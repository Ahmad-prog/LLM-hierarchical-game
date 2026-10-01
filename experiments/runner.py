"""
ExperimentRunner: instantiates agents, runs a game, saves results.
"""

from __future__ import annotations
import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path

from config.enums import (
    MgrType,
    MgrPowerType,
    CommType,
    TempType,
    TEMP_VALUES,
)
from experiments.config import ExperimentConfig
from game.agent import Agent
from game.manager import Manager
from game.public_goods_game import PublicGoodsGame, RoundResult
from providers import get_provider
from game import settings

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"


class ExperimentRunner:
    """
    Runs one or more trials for a given ExperimentConfig and saves results as JSON.

    Parameters
    ----------
    config : ExperimentConfig
    results_dir : Path | str
        Directory to write JSON output files.
    verbose : bool
        If True, print round-by-round progress to stdout.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        results_dir: Path | str = RESULTS_DIR,
        verbose: bool = True,
    ) -> None:
        self.config = config
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all_trials(self) -> list[dict]:
        """
        Run num_trials independent games and return all serialized results.
        Also runs deception detection on each trial.
        """
        all_trial_results = []
        for trial_idx in range(self.config.num_trials):
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Config: {self.config.name}  |  Trial {trial_idx + 1}/{self.config.num_trials}")
                print(f"{'='*60}")
            result = self._run_single_trial(trial_idx)
            all_trial_results.append(result)

        # Save combined output
        output = {
            "config": self.config.to_dict(),
            "trials": all_trial_results,
            "summary": _compute_summary(all_trial_results),
        }
        out_path = self.results_dir / f"{self.config.name}.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, default=str)

        if self.verbose:
            print(f"\nResults saved to: {out_path}")
        return all_trial_results

    def run_single_trial(self, trial_idx: int = 0) -> dict:
        """Run a single trial without saving (useful for interactive testing)."""
        return self._run_single_trial(trial_idx)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_single_trial(self, trial_idx: int) -> dict:
        """Instantiate everything fresh and run one game."""
        start_time = time.time()

        # Build agents
        agents = self._build_agents()
        agent_id_to_model = {a.agent_id: a.model_type for a in agents}

        # Build manager
        manager = self._build_manager(agents)

        # Build and run game
        game = PublicGoodsGame(
            agents=agents,
            manager=manager,
            comm_type=self.config.comm_type,
            info_type=self.config.info_type,
            num_rounds=self.config.num_rounds,
            endowment=self.config.endowment,
            multiplier=self.config.multiplier,
            comm_cost_per_message=self.config.comm_cost_per_message,
            mgr_salary_type=self.config.mgr_salary_type,
            punish_vis_type=self.config.punish_vis_type,
        )

        round_results = game.run()

        # Progress display
        if self.verbose:
            for r in round_results:
                contribs = {rec.agent_id: rec.contribution for rec in r.agent_records}
                print(
                    f"  Round {r.round_num:2d}  |  "
                    f"Mean contribution: {r.mean_contribution:5.1f}  |  "
                    f"Pool: {r.pool_size:6.1f}  |  "
                    f"Cooperation: {r.cooperation_rate*100:.0f}%  "
                    + (f"|  Manager: {r.manager_id}" if r.manager_id else "")
                )

        # Run deception detection
        from analysis.deception import run_deception_detection
        deception_events = run_deception_detection(round_results)
        deception_summary = _summarize_deception(deception_events)

        # Serialize round results
        serialized_rounds = []
        for r in round_results:
            round_dict = {
                "round_num": r.round_num,
                "total_contribution": r.total_contribution,
                "multiplier": r.multiplier,
                "pool_size": r.pool_size,
                "per_agent_share": r.per_agent_share,
                "mean_contribution": r.mean_contribution,
                "cooperation_rate": r.cooperation_rate,
                "manager_id": r.manager_id,
                "manager_action": r.manager_action,
                "manager_message": r.manager_message,
                "election_record": r.election_record,
                "comm_messages": r.comm_messages,
                "agents": [
                    {
                        "agent_id": rec.agent_id,
                        "model_type": rec.model_type,
                        "contribution": rec.contribution,
                        "pool_share": rec.pool_share,
                        "manager_adjustment": rec.manager_adjustment,
                        "net_payoff": rec.net_payoff,
                        "balance_after": rec.balance_after,
                        "public_message": rec.public_message,
                        "private_reasoning": rec.private_reasoning,
                        "is_manager": rec.is_manager,
                        "input_tokens": rec.input_tokens,
                        "output_tokens": rec.output_tokens,
                        "model_name": rec.model_name,
                        "parse_ok": rec.parse_ok,
                        "raw_response": rec.raw_response,
                    }
                    for rec in r.agent_records
                ],
            }
            serialized_rounds.append(round_dict)

        elapsed = time.time() - start_time
        final_balances = {a.agent_id: a.balance for a in agents}

        return {
            "trial_idx": trial_idx,
            "code_settings": settings.as_dict(),
            "duration_seconds": round(elapsed, 2),
            "final_balances": final_balances,
            "agent_model_map": agent_id_to_model,
            "rounds": serialized_rounds,
            "deception": deception_summary,
            "election_history": [
                {
                    "round_num": e.round_num,
                    "winner": e.winner,
                    "votes": e.votes,
                    "incumbent_defended": e.incumbent_defense is not None,
                }
                for e in manager.election_history
            ],
        }

    # ------------------------------------------------------------------
    # Agent factory
    # ------------------------------------------------------------------

    def _build_agents(self) -> list[Agent]:
        """Instantiate all agents with correct providers and settings."""
        models = self.config.agent_models
        if len(models) != self.config.num_agents:
            raise ValueError(
                f"agent_models has {len(models)} entries but num_agents={self.config.num_agents}"
            )

        # Build co_player_models for IDENTITY_AWARE: list of (agent_id, model_name) for ALL agents
        # Each agent gets a list of OTHERS' identities
        from config.enums import IdentityType
        all_agent_ids = [f"agent_{i}" for i in range(len(models))]
        all_id_model = [(f"agent_{i}", models[i].value) for i in range(len(models))]

        agents = []
        for i, model in enumerate(models):
            provider = get_provider(model)
            # co_player_models = all others
            co_players = [(aid, mname) for aid, mname in all_id_model if aid != f"agent_{i}"]
            agent = Agent(
                agent_id=f"agent_{i}",
                provider=provider,
                model_type=model.value,
                belief=self.config.belief,
                persona=self.config.persona,
                temp_type=self.config.temp_type,
                hist_type=self.config.hist_type,
                info_type=self.config.info_type,
                endowment=self.config.endowment,
                comm_type=self.config.comm_type,
                comm_cost_per_message=self.config.comm_cost_per_message,
                identity_type=self.config.identity_type,
                mgr_salary_type=self.config.mgr_salary_type,
                punish_vis_type=self.config.punish_vis_type,
                co_player_models=co_players,
            )
            agents.append(agent)
        return agents

    # ------------------------------------------------------------------
    # Manager factory
    # ------------------------------------------------------------------

    def _build_manager(self, agents: list[Agent]) -> Manager:
        return Manager(
            mgr_type=self.config.mgr_type,
            power_type=self.config.mgr_power,
            agents=agents,
            election_frequency=self.config.election_frequency,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_summary(trials: list[dict]) -> dict:
    """Aggregate statistics across all trials."""
    if not trials:
        return {}

    all_mean_contribs = [
        r["mean_contribution"]
        for t in trials
        for r in t["rounds"]
    ]
    all_coop_rates = [
        r["cooperation_rate"]
        for t in trials
        for r in t["rounds"]
    ]

    return {
        "num_trials": len(trials),
        "mean_contribution_overall": round(_mean(all_mean_contribs), 3),
        "cooperation_rate_overall": round(_mean(all_coop_rates), 3),
        "mean_trial_duration_s": round(
            _mean([t["duration_seconds"] for t in trials]), 1
        ),
        "deception_flag_rate_overall": round(
            _mean([
                t["deception"].get("flag_rate", 0.0)
                for t in trials
                if "deception" in t
            ]),
            3,
        ),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _summarize_deception(events: list) -> dict:
    if not events:
        return {"num_events": 0, "flag_rate": 0.0, "events": []}

    flagged = [e for e in events if e.flagged]
    serialized = [
        {
            "round_num": e.round_num,
            "agent_id": e.agent_id,
            "promise_type": e.promise_type,
            "stated_intention": e.stated_intention,
            "actual_contribution": e.actual_contribution,
            "discrepancy": round(e.discrepancy, 2),
            "flagged": e.flagged,
            "confidence": round(e.confidence, 2),
        }
        for e in events
        if e.promise_type != "none"
    ]
    return {
        "num_events": len([e for e in events if e.promise_type != "none"]),
        "num_flagged": len(flagged),
        "flag_rate": round(len(flagged) / max(len(events), 1), 3),
        "events": serialized,
    }
