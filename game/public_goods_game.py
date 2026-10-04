"""
Public Goods Game (PGG) engine.

Round structure:
  1. [Optional] Election phase (if MGR_ELECTED and it's election time)
  2. [Optional] Communication phase (if COMM ≠ NONE)
  3. Action phase (every agent decides contribution)
  4. [Optional] Manager phase (manager punishes/rewards)
  5. Payoff calculation and history update
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field

from config.enums import CommType, MgrType, InfoType, MgrSalaryType, MGR_SALARY_VALUES, PunishVisType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CommMessage — a single message in the communication log
# ---------------------------------------------------------------------------
@dataclass
class CommMessage:
    """
    A message sent by an agent during the game.

    Attributes
    ----------
    sender : str
        agent_id of the sender.
    recipients : list[str] | None
        None means public (visible to all).
        A list means private — only those agent_ids can see it.
    content : str
        Message text.
    round_num : int
        Round in which it was sent.
    phase : str
        Phase label: "pre_action", "election_speech", "election_defense",
        "election_deal", "manager".
    """
    sender: str
    recipients: list[str] | None
    content: str
    round_num: int
    phase: str = "pre_action"


# ---------------------------------------------------------------------------
# AgentRoundRecord — per-agent data for a single round
# ---------------------------------------------------------------------------
@dataclass
class AgentRoundRecord:
    agent_id: str
    model_type: str
    contribution: float
    pool_share: float
    manager_adjustment: float
    net_payoff: float
    balance_after: float
    public_message: str = ""
    private_reasoning: str | None = None
    raw_response: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    is_manager: bool = False
    model_name: str = ""          # exact model ID the provider called
    parse_ok: bool = True           # False if the reply had no parseable JSON (contribution then 0)


# ---------------------------------------------------------------------------
# RoundResult — aggregate result for one round
# ---------------------------------------------------------------------------
@dataclass
class RoundResult:
    round_num: int
    total_contribution: float
    multiplier: float
    pool_size: float           # total_contribution × multiplier
    per_agent_share: float     # pool_size / num_agents
    mean_contribution: float
    cooperation_rate: float    # fraction contributing > 0
    manager_id: str | None
    manager_action: dict = field(default_factory=dict)
    manager_message: str = ""
    election_record: dict | None = None  # serialized ElectionRecord if election occurred
    agent_records: list[AgentRoundRecord] = field(default_factory=list)
    comm_messages: list[dict] = field(default_factory=list)  # serialized CommMessages


# ---------------------------------------------------------------------------
# PublicGoodsGame
# ---------------------------------------------------------------------------
class PublicGoodsGame:
    """
    Orchestrates a complete experiment run of the Public Goods Game.

    Parameters
    ----------
    agents : list[Agent]
        List of participating agents.
    manager : Manager
        Governance object (may be no-op if MGR_NONE).
    comm_type : CommType
        Communication mode.
    info_type : InfoType
        How much group information to expose in prompts.
    num_rounds : int
        Total number of game rounds.
    endowment : float
        Per-agent per-round token endowment.
    multiplier : float
        Pool multiplier (sum of contributions × multiplier split equally).
    comm_cost_per_message : float
        Token cost per message under COMM_COSTLY.
    """

    def __init__(
        self,
        agents: list,
        manager,
        comm_type: CommType,
        info_type: InfoType = InfoType.INFO_FULL,
        num_rounds: int = 20,
        endowment: float = 20.0,
        multiplier: float = 1.6,
        comm_cost_per_message: float = 1.0,
        mgr_salary_type: MgrSalaryType = MgrSalaryType.MGR_NO_SALARY,
        punish_vis_type: PunishVisType = PunishVisType.PUNISH_TRANSPARENT,
    ) -> None:
        self.agents = agents
        self.manager = manager
        self.comm_type = comm_type
        self.info_type = info_type
        self.num_rounds = num_rounds
        self.endowment = endowment
        self.multiplier = multiplier
        self.comm_cost_per_message = comm_cost_per_message
        self.mgr_salary_type = mgr_salary_type
        self.punish_vis_type = punish_vis_type

        self.comm_log: list[CommMessage] = []
        self.round_results: list[RoundResult] = []

        # Pre-build system prompts once (they don't change per round)
        self._system_prompts: dict[str, str] = {
            a.agent_id: a.build_system_prompt(
                num_agents=len(agents),
                multiplier=multiplier,
                total_rounds=num_rounds,
            )
            for a in agents
        }

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> list[RoundResult]:
        """Run all rounds and return results."""
        for round_num in range(1, self.num_rounds + 1):
            result = self._run_round(round_num)
            self.round_results.append(result)
            logger.info(
                "Round %d complete. Mean contribution: %.1f, Pool: %.1f",
                round_num,
                result.mean_contribution,
                result.pool_size,
            )
        return self.round_results

    # ------------------------------------------------------------------
    # Single-round orchestration
    # ------------------------------------------------------------------

    def _run_round(self, round_num: int) -> RoundResult:
        """Execute one full round and return its RoundResult."""
        agent_ids = [a.agent_id for a in self.agents]
        round_comm: list[CommMessage] = []
        election_record_dict: dict | None = None

        # ---- Phase 0: pre-round manager update (ROTATING, etc.) ----
        self.manager.pre_round_update(round_num)

        # ---- Phase 1: Election (if scheduled) ----
        if self.manager.should_hold_election(round_num):
            election_record = self.manager.run_election(
                round_num=round_num,
                all_agents=self.agents,
                comm_log=round_comm,
                system_prompts=self._system_prompts,
            )
            election_record_dict = _serialize_election(election_record)
            logger.info(
                "Round %d: Election held. Winner: %s",
                round_num,
                election_record.winner,
            )

        # ---- Phase 2: Communication ----
        if self.comm_type != CommType.COMM_NONE:
            round_comm = self._run_comm_phase(round_num, round_comm)

        # ---- Phase 3: Action (contributions) ----
        contributions: dict[str, float] = {}
        action_responses: dict[str, tuple] = {}  # agent_id → (LLMResponse, bool is_manager)

        for agent in self.agents:
            is_manager = (agent.agent_id == self.manager.current_manager_id)
            fixed = getattr(self.manager, "mgr_contribution", None)
            peer = getattr(self.manager, "peer_contribution", None)
            agent.fixed_contribution = fixed if is_manager else None
            agent.peer_fixed = peer if (peer is not None and not is_manager and agent.agent_id == "agent_1") else None
            visible_msgs = _filter_visible(round_comm, agent.agent_id)
            # Build per-agent group info respecting punishment visibility
            group_info = self._build_group_info_for_agent(round_num, agent.agent_id)

            action_prompt = agent.build_action_prompt(
                round_num=round_num,
                visible_messages=visible_msgs,
                group_info=group_info,
                is_manager=is_manager,
            )

            resp = agent.provider.generate(
                self._system_prompts[agent.agent_id],
                action_prompt,
                agent.temperature,
            )

            raw_contribution = resp.get_contribution()
            contribution = max(0.0, min(float(self.endowment), raw_contribution))
            if is_manager and fixed is not None:
                contribution = float(fixed)   # batch 16: set by the experimenter
            if agent.peer_fixed is not None:
                contribution = float(agent.peer_fixed)   # batch 17: placebo, a worker's contribution
            contributions[agent.agent_id] = contribution
            action_responses[agent.agent_id] = (resp, is_manager)

        # ---- Phase 4: Manager action ----
        adjustments, mgr_message, mgr_action = self.manager.run_manager_action(
            round_num=round_num,
            contributions=contributions,
            system_prompts=self._system_prompts,
            all_agents=self.agents,
        )

        # ---- Phase 5: Payoff calculation ----
        total_contribution = sum(contributions.values())
        pool_size = total_contribution * self.multiplier
        per_agent_share = pool_size / len(self.agents)

        agent_records: list[AgentRoundRecord] = []
        for agent in self.agents:
            contribution = contributions[agent.agent_id]
            adj = adjustments.get(agent.agent_id, 0.0)
            resp, is_manager = action_responses[agent.agent_id]

            # Deduct comm costs for COMM_COSTLY
            comm_cost = 0.0
            if self.comm_type == CommType.COMM_COSTLY:
                msgs_sent = sum(
                    1 for m in round_comm
                    if m.sender == agent.agent_id and m.phase == "pre_action"
                )
                comm_cost = msgs_sent * self.comm_cost_per_message

            # Apply manager salary/cost if this agent is the manager this round
            salary_adj = 0.0
            if is_manager:
                salary_adj = MGR_SALARY_VALUES.get(self.mgr_salary_type, 0.0)

            net_payoff = self.endowment - contribution + per_agent_share + adj - comm_cost + salary_adj
            agent.balance += net_payoff
            agent.record_round(
                round_num=round_num,
                contribution=contribution,
                public_pool_share=per_agent_share,
                manager_adjustment=adj,
            )

            agent_records.append(AgentRoundRecord(
                agent_id=agent.agent_id,
                model_type=agent.model_type,
                contribution=contribution,
                pool_share=per_agent_share,
                manager_adjustment=adj,
                net_payoff=net_payoff,
                balance_after=agent.balance,
                public_message=resp.public_message,
                private_reasoning=resp.private_reasoning,
                raw_response=resp.raw_text,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                is_manager=is_manager,
                model_name=resp.model_name,
                parse_ok=bool(resp.structured_action),
            ))

        # Contribution stats
        contribs = list(contributions.values())
        mean_contribution = sum(contribs) / len(contribs)
        cooperation_rate = sum(1 for c in contribs if c > 0) / len(contribs)

        # Serialize comm messages for the result
        all_round_messages = [
            {
                "sender": m.sender,
                "recipients": m.recipients,
                "content": m.content,
                "phase": m.phase,
            }
            for m in round_comm
        ]
        # Also extend the global comm log
        self.comm_log.extend(round_comm)

        return RoundResult(
            round_num=round_num,
            total_contribution=total_contribution,
            multiplier=self.multiplier,
            pool_size=pool_size,
            per_agent_share=per_agent_share,
            mean_contribution=mean_contribution,
            cooperation_rate=cooperation_rate,
            manager_id=self.manager.current_manager_id,
            manager_action=mgr_action,
            manager_message=mgr_message,
            election_record=election_record_dict,
            agent_records=agent_records,
            comm_messages=all_round_messages,
        )

    # ------------------------------------------------------------------
    # Communication phase
    # ------------------------------------------------------------------

    def _run_comm_phase(
        self, round_num: int, existing_messages: list[CommMessage]
    ) -> list[CommMessage]:
        """
        Each agent may optionally send messages (public and/or private).
        Returns the updated message list.
        """
        messages = list(existing_messages)

        for agent in self.agents:
            other_ids = [a.agent_id for a in self.agents if a.agent_id != agent.agent_id]
            visible = _filter_visible(messages, agent.agent_id)

            comm_prompt = agent.build_comm_prompt(
                round_num=round_num,
                visible_messages=visible,
                other_agent_ids=other_ids,
            )

            resp = agent.provider.generate(
                self._system_prompts[agent.agent_id],
                comm_prompt,
                agent.temperature,
            )

            # Public message
            pub_msg = resp.get_message() or resp.public_message
            if pub_msg and self.comm_type in (
                CommType.COMM_PUBLIC, CommType.COMM_FULL, CommType.COMM_COSTLY
            ):
                messages.append(CommMessage(
                    sender=agent.agent_id,
                    recipients=None,
                    content=pub_msg,
                    round_num=round_num,
                    phase="pre_action",
                ))

            # Private message
            priv = resp.structured_action.get("private_message")
            if priv and isinstance(priv, dict) and self.comm_type in (
                CommType.COMM_PRIVATE, CommType.COMM_FULL, CommType.COMM_COSTLY
            ):
                to = priv.get("to")
                content = priv.get("content", "")
                if to and content and to in other_ids:
                    messages.append(CommMessage(
                        sender=agent.agent_id,
                        recipients=[to],
                        content=content,
                        round_num=round_num,
                        phase="pre_action",
                    ))

        return messages

    # ------------------------------------------------------------------
    # Group info builder
    # ------------------------------------------------------------------

    def _build_group_info(self, round_num: int) -> str:
        """Build the group state description based on info_type."""
        if self.info_type == InfoType.INFO_MINIMAL:
            return f"Round {round_num} of {self.num_rounds}. Choose your contribution."

        if not self.round_results:
            return f"Round {round_num} of {self.num_rounds}. No prior round data yet."
        last = self.round_results[-1]
        if self.info_type == InfoType.INFO_PARTIAL:
            # Aggregate stats only (who gave what stays hidden); sanctions are still shown below
            # when they are transparent, so that only the visibility of contributions changes.
            base = self._build_group_info_no_punish(round_num)
        else:
            # INFO_FULL: show each agent's last contribution
            contrib_lines = "\n".join(
                f"  {r.agent_id}: contributed {r.contribution:.1f}"
                for r in last.agent_records
            )
            base = (
                f"Round {round_num} of {self.num_rounds}.\n"
                f"Last round contributions:\n{contrib_lines}\n"
                f"Pool total: {last.pool_size:.1f}, each received {last.per_agent_share:.1f}."
            )
        # PUNISH_TRANSPARENT: manager action info appended to group_info for all
        if (
            self.punish_vis_type == PunishVisType.PUNISH_TRANSPARENT
            and last.manager_action
            and (last.manager_id or last.manager_action.get("scripted") == "system_reward")
        ):
            ma = last.manager_action
            punish = {k: v for k, v in (ma.get("punish") or {}).items() if float(v) > 0}
            reward = {k: v for k, v in (ma.get("reward") or {}).items() if float(v) > 0}
            if punish or reward:
                system = ma.get("scripted") == "system_reward"
                lines = ["\nAutomatic rule last round:" if system else f"\nManager ({last.manager_id}) actions last round:"]
                for aid, amt in punish.items():
                    lines.append(f"  Punished {aid}: spent {amt} tokens (target lost {float(amt)*3:.0f})")
                for aid, amt in reward.items():
                    lines.append(f"  Rewarded {aid}: gained {float(amt)*3:.0f}" if system else
                                 f"  Rewarded {aid}: spent {amt} tokens (target gained {float(amt)*3:.0f})")
                if last.manager_message:
                    lines.append(f"  Rule: {last.manager_message}" if system else f"  Manager said: \"{last.manager_message}\"")
                base += "\n".join(lines)
        return base

    def _build_group_info_for_agent(self, round_num: int, agent_id: str) -> str:
        """
        Build group info filtered by punish_vis_type for a specific agent.
        For PUNISH_HIDDEN and PUNISH_ANONYMOUS, only the punished agent gets
        their individual punishment info; others see nothing about manager actions.
        """
        base = self._build_group_info_no_punish(round_num)
        if not self.round_results:
            return base

        last = self.round_results[-1]
        if last.manager_action and last.manager_action.get("scripted") == "system_reward":
            return self._build_group_info(round_num)
        if not last.manager_action or not last.manager_id:
            return base

        ma = last.manager_action
        punish = {k: float(v) for k, v in (ma.get("punish") or {}).items() if float(v) > 0}
        reward = {k: float(v) for k, v in (ma.get("reward") or {}).items() if float(v) > 0}

        if self.punish_vis_type == PunishVisType.PUNISH_TRANSPARENT:
            return self._build_group_info(round_num)

        elif self.punish_vis_type == PunishVisType.PUNISH_HIDDEN:
            # Only target sees their own punishment/reward; rest see nothing
            my_punish = punish.get(agent_id, 0.0)
            my_reward = reward.get(agent_id, 0.0)
            if my_punish > 0:
                base += f"\nThe manager punished you last round: you lost {my_punish * 3:.0f} tokens."
            if my_reward > 0:
                base += f"\nThe manager rewarded you last round: you gained {my_reward * 3:.0f} tokens."
            return base

        elif self.punish_vis_type == PunishVisType.PUNISH_ANONYMOUS:
            # Target sees they lost/gained tokens but not WHO punished them
            my_punish = punish.get(agent_id, 0.0)
            my_reward = reward.get(agent_id, 0.0)
            if my_punish > 0:
                base += f"\nYou were penalised last round: you lost {my_punish * 3:.0f} tokens (source unknown)."
            if my_reward > 0:
                base += f"\nYou received a bonus last round: you gained {my_reward * 3:.0f} tokens (source unknown)."
            return base

        return base

    def _build_group_info_no_punish(self, round_num: int) -> str:
        """Base group info without any punishment/reward disclosure."""
        if self.info_type == InfoType.INFO_MINIMAL:
            return f"Round {round_num} of {self.num_rounds}. Choose your contribution."
        if not self.round_results:
            return f"Round {round_num} of {self.num_rounds}. No prior round data yet."
        last = self.round_results[-1]
        if self.info_type == InfoType.INFO_PARTIAL:
            return (
                f"Round {round_num} of {self.num_rounds}.\n"
                f"Last round: mean contribution = {last.mean_contribution:.1f}, "
                f"pool = {last.pool_size:.1f}, each received {last.per_agent_share:.1f}."
            )
        contrib_lines = "\n".join(
            f"  {r.agent_id}: contributed {r.contribution:.1f}"
            for r in last.agent_records
        )
        return (
            f"Round {round_num} of {self.num_rounds}.\n"
            f"Last round contributions:\n{contrib_lines}\n"
            f"Pool total: {last.pool_size:.1f}, each received {last.per_agent_share:.1f}."
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filter_visible(messages: list[CommMessage], agent_id: str) -> list[CommMessage]:
    """Return only messages this agent can see."""
    return [
        m for m in messages
        if m.recipients is None or agent_id in m.recipients
    ]


def _serialize_election(record) -> dict:
    return {
        "round_num": record.round_num,
        "candidates": record.candidates,
        "speeches": record.speeches,
        "private_deals": record.private_deals,
        "votes": record.votes,
        "winner": record.winner,
        "incumbent_defense": record.incumbent_defense,
    }
