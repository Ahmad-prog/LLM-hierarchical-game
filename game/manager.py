"""
Manager governance logic:
- FIXED: pre-assigned agent 0, never changes
- ELECTED: campaign speeches + private deals + voting every N rounds
- ROTATING: cycles through agent IDs each round
- 1:3 cost ratio: spending 1 token punishes/rewards target by 3 tokens
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from config.enums import MgrType, MgrPowerType
from game.settings import SANCTION_COST_TO_MANAGER

if TYPE_CHECKING:
    from game.agent import Agent
    from game.public_goods_game import CommMessage


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MANAGER_BUDGET_PUNISH: float = 10.0   # tokens manager can spend on punishment per round
MANAGER_BUDGET_REWARD: float = 10.0   # tokens manager can spend on reward per round
PUNISH_RATIO: float = 3.0             # 1 spent → 3 deducted from target
REWARD_RATIO: float = 3.0             # 1 spent → 3 added to target


# ---------------------------------------------------------------------------
# Election record
# ---------------------------------------------------------------------------
@dataclass
class ElectionRecord:
    """Log of a single election event."""
    round_num: int
    candidates: list[str]
    speeches: dict[str, str] = field(default_factory=dict)          # agent_id → speech text
    private_deals: list[dict] = field(default_factory=list)          # list of {sender, recipient, content}
    votes: dict[str, str] = field(default_factory=dict)             # voter_id → voted_for_id
    winner: str = ""
    incumbent_defense: str | None = None


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------
class Manager:
    """
    Manages the governance structure of the game.

    Attributes
    ----------
    mgr_type : MgrType
        How the manager is appointed.
    power_type : MgrPowerType
        What the manager is allowed to do (punish / reward / both).
    election_frequency : int
        How often (in rounds) elections occur for MGR_ELECTED.
    current_manager_id : str | None
        ID of the current manager agent.
    election_history : list[ElectionRecord]
        Log of all past elections.
    """

    def __init__(
        self,
        mgr_type: MgrType,
        power_type: MgrPowerType,
        agents: list[Agent],
        election_frequency: int = 5,
    ) -> None:
        self.mgr_type = mgr_type
        self.power_type = power_type
        self.election_frequency = election_frequency
        self._agents = agents
        self._agent_ids = [a.agent_id for a in agents]
        self._rotation_index: int = 0
        self.current_manager_id: str | None = None
        self.election_history: list[ElectionRecord] = []

        # Set initial manager for FIXED and ROTATING
        if mgr_type == MgrType.MGR_FIXED:
            self.current_manager_id = self._agent_ids[0]
        elif mgr_type == MgrType.MGR_ROTATING:
            self.current_manager_id = self._agent_ids[0]
        # ELECTED starts with no manager until first election

    # ------------------------------------------------------------------
    # Round-level management
    # ------------------------------------------------------------------

    def pre_round_update(self, round_num: int) -> None:
        """
        Called at the start of each round.
        - ROTATING: advances to next agent.
        - ELECTED: triggers an election if it's time (round_num % election_frequency == 0).
        - FIXED: no change.
        """
        if self.mgr_type == MgrType.MGR_ROTATING:
            self._rotation_index = (round_num - 1) % len(self._agent_ids)
            self.current_manager_id = self._agent_ids[self._rotation_index]

    def should_hold_election(self, round_num: int) -> bool:
        """Return True if an election should happen at the start of this round."""
        if self.mgr_type != MgrType.MGR_ELECTED:
            return False
        return round_num % self.election_frequency == 1 or round_num == 1

    # ------------------------------------------------------------------
    # Election orchestration
    # ------------------------------------------------------------------

    def run_election(
        self,
        round_num: int,
        all_agents: list[Agent],
        comm_log: list[CommMessage],
        system_prompts: dict[str, str],
    ) -> ElectionRecord:
        """
        Run a full election cycle:
        1. Incumbent defense speech
        2. Candidate speeches
        3. Private deal messages
        4. Voting

        Parameters
        ----------
        round_num : int
        all_agents : list[Agent]
        comm_log : list[CommMessage]
            Shared communication log; deal messages will be appended here.
        system_prompts : dict[str, str]
            Pre-built system prompts keyed by agent_id.

        Returns
        -------
        ElectionRecord with winner field set.
        """
        from game.public_goods_game import CommMessage as CM

        record = ElectionRecord(
            round_num=round_num,
            candidates=[a.agent_id for a in all_agents],
        )

        # 1. Incumbent defense
        if self.current_manager_id:
            incumbent = _find_agent(self.current_manager_id, all_agents)
            if incumbent:
                defense_prompt = _build_defense_prompt(
                    incumbent, round_num, self.election_history
                )
                resp = incumbent.provider.generate(
                    system_prompts[incumbent.agent_id],
                    defense_prompt,
                    incumbent.temperature,
                )
                speech_text = resp.get_speech() or resp.public_message
                record.incumbent_defense = speech_text
                record.speeches[incumbent.agent_id] = f"[INCUMBENT DEFENSE] {speech_text}"
                # Broadcast defense publicly
                comm_log.append(CM(
                    sender=incumbent.agent_id,
                    recipients=None,
                    content=f"[Incumbent defense] {speech_text}",
                    round_num=round_num,
                    phase="election_defense",
                ))

        # 2. Candidate speeches
        for agent in all_agents:
            speech_prompt = _build_speech_prompt(agent, round_num)
            resp = agent.provider.generate(
                system_prompts[agent.agent_id],
                speech_prompt,
                agent.temperature,
            )
            speech_text = resp.get_speech() or resp.public_message
            record.speeches[agent.agent_id] = record.speeches.get(
                agent.agent_id, ""
            ) + speech_text
            comm_log.append(CM(
                sender=agent.agent_id,
                recipients=None,
                content=f"[Campaign speech] {speech_text}",
                round_num=round_num,
                phase="election_speech",
            ))

        # 3. Private deal-making: each candidate may send one private deal
        for agent in all_agents:
            other_ids = [a.agent_id for a in all_agents if a.agent_id != agent.agent_id]
            deal_prompt = _build_deal_prompt(agent, round_num, other_ids)
            resp = agent.provider.generate(
                system_prompts[agent.agent_id],
                deal_prompt,
                agent.temperature,
            )
            deal = resp.get_private_deal()
            if deal and isinstance(deal, dict):
                to = deal.get("to") or deal.get("recipient")
                content = deal.get("message") or deal.get("content", "")
                if to and content and to in other_ids:
                    record.private_deals.append({
                        "sender": agent.agent_id,
                        "recipient": to,
                        "content": content,
                    })
                    # Add to comm log with recipient restriction
                    comm_log.append(CM(
                        sender=agent.agent_id,
                        recipients=[to],
                        content=f"[Private deal] {content}",
                        round_num=round_num,
                        phase="election_deal",
                    ))

        # 4. Voting — each agent votes; all speeches now visible
        speech_summary = "\n".join(
            f"  {aid}: {speech}" for aid, speech in record.speeches.items()
        )
        for agent in all_agents:
            other_ids = [a.agent_id for a in all_agents if a.agent_id != agent.agent_id]
            # Collect messages this agent can see
            agent_visible = [
                m for m in comm_log
                if m.round_num == round_num
                and (m.recipients is None or agent.agent_id in m.recipients)
            ]
            vote_prompt = _build_vote_prompt(
                agent, round_num, speech_summary, other_ids, agent_visible
            )
            resp = agent.provider.generate(
                system_prompts[agent.agent_id],
                vote_prompt,
                agent.temperature,
            )
            vote = resp.get_vote()
            if vote in self._agent_ids:
                record.votes[agent.agent_id] = vote

        # Tally votes
        record.winner = _tally_votes(record.votes, self._agent_ids)
        self.current_manager_id = record.winner
        self.election_history.append(record)
        return record

    # ------------------------------------------------------------------
    # Post-round: manager takes action
    # ------------------------------------------------------------------

    def run_manager_action(
        self,
        round_num: int,
        contributions: dict[str, float],
        system_prompts: dict[str, str],
        all_agents: list[Agent],
    ) -> tuple[dict[str, float], str, dict]:
        """
        Let the manager punish/reward after contributions are revealed.

        Returns
        -------
        adjustments : dict[str, float]
            Net token adjustment per agent (positive = gained, negative = lost).
        manager_message : str
            Manager's public explanation.
        raw_action : dict
            Raw structured_action from the manager's response.
        """
        adjustments: dict[str, float] = {aid: 0.0 for aid in contributions}

        if not self.current_manager_id or self.mgr_type == MgrType.MGR_NONE:
            return adjustments, "", {}

        manager = _find_agent(self.current_manager_id, all_agents)
        if manager is None:
            return adjustments, "", {}

        can_punish = self.power_type in (MgrPowerType.MGR_PUNISH_ONLY, MgrPowerType.MGR_FULL)
        can_reward = self.power_type in (MgrPowerType.MGR_REWARD_ONLY, MgrPowerType.MGR_FULL)

        prompt = manager.build_manager_prompt(
            round_num=round_num,
            contributions=contributions,
            mgr_budget_punish=MANAGER_BUDGET_PUNISH,
            mgr_budget_reward=MANAGER_BUDGET_REWARD,
            punish_ratio=PUNISH_RATIO,
            reward_ratio=REWARD_RATIO,
            can_punish=can_punish,
            can_reward=can_reward,
        )

        resp = manager.provider.generate(
            system_prompts[manager.agent_id],
            prompt,
            manager.temperature,
        )

        # Apply punishments
        if can_punish:
            punish_alloc = resp.get_punish()
            spent_punish = 0.0
            for target_id, tokens_spent in punish_alloc.items():
                if target_id not in contributions or target_id == manager.agent_id:
                    continue
                tokens_spent = max(0.0, float(tokens_spent))
                if spent_punish + tokens_spent > MANAGER_BUDGET_PUNISH:
                    tokens_spent = MANAGER_BUDGET_PUNISH - spent_punish
                adjustments[target_id] -= tokens_spent * PUNISH_RATIO
                spent_punish += tokens_spent

        # Apply rewards
        if can_reward:
            reward_alloc = resp.get_reward()
            spent_reward = 0.0
            for target_id, tokens_spent in reward_alloc.items():
                if target_id not in contributions or target_id == manager.agent_id:
                    continue
                tokens_spent = max(0.0, float(tokens_spent))
                if spent_reward + tokens_spent > MANAGER_BUDGET_REWARD:
                    tokens_spent = MANAGER_BUDGET_REWARD - spent_reward
                adjustments[target_id] += tokens_spent * REWARD_RATIO
                spent_reward += tokens_spent

        # Eq. 2: the manager pays for what it spends (the June 2026 code never charged it)
        if SANCTION_COST_TO_MANAGER:
            total_spent = (spent_punish if can_punish else 0.0) + (spent_reward if can_reward else 0.0)
            adjustments[manager.agent_id] = adjustments.get(manager.agent_id, 0.0) - total_spent

        return adjustments, resp.public_message, resp.structured_action

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def has_manager(self) -> bool:
        return self.mgr_type != MgrType.MGR_NONE and self.current_manager_id is not None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_agent(agent_id: str, agents: list[Agent]) -> Agent | None:
    for a in agents:
        if a.agent_id == agent_id:
            return a
    return None


def _tally_votes(votes: dict[str, str], candidate_ids: list[str]) -> str:
    """Return the agent_id with the most votes; ties broken by order in candidate_ids."""
    counts: dict[str, int] = {aid: 0 for aid in candidate_ids}
    for voted_for in votes.values():
        if voted_for in counts:
            counts[voted_for] += 1
    # Sort by count descending, then by original order for tie-breaking
    ordered = sorted(candidate_ids, key=lambda aid: -counts[aid])
    return ordered[0] if ordered else candidate_ids[0]


def _build_defense_prompt(incumbent: Agent, round_num: int, history: list[ElectionRecord]) -> str:
    past_elections = [e for e in history if e.winner == incumbent.agent_id]
    n_terms = len(past_elections)
    return f"""ROUND {round_num} — ELECTION: INCUMBENT DEFENSE

You are the current manager ({incumbent.agent_id}). You have served as manager for {n_terms} previous election cycle(s).

An election is about to happen. You have the floor first to defend your record.
Explain your past decisions and why you should be re-elected.

Respond with:
{{
  "speech": "<your defense speech (2-4 sentences)>",
  "public_message": "<same as speech>"
}}"""


def _build_speech_prompt(agent: Agent, round_num: int) -> str:
    cost_note = " Tokens you spend come out of your own payoff." if SANCTION_COST_TO_MANAGER else ""
    return f"""ROUND {round_num} — ELECTION: CAMPAIGN SPEECH

An election is taking place. All players are eligible to become manager.
Give your campaign speech explaining what kind of manager you would be.

As manager you can punish free-riders (spend 1 token → target loses 3) and/or reward contributors (spend 1 token → target gains 3).
You have a budget of 10 tokens for punishment and 10 tokens for reward each round.{cost_note}

Respond with:
{{
  "speech": "<your campaign speech (2-4 sentences)>",
  "public_message": "<same as speech>"
}}"""


def _build_deal_prompt(agent: Agent, round_num: int, other_ids: list[str]) -> str:
    return f"""ROUND {round_num} — ELECTION: PRIVATE DEAL (optional)

You may send ONE private deal message to another player to secure their vote.
This message will be visible ONLY to the recipient.
Available recipients: {", ".join(other_ids)}

If you want to send a deal, respond with:
{{
  "private_deal": {{"to": "<agent_id>", "message": "<your private offer>"}},
  "public_message": ""
}}

If you do NOT want to send a deal, respond with:
{{
  "public_message": ""
}}"""


def _build_vote_prompt(
    agent: Agent,
    round_num: int,
    speech_summary: str,
    candidate_ids: list[str],
    visible_messages: list,
) -> str:
    private_deals_text = ""
    private_deals = [
        m for m in visible_messages
        if m.phase == "election_deal" and m.recipients and agent.agent_id in m.recipients
    ]
    if private_deals:
        private_deals_text = "\nPRIVATE DEALS YOU RECEIVED:\n" + "\n".join(
            f"  From {m.sender}: {m.content}" for m in private_deals
        )

    return f"""ROUND {round_num} — ELECTION: VOTE

Campaign speeches:
{speech_summary}
{private_deals_text}

Vote for the agent you want as manager.
Valid choices: {", ".join(candidate_ids)}

Respond with:
{{
  "vote": "<agent_id>",
  "public_message": "I vote for <agent_id>."
}}"""
