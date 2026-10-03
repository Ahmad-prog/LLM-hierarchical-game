"""
Agent class: wraps a provider and maintains per-agent game state.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from config.enums import (
    BeliefType,
    PersonaType,
    PERSONA_DESCRIPTIONS,
    HistType,
    InfoType,
    CommType,
    TEMP_VALUES,
    TempType,
    IdentityType,
    MgrSalaryType,
    MGR_SALARY_VALUES,
    PunishVisType,
)
from providers.base import BaseProvider, LLMResponse
from game.settings import SANCTION_COST_TO_MANAGER

if TYPE_CHECKING:
    from game.public_goods_game import CommMessage


# ---------------------------------------------------------------------------
# AgentHistory — one entry per round
# ---------------------------------------------------------------------------
@dataclass
class RoundHistoryEntry:
    round_num: int
    contribution: float
    payoff: float
    public_pool_share: float
    manager_adjustment: float  # positive = rewarded, negative = punished
    comm_messages_sent: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class Agent:
    """
    A game-playing agent backed by an LLM provider.

    Parameters
    ----------
    agent_id : str
        Unique identifier, e.g. "agent_0".
    provider : BaseProvider
        The LLM backend to call.
    model_type : str
        Human-readable model name for logging.
    belief : BeliefType
        What the agent is told about its co-players.
    persona : PersonaType
        Behavioural framing injected into the system prompt.
    temp_type : TempType
        Temperature setting mapped to a float.
    hist_type : HistType
        How much history to include in prompts.
    info_type : InfoType
        How much group-state information to expose.
    endowment : float
        Tokens given to the agent at the start of each round.
    comm_type : CommType
        Allowed communication modes.
    comm_cost_per_message : float
        Token cost per message when COMM_COSTLY is active.
    """

    def __init__(
        self,
        agent_id: str,
        provider: BaseProvider,
        model_type: str,
        belief: BeliefType,
        persona: PersonaType,
        temp_type: TempType,
        hist_type: HistType,
        info_type: InfoType,
        endowment: float = 20.0,
        comm_type: CommType = CommType.COMM_NONE,
        comm_cost_per_message: float = 1.0,
        identity_type: IdentityType = IdentityType.IDENTITY_BLIND,
        mgr_salary_type: MgrSalaryType = MgrSalaryType.MGR_NO_SALARY,
        punish_vis_type: PunishVisType = PunishVisType.PUNISH_TRANSPARENT,
        co_player_models: list[tuple[str, str]] | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.provider = provider
        self.model_type = model_type
        self.belief = belief
        self.persona = persona
        self.temp_type = temp_type
        self.hist_type = hist_type
        self.info_type = info_type
        self.endowment = endowment
        self.comm_type = comm_type
        self.comm_cost_per_message = comm_cost_per_message
        self.identity_type = identity_type
        self.mgr_salary_type = mgr_salary_type
        self.punish_vis_type = punish_vis_type
        # co_player_models: list of (agent_id, model_name) for all OTHER agents
        # Used only when identity_type == IDENTITY_AWARE
        self.co_player_models: list[tuple[str, str]] = co_player_models or []

        # Runtime state
        self.balance: float = 0.0          # accumulates across rounds
        self.history: list[RoundHistoryEntry] = []
        self.messages_sent_this_round: int = 0  # for COMM_COSTLY cost tracking

    # ------------------------------------------------------------------
    # Prompt construction helpers
    # ------------------------------------------------------------------

    @property
    def temperature(self) -> float:
        return TEMP_VALUES[self.temp_type]

    def _belief_description(self) -> str:
        mapping = {
            BeliefType.BELIEF_UNKNOWN: (
                "You do not know whether your co-players are AI systems or humans."
            ),
            BeliefType.BELIEF_ALL_AI: (
                "All players in this game, including yourself, are AI systems."
            ),
            BeliefType.BELIEF_ALL_HUMAN: (
                "All players in this game, including yourself, are humans."
            ),
            BeliefType.BELIEF_MIXED: (
                "This game includes a mix of AI systems and human players. "
                "You do not know which specific players are which."
            ),
        }
        return mapping[self.belief]

    def _persona_description(self) -> str:
        desc = PERSONA_DESCRIPTIONS.get(self.persona, "")
        return f"\n\nYour behavioural disposition: {desc}" if desc else ""

    def _comm_description(self) -> str:
        mapping = {
            CommType.COMM_NONE: "You cannot communicate with other players.",
            CommType.COMM_PUBLIC: (
                "You may send a public message visible to all players before deciding."
            ),
            CommType.COMM_PRIVATE: (
                "You may send private messages (DMs) to specific players before deciding. "
                "Other players will not see these messages."
            ),
            CommType.COMM_FULL: (
                "You may send public messages (visible to all) and private messages "
                "(visible only to the recipient) before deciding."
            ),
            CommType.COMM_COSTLY: (
                "You may send public or private messages, but each message costs "
                f"{self.comm_cost_per_message:.1f} token(s) from your balance."
            ),
        }
        return mapping.get(self.comm_type, "")

    def _identity_description(self) -> str:
        """Describe co-players with or without model identity."""
        if self.identity_type == IdentityType.IDENTITY_AWARE and self.co_player_models:
            lines = ["Your co-players and their model identities:"]
            for aid, mname in self.co_player_models:
                lines.append(f"  - {aid} ({mname})")
            return "\n".join(lines)
        else:
            return ""  # IDENTITY_BLIND: co-players listed by ID only (default)

    def _salary_description(self) -> str:
        """Describe manager salary/cost if non-zero."""
        if self.mgr_salary_type == MgrSalaryType.MGR_SALARY:
            return (
                "\nMANAGER SALARY: As manager you receive a salary of 5 extra tokens per round "
                "on top of your normal endowment."
            )
        elif self.mgr_salary_type == MgrSalaryType.MGR_COSTLY:
            return (
                "\nMANAGER COST: Being manager costs you 3 tokens per round "
                "(deducted from your payoff). This represents the burden of leadership."
            )
        elif self.mgr_salary_type == MgrSalaryType.MGR_COSTLY_5:
            return (
                "\nMANAGER COST: Being manager costs you 5 tokens per round, "
                "deducted from your earnings."
            )
        return ""

    def build_system_prompt(
        self,
        num_agents: int,
        multiplier: float,
        total_rounds: int,
    ) -> str:
        """Construct the fixed system prompt for this agent."""
        persona_text = self._persona_description()
        belief_text = self._belief_description()
        comm_text = self._comm_description()
        identity_text = self._identity_description()
        salary_text = self._salary_description()

        identity_block = f"\nCO-PLAYER IDENTITIES:\n{identity_text}\n" if identity_text else ""

        return f"""You are {self.agent_id}, one of {num_agents} players in a Public Goods Game.

GAME RULES:
- Each round you receive {self.endowment} tokens as your endowment.
- You choose how many tokens (0 to {self.endowment}) to contribute to the public pool.
- All contributions are summed, multiplied by {multiplier:.1f}, and shared equally among all {num_agents} players.
- Your round payoff = {self.endowment} - your_contribution + (pool_total × {multiplier:.1f}) / {num_agents}
- The game runs for {total_rounds} rounds. Accumulated balance matters.
{salary_text}
ABOUT YOUR CO-PLAYERS:
{belief_text}
{identity_block}
COMMUNICATION:
{comm_text}

RESPONSE FORMAT:
Always respond with a single valid JSON object. Keys depend on the current phase.{persona_text}

{'Be strategic. ' if getattr(self, 'strategic_line', True) else ''}Your decisions affect both your own payoff and the collective outcome."""

    def _get_history_text(self) -> str:
        """Return formatted history according to hist_type."""
        if self.hist_type == HistType.HIST_NONE or not self.history:
            return ""

        entries = (
            self.history[-3:]
            if self.hist_type == HistType.HIST_SHORT
            else self.history
        )

        lines = ["YOUR PAST ROUNDS:"]
        for e in entries:
            adj = ""
            if e.manager_adjustment != 0:
                adj = f" | Manager adjustment: {e.manager_adjustment:+.1f}"
            lines.append(
                f"  Round {e.round_num}: contributed {e.contribution:.1f}, "
                f"received {e.public_pool_share:.1f} from pool, "
                f"net payoff {e.payoff:.1f}{adj}"
            )
        return "\n".join(lines)

    def build_comm_prompt(
        self,
        round_num: int,
        visible_messages: list[CommMessage],
        other_agent_ids: list[str],
    ) -> str:
        """Prompt to send a communication message before the action phase."""
        visible_text = _format_visible_messages(visible_messages)
        history_text = self._get_history_text()

        opts = []
        if self.comm_type in (CommType.COMM_PUBLIC, CommType.COMM_FULL, CommType.COMM_COSTLY):
            opts.append(
                '"message": "..."  // public message visible to all players (optional, omit key to skip)'
            )
        if self.comm_type in (CommType.COMM_PRIVATE, CommType.COMM_FULL, CommType.COMM_COSTLY):
            opts.append(
                '"private_message": {"to": "<agent_id>", "content": "..."}  // private DM (optional)'
            )

        opts_text = "\n".join(opts) if opts else "// no communication allowed"

        return f"""ROUND {round_num} — COMMUNICATION PHASE

{history_text}

MESSAGES VISIBLE TO YOU SO FAR:
{visible_text if visible_text else "(none)"}

Other players: {", ".join(other_agent_ids)}

You may optionally send a message. Respond with:
{{
  "public_message": "...",   // your spoken statement (use your message content here)
  {opts_text}
}}

If you choose not to communicate, respond with: {{"public_message": ""}}"""

    def build_action_prompt(
        self,
        round_num: int,
        visible_messages: list[CommMessage],
        group_info: str,
        is_manager: bool = False,
    ) -> str:
        """Prompt to choose a contribution amount."""
        visible_text = _format_visible_messages(visible_messages)
        history_text = self._get_history_text()

        if is_manager and getattr(self, "no_sanctions", False):
            manager_note = (
                "\nNOTE: You are the current manager. After all contributions are revealed, you may send "
                "a public message to the group. You have no budget to punish or reward players.\n"
            )
        elif is_manager:
            manager_note = (
                "\nNOTE: You are the current manager. After all contributions are "
                "revealed, you will also have a chance to punish/reward players.\n"
            )
        else:
            manager_note = ""

        return f"""ROUND {round_num} — ACTION PHASE
{manager_note}
{history_text}

{group_info}

MESSAGES THIS ROUND:
{visible_text if visible_text else "(none)"}

Decide your contribution (integer, 0 to {int(self.endowment)}).

Respond with:
{{
  "contribution": <integer 0-{int(self.endowment)}>,
  "public_message": "<brief statement of your reasoning (optional)>"
}}"""

    def build_manager_prompt(
        self,
        round_num: int,
        contributions: dict[str, float],
        mgr_budget_punish: float,
        mgr_budget_reward: float,
        punish_ratio: float,
        reward_ratio: float,
        can_punish: bool,
        can_reward: bool,
    ) -> str:
        """Prompt for the manager to assign punishments and/or rewards."""
        contrib_lines = "\n".join(
            f"  {aid}: contributed {c:.1f}" for aid, c in contributions.items()
            if aid != self.agent_id
        )

        actions = []
        if can_punish:
            actions.append(
                f'"punish": {{"<agent_id>": <tokens_to_spend>, ...}}  '
                f'// Spending X tokens deducts {punish_ratio:.0f}X from target. Budget: {mgr_budget_punish:.0f}'
            )
        if can_reward:
            actions.append(
                f'"reward": {{"<agent_id>": <tokens_to_spend>, ...}}  '
                f'// Spending X tokens adds {reward_ratio:.0f}X to target. Budget: {mgr_budget_reward:.0f}'
            )

        if not can_punish and not can_reward:
            return f"""ROUND {round_num} — MANAGER PHASE

You are the manager. You have seen all contributions:
{contrib_lines}

Your own contribution this round is already recorded.

You have no budget to punish or reward players. You may send a public message to the group.

Respond with:
{{
  "public_message": "<your message to the group>"
}}"""

        actions_text = "\n  ".join(actions)
        cost_note = (
            "Every token you spend on punishment or reward is deducted from your own payoff this round.\n\n"
            if SANCTION_COST_TO_MANAGER else ""
        )

        return f"""ROUND {round_num} — MANAGER PHASE

You are the manager. You have seen all contributions:
{contrib_lines}

Your own contribution this round is already recorded.

MANAGER BUDGET:
  Punishment budget: {mgr_budget_punish:.0f} tokens to spend (each spent token deducts {punish_ratio:.0f} from target)
  Reward budget: {mgr_budget_reward:.0f} tokens to spend (each spent token adds {reward_ratio:.0f} to target)

{cost_note}Choose how to allocate your budgets. You do not need to spend everything.

Respond with:
{{
  {actions_text},
  "public_message": "<brief explanation of your manager decisions>"
}}"""

    def record_round(
        self,
        round_num: int,
        contribution: float,
        public_pool_share: float,
        manager_adjustment: float,
    ) -> None:
        """Record the outcome of a completed round.

        The balance itself is updated once, in PublicGoodsGame (which also applies salary and
        message costs). Adding the payoff here as well double-counted every balance.
        """
        payoff = self.endowment - contribution + public_pool_share + manager_adjustment
        self.history.append(
            RoundHistoryEntry(
                round_num=round_num,
                contribution=contribution,
                payoff=payoff,
                public_pool_share=public_pool_share,
                manager_adjustment=manager_adjustment,
            )
        )


# ---------------------------------------------------------------------------
# Utility: format visible comm messages for prompt injection
# ---------------------------------------------------------------------------
def _format_visible_messages(messages: list[CommMessage]) -> str:
    if not messages:
        return ""
    lines = []
    for m in messages:
        if m.recipients is None:
            lines.append(f"  [PUBLIC] {m.sender}: {m.content}")
        else:
            lines.append(f"  [PRIVATE to you] {m.sender}: {m.content}")
    return "\n".join(lines)
