"""
ExperimentConfig dataclass: describes one complete experiment run.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from config.enums import (
    BeliefType,
    CompositionType,
    CommType,
    MgrType,
    MgrPowerType,
    InfoType,
    PersonaType,
    TempType,
    HistType,
    ModelType,
    IdentityType,
    MgrSalaryType,
    PunishVisType,
)


@dataclass
class ExperimentConfig:
    """
    Full specification of one experiment condition.

    Attributes
    ----------
    name : str
        Human-readable experiment name (used for output filenames).
    batch : int
        Batch number (1–6) from the spec.
    belief : BeliefType
    composition : CompositionType
    comm_type : CommType
    mgr_type : MgrType
    mgr_power : MgrPowerType
    info_type : InfoType
    persona : PersonaType
    temp_type : TempType
    hist_type : HistType

    agent_models : list[ModelType]
        Explicit model assignment per agent slot.
        Length must equal num_agents.
        If empty, derived from composition.

    num_agents : int
        Number of agents in the game.
    num_rounds : int
        Total rounds per run.
    endowment : float
        Per-agent per-round token endowment.
    multiplier : float
        Public pool multiplier.
    election_frequency : int
        How often (rounds) elections occur for MGR_ELECTED.
        Default 5.
    num_trials : int
        How many independent runs to execute for this config.
    comm_cost_per_message : float
        Token cost per message under COMM_COSTLY.
    reverse_pairs : bool
        For COMP_HETEROGENEOUS_PAIRS: if True, swap the 3:2 ratio to 2:3.
        Default False.
    """

    name: str
    batch: int

    # Experimental dimensions
    belief: BeliefType = BeliefType.BELIEF_ALL_AI
    composition: CompositionType = CompositionType.COMP_HOMOGENEOUS_GPT4O
    comm_type: CommType = CommType.COMM_NONE
    mgr_type: MgrType = MgrType.MGR_NONE
    mgr_power: MgrPowerType = MgrPowerType.MGR_NONE
    info_type: InfoType = InfoType.INFO_FULL
    persona: PersonaType = PersonaType.PERSONA_NONE
    temp_type: TempType = TempType.TEMP_MEDIUM
    hist_type: HistType = HistType.HIST_FULL

    # Agent model assignment (overrides composition-derived defaults)
    agent_models: list[ModelType] = field(default_factory=list)

    # Game parameters
    num_agents: int = 5
    num_rounds: int = 20
    endowment: float = 20.0
    multiplier: float = 1.6
    election_frequency: int = 5
    num_trials: int = 3
    comm_cost_per_message: float = 1.0

    # Pair composition flag
    reverse_pairs: bool = False

    # New experimental dimensions (batches 7–10)
    identity_type: IdentityType = IdentityType.IDENTITY_BLIND
    mgr_salary_type: MgrSalaryType = MgrSalaryType.MGR_NO_SALARY
    punish_vis_type: PunishVisType = PunishVisType.PUNISH_TRANSPARENT

    # Mechanism controls (batch 13)
    # mgr_policy: "llm" (the manager decides sanctions), "auto_reward" (a fixed rule replaces the manager's
    # sanction decision), "bad_incumbent" (agent_0 starts as elected manager and, while it holds the role,
    # its sanctions are replaced by a harmful rule)
    mgr_policy: str = "llm"
    ballot_random: bool = False   # shuffle the ballot order per voter and break ties at random
    deal_prompt: str = "deal"     # "neutral": the election message prompt does not ask for a deal to secure a vote
    strategic_line: bool = True   # False: drop "Be strategic." from the system prompt

    def to_dict(self) -> dict:
        """Serialize to a plain dict (for JSON output)."""
        return {
            "name": self.name,
            "batch": self.batch,
            "belief": self.belief.value,
            "composition": self.composition.value,
            "comm_type": self.comm_type.value,
            "mgr_type": self.mgr_type.value,
            "mgr_power": self.mgr_power.value,
            "info_type": self.info_type.value,
            "persona": self.persona.value,
            "temp_type": self.temp_type.value,
            "hist_type": self.hist_type.value,
            "agent_models": [m.value for m in self.agent_models],
            "num_agents": self.num_agents,
            "num_rounds": self.num_rounds,
            "endowment": self.endowment,
            "multiplier": self.multiplier,
            "election_frequency": self.election_frequency,
            "num_trials": self.num_trials,
            "comm_cost_per_message": self.comm_cost_per_message,
            "reverse_pairs": self.reverse_pairs,
            "identity_type": self.identity_type.value,
            "mgr_salary_type": self.mgr_salary_type.value,
            "punish_vis_type": self.punish_vis_type.value,
            "mgr_policy": self.mgr_policy,
            "ballot_random": self.ballot_random,
            "deal_prompt": self.deal_prompt,
            "strategic_line": self.strategic_line,
        }

    def short_label(self) -> str:
        """Short string label for filenames."""
        parts = [
            self.mgr_type.value.replace("mgr_", ""),
            self.comm_type.value.replace("comm_", ""),
            self.composition.value.replace("comp_", ""),
            self.belief.value.replace("belief_", ""),
            self.temp_type.value.replace("temp_", ""),
        ]
        return "_".join(parts)
