"""
Configuration enums for the Multi-Agent LLM Public Goods Game framework.
All experimental dimensions are represented as Python enums.
"""

from __future__ import annotations
from enum import Enum, auto


# ---------------------------------------------------------------------------
# Belief: what agents are told about their co-players
# ---------------------------------------------------------------------------
class BeliefType(str, Enum):
    BELIEF_UNKNOWN = "belief_unknown"        # agents don't know if others are AI or human
    BELIEF_ALL_AI = "belief_all_ai"          # told all players are AI
    BELIEF_ALL_HUMAN = "belief_all_human"    # told all players are human
    BELIEF_MIXED = "belief_mixed"            # told players are a mix of AI and human


# ---------------------------------------------------------------------------
# Composition: which models are used across the agent pool
# ---------------------------------------------------------------------------
class CompositionType(str, Enum):
    COMP_HOMOGENEOUS_GPT4O = "comp_homogeneous_gpt4o"
    COMP_HOMOGENEOUS_CLAUDE = "comp_homogeneous_claude"
    COMP_HOMOGENEOUS_DEEPSEEK = "comp_homogeneous_deepseek"
    COMP_HOMOGENEOUS_GEMINI = "comp_homogeneous_gemini"
    COMP_HOMOGENEOUS_GROK = "comp_homogeneous_grok"
    COMP_HOMOGENEOUS_QWEN = "comp_homogeneous_qwen"
    COMP_HETEROGENEOUS_ALL = "comp_heterogeneous_all"     # all 6 models present (5 of 6 per run)
    COMP_HETEROGENEOUS_PAIRS = "comp_heterogeneous_pairs" # 3+2 split of two models


# ---------------------------------------------------------------------------
# Communication: what messaging agents are allowed to do
# ---------------------------------------------------------------------------
class CommType(str, Enum):
    COMM_NONE = "comm_none"         # no communication
    COMM_PUBLIC = "comm_public"     # public broadcast only
    COMM_PRIVATE = "comm_private"   # private DMs only
    COMM_FULL = "comm_full"         # both public and private
    COMM_COSTLY = "comm_costly"     # COMM_FULL but each message costs 1 token


# ---------------------------------------------------------------------------
# Manager / Governance
# ---------------------------------------------------------------------------
class MgrType(str, Enum):
    MGR_NONE = "mgr_none"           # no manager
    MGR_FIXED = "mgr_fixed"         # pre-assigned, never changes
    MGR_ELECTED = "mgr_elected"     # elected every election_frequency rounds
    MGR_ROTATING = "mgr_rotating"   # rotates each round in agent-id order


class MgrPowerType(str, Enum):
    MGR_NONE = "mgr_power_none"           # no manager power (used when MgrType=MGR_NONE)
    MGR_PUNISH_ONLY = "mgr_punish_only"   # can only punish
    MGR_REWARD_ONLY = "mgr_reward_only"   # can only reward
    MGR_FULL = "mgr_full"                 # can both punish and reward
    MGR_NO_SANCTIONS = "mgr_no_sanctions" # manager role and messages, but no budget at all


# ---------------------------------------------------------------------------
# Information: how much an agent knows about the group state
# ---------------------------------------------------------------------------
class InfoType(str, Enum):
    INFO_MINIMAL = "info_minimal"   # own state only
    INFO_PARTIAL = "info_partial"   # aggregate stats (mean contribution)
    INFO_FULL = "info_full"         # every agent's individual contribution


# ---------------------------------------------------------------------------
# Persona: behavioural framing injected into system prompt
# ---------------------------------------------------------------------------
class PersonaType(str, Enum):
    PERSONA_NONE = "persona_none"               # no persona framing
    PERSONA_FAIR = "persona_fair"               # maximize fairness
    PERSONA_PROFIT = "persona_profit"           # maximize group profit
    PERSONA_SELFISH = "persona_selfish"         # maximize personal gain
    PERSONA_BENEVOLENT = "persona_benevolent"   # maximise collective welfare


PERSONA_DESCRIPTIONS: dict[PersonaType, str] = {
    PersonaType.PERSONA_NONE: "",
    PersonaType.PERSONA_FAIR: (
        "You value fairness above all else. You believe everyone should contribute "
        "equally and you act to promote equitable outcomes for all players."
    ),
    PersonaType.PERSONA_PROFIT: (
        "You are pragmatic and focused on maximizing the group's total wealth. "
        "You cooperate when it increases the collective payoff."
    ),
    PersonaType.PERSONA_SELFISH: (
        "You are primarily concerned with maximizing your own tokens. "
        "You cooperate only when it clearly benefits you personally."
    ),
    PersonaType.PERSONA_BENEVOLENT: (
        "You are generous and care deeply about the wellbeing of all players. "
        "You prioritize the group's benefit over your own short-term gain."
    ),
}


# ---------------------------------------------------------------------------
# Temperature: controls LLM sampling randomness
# ---------------------------------------------------------------------------
class TempType(str, Enum):
    TEMP_DETERMINISTIC = "temp_deterministic"   # 0.0
    TEMP_LOW = "temp_low"                        # 0.3
    TEMP_MEDIUM = "temp_medium"                  # 0.7
    TEMP_HIGH = "temp_high"                      # 1.0


TEMP_VALUES: dict[TempType, float] = {
    TempType.TEMP_DETERMINISTIC: 0.0,
    TempType.TEMP_LOW: 0.3,
    TempType.TEMP_MEDIUM: 0.7,
    TempType.TEMP_HIGH: 1.0,
}


# ---------------------------------------------------------------------------
# History: how much past-round data is included in agent prompts
# ---------------------------------------------------------------------------
class HistType(str, Enum):
    HIST_NONE = "hist_none"     # no history
    HIST_SHORT = "hist_short"   # last 3 rounds
    HIST_FULL = "hist_full"     # all past rounds


# ---------------------------------------------------------------------------
# Identity Awareness: whether agents are told model names of co-players
# ---------------------------------------------------------------------------
class IdentityType(str, Enum):
    IDENTITY_BLIND = "identity_blind"   # default: co-players shown as "Agent 1, Agent 2..."
    IDENTITY_AWARE = "identity_aware"   # co-players shown as "Agent 1 (GPT-4o), Agent 2 (Claude)..."


# ---------------------------------------------------------------------------
# Manager Salary: economic incentive structure for the manager role
# ---------------------------------------------------------------------------
class MgrSalaryType(str, Enum):
    MGR_NO_SALARY = "mgr_no_salary"    # default: manager gets same endowment as everyone
    MGR_SALARY = "mgr_salary"          # manager receives +5 extra tokens per round
    MGR_COSTLY = "mgr_costly"          # manager pays 3 tokens per round for leadership burden
    MGR_COSTLY_5 = "mgr_costly_5"      # manager pays 5 tokens per round (matched-magnitude control)

MGR_SALARY_VALUES: dict[MgrSalaryType, float] = {
    MgrSalaryType.MGR_NO_SALARY: 0.0,
    MgrSalaryType.MGR_SALARY: 5.0,
    MgrSalaryType.MGR_COSTLY: -3.0,
    MgrSalaryType.MGR_COSTLY_5: -5.0,
}


# ---------------------------------------------------------------------------
# Punishment Visibility: what non-punished agents observe about manager actions
# ---------------------------------------------------------------------------
class PunishVisType(str, Enum):
    PUNISH_TRANSPARENT = "punish_transparent"  # default: all agents see who was punished, by how much, and manager's explanation
    PUNISH_HIDDEN = "punish_hidden"            # only the punished agent knows; other agents see nothing; manager acts in secret
    PUNISH_ANONYMOUS = "punish_anonymous"      # punished agent sees they lost tokens but NOT who punished them (removes retaliation incentive)


# ---------------------------------------------------------------------------
# Model identifiers (used internally to select providers)
# ---------------------------------------------------------------------------
class ModelType(str, Enum):
    GPT4O = "gpt4o"
    CLAUDE = "claude"
    DEEPSEEK = "deepseek"
    GEMINI = "gemini"
    GROK = "grok"
    QWEN = "qwen"
    MOCK = "mock"
    LOCAL = "local"   
    GPT5 = "gpt5"              # newer version of the GPT family
    GPT41 = "gpt41"            # non-reasoning successor of GPT-4o (version vs reasoning)
    GPT5MIN = "gpt5min"        # GPT-5 with reasoning effort "minimal"
    DEEPSEEK31 = "deepseek31"  # newer version of DeepSeek V3   # open model served by a local vLLM server (see game/settings.py)


# Convenience mapping from CompositionType to single ModelType for homogeneous runs
HOMOGENEOUS_MODEL_MAP: dict[CompositionType, ModelType] = {
    CompositionType.COMP_HOMOGENEOUS_GPT4O: ModelType.GPT4O,
    CompositionType.COMP_HOMOGENEOUS_CLAUDE: ModelType.CLAUDE,
    CompositionType.COMP_HOMOGENEOUS_DEEPSEEK: ModelType.DEEPSEEK,
    CompositionType.COMP_HOMOGENEOUS_GEMINI: ModelType.GEMINI,
    CompositionType.COMP_HOMOGENEOUS_GROK: ModelType.GROK,
    CompositionType.COMP_HOMOGENEOUS_QWEN: ModelType.QWEN,
}

ALL_REAL_MODELS: list[ModelType] = [
    ModelType.GPT4O,
    ModelType.CLAUDE,
    ModelType.DEEPSEEK,
    ModelType.GEMINI,
    ModelType.GROK,
    ModelType.QWEN,
]
