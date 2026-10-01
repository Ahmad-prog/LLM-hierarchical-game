"""
All predefined experiment conditions as specified in the project spec.

Batch 1: Baseline — MGR_NONE × COMM_NONE × COMP_HOMOGENEOUS_{each model} × BELIEF_ALL_AI × TEMP_MEDIUM × HIST_FULL
Batch 2: Comm sweep — MGR_NONE × {COMM modes} × COMP_HOMOGENEOUS_GPT4O × BELIEF_ALL_AI × TEMP_MEDIUM × HIST_FULL
Batch 3: Mgmt sweep — {MGR_FIXED, MGR_ELECTED, MGR_ROTATING} × MGR_FULL × COMM_FULL × COMP_HOMOGENEOUS_{all 6} × BELIEF_ALL_AI
Batch 4: Heterogeneous all — MGR_ELECTED × MGR_FULL × COMM_FULL × COMP_HETEROGENEOUS_ALL × BELIEF_ALL_AI
Batch 5: Belief sweep — MGR_ELECTED × COMM_FULL × COMP_HOMOGENEOUS_GPT4O × {BELIEF modes} × TEMP_MEDIUM
Batch 6: Heterogeneous pairs — MGR_ELECTED × COMM_FULL × COMP_HETEROGENEOUS_PAIRS × BELIEF_ALL_AI
         15 configs (3A+2B orientation); set reverse_pairs=True in runner for the 2A+3B direction.
"""

from __future__ import annotations
from itertools import combinations

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
    HOMOGENEOUS_MODEL_MAP,
    ALL_REAL_MODELS,
    IdentityType,
    MgrSalaryType,
    PunishVisType,
)
from experiments.config import ExperimentConfig


# ---------------------------------------------------------------------------
# Batch 1: Baseline
# MGR_NONE × COMM_NONE × COMP_HOMOGENEOUS_{each model} × BELIEF_ALL_AI × TEMP_MEDIUM × HIST_FULL
# ---------------------------------------------------------------------------

def generate_batch1_baseline() -> list[ExperimentConfig]:
    """6 configs — one per model, all homogeneous, no manager, no comm."""
    configs = []
    for ctype, model in HOMOGENEOUS_MODEL_MAP.items():
        configs.append(ExperimentConfig(
            name=f"batch1_baseline_{model.value}",
            batch=1,
            belief=BeliefType.BELIEF_ALL_AI,
            composition=ctype,
            comm_type=CommType.COMM_NONE,
            mgr_type=MgrType.MGR_NONE,
            mgr_power=MgrPowerType.MGR_NONE,
            info_type=InfoType.INFO_FULL,
            persona=PersonaType.PERSONA_NONE,
            temp_type=TempType.TEMP_MEDIUM,
            hist_type=HistType.HIST_FULL,
            agent_models=[model] * 5,
            num_agents=5,
        ))
    return configs


# ---------------------------------------------------------------------------
# Batch 2: Communication sweep
# MGR_NONE × {COMM_NONE, COMM_PUBLIC, COMM_PRIVATE, COMM_FULL} × GPT4O × BELIEF_ALL_AI × TEMP_MEDIUM × HIST_FULL
# ---------------------------------------------------------------------------

def generate_batch2_comm_sweep() -> list[ExperimentConfig]:
    """4 configs — GPT-4o only, varying communication mode."""
    comm_modes = [
        CommType.COMM_NONE,
        CommType.COMM_PUBLIC,
        CommType.COMM_PRIVATE,
        CommType.COMM_FULL,
    ]
    configs = []
    for comm in comm_modes:
        configs.append(ExperimentConfig(
            name=f"batch2_comm_{comm.value.replace('comm_', '')}_gpt4o",
            batch=2,
            belief=BeliefType.BELIEF_ALL_AI,
            composition=CompositionType.COMP_HOMOGENEOUS_GPT4O,
            comm_type=comm,
            mgr_type=MgrType.MGR_NONE,
            mgr_power=MgrPowerType.MGR_NONE,
            info_type=InfoType.INFO_FULL,
            persona=PersonaType.PERSONA_NONE,
            temp_type=TempType.TEMP_MEDIUM,
            hist_type=HistType.HIST_FULL,
            agent_models=[ModelType.GPT4O] * 5,
            num_agents=5,
        ))
    return configs


# ---------------------------------------------------------------------------
# Batch 3: Management sweep
# {MGR_FIXED, MGR_ELECTED, MGR_ROTATING} × MGR_FULL × COMM_FULL × COMP_HOMOGENEOUS_{all 6 models} × BELIEF_ALL_AI
# → 3 mgr_types × 6 models = 18 configs
# ---------------------------------------------------------------------------

def generate_batch3_mgmt_sweep() -> list[ExperimentConfig]:
    """18 configs — 3 management types × 6 models."""
    mgr_types = [MgrType.MGR_FIXED, MgrType.MGR_ELECTED, MgrType.MGR_ROTATING]
    configs = []
    for mgr in mgr_types:
        for ctype, model in HOMOGENEOUS_MODEL_MAP.items():
            configs.append(ExperimentConfig(
                name=f"batch3_mgr_{mgr.value.replace('mgr_', '')}_{model.value}",
                batch=3,
                belief=BeliefType.BELIEF_ALL_AI,
                composition=ctype,
                comm_type=CommType.COMM_FULL,
                mgr_type=mgr,
                mgr_power=MgrPowerType.MGR_FULL,
                info_type=InfoType.INFO_FULL,
                persona=PersonaType.PERSONA_NONE,
                temp_type=TempType.TEMP_MEDIUM,
                hist_type=HistType.HIST_FULL,
                agent_models=[model] * 5,
                num_agents=5,
                election_frequency=5,
            ))
    return configs


# ---------------------------------------------------------------------------
# Batch 4: Heterogeneous all — 5-of-6 mixes
# MGR_ELECTED × MGR_FULL × COMM_FULL × COMP_HETEROGENEOUS_ALL × BELIEF_ALL_AI
# C(6,5) = 6 configs, each dropping one model
# ---------------------------------------------------------------------------

def generate_batch4_heterogeneous_all() -> list[ExperimentConfig]:
    """
    6 configs — each run uses 5 of the 6 models (one agent per model).
    One model is dropped per config so every 5-of-6 combination is covered.
    """
    configs = []
    for drop_model in ALL_REAL_MODELS:
        selected = [m for m in ALL_REAL_MODELS if m != drop_model]
        assert len(selected) == 5
        name = f"batch4_hetero_all_drop_{drop_model.value}"
        configs.append(ExperimentConfig(
            name=name,
            batch=4,
            belief=BeliefType.BELIEF_ALL_AI,
            composition=CompositionType.COMP_HETEROGENEOUS_ALL,
            comm_type=CommType.COMM_FULL,
            mgr_type=MgrType.MGR_ELECTED,
            mgr_power=MgrPowerType.MGR_FULL,
            info_type=InfoType.INFO_FULL,
            persona=PersonaType.PERSONA_NONE,
            temp_type=TempType.TEMP_MEDIUM,
            hist_type=HistType.HIST_FULL,
            agent_models=selected,
            num_agents=5,
            election_frequency=5,
        ))
    return configs


# ---------------------------------------------------------------------------
# Batch 5: Belief sweep
# MGR_ELECTED × COMM_FULL × COMP_HOMOGENEOUS_GPT4O × {BELIEF modes} × TEMP_MEDIUM
# ---------------------------------------------------------------------------

def generate_batch5_belief_sweep() -> list[ExperimentConfig]:
    """4 configs — GPT-4o, elected manager, varying belief about co-players."""
    belief_modes = [
        BeliefType.BELIEF_UNKNOWN,
        BeliefType.BELIEF_ALL_AI,
        BeliefType.BELIEF_ALL_HUMAN,
        BeliefType.BELIEF_MIXED,
    ]
    configs = []
    for belief in belief_modes:
        configs.append(ExperimentConfig(
            name=f"batch5_belief_{belief.value.replace('belief_', '')}_gpt4o",
            batch=5,
            belief=belief,
            composition=CompositionType.COMP_HOMOGENEOUS_GPT4O,
            comm_type=CommType.COMM_FULL,
            mgr_type=MgrType.MGR_ELECTED,
            mgr_power=MgrPowerType.MGR_FULL,
            info_type=InfoType.INFO_FULL,
            persona=PersonaType.PERSONA_NONE,
            temp_type=TempType.TEMP_MEDIUM,
            hist_type=HistType.HIST_FULL,
            agent_models=[ModelType.GPT4O] * 5,
            num_agents=5,
            election_frequency=5,
        ))
    return configs


# ---------------------------------------------------------------------------
# Batch 6: Heterogeneous pairs (3A + 2B orientation)
# MGR_ELECTED × COMM_FULL × COMP_HETEROGENEOUS_PAIRS × BELIEF_ALL_AI
# C(6,2) = 15 model pairs; each run has 3 agents of model A and 2 of model B.
# Set reverse_pairs=True on a config to swap to 2A+3B for the reverse direction.
# ---------------------------------------------------------------------------

def generate_batch6_heterogeneous_pairs(
    reverse_pairs: bool = False,
) -> list[ExperimentConfig]:
    """
    15 configs — all unique unordered pairs of models with a 3:2 majority split.

    Parameters
    ----------
    reverse_pairs : bool
        If False (default): first model gets 3 agents, second gets 2.
        If True: first model gets 2, second gets 3 (minority/majority swap).
        Adding 15 reverse configs is trivial — call with reverse_pairs=True.
    """
    configs = []
    for model_a, model_b in combinations(ALL_REAL_MODELS, 2):
        if reverse_pairs:
            agent_models = [model_a] * 2 + [model_b] * 3
            label = f"batch6_pairs_{model_a.value}2_{model_b.value}3"
        else:
            agent_models = [model_a] * 3 + [model_b] * 2
            label = f"batch6_pairs_{model_a.value}3_{model_b.value}2"

        configs.append(ExperimentConfig(
            name=label,
            batch=6,
            belief=BeliefType.BELIEF_ALL_AI,
            composition=CompositionType.COMP_HETEROGENEOUS_PAIRS,
            comm_type=CommType.COMM_FULL,
            mgr_type=MgrType.MGR_ELECTED,
            mgr_power=MgrPowerType.MGR_FULL,
            info_type=InfoType.INFO_FULL,
            persona=PersonaType.PERSONA_NONE,
            temp_type=TempType.TEMP_MEDIUM,
            hist_type=HistType.HIST_FULL,
            agent_models=agent_models,
            num_agents=5,
            election_frequency=5,
            reverse_pairs=reverse_pairs,
        ))
    return configs


# ---------------------------------------------------------------------------
# Master generator
# ---------------------------------------------------------------------------

def generate_all_conditions(
    include_reverse_pairs: bool = False,
    include_new_batches: bool = True,
) -> list[ExperimentConfig]:
    """
    Generate the complete set of experiment conditions (all batches 1–10).

    Parameters
    ----------
    include_reverse_pairs : bool
        If True, also adds the 15 reverse-pair configs for Batch 6.
    include_new_batches : bool
        If True (default), includes Batches 7–10 (identity, salary, transparency, belief).
    """
    configs: list[ExperimentConfig] = []
    configs.extend(generate_batch1_baseline())                  # 6
    configs.extend(generate_batch2_comm_sweep())                # 4  (GPT-4o only)
    configs.extend(generate_batch2_comm_sweep_expanded())       # 20 (other 5 models)
    configs.extend(generate_batch3_mgmt_sweep())                # 18
    configs.extend(generate_batch4_heterogeneous_all())         # 6
    configs.extend(generate_batch5_belief_sweep())              # 4  (GPT-4o only)
    configs.extend(generate_batch5_belief_sweep_expanded())     # 16 (other 5 models)
    configs.extend(generate_batch6_heterogeneous_pairs(reverse_pairs=False))  # 15

    if include_reverse_pairs:
        configs.extend(generate_batch6_heterogeneous_pairs(reverse_pairs=True))  # +15

    if include_new_batches:
        configs.extend(generate_batch7_identity_awareness())   # 6
        configs.extend(generate_batch8_manager_incentive())    # 18
        configs.extend(generate_batch8_costly5())              # 6  ← MGR_COSTLY_5 control
        configs.extend(generate_batch9_punish_transparency())  # 18
        configs.extend(generate_batch10_belief_expanded())     # 4
        configs.extend(generate_batch11_cross_model_mgr())     # 6
        configs.extend(generate_batch12_hetero_no_manager())   # 7

    return configs


def generate_batch7_identity_awareness() -> list[ExperimentConfig]:
    """6 configs — IDENTITY_AWARE heterogeneous 5-of-6 mixes."""
    configs = []
    for drop_model in ALL_REAL_MODELS:
        selected = [m for m in ALL_REAL_MODELS if m != drop_model]
        configs.append(ExperimentConfig(
            name=f"batch7_identity_aware_drop_{drop_model.value}",
            batch=7,
            belief=BeliefType.BELIEF_ALL_AI,
            composition=CompositionType.COMP_HETEROGENEOUS_ALL,
            comm_type=CommType.COMM_FULL,
            mgr_type=MgrType.MGR_ELECTED,
            mgr_power=MgrPowerType.MGR_FULL,
            info_type=InfoType.INFO_FULL,
            persona=PersonaType.PERSONA_NONE,
            temp_type=TempType.TEMP_MEDIUM,
            hist_type=HistType.HIST_FULL,
            agent_models=selected,
            num_agents=5,
            election_frequency=5,
            identity_type=IdentityType.IDENTITY_AWARE,
        ))
    return configs


def generate_batch8_manager_incentive() -> list[ExperimentConfig]:
    """18 configs — MGR_NO_SALARY + MGR_SALARY + MGR_COSTLY × 6 models."""
    salary_types = [MgrSalaryType.MGR_NO_SALARY, MgrSalaryType.MGR_SALARY, MgrSalaryType.MGR_COSTLY]
    configs = []
    for salary_type in salary_types:
        tag = salary_type.value.replace("mgr_", "")
        for ctype, model in HOMOGENEOUS_MODEL_MAP.items():
            configs.append(ExperimentConfig(
                name=f"batch8_mgr_{tag}_{model.value}",
                batch=8,
                belief=BeliefType.BELIEF_ALL_AI,
                composition=ctype,
                comm_type=CommType.COMM_FULL,
                mgr_type=MgrType.MGR_ELECTED,
                mgr_power=MgrPowerType.MGR_FULL,
                info_type=InfoType.INFO_FULL,
                persona=PersonaType.PERSONA_NONE,
                temp_type=TempType.TEMP_MEDIUM,
                hist_type=HistType.HIST_FULL,
                agent_models=[model] * 5,
                num_agents=5,
                election_frequency=5,
                mgr_salary_type=salary_type,
            ))
    return configs


_B9_COMP_MAP = {
    ModelType.GPT4O:    CompositionType.COMP_HOMOGENEOUS_GPT4O,
    ModelType.CLAUDE:   CompositionType.COMP_HOMOGENEOUS_CLAUDE,
    ModelType.QWEN:     CompositionType.COMP_HOMOGENEOUS_QWEN,
    ModelType.GROK:     CompositionType.COMP_HOMOGENEOUS_GROK,
    ModelType.GEMINI:   CompositionType.COMP_HOMOGENEOUS_GEMINI,
    ModelType.DEEPSEEK: CompositionType.COMP_HOMOGENEOUS_DEEPSEEK,
}

def generate_batch9_punish_transparency() -> list[ExperimentConfig]:
    """18 configs — PUNISH_TRANSPARENT + PUNISH_HIDDEN + PUNISH_ANONYMOUS × 6 models."""
    vis_types = [PunishVisType.PUNISH_TRANSPARENT, PunishVisType.PUNISH_HIDDEN, PunishVisType.PUNISH_ANONYMOUS]
    models = [ModelType.GPT4O, ModelType.CLAUDE, ModelType.QWEN, ModelType.GROK, ModelType.GEMINI, ModelType.DEEPSEEK]
    configs = []
    for vis_type in vis_types:
        tag = vis_type.value.replace("punish_", "")
        for model in models:
            configs.append(ExperimentConfig(
                name=f"batch9_punish_{tag}_{model.value}",
                batch=9,
                belief=BeliefType.BELIEF_ALL_AI,
                composition=_B9_COMP_MAP[model],
                comm_type=CommType.COMM_FULL,
                mgr_type=MgrType.MGR_ELECTED,
                mgr_power=MgrPowerType.MGR_FULL,
                info_type=InfoType.INFO_FULL,
                persona=PersonaType.PERSONA_NONE,
                temp_type=TempType.TEMP_MEDIUM,
                hist_type=HistType.HIST_FULL,
                agent_models=[model] * 5,
                num_agents=5,
                election_frequency=5,
                punish_vis_type=vis_type,
            ))
    return configs


def generate_batch8_costly5() -> list[ExperimentConfig]:
    """6 configs — MGR_COSTLY_5 (-5 tokens) × 6 models (matched-magnitude control vs MGR_SALARY +5)."""
    configs = []
    for ctype, model in HOMOGENEOUS_MODEL_MAP.items():
        configs.append(ExperimentConfig(
            name=f"batch8_mgr_costly5_{model.value}",
            batch=8,
            belief=BeliefType.BELIEF_ALL_AI,
            composition=ctype,
            comm_type=CommType.COMM_FULL,
            mgr_type=MgrType.MGR_ELECTED,
            mgr_power=MgrPowerType.MGR_FULL,
            info_type=InfoType.INFO_FULL,
            persona=PersonaType.PERSONA_NONE,
            temp_type=TempType.TEMP_MEDIUM,
            hist_type=HistType.HIST_FULL,
            agent_models=[model] * 5,
            num_agents=5,
            election_frequency=5,
            mgr_salary_type=MgrSalaryType.MGR_COSTLY_5,
        ))
    return configs


# ---------------------------------------------------------------------------
# Batch 2 expansion: comm sweep for Claude, Gemini, DeepSeek, Grok, Qwen
# Same structure as generate_batch2_comm_sweep but for the other 5 models.
# 5 models × 4 comm levels = 20 new configs
# ---------------------------------------------------------------------------

def generate_batch2_comm_sweep_expanded() -> list[ExperimentConfig]:
    """20 configs — comm sweep for all non-GPT4O models."""
    comm_modes = [
        CommType.COMM_NONE,
        CommType.COMM_PUBLIC,
        CommType.COMM_PRIVATE,
        CommType.COMM_FULL,
    ]
    NON_GPT4O = {
        CompositionType.COMP_HOMOGENEOUS_CLAUDE:    ModelType.CLAUDE,
        CompositionType.COMP_HOMOGENEOUS_GEMINI:    ModelType.GEMINI,
        CompositionType.COMP_HOMOGENEOUS_DEEPSEEK:  ModelType.DEEPSEEK,
        CompositionType.COMP_HOMOGENEOUS_GROK:      ModelType.GROK,
        CompositionType.COMP_HOMOGENEOUS_QWEN:      ModelType.QWEN,
    }
    configs = []
    for ctype, model in NON_GPT4O.items():
        for comm in comm_modes:
            comm_tag = comm.value.replace("comm_", "")
            configs.append(ExperimentConfig(
                name=f"batch2_comm_{comm_tag}_{model.value}",
                batch=2,
                belief=BeliefType.BELIEF_ALL_AI,
                composition=ctype,
                comm_type=comm,
                mgr_type=MgrType.MGR_NONE,
                mgr_power=MgrPowerType.MGR_NONE,
                info_type=InfoType.INFO_FULL,
                persona=PersonaType.PERSONA_NONE,
                temp_type=TempType.TEMP_MEDIUM,
                hist_type=HistType.HIST_FULL,
                agent_models=[model] * 5,
                num_agents=5,
            ))
    return configs


# ---------------------------------------------------------------------------
# Batch 5 expansion: belief sweep for Claude, Gemini, DeepSeek, Grok, Qwen
# Grok + Qwen already have all_ai and all_human in B10 → only run unknown + mixed.
# Claude + Gemini + DeepSeek: 4 beliefs each = 12 configs
# Grok + Qwen: 2 beliefs each (unknown, mixed) = 4 configs
# Total = 16 new configs
# ---------------------------------------------------------------------------

def generate_batch5_belief_sweep_expanded() -> list[ExperimentConfig]:
    """16 configs — belief sweep for non-GPT4O models (Grok/Qwen skip B10-covered beliefs)."""
    belief_modes_full = [
        BeliefType.BELIEF_UNKNOWN,
        BeliefType.BELIEF_ALL_AI,
        BeliefType.BELIEF_ALL_HUMAN,
        BeliefType.BELIEF_MIXED,
    ]
    # Grok + Qwen already have all_ai and all_human covered by B10
    belief_modes_partial = [
        BeliefType.BELIEF_UNKNOWN,
        BeliefType.BELIEF_MIXED,
    ]

    model_belief_map = {
        (CompositionType.COMP_HOMOGENEOUS_CLAUDE,   ModelType.CLAUDE):    belief_modes_full,
        (CompositionType.COMP_HOMOGENEOUS_GEMINI,   ModelType.GEMINI):    belief_modes_full,
        (CompositionType.COMP_HOMOGENEOUS_DEEPSEEK, ModelType.DEEPSEEK):  belief_modes_full,
        (CompositionType.COMP_HOMOGENEOUS_GROK,     ModelType.GROK):      belief_modes_partial,
        (CompositionType.COMP_HOMOGENEOUS_QWEN,     ModelType.QWEN):      belief_modes_partial,
    }

    configs = []
    for (ctype, model), beliefs in model_belief_map.items():
        for belief in beliefs:
            btag = belief.value.replace("belief_", "")
            configs.append(ExperimentConfig(
                name=f"batch5_belief_{btag}_{model.value}",
                batch=5,
                belief=belief,
                composition=ctype,
                comm_type=CommType.COMM_FULL,
                mgr_type=MgrType.MGR_ELECTED,
                mgr_power=MgrPowerType.MGR_FULL,
                info_type=InfoType.INFO_FULL,
                persona=PersonaType.PERSONA_NONE,
                temp_type=TempType.TEMP_MEDIUM,
                hist_type=HistType.HIST_FULL,
                agent_models=[model] * 5,
                num_agents=5,
                election_frequency=5,
            ))
    return configs


def generate_batch10_belief_expanded() -> list[ExperimentConfig]:
    """4 configs — Qwen + Grok under AI/Human belief (B5 already covers GPT-4o)."""
    configs = []
    for belief in [BeliefType.BELIEF_ALL_AI, BeliefType.BELIEF_ALL_HUMAN]:
        btag = belief.value.replace("belief_", "")
        for model, ctype in [
            (ModelType.QWEN, CompositionType.COMP_HOMOGENEOUS_QWEN),
            (ModelType.GROK, CompositionType.COMP_HOMOGENEOUS_GROK),
        ]:
            configs.append(ExperimentConfig(
                name=f"batch10_belief_{btag}_{model.value}",
                batch=10,
                belief=belief,
                composition=ctype,
                comm_type=CommType.COMM_FULL,
                mgr_type=MgrType.MGR_ELECTED,
                mgr_power=MgrPowerType.MGR_FULL,
                info_type=InfoType.INFO_FULL,
                persona=PersonaType.PERSONA_NONE,
                temp_type=TempType.TEMP_MEDIUM,
                hist_type=HistType.HIST_FULL,
                agent_models=[model] * 5,
                num_agents=5,
                election_frequency=5,
            ))
    return configs


# ---------------------------------------------------------------------------
# Batch 11: Cross-model management
# One forced manager of model A governing 4 workers of model B.
# MGR_FIXED × MGR_FULL × COMM_FULL × BELIEF_ALL_AI × IDENTITY_BLIND
# 6 manager→workers pairs
# ---------------------------------------------------------------------------

def generate_batch11_cross_model_mgr() -> list[ExperimentConfig]:
    """6 configs — cross-model fixed management: manager model A, 4 workers model B."""
    pairs = [
        ("claude",   ModelType.CLAUDE,   ModelType.QWEN),
        ("qwen",     ModelType.QWEN,     ModelType.CLAUDE),
        ("grok",     ModelType.GROK,     ModelType.QWEN),
        ("gpt4o",    ModelType.GPT4O,    ModelType.QWEN),
        ("claude2",  ModelType.CLAUDE,   ModelType.GROK),
        ("grok2",    ModelType.GROK,     ModelType.CLAUDE),
    ]
    # Better name mapping
    name_pairs = [
        ("claude_qwen",  ModelType.CLAUDE,   ModelType.QWEN),
        ("qwen_claude",  ModelType.QWEN,     ModelType.CLAUDE),
        ("grok_qwen",    ModelType.GROK,     ModelType.QWEN),
        ("gpt4o_qwen",   ModelType.GPT4O,    ModelType.QWEN),
        ("claude_grok",  ModelType.CLAUDE,   ModelType.GROK),
        ("grok_claude",  ModelType.GROK,     ModelType.CLAUDE),
    ]
    configs = []
    for tag, mgr_model, worker_model in name_pairs:
        configs.append(ExperimentConfig(
            name=f"batch11_mgr_{tag}",
            batch=11,
            belief=BeliefType.BELIEF_ALL_AI,
            composition=CompositionType.COMP_HETEROGENEOUS_PAIRS,
            comm_type=CommType.COMM_FULL,
            mgr_type=MgrType.MGR_FIXED,
            mgr_power=MgrPowerType.MGR_FULL,
            info_type=InfoType.INFO_FULL,
            persona=PersonaType.PERSONA_NONE,
            temp_type=TempType.TEMP_MEDIUM,
            hist_type=HistType.HIST_FULL,
            # agent_0 = manager (MGR_FIXED pins agent_ids[0])
            agent_models=[mgr_model] + [worker_model] * 4,
            num_agents=5,
        ))
    return configs


# ---------------------------------------------------------------------------
# Batch 12: Heterogeneous groups WITHOUT manager
# Same 5-of-6 mixes as B4 but MGR_NONE — isolates peer contagion from enforcement.
# Also includes 4 Claude + 1 Qwen majority-influence cell.
# MGR_NONE × COMM_FULL × COMP_HETEROGENEOUS_ALL × BELIEF_ALL_AI
# 7 configs total
# ---------------------------------------------------------------------------

def generate_batch12_hetero_no_manager() -> list[ExperimentConfig]:
    """7 configs — heterogeneous groups with no manager (B4 clone without governance)."""
    configs = []
    # (a) Six 5-of-6 mixes (same as B4 but MGR_NONE)
    for drop_model in ALL_REAL_MODELS:
        selected = [m for m in ALL_REAL_MODELS if m != drop_model]
        configs.append(ExperimentConfig(
            name=f"batch12_hetero_nomgr_drop_{drop_model.value}",
            batch=12,
            belief=BeliefType.BELIEF_ALL_AI,
            composition=CompositionType.COMP_HETEROGENEOUS_ALL,
            comm_type=CommType.COMM_FULL,
            mgr_type=MgrType.MGR_NONE,
            mgr_power=MgrPowerType.MGR_NONE,
            info_type=InfoType.INFO_FULL,
            persona=PersonaType.PERSONA_NONE,
            temp_type=TempType.TEMP_MEDIUM,
            hist_type=HistType.HIST_FULL,
            agent_models=selected,
            num_agents=5,
        ))
    # (b) Majority-influence cell: 4 Claude + 1 Qwen, no manager
    configs.append(ExperimentConfig(
        name="batch12_claude4_qwen1_nomgr",
        batch=12,
        belief=BeliefType.BELIEF_ALL_AI,
        composition=CompositionType.COMP_HETEROGENEOUS_PAIRS,
        comm_type=CommType.COMM_FULL,
        mgr_type=MgrType.MGR_NONE,
        mgr_power=MgrPowerType.MGR_NONE,
        info_type=InfoType.INFO_FULL,
        persona=PersonaType.PERSONA_NONE,
        temp_type=TempType.TEMP_MEDIUM,
        hist_type=HistType.HIST_FULL,
        agent_models=[ModelType.CLAUDE] * 4 + [ModelType.QWEN],
        num_agents=5,
    ))
    return configs


def generate_batch(batch_num: int, include_reverse_pairs: bool = False) -> list[ExperimentConfig]:
    """Generate configs for a single batch number (1–12)."""
    generators = {
        1:  generate_batch1_baseline,
        2:  lambda: generate_batch2_comm_sweep() + generate_batch2_comm_sweep_expanded(),
        3:  generate_batch3_mgmt_sweep,
        4:  generate_batch4_heterogeneous_all,
        5:  lambda: generate_batch5_belief_sweep() + generate_batch5_belief_sweep_expanded(),
        6:  lambda: generate_batch6_heterogeneous_pairs(include_reverse_pairs),
        7:  generate_batch7_identity_awareness,
        8:  generate_batch8_manager_incentive,
        9:  generate_batch9_punish_transparency,
        10: generate_batch10_belief_expanded,
        11: generate_batch11_cross_model_mgr,
        12: generate_batch12_hetero_no_manager,
    }
    gen = generators.get(batch_num)
    if gen is None:
        raise ValueError(f"Unknown batch number: {batch_num}. Must be 1–12.")
    return gen()
