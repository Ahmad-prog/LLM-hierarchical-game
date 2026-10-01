# The Hierarchical Game: code and game logs

This repository contains the code and all 492 game logs behind the paper *The Politician, the Liar,
and the Obedient Worker: Strategic Behavior of LLM Agents in Hierarchical Games*.

The **Hierarchical Game (HG)** is a 5-player, 20-round public goods game. On top of the base game it
adds:
- a **manager** who can reward and punish, and pays for this out of its own payoff;
- **elections** for the manager role;
- **public and private messages** between agents.

Every number, table and figure in the paper can be rebuilt from the logs in `results/` with one
command, without an API key or a GPU.

## Reproduce the paper's numbers (about 1 minute)

```bash
pip install -r requirements.txt
python reproduce.py --check    # rebuild everything in a temp folder and compare with paper_data/
python reproduce.py            # (re)write paper_data/ and figures/
```

`--check` prints `all numbers reproduce exactly` when every output matches the committed files.

| Output | Built by | Used in the paper for |
|---|---|---|
| `paper_data/paper_numbers.json` | `analysis/paper_tables.py` | Per-setup metrics: cooperation, contribution, vote-buying offers, promises, welfare, elections, valid replies (mean ± s.d. over games, counts summed) |
| `paper_data/extras.json`, `extras.txt` | `analysis/revision_extras.py` | Manager spending (punishment and reward), election details (votes, ties, ballot position), promise robustness, welfare split, Welch tests |
| `paper_data/digest.txt` | `analysis/digest.py` | A readable listing of `paper_numbers.json`, used to write the text |
| `paper_data/appendix_tables.tex` | `analysis/appendix_tables.py` | All appendix tables (included in the paper as-is) |
| `figures/rr_frontier_contribution.pdf` | `analysis/make_figures.py` | Figure 1: frontier models, contribution by institution |
| `figures/rr_open_cooperation.pdf` | `analysis/make_figures.py` | Figure 2: open-weight models, cooperation by institution |
| `figures/rr_deals_by_pay.pdf` | `analysis/make_figures.py` | Figure 3: vote-buying offers by manager pay |
| `figures/rr_open_broken_promises.pdf` | `analysis/make_figures.py` | Appendix figure: broken promises, open-weight models |

Table 1 of the main text (broken promises) sums the fields `broken_explicit` / `explicit` of
`paper_numbers.json` over the six frontier models.

## The game logs (`results/`)

One JSON file per game: `results/<track>/<setup>__t<k>.json`, where `<k>` is the game index.
- `results/api/` holds 282 games. The six frontier models are GPT-4o, Claude Sonnet 4.5, Gemini 2.5
  Flash, DeepSeek V3, Grok 4.3 and Qwen Plus. The two newer versions are GPT-5 and DeepSeek V3.1.
- `results/oss/` holds 210 games: 42 for each of gpt-oss-120B, Nemotron 3 Super 120B, Ling 3.0
  flash, Qwen3.8 27B and Gemma 4 31B.

Each file has the following keys:

| Key | Content |
|---|---|
| `config` | The setup (manager type, communication, salary, visibility, belief, models, rounds, …) |
| `trials[0].rounds[]` | Per round: every agent's `contribution`, `net_payoff`, `manager_adjustment`, `is_manager`, `public_message`, `model_name` (exact API or Hugging Face ID), `parse_ok` and `raw_response`; all `comm_messages` (public, private and election messages); the manager's `manager_action` (punish/reward per agent) and message; and the election record |
| `trials[0].election_history` | Every election: votes per agent, winner, and whether the incumbent defended |
| `trials[0].deception.events` | Every public message classified as an explicit, implicit or no promise, with the stated and actual contribution |
| `trials[0].agent_model_map`, `final_balances`, `duration_seconds`, `code_settings` | Model per agent, final balances, run time, and whether the manager pays for sanctions |

Setup names keep the original batch numbering:

| Name pattern | Setup |
|---|---|
| `batch1_baseline_*` | No chat, no manager |
| `batch2_comm_{full,public,private}_*` | Chat only; full chat is the control for every managed setup |
| `batch3_mgr_{fixed,elected,rotating}_*` | Manager type |
| `batch4_hetero_all_drop_*` | Mixed groups: 5 of the 6 frontier families |
| `batch5_belief_*` | Belief about the other players |
| `batch8_mgr_{salary,costly}_*` | Manager pay |
| `batch9_punish_{hidden,anonymous}_*` | Sanction visibility |
| `batch11_mgr_<manager>_<workers>` | Cross-rule |

The suffix is the model: one of the six frontier families (`gpt4o`, `claude`, `gemini`, `deepseek`,
`grok`, `qwen`), a newer version (`gpt5`, `deepseek31`), or an open-model tag (`gpt-oss-120b`,
`nemotron-3-super-120b`, `ling-3.0-flash`, `qwen3.8-27b`, `gemma-4-31b`). The exact model ID is
in every agent record (`model_name`).

The `code_commit` field in `code_settings` refers to the development history of this code, which
is not part of this repository.

## Run new games

```bash
cp .env.example .env               # add your OPENROUTER_API_KEY
python tests_rerun/test_sanction_cost.py
python hg_jobs.py --track api --dry-run --out results   # list the jobs
```

- **API models** (OpenRouter): `bash scripts/run_api_games.sh` runs all 282 API games of the paper.
  Our runs cost about US$237. The runner is resumable (finished games are skipped), runs games in
  parallel (`--workers`), and stops starting new games at `--budget-usd`.
- **Open-weight models** (vLLM, one 96 GB GPU): `bash scripts/serve_open_model.sh <tag> [gpu] [port]`
  starts vLLM and runs that model's 42 games. Tags: `gpt-oss-120b`, `nemotron-3-super-120b`,
  `ling-3.0-flash`, `qwen3.8-27b`, `gemma-4-31b`.

Generation settings used in the paper:

| Setting | Value |
|---|---|
| Temperature | 0.7 for every model except GPT-5, which does not accept a temperature |
| Reasoning | Every provider's default; no reasoning effort is set |
| Output tokens | Capped at 800 for DeepSeek V3 and 4,096 for the open models; other API models use the provider default |
| Open models | vLLM 0.30, JSON output mode, default chat templates |
| Seeds | Sampling is not seeded, so new runs reproduce the distribution of outcomes, not individual games |

On the open-model track no API key is used. Promises are therefore extracted by the regular
expression in `analysis/deception.py` rather than by GPT-4o-mini.

## Code layout

| Path | Content |
|---|---|
| `game/` | Game engine: payoffs, manager and sanctions (`manager.py`), elections, prompts (`agent.py`), `settings.py` (environment switches) |
| `providers/` | OpenRouter models and the local vLLM model (`local_provider.py`); JSON parsing |
| `config/`, `experiments/` | Enumerations and the definition of every setup (`conditions.py`) |
| `hg_jobs.py` | Parallel, resumable runner for the API and open-model tracks |
| `analysis/` | Promise and message classification, and every analysis behind the paper |
| `reproduce.py` | Rebuilds `paper_data/` and `figures/` from `results/` |

### Corrections relative to an earlier version of this code

1. An agent's displayed balance had each round's payoff added twice.
2. The manager was not charged for sanctions.

Both are fixed here. The manager now pays for every token it spends, and its prompt says so
(`HG_SANCTION_COST=1`, the default). Every game in `results/` was run with the corrected code.
