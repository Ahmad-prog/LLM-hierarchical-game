# The Hierarchical Game: code, game logs and annotation

This repository contains the code, all 1,863 game logs, 100 fresh confirmatory games,, and the human annotation behind the paper
*The Politician, the Liar, and the Obedient Worker: Strategic Behavior of LLM Agents in Hierarchical Games*.

The **Hierarchical Game (HG)** is a 5-player, 20-round public goods game. On top of the base game it
adds:
- a **manager** who can reward and punish, and pays for this out of its own payoff;
- **elections** for the manager role;
- **public and private messages** between agents.

Every number, table and figure in the paper can be rebuilt from the logs in `results/` and the labels in
`annotation/` with one command, without an API key or a GPU.

## Reproduce the paper's numbers (about 2 minutes)

```bash
pip install -r requirements.txt
python reproduce.py --check    # rebuild everything in a temp folder and compare with paper_data/
python reproduce.py            # (re)write paper_data/ and figures/
```

`--check` prints `all numbers reproduce exactly` when every output matches the committed files. The figures
are typeset with LaTeX (Helvetica) when `latex` is on PATH with the packages helvet, sfmath, type1cm and
cm-super, as in the paper; without LaTeX they are drawn with matplotlib's own font and the same data.

| Output | Built by | Used in the paper for |
|---|---|---|
| `paper_data/paper_numbers.json` | `analysis/paper_tables.py` | Per-setup metrics: cooperation, contribution, vote-contingent offers, stated intentions, welfare, elections, valid replies (mean ± s.d. over games, counts summed) |
| `paper_data/extras.json`, `extras.txt` | `analysis/revision_extras.py` | Manager spending, election details (votes, ties, ballot position), welfare split, secondary Welch tests |
| `paper_data/revision2.json`, `revision2.txt` | `analysis/revision2.py` | Stated intentions with one rule for every model (thresholds 3/5/8, first-person commitments), offer types, re-election, model × institution ANOVA (partial η² and ω²), worker-only comparisons, Nemotron over valid replies |
| `paper_data/round2.json`, `round2.txt` | `analysis/round2.py` | Mechanism controls of batch 13 per model |
| `paper_data/round3.json`, `round3.txt` | `analysis/round3.py` | Primary contrasts (one Holm family, permutation tests), pooled manager model, neutral election prompt, mechanism controls, incumbents and random ballot, aggregate-only visibility, GPT-4.1 / GPT-5 minimal, prompt robustness, quantization, last-round analysis, population-weighted offer recall, clustered re-election, manipulation check, LLM-judge validation |
| `paper_data/scores_human.json` | `tools/score_manual_labels.py` | Agreement of the fixed rules with the human labels (Table "validation") |
| `paper_data/round5.md` | `analysis/round5.py` | Fair-ballot equivalence test, batch-16 controls, and the confirmatory batch (`CONFIRMATORY_PLAN.md`) |
| `paper_data/scores_annotator2.json`, `round6_annotation.json` | `tools/score_manual_labels.py`, `tools/score_round6.py` | The rules against the second annotator, agreement between annotators, and the LLM judge against hand labels of neutral-prompt messages |
| `paper_data/digest.txt` | `analysis/digest.py` | A readable listing of `paper_numbers.json` |
| `paper_data/appendix_tables.tex`, `round3_tables.tex` | `analysis/appendix_tables.py`, `analysis/round3_tables.py` | All generated tables, defined as LaTeX macros and placed in the paper as they are |
| `figures/rr_frontier_contribution.pdf` | `analysis/make_figures.py` | Figure 1: main API models, contribution by institution |
| `figures/rr_open_cooperation.pdf` | `analysis/make_figures.py` | Figure 2: self-hosted models, cooperation by institution |
| `figures/rr_deals_by_pay.pdf` | `analysis/make_figures.py` | Figure 3: vote-contingent offers by manager pay |
| `figures/rr_open_broken_promises.pdf` | `analysis/make_figures.py` | Appendix figure: broken stated intentions, self-hosted models |

## The game logs (`results/`)

One JSON file per game: `results/<track>/<setup>__t<k>.json`, where `<k>` is the game index.
- `results/api/` holds 820 games: the six main API models (GPT-4o, Claude Sonnet 4.5, Gemini 2.5 Flash,
  DeepSeek V3, Grok 4.3, Qwen Plus), the newer versions GPT-5 and DeepSeek V3.1, and GPT-4.1 and GPT-5 with
  reasoning effort "minimal".
- `results/oss/` holds 1,043 games: 187 to 225 for each of gpt-oss-120B, Nemotron 3 Super 120B, Ling 3.0 flash,
  Qwen3.8 27B and Gemma 4 31B, and 32 for the 4-bit NVFP4 build of Qwen3.8 (quantization check).

Each file has the following keys:

| Key | Content |
|---|---|
| `config` | The setup (manager type, communication, salary, visibility, belief, models, rounds, election-prompt wording, scripted manager policy, ballot order, …) |
| `trials[0].rounds[]` | Per round: every agent's `contribution`, `net_payoff`, `manager_adjustment`, `is_manager`, `public_message`, `model_name` (exact API or Hugging Face ID), `parse_ok` and `raw_response`; all `comm_messages` (public, private and election messages); the manager's `manager_action` (punish/reward per agent) and message; and the election record |
| `trials[0].election_history` | Every election: votes per agent, winner, and whether the incumbent defended |
| `trials[0].deception.events` | Every public message classified as an explicit, implicit or no promise, with the stated and actual contribution |
| `trials[0].agent_model_map`, `final_balances`, `duration_seconds`, `code_settings` | Model per agent, final balances, run time, and whether the manager pays for sanctions |
| `api_usage` (newest API games) | Calls, tokens and the exact OpenRouter cost of the game |

Setup names keep the batch numbering:

| Name pattern | Setup |
|---|---|
| `batch1_baseline_*` | No chat, no manager |
| `batch2_comm_{full,public,private}_*` | Chat only; full chat is the control for every managed setup |
| `batch3_mgr_{fixed,elected,rotating}_*` | Manager type |
| `batch4_hetero_all_drop_*` | Mixed groups: 5 of the 6 main API families |
| `batch5_belief_*` | Belief about the other players |
| `batch8_mgr_{salary,costly}_*` | Manager pay |
| `batch9_punish_{hidden,anonymous}_*` | Sanction visibility |
| `batch11_mgr_<manager>_<workers>` | Cross-rule |
| `batch13_info_aggregate_*` | Elected manager; agents see only the pool total, not who gave what |
| `batch13_mgr_nosanction_*` | Fixed manager without a budget (authority and messages only) |
| `batch13_mgr_autoreward_*` | Fixed manager whose sanctions follow a rule (10 reward tokens split among those who gave at least 10) |
| `batch13_mgr_badincumbent_*` | `agent_0` starts in office and punishes the two highest contributors; first election at round 6 |
| `batch13_ballot_random_*` | Elected manager; ballot shuffled per voter, ties broken at random |
| `batch14_mgr_goodincumbent_*` | `agent_0` starts in office and rewards contributors; first election at round 6 |
| `batch14_system_reward_*` | No manager; the game applies the reward rule at no cost to any player |
| `batch14_aggregate_chat_*` | Chat only; agents see only the pool total |
| `batch14_neutral_{elected,salary}_*` | Election message offered without asking for a deal ("ONE private message ... before the vote") |
| `batch14_nostrategic_{chat,elected}_*` | "Be strategic." removed from the system prompt |
| `batch15_mgr_elected_nobudget_*` | Elected manager without a budget (authority, elections and messages only) |
| `batch15_mgr_{bad,good}incumbent_rb_*` | Bad or good incumbent, with a ballot shuffled per voter and ties broken at random |
| `batch16_mgr_fixed_rewardonly_*` | Fixed manager that can reward but not punish |
| `batch16_mgr_fixed_contrib{10,20}_*` | Fixed manager whose own contribution the game sets to 10 or 20 tokens |

The suffix is the model: a main API family (`gpt4o`, `claude`, `gemini`, `deepseek`, `grok`, `qwen`), a newer
model (`gpt5`, `deepseek31`, `gpt41`, `gpt5min`), or a self-hosted tag (`gpt-oss-120b`, `nemotron-3-super-120b`,
`ling-3.0-flash`, `qwen3.8-27b`, `gemma-4-31b`, `qwen3.8-27b-nvfp4`). The exact model ID is in every agent
record (`model_name`). The `code_commit` field in `code_settings` refers to the development history of this
code, which is not part of this repository.

## Confirmatory games (`results_confirm/`)

100 fresh games that re-test six central findings. The hypotheses, outcomes, tests and sample sizes are in
`CONFIRMATORY_PLAN.md`, which was committed before any of these games ran; none of the games in `results/` enters
these tests. Their election messages are labeled by the LLM judge in `annotation/confirm_llm_judge.jsonl`.

## Human annotation (`annotation/`)

Two annotators (both authors) labelled the same 400 messages independently, drawn with a fixed seed and stratified by model, without seeing the model,
the setup, or the rules' output (`tools/build_manual_work.py` draws the sample).

| File | Content |
|---|---|
| `CODEBOOK.md` | The instructions the annotator followed (definitions and examples for every label) |
| `offers_human_labels.csv` | 200 private election messages: `is_offer`, `category` (reply, targeted, reciprocal, pledge, none), `impossible`, notes |
| `intentions_human_labels.csv` | 200 public messages sent before contributing, with what the speaker and the others gave: `statement_type` (commitment, conditional, proposal, other_round, none), `stated_amount`, `condition_met`, notes |
| `offers_blank_sheet.csv`, `intentions_blank_sheet.csv` | The same sheets as handed to the annotator, before labelling |
| `offers_key.csv`, `intentions_key.csv` | For each row: model, setup, game file, round, sender, and what the fixed rules said |
| `scores_human.json` | Precision, recall and kappa of the rules against the human labels |
| `offers_human_labels_annotator2.csv`, `intentions_human_labels_annotator2.csv` | The same 400 messages labeled independently by a second annotator with the same codebook |
| `neutral_offers_human_labels.csv`, `neutral_offers_key.csv` | 120 election messages from the neutral-prompt games, hand-labeled to check the LLM judge there; the key gives model, setup and the judge's label |
| `confirm_llm_judge.jsonl` | LLM-judge labels of the election messages of the confirmatory games |
| `offers_llm_judge.jsonl` | Labels of an LLM judge (GPT-4.1 mini, temperature 0, `tools/judge_offers.py`) that applies the same codebook to the 200 labelled messages (`human|O###`) and to every election message of the elected, pay, visibility, neutral-prompt and prompt-robustness setups (`<game file>|<round>|<sender>`) |

## Run new games

```bash
cp .env.example .env               # add your OPENROUTER_API_KEY
python tests_rerun/test_sanction_cost.py
python hg_jobs.py --track api --dry-run --out results   # list the jobs
```

- **API models** (OpenRouter): `bash scripts/run_api_games.sh` runs all API games of the paper. Our runs
  cost about US$471. The runner is resumable (finished games are skipped), runs games in parallel
  (`--workers`), and stops starting new games at `--budget-usd`; raising `--trials` adds games to a setup.
- **Open-weight models** (vLLM, one 96 GB GPU): `bash scripts/serve_open_model.sh <tag> [gpu] [port]` starts
  vLLM and runs all of that model's games (32 for `qwen3.8-27b-nvfp4`).
- **LLM judge**: `python tools/judge_offers.py results annotation/offers_human_labels.csv out.jsonl --max-usd 5`
  (about US$1).

Generation settings used in the paper:

| Setting | Value |
|---|---|
| Temperature | 0.7 for every model except GPT-5, which does not accept a temperature |
| Reasoning | Every provider's default, except GPT-5 minimal (reasoning effort "minimal") |
| Output tokens | Capped at 800 for DeepSeek V3 and 4,096 for the open models; other API models use the provider default |
| Open models | vLLM 0.30, JSON output mode, default chat templates |
| Seeds | Sampling is not seeded, so new runs reproduce the distribution of outcomes, not individual games |

On the open-model track no API key is used. Promises are therefore extracted by the regular
expression in `analysis/deception.py` rather than by GPT-4o-mini.

## Code layout

| Path | Content |
|---|---|
| `game/` | Game engine: payoffs, manager and sanctions, scripted managers and ballots (`manager.py`), elections, prompts (`agent.py`), `settings.py` (environment switches) |
| `providers/` | OpenRouter models, the local vLLM model (`local_provider.py`), JSON parsing, and per-game API cost accounting |
| `config/`, `experiments/` | Enumerations and the definition of every setup (`conditions.py`) |
| `hg_jobs.py` | Parallel, resumable runner for the API and open-model tracks |
| `analysis/` | Promise and message classification, and every analysis behind the paper |
| `tools/` | Drawing the annotation sample, scoring the human labels, and the LLM judge |
| `reproduce.py` | Rebuilds `paper_data/` and `figures/` from `results/` and `annotation/` |

The balance shown to agents is their true accumulated payoff, and the manager pays for every token it spends
on sanctions; its prompt says so (`HG_SANCTION_COST=1`, the default). Every game in `results/` was run with
this code.
