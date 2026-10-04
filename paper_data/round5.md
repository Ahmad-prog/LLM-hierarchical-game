# Round 5 results (raw, for the write-up)

## 1. Bad vs good incumbent with a fair ballot (removed at the first election, round 6)

| Model | Bad removed | Good removed |
|---|---|---|
| gpt4o | 11/15 | 7/15 |
| gemini | 11/15 | 15/15 |
| qwen | 5/15 | 9/15 |
| deepseek | 9/15 | 8/15 |
| gpt-oss-120b | 4/20 | 6/20 |
| nemotron-3-super-120b | 11/20 | 8/20 |
| ling-3.0-flash | 2/20 | 6/20 |
| qwen3.8-27b | 17/20 | 15/20 |
| gemma-4-31b | 0/20 | 1/20 |
| **Pooled** | 70/160 | 75/160 |

- Stratified risk difference (bad − good): **-3.1 points**, 95% CI -12.1 to 5.9; 90% CI -10.7 to 4.4.
- CMH odds ratio 0.86 (95% CI 0.53–1.40), p = 0.618.
- Equivalence (TOST) at ±10 points: p = 0.0677 → not shown.
- Equivalence (TOST) at ±15 points: p = 0.0050 → equivalent.
- Equivalence (TOST) at ±20 points: p = 0.0001 → equivalent.

## 2. Batch 16 controls

### Reward-only fixed manager (mean contribution per agent-round)

| Model | Chat only | Fixed, no budget | Fixed, reward only | Fixed, full | p reward-only vs full | p reward-only vs chat |
|---|---|---|---|---|---|---|
| gemini | 12.48 ± 3.52 (n=10) | 11.98 ± 2.10 (n=5) | 17.37 ± 2.55 (n=10) | 13.11 ± 3.56 (n=10) | 0.009 | 0.003 |
| gpt-oss-120b | 6.95 ± 3.76 (n=8) | 7.53 ± 4.30 (n=5) | 6.11 ± 1.35 (n=10) | 12.83 ± 2.21 (n=8) | 0.000 | 0.659 |
| qwen3.8-27b | 6.17 ± 1.36 (n=8) | 4.04 ± 1.44 (n=5) | 9.10 ± 2.50 (n=10) | 9.31 ± 2.06 (n=8) | 0.858 | 0.002 |

### Manager contribution fixed by the experimenter (workers' mean contribution, manager excluded)

| Model | Manager gives 10 | Manager gives 20 | Diff (20 − 10) | 95% CI | p (two-sided) | Own fixed manager (workers) |
|---|---|---|---|---|---|---|
| qwen | 12.15 ± 1.49 (n=10) | 16.27 ± 1.32 (n=10) | +4.12 | [2.95, 5.31] | 0.0000 | 14.00 ± 3.07 (n=10) |
| gemini | 12.50 ± 2.48 (n=10) | 19.11 ± 1.38 (n=10) | +6.61 | [4.84, 8.13] | 0.0000 | 13.10 ± 3.55 (n=10) |
| gpt-oss-120b | 14.53 ± 1.90 (n=10) | 14.25 ± 2.79 (n=10) | -0.28 | [-2.24, 1.74] | 0.7945 | 13.42 ± 2.22 (n=8) |
| qwen3.8-27b | 8.21 ± 0.81 (n=10) | 13.74 ± 2.37 (n=10) | +5.54 | [4.03, 6.93] | 0.0000 | 9.06 ± 1.91 (n=8) |

## 3. Confirmatory batch (fresh games, plan fixed in review_plans/CONFIRMATORY_PLAN.md)

Judge labels: 848 messages.

| Hypothesis | Control | Treatment | Diff | 95% bootstrap CI | one-sided p | Holm p | Confirmed |
|---|---|---|---|---|---|---|---|
| H1 Gemini: elected > chat only (contribution) | 13.35 ± 3.26 (n=12) | 17.86 ± 3.35 (n=12) | +4.51 | [1.87, 6.9] | 0.0020 | 0.0020 | yes |
| H2 Qwen3.8: elected > chat only (contribution) | 5.95 ± 2.07 (n=12) | 12.71 ± 2.25 (n=12) | +6.77 | [5.04, 8.38] | 0.0000 | 0.0000 | yes |
| H3 Claude, neutral prompt: salary > none (judged offers, % of opportunities) | 1.00 ± 3.16 (n=10) | 15.00 ± 8.50 (n=10) | +14.00 | [8.5, 19.0] | 0.0002 | 0.0003 | yes |
| H4a Gemini: deal prompt > neutral (judged offers) | 0.00 ± 0.00 (n=12) | 10.83 ± 7.33 (n=12) | +10.83 | [7.08, 15.0] | 0.0000 | 0.0000 | yes |
| H4b DeepSeek V3.1: deal prompt > neutral (judged offers) | 26.00 ± 7.38 (n=10) | 78.50 ± 6.26 (n=10) | +52.50 | [47.0, 58.0] | 0.0000 | 0.0000 | yes |
| H5 broken first-person commitments, chat only: Qwen3.8 > Gemini (% per game) | 0.00 ± 0.00 (n=12) | 20.34 ± 8.32 (n=12) | +20.34 | [15.45, 24.47] | 0.0000 | 0.0000 | yes |
