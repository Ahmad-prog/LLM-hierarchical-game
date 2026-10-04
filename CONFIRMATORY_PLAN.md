# Confirmatory batch: analysis plan (fixed before any of these games were run)

Written 2026-10-03, before launch. Fresh games only: they are written to `results_confirm/` and none of the
1,573 earlier games is used in these tests. Games are the unit of analysis.

## Hypotheses and tests (one family of 6, Holm-corrected, alpha 0.05)

| # | Hypothesis | Model | Setups (games per side) | Outcome | Test |
|---|---|---|---|---|---|
| H1 | An elected manager raises contributions over chat only | Gemini 2.5 Flash | `batch2_comm_full` vs `batch3_mgr_elected` (12 v 12) | mean contribution per agent-round | one-sided permutation (elected > chat) |
| H2 | Same | Qwen3.8 27B | same (12 v 12) | same | same |
| H3 | Under the neutral election prompt, a salary raises vote-contingent offers | Claude Sonnet 4.5 | `batch14_neutral_elected` vs `batch14_neutral_salary` (10 v 10) | offers per 20 election-message opportunities, LLM judge (GPT-4.1 mini, the codebook of the paper) | one-sided permutation (salary > none) |
| H4a | Asking for a deal raises offers over the neutral prompt | Gemini 2.5 Flash | `batch14_neutral_elected` vs `batch3_mgr_elected` (12 v 12; the elected games are those of H1) | same as H3 | one-sided permutation (deal > neutral) |
| H4b | Same | DeepSeek V3.1 | same (10 v 10) | same | same |
| H5 | Self-hosted models break first-person unconditional commitments more often than API models | Qwen3.8 vs Gemini, chat only (games of H1, H2) | 12 v 12 | per game: broken / stated first-person unconditional commitments (rule of the paper, more than 5 tokens below); games with no such statement dropped | one-sided permutation (Qwen3.8 > Gemini) |

Permutation tests: exact when feasible, otherwise 100,000 Monte Carlo permutations, seed 20261003, difference in means.
Effect sizes reported with 95% bootstrap intervals (games resampled, 10,000 draws).
Games that fail (API errors) are re-run until the planned number is reached; no game is excluded for its outcome.
No interim analysis: tests are run once, after all games are done.
