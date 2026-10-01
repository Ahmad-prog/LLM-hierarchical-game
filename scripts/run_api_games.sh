#!/bin/bash
# All API games of the paper (282 games). Needs OPENROUTER_API_KEY in .env.
# The runner is resumable: finished games in results/api are skipped, so it is safe to re-run.
# The paper's API runs cost about US$237 in total (GPT-5 about $4.6 per game).
set -e
cd "$(dirname "$0")/.."

# Six frontier models: baseline, manager type, manager pay, sanction visibility,
# mixed groups, chat-only control, cross-rule (3 games per setup)
python hg_jobs.py --track api --trials 3 --workers 10 --budget-usd 400 --out results

# Manager type (fixed, elected, rotating) to 5 games per setup
python hg_jobs.py --track api --only batch3_mgr --trials 5 --workers 10 --budget-usd 400 --out results

# Newer versions (GPT-5, DeepSeek V3.1) on the key setups, 3 games each
python hg_jobs.py --track api --swap-model gpt5 --trials 3 --workers 10 --budget-usd 400 --out results
python hg_jobs.py --track api --swap-model deepseek31 --trials 3 --workers 10 --budget-usd 400 --out results
