#!/bin/bash
# All API games of the paper (642 games). Needs OPENROUTER_API_KEY in .env.
# The runner is resumable: finished games in results/api are skipped, so it is safe to re-run, and raising
# --trials adds games to a setup. Each game records its exact OpenRouter cost in its log (`api_usage`).
# The paper's API runs cost about US$393 in total (GPT-5 about $4.6 per game, Claude about $1.8, most others < $0.8).
set -e
cd "$(dirname "$0")/.."
B="--budget-usd 450 --out results"
api()  { python hg_jobs.py --track api $B "$@"; }
swap() { m=$1; shift; python hg_jobs.py --track api --swap-model $m $B "$@"; }

# 1. Six main API models: baseline, manager type, manager pay, sanction visibility, mixed groups, chat-only
#    control, cross-rule (3 games per setup); manager type to 5 games
api --trials 3 --workers 10
api --only batch3_mgr --trials 5 --workers 10

# 2. Newer versions (GPT-5, DeepSeek V3.1): baseline, manager type, pay, visibility, chat-only control (3 games)
for m in gpt5 deepseek31; do
  swap $m --only batch1_baseline,batch2_comm_full,batch3_mgr,batch8_mgr,batch9_punish --trials 3 --workers 10
done

# 3. Replication of the main contrasts
api --only batch1_baseline --trials 6 --workers 10                                   # no chat: 6 games
for m in claude grok; do
  api --only batch2_comm_full_$m --trials 6 --workers 6
  api --only batch3_mgr_elected_$m,batch8_mgr_salary_$m,batch8_mgr_costly_$m --trials 8 --workers 8
done
for m in gpt4o gemini qwen; do
  api --only batch2_comm_full_$m,batch3_mgr_fixed_$m,batch3_mgr_elected_$m --trials 10 --workers 8
done
api --only batch2_comm_full_deepseek,batch3_mgr_fixed_deepseek,batch3_mgr_elected_deepseek --trials 8 --workers 6
api --only batch8_mgr_salary_gemini,batch8_mgr_costly_gemini --trials 8 --workers 6
swap deepseek31 --only batch3_mgr_elected,batch8_mgr_salary,batch8_mgr_costly --trials 8 --workers 8

# 4. Controls (batches 13 and 14)
M13="batch13_mgr_nosanction,batch13_mgr_autoreward,batch14_system_reward,batch13_mgr_badincumbent,batch14_mgr_goodincumbent,batch13_ballot_random,batch13_info_aggregate,batch14_aggregate_chat"
for m in gpt4o gemini qwen; do swap $m --only $M13 --trials 5 --workers 8; done
swap deepseek --only batch13_mgr_badincumbent,batch14_mgr_goodincumbent,batch13_ballot_random --trials 5 --workers 6
for m in claude gemini grok deepseek31; do swap $m --only batch14_neutral --trials 5 --workers 6; done
for m in gpt4o claude gemini; do swap $m --only batch14_nostrategic --trials 3 --workers 6; done

# 5. Version vs reasoning: GPT-4.1 and GPT-5 with reasoning effort "minimal"
for m in gpt41 gpt5min; do
  swap $m --only batch2_comm_full,batch3_mgr_elected,batch8_mgr_salary,batch8_mgr_costly --trials 3 --workers 6
done
