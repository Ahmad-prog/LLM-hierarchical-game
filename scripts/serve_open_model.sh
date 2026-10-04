#!/bin/bash
# Serve one open-weight model with vLLM and run all of its games in the paper (187 per model; 225 for gpt-oss, 249 with the confirmatory games for Qwen3.8).
# The tag qwen3.8-27b-nvfp4 runs the quantization check (32 games).
#
#   bash scripts/serve_open_model.sh <tag> [gpu] [port]
#   tags: gpt-oss-120b | nemotron-3-super-120b | ling-3.0-flash | qwen3.8-27b | gemma-4-31b | qwen3.8-27b-nvfp4
#
# The paper used vLLM 0.30 on one 96 GB GPU with up to 48 games in parallel. The extra vLLM options
# below are the ones each model needed on that setup. The runner is resumable: finished games are skipped.
set -e
TAG=$1; GPU=${2:-0}; PORT=${3:-8000}
case "$TAG" in
  gpt-oss-120b)          HF=openai/gpt-oss-120b;                            EXTRA="--attention-backend TRITON_ATTN" ;;
  nemotron-3-super-120b) HF=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4; EXTRA="--max-num-seqs 128 --attention-backend TRITON_ATTN" ;;
  ling-3.0-flash)        HF=inclusionAI/Ling-3.0-flash-fp4;                 EXTRA="" ;;
  qwen3.8-27b)           HF=Qwen/Qwen3.8-27B;                               EXTRA="--max-num-seqs 128" ;;
  gemma-4-31b)           HF=google/gemma-4-31B-it;                          EXTRA="" ;;
  qwen3.8-27b-nvfp4)     HF=nvidia/Qwen3.8-27B-NVFP4;                       EXTRA="--max-num-seqs 128" ;;
  *) echo "unknown tag: $TAG"; exit 1 ;;
esac
cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES=$GPU vllm serve $HF --served-model-name $HF --host 127.0.0.1 --port $PORT \
  --max-model-len 32768 --gpu-memory-utilization 0.90 --trust-remote-code $EXTRA > vllm_$TAG.log 2>&1 &
VPID=$!
trap 'kill $VPID 2>/dev/null' EXIT
echo "waiting for vLLM ($HF) on port $PORT ..."
until curl -sf http://127.0.0.1:$PORT/v1/models > /dev/null; do
  kill -0 $VPID 2>/dev/null || { echo "vLLM exited, see vllm_$TAG.log"; exit 1; }
  sleep 20
done

export HG_LOCAL_BASE_URL=http://127.0.0.1:$PORT/v1 HG_LOCAL_MODEL=$HF
J="python hg_jobs.py --track oss --tag $TAG --out results"
if [ "$TAG" = "qwen3.8-27b-nvfp4" ]; then
  $J --only batch1_baseline,batch2_comm_full,batch3_mgr_elected,batch8_mgr_salary --trials 8 --workers 32
  exit 0
fi
$J --only batch1,batch2,batch3,batch5,batch8,batch9 --trials 3 --workers 48       # every setup, 3 games
$J --only batch2_comm_full,batch3_mgr,batch8_mgr --trials 8 --workers 48       # main contrasts, 8 games
$J --only batch13_mgr_badincumbent,batch13_ballot_random,batch14_mgr_goodincumbent --trials 10 --workers 16
$J --only batch13_info_aggregate,batch13_mgr_nosanction,batch13_mgr_autoreward,batch14_aggregate_chat,batch14_system_reward,batch14_neutral,batch14_nostrategic --trials 5 --workers 32
$J --only batch15_mgr_badincumbent_rb,batch15_mgr_goodincumbent_rb --trials 20 --workers 20
case "$TAG" in gpt-oss-120b|qwen3.8-27b)
  $J --only batch15_mgr_elected_nobudget --trials 8 --workers 8
  $J --only batch16 --trials 10 --workers 15 ;; esac
[ "$TAG" = "qwen3.8-27b" ] && python hg_jobs.py --track oss --tag $TAG --out results_confirm --only batch2_comm_full,batch3_mgr_elected --trials 12 --workers 12
