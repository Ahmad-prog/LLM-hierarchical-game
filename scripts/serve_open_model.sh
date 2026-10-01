#!/bin/bash
# Serve one open-weight model with vLLM and run its 42 games (14 setups x 3 games).
#
#   bash scripts/serve_open_model.sh <tag> [gpu] [port]
#   tags: gpt-oss-120b | nemotron-3-super-120b | ling-3.0-flash | qwen3.8-27b | gemma-4-31b
#
# The paper used vLLM 0.30 on one 96 GB GPU with 48 games in parallel. The extra vLLM options
# below are the ones each model needed on that setup.
set -e
TAG=$1; GPU=${2:-0}; PORT=${3:-8000}
case "$TAG" in
  gpt-oss-120b)          HF=openai/gpt-oss-120b;                            EXTRA="--attention-backend TRITON_ATTN" ;;
  nemotron-3-super-120b) HF=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4; EXTRA="--max-num-seqs 128 --attention-backend TRITON_ATTN" ;;
  ling-3.0-flash)        HF=inclusionAI/Ling-3.0-flash-fp4;                 EXTRA="" ;;
  qwen3.8-27b)           HF=Qwen/Qwen3.8-27B;                               EXTRA="--max-num-seqs 128" ;;
  gemma-4-31b)           HF=google/gemma-4-31B-it;                          EXTRA="" ;;
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

HG_LOCAL_BASE_URL=http://127.0.0.1:$PORT/v1 HG_LOCAL_MODEL=$HF \
  python hg_jobs.py --track oss --tag $TAG --workers 48 --trials 3 --out results
