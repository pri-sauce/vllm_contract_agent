#!/bin/bash
# ============================================================
#  start_vllm.sh — Launch vLLM server for contract agent
#  RTX 6000 Pro Blackwell 95GB — Qwen2.5 7B fp8
# ============================================================

MODEL="Qwen/Qwen2.5-7B-Instruct"

# fp8 quantization: cuts VRAM from ~14GB to ~7GB, doubles throughput
# On 95GB you have room for the full fp16 too, but fp8 is faster
QUANTIZATION="fp8"

# How much GPU memory to reserve for vLLM's KV cache
# 0.85 = 85% of 95GB = ~80GB for model + KV cache
GPU_UTIL="0.85"

# Max sequence length — 8192 covers all contract clauses comfortably
MAX_LEN="8192"

# Max concurrent sequences in flight (continuous batching)
# Higher = better GPU utilisation. 512 is solid for your card.
MAX_SEQS="512"

# Tensor parallelism — 1 GPU, so 1
TP="1"

echo "Starting vLLM server..."
echo "Model:         $MODEL"
echo "Quantization:  $QUANTIZATION"
echo "GPU Util:      $GPU_UTIL"
echo "Max seqs:      $MAX_SEQS"
echo ""
echo "Server will be ready at: http://localhost:8000"
echo "Test with: curl http://localhost:8000/v1/models"
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --dtype auto \
    --quantization "$QUANTIZATION" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAX_LEN" \
    --max-num-seqs "$MAX_SEQS" \
    --tensor-parallel-size "$TP" \
    --port 8000 \
    --host 0.0.0.0 \
    --enable-prefix-caching \
    --disable-log-requests
