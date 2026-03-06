#!/bin/bash
# start_vllm.sh - Launch vLLM server for contract agent
# RTX 6000 Pro Blackwell 95GB - Qwen2.5 7B fp8

set -e

# Find correct python binary
if command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON="python"
else
    echo "ERROR: python not found. Install Python 3.10+"
    exit 1
fi

echo "Using: $($PYTHON --version)"

# Check vLLM is installed
if ! $PYTHON -c "import vllm" >/dev/null 2>&1; then
    echo "ERROR: vllm not installed. Run: pip install vllm"
    exit 1
fi

echo "vLLM: $($PYTHON -c 'import vllm; print(vllm.__version__)')"

# Settings
MODEL="Qwen/Qwen2.5-7B-Instruct"
QUANTIZATION="fp8"
GPU_UTIL="0.85"
MAX_LEN="8192"
MAX_SEQS="512"
PORT="8000"

echo ""
echo "Model:    $MODEL"
echo "Quant:    $QUANTIZATION"
echo "GPU util: $GPU_UTIL"
echo "Port:     $PORT"
echo ""
echo "Ready at: http://localhost:$PORT"
echo "Test:     curl http://localhost:$PORT/v1/models"
echo ""

# Launch vLLM
exec $PYTHON -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --dtype auto \
    --quantization "$QUANTIZATION" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAX_LEN" \
    --max-num-seqs "$MAX_SEQS" \
    --tensor-parallel-size 1 \
    --port "$PORT" \
    --host 0.0.0.0 \
    --enable-prefix-caching \
    --disable-log-requests