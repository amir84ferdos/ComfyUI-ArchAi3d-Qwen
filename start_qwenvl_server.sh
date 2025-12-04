#!/bin/bash
# ============================================================================
# QwenVL GGUF Server for ComfyUI
# ============================================================================
#
# This script starts llama-server with Qwen3-VL GGUF model for fast VLM inference.
#
# Usage:
#   ./start_qwenvl_server.sh                    # Start with defaults (4B on port 8033)
#   ./start_qwenvl_server.sh 2B                 # Use 2B model (port 8032)
#   ./start_qwenvl_server.sh 4B                 # Use 4B model (port 8033)
#   ./start_qwenvl_server.sh 8B                 # Use 8B model (port 8034)
#
# In ComfyUI node, use server_url:
#   - 2B: http://localhost:8032
#   - 4B: http://localhost:8033
#   - 8B: http://localhost:8034
#
# Requirements:
#   - llama.cpp with llama-server built
#   - CUDA support for GPU acceleration
#
# Author: Amir Ferdos (ArchAi3d)
# ============================================================================

# Configuration
CTX="${CTX:-8192}"
GPU_LAYERS="${GPU_LAYERS:-99}"  # 99 for all layers on GPU
MODEL_DIR="$HOME/.cache/llama-models"

# Model selection with automatic port assignment
if [ "$1" = "2B" ]; then
    MODEL_PATH="$MODEL_DIR/qwen3-vl-2b/Qwen3VL-2B-Instruct-Q4_K_M.gguf"
    MMPROJ_PATH="$MODEL_DIR/qwen3-vl-2b/mmproj-Qwen3VL-2B-Instruct-F16.gguf"
    MODEL_NAME="Qwen3-VL-2B-Instruct"
    DEFAULT_PORT=8032
elif [ "$1" = "8B" ]; then
    MODEL_PATH="$MODEL_DIR/qwen3-vl-8b/Qwen3VL-8B-Instruct-Q4_K_M.gguf"
    MMPROJ_PATH="$MODEL_DIR/qwen3-vl-8b/mmproj-Qwen3VL-8B-Instruct-F16.gguf"
    MODEL_NAME="Qwen3-VL-8B-Instruct"
    DEFAULT_PORT=8034
else
    # Default to 4B
    MODEL_PATH="$MODEL_DIR/qwen3-vl-4b/Qwen3VL-4B-Instruct-Q4_K_M.gguf"
    MMPROJ_PATH="$MODEL_DIR/qwen3-vl-4b/mmproj-Qwen3VL-4B-Instruct-F16.gguf"
    MODEL_NAME="Qwen3-VL-4B-Instruct"
    DEFAULT_PORT=8033
fi

# Allow port override via environment
PORT="${PORT:-$DEFAULT_PORT}"

# Allow model path override via environment
MODEL_PATH="${MODEL:-$MODEL_PATH}"
MMPROJ_PATH="${MMPROJ:-$MMPROJ_PATH}"

echo "=============================================="
echo "QwenVL GGUF Server for ComfyUI"
echo "=============================================="
echo "Model:      $MODEL_NAME"
echo "Model Path: $MODEL_PATH"
echo "mmproj:     $MMPROJ_PATH"
echo "Port:       $PORT"
echo "Context:    $CTX tokens"
echo "GPU Layers: $GPU_LAYERS"
echo "=============================================="
echo ""
echo "In ComfyUI, set server_url to:"
echo "  http://localhost:$PORT"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=============================================="
echo ""

# Check if llama-server is available
LLAMA_SERVER=""
if command -v llama-server &> /dev/null; then
    LLAMA_SERVER="llama-server"
elif [ -f "$HOME/.local/bin/llama-server" ]; then
    LLAMA_SERVER="$HOME/.local/bin/llama-server"
elif [ -f "$HOME/llama.cpp/build/bin/llama-server" ]; then
    LLAMA_SERVER="$HOME/llama.cpp/build/bin/llama-server"
else
    echo "ERROR: llama-server not found!"
    echo ""
    echo "Installation options:"
    echo ""
    echo "1. Build from source (recommended for best CUDA performance):"
    echo "   git clone https://github.com/ggml-org/llama.cpp"
    echo "   cd llama.cpp"
    echo "   cmake -B build -DGGML_CUDA=ON"
    echo "   cmake --build build --config Release -j"
    echo "   cp build/bin/llama-server ~/.local/bin/"
    echo ""
    exit 1
fi

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo ""
    echo "Download the model using Python:"
    echo ""
    echo "from huggingface_hub import hf_hub_download"
    echo "hf_hub_download('Qwen/Qwen3-VL-4B-Instruct-GGUF', 'Qwen3VL-4B-Instruct-Q4_K_M.gguf', local_dir='~/.cache/llama-models/qwen3-vl-4b')"
    echo "hf_hub_download('Qwen/Qwen3-VL-4B-Instruct-GGUF', 'mmproj-Qwen3VL-4B-Instruct-F16.gguf', local_dir='~/.cache/llama-models/qwen3-vl-4b')"
    echo ""
    exit 1
fi

if [ ! -f "$MMPROJ_PATH" ]; then
    echo "ERROR: mmproj not found at $MMPROJ_PATH"
    echo "Vision support requires the mmproj file."
    exit 1
fi

echo "Using llama-server: $LLAMA_SERVER"
echo ""

# Start the server with local model path
"$LLAMA_SERVER" \
    --jinja \
    -c "$CTX" \
    --port "$PORT" \
    -m "$MODEL_PATH" \
    --mmproj "$MMPROJ_PATH" \
    -ngl "$GPU_LAYERS" \
    --host 0.0.0.0
