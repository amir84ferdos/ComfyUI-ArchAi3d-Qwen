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

# Auto-download model if missing
if [ ! -f "$MODEL_PATH" ] || [ ! -f "$MMPROJ_PATH" ]; then
    echo "Model files not found. Downloading automatically..."
    echo ""

    # Determine repo and files based on model size
    if [ "$1" = "2B" ]; then
        HF_REPO="Qwen/Qwen3-VL-2B-Instruct-GGUF"
        MODEL_FILE="Qwen3VL-2B-Instruct-Q4_K_M.gguf"
        MMPROJ_FILE="mmproj-Qwen3VL-2B-Instruct-F16.gguf"
        LOCAL_DIR="$MODEL_DIR/qwen3-vl-2b"
    elif [ "$1" = "8B" ]; then
        HF_REPO="Qwen/Qwen3-VL-8B-Instruct-GGUF"
        MODEL_FILE="Qwen3VL-8B-Instruct-Q4_K_M.gguf"
        MMPROJ_FILE="mmproj-Qwen3VL-8B-Instruct-F16.gguf"
        LOCAL_DIR="$MODEL_DIR/qwen3-vl-8b"
    else
        HF_REPO="Qwen/Qwen3-VL-4B-Instruct-GGUF"
        MODEL_FILE="Qwen3VL-4B-Instruct-Q4_K_M.gguf"
        MMPROJ_FILE="mmproj-Qwen3VL-4B-Instruct-F16.gguf"
        LOCAL_DIR="$MODEL_DIR/qwen3-vl-4b"
    fi

    mkdir -p "$LOCAL_DIR"

    echo "Downloading from: $HF_REPO"
    echo "To: $LOCAL_DIR"
    echo ""

    # Download using huggingface-cli or Python
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download "$HF_REPO" "$MODEL_FILE" "$MMPROJ_FILE" --local-dir "$LOCAL_DIR"
    else
        # Fallback to Python
        python3 -c "
from huggingface_hub import hf_hub_download
import os
local_dir = '$LOCAL_DIR'
os.makedirs(local_dir, exist_ok=True)
print('Downloading model...')
hf_hub_download('$HF_REPO', '$MODEL_FILE', local_dir=local_dir)
print('Downloading mmproj...')
hf_hub_download('$HF_REPO', '$MMPROJ_FILE', local_dir=local_dir)
print('Download complete!')
"
    fi

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to download model files."
        echo "Please check your internet connection and try again."
        exit 1
    fi

    echo ""
    echo "Download complete!"
    echo ""
fi

# Verify files exist after download
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH after download attempt."
    exit 1
fi

if [ ! -f "$MMPROJ_PATH" ]; then
    echo "ERROR: mmproj not found at $MMPROJ_PATH after download attempt."
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
