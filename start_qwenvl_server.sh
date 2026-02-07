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

# ============================================================================
# SPEED OPTIMIZATIONS (llama.cpp 2026 improvements)
# ============================================================================
# Flash attention - faster GPU attention computation
# Options: auto (let llama.cpp decide), on (force enable), off (force disable)
FLASH_ATTN="${FLASH_ATTN:-auto}"

# KV cache type - affects quality and VRAM usage
# Options: f16 (best quality), q8_0 (good balance), q4_0 (fastest, less quality)
CACHE_TYPE_K="${CACHE_TYPE_K:-f16}"
CACHE_TYPE_V="${CACHE_TYPE_V:-f16}"

# Parallel sequences - number of concurrent requests (1-8)
PARALLEL="${PARALLEL:-2}"

# Continuous batching - batch multiple requests for throughput
CONT_BATCHING="${CONT_BATCHING:-1}"  # 1=enabled, 0=disabled

# Tensor split for multi-GPU (comma-separated ratios, e.g., "0.5,0.5" for 2 GPUs)
TENSOR_SPLIT="${TENSOR_SPLIT:-}"

# Model quantization selection (Q4_K_M or Q8_0)
MODEL_QUANT="${MODEL_QUANT:-Q4_K_M}"

# CPU threads for non-GPU work
THREADS="${THREADS:-4}"

# Get script directory for relative paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ============================================================================
# SMART CUDA LIBS: Use persistent, or install+copy if missing
# ============================================================================
setup_cuda_libs() {
    LIB_DIR="$SCRIPT_DIR/lib"

    # Fast path: use persistent libs if they exist
    if ls "$LIB_DIR"/libcublas*.so* 1>/dev/null 2>&1; then
        export LD_LIBRARY_PATH="$LIB_DIR:${LD_LIBRARY_PATH:-}"
        return 0
    fi

    # First run: install and copy to persistent location
    echo "First run: Installing CUDA libraries to persistent location..."

    # Install cuBLAS (try CUDA 12.8 for Blackwell, fallback to others)
    apt-get update -qq 2>/dev/null
    apt-get install -y -qq libcublas-12-8 libcublas-dev-12-8 2>/dev/null || \
    apt-get install -y -qq libcublas-12-4 libcublas-dev-12-4 2>/dev/null || \
    apt-get install -y -qq libcublas-dev 2>/dev/null || true

    # Copy to persistent location
    mkdir -p "$LIB_DIR"
    for src_dir in /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu; do
        [ -d "$src_dir" ] || continue
        for lib in libcublas libcublasLt libcudart; do
            find "$src_dir" -maxdepth 1 -name "${lib}.so*" -type f \
                -exec cp -n {} "$LIB_DIR/" \; 2>/dev/null
        done
    done

    # Create symlinks (e.g., libcublas.so.12 -> libcublas.so.12.8.4.1)
    cd "$LIB_DIR"
    for lib in libcublas libcublasLt libcudart; do
        # Find the full versioned file and create .so.12 symlink
        full_lib=$(ls ${lib}.so.* 2>/dev/null | grep -v "\.so\.12$" | head -1)
        if [ -n "$full_lib" ] && [ ! -e "${lib}.so.12" ]; then
            ln -sf "$full_lib" "${lib}.so.12"
        fi
    done
    cd - > /dev/null

    export LD_LIBRARY_PATH="$LIB_DIR:${LD_LIBRARY_PATH:-}"
    echo "Done. Future starts will be instant."
}

setup_cuda_libs

# Local SSD cache for RunPod (models copied here load much faster)
LOCAL_CACHE="/tmp/comfyui-local-models"

# Multiple search paths for models (in priority order)
# 0. Local SSD cache (RunPod optimization - fastest)
# 1. Environment variable MODEL_DIR
# 2. ComfyUI models/LLM/GGUF folder (preferred - organized with other LLM models)
# 3. ComfyUI models/llama-models folder (legacy)
# 4. Default cache location (fallback - not persistent on RunPod!)
SEARCH_PATHS=(
    "$LOCAL_CACHE"
    "${MODEL_DIR:-}"
    "$SCRIPT_DIR/../../models/LLM/GGUF"
    "$SCRIPT_DIR/../../models/LLM"
    "$SCRIPT_DIR/../../models/llama-models"
    "/workspace/runpod-slim/ComfyUI/models/LLM/GGUF"
    "/workspace/runpod-slim/ComfyUI/models/LLM"
    "/workspace/runpod-slim/ComfyUI/models/llama-models"
    "$HOME/.cache/llama-models"
)

# Function to find model file in search paths
find_model() {
    local filename="$1"
    local subdir="$2"  # Optional subdirectory (e.g., qwen3-vl-8b)

    for base_path in "${SEARCH_PATHS[@]}"; do
        [ -z "$base_path" ] && continue

        # Try with subdirectory first
        if [ -n "$subdir" ] && [ -f "$base_path/$subdir/$filename" ]; then
            echo "$base_path/$subdir/$filename"
            return 0
        fi

        # Try flat structure (all models in same folder)
        if [ -f "$base_path/$filename" ]; then
            echo "$base_path/$filename"
            return 0
        fi
    done

    return 1
}

# Model selection with automatic port assignment
if [ "$1" = "2B" ]; then
    MODEL_FILE="Qwen3VL-2B-Instruct-${MODEL_QUANT}.gguf"
    MMPROJ_FILE="mmproj-Qwen3VL-2B-Instruct-F16.gguf"
    SUBDIR="qwen3-vl-2b"
    MODEL_NAME="Qwen3-VL-2B-Instruct (${MODEL_QUANT})"
    DEFAULT_PORT=8032
elif [ "$1" = "8B" ]; then
    MODEL_FILE="Qwen3VL-8B-Instruct-${MODEL_QUANT}.gguf"
    MMPROJ_FILE="mmproj-Qwen3VL-8B-Instruct-F16.gguf"
    SUBDIR="qwen3-vl-8b"
    MODEL_NAME="Qwen3-VL-8B-Instruct (${MODEL_QUANT})"
    DEFAULT_PORT=8034
else
    # Default to 4B
    MODEL_FILE="Qwen3VL-4B-Instruct-${MODEL_QUANT}.gguf"
    MMPROJ_FILE="mmproj-Qwen3VL-4B-Instruct-F16.gguf"
    SUBDIR="qwen3-vl-4b"
    MODEL_NAME="Qwen3-VL-4B-Instruct (${MODEL_QUANT})"
    DEFAULT_PORT=8033
fi

# Copy file from /workspace/ to local SSD cache for faster loading
copy_to_local() {
    local src="$1"
    [ -z "$src" ] && return
    # Only copy files from network /workspace/ mount
    case "$src" in /workspace/*) ;; *) echo "$src"; return ;; esac

    local rel="${src#/workspace/}"
    local dst="$LOCAL_CACHE/$rel"

    # Already cached? (check file size matches)
    if [ -f "$dst" ]; then
        local src_size dst_size
        src_size=$(stat -c%s "$src" 2>/dev/null)
        dst_size=$(stat -c%s "$dst" 2>/dev/null)
        if [ "$src_size" = "$dst_size" ]; then
            echo "[LocalCache] Using cached: $(basename "$dst")" >&2
            echo "$dst"
            return
        fi
    fi

    # Check disk space (keep 20GB headroom)
    local free_kb file_kb
    free_kb=$(df --output=avail /tmp 2>/dev/null | tail -1)
    file_kb=$(du -k "$src" 2>/dev/null | cut -f1)
    local headroom_kb=$((20 * 1024 * 1024))  # 20GB in KB
    if [ -n "$free_kb" ] && [ -n "$file_kb" ] && [ $((free_kb - file_kb)) -lt "$headroom_kb" ]; then
        echo "[LocalCache] Not enough space, loading from network" >&2
        echo "$src"
        return
    fi

    # Copy to local SSD
    mkdir -p "$(dirname "$dst")"
    local file_mb=$((file_kb / 1024))
    echo "[LocalCache] Copying $(basename "$src") (${file_mb}MB) to local SSD..." >&2
    if cp "$src" "$dst" 2>/dev/null; then
        echo "[LocalCache] Done." >&2
        echo "$dst"
    else
        echo "[LocalCache] Copy failed, loading from network" >&2
        rm -f "$dst" 2>/dev/null
        echo "$src"
    fi
}

# Find model files in search paths
MODEL_PATH=$(find_model "$MODEL_FILE" "$SUBDIR")
MMPROJ_PATH=$(find_model "$MMPROJ_FILE" "$SUBDIR")

# Copy to local SSD if on RunPod network mount (much faster loading)
if [ -n "$MODEL_PATH" ]; then
    MODEL_PATH=$(copy_to_local "$MODEL_PATH")
fi
if [ -n "$MMPROJ_PATH" ]; then
    MMPROJ_PATH=$(copy_to_local "$MMPROJ_PATH")
fi

# Allow port override via environment
PORT="${PORT:-$DEFAULT_PORT}"

# Allow model path override via environment (explicit override takes priority)
MODEL_PATH="${MODEL:-$MODEL_PATH}"
MMPROJ_PATH="${MMPROJ:-$MMPROJ_PATH}"

# ============================================================================
# PORT CONFLICT DETECTION
# ============================================================================
# Check if server is responding at all on this port (any HTTP response = server exists)
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 2 "http://localhost:$PORT/health" 2>/dev/null)
# Default to 000 if curl failed completely
HTTP_CODE="${HTTP_CODE:-000}"

if [ "$HTTP_CODE" = "200" ]; then
    # Server is running and healthy
    echo "=============================================="
    echo "✅ Server already running on port $PORT"
    echo "=============================================="
    echo ""
    echo "The llama-server is active and healthy."
    echo "You can use it directly in ComfyUI."
    echo ""
    echo "To restart with different settings, first stop it:"
    echo "  killall llama-server"
    echo ""
    exit 0
elif [ "$HTTP_CODE" = "503" ]; then
    # Server is running but model is still loading
    echo "=============================================="
    echo "⏳ Server is starting on port $PORT (model loading)"
    echo "=============================================="
    echo ""
    echo "The llama-server is active but still loading the model."
    echo "Please wait for it to finish loading."
    echo ""
    echo "To restart with different settings, first stop it:"
    echo "  killall llama-server"
    echo ""
    exit 0
# Any other 2xx or 5xx code means server is there
elif [[ "$HTTP_CODE" =~ ^[2345][0-9][0-9]$ ]]; then
    # Some other HTTP response - server exists
    echo "=============================================="
    echo "⚠️  Server responding with HTTP $HTTP_CODE on port $PORT"
    echo "=============================================="
    echo ""
    echo "A server is already running. Stop it first:"
    echo "  killall llama-server"
    echo ""
    exit 0
fi

# HTTP_CODE is 000 or connection failed - no server responding
# Check if port is occupied (zombie/crashed)
if (lsof -i :$PORT > /dev/null 2>&1) || (netstat -tlnp 2>/dev/null | grep -q ":$PORT ") || (fuser $PORT/tcp > /dev/null 2>&1); then
    echo "=============================================="
    echo "⚠️  Port $PORT is blocked by a stale process"
    echo "=============================================="
    echo ""
    echo "Killing stale process..."

    # Try multiple methods to kill the process
    fuser -k $PORT/tcp 2>/dev/null || \
    pkill -f "llama-server.*--port.*$PORT" 2>/dev/null || \
    pkill -f "llama-server" 2>/dev/null || \
    true

    sleep 2
    echo "Port cleared, starting fresh server..."
    echo ""
fi

echo "=============================================="
echo "QwenVL GGUF Server for ComfyUI"
echo "=============================================="
echo "Model:      $MODEL_NAME"
echo "Quant:      $MODEL_QUANT"
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
# Priority: 1) This node's bin folder (persistent on RunPod)
#           2) System PATH
#           3) Legacy locations
LLAMA_SERVER=""
if [ -f "$SCRIPT_DIR/bin/llama-server" ]; then
    # New persistent location (installed by LlamaCpp Installer node)
    LLAMA_SERVER="$SCRIPT_DIR/bin/llama-server"
elif command -v llama-server &> /dev/null; then
    LLAMA_SERVER="llama-server"
elif [ -f "$HOME/.local/bin/llama-server" ]; then
    LLAMA_SERVER="$HOME/.local/bin/llama-server"
elif [ -f "$HOME/llama.cpp/build/bin/llama-server" ]; then
    LLAMA_SERVER="$HOME/llama.cpp/build/bin/llama-server"
elif [ -f "$SCRIPT_DIR/llama_cpp/build/bin/llama-server" ]; then
    # Source build location
    LLAMA_SERVER="$SCRIPT_DIR/llama_cpp/build/bin/llama-server"
else
    echo "ERROR: llama-server not found!"
    echo ""
    echo "Installation options:"
    echo ""
    echo "1. Use the LlamaCpp Installer node in ComfyUI (recommended)"
    echo "   Add 'ArchAi3d > Setup > LlamaCpp Installer' node and run 'full_install'"
    echo ""
    echo "2. Build from source:"
    echo "   git clone https://github.com/ggml-org/llama.cpp"
    echo "   cd llama.cpp"
    echo "   cmake -B build -DGGML_CUDA=ON"
    echo "   cmake --build build --config Release -j"
    echo "   mkdir -p $SCRIPT_DIR/bin"
    echo "   cp build/bin/llama-server $SCRIPT_DIR/bin/"
    echo ""
    exit 1
fi

# Auto-download model if missing
if [ -z "$MODEL_PATH" ] || [ -z "$MMPROJ_PATH" ]; then
    echo "Model files not found in any search path. Downloading automatically..."
    echo ""
    echo "Searched in:"
    for path in "${SEARCH_PATHS[@]}"; do
        [ -n "$path" ] && echo "  - $path"
    done
    echo ""

    # Determine HuggingFace repo based on model size
    if [ "$1" = "2B" ]; then
        HF_REPO="Qwen/Qwen3-VL-2B-Instruct-GGUF"
    elif [ "$1" = "8B" ]; then
        HF_REPO="Qwen/Qwen3-VL-8B-Instruct-GGUF"
    else
        HF_REPO="Qwen/Qwen3-VL-4B-Instruct-GGUF"
    fi

    # Choose download directory (prefer ComfyUI models/LLM/GGUF folder - persistent on RunPod)
    if [ -d "$SCRIPT_DIR/../../models" ]; then
        LOCAL_DIR="$SCRIPT_DIR/../../models/LLM/GGUF"
    elif [ -d "/workspace/runpod-slim/ComfyUI/models" ]; then
        LOCAL_DIR="/workspace/runpod-slim/ComfyUI/models/LLM/GGUF"
    else
        LOCAL_DIR="$HOME/.cache/llama-models"
        echo "⚠️ Warning: Downloading to $LOCAL_DIR (NOT persistent on RunPod!)"
    fi

    mkdir -p "$LOCAL_DIR"

    echo "Downloading from: $HF_REPO"
    echo "To: $LOCAL_DIR"
    echo ""

    # Download using huggingface-cli or Python
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download "$HF_REPO" "$MODEL_FILE" "$MMPROJ_FILE" --local-dir "$LOCAL_DIR"
    elif command -v hf &> /dev/null; then
        hf download "$HF_REPO" "$MODEL_FILE" "$MMPROJ_FILE" --local-dir "$LOCAL_DIR"
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

    # Re-find models after download
    MODEL_PATH=$(find_model "$MODEL_FILE" "$SUBDIR")
    MMPROJ_PATH=$(find_model "$MMPROJ_FILE" "$SUBDIR")
fi

# Verify files exist after download
if [ -z "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model file '$MODEL_FILE' not found in any search path."
    echo ""
    echo "Manual download:"
    echo "  huggingface-cli download Qwen/Qwen3-VL-${1:-4B}-Instruct-GGUF $MODEL_FILE --local-dir /path/to/models"
    exit 1
fi

if [ -z "$MMPROJ_PATH" ] || [ ! -f "$MMPROJ_PATH" ]; then
    echo "ERROR: mmproj file '$MMPROJ_FILE' not found in any search path."
    echo ""
    echo "Manual download:"
    echo "  huggingface-cli download Qwen/Qwen3-VL-${1:-4B}-Instruct-GGUF $MMPROJ_FILE --local-dir /path/to/models"
    exit 1
fi

echo "Using llama-server: $LLAMA_SERVER"
echo ""

# Build server command with optimizations
SERVER_ARGS=(
    --jinja
    -c "$CTX"
    --port "$PORT"
    -m "$MODEL_PATH"
    --mmproj "$MMPROJ_PATH"
    -ngl "$GPU_LAYERS"
    --host 0.0.0.0
    -t "$THREADS"
)

# Flash attention (significant speedup on modern GPUs)
if [ "$FLASH_ATTN" = "auto" ]; then
    # Don't pass -fa flag; let llama.cpp use its default (auto)
    echo "✓ Flash attention: AUTO (llama.cpp default)"
elif [ "$FLASH_ATTN" = "on" ] || [ "$FLASH_ATTN" = "1" ]; then
    SERVER_ARGS+=(-fa on)
    echo "✓ Flash attention: ENABLED"
elif [ "$FLASH_ATTN" = "off" ] || [ "$FLASH_ATTN" = "0" ]; then
    SERVER_ARGS+=(-fa off)
    echo "✓ Flash attention: DISABLED"
fi

# KV cache type (best quality = f16, balanced = q8_0, fastest = q4_0)
if [ -n "$CACHE_TYPE_K" ]; then
    SERVER_ARGS+=(--cache-type-k "$CACHE_TYPE_K" --cache-type-v "$CACHE_TYPE_V")
    echo "✓ KV cache: K=$CACHE_TYPE_K, V=$CACHE_TYPE_V"
fi

# Parallel sequences for concurrent requests
if [ "$PARALLEL" -gt 1 ]; then
    SERVER_ARGS+=(-np "$PARALLEL")
    echo "✓ Parallel sequences: $PARALLEL"
fi

# Continuous batching
if [ "$CONT_BATCHING" = "1" ]; then
    SERVER_ARGS+=(-cb)
    echo "✓ Continuous batching: ENABLED"
fi

# Multi-GPU tensor split
if [ -n "$TENSOR_SPLIT" ]; then
    SERVER_ARGS+=(--tensor-split "$TENSOR_SPLIT")
    echo "✓ Tensor split: $TENSOR_SPLIT"
fi

echo ""

# Start the server with all optimizations
exec "$LLAMA_SERVER" "${SERVER_ARGS[@]}"
