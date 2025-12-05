# QwenVL GGUF Setup Guide

This guide explains how to install llama.cpp and set up the QwenVL GGUF node for fast VLM inference in ComfyUI.

## Quick Start (Automated)

Run the installation script:

```bash
cd /path/to/ComfyUI/custom_nodes/comfyui-archai3d-qwen
chmod +x install_llama_cpp.sh
./install_llama_cpp.sh
```

The script will:
1. Install build dependencies
2. Build llama.cpp with CUDA support
3. Download QwenVL GGUF models (2B and 4B)
4. Install llama-server to ~/.local/bin/

---

## Manual Installation

### Step 1: System Requirements

| Component | Requirement |
|-----------|-------------|
| OS | Ubuntu 20.04+ / Debian 11+ |
| GPU | NVIDIA with CUDA support |
| VRAM | 4GB minimum (2B model), 8GB+ recommended (4B/8B) |
| Disk | ~10GB for models |

### Step 2: Install NVIDIA Drivers

```bash
# Check if NVIDIA driver is installed
nvidia-smi

# If not installed:
sudo apt update
sudo apt install nvidia-driver-535  # or 570+ for RTX 5090
sudo reboot
```

### RTX 5090 / Blackwell Note

If you have an **RTX 5090, 5080, or 5070** (Blackwell architecture):

| Requirement | Version |
|-------------|---------|
| Driver | 570+ |
| CUDA | 12.8+ |

```bash
# For Blackwell GPUs, install driver 570+
sudo apt install nvidia-driver-570
```

### Step 3: Install CUDA Toolkit

```bash
# Option A: Install from Ubuntu repos
sudo apt install nvidia-cuda-toolkit

# Option B: Install from NVIDIA (recommended for latest)
# Visit: https://developer.nvidia.com/cuda-downloads
# Select: Linux > x86_64 > Ubuntu > 22.04 > deb (network)
```

Verify installation:
```bash
nvcc --version
# Should show CUDA version
```

### Step 4: Install Build Tools

```bash
sudo apt install -y build-essential cmake git curl wget
```

### Step 5: Build llama.cpp

```bash
# Clone repository
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cd ~/llama.cpp

# Build with CUDA support
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)

# For RTX 5090 (Blackwell) - explicitly set architecture
# cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_ARCHITECTURES="120"

# Install to local bin
mkdir -p ~/.local/bin
cp build/bin/llama-server ~/.local/bin/
chmod +x ~/.local/bin/llama-server

# Add to PATH (add to ~/.bashrc for permanent)
export PATH="$HOME/.local/bin:$PATH"
```

### Step 6: Download Models

```bash
# Create model directory
mkdir -p ~/.cache/llama-models/qwen3-vl-4b

# Download using Python
pip install huggingface_hub

python3 << 'EOF'
from huggingface_hub import hf_hub_download

# 4B Model (recommended)
hf_hub_download(
    "Qwen/Qwen3-VL-4B-Instruct-GGUF",
    "Qwen3VL-4B-Instruct-Q4_K_M.gguf",
    local_dir="~/.cache/llama-models/qwen3-vl-4b"
)
hf_hub_download(
    "Qwen/Qwen3-VL-4B-Instruct-GGUF",
    "mmproj-Qwen3VL-4B-Instruct-F16.gguf",
    local_dir="~/.cache/llama-models/qwen3-vl-4b"
)
EOF
```

---

## Model Sizes and VRAM Requirements

| Model | Quantization | VRAM Required | Speed | Quality |
|-------|--------------|---------------|-------|---------|
| 2B | Q4_K_M | ~4 GB | Fastest | Good |
| 4B | Q4_K_M | ~7 GB | Fast | Better |
| 8B | Q4_K_M | ~12 GB | Medium | Best |

**Recommendation by GPU:**

| GPU VRAM | Recommended Model |
|----------|-------------------|
| 6 GB | 2B |
| 8 GB | 4B |
| 12 GB+ | 4B or 8B |
| 24 GB (RTX 4090) | 8B (full speed) |
| 32 GB (RTX 5090) | 8B (max performance) |

---

## Starting the Server

### Basic Usage

```bash
cd /path/to/comfyui-archai3d-qwen
./start_qwenvl_server.sh 4B
```

### With Custom Settings

```bash
# Lower GPU layers to save VRAM (for running alongside ComfyUI)
GPU_LAYERS=30 ./start_qwenvl_server.sh 4B

# Smaller context for lower VRAM
CTX=4096 ./start_qwenvl_server.sh 4B

# Combined
CTX=4096 GPU_LAYERS=30 ./start_qwenvl_server.sh 4B
```

### Server Ports

| Model | Port |
|-------|------|
| 2B | 8032 |
| 4B | 8033 |
| 8B | 8034 |

---

## Using in ComfyUI

1. Start the server (terminal): `./start_qwenvl_server.sh 4B`
2. Open ComfyUI
3. Add node: **ArchAi3d > Qwen > VLM > 🚀 QwenVL GGUF (Fast)**
4. Connect an image and run!

### Node Settings

**Simple Mode (Presets):**
- `quality_preset`: Choose from Fast/Balanced/Interior Design/Creative
- `creativity`: 0.0 (focused) to 1.0 (creative)

**For Interior Design:**
- `preset_prompt`: 🏠 Interior Design Prompt
- `quality_preset`: 🏠 Interior Design (Best)
- `creativity`: 0.3 (accurate descriptions)

---

## Troubleshooting

### Error: "Cannot connect to server"

1. Check if server is running:
   ```bash
   curl http://localhost:8033/health
   ```

2. Start the server:
   ```bash
   ./start_qwenvl_server.sh 4B
   ```

### Error: "Out of VRAM"

Reduce GPU layers to use CPU for some layers:
```bash
GPU_LAYERS=30 ./start_qwenvl_server.sh 4B
```

Or use the smaller 2B model:
```bash
./start_qwenvl_server.sh 2B
```

### Error: "Model not found"

Download the models:
```bash
python3 << 'EOF'
from huggingface_hub import hf_hub_download
import os

local_dir = os.path.expanduser("~/.cache/llama-models/qwen3-vl-4b")
os.makedirs(local_dir, exist_ok=True)

hf_hub_download("Qwen/Qwen3-VL-4B-Instruct-GGUF", "Qwen3VL-4B-Instruct-Q4_K_M.gguf", local_dir=local_dir)
hf_hub_download("Qwen/Qwen3-VL-4B-Instruct-GGUF", "mmproj-Qwen3VL-4B-Instruct-F16.gguf", local_dir=local_dir)
EOF
```

### Error: "CUDA not found" during build

Install CUDA toolkit:
```bash
sudo apt install nvidia-cuda-toolkit
```

Or specify CUDA path:
```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

### Server starts but model loading fails

Check VRAM usage:
```bash
nvidia-smi
```

If ComfyUI is using too much VRAM, either:
1. Free ComfyUI VRAM (unload models)
2. Use fewer GPU layers: `GPU_LAYERS=20 ./start_qwenvl_server.sh 4B`

---

## Performance Tips

1. **Keep server running**: Don't stop/start for each inference
2. **Use caching**: Enable `use_cache` in the node
3. **Resize images**: Use `max_image_size=1024` or `1536` instead of `Original`
4. **Match model to GPU**: Don't run 8B on a 6GB GPU

---

## Windows Installation

For Windows, use WSL2 (Windows Subsystem for Linux):

1. Install WSL2:
   ```powershell
   wsl --install -d Ubuntu-22.04
   ```

2. Install NVIDIA drivers for WSL:
   - Download from: https://developer.nvidia.com/cuda/wsl

3. Follow the Linux instructions above inside WSL

Alternatively, use pre-built Windows binaries:
1. Download from: https://github.com/ggerganov/llama.cpp/releases
2. Get the `llama-server-cuda.exe` file
3. Place in a folder and add to PATH

---

## File Locations

| File | Location |
|------|----------|
| llama-server | `~/.local/bin/llama-server` |
| Models | `~/.cache/llama-models/` |
| Server script | `comfyui-archai3d-qwen/start_qwenvl_server.sh` |
| Install script | `comfyui-archai3d-qwen/install_llama_cpp.sh` |

---

## Support

- GitHub: https://github.com/amir84ferdos
- LinkedIn: https://www.linkedin.com/in/archai3d/
- Email: Amir84ferdos@gmail.com
