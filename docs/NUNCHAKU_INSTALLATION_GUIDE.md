# Nunchaku Installation Guide for RunPod

A step-by-step guide for installing Nunchaku (quantized Flux/Qwen models) on RunPod with ComfyUI.

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | RTX 30-series (sm_86) | RTX 40-series (sm_89) |
| **VRAM** | 12GB | 24GB |
| **CUDA** | 12.2+ (Linux) | 12.4+ |
| **Python** | 3.10, 3.11 | 3.11 or 3.12 |
| **PyTorch** | 2.5+ | 2.6+ |

### Supported GPU Architectures
- sm_75 (Turing: RTX 2080)
- sm_80 (Ampere: A100)
- sm_86 (Ampere: RTX 3090, A6000)
- sm_89 (Ada: RTX 4090)
- sm_120 (Blackwell: RTX 5090) - requires PyTorch 2.7+ and CUDA 12.8+

## Quick Check: Your System Info

Run these commands on your RunPod to check compatibility:

```bash
# Check GPU
nvidia-smi --query-gpu=name,memory.total --format=csv

# Check CUDA version
nvcc --version

# Check Python version
python3 --version

# Check PyTorch version (in ComfyUI venv)
/workspace/runpod-slim/ComfyUI/.venv/bin/python -c "import torch; print(torch.__version__)"
```

## Installation Steps

### Step 1: Clone ComfyUI-nunchaku

```bash
cd /workspace/runpod-slim/ComfyUI/custom_nodes
git clone https://github.com/nunchaku-tech/ComfyUI-nunchaku.git
```

### Step 2: Install Requirements

```bash
/workspace/runpod-slim/ComfyUI/.venv/bin/pip install -r /workspace/runpod-slim/ComfyUI/custom_nodes/ComfyUI-nunchaku/requirements.txt
```

This installs:
- diffusers>=0.35
- transformers>=4.54
- sentencepiece
- protobuf
- huggingface_hub>=0.34
- tomli
- peft>=0.17
- accelerate>=1.10
- insightface
- opencv-python
- facexlib
- onnxruntime
- timm

### Step 3: Install Nunchaku Backend Wheel

**Important:** Choose the correct wheel based on your PyTorch version and Python version.

#### Find Your PyTorch Version
```bash
/workspace/runpod-slim/ComfyUI/.venv/bin/python -c "import torch; print(torch.__version__)"
```

#### Install the Matching Wheel

**For PyTorch 2.6 + Python 3.12 (most common on RunPod):**
```bash
/workspace/runpod-slim/ComfyUI/.venv/bin/pip install https://github.com/nunchaku-tech/nunchaku/releases/download/v1.0.1/nunchaku-1.0.1+torch2.6-cp312-cp312-linux_x86_64.whl
```

**For PyTorch 2.6 + Python 3.11:**
```bash
/workspace/runpod-slim/ComfyUI/.venv/bin/pip install https://github.com/nunchaku-tech/nunchaku/releases/download/v1.0.1/nunchaku-1.0.1+torch2.6-cp311-cp311-linux_x86_64.whl
```

**For PyTorch 2.5 + Python 3.12:**
```bash
/workspace/runpod-slim/ComfyUI/.venv/bin/pip install https://github.com/nunchaku-tech/nunchaku/releases/download/v1.0.1/nunchaku-1.0.1+torch2.5-cp312-cp312-linux_x86_64.whl
```

**For PyTorch 2.7 + Python 3.12:**
```bash
/workspace/runpod-slim/ComfyUI/.venv/bin/pip install https://github.com/nunchaku-tech/nunchaku/releases/download/v1.0.1/nunchaku-1.0.1+torch2.7-cp312-cp312-linux_x86_64.whl
```

### Step 4: Update Versions File

```bash
cd /workspace/runpod-slim/ComfyUI/custom_nodes/ComfyUI-nunchaku
/workspace/runpod-slim/ComfyUI/.venv/bin/python scripts/update_versions.py
```

This generates `nunchaku_versions.json` with available model versions.

### Step 5: Verify Installation

```bash
/workspace/runpod-slim/ComfyUI/.venv/bin/python -c "import nunchaku; print('Nunchaku modules:', dir(nunchaku))"
```

Expected output should include:
- `NunchakuFluxTransformer2DModelV2`
- `NunchakuQwenImageTransformer2DModel`
- `NunchakuSanaTransformer2DModel`
- `NunchakuT5EncoderModel`

### Step 6: Restart ComfyUI

```bash
# Kill existing ComfyUI process
pkill -f 'python.*main.py.*8188'

# Wait a moment
sleep 3

# Start ComfyUI
cd /workspace/runpod-slim/ComfyUI
nohup /workspace/runpod-slim/ComfyUI/.venv/bin/python main.py --listen 0.0.0.0 --port 8188 > /workspace/runpod-slim/comfyui.log 2>&1 &

# Check startup
sleep 10
tail -30 /workspace/runpod-slim/comfyui.log
```

## Available Wheel Versions

All wheels are available at: https://github.com/nunchaku-tech/nunchaku/releases

| PyTorch | Python 3.10 | Python 3.11 | Python 3.12 | Python 3.13 |
|---------|-------------|-------------|-------------|-------------|
| 2.5 | ✅ | ✅ | ✅ | - |
| 2.6 | ✅ | ✅ | ✅ | ✅ |
| 2.7 | ✅ | ✅ | ✅ | ✅ |
| 2.8 | ✅ | ✅ | ✅ | ✅ |

## Model Types

### INT4 Models (for RTX 20/30/40 series)
- `svdq-int4_r128-qwen-image-edit-2509.safetensors`
- Best for non-Blackwell GPUs

### FP4 Models (for RTX 50 series / Blackwell)
- `svdq-fp4_r128-qwen-image-edit-2509.safetensors`
- Required for Blackwell architecture

## Troubleshooting

### Error: "No module named 'nunchaku'"
- Make sure you installed the wheel into the correct Python environment
- Use the full path: `/workspace/runpod-slim/ComfyUI/.venv/bin/pip install ...`

### Error: "nunchaku_versions.json not found"
- Run the update script: `python scripts/update_versions.py`

### Error: CUDA version mismatch
- Check your CUDA version: `nvcc --version`
- Linux requires CUDA 12.2+
- Blackwell GPUs require CUDA 12.8+

### Error: "Could not find a version that satisfies the requirement"
- You're trying to install from PyPI instead of the GitHub wheel
- Use the direct wheel URL from GitHub releases

### Out of Memory (OOM)
- Use INT4 quantized models instead of FP8/FP16
- Reduce image resolution
- Close other GPU processes

## One-Liner Installation Script

For RunPod with PyTorch 2.6 + Python 3.12:

```bash
cd /workspace/runpod-slim/ComfyUI/custom_nodes && \
git clone https://github.com/nunchaku-tech/ComfyUI-nunchaku.git && \
/workspace/runpod-slim/ComfyUI/.venv/bin/pip install -r ComfyUI-nunchaku/requirements.txt && \
/workspace/runpod-slim/ComfyUI/.venv/bin/pip install https://github.com/nunchaku-tech/nunchaku/releases/download/v1.0.1/nunchaku-1.0.1+torch2.6-cp312-cp312-linux_x86_64.whl && \
cd ComfyUI-nunchaku && \
/workspace/runpod-slim/ComfyUI/.venv/bin/python scripts/update_versions.py && \
echo "Nunchaku installed! Restart ComfyUI to use."
```

## References

- [Nunchaku Documentation](https://nunchaku.tech/docs/nunchaku/installation/installation.html)
- [ComfyUI-nunchaku Documentation](https://nunchaku.tech/docs/ComfyUI-nunchaku/get_started/installation.html)
- [GitHub Releases](https://github.com/nunchaku-tech/nunchaku/releases)
- [Nunchaku FAQ](https://github.com/nunchaku-tech/nunchaku/discussions/262)

---

*Last updated: December 2025*
*Tested on: RunPod RTX 4090, CUDA 12.4, Python 3.12, PyTorch 2.6*
