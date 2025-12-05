# RTX 5090 / Blackwell GPU Compatibility Guide

Complete guide for running ArchAi3D nodes and Nunchaku on NVIDIA RTX 5090 (Blackwell architecture).

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **NVIDIA Driver** | 566.03+ | **570+** |
| **CUDA** | 12.6 | **12.8** |
| **PyTorch** | 2.5.0 | **2.7.0+ (cu128)** |
| **Python** | 3.10 | 3.11 or 3.12 |
| **TensorRT** | 10.7 | 10.8 |
| **cuDNN** | 9.x | 9.x |

### GPU Architecture

| GPU | Architecture | Compute Capability |
|-----|--------------|-------------------|
| RTX 5090 | Blackwell | sm_120 (12.0) |
| RTX 5080 | Blackwell | sm_120 (12.0) |
| RTX 5070 Ti | Blackwell | sm_120 (12.0) |
| RTX 5070 | Blackwell | sm_120 (12.0) |
| B100/B200 | Blackwell | sm_120 (12.0) |

---

## Quick Start

### 1. Update PyTorch for Blackwell

```bash
# Uninstall old PyTorch
pip uninstall torch torchvision torchaudio -y

# Install PyTorch with CUDA 12.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 2. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

Expected output:
```
PyTorch: 2.7.0+cu128
CUDA: 12.8
GPU: NVIDIA GeForce RTX 5090
```

### 3. Install Nunchaku (FP4 for Blackwell)

```bash
# Use the Nunchaku Installer node in ComfyUI, or manually:
pip install https://github.com/nunchaku-tech/nunchaku/releases/download/v1.0.1/nunchaku-1.0.1+torch2.8-cp312-cp312-linux_x86_64.whl
```

---

## Known Issues & Solutions

### 1. xformers Compatibility

**Problem:** Pre-built xformers wheels don't include sm_120 support.

**Error:**
```
NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the current PyTorch installation
```

**Solution A: Build from source**
```bash
pip uninstall xformers -y
pip install ninja

export TORCH_CUDA_ARCH_LIST="12.0"
git clone --depth=1 https://github.com/facebookresearch/xformers --recursive
cd xformers && python setup.py install
```

**Solution B: Use PyTorch's native SDPA**
- xformers is optional
- PyTorch's built-in Scaled Dot Product Attention works without it
- Slightly slower but no compatibility issues

---

### 2. Triton Issues

**Problem:** Triton kernels need recompilation for sm_120.

**Solution: Update Triton**
```bash
# Linux
pip install -U --pre triton

# Windows
pip install -U --pre triton-windows
```

**Fallback:** Disable `torch.compile()` if issues persist.

---

### 3. TensorRT Not Working

**Problem:** TensorRT engines compiled for older architectures crash on sm_120.

**Solution:** Install TensorRT 10.7+
- Download from [NVIDIA Developer Site](https://developer.nvidia.com/tensorrt)
- TensorRT 10.8 recommended for FLUX FP4 models

---

### 4. "sm_120 unsupported" Error

**Full Error:**
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**Cause:** PyTorch wasn't compiled with Blackwell support.

**Fix:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

---

## Model Recommendations for Blackwell

### Use FP4 Models (Recommended)

Blackwell GPUs have optimized FP4 tensor cores. Use FP4 quantized models for best performance:

| Model Type | Blackwell Recommendation |
|------------|-------------------------|
| Nunchaku Flux | `svdq-fp4_*.safetensors` |
| Nunchaku Qwen | `svdq-fp4_*.safetensors` |
| GGUF Models | Q4_K_M or Q5_K_M |

### Avoid INT4 on Blackwell

INT4 models are optimized for Ada (RTX 40 series). On Blackwell:
- INT4 works but doesn't utilize FP4 tensor cores
- FP4 models are ~20-30% faster on RTX 5090

---

## ComfyUI Setup Commands

### Full Blackwell Setup (Linux)

```bash
# 1. Update system
sudo apt update && sudo apt upgrade -y

# 2. Install NVIDIA Driver 570+
sudo apt install nvidia-driver-570

# 3. Create virtual environment
python3 -m venv comfy-env
source comfy-env/bin/activate

# 4. Install PyTorch cu128
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 5. Clone ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI
pip install -r requirements.txt

# 6. Install ArchAi3D nodes
cd custom_nodes
git clone https://github.com/amir84ferdos/ComfyUI-ArchAi3d-Qwen
pip install -r ComfyUI-ArchAi3d-Qwen/requirements.txt

# 7. Install Nunchaku (use installer node or manual)
pip install https://github.com/nunchaku-tech/nunchaku/releases/download/v1.0.1/nunchaku-1.0.1+torch2.8-cp312-cp312-linux_x86_64.whl

# 8. Start ComfyUI
cd ..
python main.py --listen 0.0.0.0 --port 8188
```

### Windows Setup

```powershell
# 1. Install Python 3.11 or 3.12

# 2. Create virtual environment
python -m venv comfy-env
.\comfy-env\Scripts\activate

# 3. Install PyTorch cu128
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 4. Download ComfyUI portable (v0.3.30+)
# Or clone and setup manually

# 5. Install dependencies as above
```

---

## Using ArchAi3D Installer Nodes

### Nunchaku Installer Node

1. Add node: `ArchAi3d/Utils` -> `Nunchaku Installer`
2. Select action:
   - **Check System** - Verify GPU, CUDA, PyTorch versions
   - **Install Nunchaku** - Auto-detect and install correct wheel
   - **Update Versions File** - Refresh available models

The installer automatically:
- Detects RTX 5090 (sm_120)
- Warns if PyTorch < 2.8
- Warns if CUDA < 12.8
- Recommends FP4 models

### Dependency Installer Node

1. Add node: `ArchAi3d/Utils` -> `Dependency Installer`
2. Select action:
   - **Check System** - Show all detected versions
   - **Install Core** - Basic dependencies
   - **Install SAM3** - Segmentation dependencies
   - **Install Metric3D** - Depth estimation dependencies
   - **Install All** - Everything

---

## Performance Tips for RTX 5090

### 1. Use FP4 Models
- 20-30% faster than INT4/INT8
- Better utilization of Blackwell tensor cores

### 2. Increase Batch Size
- RTX 5090 has 32GB VRAM
- Can handle larger batches than 4090

### 3. Enable torch.compile (if supported)
```python
# In your workflow or script
model = torch.compile(model, mode="reduce-overhead")
```

### 4. Use SageAttention
- Works better than xformers on Blackwell
- Install: `pip install sageattention`

---

## Troubleshooting

### Check Your Setup

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA version
nvcc --version

# Check PyTorch CUDA
python -c "import torch; print(torch.version.cuda)"

# Check compute capability
python -c "import torch; print(torch.cuda.get_device_capability(0))"
```

### Expected Values for RTX 5090

```
Driver: 570.x or higher
CUDA: 12.8
PyTorch CUDA: 12.8
Compute Capability: (12, 0)
```

### Common Fixes

| Issue | Solution |
|-------|----------|
| "sm_120 unsupported" | Install PyTorch cu128 |
| xformers crash | Build from source or use SDPA |
| TensorRT error | Upgrade to TensorRT 10.7+ |
| Slow performance | Use FP4 models, not INT4 |
| CUDA OOM | Reduce batch size or image resolution |

---

## References

- [ComfyUI RTX 50 Series Support](https://github.com/comfyanonymous/ComfyUI/discussions/6643)
- [ComfyUI Blog - Blackwell Setup](https://blog.comfy.org/p/how-to-get-comfyui-running-on-your)
- [PyTorch sm_120 Issue](https://github.com/pytorch/pytorch/issues/159207)
- [NVIDIA TensorRT FP4 Blog](https://developer.nvidia.com/blog/nvidia-tensorrt-unlocks-fp4-image-generation-for-nvidia-blackwell-geforce-rtx-50-series-gpus/)
- [Nunchaku Installation](https://deepwiki.com/mit-han-lab/nunchaku/1.3-installation)

---

*Last updated: December 2025*
*Tested on: RTX 5090, CUDA 12.8, PyTorch 2.7+, Python 3.12*
