# ⭐ GRAG Encoder Guide - Fine-Grained Editing Control

> **Node:** `ArchAi3D Qwen GRAG Encoder`
> **Category:** ArchAi3d/Qwen
> **Version:** 2.1.1
> **Status:** Experimental (Placeholder Implementation)

---

## 📋 Table of Contents

1. [What is GRAG?](#what-is-grag)
2. [How It Works](#how-it-works)
3. [Node Parameters](#node-parameters)
4. [Usage Guide](#usage-guide)
5. [Integration with Clean Room Workflow](#integration-with-clean-room-workflow)
6. [Parameter Tuning Tips](#parameter-tuning-tips)
7. [Current Limitations](#current-limitations)
8. [Future Development](#future-development)

---

## 🎯 What is GRAG?

**GRAG (Group-Relative Attention Guidance)** is a training-free technique for fine-grained image editing control.

### Key Benefits:
- **No Training Required**: Works with existing Qwen-Image-Edit models
- **Fine-Grained Control**: Continuous adjustment from 0.8 to 1.7 (0.01 increments)
- **Better Preservation**: Improved structure/window preservation in edits
- **Artifact Reduction**: Cleaner results with less noise
- **Gradual Transformations**: Precise control over edit intensity

### What Makes It Special:
GRAG manipulates **attention mechanisms** in the diffusion model by re-weighting delta values between tokens and shared attention biases. This allows precise control without retraining the model.

---

## 🔧 How It Works

### Two-Tier Resolution Scaling:

```
Tier 1 (Base Reference):
- Resolution: 512×512
- Scale: 1.0 (fixed)
- Purpose: Stable reference point

Tier 2 (Modified):
- Resolution: 4096×4096
- Scale: Controlled by cond_b and cond_delta
- Purpose: Fine-tuned attention guidance
```

### Attention Manipulation:

```python
# Simplified concept:
attention_delta = high_res_attention - base_attention
weighted_delta = attention_delta * grag_strength * cond_b * cond_delta
final_attention = base_attention + weighted_delta
```

This gives you **continuous control** over how strongly edits are applied.

---

## 📊 Node Parameters

### 🖼️ Image Inputs (Required)

| Parameter | Type | Description |
|-----------|------|-------------|
| `image1` | IMAGE | First image for vision encoder |
| `image2` | IMAGE | Second image for vision encoder |
| `image3` | IMAGE | Third image for vision encoder |
| `image1_vae` | IMAGE | First image for VAE latents |
| `image2_vae` | IMAGE | Second image for VAE latents |
| `image3_vae` | IMAGE | Third image for VAE latents |

### 📝 Text Inputs (Required)

| Parameter | Type | Description |
|-----------|------|-------------|
| `user_prompt` | STRING | Main editing instruction |
| `system_prompt` | STRING (optional) | System-level guidance |

### ⭐ GRAG Parameters (Main Controls)

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| **`grag_strength`** | 0.8-1.7 | 1.0 | **Main GRAG intensity control** |
| | | | 0.8 = Subtle edits (preserves more) |
| | | | 1.0 = Balanced edits (recommended) |
| | | | 1.7 = Strong edits (maximum transformation) |
| | | | Adjust in 0.01 increments for fine control |
| **`grag_cond_b`** | 0.0-2.0 | 1.0 | Base conditioning strength |
| | | | Controls base attention weighting |
| | | | Lower = more preservation |
| | | | Higher = more change |
| **`grag_cond_delta`** | 0.0-2.0 | 1.0 | Delta conditioning strength |
| | | | Controls attention delta intensity |
| | | | Fine-tunes divergence from reference |

### 🎨 Standard Qwen Controls

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `context_strength` | 0.0-1.5 | 1.0 | System prompt influence (Stage A) |
| `user_strength` | 0.0-1.5 | 0.6 | User text influence (Stage B) |
| `image1_latent_strength` | 0.0-2.0 | 1.0 | First image reference strength |
| `image2_latent_strength` | 0.0-2.0 | 1.0 | Second image reference strength |
| `image3_latent_strength` | 0.0-2.0 | 1.0 | Third image reference strength |

---

## 📖 Usage Guide

### Basic Workflow:

```
1. Load your images (construction site, reference photos)
2. Connect to GRAG Encoder
3. Connect output to Qwen Sampler (when available)
4. Adjust GRAG parameters for desired intensity
```

### Example Parameter Sets:

#### 🟢 Subtle Preservation (Windows/Structure Critical)
```
grag_strength: 0.85
grag_cond_b: 0.8
grag_cond_delta: 0.9
context_strength: 1.0
user_strength: 0.5
```
**Use when:** Windows must be preserved, minimal structural changes

#### 🟡 Balanced Editing (Recommended Start)
```
grag_strength: 1.0
grag_cond_b: 1.0
grag_cond_delta: 1.0
context_strength: 1.0
user_strength: 0.6
```
**Use when:** General room cleaning, material changes

#### 🔴 Strong Transformation (Maximum Change)
```
grag_strength: 1.5
grag_cond_b: 1.3
grag_cond_delta: 1.4
context_strength: 1.2
user_strength: 0.8
```
**Use when:** Major renovations, complete redesigns

---

## 🏗️ Integration with Clean Room Workflow

### Standard Clean Room Workflow:
```
[Images] → [Clean Room Prompt] → [Qwen Encoder V2] → [Sampler] → [Output]
```

### Enhanced GRAG Workflow:
```
[Images] → [Clean Room Prompt] → [GRAG Encoder] → [Sampler*] → [Output]
                                         ↓
                                 Fine-grained control
                                 Better preservation
                                 Adjustable intensity
```

**Note:** `*` Requires GRAG-compatible sampler (future development)

### Benefits for Clean Room:
- **Better Window Preservation**: GRAG's fine control helps maintain windows
- **Gradual Material Changes**: Test different intensities before final render
- **Artifact Reduction**: Cleaner edges, fewer halos
- **Precise Scaffolding Removal**: Adjustable removal strength

---

## 💡 Parameter Tuning Tips

### Finding the Right GRAG Strength:

**Start with default (1.0), then:**

| Problem | Solution |
|---------|----------|
| Windows changing/disappearing | Reduce to 0.85-0.9 |
| Edits too weak | Increase to 1.1-1.3 |
| Too many artifacts | Reduce cond_delta to 0.8-0.9 |
| Not enough change | Increase cond_b to 1.2-1.5 |
| Halos around objects | Reduce grag_strength + increase user_strength |

### Iterative Tuning Process:

```
1. Start: grag_strength = 1.0
2. Test render
3. Adjust by 0.1 increments
4. When close, adjust by 0.01 increments
5. Fine-tune cond_b and cond_delta last
```

### Pro Tips:

✅ **DO:**
- Start conservative (lower values)
- Adjust one parameter at a time
- Test with same seed for comparison
- Document working parameter sets

❌ **DON'T:**
- Max all parameters at once
- Change multiple values between tests
- Ignore structure preservation warnings
- Skip baseline testing (1.0, 1.0, 1.0)

---

## ⚠️ Current Limitations

### ⚠️ **IMPORTANT: Placeholder Implementation**

This node is currently a **placeholder/metadata preparation** implementation:

**What It Does Now:**
- ✅ Builds GRAG scale configuration
- ✅ Prepares conditioning with GRAG metadata
- ✅ Returns standard Qwen conditioning format with GRAG hints

**What It Needs for Full Functionality:**
- ❌ GRAG-modified QwenImageTransformer2DModel
- ❌ GRAG-modified QwenImageEditPipeline
- ❌ Custom attention reweighting in forward pass
- ❌ Integration with actual GRAG codebase

### Technical Requirements:

To make this fully functional, you need:

1. **GRAG Repository Integration**
   ```bash
   git clone https://github.com/little-misfit/GRAG-Image-Editing.git
   cd GRAG-Image-Editing/Qwen-Edit-GRAG
   pip install -r requirements.txt
   ```

2. **Modified Attention Modules**
   - Replace standard Qwen attention with GRAG-modified version
   - Implement attention delta reweighting
   - Handle multi-resolution tier system

3. **Pipeline Integration**
   - Wrap QwenImageEditPipeline in ComfyUI node
   - Pass GRAG scale configuration through pipeline
   - Handle CUDA device management

---

## 🚀 Future Development

### Phase 1: Core Integration (Current Goal)
- [ ] Integrate actual GRAG pipeline code
- [ ] Create GRAG-compatible sampler node
- [ ] Test with Clean Room workflow
- [ ] Benchmark quality improvements

### Phase 2: Enhancement
- [ ] Add preset parameter sets (subtle/balanced/strong)
- [ ] Create visual parameter guides
- [ ] Add batch processing support
- [ ] Optimize for performance

### Phase 3: Advanced Features
- [ ] Per-region GRAG strength control
- [ ] Mask-guided attention weighting
- [ ] Automatic parameter tuning
- [ ] Real-time preview mode

---

## 📚 Resources

### Original GRAG Research:
- **Repository**: https://github.com/little-misfit/GRAG-Image-Editing
- **Qwen Support**: Added November 2025
- **Paper**: (Link TBD when available)

### Related Documentation:
- [Qwen Encoder V2 Guide](./QWEN_ENCODER_V2_GUIDE.md)
- [Clean Room Prompt Guide](./CLEAN_ROOM_PROMPT_GUIDE.md)
- [Camera Control Guide](./CAMERA_CONTROL_GUIDE.md)

---

## 🆘 Troubleshooting

### Node doesn't appear in ComfyUI
- Restart ComfyUI after installing
- Check console for loading errors
- Verify `__init__.py` includes GRAG encoder

### Parameters have no effect
- **Expected**: This is a placeholder implementation
- **Solution**: Wait for Phase 1 integration or contribute to development

### How to help development?
1. Test placeholder with different parameters
2. Report parameter combinations that would be useful
3. Contribute GRAG pipeline integration code
4. Share use cases and requirements

---

## 🤝 Contributing

Want to help make GRAG fully functional?

**Priority Needs:**
1. GRAG pipeline integration expertise
2. Qwen-Image-Edit pipeline modification
3. Attention mechanism implementation
4. Testing and benchmarking

**Contact:**
- Email: Amir84ferdos@gmail.com
- LinkedIn: https://www.linkedin.com/in/archai3d/
- GitHub: https://github.com/amir84ferdos/ComfyUI-ArchAi3d-Qwen

---

**Version:** 2.1.1
**Last Updated:** 2025-11-03
**Status:** Experimental - Placeholder Implementation
**Author:** Amir Ferdos (ArchAi3d)
**Based on:** GRAG-Image-Editing by little-misfit
