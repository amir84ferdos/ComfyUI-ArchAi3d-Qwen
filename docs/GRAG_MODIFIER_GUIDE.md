# 🎚️ GRAG Modifier Guide - Universal Fine-Grained Control

> **Node:** `ArchAi3D GRAG Modifier`
> **Category:** ArchAi3d/Qwen → Core - Utils
> **Version:** 2.1.1 (Phase 2A - Functional)
> **Type:** Universal Conditioning Modifier
> **Status:** ✅ Fully Functional (requires GRAG Sampler)

---

## 🎯 What Is GRAG Modifier?

**Universal conditioning modifier** that adds GRAG (Group-Relative Attention Guidance) to ANY encoder's output.

**⚠️ IMPORTANT:** To see actual GRAG effects, you MUST use the **GRAG Sampler** node. The GRAG Modifier only prepares metadata - the GRAG Sampler applies the actual attention reweighting during generation.

### Why Use This Instead of GRAG Encoder?

| Feature | GRAG Modifier ✅ | GRAG Encoder |
|---------|-----------------|--------------|
| Works with ALL encoders | ✅ Yes | ❌ GRAG only |
| Code duplication | ✅ None | ❌ Duplicates encoder |
| Workflow flexibility | ✅ Optional (skip it) | ⚠️ Replace encoder |
| A/B testing | ✅ Add/remove node | ⚠️ Swap encoders |
| Maintenance | ✅ Update once | ❌ Update each encoder |
| **Recommended** | ✅ **Yes** | ⚠️ Testing only |

---

## 📋 Quick Start

### ✅ Complete Functional Workflow (REQUIRED):

```
[Images] → [Any Encoder V2] → [GRAG Modifier] → [GRAG Sampler] → [VAE Decode] → [Output]
                                     ↓                  ↓
                              Prepare metadata    Apply reweighting
```

**Critical:** You MUST use `🎚️ GRAG Sampler` instead of standard KSampler to see GRAG effects!

### Without GRAG (Standard):

```
[Images] → [Any Encoder V2] → [Standard KSampler] → [VAE Decode] → [Output]
                                ↓
                           Skip GRAG entirely
```

---

## 🎮 Parameters

### Required Input:

| Parameter | Type | Description |
|-----------|------|-------------|
| `conditioning` | CONDITIONING | Output from ANY encoder (V1, V2, V3, Simple) |

### GRAG Controls:

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| **`enable_grag`** | Boolean | False | **Master toggle** - Passthrough if disabled |
| `grag_strength` | 0.8-1.7 | 1.0 | **Main control** - Edit intensity |
| | | | 0.8 = Subtle (preserve more) |
| | | | 1.0 = Balanced (recommended) |
| | | | 1.7 = Strong (maximum change) |
| `grag_cond_b` | 0.0-2.0 | 1.0 | Base conditioning strength |
| | | | Lower = more preservation |
| | | | Higher = more change |
| `grag_cond_delta` | 0.0-2.0 | 1.0 | Delta conditioning strength |
| | | | Controls attention difference |

---

## 💡 Usage Examples

### Example 1: Basic GRAG Enhancement

**Setup:**
```
Clean Room Prompt → Encoder V2 → GRAG Modifier → Sampler
```

**GRAG Settings:**
```
enable_grag: True
grag_strength: 1.0
grag_cond_b: 1.0
grag_cond_delta: 1.0
```

**Result:** Balanced fine-grained control with better structure preservation

---

### Example 2: Window Preservation Mode

**Scenario:** Construction site with windows - must preserve windows

**GRAG Settings:**
```
enable_grag: True
grag_strength: 0.85  ← Lower for preservation
grag_cond_b: 0.8     ← Reduce change
grag_cond_delta: 0.9
```

**Result:** Subtle edits that keep windows intact

---

### Example 3: Maximum Transformation

**Scenario:** Complete room redesign - change everything

**GRAG Settings:**
```
enable_grag: True
grag_strength: 1.5   ← Higher for change
grag_cond_b: 1.3
grag_cond_delta: 1.4
```

**Result:** Strong transformation with controlled quality

---

### Example 4: A/B Testing

**Test GRAG vs Standard:**

1. **Run 1**: Remove GRAG Modifier node → Standard workflow
2. **Run 2**: Add GRAG Modifier with `enable_grag: True`
3. **Compare**: Same seed, same settings, only GRAG differs

---

## 🔄 Workflow Patterns

### Pattern 1: Optional Enhancement

```
┌─────────┐    ┌──────────┐    ┌──────────────┐    ┌─────────┐
│ Images  │───→│Encoder V2│───→│GRAG Modifier │───→│ Sampler │
└─────────┘    └──────────┘    │(enabled=True)│    └─────────┘
                                └──────────────┘
                                       ↓
                                Skip by removing node
```

### Pattern 2: Encoder Comparison

```
Test different encoders with same GRAG:

┌──────────┐
│Encoder V1│───┐
└──────────┘   │
               ├─→ GRAG Modifier → Sampler
┌──────────┐   │
│Encoder V2│───┘
└──────────┘
```

### Pattern 3: Multiple GRAG Tests

```
Same encoder, different GRAG settings:

Encoder V2 ─→ GRAG (0.85) ─→ Test 1
           ─→ GRAG (1.0)  ─→ Test 2
           ─→ GRAG (1.5)  ─→ Test 3
```

---

## 🎯 Best Practices

### ✅ DO:

1. **Start with default** (enable_grag=False, strength=1.0)
2. **Test incrementally** - Adjust one parameter at a time
3. **Use same seed** for A/B comparison
4. **Document settings** that work for your use case
5. **Keep enable_grag=False** when GRAG not needed

### ❌ DON'T:

1. **Don't max all parameters** - Start conservative
2. **Don't change multiple values** between tests
3. **Don't forget to enable** - Check enable_grag=True
4. **Don't use with wrong sampler** - Needs GRAG-aware sampler (future)

---

## 🔬 Parameter Tuning Guide

### Finding Your Sweet Spot:

#### Step 1: Enable GRAG
```
enable_grag: True
grag_strength: 1.0  ← Start here
grag_cond_b: 1.0
grag_cond_delta: 1.0
```

#### Step 2: Adjust Main Strength
```
Test: 0.8, 0.9, 1.0, 1.1, 1.2, 1.3
Find where quality is best for your use case
```

#### Step 3: Fine-Tune Secondary Parameters
```
If too much change: Reduce cond_b to 0.8-0.9
If too weak: Increase cond_b to 1.2-1.5
If artifacts: Reduce cond_delta to 0.8-0.9
```

#### Step 4: Final Polish
```
Adjust in 0.01 increments for perfect result
```

---

## 📊 Troubleshooting

### Problem: No visual difference when GRAG enabled

**Cause:** You're using standard KSampler instead of GRAG Sampler
**Solution:** Replace KSampler with `🎚️ GRAG Sampler` node
**Why:** GRAG Modifier only prepares metadata. GRAG Sampler actually applies the attention reweighting.

**Correct Workflow:**
```
Encoder → GRAG Modifier (enable_grag=True) → GRAG Sampler → Output ✅
```

**Incorrect Workflow:**
```
Encoder → GRAG Modifier (enable_grag=True) → KSampler → Output ❌ (no effect)
```

---

### Problem: Can't find GRAG Sampler node

**Solution:** Look for `🎚️ GRAG Sampler (Fine-Grained Control)` in:
- Category: `ArchAi3d/Qwen` → Sampling section
- Alternative: Search "GRAG Sampler" in node browser

---

### Problem: Can't find GRAG Modifier node

**Check:**
1. ComfyUI restarted after installation?
2. Node appears in: `ArchAi3d/Qwen` → `🎚️ GRAG Modifier`
3. Console shows: "Core Utils: 2 nodes"

---

### Problem: What's difference from GRAG Encoder?

**GRAG Modifier** (Recommended):
- ✅ Works with ANY encoder
- ✅ Optional (skip if not needed)
- ✅ Clean separation of concerns

**GRAG Encoder**:
- ⚠️ Standalone encoder with GRAG built-in
- ⚠️ May be deprecated later
- ⚠️ Less flexible

---

## 🚀 Advanced Usage

### Conditional GRAG Application

```python
# In your custom workflow:
if scene_has_windows:
    grag_strength = 0.85  # Preserve
else:
    grag_strength = 1.3   # Transform
```

### Per-Material GRAG Settings

```
Material Change: grag_strength = 1.2
Scaffolding Removal: grag_strength = 0.9
Watermark Removal: grag_strength = 1.0
```

---

## 📈 Expected Results

### With GRAG vs Without:

| Aspect | Without GRAG | With GRAG (0.85) | With GRAG (1.5) |
|--------|--------------|------------------|-----------------|
| Window Preservation | ⚠️ Inconsistent | ✅ Excellent | ⚠️ May change |
| Structure Accuracy | ✅ Good | ✅ Excellent | ⚠️ Less accurate |
| Edit Strength | 🔒 Fixed | 🎚️ Adjustable | 🎚️ Maximum |
| Artifacts | ⚠️ Some | ✅ Fewer | ⚠️ More |
| Use Case | General | **Preservation** | **Transformation** |

---

## 🔮 Development Status

### Phase 1: Metadata Preparation (✅ Completed)
- ✅ Node creates GRAG configuration
- ✅ Adds metadata to conditioning
- ✅ Tested and working

### Phase 2A: Core Integration (✅ Completed)
- ✅ GRAG attention reweighting utility
- ✅ GRAG-aware sampler node
- ✅ Real attention manipulation working
- ✅ Functional fine-grained control (0.8-1.7)

### Phase 2B: Advanced Features (Future)
- [ ] Multi-resolution tier support
- [ ] Per-layer GRAG control
- [ ] Layer-wise strength scheduling
- [ ] Attention map visualization

### Phase 3: Production Hardening (Future)
- [ ] Preset parameter sets (Subtle/Balanced/Strong)
- [ ] Per-region GRAG control with masks
- [ ] Auto parameter tuning based on content
- [ ] Performance optimization (JIT compilation)

---

## 💬 Comparison: Modifier vs Encoder

### When to Use GRAG Modifier (Recommended):

✅ Testing GRAG with different encoders
✅ Optional fine-grained control
✅ Clean, modular workflows
✅ Future-proof approach
✅ A/B testing ease

### When to Use GRAG Encoder:

⚠️ Testing GRAG-specific encoder configs
⚠️ Standalone GRAG experiments
⚠️ Temporary use (may be deprecated)

---

## 📚 Related Documentation

- [GRAG Encoder Guide](./GRAG_ENCODER_GUIDE.md) - Standalone encoder version
- [Qwen Encoder V2 Guide](./QWEN_ENCODER_V2_GUIDE.md) - Compatible encoder
- [Clean Room Prompt Guide](./CLEAN_ROOM_PROMPT_GUIDE.md) - Prompt building

---

## 🆘 Support

### Getting Help:

**Issues:** [GitHub Issues](https://github.com/amir84ferdos/ComfyUI-ArchAi3d-Qwen/issues)
**Email:** Amir84ferdos@gmail.com
**LinkedIn:** https://www.linkedin.com/in/archai3d/

### Contributing:

Want to help integrate full GRAG pipeline?
1. Study [GRAG-Image-Editing](https://github.com/little-misfit/GRAG-Image-Editing)
2. Understand Qwen attention mechanisms
3. Contact for collaboration

---

**Version:** 2.1.1
**Last Updated:** 2025-11-03
**Status:** Experimental - Metadata Preparation
**Author:** Amir Ferdos (ArchAi3d)
**Based on:** GRAG-Image-Editing by little-misfit

---

## ✨ Quick Reference Card

```
┌─────────────────────────────────────────────┐
│  🎚️ GRAG Modifier - Quick Settings        │
├─────────────────────────────────────────────┤
│                                             │
│  Subtle (Windows):                          │
│    enable: True                             │
│    strength: 0.85                           │
│    cond_b: 0.8                              │
│    cond_delta: 0.9                          │
│                                             │
│  Balanced (Recommended):                    │
│    enable: True                             │
│    strength: 1.0                            │
│    cond_b: 1.0                              │
│    cond_delta: 1.0                          │
│                                             │
│  Strong (Transform):                        │
│    enable: True                             │
│    strength: 1.5                            │
│    cond_b: 1.3                              │
│    cond_delta: 1.4                          │
│                                             │
│  Standard (No GRAG):                        │
│    enable: False                            │
│    (or remove node)                         │
│                                             │
└─────────────────────────────────────────────┘
```
