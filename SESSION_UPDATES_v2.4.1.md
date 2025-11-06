# Session Updates - v2.4.1 (2025-01-07)

## Overview
This document summarizes all changes made during the v2.4.1 development session.

## Package Cleanup
**Removed redundant GRAG sampler** - The full GRAG Advanced Sampler is now maintained in the separate [ComfyUI-GRAG-ArchAi3D](https://github.com/amir84ferdos/ComfyUI-GRAG-ArchAi3D) repository. This package retains GRAG utility nodes (GRAG Modifier, GRAG Encoder) for conditioning metadata injection.

---

## 1. Auto-Facing Feature Added to Cinematography Prompt Builder

### What Changed
Added `auto_facing` parameter to **Cinematography Prompt Builder** node, previously only available in Object Focus Camera v7.

### Why Important
User insight: "i know it is important if you merg it to prompt at begiing it will have more affect base on my experince"

Based on vision-language model attention mechanisms, placing the facing directive at the **beginning** of prompts provides maximum attention weight and effectiveness.

### Implementation Details

**File**: `nodes/camera/cinematography_prompt_builder.py`

1. **Added Parameter** (Lines 159-165):
```python
"auto_facing": ("BOOLEAN", {
    "default": True,
    "tooltip": "Automatically face camera toward target subject (recommended for object photography).\n"
               "• True = Camera points directly at subject from chosen angle\n"
               "• False = Camera positioned at angle but may not face subject directly"
}),
```

2. **Simple Prompt Generation** (Lines 685-688):
```python
# AUTO-FACING: Add at the VERY BEGINNING for maximum attention weight
# Only add if enabled AND not front view (front view already implies facing)
if auto_facing and horizontal_angle != "Front View (0°)":
    parts.append(f"Facing {subject} directly")
```

3. **Professional Prompt Generation** (Lines 757-763):
```python
# AUTO-FACING: Add at BEGINNING for maximum attention (before "Next Scene:")
# Only add if enabled AND not front view
if auto_facing and horizontal_angle != "Front View (0°)":
    if language in ["Chinese (Best for dx8152 LoRAs)", "Hybrid (Chinese + English)"]:
        prompt_parts.append(f"面对{subject}")  # "Facing {subject}"
    else:
        prompt_parts.append(f"Facing {subject} directly")
```

### Behavior
- **Active**: When `auto_facing=True` AND `horizontal_angle != "Front View (0°)"`
- **Inactive**: When `auto_facing=False` OR `horizontal_angle == "Front View (0°)"` (redundant)
- **Language Support**: Full Chinese/English/Hybrid support

---

## 2. Parameter Order Bug Fix

### Problem
User reported: "i saw it is not working , the auto facing option is not working check it"

### Root Cause
Parameter order mismatch between INPUT_TYPES definition and function signature.

ComfyUI passes parameters **positionally** based on INPUT_TYPES order. The function signature had parameters in wrong positions.

**Before**:
- INPUT_TYPES position 5: `auto_facing`
- Function signature position 8: `auto_facing`

### Fix
Reordered function signature to match INPUT_TYPES exactly (Lines 591-601):
```python
def generate_cinematography_prompt(self, target_subject, shot_type, camera_angle,
                                  horizontal_angle, auto_facing,  # CRITICAL: Must match INPUT_TYPES order
                                  depth_of_field, style_mood, prompt_language,
                                  ...)
```

**File**: `nodes/camera/cinematography_prompt_builder.py`

---

## 3. Chinese Distance Format Improvement

### Problem
User showed prompt: "距离远距离" (distance far distance) - redundant and unclear

### Solution
Changed `_get_distance_chinese()` function to return specific meter values instead of generic descriptions.

**Before**: "远距离" (far distance)
**After**: "四米" (4 meters)

### Implementation (Lines 903-943)
```python
def _get_distance_chinese(self, distance):
    """Convert distance to Chinese words with specific meter values"""
    chinese_numbers = {
        0: "零", 1: "一", 2: "两", 3: "三", 4: "四",
        5: "五", 6: "六", 7: "七", 8: "八", 9: "九",
        10: "十", 15: "十五", 20: "二十"
    }

    if distance == int(distance):
        dist_int = int(distance)
        if dist_int in chinese_numbers:
            return f"{chinese_numbers[dist_int]}米"
        else:
            return f"{dist_int}米"
    # ... handles half meters and decimals
```

**File**: `nodes/camera/cinematography_prompt_builder.py`

---

## 4. GRAG Nodes Fixed for ComfyUI Update

### Problem
User reported: "there is an update for comfyui and t broken my GRAG nodes"

Error: `RuntimeError: The size of tensor a (8430) must match the size of tensor b (24)`

### Root Cause
ComfyUI commit `4cd881866bad0cde70273cc123d725693c1f2759` changed:
- Tensor format: **BSHD → BHND** (Batch, Heads, Sequence, Dim)
- RoPE function: `apply_rotary_emb` → `apply_rope1`
- Import location: `comfy.ldm.qwen_image.model` → `comfy.ldm.flux.math`

### Solution Applied

**File**: `nodes/sampling/archai3d_grag_sampler.py`

#### 1. QKV Projection Format (Lines 187-195)
**Before**:
```python
img_query = attn_module.to_q(hidden_states).unflatten(-1, (attn_module.heads, -1))
```

**After**:
```python
img_query = attn_module.to_q(hidden_states).view(batch_size, seq_img, attn_module.heads, -1).transpose(1, 2).contiguous()
```

Changes to BHND format: `[B, H, N, D]`

#### 2. Concatenation Dimension (Lines 203-206)
**Before**: `dim=1` (sequence in BSHD)
**After**: `dim=2` (sequence in BHND)

```python
joint_query = torch.cat([txt_query, img_query], dim=2)
```

#### 3. RoPE Function Update (Lines 208-211)
**Before**:
```python
from comfy.ldm.qwen_image.model import apply_rotary_emb
joint_query = apply_rotary_emb(joint_query, image_rotary_emb)
```

**After**:
```python
from comfy.ldm.flux.math import apply_rope1
joint_query = apply_rope1(joint_query, image_rotary_emb)
```

#### 4. GRAG Processing Format Conversion (Lines 216-232)
```python
# Convert BHND to BSHD format for GRAG, then flatten
# BHND: [B, H, S, D] -> BSHD: [B, S, H, D] -> [B, S, H*D]
joint_key_for_grag = joint_key.transpose(1, 2).contiguous()  # BHND -> BSHD
joint_key_flat = joint_key_for_grag.flatten(start_dim=2)  # [B, S, H*D]

# Apply GRAG reweighting
joint_key_flat = apply_grag_to_keys(...)

# Unflatten back to BSHD then transpose back to BHND
joint_key_for_grag = joint_key_flat.unflatten(-1, (attn_module.heads, -1))  # [B, S, H, D]
joint_key = joint_key_for_grag.transpose(1, 2).contiguous()  # BSHD -> BHND
```

#### 5. Attention Call with skip_reshape (Lines 241-252)
**Key Insight**: With `skip_reshape=True` and default `skip_output_reshape=False`:
- **Input**: BHND format
- **Output**: BSD format (not BHND!)

```python
# Pass tensors in BHND format with skip_reshape=True (new Qwen format)
# Output will be BSD format (batch, seq, heads*dim) due to default skip_output_reshape=False
joint_hidden_states = optimized_attention_masked(
    joint_query, joint_key, joint_value, attn_module.heads,
    attention_mask, transformer_options=transformer_options,
    skip_reshape=True  # Input is BHND, output is BSD (due to default reshape)
)

# Split streams - output is already in BSD format, no transpose needed
txt_attn_output = joint_hidden_states[:, :seq_txt, :]
img_attn_output = joint_hidden_states[:, seq_txt:, :]
```

**Critical Fix**: Removed incorrect transpose that was treating output as BHND when it's actually BSD.

### Testing
User confirmed: "ok GRAG is working"

---

## Files Modified

1. **nodes/camera/cinematography_prompt_builder.py**
   - Added auto_facing parameter
   - Fixed parameter order
   - Improved Chinese distance formatting
   - Lines: 159-165, 591-601, 685-688, 757-763, 903-943

2. **nodes/sampling/archai3d_grag_sampler.py**
   - Complete GRAG tensor format refactor for ComfyUI update
   - Lines: 183-252 (entire attention forward pass)

---

## Documentation Created

1. **AUTO_FACING_FEATURE.md** - Complete auto_facing documentation
2. **SESSION_UPDATES_v2.4.1.md** - This file

---

## Version
- **Version**: v2.4.1
- **Date**: 2025-01-07
- **Branch**: main

---

## Next Steps

User should:
1. Test auto_facing feature in ComfyUI workflows
2. Test GRAG sampler with latest ComfyUI
3. Consider updating version in `__init__.py` and `pyproject.toml` if releasing

---

**Author**: Amir Ferdos (ArchAi3d)
**Assisted by**: Claude Code (Anthropic)
