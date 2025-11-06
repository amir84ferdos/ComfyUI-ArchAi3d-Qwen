# Auto-Facing Feature Documentation

## Overview

The `auto_facing` parameter ensures the camera automatically points directly at the target subject from any horizontal angle position. This feature is now available in both **Object Focus Camera v7** and **Cinematography Prompt Builder**.

---

## Purpose

When positioning the camera at angles (left, right, side, back), `auto_facing` controls whether the camera:
- ✅ **Points directly at the subject** (auto_facing = True)
- ❌ **Maintains forward orientation** without explicitly facing the subject (auto_facing = False)

---

## Implementation Details

### Parameter Specification

```python
"auto_facing": ("BOOLEAN", {
    "default": True,
    "tooltip": "Automatically face camera toward target subject (recommended for object photography).\n"
               "• True = Camera points directly at subject from chosen angle\n"
               "• False = Camera positioned at angle but may not face subject directly"
})
```

### Prompt Positioning Strategy

**Key Finding**: Based on user experience with vision-language models, placing `auto_facing` guidance **at the beginning of the prompt** provides maximum attention weight and effectiveness.

**Prompt Structure:**

```
[FACING DIRECTIVE] + [Main Camera Prompt] + [Details]
```

**Examples:**

#### Simple Prompt (English):
```
Facing the dishwasher directly, An eye-level medium shot of the dishwasher, taken from a vantage point two meters away, positioned from thirty degrees to the left for a corner perspective, with deep depth of field keeping everything in focus, in architectural style
```

#### Professional Prompt (Chinese):
```
面对dishwasher，Next Scene: 将镜头转为标准镜头(50mm)，中景构图，平视查看dishwasher，从左侧30度拍摄,呈现转角视角，距离两米
```

---

## When Auto-Facing Is Applied

### ✅ Active Conditions:
- `auto_facing = True` (default)
- `horizontal_angle != "Front View (0°)"` (since front view already implies facing)

### ❌ Not Applied When:
- `auto_facing = False`
- `horizontal_angle = "Front View (0°)"` (redundant - front view inherently faces subject)

---

## Usage Examples

### Example 1: Dishwasher Side View with Auto-Facing

**Settings:**
- Target Subject: `dishwasher`
- Shot Type: `Medium Shot (MS)`
- Camera Angle: `Eye Level`
- Horizontal Angle: `Side Left (90°)`
- **auto_facing: `True`** ✅

**Result:**
Camera positions at the left side (90°) AND rotates to face the dishwasher directly, ensuring the dishwasher is centered in frame despite the side positioning.

---

### Example 2: Architectural Context Shot without Auto-Facing

**Settings:**
- Target Subject: `kitchen counter`
- Shot Type: `Wide Shot (WS)`
- Camera Angle: `Eye Level`
- Horizontal Angle: `Angled Right 30°`
- **auto_facing: `False`** ❌

**Result:**
Camera positions at 30° to the right but maintains forward orientation, potentially showing the counter as part of a broader environmental context rather than centered.

---

## Technical Implementation

### Cinematography Prompt Builder

#### Simple Prompt Generation ([cinematography_prompt_builder.py:685-688](nodes/camera/cinematography_prompt_builder.py#L685-L688)):

```python
# AUTO-FACING: Add at the VERY BEGINNING for maximum attention weight
# Only add if enabled AND not front view (front view already implies facing)
if auto_facing and horizontal_angle != "Front View (0°)":
    parts.append(f"Facing {subject} directly")
```

#### Professional Prompt Generation ([cinematography_prompt_builder.py:757-763](nodes/camera/cinematography_prompt_builder.py#L757-L763)):

```python
# AUTO-FACING: Add at BEGINNING for maximum attention (before "Next Scene:")
# Only add if enabled AND not front view
if auto_facing and horizontal_angle != "Front View (0°)":
    if language in ["Chinese (Best for dx8152 LoRAs)", "Hybrid (Chinese + English)"]:
        prompt_parts.append(f"面对{subject}")  # "Facing {subject}"
    else:
        prompt_parts.append(f"Facing {subject} directly")
```

---

## Why Positioning Matters

### User Observation:
> "i know it is important if you merg it to prompt at begiing it will have more affect base on my experince"

This aligns with attention mechanisms in transformer-based vision-language models:

1. **Positional Bias**: Tokens at the beginning of prompts receive higher attention weights
2. **Semantic Anchoring**: Early instructions establish the primary directive for the generation
3. **Context Precedence**: Models process sequential information with recency and primacy effects

By placing `auto_facing` directive **first**, we ensure maximum model attention to this critical orientation instruction.

---

## Integration with Other Features

### Compatible with:
- ✅ All horizontal angles (15°, 30°, 45°, 90°, 180°)
- ✅ All vertical camera angles (Eye Level, High Angle, Low Angle, etc.)
- ✅ All shot sizes (ECU to EWS)
- ✅ Perspective correction modes (Natural, Architectural, Tilt-Shift)
- ✅ All lens types
- ✅ Chinese/English/Hybrid language modes

### Automatically Disabled:
- Front View (0°) - redundant since front view inherently faces subject
- When explicitly disabled by user (`auto_facing = False`)

---

## Practical Use Cases

### 🎯 Object Photography (Recommended: True)
- Product photography requiring subject prominence
- Furniture visualization from multiple angles
- Appliance close-ups (dishwashers, ovens, refrigerators)
- Detail shots of architectural elements

### 🏛️ Environmental Photography (Consider: False)
- Architectural context shots
- Room overview with subject as part of environment
- Documentary-style environmental capture
- Spatial relationship emphasis over subject focus

---

## Version History

- **v2.4.1** (2025-01-07): Added `auto_facing` to Cinematography Prompt Builder
  - Placed at beginning of prompts for maximum attention weight
  - Full Chinese translation support (面对)
  - Automatic disable for Front View (0°)

- **v2.3.0** (2025-01-06): Original implementation in Object Focus Camera v7
  - Vantage point mode support
  - Boolean toggle for camera orientation control

---

## References

- User feedback: Prompt positioning significantly affects model attention
- Vision-language model research: Positional encoding and attention weights
- Object Focus Camera v7: Original auto_facing implementation

---

**Author**: Amir Ferdos (ArchAi3d)
**Feature Version**: v2.4.1
**Implementation Date**: 2025-01-07
**Based on**: User experience and vision-language model attention mechanisms
