# Custom Details Tooltip Enhancement

## Summary

Enhanced the `custom_details` parameter tooltip in Cinematography Prompt Builder to provide clear examples of compositional specifics that go beyond the 5 core ingredients (Subject, Shot Type, Angle, Focus/DOF, Style).

---

## Changes Made

### File: nodes/camera/cinematography_prompt_builder.py

**Lines 274-284**: Updated `custom_details` tooltip

**Before:**
```python
"custom_details": ("STRING", {
    "default": "",
    "multiline": True,
    "tooltip": "Add any custom details (e.g., 'showing dial and hands', 'with marble backsplash visible')"
}),
```

**After:**
```python
"custom_details": ("STRING", {
    "default": "",
    "multiline": True,
    "tooltip": "Add compositional specifics beyond the 5 ingredients. Examples:\n"
               "• 'The entire stove is centered in the frame, clearly showing it, the marble backsplash, and the range hood above'\n"
               "• 'focusing on the intricate details of a single burner and the cast-iron grate'\n"
               "• 'showing dial and hands clearly'\n"
               "• 'The vantage point is inches away, creating an extremely shallow depth of field'\n"
               "• 'dissolves into a soft, blurred bokeh'\n"
               "• 'The lighting is bright and even, keeping the entire area in sharp focus'"
}),
```

---

## Why This Matters

### Problem
The 5-ingredient formula (Subject, Shot Type, Angle, Focus/DOF, Style) provides the **foundation** for camera prompts, but working examples show that **rich compositional details** make the difference between good and great results.

**Example:**

**5 Ingredients Only:**
```
An eye-level full shot of the green stove, taken from a vantage point four and a half meters away, with deep depth of field, in clean and modern style
```

**5 Ingredients + Custom Details:**
```
An eye-level full shot of the green stove, taken from a vantage point four and a half meters away, with deep depth of field, in clean and modern style. The entire stove is centered in the frame, clearly showing it, the marble backsplash, and the range hood above. The lighting is bright and even, keeping the entire cooking area in sharp focus.
```

The second prompt provides:
- **Composition guidance** ("entire stove is centered")
- **Context inclusion** ("marble backsplash, range hood above")
- **Lighting specifics** ("bright and even")
- **Focus distribution** ("entire cooking area in sharp focus")

---

## Examples from Working Prompts

All examples are taken directly from the working prompts documented in [CAMERA_PROMPTING_GUIDE.md](CAMERA_PROMPTING_GUIDE.md):

### Example 1: Full Shot - Compositional Framing
**Custom Detail:**
```
The entire stove is centered in the frame, clearly showing it, the marble backsplash, and the range hood above
```

**Why It Works:**
- Specifies exact subject placement ("centered in the frame")
- Lists contextual elements to include ("marble backsplash", "range hood")
- Ensures comprehensive view ("entire stove")

---

### Example 2: Extreme Macro - Focus Control
**Custom Detail:**
```
focusing on the intricate details of a single burner and the cast-iron grate
```

**Why It Works:**
- Specifies what to isolate ("single burner")
- Emphasizes detail level ("intricate details")
- Names specific components ("cast-iron grate")

---

### Example 3: Extreme Macro - Bokeh Description
**Custom Detail:**
```
The vantage point is inches away, creating an extremely shallow depth of field where only the front edge of the burner is in sharp focus, and the rest of the stove and kitchen dissolves into a soft, blurred bokeh
```

**Why It Works:**
- Reinforces proximity ("inches away")
- Describes focus falloff precisely ("only the front edge")
- Uses evocative language for blur ("dissolves into soft, blurred bokeh")

---

### Example 4: Full Shot - Lighting Details
**Custom Detail:**
```
The lighting is bright and even, keeping the entire area in sharp focus
```

**Why It Works:**
- Specifies lighting quality ("bright and even")
- Connects lighting to focus ("keeping entire area in sharp focus")

---

### Example 5: Watch Detail - Component Naming
**Custom Detail:**
```
showing dial and hands clearly
```

**Why It Works:**
- Names specific components to emphasize
- Ensures clarity ("clearly")

---

## How Users Should Use Custom Details

### 1. Start with the 5 Ingredients (Foundation)
Set these parameters in the node:
- **Subject:** "the green stove"
- **Shot Type:** "Full Shot (FS)"
- **Angle:** "Eye Level"
- **Depth of Field:** "Deep"
- **Style:** "Clean/Modern"

### 2. Add Custom Details (Enhancement)
In the `custom_details` field, add compositional specifics:
```
The entire stove is centered in the frame, clearly showing it, the marble backsplash, and the range hood above. The lighting is bright and even, keeping the entire cooking area in sharp focus.
```

### 3. Result
The node generates a complete prompt combining both:
```
An eye-level full shot of the green stove, taken from a vantage point four and a half meters away, with deep depth of field keeping everything in focus, in clean and modern style. The entire stove is centered in the frame, clearly showing it, the marble backsplash, and the range hood above. The lighting is bright and even, keeping the entire cooking area in sharp focus.
```

---

## Categories of Custom Details

The tooltip examples cover 6 essential categories:

### 1. Compositional Framing
```
"The entire stove is centered in the frame, clearly showing it, the marble backsplash, and the range hood above"
```
**Use for:** Subject placement, contextual elements, framing guidance

---

### 2. Detail Isolation
```
"focusing on the intricate details of a single burner and the cast-iron grate"
```
**Use for:** Macro shots, close-ups, component emphasis

---

### 3. Component Naming
```
"showing dial and hands clearly"
```
**Use for:** Specific parts to emphasize, clarity requirements

---

### 4. Vantage Point Reinforcement
```
"The vantage point is inches away, creating an extremely shallow depth of field"
```
**Use for:** Extreme close-ups, macro, proximity emphasis

---

### 5. Bokeh Description
```
"dissolves into a soft, blurred bokeh"
```
**Use for:** Shallow DOF shots, background treatment, artistic blur

---

### 6. Lighting Specifics
```
"The lighting is bright and even, keeping the entire area in sharp focus"
```
**Use for:** Lighting quality, brightness, mood, focus relationship

---

## Integration with CAMERA_PROMPTING_GUIDE.md

The tooltip examples are taken directly from the 15 annotated working examples in the comprehensive camera prompting guide. Users can reference [CAMERA_PROMPTING_GUIDE.md](CAMERA_PROMPTING_GUIDE.md) for:

- Full context of each example
- Ingredient breakdowns
- Before/after comparisons
- Advanced tips for combining custom details

---

## Testing Status

✅ **Python Syntax:** VALID - File compiles successfully
✅ **Tooltip Format:** VALID - Multi-line tooltip with bullet points
✅ **Examples:** VALID - All taken from working prompts
✅ **Integration:** READY - Node will display enhanced tooltip in ComfyUI

---

## User Benefits

### 1. **Clear Guidance**
Users now see concrete examples of what to add beyond the 5 ingredients, reducing guesswork.

### 2. **Working Examples**
All tooltip examples are from validated working prompts, ensuring they produce good results.

### 3. **Category Coverage**
Examples span 6 essential categories (framing, detail, components, vantage, bokeh, lighting).

### 4. **Progressive Learning**
Users can start with the 5 ingredients (simple), then enhance with custom details (advanced).

### 5. **Consistent Pattern**
Matches the teaching approach in CAMERA_PROMPTING_GUIDE.md for unified learning experience.

---

## Version Info

- **Feature Version**: v2.4.0 (pending)
- **Based On**: Nanobanan's 5-ingredient framework
- **Enhanced With**: Working examples from CAMERA_PROMPTING_GUIDE.md
- **Compatibility**: All shot types (ECU to EWS)

---

## Files Modified

1. **nodes/camera/cinematography_prompt_builder.py**
   - Lines 274-284: Enhanced `custom_details` tooltip with 6 examples covering essential categories

**Total changes:** ~10 lines modified

---

## Next Steps for User

1. **Load Node in ComfyUI** - Verify enhanced tooltip displays correctly
2. **Test with Examples** - Try the tooltip examples with different shot types
3. **Reference Guide** - Use [CAMERA_PROMPTING_GUIDE.md](CAMERA_PROMPTING_GUIDE.md) for full context
4. **Experiment** - Create custom details combining multiple categories

---

**Implementation Date:** 2025-01-06
**Author:** Amir Ferdos (ArchAi3d)
**Based On:** Working examples from CAMERA_PROMPTING_GUIDE.md
