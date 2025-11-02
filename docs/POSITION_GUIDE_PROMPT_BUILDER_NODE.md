# Position Guide Prompt Builder Node

**Node Name:** `ArchAi3D_Position_Guide_Prompt_Builder`
**Display Name:** 📝 Position Guide Prompt Builder
**Category:** `ArchAi3d/Utils`
**Version:** 1.0.1
**Created:** 2025-10-17
**Updated:** 2025-10-17 - Removed parentheses (user feedback)

---

## Overview

The **Position Guide Prompt Builder** node automatically formats position guide prompts for Qwen image editing workflows. It takes user-defined object descriptions (separated by `/`) and converts them into properly formatted prompts with numbered rectangle mappings.

This node is part of the **Position Guide Workflow System** and works seamlessly with the [Mask to Position Guide](MASK_TO_POSITION_GUIDE_NODE.md) node.

---

## Why This Node?

### Problem It Solves:
When using Qwen's position guide workflow, you need to:
1. Create numbered rectangles on a guide image
2. Write a complex prompt mapping each rectangle number to an object description
3. Maintain exact formatting for Qwen to understand the mappings

**Manual prompt writing is:**
- Time-consuming (3-5 minutes per prompt)
- Error-prone (typos, formatting mistakes)
- Inconsistent (hard to remember exact syntax)

### Solution:
This node **automates prompt generation**:
- Type descriptions separated by `/`
- Choose a template preset
- Get perfectly formatted prompt instantly
- Save 3-5 minutes per workflow
- Consistent formatting every time

---

## Node Schema

### Inputs:

| Input | Type | Required | Description |
|-------|------|----------|-------------|
| **object_descriptions** | STRING (multiline) | Yes | Object descriptions separated by `/`<br>Example: `"boy seats on sofa / man seats on sofa / woman seats on sofa"` |
| **prompt_template** | DROPDOWN | Yes | Template preset:<br>- `standard` (recommended)<br>- `no_removal`<br>- `minimal`<br>- `custom` |
| **custom_template** | STRING (multiline) | No | Custom template (only used when `prompt_template = "custom"`)<br>Use `{MAPPINGS}` and `{ADDITIONAL}` placeholders |
| **additional_instructions** | STRING | No | Extra details to append<br>Example: `"man is close to boy"` or `"warm lighting"` |

### Outputs:

| Output | Type | Description |
|--------|------|-------------|
| **formatted_prompt** | STRING | Complete formatted prompt ready for Qwen |

---

## Template Presets

### 1. **standard** (Recommended)

**Full prompt with removal + preservation commands**

```
using the second image as a position reference guide, the red rectangles are numbered, add objects to the first image according to this mapping: {MAPPINGS}, place each object inside its numbered rectangle area, then remove all red rectangles and numbers from the image, keep everything else in the first image identical{ADDITIONAL}
```

**When to use:**
- Most common use case
- Want clean output without visible rectangles
- Need layout preservation
- Production workflows

**Example output:**
```
using the second image as a position reference guide, the red rectangles are numbered, add objects to the first image according to this mapping: (rectangle 1= boy seats on sofa), (rectangle 2= man seats on sofa), (rectangle 3= woman seats on sofa), place each object inside its numbered rectangle area, then remove all red rectangles and numbers from the image, keep everything else in the first image identical, man is close to boy
```

---

### 2. **no_removal**

**Keep rectangles visible in output**

```
using the second image as a position reference guide, the red rectangles are numbered, add objects to the first image according to this mapping: {MAPPINGS}, place each object inside its numbered rectangle area, keep everything else in the first image identical{ADDITIONAL}
```

**When to use:**
- Debugging positioning issues
- Verifying rectangle placement
- Testing workflows
- Documentation/tutorials

**Example output:**
```
using the second image as a position reference guide, the red rectangles are numbered, add objects to the first image according to this mapping: (rectangle 1= flower vase), (rectangle 2= table lamp), place each object inside its numbered rectangle area, keep everything else in the first image identical
```

---

### 3. **minimal**

**Just object mappings, no extra commands**

```
add objects to the first image: {MAPPINGS}{ADDITIONAL}
```

**When to use:**
- Quick tests
- Experimental prompts
- Combining with other prompt builders
- Custom workflows

**Example output:**
```
add objects to the first image: (rectangle 1= cat), (rectangle 2= dog), (rectangle 3= bird), sunny day lighting
```

---

### 4. **custom**

**User-defined template with placeholders**

**Placeholders:**
- `{MAPPINGS}` - Replaced with: `(rectangle 1= desc1), (rectangle 2= desc2), ...`
- `{ADDITIONAL}` - Replaced with: `, additional_instructions` (with leading comma if present)

**When to use:**
- Specialized workflows
- Experimenting with new prompt patterns
- Advanced users
- Custom Qwen versions

**Example custom template:**
```
Place these items using the guide: {MAPPINGS}. Style: photorealistic. {ADDITIONAL}
```

**Example output:**
```
Place these items using the guide: (rectangle 1= modern chair), (rectangle 2= floor lamp). Style: photorealistic. warm lighting, high contrast
```

---

## Mapping Format

All templates use the same mapping format:

```
rectangle 1= description, rectangle 2= description, rectangle 3= description, ...
```

**Format rules:**
- No parentheses (user-validated for better Qwen performance)
- Space after "rectangle": `rectangle 1` not `rectangle1`
- Equals sign with spaces: `1= description` not `1=description`
- Comma separation: `, ` between mappings
- Sequential numbering: 1, 2, 3, 4, ...

---

## Usage Examples

### Example 1: Standard Living Room Scene

**Inputs:**
```
object_descriptions: "boy seats on sofa / man seats on sofa / woman seats on sofa"
prompt_template: "standard"
additional_instructions: "man is close to boy"
```

**Output:**
```
using the second image as a position reference guide, the red rectangles are numbered, add objects to the first image according to this mapping: rectangle 1= boy seats on sofa, rectangle 2= man seats on sofa, rectangle 3= woman seats on sofa, place each object inside its numbered rectangle area, then remove all red rectangles and numbers from the image, keep everything else in the first image identical, man is close to boy
```

---

### Example 2: Product Photography

**Inputs:**
```
object_descriptions: "red coffee mug / succulent plant / open book"
prompt_template: "standard"
additional_instructions: "professional product photography, soft shadows, white background"
```

**Output:**
```
using the second image as a position reference guide, the red rectangles are numbered, add objects to the first image according to this mapping: rectangle 1= red coffee mug, rectangle 2= succulent plant, rectangle 3= open book, place each object inside its numbered rectangle area, then remove all red rectangles and numbers from the image, keep everything else in the first image identical, professional product photography, soft shadows, white background
```

---

### Example 3: Debugging (No Removal)

**Inputs:**
```
object_descriptions: "ceiling light fixture / wall painting / floor rug"
prompt_template: "no_removal"
additional_instructions: ""
```

**Output:**
```
using the second image as a position reference guide, the red rectangles are numbered, add objects to the first image according to this mapping: rectangle 1= ceiling light fixture, rectangle 2= wall painting, rectangle 3= floor rug, place each object inside its numbered rectangle area, keep everything else in the first image identical
```

---

### Example 4: Minimal Quick Test

**Inputs:**
```
object_descriptions: "tree / bench / fountain"
prompt_template: "minimal"
additional_instructions: "sunset lighting"
```

**Output:**
```
add objects to the first image: rectangle 1= tree, rectangle 2= bench, rectangle 3= fountain, sunset lighting
```

---

### Example 5: Custom Template

**Inputs:**
```
object_descriptions: "modern chair / minimalist desk / standing lamp"
prompt_template: "custom"
custom_template: "Add furniture to room using numbered guide: {MAPPINGS}. Interior design style: Scandinavian minimalism. {ADDITIONAL}"
additional_instructions: "natural wood tones, soft white palette"
```

**Output:**
```
Add furniture to room using numbered guide: rectangle 1= modern chair, rectangle 2= minimalist desk, rectangle 3= standing lamp. Interior design style: Scandinavian minimalism. natural wood tones, soft white palette
```

---

## Complete Workflow Integration

### Full Position Guide Workflow:

```
Step 1: Create Mask
↓
[User draws mask with white regions marking desired object positions]

Step 2: Generate Position Guide
↓
[Mask to Position Guide Node]
- Input: Mask
- Output: guide_image (numbered rectangles)

Step 3: Build Prompt
↓
[Position Guide Prompt Builder Node] ← THIS NODE
- Input: "boy seats / man seats / woman seats"
- Output: formatted_prompt

Step 4: Encode & Generate
↓
[Qwen Encoder]
- Image 1: Original scene
- Image 2: guide_image (from Step 2)
- Prompt: formatted_prompt (from Step 3)
- Output: Edited image with objects placed
```

---

## ComfyUI Workflow Example

```
┌─────────────────┐
│  Load Image     │ (Original scene)
└────────┬────────┘
         │
         ├──────────────────────────┐
         │                          │
         │                    ┌─────▼──────────┐
         │                    │ Create Mask    │ (Draw positions)
         │                    └─────┬──────────┘
         │                          │
         │                    ┌─────▼────────────────────┐
         │                    │ Mask to Position Guide   │
         │                    └─────┬────────────────────┘
         │                          │
         │                      guide_image
         │                          │
    Image 1                         │
         │                       Image 2
         │                          │
         └──────────┬───────────────┘
                    │
              ┌─────▼──────────────────────────┐
              │ Position Guide Prompt Builder  │ ← THIS NODE
              └─────┬──────────────────────────┘
                    │
              formatted_prompt
                    │
              ┌─────▼──────────────┐
              │  Qwen Encoder      │
              └─────┬──────────────┘
                    │
              ┌─────▼──────────────┐
              │  Output Image      │
              └────────────────────┘
```

---

## Debug Output

When the node executes, it prints detailed debug information to the console:

```
================================================================================
📝 Position Guide Prompt Builder - v1.0.1
================================================================================
Template: standard
Object count: 3

Object descriptions:
  Rectangle 1: boy seats on sofa
  Rectangle 2: man seats on sofa
  Rectangle 3: woman seats on sofa

Additional instructions: man is close to boy

📋 Generated prompt:
using the second image as a position reference guide, the red rectangles are numbered, add objects to the first image according to this mapping: rectangle 1= boy seats on sofa, rectangle 2= man seats on sofa, rectangle 3= woman seats on sofa, place each object inside its numbered rectangle area, then remove all red rectangles and numbers from the image, keep everything else in the first image identical, man is close to boy
================================================================================
```

---

## Tips & Best Practices

### 1. **Description Clarity**
- ✅ Good: `"boy seats on sofa"`
- ✅ Good: `"realistic red coffee mug on wooden table"`
- ❌ Avoid: `"boy"` (too vague)
- ❌ Avoid: `"rectangle with boy"` (don't repeat "rectangle")

### 2. **Separator Usage**
- Use `/` as separator: `"cat / dog / bird"`
- Spaces around `/` are optional: `"cat/dog/bird"` works too
- Node auto-trims whitespace

### 3. **Additional Instructions**
- Use for spatial relationships: `"man is close to boy"`
- Use for style: `"photorealistic, warm lighting"`
- Use for atmosphere: `"cozy evening ambiance"`
- Don't repeat object descriptions already in mappings

### 4. **Template Selection**
- Start with `"standard"` for most workflows
- Use `"no_removal"` when debugging positioning
- Use `"minimal"` for quick tests
- Use `"custom"` when you're comfortable with the workflow

### 5. **Empty Descriptions**
- If `object_descriptions` is empty, output is empty string
- Node skips empty descriptions from splitting: `"cat / / dog"` → 2 rectangles (not 3)

---

## Troubleshooting

### Issue 1: Prompt Not Working as Expected

**Symptoms:**
- Objects not appearing in correct positions
- Qwen ignoring rectangle mappings

**Solutions:**
1. Check guide image has visible numbered rectangles
2. Verify object count matches rectangle count in guide image
3. Try `"no_removal"` template to see if removal command is interfering
4. Check console debug output for formatting issues

---

### Issue 2: Too Many/Few Objects

**Symptoms:**
- More descriptions than rectangles in guide image
- Fewer descriptions than rectangles

**Solutions:**
1. Count rectangles in guide image output
2. Count `/` separators in object_descriptions (should be N-1 for N objects)
3. Use debug output to verify object count
4. Note: No automatic validation in this version (manual check required)

---

### Issue 3: Custom Template Not Working

**Symptoms:**
- `{MAPPINGS}` or `{ADDITIONAL}` appearing literally in output

**Solutions:**
1. Check spelling: Must be exactly `{MAPPINGS}` and `{ADDITIONAL}`
2. Ensure template is provided when `prompt_template = "custom"`
3. Check console debug output for generated prompt
4. Verify placeholders are in correct format (curly braces required)

---

### Issue 4: Parentheses Appearing in Output

**Note:** This was fixed in v1.0.1 based on user feedback.

**Old format (v1.0.0):** `(rectangle 1= description), (rectangle 2= description)`
**New format (v1.0.1):** `rectangle 1= description, rectangle 2= description`

If you see parentheses in your output, restart ComfyUI to load v1.0.1.

---

## Version History

### v1.0.1 (2025-10-17)
- **Breaking change:** Removed parentheses from mapping format (user feedback)
- New format: `rectangle 1= description, rectangle 2= description`
- Old format: `(rectangle 1= description), (rectangle 2= description)`
- Reason: Better Qwen performance without parentheses
- All documentation updated

### v1.0.0 (2025-10-17)
- Initial release
- 4 template presets: standard, no_removal, minimal, custom
- `/` separator support with auto-trim
- Custom template with `{MAPPINGS}` and `{ADDITIONAL}` placeholders
- Optional additional instructions field
- Debug console output
- Original formatting: `(rectangle 1= description)` (deprecated)

---

## Related Documentation

- [Mask to Position Guide Node](MASK_TO_POSITION_GUIDE_NODE.md) - Companion node for generating guide images
- [Position Guide Workflow Discovery](POSITION_GUIDE_WORKFLOW_DISCOVERY.md) - Complete workflow documentation
- [Position Guide Quick Reference](POSITION_GUIDE_QUICK_REFERENCE.md) - Fast lookup guide
- [Qwen Prompt Guide](QWEN_PROMPT_GUIDE.md) - General Qwen prompting patterns

---

## Technical Details

### File Location:
```
E:\Comfy\Qwen\ComfyUI-Easy-Install\ComfyUI\custom_nodes\ComfyUI-ArchAi3d-Qwen\
└── nodes/
    └── utils/
        └── archai3d_position_guide_prompt_builder.py
```

### Dependencies:
- `comfy_io` (for node schema)
- Python standard library only

### Node Registration:
- **Class:** `ArchAi3D_Position_Guide_Prompt_Builder`
- **Display Name:** `📝 Position Guide Prompt Builder`
- **Category:** `ArchAi3d/Utils`

---

## Support

For issues or questions:
- Check console debug output first
- Review related documentation
- Verify input format (especially `/` separator)
- Test with `"minimal"` template for simplest case

---

**Author:** Amir Ferdos (ArchAi3d)
**Email:** Amir84ferdos@gmail.com
**LinkedIn:** https://www.linkedin.com/in/archai3d/
**License:** MIT

---

*Last Updated: 2025-10-17*
