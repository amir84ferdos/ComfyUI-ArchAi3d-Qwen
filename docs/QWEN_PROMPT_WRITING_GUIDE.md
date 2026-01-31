# Image Edit AI Prompt Guide

**Author:** ArchAi3d
**Last Updated:** 2026-01-26

---

## Quick Reference

| Model | Prompt Style | Best For |
|-------|-------------|----------|
| **Qwen-Image-Edit** | Structured commands | Precise editing, object placement, compositing |
| **FLUX.2 Klein** | Narrative prose | Creative generation, lighting, atmosphere |

---

## 1. Universal Principles

### Be Specific, Not Vague

| Bad | Good |
|-----|------|
| "add objects" | "add a red bicycle near the left wall" |
| "good lighting" | "soft diffused natural light from the left" |
| "make it better" | "increase contrast, add warm orange tones" |

### Word Order Matters

Put the most important elements **first**. Models pay more attention to the beginning of prompts.

```
Subject → Setting → Details → Lighting → Atmosphere
```

### Every Word Should Serve a Purpose

Avoid filler. If a word doesn't change the output, remove it.

---

## 2. Prompt Length Guide

| Length | Use Case | Example |
|--------|----------|---------|
| **10-30 words** | Quick concepts | "Portrait in soft window light, natural skin tones, warm atmosphere" |
| **30-80 words** | Standard work | Full scene descriptions with lighting details |
| **80-300 words** | Complex scenes | Detailed editorial or product photography |

---

## 3. Lighting (High Impact)

Lighting has the **greatest impact** on output quality. Always specify:

| Element | Options |
|---------|---------|
| **Source** | natural, artificial, ambient, studio |
| **Quality** | soft, harsh, diffused, direct |
| **Direction** | side, back, overhead, fill, rim |
| **Temperature** | warm, cool, golden, blue |

**Example:**
```
soft diffused natural light filtering through sheer curtains, warm golden tones
```

---

## 4. Qwen-Image-Edit Specifics

### Position Guide Workflow

When using numbered rectangles as position guides:

**System Prompt:**
```
You are an expert image compositor. Image 1 is the scene to edit. Image 2 is the position guide with numbered red rectangles. Add objects at each rectangle's position. Red rectangles are invisible guides - do not draw them.
```

**Main Prompt Template:**
```
remove all red rectangles and numbers from the image. add objects according to this mapping: rectangle 1 = [OBJECT], rectangle 2 = [OBJECT], place each object inside its numbered rectangle, then remove all red rectangles and numbers, keep everything else identical
```

### Key Rules

1. **Spaces around equals:** `rectangle 1 = bicycle` (not `rectangle 1=bicycle`)
2. **Dual removal:** Put "remove rectangles" at BOTH start and end
3. **Explicit preservation:** Always add "keep everything else identical"

### Object Mapping Format

```
rectangle 1 = red bicycle leaning against wall
rectangle 2 = man sitting on chair
rectangle 3 = golden retriever lying on floor
```

---

## 5. FLUX.2 Klein Specifics

### Write Like a Story

FLUX responds to **narrative descriptions**, not keyword lists.

**Bad (keyword style):**
```
woman, portrait, soft light, warm tones, professional
```

**Good (narrative style):**
```
A woman in soft window light with natural skin tones and gentle shadows, warm golden hour atmosphere creating a professional portrait feel
```

### Model Variants

| Variant | Speed | License | Use Case |
|---------|-------|---------|----------|
| **4B Klein** | Sub-second | Apache 2.0 | Local deployment, quick iterations |
| **9B Klein** | Standard | Non-Commercial | Best quality, complex prompts |

---

## 6. Common Patterns

### Object Addition
```
add [object] [position] in the image
```
Example: "add a floor lamp next to the sofa on the left side"

### Object Removal
```
remove [object] from the image
```
Example: "remove the watermark from the bottom right corner"

### Material Change
```
change the [object] material to [new material], keep everything else identical
```
Example: "change the countertop material to white marble with gray veining, keep everything else identical"

### Lighting Change
```
change the lighting to [description], keep all furniture identical
```
Example: "change the lighting to golden hour sunset with warm orange light, keep all furniture identical"

### Camera/Perspective
```
Rotate the angle to [angle description], keep the subject's identity identical
```
Example: "Rotate the angle to a low-angle shot looking up at the subject, keep the subject's id, clothes, and pose identical"

---

## 7. Troubleshooting

| Problem | Solution |
|---------|----------|
| Rectangles/guides appear in output | Use dual removal (start AND end of prompt) |
| Objects placed incorrectly | Add "place each object inside its numbered rectangle area" |
| Background changed unexpectedly | Add "keep everything else in the image identical" |
| Identity/appearance changed | Add "keep the subject's id, clothes, facial features identical" |
| Colors shifted | Add "maintain all original colors unchanged" |

---

## 8. Quick Templates

### Position Guide (Copy & Paste)

```
remove all red rectangles and numbers from the image. using the second image as position guide, add objects: rectangle 1 = [OBJECT_1], rectangle 2 = [OBJECT_2], place each inside its rectangle, then remove all red rectangles and numbers, keep everything else identical
```

### Product Photography

```
[product] on [surface], soft diffused studio lighting from the left, clean white background, professional product photography, sharp focus
```

### Interior Scene

```
[room type] with [key furniture], [lighting description] streaming through [light source], [atmosphere/mood], [style] interior design
```

### Portrait Edit

```
[action/change], keep the subject's id, clothes, facial features, pose, and hairstyle identical
```

---

## 9. Checklist Before Submitting

- [ ] Most important elements at the beginning
- [ ] Specific object descriptions (not vague)
- [ ] Lighting described (source, quality, direction)
- [ ] Preservation clause included if needed
- [ ] Removal instructions (for position guide workflows)
- [ ] No filler words or redundant phrases

---

## 10. Examples

### Interior Design - Material Variation
```
modern kitchen with white cabinets, change the countertop material to black granite with subtle sparkle, soft natural light from the window, keep everything else identical
```

### Product - 360 View
```
wireless headphones on white background, orbit around the product showing all sides, maintaining distance, keeping camera level, studio lighting
```

### Portrait - Angle Change
```
Rotate the angle to a low-angle shot with camera below the subject looking up, keep the subject's id, clothes, facial features, pose, and hairstyle identical
```

### Scene Composite - Position Guide
```
remove all red rectangles and numbers. add objects: rectangle 1 = brown leather sofa against wall, rectangle 2 = glass coffee table in center, rectangle 3 = potted plant in corner, place each inside its rectangle, remove all guides, keep everything else identical
```

---

**End of Guide**
