# Qwen Prompt Writing Guide - Based on Real Experience
**Author:** ArchAi3d
**Last Updated:** 2025-10-18
**Model:** Qwen-Image-Edit (Qwen2.5-VL based)

---

## Table of Contents
1. [Introduction](#introduction)
2. [Core Principles](#core-principles)
3. [Punctuation Usage](#punctuation-usage)
4. [Position Guide Workflow](#position-guide-workflow)
5. [System Prompts](#system-prompts)
6. [Removal Instructions](#removal-instructions)
7. [Proven Working Formats](#proven-working-formats)
8. [Common Patterns](#common-patterns)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

---

## Introduction

This guide is based on extensive real-world testing with Qwen-Image-Edit model. These are **validated patterns** that actually work in production, not theoretical suggestions.

**Key Insight:** Qwen is sensitive to:
- Prompt structure and punctuation
- Clear, explicit instructions
- Consistent formatting
- Removal command positioning

---

## Core Principles

### 1. **Be Explicit and Clear**
❌ **Bad:** "add objects"
✅ **Good:** "add objects to the first image according to this mapping: rectangle 1 = bicycle, rectangle 2 = dog"

### 2. **Use Consistent Formatting**
Stick to one format throughout your prompt. Don't mix styles.

### 3. **Separation is Key**
Use proper punctuation to separate:
- Major instructions (use periods `.`)
- Related clauses (use commas `,`)
- List items (use commas `,`)
- Mappings (use equals `=` with spaces)

### 4. **Removal Commands Matter**
Position guide rectangles/numbers **will appear in output** unless you explicitly tell Qwen to remove them (sometimes multiple times).

---

## Punctuation Usage

### **Period (.) - Sentence Separator**

**Purpose:** Separate distinct, independent instructions

**Examples:**
```
remove all red rectangles and numbers from the image. using the second image as a position reference guide, add objects to the first image
```

**When to use:**
- End of major commands
- Between removal instruction and main instruction
- Between independent tasks

**Effect:** Creates clear mental breaks for the model

---

### **Comma (,) - Flow Connector**

**Purpose:** Connect related clauses and separate list items

**Examples:**
```
rectangle 1 = bicycle, rectangle 2 = dog, rectangle 3 = man
```
```
add objects to the image, place each inside its rectangle, then remove guides
```

**When to use:**
- Separating mapping items
- Connecting flowing instructions
- Adding qualifying details

**Effect:** Creates natural flow and groups related concepts

---

### **Colon (:) - List Introduction**

**Purpose:** Signal "here comes the important data"

**Examples:**
```
add objects according to this mapping: rectangle 1 = bicycle, rectangle 2 = dog
```

**When to use:**
- Before mapping lists
- Before enumerated instructions
- After "according to this" or "following these"

**Effect:** Draws model attention to what follows

---

### **Equals (=) - Position Mapping**

**Purpose:** Create clear 1:1 relationships

**Format:** **ALWAYS use spaces around equals!**
- ✅ `rectangle 1 = bicycle` (CORRECT)
- ❌ `rectangle 1=bicycle` (WRONG - less reliable)

**Examples:**
```
rectangle 1 = bicycle near the wall
rectangle 2 = new man seats on chair
```

**Critical:** Spacing affects model parsing. Always use spaces.

---

### **Parentheses ( ) - Grouped Instructions**

**Purpose:** Wrap individual instructions (alternative format)

**Format:**
```
(rectangle 1 = add a bicycle near the wall.)
(rectangle 2 = add a new man seats on chair.)
(rectangle 3 = add many dogs are playing in near and far.)

(Maintain all original elements in Image 1 unchanged.)
```

**When to use:**
- Alternative to comma-separated format
- When you want each instruction visually separated
- Proven to work well with Qwen

**Effect:** Each line is a discrete, contained instruction

---

## Position Guide Workflow

### **What is Position Guide?**
A two-image workflow where:
- **Image 1:** Scene to edit (original photo)
- **Image 2:** Position guide with numbered red rectangles showing where to add objects

### **The Challenge:**
Qwen tends to draw the red rectangles and numbers in the output unless explicitly instructed otherwise.

---

## System Prompts

### **Validated Working System Prompt (Recommended)**

```
You are an expert image compositor. You receive two inputs: Image 1 (scene to edit) and Image 2 (numbered position guide with red rectangles). Each rectangle in Image 2 has a number. The prompt provides a mapping of which object goes in which numbered rectangle. Read the number in each rectangle, find its mapping in the prompt, then add that specific object to Image 1 at the rectangle's position. Red rectangles and numbers are invisible reference guides only - do not draw them. Maintain all original elements in Image 1 unchanged.
```

**Why it works:**
- ✅ Explains the two-image input clearly
- ✅ Describes the numbered rectangle mapping system
- ✅ Emphasizes "invisible reference guides only - do not draw them"
- ✅ States preservation requirement

---

### **Three-Stage System Prompt (Alternative)**

```
You are an expert image compositor with cleanup capability. You receive two inputs: Image 1 (main scene to edit) and Image 2 (numbered position guide with red rectangles). Your task has three stages: Stage 1 - Read: Each red rectangle in Image 2 has a number. The prompt maps numbers to objects (example: rectangle 1 = flower). Stage 2 - Add: Add each mapped object to Image 1 at its numbered rectangle's position. Stage 3 - Clean: Remove all red rectangles and all numbers from the final image. These are temporary reference guides and must not appear in output. Use the remove command to clean them. Maintain all original elements in Image 1 unchanged. Only the mapped objects should be added, and all guide markers should be removed.
```

**When to use:**
- If Qwen is forgetting the cleanup step
- When you need very explicit stage-by-stage instructions
- For complex multi-step operations

---

## Removal Instructions

### **The Problem:**
Red rectangles and numbers from position guides **frequently appear in output** if not explicitly removed.

### **Solution Patterns (Tested & Validated):**

#### **Pattern 1: Dual Removal (Recommended)**
Place removal command at **both start and end**:

```
remove all red rectangles and numbers from the image. using the second image as a position reference guide, the red rectangles are numbered, add objects to the first image according to this mapping: rectangle 1 = bicycle near the wall, rectangle 2 = new man seats on chair, rectangle 3 = dog, place each object inside its numbered rectangle area, then remove all red rectangles and numbers from the image, keep everything else in the first image identical
```

**Why it works:**
- Sets expectation upfront
- Reminds at the end
- Reinforces the removal requirement

**Effectiveness:** ⭐⭐⭐⭐⭐ (Most reliable)

---

#### **Pattern 2: Split Removal Commands**
Separate rectangles and numbers:

```
using the second image as a position reference guide, the red rectangles are numbered, add objects to the first image according to this mapping: rectangle 1 = bicycle, rectangle 2 = dog, rectangle 3 = man, place each object inside its numbered rectangle area, remove all red rectangles from the image, remove all numbers from the image, keep everything else in the first image identical
```

**Why it works:**
- Two separate commands are clearer than one combined
- Targets rectangles and numbers independently

**Effectiveness:** ⭐⭐⭐⭐ (Very reliable)

---

#### **Pattern 3: Ultra-Explicit with IMPORTANT**
Use emphasis for critical removal:

```
using the second image as a position reference guide, the red rectangles are numbered, add objects to the first image according to this mapping: rectangle 1 = bicycle, rectangle 2 = dog, place each object inside its numbered rectangle area, IMPORTANT: remove all red rectangles from the image, IMPORTANT: remove all numbers from the image, the rectangles and numbers are temporary guides only and must not be drawn or visible, keep everything else in the first image identical
```

**Why it works:**
- IMPORTANT prefix draws strong attention
- Explicit statement that guides "must not be drawn or visible"
- Triple reinforcement (IMPORTANT + must not be drawn + temporary guides)

**Effectiveness:** ⭐⭐⭐⭐⭐ (Use if other patterns fail)

---

#### **Pattern 4: Combined Removal (Standard)**
Single removal command at end:

```
using the second image as a position reference guide, the red rectangles are numbered, add objects to the first image according to this mapping: rectangle 1 = bicycle, rectangle 2 = dog, place each object inside its numbered rectangle area, then remove all red rectangles and numbers from the image, keep everything else in the first image identical
```

**Effectiveness:** ⭐⭐⭐ (Works, but less reliable than dual removal)

---

### **Removal Instruction Positioning**

**Tested positions:**
1. **Start only:** Moderate effectiveness
2. **End only:** Standard effectiveness
3. **Both start AND end:** **Highest effectiveness** ⭐⭐⭐⭐⭐
4. **System prompt only:** Not sufficient (still need in main prompt)

**Recommendation:** Use dual removal (start + end) for best results.

---

## Proven Working Formats

### **Format 1: Comma-Separated with Dual Removal**

**Main Prompt:**
```
remove all red rectangles and numbers from the image. using the second image as a position reference guide, the red rectangles are numbered, add objects to the first image according to this mapping: rectangle 1 = bicycle near the wall, rectangle 2 = new man seats on chair, rectangle 3 = dog, place each object inside its numbered rectangle area, then remove all red rectangles and numbers from the image, keep everything else in the first image identical
```

**System Prompt:**
```
You are an expert image compositor. You receive two inputs: Image 1 (scene to edit) and Image 2 (numbered position guide with red rectangles). Each rectangle in Image 2 has a number. The prompt provides a mapping of which object goes in which numbered rectangle. Read the number in each rectangle, find its mapping in the prompt, then add that specific object to Image 1 at the rectangle's position. Red rectangles and numbers are invisible reference guides only - do not draw them. Maintain all original elements in Image 1 unchanged.
```

**Status:** ✅ Validated working format
**Use Case:** Standard position guide workflow

---

### **Format 2: Parentheses Format**

**Main Prompt:**
```
(rectangle 1 = add a bicycle near the wall.)
(rectangle 2 = add a new man seats on chair.)
(rectangle 3 = add many dogs are playing in near and far.)

(Maintain all original elements in Image 1 unchanged.)
```

**System Prompt:** (same as Format 1)

**Status:** ✅ Validated working format
**Use Case:** When you prefer visual separation of instructions

---

### **Format 3: Single Image (Rectangles Already on Image)**

**Main Prompt:**
```
remove all red rectangles and numbers from the image. the red rectangles are numbered, add objects to the image according to this mapping: rectangle 1 = bicycle near the wall, rectangle 2 = new man seats on chair, rectangle 3 = dog, place each object inside its numbered rectangle area, then remove all red rectangles and numbers from the image, keep everything else in the image identical
```

**System Prompt:**
```
You are an expert image compositor. The input image has numbered red rectangles drawn on it. Each rectangle has a number. The prompt provides a mapping of which object goes in which numbered rectangle. Read the number in each rectangle, find its mapping in the prompt, then add that specific object to the image at the rectangle's position. Red rectangles and numbers are invisible reference guides only - remove them completely from the final output. Do not draw any rectangles or numbers in the result. Maintain all original elements in the image unchanged.
```

**Status:** ✅ Validated for single-image workflow
**Use Case:** When position guide is drawn directly on the image

---

## Common Patterns

### **Mapping Pattern**
Always use this exact format (with spaces around `=`):

```
rectangle 1 = [object description], rectangle 2 = [object description], rectangle 3 = [object description]
```

**Examples:**
```
rectangle 1 = bicycle near the wall
rectangle 2 = man seats on chair
rectangle 3 = dog sitting
rectangle 4 = flower vase on table
```

---

### **Placement Pattern**
Specify where to place objects:

```
place each object inside its numbered rectangle area
```

**Alternatives:**
```
add each mapped object to Image 1 at the rectangle's position
position each object within its corresponding numbered rectangle
```

---

### **Preservation Pattern**
Tell Qwen what NOT to change:

```
keep everything else in the first image identical
```

**Alternatives:**
```
Maintain all original elements in Image 1 unchanged
preserve all existing elements outside the numbered areas
do not modify background, lighting, or other objects
```

---

### **Context Pattern**
Set up the two-image scenario:

```
using the second image as a position reference guide, the red rectangles are numbered
```

**Alternative (single image):**
```
the red rectangles are numbered
```

---

## Troubleshooting

### **Problem 1: Rectangles/Numbers Appear in Output**

**Solutions (in order of effectiveness):**

1. **Use dual removal** (start + end):
   ```
   remove all red rectangles and numbers from the image. [main prompt]... then remove all red rectangles and numbers from the image
   ```

2. **Split removal commands**:
   ```
   remove all red rectangles from the image, remove all numbers from the image
   ```

3. **Add IMPORTANT prefix**:
   ```
   IMPORTANT: remove all red rectangles from the image, IMPORTANT: remove all numbers from the image
   ```

4. **Emphasize in system prompt**:
   ```
   Red rectangles and numbers are invisible reference guides only - do not draw them.
   ```

5. **Use three-stage system prompt** (explicitly mentions cleanup stage)

---

### **Problem 2: Objects Not Placed Correctly**

**Solutions:**

1. **Be more specific in descriptions**:
   - ❌ "bicycle"
   - ✅ "bicycle near the wall, facing left"

2. **Use "place each object inside its numbered rectangle area"** in prompt

3. **Add positioning details**:
   ```
   rectangle 1 = bicycle near the left wall, leaning against it
   rectangle 2 = man seats on chair in center of room
   ```

4. **Check your position guide image**:
   - Are rectangles clearly visible?
   - Are numbers readable?
   - Is there enough contrast?

---

### **Problem 3: Background/Original Elements Changed**

**Solutions:**

1. **Add explicit preservation instruction**:
   ```
   keep everything else in the first image identical
   ```

2. **Use "preserve_everything" system prompt variant**:
   ```
   CRITICAL: Keep ALL existing elements in Image 1 EXACTLY as they are: furniture, walls, floors, ceilings, lighting, decorations, colors, textures, and spatial relationships. Only add the specified objects in the marked positions.
   ```

3. **Be specific about what to preserve**:
   ```
   maintain original lighting, shadows, background, and all furniture unchanged
   ```

---

### **Problem 4: Objects Look Out of Place**

**Solutions:**

1. **Add style/quality instructions**:
   ```
   add objects with photorealistic quality, matching the lighting and perspective of the original scene
   ```

2. **Specify lighting consistency**:
   ```
   maintain identical lighting, shadows, and perspective when adding objects
   ```

3. **Use quality system prompt**:
   ```
   Maintain photorealistic quality, lighting consistency, and do not modify any existing elements in Image 1 outside the numbered areas.
   ```

---

## Best Practices

### **1. Prompt Structure**

**Recommended order:**
1. Removal instruction (optional, for dual removal)
2. Context setting (two-image explanation)
3. Main instruction (add objects)
4. Mapping list (rectangle 1 = ..., rectangle 2 = ...)
5. Placement instruction
6. Removal instruction (required)
7. Preservation instruction

**Example:**
```
[1. REMOVAL] remove all red rectangles and numbers from the image.
[2. CONTEXT] using the second image as a position reference guide, the red rectangles are numbered,
[3. MAIN] add objects to the first image according to this mapping:
[4. MAPPINGS] rectangle 1 = bicycle, rectangle 2 = dog, rectangle 3 = man,
[5. PLACEMENT] place each object inside its numbered rectangle area,
[6. REMOVAL] then remove all red rectangles and numbers from the image,
[7. PRESERVE] keep everything else in the first image identical
```

---

### **2. Spacing in Mappings**

**Always use spaces around equals sign:**
- ✅ `rectangle 1 = bicycle`
- ❌ `rectangle 1=bicycle`

**Why:** Model parses this more reliably with spaces.

---

### **3. Consistent Terminology**

**Pick one set of terms and stick with it:**

**Option A (Recommended):**
- "Image 1" and "Image 2"
- "red rectangles"
- "numbers"
- "mapping"

**Option B:**
- "first image" and "second image"
- "position guide"
- "numbered rectangles"
- "mapping"

**Don't mix:** Don't call it "Image 1" in one place and "first image" in another.

---

### **4. Period Placement**

**End each mapping description with appropriate punctuation:**

**Comma-separated format:**
```
rectangle 1 = bicycle near the wall, rectangle 2 = dog sitting
```
(No period until end of sentence)

**Parentheses format:**
```
(rectangle 1 = bicycle near the wall.)
(rectangle 2 = dog sitting.)
```
(Period inside each parenthesis)

---

### **5. System Prompt Usage**

**Always use a system prompt for position guide workflows.**

**Minimal acceptable system prompt:**
```
Red rectangles and numbers are invisible reference guides only - do not draw them.
```

**Recommended system prompt:**
```
You are an expert image compositor. You receive two inputs: Image 1 (scene to edit) and Image 2 (numbered position guide with red rectangles). Each rectangle in Image 2 has a number. The prompt provides a mapping of which object goes in which numbered rectangle. Read the number in each rectangle, find its mapping in the prompt, then add that specific object to Image 1 at the rectangle's position. Red rectangles and numbers are invisible reference guides only - do not draw them. Maintain all original elements in Image 1 unchanged.
```

---

### **6. Testing Strategy**

**When testing new prompts:**

1. **Start with validated format** (use Format 1 or Format 2 above)
2. **Change ONE thing at a time**
3. **Test with same images** for consistency
4. **Document what works** (keep a log)
5. **Build on success** (iterate from working prompts)

**Don't:**
- Change multiple things at once
- Use completely new untested formats
- Assume something will work without testing

---

### **7. Object Descriptions**

**Be specific but concise:**

**Good examples:**
```
rectangle 1 = bicycle near the wall
rectangle 2 = man seats on chair
rectangle 3 = dog sitting on floor
rectangle 4 = flower vase on table
rectangle 5 = painting on wall
```

**Too vague:**
```
rectangle 1 = bike
rectangle 2 = person
rectangle 3 = animal
```

**Too verbose:**
```
rectangle 1 = a beautiful red bicycle with chrome handlebars leaning against the left side of the wall near the window with sunlight streaming through creating interesting shadows on the floor
```

**Sweet spot:**
```
rectangle 1 = red bicycle leaning against left wall
```

---

## Qwen-Specific Behaviors (Observed)

### **1. Removal Command Sensitivity**

**Observation:** Qwen often "forgets" to remove guide elements unless explicitly reminded multiple times.

**Implication:** Use dual removal (start + end) for reliability.

---

### **2. Equals Sign Spacing**

**Observation:** `rectangle 1 = bicycle` works more reliably than `rectangle 1=bicycle`

**Implication:** Always use spaces around `=` in mappings.

---

### **3. Punctuation Matters**

**Observation:** Periods create stronger breaks than commas in instruction flow.

**Implication:** Use periods to separate major instructions, commas for related clauses.

---

### **4. System Prompt Importance**

**Observation:** Position guide workflow performs significantly better with proper system prompt.

**Implication:** Always include system prompt for two-image workflows.

---

### **5. Explicit > Implicit**

**Observation:** Qwen responds better to explicit instructions than implied ones.

**Examples:**
- ❌ "clean up the guides" (implicit - what guides?)
- ✅ "remove all red rectangles and numbers from the image" (explicit)

- ❌ "maintain scene" (implicit - what aspects?)
- ✅ "keep everything else in the first image identical" (explicit)

---

## Quick Reference Card

### **Position Guide Workflow - Copy & Paste Template**

**Main Prompt (Dual Removal):**
```
remove all red rectangles and numbers from the image. using the second image as a position reference guide, the red rectangles are numbered, add objects to the first image according to this mapping: rectangle 1 = [OBJECT_1], rectangle 2 = [OBJECT_2], rectangle 3 = [OBJECT_3], place each object inside its numbered rectangle area, then remove all red rectangles and numbers from the image, keep everything else in the first image identical
```

**System Prompt:**
```
You are an expert image compositor. You receive two inputs: Image 1 (scene to edit) and Image 2 (numbered position guide with red rectangles). Each rectangle in Image 2 has a number. The prompt provides a mapping of which object goes in which numbered rectangle. Read the number in each rectangle, find its mapping in the prompt, then add that specific object to Image 1 at the rectangle's position. Red rectangles and numbers are invisible reference guides only - do not draw them. Maintain all original elements in Image 1 unchanged.
```

**Replace:**
- `[OBJECT_1]` with your first object description
- `[OBJECT_2]` with your second object description
- `[OBJECT_3]` with your third object description
- Add more `rectangle N = [OBJECT]` as needed

---

### **Parentheses Format - Copy & Paste Template**

**Main Prompt:**
```
(rectangle 1 = [OBJECT_1].)
(rectangle 2 = [OBJECT_2].)
(rectangle 3 = [OBJECT_3].)

(Maintain all original elements in Image 1 unchanged.)
```

**System Prompt:** (same as above)

---

## Key Discoveries from Real Testing

### **Discovery 1: "add" Prefix is Optional**

**Finding:** Qwen understands object addition from context, "add" prefix not required.

**Both work:**
```
(rectangle 1 = painting on the wall.)              ← Implicit add
(rectangle 2 = add flower on the table.)           ← Explicit add
```

**Recommendation:** Use whichever reads more naturally for your description.

---

### **Discovery 2: Action Verbs Work**

**Finding:** You can include action verbs and modifications in descriptions.

**Examples that work:**
```
(rectangle 3 = a girl seats on the chair, move the cair.)
(rectangle 5 = the mother standing and talking to girl)
```

**Pattern:**
```
(rectangle N = [object/person] [action], [additional instruction].)
```

**Use cases:**
- Moving existing objects
- Describing poses/actions ("standing", "talking")
- Combining add + modify operations

---

### **Discovery 3: Relationship Context**

**Finding:** You can describe relationships between objects.

**Examples:**
```
the mother standing and talking to girl         ← Spatial + action relationship
a boy standing near the wall                    ← Spatial relationship
dog sitting, looking at the man                 ← Action + direction relationship
```

**Effect:** Helps Qwen understand scene composition and object interactions.

---

### **Discovery 4: Color Preservation Enhancement**

**Finding:** Adding "and colors" to preservation instruction improves color fidelity.

**Standard:**
```
(Maintain all original elements in Image 1 unchanged.)
```

**Enhanced (recommended for color-critical work):**
```
(Maintain all other elements and colors in Image 1 unchanged.)
```

**Benefit:** Prevents unwanted color shifts in original scene elements.

---

### **Discovery 5: Scales Well (6+ Objects)**

**Finding:** Parentheses format works reliably with 6+ rectangles.

**Tested:** Successfully validated with 6 objects in single prompt.

**Implication:** No need to split into multiple prompts for complex scenes.

---

## Advanced Techniques

### **1. Progressive Refinement**

Start with simple descriptions, then add details:

**Iteration 1:**
```
rectangle 1 = bicycle
```

**Iteration 2:**
```
rectangle 1 = bicycle near the wall
```

**Iteration 3:**
```
rectangle 1 = red bicycle near the left wall, leaning against it
```

---

### **2. Context-Specific Instructions**

Add instructions relevant to specific objects:

```
rectangle 1 = man seats on chair, facing the camera
rectangle 2 = dog sitting, looking at the man
rectangle 3 = lamp on table, turned on
```

---

### **3. Lighting/Style Consistency**

Add global style instructions after mappings:

```
rectangle 1 = bicycle, rectangle 2 = dog, rectangle 3 = man, place each object inside its numbered rectangle area with photorealistic quality and lighting that matches the original scene, then remove all red rectangles and numbers from the image, keep everything else in the first image identical
```

---

### **4. Combining with Additional Instructions**

Use the additional_instructions field in nodes:

**Mappings:**
```
rectangle 1 = bicycle, rectangle 2 = dog
```

**Additional Instructions:**
```
man is close to boy, warm lighting, photorealistic style
```

**Result:**
```
...rectangle 1 = bicycle, rectangle 2 = dog, place each object inside its numbered rectangle area, then remove all red rectangles and numbers from the image, keep everything else in the first image identical, man is close to boy, warm lighting, photorealistic style
```

---

### **5. Mixed Instructions - Add, Move, Modify (ADVANCED)**

**Discovery (2025-10-18):** You can mix different instruction types in the same prompt!

**Pattern:**
```
(rectangle 1 = [description without action])
(rectangle 2 = [description with implicit add])
(rectangle 3 = [description], [action verb].)
(rectangle 4 = add [object].)
(rectangle 5 = [description with relationship])
```

**Working Example:**
```
(rectangle 1 = painting on the wall.)                           ← No "add" prefix
(rectangle 2 = a boy standing near the wall.)                   ← Implicit add
(rectangle 3 = a girl seats on the chair, move the cair.)       ← Add + Move action
(rectangle 4 = add flower on the table.)                        ← Explicit "add"
(rectangle 5 = the mother standing and talking to girl)         ← Relationship context
(rectangle 6 = add a cat on the ground.)                        ← Explicit "add"
```

**Key Insights:**
- ✅ "add" prefix is **optional** (Qwen understands context)
- ✅ Can include **action verbs** ("move the chair")
- ✅ Can add **relationships** ("talking to girl")
- ✅ Can mix **explicit and implicit** add instructions
- ✅ **Period placement** critical (inside parentheses)

**What works:**
- `painting on the wall` (implicit add)
- `add flower on the table` (explicit add)
- `a girl seats on the chair, move the cair` (add + modify)
- `the mother standing and talking to girl` (add with relationship)

**Status:** ✅ Validated working pattern

---

### **6. Color Preservation Specificity**

**Discovery:** You can be explicit about **what** to preserve.

**Standard preservation:**
```
(Maintain all original elements in Image 1 unchanged.)
```

**Enhanced preservation (more explicit):**
```
(Maintain all other elements and colors in Image 1 unchanged.)
```

**Ultra-specific preservation:**
```
(Maintain all original elements, colors, lighting, shadows, and textures in Image 1 unchanged.)
```

**When to use:**
- Standard: General workflows
- Enhanced: When colors are critical (architectural visualization, product photos)
- Ultra-specific: When quality is paramount

---

## Summary Checklist

**Before submitting your prompt, verify:**

- [ ] Using spaces around equals signs (`rectangle 1 = bicycle`)
- [ ] Removal instruction included (preferably at both start AND end)
- [ ] System prompt included for two-image workflows
- [ ] Mapping format is consistent throughout
- [ ] Period placement is correct for your chosen format
- [ ] Preservation instruction included ("keep everything else identical")
- [ ] Object descriptions are specific but concise
- [ ] Using validated format (Format 1 or Format 2) as base
- [ ] Consistent terminology throughout (Image 1/Image 2 or first/second)

---

## Examples from Real Testing

### **Example 1: Interior Scene with Furniture**

**Main Prompt:**
```
remove all red rectangles and numbers from the image. using the second image as a position reference guide, the red rectangles are numbered, add objects to the first image according to this mapping: rectangle 1 = brown leather sofa against the wall, rectangle 2 = wooden coffee table in center, rectangle 3 = floor lamp in corner, rectangle 4 = large painting on wall, place each object inside its numbered rectangle area, then remove all red rectangles and numbers from the image, keep everything else in the first image identical
```

**Result:** ✅ Success - Objects placed correctly, guides removed

---

### **Example 2: Outdoor Scene with People and Animals**

**Main Prompt:**
```
(rectangle 1 = bicycle near the wall.)
(rectangle 2 = new man seats on chair.)
(rectangle 3 = many dogs are playing in near and far.)

(Maintain all original elements in Image 1 unchanged.)
```

**Result:** ✅ Success - Parentheses format works well

---

### **Example 3: Complex Indoor Scene with 6 Objects (VALIDATED 2025-10-18)**

**Main Prompt:**
```
(rectangle 1 = painting on the wall.)
(rectangle 2 = a boy standing near the wall.)
(rectangle 3 = a girl seats on the chair, move the cair.)
(rectangle 4 = add flower on the table.)
(rectangle 5 = the mother standing and talking to girl)
(rectangle 6 = add a cat on the ground.)

(Maintain all other elements and colors in Image 1 unchanged.)
```

**Result:** ✅ Success - Works with 6+ objects, mixed instructions (add/move)

**Key Observations:**
- ✅ Mixing "add" prefix and no prefix works
- ✅ Can include action verbs ("move the chair", "talking to girl")
- ✅ "Maintain all other elements **and colors**" - explicit color preservation
- ✅ Scales well (6 rectangles tested successfully)
- ✅ Period inside parentheses is critical

---

### **Example 3: Single Image with Guides Already Drawn**

**Main Prompt:**
```
remove all red rectangles and numbers from the image. the red rectangles are numbered, add objects to the image according to this mapping: rectangle 1 = bicycle near the wall, rectangle 2 = man seats on chair, rectangle 3 = dog, place each object inside its numbered rectangle area, then remove all red rectangles and numbers from the image, keep everything else in the image identical
```

**System Prompt:**
```
You are an expert image compositor. The input image has numbered red rectangles drawn on it. Each rectangle has a number. The prompt provides a mapping of which object goes in which numbered rectangle. Read the number in each rectangle, find its mapping in the prompt, then add that specific object to the image at the rectangle's position. Red rectangles and numbers are invisible reference guides only - remove them completely from the final output. Do not draw any rectangles or numbers in the result. Maintain all original elements in the image unchanged.
```

**Result:** ✅ Success - Single-image workflow works

---

## Mask to Position Guide - Numbering Orders

The **ArchAi3D_Mask_To_Position_Guide** node supports 4 directional numbering orders:

### **Visual Guide:**

```
LEFT TO RIGHT (→)
┌─────────────────────────────────┐
│  ┌───┐    ┌───┐    ┌───┐       │
│  │ 1 │    │ 2 │    │ 3 │       │
│  └───┘    └───┘    └───┘       │
└─────────────────────────────────┘

RIGHT TO LEFT (←)
┌─────────────────────────────────┐
│  ┌───┐    ┌───┐    ┌───┐       │
│  │ 3 │    │ 2 │    │ 1 │       │
│  └───┘    └───┘    └───┘       │
└─────────────────────────────────┘

TOP TO BOTTOM (↓)
┌─────────────────────────────────┐
│  ┌───┐                          │
│  │ 1 │                          │
│  └───┘                          │
│  ┌───┐                          │
│  │ 2 │                          │
│  └───┘                          │
│  ┌───┐                          │
│  │ 3 │                          │
│  └───┘                          │
└─────────────────────────────────┘

BOTTOM TO TOP (↑)
┌─────────────────────────────────┐
│  ┌───┐                          │
│  │ 3 │                          │
│  └───┘                          │
│  ┌───┐                          │
│  │ 2 │                          │
│  └───┘                          │
│  ┌───┐                          │
│  │ 1 │                          │
│  └───┘                          │
└─────────────────────────────────┘
```

### **When to Use Each Order:**

**Left to Right (→):** Default, natural reading order
- Horizontal layouts
- Objects arranged horizontally in scene

**Right to Left (←):** Reverse horizontal
- Right-to-left languages (Arabic, Hebrew)
- Objects arranged right-to-left in scene

**Top to Bottom (↓):** Vertical descending
- Vertical stacks
- Floor-by-floor (top floor → ground floor)

**Bottom to Top (↑):** Vertical ascending
- Ground-up ordering (foundation → roof)
- Bottom-to-top priority

---

## Version History

**v1.0.1 (2025-10-18):**
- Added 4-directional numbering orders (left-to-right, right-to-left, top-to-bottom, bottom-to-top)
- Updated Mask to Position Guide node

**v1.0.0 (2025-10-18):**
- Initial guide based on real testing experience
- Documented validated working formats
- Included removal instruction patterns
- Added punctuation usage guide
- Troubleshooting section
- Quick reference templates

---

## Credits

**Testing & Validation:** ArchAi3d
**Model:** Qwen-Image-Edit (Qwen2.5-VL based)
**Research Documents Used:**
- POSITION_GUIDE_FINAL_SOLUTION.md
- Qwen Function 8 (Object Removal) patterns
- Real user testing sessions

---

## Appendix: Qwen Function 8 (Object Removal)

**Official Pattern from Qwen Research:**

**Reliability:** ⭐⭐⭐⭐ Very Reliable

**Format:**
```
remove [object/guide] from the image
```

**Examples:**
```
remove all red rectangles from the image
remove all numbers from the image
remove watermark from the image
remove person from background
```

**Key Insights:**
- Simple, direct command works best
- "from the image" suffix helps
- Specific object names better than generic "guides"
- Can be combined: "remove all red rectangles and numbers from the image"

**When to use:**
- Removing position guide markers
- Watermark removal
- Object removal from scenes
- Cleanup operations

---

**End of Guide**

For questions or updates, contact: ArchAi3d
For latest version, check: `/docs/QWEN_PROMPT_WRITING_GUIDE.md`
