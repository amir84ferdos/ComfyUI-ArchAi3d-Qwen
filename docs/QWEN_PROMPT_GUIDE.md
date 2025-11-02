# Qwen Edit 2509 - Complete Prompt Engineering Guide

**Version:** 1.0
**Last Updated:** 2025-10-15
**Author:** ArchAi3d (Amir Ferdos)
**Purpose:** Daily reference for creating perfect Qwen prompts and new nodes

---

## Table of Contents

1. [Core Prompting Principles](#core-prompting-principles)
2. [Camera Control Functions](#camera-control-functions)
3. [Image Editing Functions](#image-editing-functions)
4. [Prompt Formula Templates](#prompt-formula-templates)
5. [Best Practices & Anti-Patterns](#best-practices--anti-patterns)
6. [Function Reference Library](#function-reference-library)
7. [Node Design Guidelines](#node-design-guidelines)

---

## Core Prompting Principles

### The Golden Rules

1. **Mention What You Want in Frame**
   - ✅ GOOD: "orbit around the wooden table showing all sides"
   - ❌ BAD: "rotate camera 90 degrees"
   - **Why:** Qwen responds better to target objects than abstract camera movements

2. **Use Spatial Relationships**
   - ✅ GOOD: "move to vantage point 5m to the left of the sofa"
   - ❌ BAD: "move camera left"
   - **Why:** Relative positioning to scene elements is more reliable

3. **Preserve Identity & Context**
   - ✅ GOOD: "keep the subject's id, clothes, facial features, pose, and hairstyle identical"
   - ❌ BAD: No identity preservation mentioned
   - **Why:** Qwen Edit 2509 focuses on consistency - you must explicitly request it

4. **Natural Language Over Coordinates**
   - ✅ GOOD: "from the right side of the room"
   - ❌ BAD: Pixel coordinates (x=450, y=300)
   - **Why:** No community evidence that Qwen supports coordinate-based prompting

5. **Distance Over Degrees**
   - ✅ GOOD: "5 meters to the right"
   - ❌ BAD: "rotate 45 degrees"
   - **Why:** Distance measurements are more consistent than angular measurements

---

## Camera Control Functions

### Function 1: Object Rotation (Orbit Around)

**Purpose:** Rotate camera around a subject while maintaining view of the subject

**Reliability:** ⭐⭐⭐⭐⭐ (Most Reliable)

**Best For:**
- Product photography (360° turntables)
- Architectural exteriors
- Environment-only scenes
- E-commerce showcases

**Works Poorly With:**
- Scenes with people (model rotates person instead of camera)
- Cluttered scenes without clear focal point

**Prompt Formula:**
```
[SCENE_CONTEXT] orbit around [TARGET_OBJECT] [DISTANCE] showing [WHAT_TO_REVEAL], [MODIFIERS]
```

**Parameters:**
- `TARGET_OBJECT`: The subject to orbit around (e.g., "the wooden dining table", "the building")
- `DISTANCE`: Optional - "close", "wide", or omit for medium
- `WHAT_TO_REVEAL`: What should be visible (e.g., "all sides", "the back", "different angles")
- `MODIFIERS`:
  - `maintaining distance` - Keep same distance throughout
  - `keeping camera level` - No vertical movement
  - `smooth camera movement` - Cinematic smoothness

**Examples:**

```
# Product Photography
"orbit around the leather chair showing all sides, maintaining distance, keeping camera level"

# Architectural Exterior
"orbit around the modern house close showing different facades, smooth camera movement"

# Environment Showcase
"in a bright living room with large windows, orbit around the room wide revealing the entire space layout"
```

**Multi-Step Approach:**
For 360° rotations, break into steps:
- **4 steps:** 90° increments (cardinal directions)
- **8 steps:** 45° increments (smooth e-commerce)
- **12 steps:** 30° increments (ultra-smooth cinematic)

---

### Function 2: Vantage Point Change (Photographer Movement)

**Purpose:** Move camera to a completely new position in the scene

**Reliability:** ⭐⭐⭐⭐ (Very Reliable)

**Best For:**
- Changing room perspective
- Moving to opposite side of scene
- Exploring different areas of environment
- Framing specific subjects

**Prompt Formula:**
```
change the view to [HEIGHT_DESCRIPTOR] vantage point [DISTANCE]m to the [DIRECTION] [OPTIONAL_TILT]
```

**Parameters:**
- `HEIGHT_DESCRIPTOR`:
  - `ground level` - Worm's eye view
  - `lower` - Below current height
  - `higher` - Above current height
  - `at face level` - Human eye level
  - Omit for same level
- `DISTANCE`: Meters (1-20m)
- `DIRECTION`:
  - `left` / `right` (picture left/right, NOT subject's left/right)
  - `front` / `back`
  - `left side of room` / `right side of room`
- `OPTIONAL_TILT`:
  - `camera tilted up slightly`
  - `camera tilted down slightly`
  - `camera tilted way up towards the ceiling`
  - `camera aiming upwards`

**Examples:**

```
# Interior Design - Different Room View
"change the view to a new vantage point 8m to the right side of room"

# Architectural - Ground Up View
"change the view to a vantage point at ground level 3m to the front camera tilted way up towards the ceiling"

# Portrait - Face Level View
"change the view to a vantage point at face level 2m to the left"

# Dramatic Angle
"change the view to a higher vantage point 5m to the back camera aiming downwards"
```

**Scene Photographer Pattern:**
This is perfect for your "Scene Photographer" node concept:

```
# Step 1: Describe what you want to frame
scene_context = "modern kitchen with marble countertop and stainless appliances"

# Step 2: Identify the subject
target = "the espresso machine on the counter"

# Step 3: Calculate position
position = "at face level 1m to the right of the espresso machine"

# Step 4: Build prompt
prompt = f"{scene_context}, change the view to a vantage point {position} facing the espresso machine"
```

---

### Function 3: Camera Tilt (Vertical Angle Adjustment)

**Purpose:** Tilt camera up/down without moving position

**Reliability:** ⭐⭐⭐⭐ (Very Reliable)

**Best For:**
- Looking up at tall subjects
- Looking down at ground-level subjects
- Creating psychological effects (power/vulnerability)

**Prompt Formula:**
```
change the view and tilt the camera [TILT_DIRECTION]
```

**Parameters:**
- `TILT_DIRECTION`:
  - `up slightly` - Gentle upward tilt
  - `down slightly` - Gentle downward tilt
  - `way up` - Extreme upward (towards ceiling/sky)
  - `way down` - Extreme downward (bird's eye)

**Psychological Effects:**
- **High angle (looking down):** Subject appears vulnerable, weak, innocent
- **Low angle (looking up):** Subject appears powerful, dominant, imposing
- **Eye level:** Neutral, equal relationship

**Examples:**

```
# Architecture - Emphasize Height
"change the view and tilt the camera way up"

# Portrait - Create Vulnerability
"change the view to extreme top-down view, bird's eye perspective"

# Product - Detail Focus
"change the view and tilt the camera down slightly"
```

---

### Function 4: Combined Movement (Move + Tilt)

**Purpose:** Move camera position while simultaneously tilting

**Reliability:** ⭐⭐⭐ (Reliable with clear instructions)

**Best For:**
- Complex camera choreography
- Cinematic reveals
- Following subjects

**Prompt Formula:**
```
change the view and move the camera [DIRECTION] while tilting it [TILT]
```

**Parameters:**
- `DIRECTION`: `up`, `down`, `way left`, `way right`
- `TILT`: `up`, `down`, `left`, `right`

**Examples:**

```
# Dramatic Reveal
"change the view and move the camera up while tilting it down"

# Sweeping Motion
"change the view and move the camera way right while tilting it left"
```

---

### Function 5: Field of View Change (FOV/Lens)

**Purpose:** Change how much of the scene fits in frame (zoom effect without moving)

**Reliability:** ⭐⭐⭐⭐ (Very Reliable)

**Best For:**
- Creating distortion effects
- Fitting more/less in frame
- Artistic effects

**Prompt Formula:**
```
change the view to [FOV_DESCRIPTOR]
```

**Parameters:**
- `wide 100 degrees FOV` - Wide angle lens
- `ultrawide 180 degrees FOV shot on ultrawide lens more of the scene fits the view` - Ultra wide
- `fisheye 180 fov` - Fisheye distortion
- `ultrawide fisheye lens` - Extreme fisheye

**Examples:**

```
# Real Estate - Show Entire Room
"change the view to ultrawide 180 degrees FOV shot on ultrawide lens more of the scene fits the view"

# Artistic Effect
"change the view to ultrawide fisheye lens"

# Standard Wide
"change the view to wide 100 degrees FOV"
```

---

### Function 6: Dolly (Zoom In/Out)

**Purpose:** Move camera closer to or farther from subject along viewing axis

**Reliability:** ⭐⭐⭐⭐⭐ (Most Consistent for Zoom)

**Best For:**
- Product close-ups
- Zooming in on details
- Zooming out for context

**Prompt Formula:**
```
change the view dolly [DIRECTION] [PREPOSITION] the [TARGET]
```

**Parameters:**
- `DIRECTION`: `in` (closer) or `out` (farther)
- `PREPOSITION`: `towards` (dolly in) or `from` (dolly out)
- `TARGET`: The subject

**Examples:**

```
# Product Detail
"change the view dolly in towards the watch face"

# Context Reveal
"change the view dolly out from the sculpture"

# Cinematic Focus
"change the view dolly in towards the subject smooth camera movement"
```

---

## Image Editing Functions

### Function 7: Person Perspective Change

**Purpose:** Change camera angle for portraits while preserving identity

**Reliability:** ⭐⭐⭐⭐ (Very Reliable with identity preservation)

**Best For:**
- Portrait photography variations
- Social media content
- Character shots

**CRITICAL:** Always include identity preservation clause

**Prompt Formula:**
```
Rotate the angle of the photo to [ANGLE_DESCRIPTION], keep the subject's id, clothes, facial features, pose, and hairstyle identical
```

**Parameters:**
- `ANGLE_DESCRIPTION`:
  - `an ultra-high angle shot (bird's eye view) of the subject, with the camera's point of view positioned very high up, directly looking down at the subject from above`
  - `a high-angle shot of the subject, with the camera positioned above the subject and angled downward`
  - `an eye-level shot of the subject, with the camera positioned at the same height as the subject's eyes`
  - `a low-angle shot of the subject, with the camera positioned below the subject and angled upward`
  - `an ultra-low angle shot (worm's eye view) of the subject, with the camera's point of view positioned very low to the ground, directly looking up at the subject`

**Identity Preservation Clause (REQUIRED):**
```
keep the subject's id, clothes, facial features, pose, and hairstyle identical
```

**Examples:**

```
# High Angle Portrait
"Rotate the angle of the photo to a high-angle shot of the subject, with the camera positioned above the subject and angled downward, keep the subject's id, clothes, facial features, pose, and hairstyle identical"

# Worm's Eye Hero Shot
"Rotate the angle of the photo to an ultra-low angle shot (worm's eye view) of the subject, with the camera's point of view positioned very low to the ground, directly looking up at the subject, keep the subject's id, clothes, facial features, pose, and hairstyle identical"

# Natural Portrait
"Rotate the angle of the photo to an eye-level shot of the subject, with the camera positioned at the same height as the subject's eyes, keep the subject's id, clothes, facial features, pose, and hairstyle identical"
```

---

### Function 8: Object Removal / Adding Elements

**Purpose:** Remove unwanted objects or add new elements to scene

**Reliability:** ⭐⭐⭐⭐ (Very Reliable)

**Best For:**
- Cleaning up scenes
- Adding decorative elements
- Photo manipulation

**Prompt Formula (Remove):**
```
remove [OBJECT_DESCRIPTION] from the image
```

**Prompt Formula (Add):**
```
add [OBJECT_DESCRIPTION] [POSITION] in the image
```

**Examples:**

```
# Watermark Removal (Your Interest)
"remove the watermark from the bottom right corner of the image"

# Clean Up Scene
"remove the electrical wires from the sky"

# Add Decoration
"add a modern floor lamp next to the sofa on the left side"

# Add Context
"add a coffee cup on the table in front of the person"
```

---

### Function 9: Material & Texture Change

**Purpose:** Change materials/textures of objects in scene

**Reliability:** ⭐⭐⭐⭐ (Very Reliable for Interior Design)

**Best For:**
- Interior design visualization
- Product variations
- Material exploration

**Prompt Formula:**
```
change the [OBJECT] material to [NEW_MATERIAL], keep everything else identical
```

**Examples:**

```
# Interior Design Visualization (Your Interest)
"change the kitchen countertop material to white marble with gray veining, keep everything else identical"

"change the flooring material to light oak hardwood, keep everything else identical"

"change the wall paint to sage green, keep everything else identical"

# Product Variations
"change the chair upholstery to navy blue velvet fabric, keep everything else identical"

"change the table surface to polished concrete, keep everything else identical"
```

---

### Function 10: Color Grading / Colorization

**Purpose:** Adjust colors or colorize black & white images

**Reliability:** ⭐⭐⭐⭐ (Very Reliable)

**Best For:**
- Photo restoration
- Mood adjustment
- Creative effects

**Prompt Formula (Colorize):**
```
colorize this black and white photo with realistic colors
```

**Prompt Formula (Color Grade):**
```
adjust the color grading to [MOOD/STYLE]
```

**Examples:**

```
# Colorization (Your Interest)
"colorize this black and white photo with realistic colors appropriate for the 1950s era"

"colorize this vintage photo maintaining natural skin tones and period-accurate colors"

# Color Grading
"adjust the color grading to warm golden hour lighting"

"adjust the color grading to cool blue tones for a modern cinematic look"

"adjust the color grading to vibrant saturated colors"
```

---

### Function 11: Style Transfer & Artistic Effects

**Purpose:** Apply artistic styles or effects to images

**Reliability:** ⭐⭐⭐⭐ (Very Reliable based on Qwen-Image capabilities)

**Best For:**
- Creative visualization
- Artistic interpretation
- Presentation styles

**Prompt Formula:**
```
render this image in [STYLE] style, maintaining [WHAT_TO_PRESERVE]
```

**Examples:**

```
# Architectural Visualization
"render this interior in photorealistic style with professional architectural photography lighting, maintaining all furniture positions and room layout"

# Artistic Interpretation
"render this scene in watercolor painting style, maintaining the composition"

# Technical Visualization
"render this building facade in clean line drawing style for architectural presentation, maintaining accurate proportions"
```

---

### Function 12: Lighting & Time of Day Changes

**Purpose:** Change lighting conditions or time of day

**Reliability:** ⭐⭐⭐⭐ (Very Reliable)

**Best For:**
- Architectural visualization
- Mood changes
- Product photography

**Prompt Formula:**
```
change the lighting to [LIGHTING_DESCRIPTION], keep [WHAT_TO_PRESERVE]
```

**Examples:**

```
# Architectural Visualization
"change the lighting to golden hour sunset with warm orange light streaming through windows, keep all furniture and room layout identical"

"change the lighting to overcast daylight with soft diffused shadows, keep everything else identical"

# Interior Design
"change the lighting to evening ambiance with warm practical lights turned on, keep all furniture positions identical"

# Product Photography
"change the lighting to professional studio lighting with soft shadows, keep the product position identical"
```

---

## Prompt Formula Templates

### Universal Template Structure

```
[SCENE_CONTEXT] + [ACTION_VERB] + [TARGET] + [SPATIAL_RELATIONSHIP] + [MODIFIERS] + [PRESERVATION_CLAUSE]
```

**Components:**

1. **SCENE_CONTEXT** (Optional but Recommended)
   - Describes environment
   - Helps consistency
   - Example: "in a modern minimalist living room with large windows"

2. **ACTION_VERB**
   - `orbit around` - Rotation
   - `change the view to` - Movement
   - `tilt the camera` - Tilt only
   - `dolly in/out` - Zoom
   - `remove` - Object removal
   - `add` - Object addition
   - `change the [object]` - Modification
   - `colorize` - Colorization
   - `adjust` - Fine-tuning

3. **TARGET**
   - The subject of the action
   - Be specific: "the wooden dining table" not "the table"
   - Include material/color for clarity

4. **SPATIAL_RELATIONSHIP**
   - How action relates to target
   - Examples: "5m to the left", "at ground level", "from above"

5. **MODIFIERS** (Optional)
   - `maintaining distance`
   - `keeping camera level`
   - `smooth camera movement`
   - `with soft shadows`
   - `realistic colors`

6. **PRESERVATION_CLAUSE** (When Needed)
   - For person edits: `keep the subject's id, clothes, facial features, pose, and hairstyle identical`
   - For object edits: `keep everything else identical`
   - For scene edits: `keep all furniture and room layout identical`

---

### Template 1: Camera Orbit (360° Product View)

```python
def build_orbit_prompt(
    target_object: str,
    scene_context: str = "",
    distance: str = "medium",  # "close", "medium", "wide"
    maintain_distance: bool = True,
    keep_level: bool = True,
    smooth: bool = True
) -> str:
    """
    Build orbit camera prompt for product/object rotation.

    Example:
        build_orbit_prompt(
            target_object="the leather office chair",
            scene_context="in a modern office with wooden desk",
            distance="close",
            maintain_distance=True,
            keep_level=True,
            smooth=True
        )

    Returns:
        "in a modern office with wooden desk, orbit around the leather office chair
         close showing all sides, maintaining distance, keeping camera level,
         smooth camera movement"
    """
    parts = []

    if scene_context:
        parts.append(scene_context)

    orbit_phrase = f"orbit around {target_object}"

    if distance == "close":
        orbit_phrase += " close"
    elif distance == "wide":
        orbit_phrase += " wide"

    orbit_phrase += " showing all sides"

    modifiers = []
    if maintain_distance:
        modifiers.append("maintaining distance")
    if keep_level:
        modifiers.append("keeping camera level")
    if smooth:
        modifiers.append("smooth camera movement")

    if modifiers:
        orbit_phrase += ", " + ", ".join(modifiers)

    parts.append(orbit_phrase)

    return ", ".join(parts)
```

---

### Template 2: Vantage Point Change (Scene Photographer)

```python
def build_vantage_point_prompt(
    target_subject: str,
    direction: str,  # "left", "right", "front", "back"
    distance_meters: int,
    height: str = "same",  # "ground", "lower", "same", "higher", "face_level"
    tilt: str = "none",  # "none", "up_slightly", "down_slightly", "way_up", "way_down"
    scene_context: str = ""
) -> str:
    """
    Build vantage point change prompt for photographer positioning.

    Example:
        build_vantage_point_prompt(
            target_subject="the marble kitchen island",
            direction="right",
            distance_meters=3,
            height="face_level",
            tilt="down_slightly",
            scene_context="modern kitchen with white cabinets and stainless appliances"
        )

    Returns:
        "modern kitchen with white cabinets and stainless appliances, change the view to
         a vantage point at face level 3m to the right camera tilted down slightly"
    """
    parts = []

    if scene_context:
        parts.append(scene_context)

    vantage_phrase = "change the view to"

    # Height descriptor
    if height == "ground":
        vantage_phrase += " a vantage point at ground level"
    elif height == "lower":
        vantage_phrase += " a lower vantage point"
    elif height == "higher":
        vantage_phrase += " a higher vantage point"
    elif height == "face_level":
        vantage_phrase += " a vantage point at face level"
    else:  # same
        vantage_phrase += " a new vantage point"

    # Distance and direction
    vantage_phrase += f" {distance_meters}m to the {direction}"

    # Optional tilt
    if tilt == "up_slightly":
        vantage_phrase += " camera tilted up slightly"
    elif tilt == "down_slightly":
        vantage_phrase += " camera tilted down slightly"
    elif tilt == "way_up":
        vantage_phrase += " camera tilted way up towards the ceiling"
    elif tilt == "way_down":
        vantage_phrase += " camera aiming downwards"

    parts.append(vantage_phrase)

    return ", ".join(parts)
```

---

### Template 3: Person Portrait Angle

```python
def build_person_angle_prompt(
    angle_type: str,  # "ultra_high", "high", "eye_level", "low", "ultra_low"
    preserve_identity: bool = True
) -> str:
    """
    Build person portrait angle change prompt with identity preservation.

    Example:
        build_person_angle_prompt(angle_type="ultra_low", preserve_identity=True)

    Returns:
        "Rotate the angle of the photo to an ultra-low angle shot (worm's eye view)
         of the subject, with the camera's point of view positioned very low to the
         ground, directly looking up at the subject, keep the subject's id, clothes,
         facial features, pose, and hairstyle identical"
    """
    angle_descriptions = {
        "ultra_high": (
            "Rotate the angle of the photo to an ultra-high angle shot (bird's eye view) "
            "of the subject, with the camera's point of view positioned very high up, "
            "directly looking down at the subject from above"
        ),
        "high": (
            "Rotate the angle of the photo to a high-angle shot of the subject, "
            "with the camera positioned above the subject and angled downward"
        ),
        "eye_level": (
            "Rotate the angle of the photo to an eye-level shot of the subject, "
            "with the camera positioned at the same height as the subject's eyes"
        ),
        "low": (
            "Rotate the angle of the photo to a low-angle shot of the subject, "
            "with the camera positioned below the subject and angled upward"
        ),
        "ultra_low": (
            "Rotate the angle of the photo to an ultra-low angle shot (worm's eye view) "
            "of the subject, with the camera's point of view positioned very low to the "
            "ground, directly looking up at the subject"
        )
    }

    prompt = angle_descriptions.get(angle_type, angle_descriptions["eye_level"])

    if preserve_identity:
        prompt += ", keep the subject's id, clothes, facial features, pose, and hairstyle identical"

    return prompt
```

---

### Template 4: Material Change (Interior Design)

```python
def build_material_change_prompt(
    object_to_change: str,
    new_material: str,
    preserve_rest: bool = True,
    scene_context: str = ""
) -> str:
    """
    Build material change prompt for interior design.

    Example:
        build_material_change_prompt(
            object_to_change="the kitchen countertop",
            new_material="white Carrara marble with gray veining",
            preserve_rest=True,
            scene_context="modern kitchen"
        )

    Returns:
        "modern kitchen, change the kitchen countertop material to white Carrara marble
         with gray veining, keep everything else identical"
    """
    parts = []

    if scene_context:
        parts.append(scene_context)

    change_phrase = f"change {object_to_change} material to {new_material}"

    if preserve_rest:
        change_phrase += ", keep everything else identical"

    parts.append(change_phrase)

    return ", ".join(parts)
```

---

### Template 5: Object Removal (Watermark, Cleanup)

```python
def build_removal_prompt(
    object_to_remove: str,
    location_description: str = ""
) -> str:
    """
    Build object removal prompt for cleanup tasks.

    Example:
        build_removal_prompt(
            object_to_remove="the watermark",
            location_description="from the bottom right corner"
        )

    Returns:
        "remove the watermark from the bottom right corner of the image"
    """
    if location_description:
        return f"remove {object_to_remove} {location_description} of the image"
    else:
        return f"remove {object_to_remove} from the image"
```

---

### Template 6: Lighting Change

```python
def build_lighting_change_prompt(
    new_lighting: str,
    preserve_elements: str = "everything else",
    scene_context: str = ""
) -> str:
    """
    Build lighting change prompt for mood adjustment.

    Example:
        build_lighting_change_prompt(
            new_lighting="golden hour sunset with warm orange light streaming through windows",
            preserve_elements="all furniture and room layout",
            scene_context="living room"
        )

    Returns:
        "living room, change the lighting to golden hour sunset with warm orange light
         streaming through windows, keep all furniture and room layout identical"
    """
    parts = []

    if scene_context:
        parts.append(scene_context)

    lighting_phrase = f"change the lighting to {new_lighting}"

    if preserve_elements:
        lighting_phrase += f", keep {preserve_elements} identical"

    parts.append(lighting_phrase)

    return ", ".join(parts)
```

---

## Best Practices & Anti-Patterns

### ✅ DO: Best Practices

1. **Always Describe Target Objects**
   ```python
   # ✅ GOOD - Specific target
   "orbit around the red velvet armchair"

   # ❌ BAD - Generic target
   "orbit around the chair"
   ```

2. **Use Scene Context for Consistency**
   ```python
   # ✅ GOOD - Context provided
   "in a minimalist bedroom with white walls, change the view to..."

   # ❌ BAD - No context
   "change the view to..."
   ```

3. **Preserve Identity for Person Edits**
   ```python
   # ✅ GOOD - Identity preserved
   "Rotate angle to low-angle shot, keep the subject's id, clothes, facial features, pose, and hairstyle identical"

   # ❌ BAD - No preservation
   "Rotate angle to low-angle shot"
   ```

4. **Use Distance Over Degrees**
   ```python
   # ✅ GOOD - Distance-based
   "change the view to a vantage point 5m to the right"

   # ❌ BAD - Degree-based
   "rotate camera 90 degrees right"
   ```

5. **Break Complex Movements into Steps**
   ```python
   # ✅ GOOD - Multi-step for 360°
   steps = [
       "orbit around table from front to right side",
       "orbit around table from right side to back",
       "orbit around table from back to left side",
       "orbit around table from left side to front"
   ]

   # ❌ BAD - Single 360° command
   "orbit around table 360 degrees"
   ```

6. **Mention What You Want in Frame**
   ```python
   # ✅ GOOD - Target mentioned
   "orbit around the glass coffee table showing the transparent top and metal legs"

   # ❌ BAD - No specific target
   "move camera in a circle"
   ```

---

### ❌ DON'T: Anti-Patterns

1. **Don't Use Pixel Coordinates**
   ```python
   # ❌ BAD - Coordinates not supported
   "move camera to position (450, 300)"

   # ✅ GOOD - Natural language
   "move to vantage point 3m to the left of the sofa"
   ```

2. **Don't Forget Scene Type Warnings**
   ```python
   # ❌ BAD - Ignoring scene type
   # Using orbit on person portrait (will rotate person, not camera)

   # ✅ GOOD - Use appropriate function
   # Use "Person Perspective Change" for portraits
   ```

3. **Don't Use Abstract Camera Terms Only**
   ```python
   # ❌ BAD - Abstract only
   "dolly left and pan right"

   # ✅ GOOD - With target reference
   "move camera left while keeping the sculpture centered in frame"
   ```

4. **Don't Omit Modifiers for Critical Constraints**
   ```python
   # ❌ BAD - No constraints
   "orbit around the building"
   # (Camera might drift up/down, change distance)

   # ✅ GOOD - With constraints
   "orbit around the building, maintaining distance, keeping camera level"
   ```

5. **Don't Mix Incompatible Modifiers**
   ```python
   # ❌ BAD - Conflicting modifiers
   "orbit around table while moving closer and farther"

   # ✅ GOOD - Consistent modifiers
   "orbit around table maintaining distance"
   ```

---

## Function Reference Library

### Quick Reference Card

| Function | Use When | Reliability | Key Term | Avoid |
|----------|----------|-------------|----------|-------|
| **Orbit Around** | Rotate around object 360° | ⭐⭐⭐⭐⭐ | "orbit around [target]" | Scenes with people |
| **Vantage Point** | Move to new position | ⭐⭐⭐⭐ | "change the view to vantage point Xm to [direction]" | Abstract directions |
| **Tilt** | Look up/down only | ⭐⭐⭐⭐ | "tilt the camera [direction]" | Extreme angles without context |
| **Combined Movement** | Move + tilt together | ⭐⭐⭐ | "move camera [dir] while tilting [dir]" | Too many simultaneous changes |
| **FOV Change** | Change lens/zoom effect | ⭐⭐⭐⭐ | "ultrawide 180 degrees FOV" | Extreme FOV without reason |
| **Dolly** | Zoom in/out along axis | ⭐⭐⭐⭐⭐ | "dolly in towards [target]" | Generic "zoom" |
| **Person Angle** | Portrait angle change | ⭐⭐⭐⭐ | "Rotate angle to [angle], keep identity" | Without identity preservation |
| **Material Change** | Interior design variations | ⭐⭐⭐⭐ | "change [object] material to [material]" | Without preservation clause |
| **Object Removal** | Cleanup, watermark removal | ⭐⭐⭐⭐ | "remove [object] from [location]" | Vague object descriptions |
| **Colorization** | B&W to color | ⭐⭐⭐⭐ | "colorize with realistic colors" | Without era/context |
| **Lighting Change** | Mood, time of day | ⭐⭐⭐⭐ | "change lighting to [description]" | Without preservation |

---

### Scene Type Decision Tree

```
Is there a PERSON in the frame?
│
├─ YES → Is the person the PRIMARY subject?
│   │
│   ├─ YES → Use "Person Perspective Change" function
│   │         (Avoid orbit - will rotate person instead of camera)
│   │
│   └─ NO → Can you crop/frame without the person?
│       │
│       ├─ YES → Reframe to environment-only, then use "Orbit Around"
│       │
│       └─ NO → Use "Vantage Point Change" (more reliable with people in scene)
│
└─ NO → What type of movement?
    │
    ├─ 360° rotation around object → "Orbit Around" (most reliable)
    │
    ├─ Move to different position → "Vantage Point Change"
    │
    ├─ Look up/down only → "Tilt"
    │
    ├─ Zoom in/out → "Dolly" (most consistent)
    │
    └─ Change lens effect → "FOV Change"
```

---

### Modifier Combinations

**Compatible Modifier Sets:**

```python
# Orbit Around
modifiers = [
    "maintaining distance",      # Keep same distance
    "keeping camera level",      # No vertical drift
    "smooth camera movement"     # Cinematic smoothness
]

# Vantage Point
modifiers = [
    "camera tilted up slightly",
    "camera tilted down slightly",
    "camera tilted way up towards the ceiling",
    "camera aiming upwards"
]

# Material Change
modifiers = [
    "keep everything else identical",
    "keep all other materials identical",
    "maintain the same lighting"
]

# Lighting Change
modifiers = [
    "keep all furniture and room layout identical",
    "keep everything else identical",
    "maintain the same camera angle"
]
```

---

## Node Design Guidelines

### Creating New Qwen Nodes - Checklist

When creating a new ComfyUI node for Qwen Edit 2509, follow this checklist:

#### 1. Node Purpose & Scope

- [ ] **Single Responsibility:** Does this node do ONE thing well?
- [ ] **Distinct from Existing Nodes:** Is this different enough from existing nodes?
- [ ] **Real User Need:** Does this solve a real prompting challenge?

#### 2. Input Parameters

- [ ] **Essential Parameters Only:** No "nice to have" options
- [ ] **Logical Grouping:** Group related parameters together
- [ ] **Clear Defaults:** Sensible defaults that work for 80% of use cases
- [ ] **Helpful Tooltips:** Each input has clear, concise tooltip
- [ ] **Value Ranges:** Min/max values make sense for Qwen's capabilities

#### 3. Prompt Building Logic

- [ ] **Use Template Functions:** Leverage prompt formula templates
- [ ] **Scene Context Integration:** Support optional scene description
- [ ] **Modifier System:** Implement consistent modifier pattern
- [ ] **Identity/Object Preservation:** Include preservation clauses where needed
- [ ] **Debug Mode:** Include option to print generated prompt

#### 4. Multi-Step Support

- [ ] **Breaking Down Complex Operations:** Option to split into multiple frames
- [ ] **Consistent Step Calculation:** Clear algorithm for determining steps
- [ ] **Per-Step Context:** Each step gets scene context if provided
- [ ] **Per-Step Modifiers:** Rebuild modifiers for each step
- [ ] **Progress Indication:** Clear frame numbering (Frame 1/8)

#### 5. Code Quality

- [ ] **Constants at Top:** All text mappings in constants section
- [ ] **Helper Functions:** Extract reusable logic
- [ ] **Type Hints:** Use proper type annotations
- [ ] **Docstrings:** Clear documentation for all functions
- [ ] **DRY Principle:** Use defaults dict for common preset values

#### 6. User Experience

- [ ] **Preset System:** Include 5-10 useful presets for common use cases
- [ ] **Clear Naming:** Preset names match their actual behavior
- [ ] **Scene Type Detection:** Warn about known limitations (e.g., people in frame)
- [ ] **Smart Defaults:** Auto-configure options when using presets
- [ ] **Output Variants:** Provide both minimal prompt and full prompt with context

---

### Node Template Structure

```python
# Standard Node Structure for Qwen Edit 2509

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override
from typing import Dict, List, Tuple

# ============================================================================
# CONSTANTS - All text mappings at top for easy modification
# ============================================================================

OPTION_MAP = {
    "option1": "prompt text 1",
    "option2": "prompt text 2",
}

PRESET_DEFAULTS = {
    "common_param1": default_value1,
    "common_param2": default_value2,
}

PRESETS = {
    "preset_name": {
        "description": "Clear description of what this does",
        "specific_param": value,
        **PRESET_DEFAULTS  # Spread common defaults
    }
}

# ============================================================================
# HELPER FUNCTIONS - Pure functions for prompt building
# ============================================================================

def build_base_prompt(param1: str, param2: str) -> str:
    """Build the base prompt structure.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Base prompt string
    """
    pass


def build_modifiers(modifier_params: Dict) -> List[str]:
    """Build list of prompt modifiers.

    Args:
        modifier_params: Dict of modifier settings

    Returns:
        List of modifier strings
    """
    pass


def add_scene_context(prompt: str, scene_context: str) -> str:
    """Add scene context to prompt if provided.

    Args:
        prompt: The base prompt
        scene_context: Optional scene description

    Returns:
        Prompt with scene context prepended
    """
    if scene_context.strip():
        return f"{scene_context.strip()}, {prompt}"
    return prompt


# ============================================================================
# NODE DEFINITION
# ============================================================================

class ArchAi3D_Qwen_YourNode(io.ComfyNode):
    """Short description of what this node does.

    Longer explanation of:
    - What problem it solves
    - When to use it
    - What it's good for
    - What to avoid
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ArchAi3D_Qwen_YourNode",
            category="ArchAi3d/Qwen/CategoryName",
            inputs=[
                # Group 1: Main Function Selection
                io.Combo.Input(
                    "main_option",
                    options=list(OPTION_MAP.keys()),
                    default="option1",
                    tooltip="Clear description of what this controls"
                ),

                # Group 2: Function-Specific Parameters
                io.Int.Input(
                    "numeric_param",
                    default=5,
                    min=1,
                    max=20,
                    step=1,
                    tooltip="What this number controls and why"
                ),

                # Group 3: Scene Context
                io.String.Input(
                    "scene_context",
                    multiline=True,
                    default="",
                    tooltip="Optional scene description for better consistency"
                ),

                # Group 4: Advanced Options
                io.Boolean.Input(
                    "debug_mode",
                    default=False,
                    tooltip="Print generated prompt to console"
                ),
            ],
            outputs=[
                io.String.Output("prompt", tooltip="Generated prompt for Qwen"),
                io.String.Output("full_prompt", tooltip="Prompt with scene context"),
            ],
        )

    @classmethod
    def execute(cls, main_option, numeric_param, scene_context, debug_mode) -> io.NodeOutput:
        """Execute the node logic.

        Steps:
        1. Build base prompt from parameters
        2. Add modifiers
        3. Add scene context
        4. Debug output if requested
        5. Return outputs
        """

        # Step 1: Build base prompt
        base_prompt = build_base_prompt(main_option, numeric_param)

        # Step 2: Add modifiers
        modifiers = build_modifiers({"param": numeric_param})
        if modifiers:
            base_prompt += ", " + ", ".join(modifiers)

        # Step 3: Add scene context
        full_prompt = add_scene_context(base_prompt, scene_context)

        # Step 4: Debug output
        if debug_mode:
            print("=" * 70)
            print(f"ArchAi3D_Qwen_YourNode - Generated Prompt")
            print("=" * 70)
            print(f"Option: {main_option}")
            print(f"Base Prompt: {base_prompt}")
            print(f"Full Prompt: {full_prompt}")
            print("=" * 70)

        # Step 5: Return
        return io.NodeOutput(base_prompt, full_prompt)


# ============================================================================
# EXTENSION REGISTRATION
# ============================================================================

class YourNodeExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [ArchAi3D_Qwen_YourNode]

async def comfy_entrypoint():
    return YourNodeExtension()
```

---

### Testing New Nodes

**Test Case Checklist:**

1. **Baseline Test**
   - [ ] Test with all default values
   - [ ] Verify prompt is grammatically correct
   - [ ] Check output is not empty

2. **Edge Cases**
   - [ ] Test with empty scene context
   - [ ] Test with very long scene context
   - [ ] Test with min/max parameter values
   - [ ] Test with special characters in text inputs

3. **Preset Tests**
   - [ ] Test each preset individually
   - [ ] Verify preset settings override defaults correctly
   - [ ] Check preset descriptions match actual behavior

4. **Multi-Step Tests** (if applicable)
   - [ ] Verify correct number of steps generated
   - [ ] Check each step has scene context
   - [ ] Verify modifiers appropriate for each step

5. **Integration Tests**
   - [ ] Test output connects to Qwen Edit 2509 node
   - [ ] Verify prompts produce expected image changes
   - [ ] Check consistency across multiple runs

---

### Prompt Quality Checklist

Before finalizing any prompt, verify:

- [ ] **Target Specificity:** Is the target object clearly described?
- [ ] **Spatial Clarity:** Are spatial relationships clear and unambiguous?
- [ ] **Modifier Compatibility:** Do all modifiers make sense together?
- [ ] **Preservation Clause:** Is identity/context preserved when needed?
- [ ] **Grammar:** Is the prompt grammatically correct?
- [ ] **Length:** Is the prompt concise but complete? (aim for 15-30 words)
- [ ] **Scene Context:** Is scene description provided for consistency?
- [ ] **No Contradictions:** Do parts of the prompt contradict each other?

---

## Appendix: Research Sources

### Community Findings (Reddit r/StableDiffusion)

**Key Insights:**
1. "orbit around" most reliable term for rotation
2. Environment-only scenes work best (no people)
3. People in frame → model rotates person instead of camera
4. "left/right" = picture left/right, NOT subject's left/right
5. "Dolly" most consistent for zoom in/out
6. Distance measurements (meters) better than degrees
7. Describing scene elements helps accuracy

### Official Qwen Documentation

**Qwen 2.5 VL Capabilities:**
- High-resolution image understanding
- Multi-image reasoning
- Video understanding
- Text rendering in images
- Mathematical diagram recognition

**Qwen Edit 2509 Features:**
- Image editing with consistency preservation
- Camera angle manipulation
- Object addition/removal
- Material and texture changes
- Lighting adjustments
- Style transfer
- Multi-image editing (blend elements from multiple images)
- ControlNet integration (depth maps, edge maps, keypoint poses)

### Advanced Syntax Patterns (From Official Examples)

**1. Transform Syntax for Complete Changes:**
```
Transform [CURRENT_STATE] to [NEW_STATE]
```
Example: "Transform the background to a sunset beach scene"
Use for: Complete camera angle changes, background replacements

**2. Replace Syntax for Object Substitution:**
```
Replace [OBJECT_A] with [OBJECT_B]
```
Example: "Replace the wooden chair with a modern leather armchair"
Use for: Swapping specific objects while preserving scene

**3. Multi-Image Editing Pattern:**
```
[Treat Image 2 as canvas, Image 1 as donor] + [describe which elements transfer and which remain]
```
Example: "Use Image 2 as the base, transfer the person from Image 1 to the center, keep the background from Image 2"
Use for: Person + product, person + scene, object + object combinations

**4. ControlNet-Guided Edits:**
```
[EDIT_INSTRUCTION] + [following the provided {depth_map|edge_map|keypoint_map}]
```
Example: "Change the person's clothing to a business suit following the provided keypoint map"
Use for: Precise structural control over edits

**5. Concrete Prompt Structure (Recommended):**
```
[SUBJECT] + [ACTION] + [SETTING] + [STYLE]
```
Example: "The woman standing in a modern office wearing professional attire, photorealistic"
Keep it: Single-sentence goal + a few strong descriptors

**6. Studio Lighting Pattern:**
```
[MAIN_EDIT] + Studio lighting with soft, natural shadows
```
Example: "Change the product color to navy blue, Studio lighting with soft, natural shadows"
Use for: Professional product photography consistency

**7. Consistent Integration Clause:**
```
[EDIT_INSTRUCTION] + maintaining consistent lighting, perspective, and natural integration
```
Example: "Add a modern floor lamp on the left, maintaining consistent lighting, perspective, and natural integration"
Use for: Ensuring added elements match the scene naturally

### Testing Best Practices

From community experimentation:
1. Always test on environment-only scenes first
2. Test with and without scene context to see difference
3. Try both distance and degree measurements (distance wins)
4. Compare "orbit around" vs other rotation terms ("rotate", "move around")
5. Test identity preservation clause on/off for person edits
6. Experiment with modifier combinations to find compatible sets
7. For multi-image edits, clearly specify canvas vs donor image
8. Use ControlNet guides for pose/structure preservation
9. Try "Transform X to Y" for dramatic changes vs "change X" for subtle adjustments
10. Keep prompts concrete: subject + action + setting + style

### Files Read (7 Total)

**Successfully Read (5/7):**
1. ✅ qwen.pdf (1.8MB) - Community Reddit research on camera control
2. ✅ qwen-2.pdf (3.5MB) - Person perspective tutorial with identity preservation
3. ✅ qwen-vl.pdf (182KB) - Qwen 2.5 VL official documentation
4. ✅ Qwen-repo1.pdf (744KB) - Qwen-Image text rendering capabilities
5. ✅ Qwen-repo2.pdf (756KB) - Qwen-Image-Edit-2509 features and consistency

**Unable to Read (2/7 - Exceeded Size Limit):**
6. ❌ Qwen-repo3.pdf (50MB) - Exceeds 32MB read limit
7. ❌ Qwen_Image.pdf (42MB) - Exceeds 32MB read limit

**Supplemented With:**
- Web research from official Qwen documentation sites
- Community guides and tutorials (DEV Community, Atlabs AI, NextDiffusion)
- Official Hugging Face model card documentation

---

## Version History

**v1.1 - 2025-10-15**
- Added advanced syntax patterns from official Qwen examples
- Added "Transform X to Y" and "Replace X with Y" patterns
- Added multi-image editing guidelines
- Added ControlNet integration notes
- Added concrete prompt structure recommendations
- Supplemented with web research from official documentation
- Documented all 7 source files (5 read fully, 2 supplemented via web)

**v1.0 - 2025-10-15**
- Initial documentation
- 12 core functions documented
- Prompt templates for all major use cases
- Node design guidelines
- Testing checklist

---

## Quick Start Examples

### Example 1: Product Photography (360° Turntable)

```python
# Goal: Create 8-frame 360° rotation of a product

scene = "white studio background with soft lighting"
product = "the wireless headphones"

prompts = []
for i in range(8):
    angle = i * 45
    step_prompt = f"{scene}, orbit around {product} showing all sides, maintaining distance, keeping camera level, smooth camera movement"
    prompts.append(step_prompt)

# Feed each prompt sequentially to Qwen Edit 2509
```

### Example 2: Interior Design Exploration

```python
# Goal: Move photographer to different positions to showcase room

scene = "modern living room with floor-to-ceiling windows and minimalist furniture"

positions = [
    "change the view to a new vantage point 6m to the right side of room",
    "change the view to a vantage point at ground level 2m to the front camera tilted way up",
    "change the view to a higher vantage point 8m to the back",
    "change the view to a vantage point at face level 3m to the left"
]

prompts = [f"{scene}, {pos}" for pos in positions]
```

### Example 3: Material Variations for Client Presentation

```python
# Goal: Show different countertop materials for client decision

scene = "modern kitchen with white cabinets and stainless appliances"
countertop_object = "the kitchen countertop"

materials = [
    "white Carrara marble with gray veining",
    "black granite with subtle sparkle",
    "butcher block oak wood",
    "polished concrete with light gray tone",
    "white quartz with marble-like pattern"
]

prompts = []
for material in materials:
    prompt = f"{scene}, change {countertop_object} material to {material}, keep everything else identical"
    prompts.append(prompt)
```

### Example 4: Lighting Studies (Time of Day)

```python
# Goal: Show how space looks at different times

scene = "architectural interior with large windows"

lighting_conditions = [
    "early morning soft blue light streaming through windows",
    "golden hour sunset with warm orange light",
    "overcast daylight with diffused soft shadows",
    "evening ambiance with warm practical lights turned on",
    "midday bright sunlight with strong shadows"
]

prompts = []
for lighting in lighting_conditions:
    prompt = f"{scene}, change the lighting to {lighting}, keep all furniture and room layout identical"
    prompts.append(prompt)
```

### Example 5: Person Portrait Angle Variations

```python
# Goal: Create 5 different angle variations of same person

angles = ["ultra_high", "high", "eye_level", "low", "ultra_low"]

prompts = [build_person_angle_prompt(angle, preserve_identity=True) for angle in angles]

# Results in 5 prompts, each preserving identity:
# 1. Bird's eye view
# 2. High angle (looking down)
# 3. Eye level (neutral)
# 4. Low angle (looking up)
# 5. Worm's eye view (extreme low)
```

---

## Final Notes

This guide is a living document. As you discover new prompting techniques through experimentation:

1. **Document New Findings:** Add successful prompt patterns to templates
2. **Update Reliability Ratings:** Adjust ⭐ ratings based on testing
3. **Add New Functions:** Document new capabilities as Qwen evolves
4. **Refine Templates:** Optimize templates based on real-world usage
5. **Share with Community:** Contribute findings back to Qwen community

**Remember:** The best prompt is the one that gets you the result you want. Use this guide as a starting point, then experiment and iterate.

---

**Created by:** Amir Ferdos (ArchAi3d)
**Email:** Amir84ferdos@gmail.com
**LinkedIn:** https://www.linkedin.com/in/archai3d/
**GitHub:** https://github.com/amir84ferdos

**License:** MIT - Use freely, attribute when sharing
