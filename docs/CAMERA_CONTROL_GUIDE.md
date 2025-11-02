# Qwen Camera Control Guide

**Author:** Amir Ferdos (ArchAi3d)
**Version:** 2.0.0
**Date:** 2024

## Overview

This guide covers the two new camera control nodes designed specifically for Qwen Edit 2509 based on extensive community testing and research from Reddit r/StableDiffusion.

## 📦 New Nodes

### 1. ArchAi3D Qwen Camera View
Control camera viewpoints for interior/exterior scenes.

### 2. ArchAi3D Qwen Object Rotation
Rotate camera around objects using the proven "orbit around" technique.

---

## 🎯 Key Research Findings

Based on community testing documented in Reddit, here are the critical insights:

### ✅ What Works BEST:

1. **"Orbit around"** - The MOST reliable term for camera rotation
2. **Environment-only scenes** - Much more predictable than scenes with people
3. **Distance-based movement** - "10m to the left" works better than arbitrary degree rotations
4. **"Dolly"** - Most consistent term for zoom in/out operations
5. **Descriptive scene context** - Adding scene descriptions improves accuracy

### ⚠️ Important Notes:

1. **Left/Right orientation**: `left` in prompt = picture left, NOT subject's left
2. **People in frame**: Model tends to rotate the person instead of the camera
3. **Centered subjects**: Work better than off-center subjects
4. **Angle accuracy**: "90 degrees" may orbit more than "45 degrees", but direction is consistent
5. **Multiple steps**: For 360° rotation, use multiple 90° steps for better control

---

## 🎯 IMPORTANT: System Prompts for Camera Control

**⚠️ CRITICAL:** When using camera control, always use a camera-specific system prompt!

### Why System Prompts Matter:

**Without proper system prompt:**
- ❌ Qwen may redesign the room instead of moving camera
- ❌ Objects and furniture might change
- ❌ Materials and colors may vary
- ❌ Lighting could be different

**With camera system prompt:**
- ✅ Scene content stays identical
- ✅ Only viewpoint/camera position changes
- ✅ Perfect for multi-frame video sequences
- ✅ Consistent results across frames

### Camera-Specific System Prompts Added:

We've added **6 camera-specific system prompts** to the ArchAi3D System Prompt node:

1. **"Cinematographer"** ⭐ Recommended for general use
   - Professional architectural photography
   - Good balance of quality and scene preservation
   - **Use for:** Real estate, architectural visualization

2. **"Virtual Camera Operator"** ⭐ Maximum scene preservation
   - Explicitly preserves all scene content
   - Best for exact scene consistency
   - **Use for:** Product visualization, before/after comparisons

3. **"FLF Video Camera"** ⭐ Multi-frame sequences
   - Optimized for video generation
   - Frame-to-frame continuity
   - **Use for:** Walkthroughs, FLF videos, smooth transitions

4. **"3D Camera Controller"**
   - Technical precision
   - Good for orbit/pan/tilt
   - **Use for:** Technical documentation, precise control

5. **"Architectural Photographer"**
   - Real estate photography style
   - Natural framing and composition
   - **Use for:** Marketing materials, portfolio images

6. **"Scene Preservation Camera"**
   - Maximum preservation emphasis
   - 100% scene consistency goal
   - **Use for:** Scientific work, measurements, quality control

### Quick Usage:

```
1. Add "ArchAi3D Qwen System Prompt" node
2. Select preset: "Cinematographer" or "FLF Video Camera"
3. Connect to encoder's "system_prompt" input
4. Your camera prompt goes to "prompt" input
5. Generate!
```

**📚 Full guide:** See [CAMERA_SYSTEM_PROMPTS.md](CAMERA_SYSTEM_PROMPTS.md) for detailed comparison and examples.

---

## 📘 Node 1: ArchAi3D Qwen Camera View

### Purpose
Generate professional camera viewpoint changes for architectural visualization, interior design, and FLF (First Look Frame) video generation.

### Parameters

#### Movement Types:

1. **vantage_point** (Recommended)
   - Move camera to a new position
   - Most reliable for scene changes
   - Supports distance-based positioning

2. **tilt**
   - Tilt camera up/down
   - Good for revealing ceilings or floor details

3. **combined_movement**
   - Move and tilt simultaneously
   - Creates dynamic camera motion

4. **fov_change**
   - Change field of view
   - Options: normal, wide_100, ultrawide_180, fisheye_180, ultrawide_fisheye

5. **dolly**
   - Zoom in/out
   - Most consistent method for zoom operations

6. **custom**
   - Provide your own prompt

### Example Usage

#### Example 1: Simple Vantage Point Change
```
Movement Type: vantage_point
Vantage Direction: right
Vantage Height: same_level
Distance: 10m
Scene Type: interior
Scene Description: modern living room with glass coffee table

Generated Prompt:
"change the view to a new vantage point 10m to the right"
```

#### Example 2: Ground Level View (Worm's Eye)
```
Movement Type: vantage_point
Vantage Direction: left
Vantage Height: ground_level
Tilt Direction: way_up
Scene Description: luxury bedroom with high ceiling

Generated Prompt:
"change the view to a vantage point at ground level 5m to the left camera tilted way up towards the ceiling"
```

#### Example 3: Ultrawide FOV
```
Movement Type: fov_change
FOV Type: ultrawide_180
Scene Type: interior

Generated Prompt:
"change the view to ultrawide 180 degrees FOV shot on ultrawide lens more of the scene fits the view"
```

#### Example 4: Combined Movement
```
Movement Type: combined_movement
Move Direction: left
Move Tilt: tilt_right

Generated Prompt:
"change the view and move the camera way left while tilting it right"
```

---

## 📗 Node 2: ArchAi3D Qwen Object Rotation

### Purpose
Rotate camera around objects and subjects using the proven "orbit around" technique. Perfect for product visualization, architectural walkarounds, and 360° turntables.

### Parameters

#### Core Settings:

1. **subject**
   - What to orbit around
   - Examples: "the chair", "the building", "the sculpture", "SUBJECT"
   - Use "SUBJECT" as generic placeholder

2. **rotation_axis**
   - horizontal: Left/right orbit (most reliable)
   - vertical: Up/down orbit (less reliable)
   - diagonal: Combined orbit (experimental)

3. **direction**
   - left: Orbit counterclockwise (from top view)
   - right: Orbit clockwise (from top view)
   - up/down: For vertical orbits

4. **angle_preset**
   - 45°: Small rotation (reliable)
   - 90°: Quarter turn (orbits more, even if not exactly 90°)
   - 180°: Opposite view
   - 360°: Full rotation (consider multi-step mode)
   - custom: User-defined angle

#### Advanced Options:

- **rotation_style**: orbit_around (recommended), revolve_around, rotate_camera
- **maintain_distance**: Keep same distance from subject
- **keep_level**: Keep camera level during orbit
- **multi_step_mode**: Break rotation into multiple steps

### Example Usage

#### Example 1: Basic 90° Orbit
```
Subject: the chair
Rotation Axis: horizontal
Direction: right
Angle Preset: 90
Rotation Style: orbit_around

Generated Prompt:
"camera orbit right around the chair by 90 degrees"
```

#### Example 2: Full 360° with Multi-Step
```
Subject: the building
Rotation Axis: horizontal
Direction: right
Angle Preset: 360
Multi-Step Mode: true
Steps: 4

Generated Prompts:
Step 1: camera orbit right around the building by 90 degrees
Step 2: camera orbit right around the building by 90 degrees
Step 3: camera orbit right around the building by 90 degrees
Step 4: camera orbit right around the building by 90 degrees
```

#### Example 3: Product Visualization
```
Subject: the product
Rotation Axis: horizontal
Direction: left
Angle Preset: 45
Maintain Distance: true
Keep Level: true
Scene Context: on white background with studio lighting

Generated Prompt:
"on white background with studio lighting camera orbit left around the product by 45 degrees maintaining distance keeping camera level"
```

#### Example 4: Architectural Walkaround
```
Subject: the house
Rotation Axis: horizontal
Direction: right
Angle Preset: 180
Scene Context: exterior view sunny day

Generated Prompt:
"exterior view sunny day camera orbit right around the house by 180 degrees"
```

---

## 🎬 Workflow Examples

### Workflow 1: Interior Room Exploration

**Use Case:** Generate multiple views of a living room for real estate marketing

1. **Load Image** → Original room photo
2. **ArchAi3D Qwen Camera View**
   - Movement: vantage_point
   - Direction: right
   - Distance: 5m
   - Scene: "modern living room with sectional sofa"
3. **Connect to Qwen Encoder** → Generate view 1
4. **Duplicate node**, change direction to "left" → Generate view 2
5. **Change to FOV: ultrawide_180** → Generate wide-angle view

### Workflow 2: Product 360° Turntable

**Use Case:** Create 8 views around a product for e-commerce

1. **Load Image** → Product photo
2. **ArchAi3D Qwen Object Rotation**
   - Subject: "the product"
   - Angle: 360
   - Multi-Step: true
   - Steps: 8
3. **Text Multiline** → Copy multi-step prompts
4. **Loop through each step** with Qwen encoder
5. **Compile images** → 360° product view

### Workflow 3: Architectural Exterior Flyaround

**Use Case:** Generate cinematic views of a building

1. **Load Image** → Building exterior
2. **ArchAi3D Qwen Object Rotation**
   - Subject: "the building"
   - Direction: right
   - Angle: 45
   - Scene: "exterior view golden hour lighting"
3. **Generate 8 frames** (8 × 45° = 360°)
4. **Combine with video tools** → Smooth flyaround animation

---

## 📊 Parameter Comparison Table

| Parameter | Camera View Node | Object Rotation Node |
|-----------|------------------|---------------------|
| **Best For** | Viewpoint changes | Rotating around objects |
| **Movement Type** | 6 types (vantage, tilt, combined, fov, dolly, custom) | Orbit-based only |
| **Angle Control** | Distance-based (meters) | Angle-based (degrees) |
| **Most Reliable** | Vantage Point + Dolly | orbit_around |
| **Scene Type** | Interior/Exterior/Both | Best with centered subjects |
| **Multi-Step** | No | Yes (for 360°) |

---

## 💡 Best Practices

### For Camera View Node:

1. ✅ **Use distance-based positioning** (e.g., "10m to the left") instead of arbitrary angles
2. ✅ **Add scene descriptions** to improve accuracy
3. ✅ **Use "dolly" for zoom operations** - most consistent
4. ✅ **Test with environment-only scenes** first
5. ⚠️ **Avoid complex multi-axis movements** in single prompts

### For Object Rotation Node:

1. ✅ **Always use "orbit around"** - most reliable technique
2. ✅ **Center your subject** in the frame before rotation
3. ✅ **Use multi-step mode** for 360° rotations (4-8 steps)
4. ✅ **Specify the subject** explicitly (e.g., "the chair" not "it")
5. ⚠️ **Avoid rotating when people are off-center** - less predictable

### General Tips:

1. ✅ **Environment-only scenes** work best
2. ✅ **Keep people centered** if they must be in frame
3. ✅ **Use scene context** for better consistency
4. ✅ **Remember: left/right = picture left/right**, not subject's perspective
5. ✅ **Test with debug mode enabled** to see generated prompts

---

## 🔧 Troubleshooting

### Problem: Camera rotates person instead of scene
**Solution:**
- Remove people from frame if possible
- If people must be included, center them in the frame
- Use "orbit around the room" instead of "orbit around the person"

### Problem: Rotation angle isn't exact
**Solution:**
- This is normal behavior - direction is consistent even if angle varies
- Use multi-step mode for more control
- 90° will orbit more than 45°, which is expected

### Problem: Camera movement is too subtle
**Solution:**
- Increase distance for vantage point changes (try 10m instead of 5m)
- Use larger angles for rotations (90° instead of 45°)
- Add scene context descriptions

### Problem: Movement is unpredictable
**Solution:**
- Switch to environment-only scenes
- Use "orbit around" instead of other rotation terminology
- Enable "maintain_distance" and "keep_level" options

---

## 📚 Prompt Structure Reference

### Optimal Prompt Structure for Qwen Camera Control:

```
[Scene Context] + [Camera Movement] + [Modifiers]
```

**Examples:**

1. **Good:** "modern living room with glass table change the view to a new vantage point 10m to the right"

2. **Better:** "luxury bedroom interior camera orbit right around SUBJECT by 90 degrees maintaining distance"

3. **Best:** "exterior architectural view golden hour lighting camera orbit right around the building by 45 degrees keeping camera level"

### Why This Works:

1. **Scene Context First** - Helps Qwen understand the environment
2. **Clear Action** - "orbit around", "change the view", "dolly in"
3. **Specific Parameters** - "10m to the right", "by 90 degrees"
4. **Modifiers** - "maintaining distance", "keeping camera level"

---

## 🎓 Advanced Techniques

### Technique 1: Progressive Vantage Points
Create a sequence of viewpoints that tell a story:

```
Frame 1: "entrance view, change the view to ground level looking up"
Frame 2: "change the view to a new vantage point 5m to the right same level"
Frame 3: "change the view to a higher vantage point camera tilted down slightly"
Frame 4: "change the view to ultrawide 180 degrees FOV"
```

### Technique 2: Orbit + Dolly Combo
Combine rotation with zoom for dynamic shots:

```
Frame 1: "camera orbit right around the subject by 45 degrees"
Frame 2: "dolly in towards the subject"
Frame 3: "camera orbit right around the subject by 45 degrees"
Frame 4: "dolly out from the subject"
```

### Technique 3: FLF Video Generation
Use for First Look Frame video sequences:

```
1. Start: "interior view from entrance"
2. Establish: "change the view to ultrawide 180 degrees FOV"
3. Detail: "dolly in towards the focal point"
4. Context: "change the view to a new vantage point 10m to the right"
5. Finale: "camera orbit right around the room by 180 degrees"
```

---

## 📖 Research Source

This implementation is based on extensive community testing documented in:

**Reddit Post:** "Prompts for camera control in Qwen Edit 2509"
**Subreddit:** r/StableDiffusion
**Key Contributors:** Community members who tested various prompting techniques

**Key Insight from Original Post:**
> "I have noticed that if you have a person (or multiple) in the picture then these prompts are more of a hit or miss, most of the time it rotates the person around and not the entire scene... however if they are somehow in the center of the scene/frame then some of these commands still work. But for only environment are more predictable."

---

## 🔄 Version History

**v2.0.0** (Current)
- Added ArchAi3D Qwen Camera View node
- Added ArchAi3D Qwen Object Rotation node
- Implemented research-based prompt optimization
- Added multi-step rotation support
- Comprehensive documentation

---

## 📧 Support & Contribution

**Author:** Amir Ferdos (ArchAi3d)
**Email:** Amir84ferdos@gmail.com
**LinkedIn:** https://www.linkedin.com/in/archai3d/
**GitHub:** https://github.com/amir84ferdos

For questions, issues, or contributions, please contact via the channels above.

---

## 📄 License

MIT License - Feel free to use, modify, and distribute with attribution.

---

## 🙏 Acknowledgments

Special thanks to:
- **Prudent-Suspect9834** (Reddit) - Original research post and testing
- **Tomber_** (Reddit) - "orbit around" insight
- **r/StableDiffusion community** - Extensive testing and feedback
- **Qwen Team** - Qwen Edit 2509 model development

---

**Happy Rendering! 🎨📸**
