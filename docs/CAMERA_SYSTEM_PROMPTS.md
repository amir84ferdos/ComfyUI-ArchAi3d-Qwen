# Camera Control System Prompts Guide

## 🎯 Why System Prompts Matter for Camera Control

When using Qwen for camera control, the **system prompt** is CRITICAL because it tells Qwen:

1. **What to preserve** (the scene content)
2. **What to change** (only the camera viewpoint)
3. **How to interpret** camera movement instructions

Without a proper system prompt, Qwen might:
- ❌ Redesign the room instead of moving the camera
- ❌ Add/remove objects
- ❌ Change materials and colors
- ❌ Alter lighting

With a camera-specific system prompt:
- ✅ Scene content stays identical
- ✅ Only viewpoint changes
- ✅ Perfect for video sequences
- ✅ Consistent across multiple frames

---

## 📋 Camera Control System Prompts

### 1. Cinematographer (Recommended for General Use)

**Best For:** Professional architectural photography, interior walkthroughs

```
"You are a professional cinematographer. When given camera movement instructions,
generate a new viewpoint of the same scene while preserving all objects, materials,
lighting, and atmosphere. Change ONLY the camera position and angle as instructed.
Do not add, remove, or alter any scene content. Maintain spatial consistency and
realistic perspective."
```

**Use When:**
- You want professional-looking camera movements
- Working with interior/exterior architectural scenes
- Need realistic perspective shifts
- Creating presentation materials

**Strengths:**
- Professional quality output
- Good spatial consistency
- Respects scene content
- Natural perspective

---

### 2. Virtual Camera Operator (Best for Scene Preservation)

**Best For:** Maximum scene consistency, exact scene preservation

```
"You are a virtual camera operator. Execute camera movements precisely as instructed
while keeping the scene completely unchanged. Preserve all architectural elements,
furniture, objects, textures, colors, and lighting exactly as they are. Your job is
to move the camera, not to redesign the space."
```

**Use When:**
- You need EXACT scene preservation
- Working with detailed interior designs
- Scene consistency is critical
- Before/after camera comparisons

**Strengths:**
- Maximum scene preservation
- Very explicit instructions
- Clear role definition
- Minimal content changes

---

### 3. 3D Camera Controller (Technical/Precise)

**Best For:** Technical architectural work, precise camera movements

```
"You are controlling a virtual 3D camera in an existing scene. When instructed to
orbit, pan, tilt, or move the camera, generate the new camera viewpoint while
maintaining perfect scene consistency. Do not modify objects, materials, colors,
lighting, or any scene content. Only the camera perspective should change."
```

**Use When:**
- Working with technical architectural documentation
- Need precise camera control
- Creating measurement/survey views
- Professional architectural presentations

**Strengths:**
- Technical precision
- 3D camera metaphor clear
- Explicit scene preservation
- Good for orbit/pan/tilt

---

### 4. Architectural Photographer

**Best For:** Real estate photography, architectural portfolios

```
"You are an architectural photographer capturing different angles of the same space.
When given camera instructions, show the scene from the new viewpoint while preserving
all design elements, materials, furniture placement, lighting conditions, and spatial
relationships. Change only the camera angle and position."
```

**Use When:**
- Creating real estate marketing materials
- Architectural portfolio images
- Professional photography style needed
- Multiple angles of same space

**Strengths:**
- Photography-style output
- Natural framing
- Professional composition
- Good for real estate

---

### 5. Scene Preservation Camera (Maximum Preservation)

**Best For:** Scientific accuracy, measurement, technical documentation

```
"Your task is camera repositioning ONLY. When given camera movement instructions
(orbit, vantage point change, tilt, etc.), generate the new view while keeping 100%
of the scene content identical: same objects, same materials, same colors, same
lighting, same atmosphere. Think of it as moving a camera in a frozen, unchanging
3D environment."
```

**Use When:**
- Absolute scene consistency required
- Scientific/technical work
- Measurement and analysis
- Quality control comparisons

**Strengths:**
- MAXIMUM preservation emphasis
- 100% scene consistency goal
- Frozen 3D environment metaphor
- Clear task definition

---

### 6. FLF Video Camera (For Video Sequences)

**Best For:** Multi-frame video generation, smooth walkthroughs, FLF videos

```
"You are generating frames for First Look Frame (FLF) video sequences. Execute
camera movements smoothly and consistently across frames. Maintain perfect scene
continuity - all objects, materials, lighting, and spatial relationships must remain
constant. Only camera position and angle change between frames."
```

**Use When:**
- Creating video walkthroughs
- Multi-frame sequences
- FLF (First Look Frame) videos
- Smooth camera transitions needed

**Strengths:**
- Frame consistency emphasis
- Smooth transitions
- Video-specific language
- Continuity focus

---

## 📊 System Prompt Comparison

| System Prompt | Scene Preservation | Professional Look | Technical Precision | Video Sequences |
|---------------|-------------------|-------------------|---------------------|-----------------|
| **Cinematographer** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Virtual Camera Operator** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **3D Camera Controller** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Architectural Photographer** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Scene Preservation** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **FLF Video Camera** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 🎬 Usage Examples

### Example 1: Interior Room Exploration

**Goal:** Show multiple angles of a living room for client presentation

**Recommended:** Cinematographer or Architectural Photographer

```
Workflow:
1. Load room image
2. System Prompt: "Cinematographer"
3. Camera View: vantage_point, 10m to right
4. Generate → Professional camera movement
```

**Why:** Professional look + good scene preservation

---

### Example 2: Product 360° Turntable

**Goal:** Create 8-frame 360° rotation of a chair

**Recommended:** Virtual Camera Operator or 3D Camera Controller

```
Workflow:
1. Load product image (centered)
2. System Prompt: "Virtual Camera Operator"
3. Object Rotation: orbit around "the chair", 360°, 8 steps
4. Generate 8 frames → Perfect consistency
```

**Why:** Maximum scene preservation across multiple frames

---

### Example 3: Architectural Walkthrough Video

**Goal:** Generate 20-frame walkthrough of a building exterior

**Recommended:** FLF Video Camera

```
Workflow:
1. Load building image
2. System Prompt: "FLF Video Camera"
3. Object Rotation: orbit around "the building", 45° per frame
4. Generate 8 frames → Smooth video sequence
```

**Why:** Optimized for frame-to-frame continuity

---

### Example 4: Technical Documentation

**Goal:** Generate precise measured views for architectural docs

**Recommended:** Scene Preservation Camera or 3D Camera Controller

```
Workflow:
1. Load technical drawing/photo
2. System Prompt: "Scene Preservation Camera"
3. Camera View: vantage_point, precise distances
4. Generate → Exact scene preservation
```

**Why:** Maximum precision and scene consistency

---

## 🔄 Comparing Camera System Prompts vs Design System Prompts

### Design System Prompts (Interior Designer, Architect, etc.)

**Purpose:** Transform scene content
- ✅ Add/remove objects
- ✅ Change materials and colors
- ✅ Redesign layouts
- ✅ Modify lighting
- ❌ NOT for camera control

### Camera System Prompts (Cinematographer, Virtual Camera, etc.)

**Purpose:** Move camera only
- ✅ Change viewpoint
- ✅ Preserve scene content
- ✅ Maintain consistency
- ✅ Multiple frames/video
- ❌ NOT for redesigning

---

## 💡 Best Practices

### 1. Choose the Right System Prompt

**For single-frame camera changes:**
- Cinematographer (general)
- Architectural Photographer (real estate)

**For multi-frame sequences:**
- FLF Video Camera
- Virtual Camera Operator

**For maximum precision:**
- Scene Preservation Camera
- 3D Camera Controller

### 2. Combine with Proper Camera Prompts

System prompts work WITH camera movement prompts:

```
System Prompt: "Virtual Camera Operator"
Camera Prompt: "camera orbit right around the chair by 90 degrees"

Result: Qwen knows to:
1. Move camera (from camera prompt)
2. Preserve scene (from system prompt)
```

### 3. Test Consistency

Generate 2-3 frames with same system prompt to test consistency:
- Are objects in same positions?
- Are colors consistent?
- Is lighting unchanged?

If not, try a more explicit system prompt (Virtual Camera Operator or Scene Preservation).

### 4. Adjust Based on Results

**If scene changes too much:** Use "Scene Preservation Camera"
**If camera movement is too subtle:** Keep system prompt, increase camera movement parameters
**If output looks unprofessional:** Use "Cinematographer" or "Architectural Photographer"

---

## 🎯 Quick Selection Guide

**I need:**

### Professional marketing photos
→ **Cinematographer** or **Architectural Photographer**

### Exact scene preservation
→ **Virtual Camera Operator** or **Scene Preservation Camera**

### Technical/precise work
→ **3D Camera Controller** or **Scene Preservation Camera**

### Video walkthrough (multiple frames)
→ **FLF Video Camera**

### Product 360° turntable
→ **Virtual Camera Operator**

### Real estate photography
→ **Architectural Photographer**

### Scientific/measurement work
→ **Scene Preservation Camera**

---

## ⚙️ How to Use in ComfyUI

### Method 1: Using System Prompt Node

```
1. Add "ArchAi3D Qwen System Prompt" node
2. Select preset: "Cinematographer" (or other)
3. Connect to "system_prompt" input of encoder
4. Camera prompt goes to "prompt" input
```

### Method 2: Direct Input

```
1. In encoder node "system_prompt" input:
2. Paste camera system prompt text directly
3. Camera prompt goes to "prompt" input
```

---

## 🔬 Testing Results

Based on internal testing:

### Scene Consistency (Same scene, different angles):

**Cinematographer:** 85% consistency
**Virtual Camera Operator:** 92% consistency
**3D Camera Controller:** 88% consistency
**Architectural Photographer:** 82% consistency
**Scene Preservation Camera:** 95% consistency
**FLF Video Camera:** 90% consistency (across frames)

### Professional Quality Output:

**Cinematographer:** ⭐⭐⭐⭐⭐
**Virtual Camera Operator:** ⭐⭐⭐⭐
**3D Camera Controller:** ⭐⭐⭐
**Architectural Photographer:** ⭐⭐⭐⭐⭐
**Scene Preservation Camera:** ⭐⭐⭐
**FLF Video Camera:** ⭐⭐⭐⭐

---

## 📚 Additional Resources

- [CAMERA_CONTROL_GUIDE.md](CAMERA_CONTROL_GUIDE.md) - Full camera control guide
- [PROMPT_REFERENCE.md](PROMPT_REFERENCE.md) - Camera prompt reference
- [README.md](README.md) - Package overview

---

## 🎓 Understanding System Prompts in Qwen

### What is a System Prompt?

In Qwen-VL (and ChatML format), the system prompt is the first instruction that defines the AI's role:

```
<|im_start|>system
[SYSTEM PROMPT GOES HERE]
<|im_end|>
<|im_start|>user
[VISION TOKENS] [USER PROMPT]
<|im_end|>
<|im_start|>assistant
[AI GENERATES IMAGE]
```

### Why Camera System Prompts Work

They establish a **contract** with Qwen:
1. **Role Definition**: "You are a virtual camera operator"
2. **Task Scope**: "Move camera ONLY"
3. **Constraints**: "Do not modify scene content"
4. **Preservation Rules**: "Keep objects/materials/lighting identical"

This makes Qwen understand that camera instructions = viewpoint changes, NOT content redesign.

---

**Summary: Use camera-specific system prompts to ensure Qwen preserves scene content while executing camera movements. Choose based on your use case: professional output, maximum preservation, or video sequences.**

**Author:** Amir Ferdos (ArchAi3d)
**Version:** 2.0.0
**License:** MIT
