# ArchAi3D Qwen Object Rotation V2 - Complete Guide

## 🎉 What's New in V2

The Object Rotation V2 node is a **massive upgrade** with professional cinematography features, making it easier than ever to create stunning 360° views, product turntables, and architectural walkarounds.

### 🚀 Major New Features:

1. **9 Cinematography Presets** - One-click professional results
2. **Orbit Distance Control** - Close/medium/wide orbit paths
3. **8 Subject Type Presets** - Optimized for different subjects
4. **5 New Angle Presets** - 30°, 60°, 120°, 135°, 270° added
5. **Speed/Transition Hints** - Smooth, slow, fast, cinematic
6. **Elevation Control** - Spiral up/down during orbit
7. **Frame Count Output** - Better batch processing support
8. **Enhanced Multi-Step** - Up to 24 frames (was 12)

---

## 📦 Cinematography Presets (NEW!)

### What Are Cinematography Presets?

One-click configurations that automatically set ALL parameters for professional results. No need to fiddle with individual settings!

### Available Presets:

#### 1. **Product Turntable** ⭐ E-commerce

**Perfect for:** Product photography, e-commerce listings, Shopify/Amazon

**Auto-configures:**
- 360° rotation
- 8 frames
- Close orbit
- Smooth movement
- Level elevation
- Maintains distance

**Use case:** Create perfect 360° product views for online stores

---

#### 2. **Architectural Walkaround** ⭐ Buildings

**Perfect for:** Building exteriors, real estate, architectural portfolios

**Auto-configures:**
- 360° rotation
- 8 frames
- Wide orbit
- Cinematic movement
- Level elevation
- Maintains distance

**Use case:** Showcase entire building from all angles

---

#### 3. **Reveal Shot** 🎬 Dramatic

**Perfect for:** Dramatic reveals, before/after comparisons

**Auto-configures:**
- 180° rotation
- 4 frames
- Medium orbit
- Slow movement
- Level elevation

**Use case:** Cinematic 180° reveal for presentations

---

#### 4. **Inspection View** 🔍 Detailed

**Perfect for:** Quality control, detailed examination, technical documentation

**Auto-configures:**
- 360° rotation
- 12 frames
- Close orbit
- Smooth movement
- Level elevation

**Use case:** Show every detail with 12-angle coverage

---

#### 5. **Spiral Ascent** 🌀 Rising

**Perfect for:** Dynamic presentations, hero shots, impressive reveals

**Auto-configures:**
- 360° rotation
- 8 frames
- Medium orbit
- Smooth movement
- **Rising elevation** (spirals upward!)

**Use case:** Orbit while rising for dramatic effect

---

#### 6. **Spiral Descent** 🌀 Descending

**Perfect for:** Introductions, establishing shots, scene entry

**Auto-configures:**
- 360° rotation
- 8 frames
- Medium orbit
- Smooth movement
- **Descending elevation** (spirals downward!)

**Use case:** Orbit while descending from bird's eye view

---

#### 7. **Hero Shot** 💫 Wide Cinematic

**Perfect for:** Portfolio pieces, hero images, single stunning frame

**Auto-configures:**
- 180° rotation
- 1 frame (single shot!)
- Wide orbit
- Cinematic movement
- Level elevation

**Use case:** One perfect 180° opposite view

---

#### 8. **Quick Spin** ⚡ Dynamic

**Perfect for:** Social media, attention-grabbing animations, quick previews

**Auto-configures:**
- 360° rotation
- 16 frames (smooth!)
- Medium orbit
- Fast movement
- Level elevation

**Use case:** Fast 360° spin for dynamic videos

---

#### 9. **Interior Walkthrough** 🏠 Rooms

**Perfect for:** Interior design, room exploration, real estate interiors

**Auto-configures:**
- 180° rotation
- 4 frames
- Medium orbit
- Smooth movement
- Level elevation

**Use case:** Show room from multiple angles

---

## 🎯 Subject Type Presets (NEW!)

### What Are Subject Type Presets?

Optimized subject text and hints for different types of objects/scenes.

### Available Types:

| Subject Type | Generated Text | Optimized For | Orbit Hint |
|--------------|----------------|---------------|------------|
| **Generic** | "SUBJECT" | Any object | - |
| **Product** | "the product" | E-commerce items | "showcasing all angles" |
| **Building** | "the building" | Architecture | "exterior view" |
| **Furniture** | "the furniture" | Interior design | "showing details" |
| **Vehicle** | "the vehicle" | Cars, trucks, etc. | "displaying all sides" |
| **Character** | "the character" | People (⚠️ centered!) | "centered in frame" |
| **Room** | "the room" | Interior spaces | "interior view" |
| **Custom** | Your text | Anything | - |

### How to Use:

1. Select subject type from dropdown
2. Node automatically uses optimized text
3. If "Custom" selected, use "custom_subject" field

**Example:**
```
Subject Type: "product"
→ Generates: "camera orbit right around the product by 90 degrees showcasing all angles"
```

---

## 🌍 Orbit Distance Control (NEW!)

### What Is Orbit Distance?

Controls how close the camera orbits to the subject.

### Options:

#### 1. **Close**
- **Distance:** Near to subject
- **Best for:** Product details, closeup inspection, texture visibility
- **Prompt:** "close orbit around..."
- **Example:** Jewelry, small products, detailed examination

#### 2. **Medium** (Default)
- **Distance:** Standard orbit
- **Best for:** Furniture, medium objects, general use
- **Prompt:** "orbit around..." (no modifier)
- **Example:** Chairs, lamps, medium-sized objects

#### 3. **Wide**
- **Distance:** Far from subject
- **Best for:** Buildings, large objects, establishing shots
- **Prompt:** "wide orbit around..."
- **Example:** Architecture, vehicles, rooms

### Visual Comparison:

```
Close:   ⟲ 🪑  (tight orbit, see details)
Medium:  ⟲  🪑  (standard orbit)
Wide:    ⟲   🪑  (wide orbit, see context)
```

---

## 🎬 Speed/Transition Hints (NEW!)

### What Are Speed Hints?

Instructions to Qwen about movement quality. Helps with video smoothness and style.

### Options:

| Speed Hint | Effect | Best For | Prompt Addition |
|-----------|--------|----------|-----------------|
| **None** | No modifier | Quick tests | - |
| **Smooth** | Smooth motion | Product videos | "smooth camera movement" |
| **Slow** | Deliberate pace | Luxury items | "slow steady movement" |
| **Fast** | Quick rotation | Social media | "quick movement" |
| **Cinematic** | Professional quality | Portfolios | "cinematic slow movement" |

### When to Use Each:

**Smooth:** Default for most professional work
**Slow:** Luxury products, high-end presentations
**Fast:** Social media, attention-grabbing content
**Cinematic:** Portfolio pieces, impressive presentations

---

## 🏔️ Elevation During Orbit (NEW!)

### What Is Elevation Control?

Allows camera to change height WHILE orbiting - creates spiral/helical paths.

### Options:

#### 1. **Level** (Default)
- Camera stays at same height
- Traditional circular orbit
- Most predictable results

#### 2. **Rising**
- Camera rises while orbiting
- Spiral upward effect
- Dramatic reveals

#### 3. **Descending**
- Camera descends while orbiting
- Spiral downward effect
- Scene introductions

#### 4. **Eye to Bird**
- Start at eye level
- End at bird's eye view
- Progressive elevation change

#### 5. **Bird to Eye**
- Start at bird's eye view
- End at eye level
- Descending reveal

### Visual Representation:

```
Level:      ⟲→⟲→⟲→⟲     (flat circular orbit)
Rising:     ⟲↗⟲↗⟲↗⟲     (spiral upward)
Descending: ⟲↘⟲↘⟲↘⟲     (spiral downward)
```

---

## 📐 Extended Angle Presets (NEW!)

### New Angles Added in V2:

| Angle | Rotation | Use Case |
|-------|----------|----------|
| **30°** | Small step | Subtle changes, many frames |
| **60°** | Between 45/90 | 6-frame sequences |
| **120°** | Third turn | 3-frame sequences |
| **135°** | Diagonal | Opposite diagonal view |
| **270°** | Three-quarters | Almost full rotation |

### All Available Angles:

30°, 45°, 60°, 90°, 120°, 135°, 180°, 270°, 360°, Custom

### Angle Selection Guide:

**For 360° video with N frames:**
- 4 frames: 90° per step
- 6 frames: 60° per step
- 8 frames: 45° per step
- 12 frames: 30° per step
- 16 frames: ~22° (use 360° with 16 steps)

---

## 🎯 Complete Workflow Examples

### Example 1: E-commerce Product Turntable

**Goal:** Create 8-frame 360° product view for Shopify

**Setup:**
```
1. Load product image (centered, white background)
2. Add "ArchAi3D System Prompt" → "Virtual Camera Operator"
3. Add "ArchAi3D Qwen Object Rotation V2":
   - Cinematography Preset: "product_turntable"
   - (Everything else auto-configured!)
4. Connect to encoder
5. Generate 8 frames
```

**Result:** Perfect 360° product turntable with 8 views

**Prompt Generated:**
```
"camera close orbit right around the product by 45 degrees
showcasing all angles maintaining distance keeping camera level smooth camera movement"
```

---

### Example 2: Architectural Building Walkaround

**Goal:** Create impressive building exterior flyaround

**Setup:**
```
1. Load building exterior photo
2. System Prompt: "Cinematographer"
3. Object Rotation V2:
   - Cinematography Preset: "architectural_walkaround"
   - Scene Context: "modern office building exterior view golden hour lighting"
4. Generate 8 frames
```

**Result:** Professional 360° architectural visualization

---

### Example 3: Spiral Product Reveal

**Goal:** Dramatic spiral ascent for luxury product

**Setup:**
```
1. Load luxury product image
2. System Prompt: "Cinematographer"
3. Object Rotation V2:
   - Cinematography Preset: "spiral_ascent"
   - Subject Type: "product"
   - Scene Context: "luxury product on marble pedestal studio lighting"
4. Generate 8 frames
```

**Result:** Camera orbits while rising - dramatic effect!

---

### Example 4: Furniture Inspection

**Goal:** Show furniture from 12 different angles for catalog

**Setup:**
```
1. Load furniture photo
2. System Prompt: "Virtual Camera Operator"
3. Object Rotation V2:
   - Cinematography Preset: "inspection_view"
   - Subject Type: "furniture"
4. Generate 12 frames
```

**Result:** Detailed 360° view with 12 angles (30° steps)

---

### Example 5: Custom Configuration

**Goal:** Manual setup for specific needs

**Setup:**
```
1. Load image
2. System Prompt: "Cinematographer"
3. Object Rotation V2:
   - Cinematography Preset: "custom"
   - Subject Type: "building"
   - Orbit Distance: "wide"
   - Angle Preset: "60"
   - Elevation: "rising"
   - Speed Hint: "cinematic"
   - Multi-Step Mode: true
   - Steps: 6
```

**Result:** 6-frame 360° with 60° steps, wide orbit, rising elevation

---

## 📊 Preset Comparison Table

| Preset | Frames | Angle | Orbit | Speed | Elevation | Best For |
|--------|--------|-------|-------|-------|-----------|----------|
| **Product Turntable** | 8 | 360° | Close | Smooth | Level | E-commerce |
| **Architectural Walkaround** | 8 | 360° | Wide | Cinematic | Level | Buildings |
| **Reveal Shot** | 4 | 180° | Medium | Slow | Level | Drama |
| **Inspection View** | 12 | 360° | Close | Smooth | Level | Details |
| **Spiral Ascent** | 8 | 360° | Medium | Smooth | Rising | Hero shots |
| **Spiral Descent** | 8 | 360° | Medium | Smooth | Descending | Intros |
| **Hero Shot** | 1 | 180° | Wide | Cinematic | Level | Portfolio |
| **Quick Spin** | 16 | 360° | Medium | Fast | Level | Social media |
| **Interior Walkthrough** | 4 | 180° | Medium | Smooth | Level | Rooms |

---

## 💡 Pro Tips

### 1. When to Use Each Preset

**E-commerce:** Product Turntable
**Architecture:** Architectural Walkaround
**Luxury/Premium:** Spiral Ascent + Slow speed
**Social Media:** Quick Spin
**Portfolio:** Hero Shot or Inspection View
**Real Estate:** Interior Walkthrough or Architectural Walkaround

### 2. Combining with System Prompts

**For maximum scene preservation:**
```
System Prompt: "Virtual Camera Operator"
Preset: Any
Result: Identical scene across all frames
```

**For professional quality:**
```
System Prompt: "Cinematographer"
Preset: Any
Result: Professional cinematic output
```

**For video sequences:**
```
System Prompt: "FLF Video Camera"
Preset: Any with multi-step
Result: Smooth frame-to-frame transitions
```

### 3. Subject Centering

For best results:
- **Center your subject** in the frame before generating
- **Environment-only scenes** work better than people
- If people in frame, **keep them centered**

### 4. Frame Count Guidelines

| Use Case | Recommended Frames |
|----------|-------------------|
| Quick preview | 4 frames |
| Standard turntable | 8 frames |
| Smooth video | 12-16 frames |
| Ultra-smooth | 24 frames |
| Single opposite view | 1 frame (Hero Shot) |

### 5. Orbit Distance Selection

**Small objects (< 1m):** Close orbit
**Medium objects (1-3m):** Medium orbit
**Large objects (> 3m):** Wide orbit
**Buildings:** Wide orbit
**Products:** Close or medium

---

## 🆚 V1 vs V2 Comparison

### What's Different:

| Feature | V1 | V2 |
|---------|----|----|
| **Cinematography Presets** | ❌ None | ✅ 9 presets |
| **Orbit Distance** | ❌ Fixed | ✅ Close/Medium/Wide |
| **Subject Types** | ❌ Manual text | ✅ 8 optimized presets |
| **Angle Options** | 5 presets | **10 presets** |
| **Speed Hints** | ❌ None | ✅ 5 options |
| **Elevation Control** | ❌ None | ✅ 5 options |
| **Max Frames** | 12 | **24** |
| **Frame Count Output** | ❌ No | ✅ Yes |
| **Auto-Configuration** | ❌ No | ✅ Yes |

### Should You Use V2?

**Use V2 if you want:**
- One-click presets for instant results
- Professional cinematography features
- Spiral/elevation effects
- More angle options
- Better video sequences (up to 24 frames)

**Use V1 if you want:**
- Simpler interface
- Manual control only
- Legacy workflows

**Recommendation:** V2 for new projects, V1 for existing workflows

---

## 🔧 Technical Details

### Outputs:

1. **rotation_prompt** (STRING)
   - Single rotation prompt
   - Without scene context

2. **full_prompt** (STRING)
   - Rotation prompt + scene context
   - Ready to use with encoder

3. **multi_step_prompts** (STRING)
   - All steps as formatted text
   - "Frame 1/8: [prompt]"
   - "Frame 2/8: [prompt]" etc.

4. **frame_count** (INT) ⭐ NEW!
   - Number of frames
   - Useful for batch processing
   - 1 if multi-step disabled

### Auto-Configuration Logic:

When cinematography preset selected (not "custom"):
- All parameters auto-set
- Manual settings ignored
- Debug mode shows applied preset

When "custom" selected:
- Manual control of all parameters
- No auto-configuration

---

## 📚 Additional Resources

- [CAMERA_CONTROL_GUIDE.md](CAMERA_CONTROL_GUIDE.md) - General camera control guide
- [CAMERA_SYSTEM_PROMPTS.md](CAMERA_SYSTEM_PROMPTS.md) - System prompt guide
- [PROMPT_REFERENCE.md](PROMPT_REFERENCE.md) - Quick prompt reference
- [README.md](README.md) - Package overview

---

## 🎓 Learning Path

### Beginner:
1. Start with **cinematography presets**
2. Try "Product Turntable" or "Architectural Walkaround"
3. Use default system prompt "Cinematographer"
4. Don't change manual settings yet

### Intermediate:
1. Experiment with different **subject types**
2. Try different **orbit distances**
3. Use **scene context** for better results
4. Test different **system prompts**

### Advanced:
1. Switch to "custom" cinematography preset
2. Fine-tune **elevation** settings
3. Experiment with **speed hints**
4. Create custom **multi-step sequences**
5. Combine **spiral movements** with custom subjects

---

**Summary: The V2 node is a massive upgrade with 9 cinematography presets, orbit distance control, subject type optimization, elevation control, and extended angle options - making professional 360° views effortless!**

**Author:** Amir Ferdos (ArchAi3d)
**Version:** 2.0.0
**License:** MIT
