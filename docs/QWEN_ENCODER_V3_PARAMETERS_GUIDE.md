# ArchAi3D Qwen Encoder V3 - Complete Parameters Guide

**Node:** `ArchAi3D_Qwen_Encoder_V3`
**Category:** ArchAi3d/Qwen/Encoders
**Version:** 1.0.0 (Final Release)
**Author:** Amir Ferdos (ArchAi3d)
**Purpose:** Research-validated conditioning control for Qwen-Image-Edit workflows

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Core Inputs](#core-inputs)
3. [Image Inputs](#image-inputs)
4. [Conditioning Control](#conditioning-control)
5. [Manual Override](#manual-override)
6. [Image Labels](#image-labels)
7. [Latent Strength Controls](#latent-strength-controls)
8. [Debug Options](#debug-options)
9. [Outputs](#outputs)
10. [Workflow Examples](#workflow-examples)
11. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### **Minimal Setup:**
1. Connect **clip** (Qwen-VL model)
2. Connect **image1_vl** (your main image)
3. Enter **prompt** (what you want to change)
4. Keep **conditioning_balance** at "Balanced"
5. Connect **recommended_cfg** to KSampler's cfg parameter

### **Recommended Setup:**
1. Connect **clip** + **vae**
2. Connect **image1_vl** and **image1_latent** (same image)
3. Add **system_prompt** for better control
4. Choose **conditioning_balance** preset based on your needs
5. Connect **recommended_cfg** to KSampler

---

## 🎯 Core Inputs

### **1. clip** (REQUIRED)
- **Type:** CLIP model
- **Purpose:** Qwen-VL CLIP model for encoding text and vision tokens
- **What it does:** Converts your images and text into embeddings that the diffusion model understands

**Where to get it:**
```
Load Checkpoint → clip output
```

**Technical detail:** This must be a Qwen-VL compatible CLIP model (not Stable Diffusion CLIP).

---

### **2. prompt** (REQUIRED)
- **Type:** String (multiline)
- **Purpose:** Your text instructions for image editing
- **What it does:** Tells Qwen what changes you want to make to the image

**Examples:**
```
"change the wall color to blue"

"add a modern sofa in the center of the room"

"remove the person from the image"

"change the time of day to sunset with warm lighting"
```

**Tips:**
- Be specific about what you want
- Mention which image if using multiple (e.g., "change the wall in Image 1 to blue")
- Include preservation clauses if needed (e.g., "keep everything else identical")

**Technical detail:** Vision tokens are automatically inserted into this prompt in ChatML format.

---

### **3. vae** (OPTIONAL - Recommended)
- **Type:** VAE model
- **Purpose:** Encodes reference latents for better image consistency
- **What it does:** Converts your reference images into latent representations that guide the generation

**Where to get it:**
```
Load Checkpoint → vae output
```

**When to use:**
- ✅ Always recommended for best results
- ✅ Required if you want to use image1_latent, image2_latent, or image3_latent
- ❌ Can skip if only using vision encoder (vl images only)

**Technical detail:** Reference latents are attached to conditioning metadata and help preserve image structure.

---

## 🖼️ Image Inputs

### **Two Parallel Image Paths:**

The V3 encoder has **6 image inputs** organized into **2 groups of 3**:

```
GROUP 1: Vision Encoder Path (VL Images)
├─ image1_vl
├─ image2_vl
└─ image3_vl

GROUP 2: VAE Encoder Path (Latent Images)
├─ image1_latent
├─ image2_latent
└─ image3_latent
```

---

### **GROUP 1: Vision Encoder Path (VL Images)**

#### **image1_vl** (OPTIONAL)
- **Type:** Image
- **Purpose:** Image 1 for Qwen-VL vision encoder
- **What it does:** Creates vision embeddings that represent the visual content

**Typical use:**
- Main image to edit
- Scene context
- First reference image

**Format:**
- RGB only (alpha channel automatically removed)
- Expected size: 512×512 px or as per your workflow

---

#### **image2_vl** (OPTIONAL)
- **Type:** Image
- **Purpose:** Image 2 for Qwen-VL vision encoder
- **What it does:** Provides additional visual context

**Typical use:**
- Position guide (with numbered rectangles)
- Second reference image
- Style reference

**Position Guide Workflow:**
```
image1_vl: Main scene to edit
image2_vl: Position guide with numbered rectangles
prompt: "using the second image as a position reference guide..."
```

---

#### **image3_vl** (OPTIONAL)
- **Type:** Image
- **Purpose:** Image 3 for Qwen-VL vision encoder
- **What it does:** Provides third visual context

**Typical use:**
- Additional reference
- Third style example
- Comparison image

**Note:** Using multiple images increases token count and processing time.

---

### **GROUP 2: VAE Encoder Path (Latent Images)**

#### **image1_latent** (OPTIONAL - Recommended)
- **Type:** Image
- **Purpose:** Image 1 for reference latent encoding
- **What it does:** Creates latent representation that guides structure and composition

**Best practice:**
```
Use the SAME image for both:
- image1_vl (vision encoder)
- image1_latent (VAE encoder)

This gives best results: vision context + latent structure
```

**Why separate inputs?**
- Vision encoder: 512×512 RGB (Qwen-VL expects specific format)
- VAE encoder: Can be any resolution (gets encoded to latent space)
- Sometimes you want different images for vision vs structure

---

#### **image2_latent** (OPTIONAL)
- **Type:** Image
- **Purpose:** Image 2 for reference latent encoding
- **What it does:** Provides additional latent guidance

**Typical use:**
- Second structural reference
- Background consistency
- Multi-image blending

**Latent strength:** Control influence with `image2_latent_strength` parameter

---

#### **image3_latent** (OPTIONAL)
- **Type:** Image
- **Purpose:** Image 3 for reference latent encoding
- **What it does:** Provides third latent guidance

**Typical use:**
- Third structural reference
- Complex multi-image workflows

**Latent strength:** Control influence with `image3_latent_strength` parameter

---

### **📌 Image Input Summary:**

| Input | Goes To | Purpose | Typical Use |
|-------|---------|---------|-------------|
| `image1_vl` | Vision Encoder | Visual understanding | Main scene |
| `image2_vl` | Vision Encoder | Additional context | Position guide |
| `image3_vl` | Vision Encoder | More context | Extra reference |
| `image1_latent` | VAE Encoder | Structure guidance | Same as image1_vl |
| `image2_latent` | VAE Encoder | Additional structure | Background ref |
| `image3_latent` | VAE Encoder | More structure | Extra structure |

---

## 🎚️ Conditioning Control

### **system_prompt** (OPTIONAL - Recommended)
- **Type:** String (multiline)
- **Purpose:** Background instructions wrapped in ChatML system block
- **What it does:** Provides context and rules that guide ALL operations

**When to use:**
- ✅ Position guide workflows (explain the two-image system)
- ✅ Complex editing tasks (set preservation rules)
- ✅ Specialized workflows (define expert behavior)

**Examples:**

**Position Guide System Prompt:**
```
You are an expert image compositor. You receive two inputs: Image 1 (scene to edit) and Image 2 (numbered position guide with red rectangles).

Step 1: Read the number in each rectangle in Image 2 and find its mapping in the prompt.

Step 2: Add the specified object to Image 1 at each rectangle's position.

Step 3: Remove all red rectangles and all numbers from the final image - these are temporary reference guides and must not be visible in the output.

Maintain all original elements in Image 1 unchanged.
```

**Preservation System Prompt:**
```
You are a professional image editor. Make only the changes described in the prompt. Keep all other elements in the image EXACTLY as they are: lighting, colors, furniture, walls, textures, and spatial relationships. Maintain photorealistic quality.
```

**Technical detail:** Wrapped as `<|im_start|>system\n{system_prompt}\n<|im_end|>` in ChatML format.

---

### **conditioning_balance** (REQUIRED)
- **Type:** Dropdown
- **Purpose:** Choose image/text balance preset
- **What it does:** Sets context_strength, user_strength, and recommended CFG

**Options:**

#### **1. Image-Dominant** ⭐
```yaml
context_strength: 0.2
user_strength: 0.1
recommended_cfg: 2.5
```

**Use when:**
- You want to preserve the image EXACTLY
- Text should have minimal influence
- Precise material matching
- Exact reproduction

**Example:** "Change countertop to marble" - keeps everything else pixel-perfect

---

#### **2. Image-Priority**
```yaml
context_strength: 0.5
user_strength: 0.3
recommended_cfg: 3.0
```

**Use when:**
- Images should guide the result
- Text provides small hints
- Style transfer with minor tweaks

**Example:** "Add warm lighting" - subtle changes, image-driven

---

#### **3. Balanced** (DEFAULT)
```yaml
context_strength: 1.0
user_strength: 1.0
recommended_cfg: 4.0
```

**Use when:**
- Equal weight to images and text
- General editing tasks
- Qwen's default behavior

**Example:** "Add a sofa and change wall color" - balanced image/text influence

---

#### **4. Text-Priority**
```yaml
context_strength: 1.3
user_strength: 1.3
recommended_cfg: 4.75
```

**Use when:**
- Text instructions should guide
- Images provide context
- Creative reinterpretation

**Example:** "Transform to cyberpunk style" - text-driven creativity

---

#### **5. Text-Dominant**
```yaml
context_strength: 1.5
user_strength: 1.5
recommended_cfg: 5.5
```

**Use when:**
- Follow text instructions closely
- Images as loose reference only
- Dramatic changes
- Text-to-image with image hints

**Example:** "Create a completely different scene inspired by this" - text dominates

---

#### **6. Custom** (Advanced)
```yaml
context_strength: manual_context_strength (you set)
user_strength: manual_user_strength (you set)
recommended_cfg: calculated from your values
```

**Use when:**
- You know exactly what you're doing
- Need fine-tuning beyond presets
- Experimenting with specific values

**Reveals:** `manual_context_strength` and `manual_user_strength` sliders

---

### **📊 Preset Comparison:**

| Preset | Image Influence | Text Influence | CFG | Use Case |
|--------|----------------|----------------|-----|----------|
| Image-Dominant | ████████░░ 80% | ██░░░░░░░░ 20% | 2.5 | Exact reproduction |
| Image-Priority | ██████░░░░ 60% | ████░░░░░░ 40% | 3.0 | Image-guided tweaks |
| Balanced | █████░░░░░ 50% | █████░░░░░ 50% | 4.0 | General editing |
| Text-Priority | ████░░░░░░ 40% | ██████░░░░ 60% | 4.75 | Text-guided creation |
| Text-Dominant | ██░░░░░░░░ 20% | ████████░░ 80% | 5.5 | Creative freedom |

---

## ⚙️ Manual Override (Advanced)

**Only visible when `conditioning_balance = "Custom"`**

### **manual_context_strength**
- **Type:** Float slider
- **Range:** 0.0 to 3.0 (Extended range for extreme conditioning control)
- **Default:** 1.0
- **Step:** 0.01

**What it controls:**
- System prompt influence
- Image label influence
- Background context strength

**How it works:**
```python
# Stage A interpolation: vision → vision+context
cond_context = vision + context_strength * (vision_with_context - vision)
```

**Values:**

| Value | Effect | Description |
|-------|--------|-------------|
| 0.0 | No context | Pure vision-only, ignores system prompt and labels |
| 0.2 | Minimal | 20% system prompt influence (Image-Dominant preset) |
| 0.5 | Half | 50% blend between vision and context |
| 1.0 | Full | Normal system prompt and label influence (default) |
| 1.3 | Enhanced | Over-emphasizes context (Text-Priority preset) |
| 1.5 | Maximum | Strongest context influence (Text-Dominant preset) |

**When to adjust:**
- **Decrease (0.1-0.5):** System prompt is too controlling
- **Increase (1.1-1.5):** System prompt isn't having enough effect

---

### **manual_user_strength**
- **Type:** Float slider
- **Range:** 0.0 to 3.0 (Extended range for extreme conditioning control)
- **Default:** 1.0
- **Step:** 0.01

**What it controls:**
- Your text prompt influence
- How much user instructions matter
- Creative freedom vs image fidelity

**How it works:**
```python
# Stage B interpolation: context → full
conditioning = context + user_strength * (full_prompt - context)
```

**Values:**

| Value | Effect | Description |
|-------|--------|-------------|
| 0.0 | Ignore text | Your prompt is completely ignored |
| 0.1 | Minimal | 10% user text influence (Image-Dominant preset) |
| 0.5 | Half | 50% blend between context and full prompt |
| 1.0 | Full | Normal user text influence (default) |
| 1.3 | Enhanced | Over-emphasizes your text (Text-Priority preset) |
| 1.5 | Maximum | Strongest text influence (Text-Dominant preset) |

**When to adjust:**
- **Decrease (0.1-0.5):** Need exact image reproduction, minimal text changes
- **Increase (1.1-1.5):** Text prompt is being ignored, need more creative freedom

---

### **🔬 How Context & User Strength Work Together:**

```
                    Two-Stage Interpolation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Stage A: manual_context_strength
┌─────────────┐                    ┌─────────────────┐
│   Vision    │  context_strength  │ Vision+Context  │
│    Only     │ ─────────────────> │  (sys+labels)   │
│  (images)   │     (0.0-3.0)      │                 │
└─────────────┘                    └─────────────────┘

Stage B: manual_user_strength
┌─────────────────┐                ┌─────────────────┐
│ Vision+Context  │  user_strength │   Full Prompt   │
│  (sys+labels)   │ ─────────────> │ (sys+labels+    │
│                 │   (0.0-3.0)    │  user text)     │
└─────────────────┘                └─────────────────┘
                                           │
                                           v
                                     Final Output
```

**Example with Custom values:**
```python
manual_context_strength = 0.3
manual_user_strength = 0.2

# Result:
# - 30% system prompt influence
# - 20% user text influence
# - 70% vision-driven result
# → Perfect for "keep image exact, tiny text changes"
```

---

## 🏷️ Image Labels

### **image1_label**
- **Type:** String
- **Default:** "Image 1"
- **Purpose:** Custom label for Image 1 in vision tokens

**What it does:**
Adds a text label before the vision tokens to help Qwen identify which image is which.

**Default format:**
```
Image 1: <|vision_start|><|image_pad|><|vision_end|>
```

**Custom example:**
```
image1_label = "Main Scene"

Output:
Main Scene: <|vision_start|><|image_pad|><|vision_end|>
```

**When to customize:**
- Position guide workflows: "Image 1" vs "Image 2"
- Descriptive labels: "Main Scene", "Position Guide", "Style Reference"
- Multi-image editing: "Before", "After", "Reference"

---

### **image2_label**
- **Type:** String
- **Default:** "Image 2"
- **Purpose:** Custom label for Image 2 in vision tokens

**Typical custom values:**
```
"Image 2" (default)
"Position Guide"
"Reference Image"
"Style Example"
"Background"
```

---

### **image3_label**
- **Type:** String
- **Default:** "Image 3"
- **Purpose:** Custom label for Image 3 in vision tokens

**Typical custom values:**
```
"Image 3" (default)
"Additional Reference"
"Third Example"
"Comparison"
```

---

### **📌 Label Usage Example:**

**Position Guide Workflow:**
```yaml
image1_vl: main_scene.png
image1_label: "Image 1"

image2_vl: position_guide.png
image2_label: "Image 2"

system_prompt: |
  You receive Image 1 (scene to edit) and Image 2
  (position guide with numbered rectangles)...

prompt: |
  using the second image as a position reference guide,
  add objects to the first image...
```

**Result in ChatML:**
```
<|im_start|>user
Image 1: <|vision_start|><|image_pad|><|vision_end|>
Image 2: <|vision_start|><|image_pad|><|vision_end|>

using the second image as a position reference guide...
<|im_end|>
```

---

## 💪 Latent Strength Controls

### **image1_latent_strength**
- **Type:** Float slider
- **Range:** 0.0 to 2.0
- **Default:** 1.0
- **Step:** 0.01

**What it controls:**
- Influence of Image 1's latent representation
- How much Image 1's structure guides the generation

**How it works:**
```python
ref_latent = vae.encode(image1_latent)
ref_latent = ref_latent * image1_latent_strength
```

**Values:**

| Value | Effect | Description |
|-------|--------|-------------|
| 0.0 | No influence | Image 1 latent is completely ignored |
| 0.5 | Half strength | 50% of normal latent influence |
| 1.0 | Normal | Full latent influence (default) |
| 1.5 | Enhanced | 150% latent influence (stronger structure) |
| 2.0 | Maximum | 200% latent influence (very strong structure) |

**When to adjust:**
- **Decrease (0.3-0.7):** Image 1 structure is too dominant
- **Increase (1.2-2.0):** Need stronger structural guidance from Image 1

---

### **image2_latent_strength**
- **Type:** Float slider
- **Range:** 0.0 to 2.0
- **Default:** 1.0
- **Step:** 0.01

**What it controls:**
- Influence of Image 2's latent representation

**Typical use cases:**
```yaml
1.0: Normal influence (balanced with Image 1)
0.5: Secondary reference (weaker than Image 1)
0.3: Subtle hint (minimal influence)
0.0: Disable Image 2 latent (keep only for vision)
```

---

### **image3_latent_strength**
- **Type:** Float slider
- **Range:** 0.0 to 2.0
- **Default:** 1.0
- **Step:** 0.01

**What it controls:**
- Influence of Image 3's latent representation

**Typical use cases:**
```yaml
1.0: Normal influence
0.5: Tertiary reference
0.2: Very subtle hint
0.0: Disable Image 3 latent
```

---

### **📊 Latent Strength Strategies:**

**Strategy 1: Single Image Focus**
```yaml
image1_latent_strength: 1.0   # Main structure
image2_latent_strength: 0.0   # Disabled
image3_latent_strength: 0.0   # Disabled
```

**Strategy 2: Primary + Secondary**
```yaml
image1_latent_strength: 1.0   # Main structure
image2_latent_strength: 0.5   # Supporting structure
image3_latent_strength: 0.0   # Disabled
```

**Strategy 3: Balanced Multi-Image**
```yaml
image1_latent_strength: 1.0   # Equal
image2_latent_strength: 1.0   # Equal
image3_latent_strength: 1.0   # Equal
```

**Strategy 4: Enhanced Primary**
```yaml
image1_latent_strength: 1.5   # Strong structure
image2_latent_strength: 0.3   # Subtle hint
image3_latent_strength: 0.0   # Disabled
```

---

## 🐛 Debug Options

### **debug_mode**
- **Type:** Boolean checkbox
- **Default:** False (unchecked)

**What it does:**
Prints detailed information to the console about:
- Conditioning balance preset values
- Strength parameters (context, user, latent)
- Recommended CFG scale
- Two-stage interpolation process
- Conditioning shape information
- Active inputs summary

**Console Output Example:**
```
======================================================================
ArchAi3D Qwen Encoder V3 (ENHANCED) - Debug Output
======================================================================
Conditioning Balance Preset: Image-Dominant
  • Preset context_strength: 0.200
  • Preset user_strength: 0.100
  • Preset recommended_cfg: 2.50
  • Description: Follow images EXACTLY, ignore most text

⭐ CFG SCALE RECOMMENDATION:
  • Connect 'recommended_cfg' output to KSampler's cfg parameter
  • Current value: 2.50
  • Range: 2.5 (strong image) → 5.5 (strong text)
  • This is the PRIMARY control method (research-validated)

Two-Stage Interpolation:
  • Stage A (context): vision → vision+context | strength=0.200
  • Stage B (user): context → full | strength=0.100

  • Δ_context = 12.450 (vision vs no-user)
  • Δ_user = 23.789 (no-user vs full)

Latent Strengths:
  • Image1: 1.000 (default)

Active Inputs:
  • User Text: 45 chars
  • System Prompt: 234 chars
  • VL Image1: 512×512 px

Output Shapes:
  • Conditioning: torch.Size([1, 77, 768])
  • Latent: torch.Size([1, 4, 64, 64])
======================================================================
```

**When to enable:**
- ✅ Troubleshooting workflow issues
- ✅ Understanding conditioning behavior
- ✅ Verifying preset values
- ✅ Checking interpolation deltas
- ✅ Confirming image inputs are correct

**When to disable:**
- ❌ Production workflows (cleaner console)
- ❌ Batch processing (too much output)

---

## 📤 Outputs

### **1. conditioning**
- **Type:** CONDITIONING
- **Purpose:** Text+vision embeddings with reference latents attached

**What it contains:**
- Text embeddings from your prompt
- Vision embeddings from VL images
- Interpolated conditioning based on preset
- Reference latents metadata (if VAE provided)

**Connect to:**
```
KSampler → positive
ConditioningAverage → conditioning_to/conditioning_from
ConditioningCombine → conditioning_1
```

**Technical detail:**
- Shape: `[batch, tokens, embedding_dim]`
- Contains metadata: `{"reference_latents": [latent1, latent2, ...]}`

---

### **2. latent**
- **Type:** LATENT
- **Purpose:** Image 1 latent in standard ComfyUI format

**What it contains:**
- VAE-encoded representation of `image1_latent`
- Scaled by `image1_latent_strength`
- Standard format: `{"samples": tensor}`

**Connect to:**
```
VAEDecode → samples (to preview Image 1)
KSampler → latent_image (if doing img2img)
LatentUpscale → samples
```

**Note:** Only contains Image 1. Images 2 and 3 are in conditioning metadata.

**Technical detail:**
- Shape: `[batch, channels, height, width]`
- Typical: `[1, 4, 64, 64]` for 512×512 image

---

### **3. formatted_prompt**
- **Type:** STRING
- **Purpose:** Final ChatML-formatted prompt with vision tokens

**What it contains:**
Complete prompt in Qwen ChatML format, including:
- System prompt block (if provided)
- User block with vision tokens and labels
- Assistant start token

**Example output:**
```
<|im_start|>system
You are an expert image compositor...
<|im_end|>
<|im_start|>user
Image 1: <|vision_start|><|image_pad|><|vision_end|>
Image 2: <|vision_start|><|image_pad|><|vision_end|>

using the second image as a position reference guide...
<|im_end|>
<|im_start|>assistant
```

**Connect to:**
```
ShowText → text (to preview formatted prompt)
SaveText → text (to save prompt for debugging)
```

**Use for:**
- Debugging ChatML format
- Verifying vision token placement
- Checking label formatting

---

### **4. recommended_cfg** ⭐ NEW
- **Type:** FLOAT
- **Purpose:** Recommended CFG scale based on preset

**What it contains:**
Single float value representing optimal CFG (Classifier-Free Guidance) scale for the selected preset.

**Values by preset:**

| Preset | CFG Value | Meaning |
|--------|-----------|---------|
| Image-Dominant | 2.5 | Strong image fidelity |
| Image-Priority | 3.0 | Image-guided with hints |
| Balanced | 4.0 | Equal image/text weight |
| Text-Priority | 4.75 | Text-guided creation |
| Text-Dominant | 5.5 | Strong text guidance |
| Custom | Calculated | Based on your manual strengths |

**Connect to:**
```
KSampler → cfg (RECOMMENDED!) ⭐
```

**Why it matters:**
- Research shows CFG is MORE effective than embedding interpolation
- Official Qwen-Image-Edit uses CFG as primary control
- Range 2.5-5.5 is validated for optimal results

**Example workflow:**
```
V3 Encoder (Image-Dominant preset)
├─ conditioning → KSampler positive
└─ recommended_cfg (2.5) → KSampler cfg
```

---

## 🎨 Workflow Examples

### **Example 1: Simple Image Edit (Beginner)**

**Goal:** Change wall color to blue in a room photo

**Setup:**
```yaml
Inputs:
  clip: Qwen-VL CLIP
  vae: Qwen VAE
  image1_vl: room.png
  image1_latent: room.png (same image)
  prompt: "change the wall color to blue, keep everything else identical"
  conditioning_balance: "Balanced"

Connections:
  conditioning → KSampler positive
  recommended_cfg → KSampler cfg
  latent → VAEDecode samples (to preview)
```

**Result:** Clean wall color change with everything else preserved.

---

### **Example 2: Position Guide Workflow (Intermediate)**

**Goal:** Add multiple objects to scene using position guide

**Setup:**
```yaml
Inputs:
  clip: Qwen-VL CLIP
  vae: Qwen VAE

  image1_vl: empty_room.png
  image1_label: "Image 1"
  image1_latent: empty_room.png

  image2_vl: position_guide.png (red rectangles with numbers)
  image2_label: "Image 2"

  system_prompt: |
    You are an expert image compositor. You receive two inputs:
    Image 1 (scene to edit) and Image 2 (numbered position guide
    with red rectangles).

    Step 1: Read the number in each rectangle in Image 2 and find
    its mapping in the prompt.

    Step 2: Add the specified object to Image 1 at each rectangle's
    position.

    Step 3: Remove all red rectangles and all numbers from the
    final image - these are temporary reference guides and must
    not be visible in the output.

    Maintain all original elements in Image 1 unchanged.

  prompt: |
    using the second image as a position reference guide, the red
    rectangles are numbered, add objects to the first image according
    to this mapping: rectangle 1 = modern gray sofa, rectangle 2 =
    wooden coffee table, rectangle 3 = floor lamp, remove all red
    rectangles from the image, remove all numbers from the image,
    keep everything else in the first image identical

  conditioning_balance: "Balanced"

Connections:
  conditioning → KSampler positive
  recommended_cfg → KSampler cfg
```

**Result:** Objects placed exactly where rectangles were, no visible guides.

---

### **Example 3: Exact Material Match (Advanced - Image-Dominant)**

**Goal:** Match countertop material exactly from reference

**Setup:**
```yaml
Inputs:
  clip: Qwen-VL CLIP
  vae: Qwen VAE

  image1_vl: kitchen.png
  image1_latent: kitchen.png
  image1_latent_strength: 1.5  # Strong structure preservation

  image2_vl: marble_reference.png
  image2_latent: marble_reference.png
  image2_latent_strength: 0.5  # Supporting reference

  prompt: "change the kitchen countertop material to match the marble in Image 2, keep everything else identical"

  conditioning_balance: "Image-Dominant"  # Preserve image exactly

  debug_mode: true  # Check interpolation

Connections:
  conditioning → KSampler positive
  recommended_cfg (2.5) → KSampler cfg
```

**Result:** Precise material match with minimal changes to rest of scene.

---

### **Example 4: Creative Reinterpretation (Advanced - Text-Dominant)**

**Goal:** Dramatic style change with image as loose reference

**Setup:**
```yaml
Inputs:
  clip: Qwen-VL CLIP
  vae: Qwen VAE

  image1_vl: modern_room.png
  image1_latent: modern_room.png
  image1_latent_strength: 0.5  # Weak structure (allow changes)

  prompt: |
    transform this modern room into a cyberpunk aesthetic with
    neon lighting, holographic displays, futuristic furniture,
    dark color palette with blue and purple accents, maintain
    the basic room layout

  conditioning_balance: "Text-Dominant"  # Follow text closely

  debug_mode: true

Connections:
  conditioning → KSampler positive
  recommended_cfg (5.5) → KSampler cfg
```

**Result:** Creative cyberpunk transformation with original room as reference.

---

### **Example 5: ConditioningAverage Blending (Expert)**

**Goal:** Custom balance between two extreme presets

**Setup:**
```yaml
# Node 1: Image-Dominant
V3 Encoder #1:
  conditioning_balance: "Image-Dominant"
  → conditioning output

# Node 2: Text-Priority
V3 Encoder #2:
  conditioning_balance: "Text-Priority"
  → conditioning output

# Blend them
ConditioningAverage:
  conditioning_to: V3 Encoder #1 conditioning
  conditioning_from: V3 Encoder #2 conditioning
  conditioning_to_strength: 0.7  # 70% Image-Dominant, 30% Text-Priority
  → blended conditioning

# Sample
KSampler:
  positive: ConditioningAverage output
  cfg: 3.5  # Between 2.5 and 4.75
```

**Result:** Custom balance not available in presets (mostly image-driven with some text influence).

---

### **Example 6: Custom Manual Fine-Tuning (Expert)**

**Goal:** Precise control beyond presets

**Setup:**
```yaml
Inputs:
  conditioning_balance: "Custom"

  manual_context_strength: 0.4   # 40% system prompt influence
  manual_user_strength: 0.6      # 60% user text influence

  # This creates a custom profile:
  # - Less context than Balanced (1.0)
  # - More context than Image-Priority (0.5/0.3)
  # - Unique balance for specific use case

  debug_mode: true  # See calculated CFG and deltas

Connections:
  conditioning → KSampler positive
  recommended_cfg (calculated ~3.2) → KSampler cfg
```

**Result:** Fine-tuned balance for specialized workflow.

---

## 🔧 Troubleshooting

### **Problem 1: Text prompt is being ignored**

**Symptoms:**
- Output looks identical to input image
- Text changes don't appear
- Result ignores instructions

**Solutions:**

1. **Check conditioning_balance:**
   ```yaml
   Current: "Image-Dominant" (0.1 user strength)
   Try: "Balanced" (1.0 user strength)
   Or: "Text-Priority" (1.3 user strength)
   ```

2. **Check CFG scale:**
   ```yaml
   Too low CFG (2.5) = ignores text
   Try higher: 4.0-5.5
   ```

3. **Use Custom preset:**
   ```yaml
   conditioning_balance: "Custom"
   manual_user_strength: 1.5  # Maximum text influence
   ```

4. **Verify prompt is connected:**
   - Make sure prompt input has text
   - Check formatted_prompt output to verify

---

### **Problem 2: Image structure is lost**

**Symptoms:**
- Output doesn't resemble input
- Structure changes dramatically
- Unwanted modifications

**Solutions:**

1. **Check conditioning_balance:**
   ```yaml
   Current: "Text-Dominant" (high text influence)
   Try: "Image-Dominant" or "Image-Priority"
   ```

2. **Increase latent strength:**
   ```yaml
   image1_latent_strength: 1.5-2.0  # Stronger structure
   ```

3. **Lower CFG scale:**
   ```yaml
   Current: 5.5 (text-driven)
   Try: 2.5-3.0 (image-driven)
   ```

4. **Verify latent images:**
   - Make sure image1_latent is connected
   - Check VAE is connected

---

### **Problem 3: System prompt not working**

**Symptoms:**
- System instructions ignored
- Position guide workflow fails
- Preservation rules not followed

**Solutions:**

1. **Check context_strength:**
   ```yaml
   Current: "Image-Dominant" (0.2 context)
   Try: "Balanced" (1.0 context)
   Or: Custom with manual_context_strength: 1.3
   ```

2. **Verify system prompt:**
   - Check system_prompt input has text
   - Look at formatted_prompt output
   - Should see `<|im_start|>system\n...`

3. **Enable debug_mode:**
   - See Δ_context value
   - Should be > 0 if system prompt is active

---

### **Problem 4: Multiple images not being used**

**Symptoms:**
- Only Image 1 seems to affect result
- Images 2 and 3 ignored
- No multi-image blending

**Solutions:**

1. **Verify all VL images connected:**
   ```yaml
   image1_vl: ✓
   image2_vl: ✓  # Check this
   image3_vl: ✓  # Check this
   ```

2. **Check latent strengths:**
   ```yaml
   image2_latent_strength: 0.0  # PROBLEM - disabled!

   Try:
   image2_latent_strength: 1.0
   ```

3. **Reference images in prompt:**
   ```yaml
   prompt: "match the style from Image 2"
   # Explicitly mention which image
   ```

4. **Enable debug_mode:**
   - Check "Active Inputs" section
   - Should show all 3 VL images

---

### **Problem 5: Position guide rectangles visible in output**

**Symptoms:**
- Red rectangles appear in final image
- Numbers are visible
- Guide markers not removed

**Solutions:**

1. **Use explicit removal prompt:**
   ```yaml
   prompt: |
     ...add objects to mapping...
     remove all red rectangles from the image,
     remove all numbers from the image,
     keep everything else identical
   ```

2. **Enhance system prompt:**
   ```yaml
   system_prompt: |
     Step 3: Remove all red rectangles and all numbers
     from the final image - these are temporary reference
     guides and must not be visible in the output.
   ```

3. **Use Position Guide Prompt Builder:**
   - Try "standard_explicit" template
   - Or "strong_removal" template

---

### **Problem 6: Output is too different from input**

**Symptoms:**
- Unexpected changes
- Over-creative interpretation
- Not preserving enough

**Solutions:**

1. **Lower all strengths:**
   ```yaml
   conditioning_balance: "Image-Dominant"
   # OR Custom:
   manual_context_strength: 0.2
   manual_user_strength: 0.1
   ```

2. **Increase latent strength:**
   ```yaml
   image1_latent_strength: 2.0  # Maximum structure
   ```

3. **Lower CFG:**
   ```yaml
   recommended_cfg → ignore
   KSampler cfg: 2.0-2.5
   ```

4. **Add preservation clause:**
   ```yaml
   prompt: "...your changes..., keep everything else in the image EXACTLY identical"
   ```

---

### **Problem 7: Not enough visible change**

**Symptoms:**
- Output identical to input
- Subtle changes too subtle
- Need more dramatic effect

**Solutions:**

1. **Increase all strengths:**
   ```yaml
   conditioning_balance: "Text-Dominant"
   # OR Custom:
   manual_context_strength: 1.5
   manual_user_strength: 1.5
   ```

2. **Lower latent strength:**
   ```yaml
   image1_latent_strength: 0.3-0.5  # Allow more changes
   ```

3. **Increase CFG:**
   ```yaml
   recommended_cfg → ignore
   KSampler cfg: 5.5-7.0
   ```

4. **More specific prompt:**
   ```yaml
   prompt: "dramatic sunset lighting with warm orange and
            purple colors, strong shadows, cinematic mood"
   # Be very specific about what you want
   ```

---

## 📚 Quick Reference Card

```
╔══════════════════════════════════════════════════════════════╗
║         ArchAi3D Qwen Encoder V3 - Quick Reference          ║
╚══════════════════════════════════════════════════════════════╝

REQUIRED INPUTS:
  • clip (Qwen-VL CLIP model)
  • prompt (your text instructions)

RECOMMENDED INPUTS:
  • vae (for reference latents)
  • image1_vl + image1_latent (same image for both)
  • system_prompt (for complex workflows)
  • conditioning_balance preset

CRITICAL OUTPUTS:
  • conditioning → KSampler positive
  • recommended_cfg → KSampler cfg ⭐ IMPORTANT

PRESETS (conditioning_balance):
  Image-Dominant  → Exact reproduction    (CFG 2.5)
  Image-Priority  → Image-guided tweaks   (CFG 3.0)
  Balanced        → General editing       (CFG 4.0) [DEFAULT]
  Text-Priority   → Text-guided creation  (CFG 4.75)
  Text-Dominant   → Creative freedom      (CFG 5.5)
  Custom          → Manual fine-tuning

MANUAL OVERRIDE (Custom preset only):
  manual_context_strength: 0.0-3.0 (Extended range!)
    • Controls system prompt + labels influence
    • 0.0 = ignore, 1.0 = normal, 1.5 = strong, 3.0 = extreme

  manual_user_strength: 0.0-3.0 (Extended range!)
    • Controls your text prompt influence
    • 0.0 = ignore, 1.0 = normal, 1.5 = strong, 3.0 = extreme

LATENT STRENGTH:
  image1_latent_strength: 0.0-2.0
    • 0.5 = weak structure, 1.0 = normal, 2.0 = strong

TROUBLESHOOTING:
  Text ignored?     → Increase user_strength or CFG
  Image changed?    → Decrease user_strength or CFG
  System not working? → Increase context_strength
  Too similar?      → Use Text-Priority/Dominant
  Too different?    → Use Image-Priority/Dominant

DEBUG:
  • Enable debug_mode to see all values in console
  • Check formatted_prompt output for ChatML format
  • Verify recommended_cfg is connected to KSampler
```

---

## 📖 Related Documentation

- **Position Guide Workflow:** See `POSITION_GUIDE_WORKFLOW_DISCOVERY.md`
- **Qwen Prompt Guide:** See `QWEN_PROMPT_GUIDE.md`
- **Node Development:** See `ADDING_NODES_TO_COMFYUI.md`

---

## 📝 Version History

**v1.0** - Initial documentation
- Complete parameter explanations
- Workflow examples
- Troubleshooting guide

---

## 📧 Support

**Author:** Amir Ferdos (ArchAi3d)
**Email:** Amir84ferdos@gmail.com
**LinkedIn:** https://www.linkedin.com/in/archai3d/
**GitHub:** https://github.com/amir84ferdos

**License:** MIT - Free to use and modify

---

**🎯 Pro Tip:** Start with "Balanced" preset and adjust based on results. Use `recommended_cfg` output - it's research-validated and more effective than manual tuning!
