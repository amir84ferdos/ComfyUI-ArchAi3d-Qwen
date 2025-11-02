# ArchAi3D Qwen - Professional AI Interior Design Toolkit

**Transform empty rooms into stunning interior designs using AI** 

Custom ComfyUI nodes for Qwen-VL image editing, specialized for architectural visualization and interior design workflows.

---

## 🎯 What This Does

Professional AI-powered interior design with **4 powerful modes**:

1. **Text-to-Design** - Describe your vision, generate the design
2. **Mood Board Design** - Use reference images for style inspiration  
3. **Reference-Based Design** - Control with perspective reference images
4. **Room Cleaning** - Remove construction debris, tools, and clutter before design

Perfect for architects, interior designers, real estate professionals, and AI enthusiasts.

---

## 🚀 Quick Start

### Installation

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/amir84ferdos/ComfyUI-ArchAi3d-Qwen.git
# Restart ComfyUI
```

Or use **ComfyUI Manager**: Search for "ArchAi3d Qwen"

### What You Get

**17 Custom Nodes** (all under `ArchAi3d/Qwen` category):

**Core Encoding Nodes:**
- 🎨 **Qwen Encoder V1** - Standard strength controls
- 🎨 **Qwen Encoder V2** - Advanced interpolation (recommended)
- 🎨 **Qwen Encoder Simple** - Easy-to-use version
- 🎨 **Qwen Encoder Simple V2** - Multi-image direct input (no resizing, up to 3 VL images + 3 latents)
- 📏 **Qwen Image Scale** - Smart aspect ratio scaling (23 presets)
- 💬 **Qwen System Prompt** - Preset prompt loader
- 🏗️ **Room Transform Prompt** - Visual prompt builder with **103+ materials** (user-customizable via YAML)

**Camera Control Nodes (NEW!):**
- 📹 **Qwen Camera View** - Professional camera control for interior/exterior scenes
- 🔄 **Qwen Object Rotation V2** - Orbit around objects for 360° views with 19 cinematography presets
- 👤 **Qwen Person Perspective** - Person/character perspective control with identity preservation
- 📸 **Scene Photographer** - Position camera to frame specific subjects (14 presets)
- 🎬 **Camera View Selector** - Quick selection from 22 professional viewpoints
- 🚶 **Environment Navigator** - Move and rotate through scenes (14 navigation patterns)

**Image Editing Nodes (NEW!):**
- 🎨 **Material Changer** - Interior design material visualization (48 materials across 6 categories)
- 🧹 **Watermark Removal** - Remove watermarks, text, and logos
- 🎨 **Colorization** - Convert B&W to color with era context (9 historical periods)
- ✨ **Style Transfer** - Apply 8 artistic styles to objects (ice, cloud, wooden, fluffy, etc.)

---

## 💎 Professional Workflows

**Ready-to-use workflows for all 4 design modes available on my Patreon!**

👉 **[Get Premium Workflows on Patreon](https://patreon.com/archai3d)**

Your support helps me:
- ✅ Improve and maintain these nodes
- ✅ Create more presets and workflows  
- ✅ Add new features based on feedback
- ✅ Provide better documentation and tutorials

### What's Included on Patreon:
- 📦 **12+ preset workflows** for different interior styles
- 🎯 Fine-tuned parameters for each use case
- 📚 Setup guides and best practices
- 💬 Direct support and feedback
- 🔄 Regular updates with new presets

---

## 🛠️ Key Features

### ⭐ Encoder V2 (Recommended)
- **Two-stage interpolation** for precise control
- Fixes "weight spike" issues with system prompts
- Separate control for context and user text strength
- Per-image latent strength controls

### 📐 Smart Image Scaling
- **23 preferred aspect ratios** optimized for Qwen-VL
- Auto or manual aspect ratio selection
- Pixel-perfect alignment between VL and latent
- Multiple scaling strategies (crop, letterbox, stretch)

### 🎭 System Prompt Presets
- Interior Designer, Architect, Creative Director
- Luxury Designer, Minimalist, Renovation Expert
- Quick preset switching for different styles

### 🏗️ Room Transform Prompt Builder (NEW!)
- **3 workflow modes**: Remove Only, Remove + Paint All, Remove + Paint Selective
- **103+ material presets** loaded from `config/materials.yaml`
  - 32 floor materials (marble, hardwood, concrete, tile, carpet, stone)
  - 36 wall materials (paint, wallpaper, wood, brick, concrete, tile)
  - 35 ceiling materials (paint, architectural, beams, industrial, wood)
- **User-customizable material library** - edit YAML file to add your own materials!
- **Material tags system** for organization (rich_dark, bright_light, low_contrast)
- **Custom material override** for unique specifications
- **System prompt presets** (3 optimized options + existing presets)
- **Quality controls**: preserve lighting/perspective/POV, clean edges, no halos
- **Optimized prompt structure** based on proven patterns
- Perfect for creating empty rooms or complete room transformations

---

## 📋 Roadmap

### ✅ Working Features (Stable)

- ✅ **Text-based interior design** - High quality, stable
- ✅ **Mood board design** - Style transfer working well
- ✅ **Reference image control** - Perspective preservation works
- ✅ **Room cleaning mode** - Removes debris and construction materials
- ✅ **Multi-image support** - Up to 3 images per workflow
- ✅ **Aspect ratio optimization** - 23 QwenVL-optimized presets
- ✅ **ChatML formatting** - Proper Qwen-VL 2.5 integration
- ✅ **Debug tools** - Comprehensive logging and validation
- ✅ **Camera control** - Research-based viewpoint changes
- ✅ **Object rotation** - "Orbit around" technique for 360° views
- ✅ **Person perspective** - Identity-preserving perspective control for people/characters (NEW!)

### 🔧 Under Development

- 🔧 **Weight control refinement** - Fine-tuning prompt vs reference balance
- 🔧 **More preset workflows** - Expanding style library
- 🔧 **Better documentation** - Video tutorials and examples
- 🔧 **Strength presets** - Pre-configured settings for common scenarios
- 🔧 **FLF video generation** - Multi-frame camera sequences for walkthroughs

### 🎯 Planned Features

- 📅 **Style consistency mode** - Match existing room designs
- 📅 **Batch processing** - Process multiple rooms at once
- 📅 **Advanced masking** - Region-specific design control
- 📅 **Material library** - Quick material swapping
- 📅 **Lighting presets** - Pre-configured lighting scenarios
- 📅 **Animated walkthroughs** - Automatic video generation from camera paths

---

## 📖 Basic Usage

### Standard Workflow (Interior Design from Empty Room)

```
1. Load your empty room image
   ↓
2. ArchAi3D Qwen Image Scale
   ├→ Scales for VL encoder
   └→ Scales for latent processing
   ↓
3. ArchAi3D Qwen System Prompt (optional)
   └→ Choose your AI persona
   ↓
4. ArchAi3D Qwen Encoder V2
   ├─ Connect scaled images
   ├─ Add your design prompt
   ├─ Adjust strength controls
   └→ Get conditioning
   ↓
5. Connect to your sampler
   └→ Generate beautiful interior design!
```

### Room Transform Workflow (Empty Room Creation + Redesign)

```
1. Load your under-construction/cluttered room image
   ↓
2. ArchAi3D Qwen Image Scale
   ├→ Scales for VL encoder
   └→ Scales for latent processing
   ↓
3. ArchAi3D Room Transform Prompt
   ├─ Select mode (Remove Only / Remove + Paint All / Remove + Paint Selective)
   ├─ Specify objects to remove (tools/debris/cables/etc)
   ├─ Choose floor material (18+ presets or custom)
   ├─ Choose wall material (18+ presets or custom)
   ├─ Choose ceiling material (18+ presets or custom)
   ├─ Toggle quality controls (preserve lighting/perspective/etc)
   └→ Get optimized prompt
   ↓
4. ArchAi3D Qwen System Prompt (optional)
   └→ Use "Interior Designer" or "Renovation Expert" preset
   ↓
5. ArchAi3D Qwen Encoder V2
   ├─ Connect scaled images
   ├─ Connect prompt from Room Transform Prompt node
   ├─ Adjust strength controls
   └→ Get conditioning
   ↓
6. Connect to your sampler
   └→ Generate clean empty room or fully redesigned space!
```

**For detailed workflows and presets, check my Patreon!**

---

## 📹 Camera Control

### Three Powerful Camera Nodes

The camera control system is based on extensive community research from Reddit r/StableDiffusion, optimized for Qwen Edit 2509.

#### 🎥 ArchAi3D Qwen Camera View
Professional viewpoint control for interior/exterior scenes:
- **6 movement types**: vantage point, tilt, combined movement, FOV, dolly, custom
- **Distance-based positioning**: "10m to the left" (more reliable than degrees)
- **FOV presets**: Normal, wide 100°, ultrawide 180°, fisheye
- **Scene-aware**: Optimized for interior/exterior/environment-only
- **Best for**: Room exploration, architectural walkthroughs, FLF video generation

#### 🔄 ArchAi3D Qwen Object Rotation
Orbit around objects using the proven "orbit around" technique:
- **Most reliable rotation method** (based on community testing)
- **Precise angle control**: 45°, 90°, 180°, 360° or custom
- **Multi-step mode**: Break 360° into multiple steps for better control
- **Subject-aware**: Specify what to orbit around
- **Best for**: Product visualization, 360° turntables, architectural flyarounds

### 📚 Documentation

**Comprehensive guides included:**
- [CAMERA_CONTROL_GUIDE.md](CAMERA_CONTROL_GUIDE.md) - Full guide with examples and workflows
- [PROMPT_REFERENCE.md](PROMPT_REFERENCE.md) - Quick prompt reference with reliability ratings

### Key Insights from Research

✅ **What Works Best:**
- "Orbit around" is THE most reliable term for rotation
- Environment-only scenes (no people) are most predictable
- Distance-based movement ("10m to left") beats arbitrary degree rotations
- "Dolly" is most consistent for zoom operations

⚠️ **Important Notes:**
- Left/right in prompt = picture left/right, NOT subject's perspective
- Model may rotate people instead of camera if they're in frame
- Centered subjects work better than off-center
- Angles may not be exact, but direction is always consistent

### Simple Camera Workflow

```
1. Load your scene image
   ↓
2. ArchAi3D Qwen Camera View
   ├─ Choose movement type (vantage_point recommended)
   ├─ Set direction and distance
   ├─ Add scene description
   └→ Get camera prompt
   ↓
3. Connect to Qwen Encoder
   └→ Generate new viewpoint!
```

### 360° Object Rotation Workflow

```
1. Load your object/building image
   ↓
2. ArchAi3D Qwen Object Rotation
   ├─ Subject: "the building" / "the product"
   ├─ Angle: 360
   ├─ Multi-step: true (4-8 steps)
   └→ Get rotation prompts
   ↓
3. Loop through each step with Qwen Encoder
   └→ Generate 360° turntable!
```

**For complete examples and advanced techniques, see [CAMERA_CONTROL_GUIDE.md](CAMERA_CONTROL_GUIDE.md)**

---

## 👤 Person Perspective Control (NEW!)

### Specialized Node for People/Character Photography

Based on Reddit community research, the **ArchAi3D Qwen Person Perspective** node is specifically designed for changing camera perspectives when photographing **people and characters**.

#### 🎭 Key Difference: Person vs Object

**Person Perspective** (this node):
- Changes the **camera viewing angle** (high/low/side)
- Person stays in same pose, camera moves up/down/around
- **Primary focus: Identity preservation** (keep face/clothes/pose identical)
- Creates psychological effects (vulnerability, power, intimacy)
- **Best for**: Portraits, fashion, character art

**Object Rotation** (separate node):
- **Orbits camera** around an object/building
- Shows different sides of the subject
- **Best for**: Products, buildings, 360° turntables

#### 🎯 6 Perspective Presets

1. **High Angle (Bird's Eye)** - Looking down → vulnerability, intimacy
2. **Low Angle (Worm's Eye)** - Looking up → power, heroic, monumentality
3. **Eye Level Front** - Straight on → balanced, neutral, approachable
4. **Side Profile** - Full side view → silhouette, distance, elegance
5. **Three-Quarter View** - 45° angle → depth with approachability (most versatile)
6. **Dutch Angle** - Tilted camera → tension, drama, artistic flair

#### 🔒 Identity Preservation Levels

- **Strict** (recommended): Keep face, clothes, hairstyle, pose 100% identical
- **Moderate**: Maintain appearance and clothing
- **Loose**: Keep subject recognizable
- **None**: No preservation (not recommended)

#### ⚡ Simple Person Perspective Workflow

```
1. Load your person/character portrait
   ↓
2. ArchAi3D Qwen Image Scale
   ├→ Scale for VL and latent
   ↓
3. ArchAi3D Qwen System Prompt
   └→ Choose "Portrait Photographer" or "Fashion Photographer"
   ↓
4. ArchAi3D Qwen Person Perspective
   ├─ perspective_preset: Choose angle (e.g., low_angle_worms_eye for heroic)
   ├─ identity_preservation: strict (keep everything identical)
   ├─ psychological_effect: power/vulnerability/etc
   ├─ scene_context: Add environment description
   └→ Get perspective prompt
   ↓
5. Connect to Qwen Encoder
   └→ Generate new perspective while preserving identity!
```

#### 💡 Best Practices

- ✅ **Always use "strict" identity preservation** for consistent results
- ✅ **Keep subject centered in frame** for best results
- ✅ **Use person-focused system prompts** (Portrait/Fashion Photographer)
- ✅ **Match psychological effect to angle** (high=vulnerable, low=powerful)
- ✅ **Enable background/lighting adaptation** for natural results

#### 📚 Complete Guide

For full details, examples, and advanced techniques, see:
- **[PERSON_PERSPECTIVE_GUIDE.md](PERSON_PERSPECTIVE_GUIDE.md)** - Complete guide with all 6 presets, workflows, and troubleshooting

**Perfect for**: Portrait photography, fashion shoots, character concept art, editorial photography, heroic poses, emotional storytelling through camera angles

---

## 🆕 New Advanced Camera & Editing Suite (7 Nodes!)

Based on comprehensive research from 7+ sources including official Qwen documentation, Reddit community findings, and technical papers, we've created a complete professional suite:

### 📸 Scene Photographer (ArchAi3D_Qwen_Scene_Photographer)
**Position camera to frame specific subjects with natural language**

- **14 professional presets**: Product (front, hero low, overhead), Interior (corner, opposite wall, ceiling), Architectural (ground up, elevated), Food (45°, overhead), Fashion, Landscape
- **Natural language positioning**: "3m to the right facing the sofa" (NO pixel coordinates)
- **Full control**: Direction, distance (1-20m), height (ground/lower/same/higher/face level), tilt
- **Auto-facing mode**: Automatically face your target subject
- **Perfect for**: "Go in front of some object and take a photo with that subject in front of camera view" ✅

```
Example: Position 2m to the right of espresso machine at face level
→ "modern kitchen, change the view to a vantage point at face level 2m
   to the right facing the espresso machine"
```

### 🎬 Camera View Selector (ArchAi3D_Qwen_Camera_View_Selector)
**Quick selection from 22 professional viewpoints**

- **6 orthographic views**: Front, back, left, right, top, bottom
- **5 portrait angles**: Eye level, high angle, low angle, bird's eye, worm's eye
- **4 architectural views**: Section, aerial, street level, entrance
- **3 interior views**: Corner, entrance, ceiling
- **4 cinematic views**: 3/4, isometric, dutch angle, overhead, ground level
- **Perfect for**: Standard architectural elevations, product e-commerce views, quick viewpoint changes

```
Example: Three-quarter view of building
→ "modern architectural exterior, change the view to a three-quarter view
   of the building showing both the front and side"
```

### 🚶 Environment Navigator (ArchAi3D_Qwen_Environment_Navigator)
**Move and rotate through environments with fluid camera paths**

- **14 navigation patterns**: Interior walkthroughs (forward, pan right/left, strafe), Landscape navigation (forward, rise, 360° pan), Architectural (approach, circle, flyby), Cinematic (retreat, rise, descent)
- **Combined movement + rotation**: "Move forward while rotating right" for complex paths
- **Speed control**: Slow, normal, fast, smooth (cinematic)
- **Maintain focus mode**: Keep subject in frame during movement
- **Perfect for**: Interior walkthroughs, landscape exploration, building tours, cinematic shots

```
Example: Walk through forest while panning
→ "dense forest with morning mist, move smoothly 10m forward while rotating right"
```

### 🎨 Material Changer (ArchAi3D_Qwen_Material_Changer)
**Interior design material visualization with 48 presets**

- **6 material categories**: Stone (8), Wood (8), Metal (8), Fabric (8), Paint (8), Tile (8)
- **48 total materials**: Carrara marble, black granite, oak hardwood, walnut, stainless steel, brass, velvet, linen, etc.
- **14 common objects**: Countertop, flooring, wall, backsplash, cabinets, furniture, etc.
- **Automatic preservation**: "keep everything else identical" clause for consistency
- **Perfect for**: Kitchen design, living room variations, bathroom remodeling, client presentations

```
Example: Try different countertop materials
→ "modern kitchen with white cabinets, change the kitchen countertop material
   to white Carrara marble with gray veining, keep everything else identical"
```

### 🧹 Watermark Removal (ArchAi3D_Qwen_Watermark_Removal)
**Simple but powerful cleanup tool**

- **7 element types**: All text, watermark, English text, Chinese text, logo, brand mark, UI elements
- **8 locations**: Anywhere (auto-detect), bottom right/left, top right/left, center, bottom, top
- **One-step cleanup**: Remove watermarks, text overlays, screenshots UI, logos
- **Perfect for**: Stock photo cleanup, screenshot cleaning, social media prep

```
Example: Remove watermark from bottom right
→ "Remove the watermark from the bottom right corner of the image"
```

### 🎨 Colorization (ArchAi3D_Qwen_Colorization)
**Convert B&W to color with historical era context**

- **2 modes**: Auto (model chooses realistic colors) or Custom (specify "blue sky, green grass")
- **9 era presets**: 1900s, 1920s, 1940s, 1950s, 1960s, 1970s, 1980s, Victorian, Medieval
- **Skin tone preservation**: Maintains natural skin tones
- **Perfect for**: Family photo restoration, historical archives, vintage images, documentary work

```
Example: Colorize 1950s photo
→ "colorize this black and white photo with realistic colors appropriate
   for the 1950s era, maintaining natural skin tones"
```

### ✨ Style Transfer (ArchAi3D_Qwen_Style_Transfer)
**Apply 8 artistic styles to specific objects (local stylization)**

- **8 unique styles**: Ice (frozen crystalline), Cloud (soft ethereal), Chinese Lantern (red glowing), Wooden (natural grain), Blue & White Porcelain (ceramic), Fluffy (cotton-like), Weaving (knitted textile), Balloon (inflated shiny)
- **Local stylization**: Applies to specific object, not entire image
- **12 common objects**: House, building, car, furniture, product, nature elements
- **Perfect for**: Creative product visualization, social media content, artistic interior concepts

```
Example: Ice sculpture effect on building
→ "modern architectural exterior, Change the house to ice style"
```

### 📚 Complete Documentation

All 7 new nodes are fully documented with:
- **[QWEN_PROMPT_GUIDE.md](QWEN_PROMPT_GUIDE.md)** - Complete prompt engineering guide (1,630 lines)
  - 12 documented functions with reliability ratings
  - 6 ready-to-use Python templates
  - Universal template structure
  - Best practices & anti-patterns
  - Scene type decision tree
  - Quick reference card
  - Node design guidelines
  - 5 real-world examples

### 🔬 Research Foundation

All nodes based on comprehensive research from:
- ✅ Community findings (Reddit r/StableDiffusion)
- ✅ Official Qwen documentation (Qwen 2.5 VL, Qwen-Image)
- ✅ WanX API documentation (Alibaba)
- ✅ Qwen-Image technical paper
- ✅ 7 PDF files analyzed (100% coverage)

**Key Discovery**: Natural language positioning works perfectly. Pixel coordinates NOT supported by Qwen.

### 💡 Quick Start with New Nodes

**Example Workflow - Product Photography Session:**
```
1. Load product image
2. Scene Photographer → preset: "product_hero_low" → dramatic low angle
3. Camera View Selector → "three_quarter_view" → classic e-commerce angle
4. Scene Photographer → preset: "product_overhead" → flat lay style
5. Style Transfer → "fluffy" style → creative social media variation
6. You now have 4 professional product shots!
```

**Example Workflow - Interior Design Exploration:**
```
1. Load kitchen image
2. Material Changer → "the countertop" → "white Carrara marble" → Generate
3. Material Changer → "the countertop" → "black granite" → Generate
4. Material Changer → "the flooring" → "light oak hardwood" → Generate
5. Environment Navigator → "walkthrough_forward" → room walkthrough
6. Complete material exploration for client presentation!
```

**Example Workflow - Historical Photo Restoration:**
```
1. Load old B&W photo
2. Watermark Removal → "all_text" → "anywhere" → Remove text
3. Colorization → Auto mode → Era: "1950s" → Add period-accurate colors
4. Beautiful restored historical photo!
```

---

## 🎨 Customizing Materials

The Room Transform Prompt node loads materials from `config/materials.yaml`. You can easily customize this file!

### Adding Your Own Materials

Edit `config/materials.yaml` and add new materials:

```yaml
floors:
  - name: "My Custom Floor"
    description: "my custom floor material (detailed description for AI)"
    tags: [bright_light, low_contrast]  # Choose from: rich_dark, bright_light, low_contrast, all
```

### Material Tags

Each material can have multiple tags for organization:
- **`rich_dark`** - Dark, dramatic materials (black, dark wood, navy, etc.)
- **`bright_light`** - Light, bright materials (white, cream, beige, etc.)
- **`low_contrast`** - Smooth, minimal texture (flat paints, polished surfaces)
- **`all`** - Always shown (use for "Keep Original" and "Custom")

### Example Custom Material

```yaml
floors:
  - name: "Weathered Reclaimed Wood"
    description: "weathered reclaimed wood planks (rustic, aged patina, natural variations)"
    tags: [rich_dark]
```

After editing the YAML file, restart ComfyUI to load the new materials.

### Multi-Language Support

You can create language-specific files:
- `config/materials_en.yaml` (English)
- `config/materials_es.yaml` (Spanish)
- `config/materials_fa.yaml` (Persian/Farsi)

Then modify the Python file to load the appropriate language file.

---

## ⚖️ License

### Personal & Non-Commercial Use
**FREE** - Use these nodes for personal projects, learning, and non-commercial purposes.

### Commercial Use  
**Requires License** - If you want to use these nodes for:
- Commercial interior design services
- Paid client work
- Business applications
- Reselling or redistributing

**Please contact me for commercial licensing:**
- 📧 Email: Amir84ferdos@gmail.com
- 💼 LinkedIn: [linkedin.com/in/archai3d](https://www.linkedin.com/in/archai3d/)

**Commercial licenses are affordable and support continued development!**

---

## 👤 About the Author

**Amir Ferdos (ArchAi3d)**
- 🏛️ Architect & AI Developer
- 💻 2+ years ComfyUI experience
- 🎨 Specialized in AI interior design workflows

### Connect With Me

- 💬 **Patreon**: [patreon.com/archai3d](https://patreon.com/archai3d) (Premium workflows & support)
- 💼 **LinkedIn**: [linkedin.com/in/archai3d](https://www.linkedin.com/in/archai3d/)
- 📧 **Email**: Amir84ferdos@gmail.com
- 🐙 **GitHub**: [github.com/amir84ferdos](https://github.com/amir84ferdos)

---

## 🙏 Support This Project

If these nodes help your work:

1. ⭐ **Star this repository**
2. 💎 **[Support on Patreon](https://patreon.com/archai3d)** - Get premium workflows
3. 💬 **Share your results** - Tag me on LinkedIn
4. 📧 **Commercial license** - Support and get business rights

Your support keeps this project alive and improving!

---

## 🐛 Issues & Support

- **GitHub Issues**: [Report bugs here](https://github.com/amir84ferdos/ComfyUI-ArchAi3d-Qwen/issues)
- **Patreon**: Priority support for supporters
- **LinkedIn**: General questions and feedback

---

## 📜 Technical Notes

- **Qwen-VL 2.5** compatible
- **Standard 4D latent format** (compatible with all ComfyUI nodes)
- **RGB channel handling** (automatic alpha removal)
- **Even dimension padding** (ensures model compatibility)
- **ChatML formatting** (proper Qwen-VL prompt structure)

---

**Made with ❤️ for the ComfyUI community**

*Transforming spaces with AI, one room at a time.*