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

**5 Custom Nodes** (all under `ArchAi3d/Qwen` category):

- 🎨 **Qwen Encoder V1** - Standard strength controls
- 🎨 **Qwen Encoder V2** - Advanced interpolation (recommended)
- 🎨 **Qwen Encoder Simple** - Easy-to-use version
- 📏 **Qwen Image Scale** - Smart aspect ratio scaling (23 presets)
- 💬 **Qwen System Prompt** - Preset prompt loader

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

### 🔧 Under Development

- 🔧 **Weight control refinement** - Fine-tuning prompt vs reference balance
- 🔧 **More preset workflows** - Expanding style library
- 🔧 **Better documentation** - Video tutorials and examples
- 🔧 **Strength presets** - Pre-configured settings for common scenarios

### 🎯 Planned Features

- 📅 **Style consistency mode** - Match existing room designs
- 📅 **Batch processing** - Process multiple rooms at once
- 📅 **Advanced masking** - Region-specific design control
- 📅 **Material library** - Quick material swapping
- 📅 **Lighting presets** - Pre-configured lighting scenarios

---

## 📖 Basic Usage

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

**For detailed workflows and presets, check my Patreon!**

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