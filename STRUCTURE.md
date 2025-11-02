# ComfyUI-ArchAi3d-Qwen - Folder Structure

**Version:** 3.0.0
**Date:** 2025-10-15
**Status:** ✅ Reorganized and Optimized

---

## 📁 New Folder Structure

```
ComfyUI-ArchAi3d-Qwen/
├── __init__.py                          # Main registration file (18 nodes)
├── LICENSE
├── license_file.txt
│
├── nodes/                               # All node files (organized)
│   ├── __init__.py
│   │
│   ├── core/                            # Core encoding nodes (7 nodes)
│   │   ├── __init__.py
│   │   │
│   │   ├── encoders/                    # Encoder nodes (4 nodes)
│   │   │   ├── __init__.py
│   │   │   ├── archai3d_qwen_encoder.py
│   │   │   ├── archai3d_qwen_encoder_v2.py
│   │   │   ├── archai3d_qwen_encoder_simple.py
│   │   │   └── archai3d_qwen_encoder_simple_v2.py
│   │   │
│   │   ├── utils/                       # Utility nodes (2 nodes)
│   │   │   ├── __init__.py
│   │   │   ├── archai3d_qwen_image_scale.py
│   │   │   └── archai3d_qwen_system_prompt.py
│   │   │
│   │   └── prompts/                     # Prompt builder nodes (1 node)
│   │       ├── __init__.py
│   │       └── archai3d_clean_room_prompt.py
│   │
│   ├── camera/                          # Camera control nodes (7 nodes)
│   │   ├── __init__.py
│   │   ├── archai3d_qwen_camera_view.py
│   │   ├── archai3d_qwen_object_rotation.py
│   │   ├── archai3d_qwen_object_rotation_v2.py
│   │   ├── archai3d_qwen_person_perspective.py
│   │   ├── archai3d_qwen_scene_photographer.py ⭐ NEW
│   │   ├── archai3d_qwen_camera_view_selector.py ⭐ NEW
│   │   └── archai3d_qwen_environment_navigator.py ⭐ NEW
│   │
│   └── editing/                         # Image editing nodes (4 nodes)
│       ├── __init__.py
│       ├── archai3d_qwen_material_changer.py ⭐ NEW
│       ├── archai3d_qwen_watermark_removal.py ⭐ NEW
│       ├── archai3d_qwen_colorization.py ⭐ NEW
│       └── archai3d_qwen_style_transfer.py ⭐ NEW
│
├── docs/                                # All documentation (clean!)
│   ├── README.md                        # Main documentation (updated)
│   ├── NEW_NODES_README.md              # Quick reference for 7 new nodes
│   ├── QWEN_PROMPT_GUIDE.md             # Complete prompt engineering guide
│   ├── CAMERA_CONTROL_GUIDE.md          # Camera control guide
│   ├── OBJECT_ROTATION_V2_GUIDE.md      # Object rotation guide
│   ├── PERSON_PERSPECTIVE_GUIDE.md      # Person perspective guide
│   ├── CINEMATOGRAPHY_PRESETS_GUIDE.md  # Cinematography presets
│   ├── CAMERA_SYSTEM_PROMPTS.md         # System prompts
│   ├── PROMPT_REFERENCE.md              # Quick prompt reference
│   └── ROADMAP.md                       # Development roadmap
│
├── config/                              # Configuration files
│   └── materials.yaml                   # Material presets (user-editable)
│
└── web/                                 # Web UI resources
    └── (custom UI elements)
```

---

## 🎯 Benefits of New Structure

### 1. **Clean Organization**
- ✅ All nodes in `nodes/` folder
- ✅ All documentation in `docs/` folder
- ✅ Logical grouping by function

### 2. **Easy Navigation**
- ✅ Clear folder names (core/camera/editing)
- ✅ Subfolders for related nodes
- ✅ All docs in one place

### 3. **Scalability**
- ✅ Easy to add new nodes
- ✅ Clear where to put new files
- ✅ Won't get messy as project grows

### 4. **Better ComfyUI Organization**
- ✅ Nodes organized in submenus
- ✅ Emoji prefixes for visual clarity
- ✅ Grouped by category

---

## 📊 Node Organization

### Core Encoding (7 nodes)
**Menu:** `ArchAi3d/Qwen/Core`

| Node | Location | Purpose |
|------|----------|---------|
| 🎨 Qwen Encoder | `nodes/core/encoders/` | Standard encoder |
| 🎨 Qwen Encoder V2 | `nodes/core/encoders/` | Advanced encoder (recommended) |
| 🎨 Qwen Encoder Simple | `nodes/core/encoders/` | Simple encoder |
| 🎨 Qwen Encoder Simple V2 | `nodes/core/encoders/` | Multi-image encoder |
| 📏 Qwen Image Scale | `nodes/core/utils/` | Smart scaling |
| 💬 Qwen System Prompt | `nodes/core/utils/` | System prompts |
| 🏗️ Clean Room Prompt | `nodes/core/prompts/` | Room transformation |

### Camera Control (7 nodes)
**Menu:** `ArchAi3d/Qwen/Camera`

| Node | Location | Purpose |
|------|----------|---------|
| 📹 Camera View | `nodes/camera/` | Professional camera control |
| 🔄 Object Rotation | `nodes/camera/` | Basic rotation |
| 🔄 Object Rotation V2 | `nodes/camera/` | Advanced rotation (19 presets) |
| 👤 Person Perspective | `nodes/camera/` | Portrait angles |
| 📸 Scene Photographer ⭐ | `nodes/camera/` | Frame specific subjects (14 presets) |
| 🎬 Camera View Selector ⭐ | `nodes/camera/` | 22 professional views |
| 🚶 Environment Navigator ⭐ | `nodes/camera/` | Move through scenes (14 patterns) |

### Image Editing (4 nodes)
**Menu:** `ArchAi3d/Qwen/Editing`

| Node | Location | Purpose |
|------|----------|---------|
| 🎨 Material Changer ⭐ | `nodes/editing/` | 48 materials (6 categories) |
| 🧹 Watermark Removal ⭐ | `nodes/editing/` | Remove text/watermarks |
| 🌈 Colorization ⭐ | `nodes/editing/` | B&W to color (9 eras) |
| ✨ Style Transfer ⭐ | `nodes/editing/` | 8 artistic styles |

**TOTAL: 18 nodes** (7 core + 7 camera + 4 editing)

---

## 🔄 What Changed

### Before (Messy):
```
ComfyUI-ArchAi3d-Qwen/
├── archai3d_qwen_encoder.py
├── archai3d_qwen_encoder_v2.py
├── archai3d_qwen_encoder_simple.py
├── archai3d_qwen_encoder_simple_v2.py
├── archai3d_qwen_image_scale.py
├── archai3d_qwen_system_prompt.py
├── archai3d_clean_room_prompt.py
├── archai3d_qwen_camera_view.py
├── archai3d_qwen_object_rotation.py
├── archai3d_qwen_object_rotation_v2.py
├── archai3d_qwen_person_perspective.py
├── archai3d_qwen_scene_photographer.py
├── archai3d_qwen_camera_view_selector.py
├── archai3d_qwen_environment_navigator.py
├── archai3d_qwen_material_changer.py
├── archai3d_qwen_watermark_removal.py
├── archai3d_qwen_colorization.py
├── archai3d_qwen_style_transfer.py
├── README.md
├── CAMERA_CONTROL_GUIDE.md
├── OBJECT_ROTATION_V2_GUIDE.md
├── PERSON_PERSPECTIVE_GUIDE.md
├── CINEMATOGRAPHY_PRESETS_GUIDE.md
├── CAMERA_SYSTEM_PROMPTS.md
├── PROMPT_REFERENCE.md
├── QWEN_PROMPT_GUIDE.md
├── NEW_NODES_README.md
├── ROADMAP.md
├── config/
├── web/
└── __init__.py

❌ 18 Python files mixed with 9 documentation files = MESSY!
```

### After (Clean):
```
ComfyUI-ArchAi3d-Qwen/
├── nodes/            # 18 node files organized in 3 categories
│   ├── core/        # 7 core nodes
│   ├── camera/      # 7 camera nodes
│   └── editing/     # 4 editing nodes
│
├── docs/            # 9 documentation files in one place
├── config/          # Configuration
├── web/             # Web resources
└── __init__.py      # Registration (updated)

✅ Everything organized and easy to find!
```

---

## 📖 Documentation Organization

All documentation moved to `docs/` folder:

**Main Documentation:**
- `README.md` - Main documentation (updated with all 18 nodes)
- `STRUCTURE.md` - This file (folder structure guide)

**Node-Specific Guides:**
- `NEW_NODES_README.md` - Quick reference for 7 new nodes
- `QWEN_PROMPT_GUIDE.md` - Complete prompt engineering guide (1,630 lines)
- `CAMERA_CONTROL_GUIDE.md` - Camera control guide
- `OBJECT_ROTATION_V2_GUIDE.md` - Object rotation guide
- `PERSON_PERSPECTIVE_GUIDE.md` - Person perspective guide
- `CINEMATOGRAPHY_PRESETS_GUIDE.md` - Cinematography presets
- `CAMERA_SYSTEM_PROMPTS.md` - System prompts
- `PROMPT_REFERENCE.md` - Quick prompt reference

**Development:**
- `ROADMAP.md` - Development roadmap

---

## 🚀 How to Use

### 1. After Restart ComfyUI

All nodes will appear in organized submenus:

```
Add Node → ArchAi3d →
    ├── Qwen →
    │   ├── Core →
    │   │   ├── 🎨 Qwen Encoder
    │   │   ├── 🎨 Qwen Encoder V2
    │   │   ├── 🎨 Qwen Encoder Simple
    │   │   ├── 🎨 Qwen Encoder Simple V2
    │   │   ├── 📏 Qwen Image Scale
    │   │   ├── 💬 Qwen System Prompt
    │   │   └── 🏗️ Clean Room Prompt
    │   │
    │   ├── Camera →
    │   │   ├── 📹 Camera View
    │   │   ├── 🔄 Object Rotation
    │   │   ├── 🔄 Object Rotation V2
    │   │   ├── 👤 Person Perspective
    │   │   ├── 📸 Scene Photographer ⭐
    │   │   ├── 🎬 Camera View Selector ⭐
    │   │   └── 🚶 Environment Navigator ⭐
    │   │
    │   └── Editing →
    │       ├── 🎨 Material Changer ⭐
    │       ├── 🧹 Watermark Removal ⭐
    │       ├── 🌈 Colorization ⭐
    │       └── ✨ Style Transfer ⭐
```

### 2. Finding Documentation

All documentation in `docs/` folder:
```
ComfyUI-ArchAi3d-Qwen/docs/
├── README.md              ← Start here
├── NEW_NODES_README.md    ← 7 new nodes quick reference
└── QWEN_PROMPT_GUIDE.md   ← Complete prompt guide
```

### 3. Adding New Nodes

**Easy to add new nodes!**

```
1. Choose category: core/camera/editing
2. Create node file in appropriate folder
3. Add import to __init__.py
4. Add to NODE_CLASS_MAPPINGS
5. Add to NODE_DISPLAY_NAME_MAPPINGS
6. Done!
```

---

## ⚙️ Updated __init__.py

**Version:** 3.0.0

**Key Changes:**
- ✅ All 18 nodes registered
- ✅ Organized imports by category
- ✅ Emoji prefixes for visual clarity
- ✅ Informative startup message
- ✅ Shows node count by category

**Startup Message:**
```
======================================================================
[ArchAi3d-Qwen v3.0.0] Loading nodes...
  📦 Core Encoding: 7 nodes
  📸 Camera Control: 7 nodes
  🎨 Image Editing: 4 nodes
  ✅ Total: 18 nodes loaded successfully!
  📚 Documentation: ./docs/
======================================================================
```

---

## 📊 Statistics

**Before Reorganization:**
- 18 Python files in root
- 9 documentation files in root
- 1 config folder
- 1 web folder
- Total: **29 items in root = MESSY**

**After Reorganization:**
- `nodes/` folder (organized into 3 categories)
- `docs/` folder (all 9 docs in one place)
- `config/` folder
- `web/` folder
- `__init__.py`
- `LICENSE`
- Total: **6 items in root = CLEAN!** ✅

**Improvement:** **79% reduction in root clutter** (29 → 6 items)

---

## 🔮 Future Scalability

Adding new nodes is now easy:

**Example: Adding a new camera node**
```
1. Create: nodes/camera/archai3d_qwen_new_camera_node.py
2. Update __init__.py:
   - Add import: from .nodes.camera.archai3d_qwen_new_camera_node import NewNode
   - Add to NODE_CLASS_MAPPINGS
   - Add to NODE_DISPLAY_NAME_MAPPINGS
3. Done!
```

**Example: Adding a new editing node**
```
1. Create: nodes/editing/archai3d_qwen_new_editing_node.py
2. Update __init__.py (same as above)
3. Done!
```

**Example: Adding a new category**
```
1. Create: nodes/new_category/
2. Add __init__.py to folder
3. Create nodes in folder
4. Update main __init__.py
5. Done!
```

---

## ✅ Migration Checklist

- [x] Created organized folder structure
- [x] Moved all node files to `nodes/` with subfolders
- [x] Moved all documentation to `docs/`
- [x] Created `__init__.py` files for all folders
- [x] Updated main `__init__.py` with all 18 nodes
- [x] Added emoji prefixes for visual clarity
- [x] Organized imports by category
- [x] Added informative startup message
- [x] Updated README.md in docs folder
- [x] Created STRUCTURE.md (this file)
- [x] Ready for ComfyUI restart

---

## 🎯 Next Steps

1. **Restart ComfyUI** to load reorganized structure
2. **Verify all 18 nodes** appear in submenus
3. **Check console** for successful load message
4. **Test a few nodes** to ensure imports work
5. **Enjoy clean organization!** ✨

---

## 👤 Author

**Amir Ferdos (ArchAi3d)**
- 📧 Email: Amir84ferdos@gmail.com
- 💼 LinkedIn: [linkedin.com/in/archai3d](https://www.linkedin.com/in/archai3d/)
- 🐙 GitHub: [github.com/amir84ferdos](https://github.com/amir84ferdos)

---

**Structure Version:** 3.0.0
**Last Updated:** 2025-10-15
**Status:** ✅ Complete and Ready

**Enjoy your clean and organized ComfyUI-ArchAi3d-Qwen folder!** 🎉
