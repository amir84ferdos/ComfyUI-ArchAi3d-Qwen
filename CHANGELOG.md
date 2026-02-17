# Changelog

All notable changes to the ArchAi3D Qwen ComfyUI Custom Nodes project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.42] - 2026-02-17

### Added — VRAM Pinning, Auto-Free Memory, Loader Enhancements

#### VRAM Pin System (`dram_cache.py`)

- **Pin models to GPU VRAM** — protected from ComfyUI's auto-eviction under `--normalvram`
- Monkey-patches `mm.free_memory()` to inject pinned models into `keep_loaded` list
- Lazy patch installation — zero overhead if `keep_on_vram` is never used
- `WeakSet` tracking — pins auto-clean when models are garbage collected
- Functions: `pin_model()`, `unpin_model()`, `is_pinned()`, `unpin_all()`, `get_pinned_count()`

#### Auto-Free Memory System (`dram_cache.py`)

- **`ensure_free_memory()`** — threshold-based cleanup before loading new models
- Checks free VRAM and/or free RAM against user-defined thresholds
- If VRAM low: unpins all models, asks ComfyUI to evict, clears CUDA cache
- If RAM low: clears all DRAM cache entries
- Only triggers when loading a DIFFERENT model (same model reuses cache)
- Solves RunPod serverless orphaned model problem without handler modifications

#### Loader Node Enhancements (all 3 loaders)

- **`keep_on_vram` toggle** — pin model to GPU permanently, smart detection:
  - Same model already pinned → returns cached instantly (no disk I/O)
  - Model name changed → unpins old, loads + pins new
  - Toggle switched to DRAM mode → unpins and enters DRAM path
- **`auto_free_vram` toggle + `min_free_vram_gb` threshold** — auto-cleanup VRAM before disk load
- **`auto_free_dram` toggle + `min_free_dram_gb` threshold** — auto-cleanup RAM before disk load
- Full parameter descriptions in node DESCRIPTION field

#### Memory Cleanup Node Enhancement

- Added `unpin_all_vram` toggle (default: True) — removes all VRAM pins on cleanup

### Changed

- `get_memory_stats()` now shows `[PINNED]` tag on pinned models and pinned count
- DRAM_CACHE_GUIDE.md updated to v1.5.0 with new sections for VRAM pinning and auto-free

### Removed — Research Docs (moved to private repo)

- Moved 6 internal research/dev note files to `runpod-comfyui-worker` repo
- Added `.gitignore` entries to prevent re-addition
- Deleted `.bak` backup files

---

## [3.41] - 2026-02-17

### Added — DRAM Cache Memory Management System

#### New Nodes (3)

- **Offload Model to DRAM** (`nodes/utils/archai3d_offload_model.py`):
  - Moves diffusion model weights from VRAM to CPU RAM
  - Passthrough output: connects KSampler LATENT → VAE Decode
  - Outputs: memory_stats (STRING), dram_id (STRING), passthrough (*)

- **Offload CLIP to DRAM** (`nodes/utils/archai3d_offload_clip.py`):
  - Moves CLIP text encoder weights from VRAM to CPU RAM
  - Passthrough output: connects CLIPTextEncode CONDITIONING → KSampler
  - Outputs: memory_stats (STRING), dram_id (STRING), passthrough (*)

- **Memory Cleanup** (`nodes/utils/archai3d_memory_cleanup.py`):
  - Three toggles: clear DRAM cache, clear VRAM, clear CUDA cache
  - Use when switching between workflows or freeing system RAM

#### New Module

- **dram_cache.py** (`nodes/utils/dram_cache.py`):
  - Core shared DRAM cache module — persistent CPU RAM model storage
  - Module-level `_cache` dict with strong references preventing GC
  - VRAM state safety check warns about conflicting ComfyUI modes
  - Functions: `store()`, `get()`, `remove()`, `clear()`, `list_cached()`, `get_memory_stats()`

- **local_model_cache.py** (`nodes/utils/local_model_cache.py`):
  - RunPod SSD optimization — copies network models to local /tmp/ SSD
  - No-op on local PCs (only activates for /workspace/ paths)
  - Independent from DRAM cache, both can be active simultaneously

### Changed — Triggered Loaders (DRAM Integration)

- **Load Diffusion Model, Load CLIP, Load Dual CLIP**:
  - Added `use_dram` toggle (default: True) — check DRAM cache before disk load
  - Added `cache_to_local_ssd` toggle (default: True) — RunPod SSD optimization
  - Added `memory_stats` STRING output (backwards compatible extra output)
  - Added dynamic `IS_CHANGED` — returns `time.time()` when use_dram=True (forces DRAM check every run)
  - Cache key attached to model as `._dram_cache_key` for offload node to read
  - DRAM hit = ~1s reload from CPU RAM vs ~8s from disk

### Documentation

- **DRAM_CACHE_GUIDE.md**: Comprehensive developer & maintenance guide
  - Full ComfyUI API dependency reference (15 APIs documented)
  - Step-by-step upgrade procedure for new ComfyUI versions
  - Common breaking changes table with fixes
  - Linux swap prevention guide (vm.swappiness)
  - Troubleshooting section

### Added — Utility Nodes (earlier in v3.x)

- **Mask Uncrop** — stitch cropped/rotated region back into original image
- **Batch Split/Merge** — split batch to 6 images and back (cubemap workflows)
- **Circle Mask** — generate circular/elliptical masks
- **Prompt Line** — extract specific line from multiline text
- **Panorama Conversion** — equirectangular ↔ cubemap conversion (requires py360convert)
- **Extract Region Text** — extract text within bounding box coordinates

### Technical Notes

- **Tested on:** ComfyUI v0.14.0, RTX 3090 Ti (24GB), 64GB RAM
- **Startup flags required:** `--normalvram --cache-classic`
- **Linux:** Set `vm.swappiness=10` to prevent model pages being swapped to disk
- **Reference workflow:** `workflows/DRAM_qwen_image_edit_2511.json`

---

## [2.4.0] - 2025-01-07

### Added - Cinematography Prompt Builder Enhancements ⭐

#### New Parameters for Professional Architectural Photography

- **Horizontal Angle Control** - Camera position around object:
  - 10 position options: Front (0°), Angled Left/Right (15°, 30°, 45°), Side (90°), Back (180°)
  - Natural language descriptions: "from thirty degrees to the left for a corner perspective"
  - Full Chinese translation support for all angles
  - Enables precise 3D camera positioning combined with existing vertical angles

- **Perspective Correction System** - Keep vertical lines straight:
  - **Natural (Standard Lens)** - Default mode with natural perspective convergence
  - **Architectural (Keep Verticals Straight)** - Professional architectural photography mode
  - **Tilt-Shift (Full Perspective Control)** - Advanced mode with selective focus plane
  - Automatic tilt-shift lens selection when Full Perspective Control enabled
  - System prompt guidance for maintaining parallel vertical lines
  - Validation warnings for incompatible camera angle combinations

#### Enhanced Prompt Generation

- **Simple Prompt Updates**:
  - Horizontal angle positioning integrated into natural language flow
  - Perspective correction guidance added for architectural mode
  - Example: "positioned from thirty degrees to the left for a corner perspective, with careful framing to keep all vertical lines parallel"

- **Professional Prompt Updates**:
  - Chinese translations for horizontal angles (从左侧30度拍摄,呈现转角视角)
  - Chinese translations for perspective correction (保持所有垂直线平行,防止透视畸变)
  - Integrated into dx8152 LoRA-optimized prompt structure

- **System Prompt Enhancements**:
  - Architectural guidance automatically appended when perspective correction enabled
  - "IMPORTANT: Maintain parallel vertical lines in architectural photography. Keep the camera level..."
  - Applied to all 3 system prompt modes (Professional, Research-Validated, Simple/Beginner)

#### New Helper Methods

- `_get_horizontal_angle_description()` - Converts angle selections to natural language (English + Chinese)
- `_get_perspective_correction_prompting()` - Generates perspective guidance text (English + Chinese)

#### Enhanced Validation

- **Perspective Correction Compatibility Check**:
  - Warns if perspective correction enabled with non-level camera angles
  - "⚠️ Perspective correction requires eye-level camera angle. Vertical lines will converge with tilted camera positions."
  - Prevents common architectural photography mistakes

### Changed

- **Cinematography Prompt Builder**:
  - Function signature updated with `horizontal_angle` and `perspective_correction` parameters
  - Lens auto-selection logic enhanced for tilt-shift mode
  - All prompts now support full 3D positioning with horizontal + vertical angles

### Documentation

- **HORIZONTAL_ANGLE_PERSPECTIVE_CORRECTION.md**: Complete implementation guide
  - 3 perspective correction modes explained in detail
  - 10 horizontal angle options with use cases
  - Usage examples with expected outputs
  - Technical implementation details

- **CAMERA_PROMPTING_GUIDE.md**: Comprehensive 15,000+ word guide
  - Based on Nanobanan's 5-ingredient camera prompting formula
  - 15 annotated working examples covering all shot types
  - Quick reference charts for shot sizes, angles, DOF, styles
  - Integration guide for Cinematography Prompt Builder node

- **Additional Documentation**:
  - CINEMATOGRAPHY_PROMPT_BUILDER_COMPLETE.md - Full v2.4.0 feature summary
  - CUSTOM_DETAILS_TOOLTIP_ENHANCEMENT.md - Enhanced tooltip guidance
  - PROMPT_FORMAT_FIXES.md - Natural language improvements
  - SYSTEM_PROMPT_UPDATE.md - Dynamic system prompt implementation

### Technical Notes

- **Research-Validated Approach**:
  - Horizontal angles use natural language ("from thirty degrees to the left") instead of degree-based rotation commands
  - Aligns with vision-language research showing distance-based positioning more reliable than degree-based
  - Perspective correction uses explicit natural language guidance for architectural straight verticals

- **Backwards Compatibility**:
  - All new parameters have sensible defaults (Front View, Natural perspective)
  - Existing workflows continue working without modification
  - Progressive enhancement approach for advanced users

- **Language Support**:
  - Full Chinese translations for all new features
  - Optimized for dx8152 LoRAs requiring Chinese cinematography terms
  - Hybrid mode combines Chinese technical terms with English details

### Benefits

- **Precise Camera Control**: Full 3D positioning with horizontal + vertical angles + distance
- **Professional Architectural Photography**: Straight vertical lines, no keystoning distortion
- **Interior Design Workflows**: Perfect for architectural visualization and real estate photography
- **User-Friendly**: Clear tooltips, validation warnings, auto-selection features
- **Research-Backed**: Implements findings from vision-language camera control research

---

## [2.3.0] - 2025-01-06

### Added - Object Focus Camera System ⭐

#### New Camera Control Nodes (v1-v7)
- **Object Focus Camera v1-v3**: Foundation camera control nodes
  - Basic object focusing with distance and height control
  - Direction and lens type selection
  - Chinese/English/Hybrid prompt support

- **Object Focus Camera v4**: Enhanced with quality presets
  - Added professional photography quality presets
  - Improved prompt generation structure

- **Object Focus Camera v5**: Material detail system
  - 37 material detail presets for better object visualization
  - Enhanced vantage point mode

- **Object Focus Camera v6**: Unified prompt structure
  - Complete redesign with unified English/Chinese prompts
  - Enhanced vantage point features (Interior Focus style)
  - 15 photography quality presets
  - Improved plural-safe grammar for multiple objects

- **🎬 Object Focus Camera v7 (Pro Cinema)** (NEW - RECOMMENDED):
  - Professional cinematography edition with industry-standard terminology
  - **8 Shot Sizes**: ECU, CU, MCU, MS, MLS, FS, WS, EWS (replaces distance presets)
  - **7 Camera Angles**: Eye Level, High Angle, Low Angle, Bird's Eye, Worm's Eye, Dutch Angle, Over-the-Shoulder
  - **8 Camera Movements**: Static, Pan, Tilt, Dolly, Truck, Pedestal, Arc, Zoom
  - **Enhanced Lens Types**: Ultra Wide (14-24mm), Wide (24-35mm), Standard (35-50mm), Portrait (85mm), Telephoto (70-200mm), Super Telephoto (200mm+)
  - **Framing Mode**: Toggle between Shot Size Presets or Custom Meters
  - Complete cinematography reference documentation included
  - Maintains all v6 features (vantage point, presets, plural-safe grammar)

#### Supporting Nodes
- **Simple Camera Control**: Basic camera positioning and control
- **dx8152 LoRA Support Nodes**: Enhanced compatibility with dx8152's Multiple-angles LoRA

### Enhanced Features

- **Professional Cinematography Terminology**:
  - Shot sizes replace numeric distance system in v7
  - Industry-standard camera angles and movements
  - Professional lens focal length classifications
  - Comprehensive documentation from StudioBinder, MasterClass, B&H Photo

- **Plural-Safe Grammar System**:
  - Automatic singular/plural detection across all camera versions
  - 200+ grammar fixes applied to v1-v6
  - Correctly handles "chair" vs "chairs", "bottle" vs "bottles", etc.
  - Works with comma-separated object lists

- **Multi-Language Support**:
  - Chinese/English/Hybrid prompt modes
  - Optimized for dx8152 LoRAs requiring Chinese prompts
  - Seamless language switching

### Changed

- **Object Focus Camera v6**: Updated default settings
  - Target object: "chair" → more universal default
  - Height: 1.5m → better viewing angle
  - Distance: 2.5m → Medium Shot equivalent
  - Lens: "Normal (50mm)" → standard photography lens
  - Prompt mode: "Hybrid (Chinese + English)" → dx8152 LoRA compatibility

- **Object Focus Camera v7**: Parameter clarity improvements
  - Renamed `distance_mode` → `framing_mode` for clearer understanding
  - Enhanced tooltips explaining shot size to distance mapping
  - Simplified parameter structure (removed redundant camera_distance)

### Documentation

- **cinematography_reference_v7.md**: Comprehensive cinematography guide
  - 11 shot sizes with definitions and distances
  - 8 camera angles with psychological effects
  - 8 camera movements with technical details
  - Chinese translations for all terms
  - Sources from professional cinematography resources

### Technical Notes

- **v7 Design Philosophy**: Clean professional design over backward compatibility
  - v6 remains available for numeric distance workflows
  - v7 targets professional cinematographers and visualization artists
  - Shot sizes provide intuitive framing vs arbitrary meters

- **Node Count**: Now **48 custom nodes** (up from 41)
  - 8 new Object Focus Camera variants (v1-v7 + Simple Camera)
  - 1 dx8152 LoRA support node

---

## [2.2.0] - 2025-11-03

### Added - Phase 2A: Functional GRAG Implementation ⭐

- **GRAG Sampler Node** (✅ REQUIRED for GRAG to work):
  - New `🎚️ GRAG Sampler` - Functional GRAG-aware sampler
  - Extracts GRAG config from conditioning metadata
  - Injects attention reweighting patches during sampling
  - No ComfyUI core modifications (update-safe implementation)
  - Graceful fallback to standard sampling if GRAG fails
  - **Critical**: You MUST use this sampler to see GRAG effects!

- **GRAG Attention Utilities** (`nodes/core/utils/grag_attention.py`):
  - Implements full GRAG mathematical algorithm from research paper
  - `apply_grag_to_keys()`: Text/image stream separation and reweighting
  - Group mean computation and token deviation calculation
  - Formula: `k̂ = λ * k_mean + δ * (k - k_mean)`
  - `create_grag_patch()`: Factory for ComfyUI transformer_options integration
  - Helper functions for config extraction and validation
  - Preset system (Subtle/Balanced/Strong parameter sets)

### Changed

- **GRAG System Now Fully Functional**:
  - Previous v2.1.1 GRAG nodes were placeholder (metadata only)
  - Now implements actual attention manipulation during generation
  - Real fine-grained control with visible effects on output
  - Continuous control range (0.8-1.7) instead of binary on/off

- **Updated Documentation**:
  - `GRAG_MODIFIER_GUIDE.md`: Added GRAG Sampler requirement and workflow
  - `GRAG_INTEGRATION_SUMMARY.md`: Marked Phase 2A as completed
  - Added troubleshooting for "no effect" issue (missing GRAG Sampler)
  - Updated all workflow examples with correct sampler usage

- **Startup Message**:
  - Now shows "Sampling: 1 node (GRAG Sampler)"
  - Updated to 41 nodes total
  - Highlights functional GRAG implementation

### Technical Notes

- **Implementation Details**:
  - GRAG operates after RoPE (Rotary Position Embeddings)
  - Intercepts attention keys before attention computation
  - Applies independent reweighting to text and image token streams
  - Works via ComfyUI's `transformer_options["patches"]` system
  - Compatible with all existing encoders (via GRAG Modifier)

- **Performance**:
  - Minimal overhead (~5-10% per attention layer)
  - No CUDA memory increase
  - Single global λ/δ parameters (Phase 2A)
  - Multi-resolution tiers planned for Phase 2B

- **Based on**: [GRAG-Image-Editing](https://github.com/little-misfit/GRAG-Image-Editing) by little-misfit
- **Research Paper**: arXiv 2510.24657 (October 2024)

### Workflow

**Complete Functional Workflow**:
```
[Images] → [Encoder V2] → [GRAG Modifier] → [GRAG Sampler] → [VAE Decode] → [Output]
                               ↓ enable_grag=True      ↓ Applies reweighting
                          Prepares metadata
```

**Important**: Standard KSampler will NOT apply GRAG effects, even if GRAG Modifier is used!

---

## [2.1.1] - 2025-11-03

### Added
- **GRAG Modifier Node** (Recommended - Universal):
  - New `ArchAi3D GRAG Modifier` - Universal conditioning modifier
  - Works with ANY encoder (V1, V2, V3, Simple, etc.)
  - Clean passthrough mode when disabled (optional use)
  - Perfect for A/B testing and flexible workflows
  - **Benefits**: No code duplication, maximum flexibility, easy maintenance

- **GRAG Encoder Node** (Experimental - Standalone):
  - `ArchAi3D Qwen GRAG Encoder` - Standalone GRAG encoder
  - Includes full encoder + GRAG in one node
  - Useful for testing GRAG-specific configurations
  - May be deprecated in favor of modifier approach

- **GRAG Implementation**:
  - Implements GRAG (Group-Relative Attention Guidance) metadata preparation
  - Three main parameters: `grag_strength` (0.8-1.7), `grag_cond_b`, `grag_cond_delta`
  - Adjustable in 0.01 increments for precise control
  - Better structure/window preservation potential
  - Training-free fine-grained editing control

- **GRAG Documentation**:
  - Complete usage guide with parameter explanations
  - Integration examples with Clean Room workflow
  - Parameter tuning tips and troubleshooting
  - Future development roadmap
  - Comparison: Modifier vs Encoder approaches

### Changed
- Updated version number to 2.1.1 for proper release tracking
- Increased encoder count from 5 to 6 nodes
- Updated startup message to highlight GRAG feature

### Technical Notes
- Current GRAG implementation is a **placeholder** preparing metadata
- Full functionality requires integration with actual GRAG pipeline code
- Based on: [GRAG-Image-Editing](https://github.com/little-misfit/GRAG-Image-Editing)
- Qwen-Image-Edit support added to GRAG in November 2025

---

## [2.1.0] - 2025-11-03

### Enhanced Features

#### Clean Room Prompt Node v2.1.0 - Major Enhancement ⭐
- **Scene Context Field** (NEW):
  - Optional multiline text field for describing room context
  - Examples: "modern office with large windows", "bedroom with floor-to-ceiling windows"
  - Helps preserve architectural features and overall room character
  - Integrated at the start of prompts following Qwen best practices (context-first approach)
  - Auto-detects windows in context and adds explicit window preservation clause

- **Watermark/Logo Removal** (NEW):
  - Checkbox toggle to enable watermark removal during room cleaning
  - 5 watermark types: watermark, logo, text, English text, Chinese text
  - 6 location options: anywhere, bottom right, bottom left, top right, top left, center
  - Uses research-validated "Remove [TYPE] from [LOCATION]" pattern from Qwen WanX API
  - Eliminates need for separate Watermark Removal node in room cleaning workflows

- **Enhanced System Prompt**:
  - Updated "Room Transform Specialist" to emphasize window preservation (CRITICAL)
  - Added watermark/logo/text removal to supported cleanup operations
  - Improved inpainting instructions for seamless blending

- **Perfect Qwen Prompt Structure**:
  - Implements research-based pattern: [SCENE_CONTEXT] + [TRANSFORMATION] + [REMOVAL] + [SURFACES] + [PRESERVATION] + [STYLE]
  - Window preservation automatically added when windows mentioned in context
  - Example: "Transform image1: modern office with large windows, clean finished interior. Remove scaffolding/the watermark from the bottom right corner."

### Added
- **Automated Publishing**: Added GitHub Actions workflow for automatic publishing to Comfy Registry
- **PyPI Support**: Created `pyproject.toml` for PyPI package distribution
- **CHANGELOG**: Added comprehensive changelog for tracking version history
- **Package Metadata**: Complete project metadata including 38 custom nodes documentation
- **Helper Function**: `build_watermark_removal_phrase()` for watermark removal phrase generation

### Fixed
- **Clean Room Prompt Node**: Fixed material library loading path issue
  - The node was looking for `config/materials.yaml` in the wrong directory
  - Updated path to correctly navigate from `nodes/core/prompts/` to root `config/` folder
  - Now properly loads all 103+ materials (32 floors, 36 walls, 35 ceilings)
- **License Clarification**: Standardized dual licensing information across all files
  - Updated `__init__.py` to reflect dual license model
  - Clarified personal (free) vs commercial (paid) usage terms

### Changed
- **Version Synchronization**: Rolled back version from `5.0.0-alpha` to `2.1.0`
  - Aligns with actual git tag history (previous tag: v2.0.0)
  - Provides clean versioning path forward
  - Updated version in `__init__.py` (lines 9 and 223)
- **Startup Message**: Simplified console output to be cleaner and more professional
- **License Documentation**: Updated license references to be consistent with `license_file.txt`

### Technical Details
- **Backward Compatible**: All new features are optional with safe defaults
- **Research-Based**: Enhancements follow proven Qwen prompting patterns
- **Tested**: Comprehensive testing confirms backward compatibility and new feature functionality

### Infrastructure
- GitHub Actions workflow for Comfy Registry publishing
- Support for both Comfy Registry and PyPI distribution channels
- Proper Python packaging with build system configuration

---

## [2.0.0] - 2024-12-XX

### Added
- **Initial Public Release** - Professional Interior Design Toolkit
- **38 Custom Nodes** across multiple categories:
  - 5 Core Encoding nodes (V1, V2, V3, Simple, Simple V2)
  - 3 Prompt Builder nodes (Clean Room, System Prompt, Image Scale)
  - 18 Camera Control nodes (Exterior, Interior, Object, Person)
  - 4 Image Editing nodes (Material Changer, Watermark Removal, Colorization, Style Transfer)
  - 8 Utility nodes (Position Guide, Color Correction, Mask Tools)

#### Core Features
- **Qwen-VL Encoders**:
  - V1: Standard strength controls
  - V2: Two-stage interpolation for precise control (recommended)
  - V3: Preset balance with CFG control
  - Simple: Easy-to-use version
  - Simple V2: Multi-image direct input (up to 3 VL images + 3 latents)

- **Clean Room Prompt Builder**:
  - 103+ material presets loaded from YAML
  - 32 floor materials (marble, hardwood, concrete, tile, stone, carpet)
  - 36 wall materials (paint, wallpaper, wood, brick, concrete, tile)
  - 35 ceiling materials (paint, architectural, beams, industrial, wood)
  - User-customizable material library
  - 3 workflow modes (Remove Only, Remove + Paint All, Remove + Paint Selective)
  - Quality control toggles

- **Camera Control System**:
  - Professional viewpoint control for interior/exterior scenes
  - Object rotation with 360° turntable support
  - Person perspective with identity preservation
  - Scene photographer with 14 presets
  - Environment navigator with 14 navigation patterns
  - 22 professional viewpoint presets

- **Image Editing Tools**:
  - Material Changer: 48 material presets across 6 categories
  - Watermark Removal: Remove text, logos, and watermarks
  - Colorization: B&W to color with 9 historical era presets
  - Style Transfer: 8 artistic styles (ice, cloud, wooden, fluffy, etc.)

- **Smart Image Scaling**:
  - 23 Qwen-VL optimized aspect ratios
  - Auto or manual aspect ratio selection
  - Multiple scaling strategies (crop, letterbox, stretch)

- **Utility Nodes**:
  - Position Guide system for spatial control
  - BT.709 color correction
  - Advanced color correction tools
  - Average color calculation
  - Solid color image generation
  - Mask crop and rotate

#### Documentation
- Comprehensive README with usage examples
- Camera control guide with research-based techniques
- Prompt reference with reliability ratings
- Person perspective guide
- Material customization guide

#### License
- Dual License model established:
  - Free for personal and non-commercial use
  - Commercial license available for business use
- Contact information for commercial licensing

---

## Version History Summary

- **v2.3.0** (Current): Object Focus Camera v1-v7 with professional cinematography features
- **v2.2.0**: Functional GRAG implementation with sampler and attention utilities
- **v2.1.1**: GRAG Modifier and Encoder nodes
- **v2.1.0**: Bug fixes, automated publishing setup, improved documentation
- **v2.0.0** (Initial): First public release with 38 custom nodes for professional AI interior design

---

## License

This project uses a dual license model:
- **Personal/Non-Commercial**: Free to use
- **Commercial**: License required (contact Amir84ferdos@gmail.com)

For full license details, see [license_file.txt](license_file.txt)

---

## Support

- **Issues**: [GitHub Issues](https://github.com/amir84ferdos/ComfyUI-ArchAi3d-Qwen/issues)
- **Patreon**: [Support Development](https://patreon.com/archai3d)
- **Email**: Amir84ferdos@gmail.com
- **LinkedIn**: [ArchAi3d](https://www.linkedin.com/in/archai3d/)
