# Changelog

All notable changes to the ArchAi3D Qwen ComfyUI Custom Nodes project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
