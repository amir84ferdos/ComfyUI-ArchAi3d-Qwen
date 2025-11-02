# Changelog

All notable changes to the ArchAi3D Qwen ComfyUI Custom Nodes project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2025-01-XX

### Added
- **Automated Publishing**: Added GitHub Actions workflow for automatic publishing to Comfy Registry
- **PyPI Support**: Created `pyproject.toml` for PyPI package distribution
- **CHANGELOG**: Added comprehensive changelog for tracking version history
- **Package Metadata**: Complete project metadata including 38 custom nodes documentation

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

- **v2.1.0** (Current): Bug fixes, automated publishing setup, improved documentation
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
