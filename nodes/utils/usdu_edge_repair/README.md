# USDU Edge Repair - Module Documentation

## Overview

USDU Edge Repair is a ComfyUI node for tile-based image upscaling with advanced features:

- **Per-tile Conditioning** - Different prompts for each tile
- **Differential Diffusion** - Per-pixel denoise control via mask
- **Per-tile ControlNet** - ControlNet applied to each tile separately
- **Preview Mode** - Debug visualization of tiles and masks
- **Safeguard Validation** - Prevents invalid tile configurations

Based on Ultimate SD Upscale (USDU) with custom enhancements.

---

## File Structure

```
usdu_edge_repair/
├── __init__.py          (7 lines)     - Package entry point
├── node.py              (246 lines)   - Main ComfyUI node
├── tile_geometry.py     (440 lines)   - SINGLE SOURCE OF TRUTH for all geometry
├── constants.py         (27 lines)    - Mode constants
├── inputs.py            (91 lines)    - Input definitions
├── validation.py        (82 lines)    - Safeguard validation
├── diffdiff.py          (89 lines)    - Differential Diffusion
├── preview.py           (94 lines)    - Preview mode (uses TileGeometry)
├── controlnet.py        (176 lines)   - ControlNet patches
├── processing.py        (910 lines)   - Core sampling
├── utils.py             (995 lines)   - Utility functions
├── shared.py            (107 lines)   - Global state
├── upscaler.py          (126 lines)   - Upscaler wrapper
├── usdu_patch.py        (257 lines)   - Monkey patches
├── ultimate_upscale.py  (1,123 lines) - USDU core
└── Archive/             (empty)       - For deprecated files
```

---

## File Descriptions

### Entry Point

#### `__init__.py`
Package entry point. Exports `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` to register the node with ComfyUI.

```python
from .node import ArchAi3D_USDU_EdgeRepair, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
```

---

### Main Node

#### `node.py`
The main ComfyUI node class `ArchAi3D_USDU_EdgeRepair`.

**Key responsibilities:**
- Define `INPUT_TYPES` for ComfyUI UI
- Create TileGeometry instance (single source of truth)
- Orchestrate the upscaling pipeline
- Handle preview mode
- Setup Differential Diffusion and ControlNet
- Call USDU script

**Main method:** `upscale()` - processes the image through USDU

---

### Geometry (NEW)

#### `tile_geometry.py`
**SINGLE SOURCE OF TRUTH** for all tile geometry calculations.

**Class:** `TileGeometry`
- Calculates canvas size with padding margin for edge tiles
- Pre-computes all tile rectangles and padded regions
- Lazy-loads blend masks on demand
- Provides consistent padding methods for images and masks

**Key methods:**
- `pad_image(tensor)` - Mirror-pad image tensor to canvas size
- `pad_mask(tensor)` - Mirror-pad mask tensor to canvas size
- `get_tile_rect(idx)` - Get tile boundaries (non-overlapping grid)
- `get_padded_rect(idx)` - Get padded crop region for tile
- `get_blend_mask(idx)` - Get blend mask for compositing
- `get_tile_crop(pil, idx)` - Crop tile with padding context
- `create_edge_mask(idx, width, feather)` - Create edge mask for DiffDiff
- `crop_to_original(tensor)` - Crop result back to original size

**Benefits:**
- Eliminates duplicate geometry calculations
- Prevents bugs like double mirror padding
- Preview and processing use EXACT same data
- Easier to debug and test

---

### Configuration

#### `constants.py`
Mode constants used throughout the module.

**Exports:**
- `MAX_RESOLUTION = 8192` - Maximum allowed resolution
- `MODES` - Tile processing modes (Linear, Chess, None)
- `SEAM_FIX_MODES` - Seam fix modes (None, Band Pass, Half Tile, etc.)

#### `inputs.py`
Input type definitions for the ComfyUI node.

**Functions:**
- `USDU_edge_repair_inputs()` - Returns required and optional input definitions
- `prepare_inputs()` - Formats inputs for ComfyUI

---

### Validation

#### `validation.py`
Parameter validation to prevent invalid configurations.

**Function:**
- `validate_safeguard()` - Checks tile dimensions, padding, grid size
  - Ensures tiles fit within image bounds
  - Validates tile count matches conditioning count
  - Raises clear error messages for invalid configs

---

### Features

#### `diffdiff.py`
Differential Diffusion - per-pixel denoise control via grayscale mask.

**Class:** `DifferentialDiffusionAdvanced`
- White mask = more denoise
- Black mask = less denoise
- Patches model's `denoise_mask_function`
- Based on KJNodes implementation

#### `preview.py`
Preview mode for debugging tile configurations.

**Function:** `generate_tile_previews(geometry, upscaled_image, ...)`
- Takes TileGeometry instance as parameter (single source of truth)
- Returns 4 batches: `tiles_original`, `tiles_padded`, `tiles_blend_mask`, `tiles_edge_mask`
- All geometry from TileGeometry - guarantees preview matches processing

#### `controlnet.py`
ControlNet patches for per-tile application.

**Classes:**
- `ControlNetPatchBase` - Base class with shared functionality
  - `to()`, `models()` - Device/model management
  - `needs_reencode()`, `scale_image()` - Size matching helpers

- `DiffSynthCnetPatch(ControlNetPatchBase)` - For DiffSynth/Qwen models
  - Encodes control image to latent space
  - Applies control signal at each diffusion block

- `ZImageControlPatch(ControlNetPatchBase)` - For Z-Image ControlNet
  - Uses Flux latent format
  - Processes through multiple control layers

**Function:**
- `is_zimage_control(model_patch)` - Detect ControlNet type

---

### Core Processing

#### `processing.py`
Core sampling and tile compositing.

**Classes:**
- `StableDiffusionProcessing` - A1111-compatible processing container
  - Per-tile conditioning support
  - Per-tile denoise values
  - Tracks current tile index
  - Weighted accumulation for tile blending (fixes boundary artifacts)

- `Processed` - Result container

**Functions:**
- `sample()` - Run diffusion sampling with noise
- `sample_no_add_noise()` - Sample without adding noise (for Two-Latent Blend)
- `process_images()` - Main tile processing function
  - Crops tile from image
  - Crops conditioning to tile
  - Encodes to latent
  - Applies DiffDiff mask (optional)
  - Applies ControlNet (optional)
  - Samples with model
  - Decodes and composites back

#### `ultimate_upscale.py`
USDU core orchestration - the main upscaling engine.

**Classes:**
- `USDUpscaler` - Main upscaler class
- `USDURedraw` - Tile redraw pass
- `USDUSeamsFix` - Seam fixing pass
- `Script` - Entry point script

**Key methods:**
- `calc_rectangle()` - Calculate tile position (NON-OVERLAPPING grid)
- `start()` - Begin redraw/seam fix pass
- `run()` - Execute the full USDU pipeline

---

### Utilities

#### `utils.py`
25+ utility functions for image/tensor operations.

**Key functions:**
- `tensor_to_pil()` / `pil_to_tensor()` - Convert between formats
- `pad_image()` - Pad PIL image with edge fill
- `get_crop_region()` - Find white region in mask
- `expand_crop()` - Expand crop to target size
- `crop_cond()` - Crop conditioning to tile region
  - Internally calls: `crop_controlnet()`, `crop_gligen()`, `crop_area()`, `crop_mask()`, `crop_reference_latents()`

Note: Mirror padding is now handled by TileGeometry class.

#### `shared.py`
Global state container for A1111 compatibility.

**Variables:**
- `opts` - Options object with gradio compatibility
- `state` - Processing state with interruption support
- `sd_upscalers` - List of available upscalers
- `actual_upscaler` - Currently selected upscaler
- `batch` - List of PIL images being processed
- `batch_as_tensor` - Same images as tensor

#### `upscaler.py`
Image upscaling wrapper.

**Classes:**
- `Upscaler` - Base upscaler class
- `UpscalerData` - Upscaler configuration data

#### `usdu_patch.py`
Monkey patches for USDU to use 8px alignment instead of 64px.

**Patches:**
1. `USDUpscaler.__init__` - Round canvas to 8px
2. `USDURedraw.init_draw` - Round tile size to 8px
3. `USDUSeamsFix.init_draw` - Round seam fix to 8px
4. `USDUpscaler.upscale` - Handle batch images

---

## Import Chain

```
__init__.py
    └── node.py
        ├── tile_geometry.py    ← SINGLE SOURCE OF TRUTH
        ├── constants.py
        │   └── usdu_patch.py
        │       └── ultimate_upscale.py
        │           └── processing.py
        │               ├── utils.py
        │               ├── shared.py
        │               └── controlnet.py
        ├── inputs.py
        ├── validation.py
        ├── diffdiff.py
        ├── preview.py          ← uses TileGeometry
        ├── upscaler.py
        └── shared.py
```

---

## Data Flow

```
1. ComfyUI calls node.upscale()
   │
2. Create TileGeometry (single source of truth for all geometry)
   │
3. Pad image to canvas size using geometry.pad_image()
   │
4. Validate parameters (validation.py)
   │
5. Setup DiffDiff if enabled (diffdiff.py) - uses geometry.pad_mask()
   │
6. Setup ControlNet if enabled (node.py) - uses geometry.pad_image()
   │
7. Generate previews if preview_mode (preview.py uses TileGeometry)
   │
8. Create StableDiffusionProcessing (processing.py)
   │
9. Run USDU Script (usdu_patch.py → ultimate_upscale.py)
   │
   ├── USDUpscaler.upscale() - Upscale image to target size
   │
   ├── USDURedraw.start() - For each tile:
   │   └── process_images() - Sample and composite tile
   │       ├── Crop image to tile
   │       ├── Crop conditioning
   │       ├── Encode to latent
   │       ├── Apply DiffDiff mask
   │       ├── Apply ControlNet patch
   │       ├── Sample with model
   │       ├── Decode latent
   │       └── Composite back to image (weighted accumulation)
   │
   └── USDUSeamsFix.start() - Fix seams between tiles
       └── process_images() - Sample and composite seam
   │
10. Crop result to original size using geometry.crop_to_original()
   │
11. Return final image + preview outputs
```

---

## Key Design Decisions

1. **TileGeometry (Single Source of Truth)** - All geometry calculations in ONE class
   - Eliminates duplicate code and prevents bugs
   - Preview and processing use EXACT same data
   - Canvas size includes padding margin for edge tiles
2. **8px alignment** - Monkey patches change USDU from 64px to 8px for finer control
3. **Per-tile features** - Conditioning, denoise, and ControlNet can vary per tile
4. **A1111 compatibility** - StableDiffusionProcessing mimics A1111 API for USDU compatibility
5. **Weighted accumulation** - Proper tile blending using weighted average (fixes chess mode checker pattern)
6. **Modular design** - Each file has single responsibility for easy maintenance
