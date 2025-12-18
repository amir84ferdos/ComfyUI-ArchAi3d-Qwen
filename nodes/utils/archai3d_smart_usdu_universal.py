# Smart USDU Universal - ComfyUI Node
#
# A universal upscaling node with toggleable Differential Diffusion and ControlNet.
# Based on Smart USDU DiffDiff + ControlNet with added enable/disable toggles.
#
# Features:
# - Toggle Differential Diffusion ON/OFF without disconnecting wires
# - Toggle ControlNet ON/OFF without disconnecting wires
# - Per-tile ControlNet support (control image cropped per tile)
# - All features from Smart USDU DiffDiff + ControlNet
#
# Modified by Amir Ferdos (ArchAi3d)
#
# Version: 3.1.0 - Tile Feather Mask for color consistency
#          3.0.0 - NoUpscale V3 with debug outputs
#          1.0.0 - Initial release with toggleable features
# License: Dual License (Free for personal use, Commercial license required for business use)

import logging
import math
import torch
import torch.nn.functional as F
import comfy
from PIL import Image, ImageFilter
import numpy as np
from typing import List, Tuple, Optional
from .smart_usdu.usdu_patch import usdu
from .smart_usdu.utils import tensor_to_pil, pil_to_tensor
from .smart_usdu.processing import StableDiffusionProcessing
from .smart_usdu import shared
from .smart_usdu.upscaler import UpscalerData

MAX_RESOLUTION = 8192

# The modes available for Ultimate SD Upscale
MODES = {
    "Linear": usdu.USDUMode.LINEAR,
    "Chess": usdu.USDUMode.CHESS,
    "None": usdu.USDUMode.NONE,
}

# The seam fix modes
SEAM_FIX_MODES = {
    "None": usdu.USDUSFMode.NONE,
    "Band Pass": usdu.USDUSFMode.BAND_PASS,
    "Half Tile": usdu.USDUSFMode.HALF_TILE,
    "Half Tile + Intersections": usdu.USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS,
}


# ============================================================
# FEATHER MASK FUNCTIONS (v3.1)
# ============================================================

def create_tile_feather_mask(
    tile_h: int,
    tile_w: int,
    feather_width: int,
    feather_blur: int,
    is_left_edge: bool,
    is_right_edge: bool,
    is_top_edge: bool,
    is_bottom_edge: bool
) -> torch.Tensor:
    """
    Creates a feather mask for a tile.

    The mask has:
    - 1.0 in the center (full denoise)
    - Gradient from 1.0 to 0.0 at interior edges (feather zone)
    - 1.0 at outer image boundaries (no feather)

    Args:
        tile_h: Tile height in pixels
        tile_w: Tile width in pixels
        feather_width: Width of gradient zone in pixels
        feather_blur: Gaussian blur radius to smooth the gradient
        is_left_edge: True if tile is at left image boundary
        is_right_edge: True if tile is at right image boundary
        is_top_edge: True if tile is at top image boundary
        is_bottom_edge: True if tile is at bottom image boundary

    Returns:
        Tensor shape [H, W] with values 0.0-1.0
    """
    # Start with all 1.0 (full denoise everywhere)
    mask = torch.ones((tile_h, tile_w), dtype=torch.float32)

    # Create gradient for interior edges only
    # Left edge gradient (if not at image boundary)
    if not is_left_edge and feather_width > 0:
        for x in range(min(feather_width, tile_w)):
            # Gradient from 0.0 at edge to 1.0 at feather_width
            value = x / feather_width
            mask[:, x] = torch.minimum(mask[:, x], torch.tensor(value))

    # Right edge gradient (if not at image boundary)
    if not is_right_edge and feather_width > 0:
        for x in range(min(feather_width, tile_w)):
            # Gradient from 0.0 at edge to 1.0 at feather_width
            value = x / feather_width
            mask[:, tile_w - 1 - x] = torch.minimum(mask[:, tile_w - 1 - x], torch.tensor(value))

    # Top edge gradient (if not at image boundary)
    if not is_top_edge and feather_width > 0:
        for y in range(min(feather_width, tile_h)):
            # Gradient from 0.0 at edge to 1.0 at feather_width
            value = y / feather_width
            mask[y, :] = torch.minimum(mask[y, :], torch.tensor(value))

    # Bottom edge gradient (if not at image boundary)
    if not is_bottom_edge and feather_width > 0:
        for y in range(min(feather_width, tile_h)):
            # Gradient from 0.0 at edge to 1.0 at feather_width
            value = y / feather_width
            mask[tile_h - 1 - y, :] = torch.minimum(mask[tile_h - 1 - y, :], torch.tensor(value))

    # Apply Gaussian blur if requested
    if feather_blur > 0:
        # Convert to PIL for blur
        mask_np = (mask.numpy() * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_np, mode='L')
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=feather_blur))
        mask_np = np.array(mask_pil).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask_np)

    # Clamp to [0, 1]
    mask = torch.clamp(mask, 0.0, 1.0)

    return mask


def create_debug_grid(
    tiles: List[torch.Tensor],
    rows: int,
    cols: int,
    border_width: int = 2,
    border_color: Tuple[float, float, float] = (1.0, 0.0, 0.0)
) -> torch.Tensor:
    """
    Arrange tiles into a grid image for visual debugging.

    Args:
        tiles: List of tensors [H, W, C] or [H, W] (grayscale)
        rows: Number of rows in grid
        cols: Number of columns in grid
        border_width: Width of border between tiles
        border_color: RGB color for border (default red)

    Returns:
        Single tensor [1, H*rows + border*(rows-1), W*cols + border*(cols-1), C]
    """
    if len(tiles) == 0:
        return torch.zeros((1, 64, 64, 3))

    # Get tile dimensions
    tile_h, tile_w = tiles[0].shape[0], tiles[0].shape[1]

    # Handle grayscale vs RGB
    if len(tiles[0].shape) == 2:
        # Convert grayscale to RGB for visualization
        tiles = [t.unsqueeze(-1).repeat(1, 1, 3) for t in tiles]

    channels = tiles[0].shape[2]

    # Calculate grid dimensions
    grid_h = tile_h * rows + border_width * (rows - 1)
    grid_w = tile_w * cols + border_width * (cols - 1)

    # Create grid with border color
    grid = torch.ones((grid_h, grid_w, channels))
    for c in range(min(channels, 3)):
        grid[:, :, c] = border_color[c] if c < len(border_color) else 0.0

    # Place tiles in grid
    for idx, tile in enumerate(tiles):
        if idx >= rows * cols:
            break
        row = idx // cols
        col = idx % cols

        y_start = row * (tile_h + border_width)
        x_start = col * (tile_w + border_width)

        # Ensure tile fits
        actual_h = min(tile_h, tile.shape[0])
        actual_w = min(tile_w, tile.shape[1])

        grid[y_start:y_start + actual_h, x_start:x_start + actual_w, :] = tile[:actual_h, :actual_w, :]

    # Add batch dimension
    return grid.unsqueeze(0)


def create_combined_debug(
    tiles: List[torch.Tensor],
    masks: List[torch.Tensor],
    rows: int,
    cols: int,
    border_width: int = 2
) -> torch.Tensor:
    """
    Create debug image showing tiles with feather masks overlaid.

    Red tint shows the feather zones (where mask < 1.0).

    Args:
        tiles: List of tile images [H, W, C]
        masks: List of feather masks [H, W]
        rows, cols: Grid dimensions
        border_width: Border between tiles

    Returns:
        Combined debug image tensor
    """
    combined_tiles = []

    for tile, mask in zip(tiles, masks):
        # Ensure tile is RGB
        if len(tile.shape) == 2:
            tile = tile.unsqueeze(-1).repeat(1, 1, 3)

        # Create overlay: red tint where mask < 1.0
        # mask=1.0 -> no tint, mask=0.0 -> full red tint
        overlay = tile.clone()

        # Increase red channel, decrease green/blue based on inverse mask
        inverse_mask = 1.0 - mask
        overlay[:, :, 0] = torch.clamp(tile[:, :, 0] + inverse_mask * 0.5, 0, 1)  # Add red
        overlay[:, :, 1] = tile[:, :, 1] * (0.5 + mask * 0.5)  # Reduce green
        overlay[:, :, 2] = tile[:, :, 2] * (0.5 + mask * 0.5)  # Reduce blue

        combined_tiles.append(overlay)

    return create_debug_grid(combined_tiles, rows, cols, border_width, (0.0, 1.0, 0.0))  # Green border


def USDU_universal_inputs():
    """Inputs for Universal node with toggles."""
    required = [
        ("image", ("IMAGE",)),
        # Feature Toggles
        ("enable_diffdiff", ("BOOLEAN", {"default": True, "tooltip": "Enable/disable Differential Diffusion"})),
        ("enable_controlnet", ("BOOLEAN", {"default": True, "tooltip": "Enable/disable ControlNet per-tile"})),
        # Sampling Params
        ("model", ("MODEL",)),
        ("conditionings", ("CONDITIONING_LIST",)),  # Per-tile conditionings
        ("negative", ("CONDITIONING",)),
        ("vae", ("VAE",)),
        ("upscale_by", ("FLOAT", {"default": 2, "min": 0.05, "max": 4, "step": 0.05})),
        ("seed", ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})),
        ("steps", ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1})),
        ("cfg", ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0})),
        ("sampler_name", (comfy.samplers.KSampler.SAMPLERS,)),
        ("scheduler", (comfy.samplers.KSampler.SCHEDULERS,)),
        ("denoise", ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01})),
        # Upscale Params
        ("upscale_model", ("UPSCALE_MODEL",)),
        ("mode_type", (list(MODES.keys()),)),
        ("tile_width", ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8})),
        ("tile_height", ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8})),
        ("mask_blur", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1})),
        ("tile_padding", ("INT", {"default": 32, "min": 0, "max": MAX_RESOLUTION, "step": 8})),
        # Seam fix params
        ("seam_fix_mode", (list(SEAM_FIX_MODES.keys()),)),
        ("seam_fix_denoise", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})),
        ("seam_fix_width", ("INT", {"default": 64, "min": 0, "max": MAX_RESOLUTION, "step": 8})),
        ("seam_fix_mask_blur", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1})),
        ("seam_fix_padding", ("INT", {"default": 16, "min": 0, "max": MAX_RESOLUTION, "step": 8})),
        # Misc
        ("force_uniform_tiles", ("BOOLEAN", {"default": True})),
        ("tiled_decode", ("BOOLEAN", {"default": False})),
    ]

    # Optional inputs: DiffDiff mask + ControlNet
    optional = [
        # Differential Diffusion
        ("denoise_mask", ("MASK", {
            "tooltip": "Optional mask for per-pixel denoise. White=more denoise, Black=less"
        })),
        ("multiplier", ("FLOAT", {
            "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001,
            "tooltip": "Controls effect strength. <1=stronger, >1=weaker"
        })),
        # ControlNet per-tile
        ("model_patch", ("MODEL_PATCH", {
            "tooltip": "ControlNet patch (from ModelPatchLoader)"
        })),
        ("control_image", ("IMAGE", {
            "tooltip": "Control image (e.g., Canny/Depth) - will be cropped per tile"
        })),
        ("control_strength", ("FLOAT", {
            "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
            "tooltip": "ControlNet strength"
        })),
        ("control_mask", ("MASK", {
            "tooltip": "Optional mask for ControlNet (separate from denoise_mask)"
        })),
    ]

    return required, optional


def USDU_noupscale_inputs():
    """
    Inputs for NoUpscale variant - designed to work with Matrix Search.

    All tile/grid values come from Matrix Search - NO internal calculations.
    Includes safeguard validation to ensure values match exactly.
    """
    required = [
        ("upscaled_image", ("IMAGE", {"tooltip": "Pre-upscaled image from Matrix Search"})),
        # Matrix Search values - REQUIRED, no calculations
        ("output_width", ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 1,
                                   "tooltip": "Expected output width from Matrix Search"})),
        ("output_height", ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 1,
                                    "tooltip": "Expected output height from Matrix Search"})),
        ("tiles_x", ("INT", {"default": 2, "min": 1, "max": 64, "step": 1,
                             "tooltip": "Number of tile columns from Matrix Search"})),
        ("tiles_y", ("INT", {"default": 2, "min": 1, "max": 64, "step": 1,
                             "tooltip": "Number of tile rows from Matrix Search"})),
        # Safeguard toggle
        ("safe_guard", ("BOOLEAN", {"default": True,
                                     "tooltip": "Validate all values match Matrix Search exactly. Error if ANY mismatch."})),
        # Feature Toggles
        ("enable_diffdiff", ("BOOLEAN", {"default": True, "tooltip": "Enable/disable Differential Diffusion"})),
        ("enable_controlnet", ("BOOLEAN", {"default": True, "tooltip": "Enable/disable ControlNet per-tile"})),
        # Sampling Params
        ("model", ("MODEL",)),
        ("conditionings", ("CONDITIONING_LIST",)),  # Per-tile conditionings
        ("negative", ("CONDITIONING",)),
        ("vae", ("VAE",)),
        ("seed", ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})),
        ("steps", ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1})),
        ("cfg", ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0})),
        ("sampler_name", (comfy.samplers.KSampler.SAMPLERS,)),
        ("scheduler", (comfy.samplers.KSampler.SCHEDULERS,)),
        ("denoise", ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01})),
        # Tile Params - from Matrix Search
        ("mode_type", (list(MODES.keys()),)),
        ("tile_width", ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8,
                                "tooltip": "Tile width from Matrix Search"})),
        ("tile_height", ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8,
                                 "tooltip": "Tile height from Matrix Search"})),
        ("mask_blur", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1})),
        ("tile_padding", ("INT", {"default": 32, "min": 0, "max": MAX_RESOLUTION, "step": 8,
                                  "tooltip": "Overlap/padding from Matrix Search"})),
        # Seam fix params
        ("seam_fix_mode", (list(SEAM_FIX_MODES.keys()),)),
        ("seam_fix_denoise", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})),
        ("seam_fix_width", ("INT", {"default": 64, "min": 0, "max": MAX_RESOLUTION, "step": 8})),
        ("seam_fix_mask_blur", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1})),
        ("seam_fix_padding", ("INT", {"default": 16, "min": 0, "max": MAX_RESOLUTION, "step": 8})),
        # Misc
        ("tiled_decode", ("BOOLEAN", {"default": False})),
    ]

    # Optional inputs: DiffDiff mask + ControlNet
    optional = [
        # Differential Diffusion
        ("denoise_mask", ("MASK", {
            "tooltip": "Optional mask for per-pixel denoise. White=more denoise, Black=less"
        })),
        ("multiplier", ("FLOAT", {
            "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001,
            "tooltip": "Controls effect strength. <1=stronger, >1=weaker"
        })),
        # ControlNet per-tile
        ("model_patch", ("MODEL_PATCH", {
            "tooltip": "ControlNet patch (from ModelPatchLoader)"
        })),
        ("control_image", ("IMAGE", {
            "tooltip": "Control image (e.g., Canny/Depth) - will be cropped per tile"
        })),
        ("control_strength", ("FLOAT", {
            "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
            "tooltip": "ControlNet strength"
        })),
        ("control_mask", ("MASK", {
            "tooltip": "Optional mask for ControlNet (separate from denoise_mask)"
        })),
    ]

    return required, optional


def USDU_noupscale_v3_inputs():
    """
    Inputs for NoUpscale V3 - with Feather Mask + Debug Outputs.

    Adds:
    - feather_enabled: Toggle feather mask ON/OFF (default ON)
    - feather_width: Width of gradient zone
    - feather_blur: Gaussian blur radius
    - Debug outputs: tile images, feather masks, combined view
    """
    required = [
        ("upscaled_image", ("IMAGE", {"tooltip": "Pre-upscaled image from Matrix Search"})),
        # Matrix Search values
        ("output_width", ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 1})),
        ("output_height", ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 1})),
        ("tiles_x", ("INT", {"default": 2, "min": 1, "max": 64, "step": 1})),
        ("tiles_y", ("INT", {"default": 2, "min": 1, "max": 64, "step": 1})),
        ("safe_guard", ("BOOLEAN", {"default": True})),
        # Feature Toggles
        ("enable_diffdiff", ("BOOLEAN", {"default": True})),
        ("enable_controlnet", ("BOOLEAN", {"default": True})),
        # FEATHER MASK PARAMETERS (v3.1)
        ("feather_enabled", ("BOOLEAN", {"default": True, "tooltip": "Enable tile feather mask for color consistency"})),
        ("feather_width", ("INT", {"default": 16, "min": 4, "max": 128, "step": 4,
                                    "tooltip": "Width of gradient zone (must be < tile_padding)"})),
        ("feather_blur", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1,
                                   "tooltip": "Gaussian blur radius for feather gradient"})),
        # Sampling Params
        ("model", ("MODEL",)),
        ("conditionings", ("CONDITIONING_LIST",)),
        ("negative", ("CONDITIONING",)),
        ("vae", ("VAE",)),
        ("seed", ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})),
        ("steps", ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1})),
        ("cfg", ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0})),
        ("sampler_name", (comfy.samplers.KSampler.SAMPLERS,)),
        ("scheduler", (comfy.samplers.KSampler.SCHEDULERS,)),
        ("denoise", ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01})),
        # Tile Params
        ("mode_type", (list(MODES.keys()),)),
        ("tile_width", ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8})),
        ("tile_height", ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8})),
        ("mask_blur", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1})),
        ("tile_padding", ("INT", {"default": 32, "min": 0, "max": MAX_RESOLUTION, "step": 8})),
        # Seam fix params
        ("seam_fix_mode", (list(SEAM_FIX_MODES.keys()),)),
        ("seam_fix_denoise", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})),
        ("seam_fix_width", ("INT", {"default": 64, "min": 0, "max": MAX_RESOLUTION, "step": 8})),
        ("seam_fix_mask_blur", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1})),
        ("seam_fix_padding", ("INT", {"default": 16, "min": 0, "max": MAX_RESOLUTION, "step": 8})),
        # Misc
        ("tiled_decode", ("BOOLEAN", {"default": False})),
    ]

    optional = [
        ("denoise_mask", ("MASK", {"tooltip": "Optional mask for per-pixel denoise"})),
        ("multiplier", ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001})),
        ("model_patch", ("MODEL_PATCH", {"tooltip": "ControlNet patch"})),
        ("control_image", ("IMAGE", {"tooltip": "Control image for ControlNet"})),
        ("control_strength", ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})),
        ("control_mask", ("MASK", {"tooltip": "Optional mask for ControlNet"})),
    ]

    return required, optional


def prepare_inputs(required: list, optional: list = None):
    inputs = {}
    if required:
        inputs["required"] = {}
        for name, type in required:
            inputs["required"][name] = type
    if optional:
        inputs["optional"] = {}
        for name, type in optional:
            inputs["optional"][name] = type
    return inputs


def remove_input(inputs: list, input_name: str):
    for i, (n, _) in enumerate(inputs):
        if n == input_name:
            del inputs[i]
            break


def rename_input(inputs: list, old_name: str, new_name: str):
    for i, (n, t) in enumerate(inputs):
        if n == old_name:
            inputs[i] = (new_name, t)
            break


def validate_safeguard(image, output_width, output_height,
                       tile_width, tile_height, tile_padding,
                       tiles_x, tiles_y):
    """
    Validate all values from Matrix Search match exactly.

    Raises ValueError if ANY mismatch detected (even 1 pixel difference).
    This ensures USDU uses exactly what Matrix Search calculated.

    Args:
        image: Input image tensor (BHWC format)
        output_width: Expected width from Matrix Search
        output_height: Expected height from Matrix Search
        tile_width: Tile width from Matrix Search
        tile_height: Tile height from Matrix Search
        tile_padding: Overlap/padding from Matrix Search
        tiles_x: Number of tile columns from Matrix Search
        tiles_y: Number of tile rows from Matrix Search

    Returns:
        True if all checks pass

    Raises:
        ValueError: If any mismatch detected
    """
    errors = []

    # 1. Check image dimensions match expected output
    img_h, img_w = image.shape[1], image.shape[2]
    if img_w != output_width:
        errors.append(f"output_width: expected {output_width}, got {img_w} (diff: {abs(img_w - output_width)}px)")
    if img_h != output_height:
        errors.append(f"output_height: expected {output_height}, got {img_h} (diff: {abs(img_h - output_height)}px)")

    # 2. Verify coverage formula
    # coverage = tile * n - padding * (n - 1)
    if tiles_x > 1:
        coverage_w = tile_width * tiles_x - tile_padding * (tiles_x - 1)
    else:
        coverage_w = tile_width

    if tiles_y > 1:
        coverage_h = tile_height * tiles_y - tile_padding * (tiles_y - 1)
    else:
        coverage_h = tile_height

    if coverage_w < output_width:
        errors.append(f"coverage_width: {coverage_w}px < output_width {output_width}px (tiles don't cover image!)")
    if coverage_h < output_height:
        errors.append(f"coverage_height: {coverage_h}px < output_height {output_height}px (tiles don't cover image!)")

    # 3. If any errors, raise exception with detailed message
    if errors:
        error_msg = "\n".join([
            "",
            "=" * 60,
            "❌ SAFEGUARD ERROR: Mismatch detected!",
            "=" * 60,
            "",
            "The values from Matrix Search don't match the actual image.",
            "This could cause incorrect tiling. Please check your connections.",
            "",
            "ERRORS:",
        ] + [f"  • {e}" for e in errors] + [
            "",
            "EXPECTED VALUES (from Matrix Search inputs):",
            f"  output_width: {output_width}",
            f"  output_height: {output_height}",
            f"  tile_width: {tile_width}",
            f"  tile_height: {tile_height}",
            f"  tile_padding: {tile_padding}",
            f"  tiles_x: {tiles_x}",
            f"  tiles_y: {tiles_y}",
            "",
            "ACTUAL IMAGE:",
            f"  width: {img_w}",
            f"  height: {img_h}",
            "",
            "COVERAGE CHECK:",
            f"  coverage_w: {coverage_w} (needs >= {output_width})",
            f"  coverage_h: {coverage_h} (needs >= {output_height})",
            "=" * 60,
        ])
        raise ValueError(error_msg)

    return True  # All checks passed


# ============================================================
# Copied EXACTLY from KJNodes DifferentialDiffusionAdvanced
# Source: comfyui-kjnodes/nodes/nodes.py lines 1743-1779
# DO NOT MODIFY THIS CODE
# ============================================================
class DifferentialDiffusionAdvanced():
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL", ),
                    "samples": ("LATENT",),
                    "mask": ("MASK",),
                    "multiplier": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001}),
                            }}
    RETURN_TYPES = ("MODEL", "LATENT")
    FUNCTION = "apply"
    CATEGORY = "_for_testing"
    INIT = False

    def apply(self, model, samples, mask, multiplier):
        self.multiplier = multiplier
        model = model.clone()
        model.set_model_denoise_mask_function(self.forward)
        s = samples.copy()
        s["noise_mask"] = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
        return (model, s)

    def forward(self, sigma: torch.Tensor, denoise_mask: torch.Tensor, extra_options: dict):
        model = extra_options["model"]
        step_sigmas = extra_options["sigmas"]
        sigma_to = model.inner_model.model_sampling.sigma_min
        if step_sigmas[-1] > sigma_to:
            sigma_to = step_sigmas[-1]
        sigma_from = step_sigmas[0]

        ts_from = model.inner_model.model_sampling.timestep(sigma_from)
        ts_to = model.inner_model.model_sampling.timestep(sigma_to)
        current_ts = model.inner_model.model_sampling.timestep(sigma[0])

        threshold = (current_ts - ts_to) / (ts_from - ts_to) / self.multiplier

        return (denoise_mask >= threshold).to(denoise_mask.dtype)
# ============================================================
# END OF KJNODES CODE
# ============================================================


class ArchAi3D_Smart_USDU_Universal:
    """
    Smart USDU Universal - Toggleable Differential Diffusion + ControlNet.

    Features:
    - enable_diffdiff: Toggle Differential Diffusion ON/OFF
    - enable_controlnet: Toggle ControlNet ON/OFF
    - All features from Smart USDU DiffDiff + ControlNet

    How ControlNet tiling works:
    - Control image (Canny/Depth/etc.) is upscaled to output size
    - For each tile, the control image is CROPPED to same region as main image
    - ControlNet patch is applied with cropped control image
    - Each tile "sees" only its corresponding region of the control image

    Tile order: row-major (left-to-right, top-to-bottom)
    Same order as Smart Tile Prompter output.
    """

    @classmethod
    def INPUT_TYPES(s):
        required, optional = USDU_universal_inputs()
        return prepare_inputs(required, optional)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "ArchAi3d/Upscaling"

    def upscale(self, image, enable_diffdiff, enable_controlnet, model, conditionings, negative, vae, upscale_by, seed,
                steps, cfg, sampler_name, scheduler, denoise, upscale_model,
                mode_type, tile_width, tile_height, mask_blur, tile_padding,
                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode,
                custom_sampler=None, custom_sigmas=None,
                denoise_mask=None, multiplier=1.0,
                model_patch=None, control_image=None, control_strength=1.0, control_mask=None):
        # Store params
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.mask_blur = mask_blur
        self.tile_padding = tile_padding
        self.seam_fix_width = seam_fix_width
        self.seam_fix_denoise = seam_fix_denoise
        self.seam_fix_padding = seam_fix_padding
        self.seam_fix_mode = seam_fix_mode
        self.mode_type = mode_type
        self.upscale_by = upscale_by
        self.seam_fix_mask_blur = seam_fix_mask_blur

        # ===== DEBUG INFO =====
        input_h, input_w = image.shape[1], image.shape[2]
        output_w = int(input_w * upscale_by)
        output_h = int(input_h * upscale_by)
        num_conditionings = len(conditionings) if isinstance(conditionings, list) else 1

        # USDU formula: rows = ceil(height / tile_height), cols = ceil(width / tile_width)
        cols = math.ceil(output_w / tile_width)
        rows = math.ceil(output_h / tile_height)
        total_tiles = rows * cols

        # Actual sampling size (rounded up to 64 for VAE)
        sampling_width = math.ceil((tile_width + tile_padding) / 64) * 64
        sampling_height = math.ceil((tile_height + tile_padding) / 64) * 64

        print("\n" + "=" * 60)
        print("🎛️ Smart USDU Universal - DEBUG INFO")
        print("=" * 60)
        print(f"INPUT IMAGE:")
        print(f"  Size: {input_w}x{input_h}")
        print(f"  Upscale by: {upscale_by}x")
        print(f"  Output size: {output_w}x{output_h}")
        print(f"\nTILE SETTINGS:")
        print(f"  tile_width: {tile_width}px (visible area per tile)")
        print(f"  tile_height: {tile_height}px (visible area per tile)")
        print(f"  tile_padding: {tile_padding}px (extra context around each tile)")
        print(f"  mask_blur: {mask_blur}px (blending at tile edges)")
        print(f"\nCALCULATED GRID (USDU formula: ceil(size/tile_size)):")
        print(f"  Rows x Cols: {rows}x{cols} = {total_tiles} tiles")
        print(f"  Sampling size per tile: {sampling_width}x{sampling_height}px (rounded to 64)")
        print(f"\nSEAM FIX SETTINGS:")
        print(f"  Mode: {seam_fix_mode}")
        print(f"  seam_fix_denoise: {seam_fix_denoise}")
        print(f"\nSAMPLING:")
        print(f"  Mode: {mode_type}")
        print(f"  Steps: {steps}, CFG: {cfg}, Denoise: {denoise}")
        print(f"  Sampler: {sampler_name}, Scheduler: {scheduler}")
        print(f"  Seed: {seed}")
        print(f"\nCONDITIONING:")
        print(f"  Number of conditionings: {num_conditionings}")
        print(f"  Expected tiles: {total_tiles}")
        if num_conditionings != total_tiles:
            print(f"  WARNING: Conditioning count ({num_conditionings}) != tile count ({total_tiles})")

        # ===== DIFFERENTIAL DIFFUSION INFO =====
        # Check BOTH toggle AND mask
        diff_diff_enabled = enable_diffdiff and denoise_mask is not None
        print(f"\nDIFFERENTIAL DIFFUSION:")
        print(f"  Toggle: {'ENABLED' if enable_diffdiff else 'DISABLED'}")
        if enable_diffdiff:
            if denoise_mask is not None:
                print(f"  Status: ACTIVE")
                print(f"  Multiplier: {multiplier}")
                print(f"  Mask shape: {denoise_mask.shape}")
                print(f"  Effect: White=more denoise, Black=less denoise")
            else:
                print(f"  Status: INACTIVE (no mask provided)")
        else:
            print(f"  Status: DISABLED by toggle")
            if denoise_mask is not None:
                print(f"  Note: Mask is connected but feature is disabled")

        # ===== CONTROLNET INFO =====
        # Check BOTH toggle AND inputs
        controlnet_enabled = enable_controlnet and model_patch is not None and control_image is not None
        print(f"\nCONTROLNET (Per-Tile):")
        print(f"  Toggle: {'ENABLED' if enable_controlnet else 'DISABLED'}")
        if enable_controlnet:
            if model_patch is not None and control_image is not None:
                # Detect ControlNet type
                try:
                    import comfy.ldm.lumina.controlnet
                    is_zimage = isinstance(model_patch.model, comfy.ldm.lumina.controlnet.ZImage_Control)
                except (ImportError, AttributeError):
                    is_zimage = hasattr(model_patch.model, 'n_control_layers')
                cnet_type = "Z-Image" if is_zimage else "DiffSynth/Qwen"
                print(f"  Status: ACTIVE")
                print(f"  Type: {cnet_type}")
                print(f"  Control image shape: {control_image.shape}")
                print(f"  Strength: {control_strength}")
                if not is_zimage:
                    print(f"  Control mask: {'PROVIDED' if control_mask is not None else 'NONE'}")
                    if control_mask is not None:
                        print(f"  Control mask shape: {control_mask.shape}")
                else:
                    print(f"  Control mask: N/A (Z-Image doesn't support mask)")
                print(f"  Method: Crop control image per tile (same region as main image)")
            else:
                print(f"  Status: INACTIVE (missing model_patch or control_image)")
        else:
            print(f"  Status: DISABLED by toggle")
            if model_patch is not None or control_image is not None:
                print(f"  Note: Inputs are connected but feature is disabled")
        print("=" * 60 + "\n")

        # ===== APPLY DIFFERENTIAL DIFFUSION IF ENABLED =====
        denoise_mask_upscaled = None
        if diff_diff_enabled:
            # Upscale mask to match output image size
            if denoise_mask.dim() == 3:
                mask_2d = denoise_mask[0]
            else:
                mask_2d = denoise_mask

            # Resize mask to output size (same as upscaled image)
            mask_upscaled = F.interpolate(
                mask_2d.unsqueeze(0).unsqueeze(0),
                size=(output_h, output_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)

            print(f"[DiffDiff] Mask upscaled: {mask_2d.shape} -> {mask_upscaled.shape}")

            # Use DifferentialDiffusionAdvanced to patch model
            diff_diff = DifferentialDiffusionAdvanced()
            diff_diff.multiplier = multiplier
            model = model.clone()
            model.set_model_denoise_mask_function(diff_diff.forward)

            print(f"[DiffDiff] Model patched with multiplier={multiplier}")

            # Store upscaled mask for per-tile processing
            denoise_mask_upscaled = mask_upscaled

        # ===== UPSCALE CONTROL IMAGE TO OUTPUT SIZE =====
        control_image_upscaled = None
        control_mask_upscaled = None
        if controlnet_enabled:
            # Resize control image to output size (same as upscaled main image)
            # This ensures tile cropping coordinates align perfectly
            control_image_upscaled = F.interpolate(
                control_image.movedim(-1, 1),  # BHWC -> BCHW
                size=(output_h, output_w),
                mode='bilinear',
                align_corners=False
            ).movedim(1, -1)  # BCHW -> BHWC
            print(f"[ControlNet] Control image upscaled: {control_image.shape} -> {control_image_upscaled.shape}")

            # Upscale control mask if provided
            if control_mask is not None:
                # Handle 2D or 3D mask
                if control_mask.dim() == 3:
                    ctrl_mask_2d = control_mask[0]
                else:
                    ctrl_mask_2d = control_mask

                # Resize to output size (same as control image)
                control_mask_upscaled = F.interpolate(
                    ctrl_mask_2d.unsqueeze(0).unsqueeze(0),
                    size=(output_h, output_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)
                print(f"[ControlNet] Control mask upscaled: {ctrl_mask_2d.shape} -> {control_mask_upscaled.shape}")

        #
        # Set up A1111 patches
        #

        # Upscaler
        shared.sd_upscalers[0] = UpscalerData()
        shared.actual_upscaler = upscale_model

        # Set the batch of images
        shared.batch = [tensor_to_pil(image, i) for i in range(len(image))]
        shared.batch_as_tensor = image

        # Processing - Pass conditionings list + optional masks + ControlNet params
        # Only pass ControlNet params if enabled
        sdprocessing = StableDiffusionProcessing(
            shared.batch[0], model, conditionings, negative, vae,
            seed, steps, cfg, sampler_name, scheduler, denoise, upscale_by, force_uniform_tiles, tiled_decode,
            tile_width, tile_height, MODES[self.mode_type], SEAM_FIX_MODES[self.seam_fix_mode],
            custom_sampler, custom_sigmas,
            denoise_mask_tensor=denoise_mask_upscaled,
            # ControlNet per-tile parameters (only if enabled)
            model_patch=model_patch if controlnet_enabled else None,
            control_image_tensor=control_image_upscaled if controlnet_enabled else None,
            control_strength=control_strength if controlnet_enabled else 1.0,
            control_mask_tensor=control_mask_upscaled if controlnet_enabled else None,
        )

        # Disable logging
        logger = logging.getLogger()
        old_level = logger.getEffectiveLevel()
        logger.setLevel(logging.CRITICAL + 1)
        try:
            # Running the script
            script = usdu.Script()
            processed = script.run(p=sdprocessing, _=None, tile_width=self.tile_width, tile_height=self.tile_height,
                               mask_blur=self.mask_blur, padding=self.tile_padding, seams_fix_width=self.seam_fix_width,
                               seams_fix_denoise=self.seam_fix_denoise, seams_fix_padding=self.seam_fix_padding,
                               upscaler_index=0, save_upscaled_image=False, redraw_mode=MODES[self.mode_type],
                               save_seams_fix_image=False, seams_fix_mask_blur=self.seam_fix_mask_blur,
                               seams_fix_type=SEAM_FIX_MODES[self.seam_fix_mode], target_size_type=2,
                               custom_width=None, custom_height=None, custom_scale=self.upscale_by)

            # Return the resulting images
            images = [pil_to_tensor(img) for img in shared.batch]
            tensor = torch.cat(images, dim=0)
            return (tensor,)
        finally:
            # Restore the original logging level
            logger.setLevel(old_level)


class ArchAi3D_Smart_USDU_Universal_NoUpscale:
    """
    Smart USDU Universal (No Upscale) - Designed for Matrix Search.

    This variant:
    - Takes pre-upscaled image from Matrix Search
    - Uses ALL tile/grid values from Matrix Search inputs (NO internal calculations)
    - Includes safeguard validation to ensure values match exactly

    IMPORTANT: All values must come from Matrix Search:
    - output_width, output_height
    - tile_width, tile_height, tile_padding
    - tiles_x, tiles_y
    """

    @classmethod
    def INPUT_TYPES(s):
        required, optional = USDU_noupscale_inputs()
        return prepare_inputs(required, optional)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "ArchAi3d/Upscaling"

    def upscale(self, upscaled_image, output_width, output_height, tiles_x, tiles_y, safe_guard,
                enable_diffdiff, enable_controlnet, model, conditionings, negative, vae, seed,
                steps, cfg, sampler_name, scheduler, denoise,
                mode_type, tile_width, tile_height, mask_blur, tile_padding,
                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding, tiled_decode,
                denoise_mask=None, multiplier=1.0,
                model_patch=None, control_image=None, control_strength=1.0, control_mask=None):
        """
        Process upscaled image using values from Matrix Search.

        ALL tile/grid values are used directly from inputs - NO calculations.
        """
        # Store params
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.mask_blur = mask_blur
        self.tile_padding = tile_padding
        self.seam_fix_width = seam_fix_width
        self.seam_fix_denoise = seam_fix_denoise
        self.seam_fix_padding = seam_fix_padding
        self.seam_fix_mode = seam_fix_mode
        self.mode_type = mode_type
        self.seam_fix_mask_blur = seam_fix_mask_blur

        # ===== SAFEGUARD VALIDATION =====
        if safe_guard:
            validate_safeguard(
                upscaled_image, output_width, output_height,
                tile_width, tile_height, tile_padding,
                tiles_x, tiles_y
            )

        # ===== USE INPUT VALUES DIRECTLY - NO CALCULATIONS =====
        img_h, img_w = upscaled_image.shape[1], upscaled_image.shape[2]
        output_w = output_width   # FROM INPUT - not calculated
        output_h = output_height  # FROM INPUT - not calculated
        cols = tiles_x            # FROM INPUT - not calculated
        rows = tiles_y            # FROM INPUT - not calculated
        total_tiles = tiles_x * tiles_y  # FROM INPUT values

        num_conditionings = len(conditionings) if isinstance(conditionings, list) else 1

        # Actual sampling size (rounded up to 64 for VAE)
        sampling_width = math.ceil((tile_width + tile_padding) / 64) * 64
        sampling_height = math.ceil((tile_height + tile_padding) / 64) * 64

        # ===== DEBUG INFO - Values from Matrix Search =====
        print("\n" + "=" * 60)
        print("🎛️ Smart USDU Universal (No Upscale) - MATRIX SEARCH MODE")
        print("=" * 60)
        print(f"SAFEGUARD: {'✅ ENABLED' if safe_guard else '⚠️ DISABLED'}")
        print(f"\nINPUT IMAGE (pre-upscaled from Matrix Search):")
        print(f"  Actual size: {img_w}x{img_h}")
        print(f"\nVALUES FROM MATRIX SEARCH (used directly, no calculation):")
        print(f"  output_width: {output_width}px")
        print(f"  output_height: {output_height}px")
        print(f"  tile_width: {tile_width}px")
        print(f"  tile_height: {tile_height}px")
        print(f"  tile_padding (overlap): {tile_padding}px")
        print(f"  tiles_x: {tiles_x}")
        print(f"  tiles_y: {tiles_y}")
        print(f"  total_tiles: {total_tiles}")
        print(f"\nGRID LAYOUT:")
        print(f"  Rows x Cols: {rows}x{cols} = {total_tiles} tiles")
        print(f"  Sampling size per tile: {sampling_width}x{sampling_height}px (rounded to 64)")
        print(f"\nSEAM FIX SETTINGS:")
        print(f"  Mode: {seam_fix_mode}")
        print(f"  seam_fix_denoise: {seam_fix_denoise}")
        print(f"\nSAMPLING:")
        print(f"  Mode: {mode_type}")
        print(f"  Steps: {steps}, CFG: {cfg}, Denoise: {denoise}")
        print(f"  Sampler: {sampler_name}, Scheduler: {scheduler}")
        print(f"  Seed: {seed}")
        print(f"\nCONDITIONING:")
        print(f"  Number of conditionings: {num_conditionings}")
        print(f"  Expected tiles: {total_tiles}")
        if num_conditionings != total_tiles:
            print(f"  ⚠️ WARNING: Conditioning count ({num_conditionings}) != tile count ({total_tiles})")

        # ===== DIFFERENTIAL DIFFUSION INFO =====
        diff_diff_enabled = enable_diffdiff and denoise_mask is not None
        print(f"\nDIFFERENTIAL DIFFUSION:")
        print(f"  Toggle: {'ENABLED' if enable_diffdiff else 'DISABLED'}")
        if enable_diffdiff:
            if denoise_mask is not None:
                print(f"  Status: ACTIVE")
                print(f"  Multiplier: {multiplier}")
                print(f"  Mask shape: {denoise_mask.shape}")
            else:
                print(f"  Status: INACTIVE (no mask provided)")
        else:
            print(f"  Status: DISABLED by toggle")

        # ===== CONTROLNET INFO =====
        controlnet_enabled = enable_controlnet and model_patch is not None and control_image is not None
        print(f"\nCONTROLNET (Per-Tile):")
        print(f"  Toggle: {'ENABLED' if enable_controlnet else 'DISABLED'}")
        if enable_controlnet:
            if model_patch is not None and control_image is not None:
                try:
                    import comfy.ldm.lumina.controlnet
                    is_zimage = isinstance(model_patch.model, comfy.ldm.lumina.controlnet.ZImage_Control)
                except (ImportError, AttributeError):
                    is_zimage = hasattr(model_patch.model, 'n_control_layers')
                cnet_type = "Z-Image" if is_zimage else "DiffSynth/Qwen"
                print(f"  Status: ACTIVE")
                print(f"  Type: {cnet_type}")
                print(f"  Control image shape: {control_image.shape}")
                print(f"  Strength: {control_strength}")
            else:
                print(f"  Status: INACTIVE (missing model_patch or control_image)")
        else:
            print(f"  Status: DISABLED by toggle")
        print("=" * 60 + "\n")

        # ===== APPLY DIFFERENTIAL DIFFUSION IF ENABLED =====
        denoise_mask_upscaled = None
        if diff_diff_enabled:
            if denoise_mask.dim() == 3:
                mask_2d = denoise_mask[0]
            else:
                mask_2d = denoise_mask

            # Resize mask to output size
            mask_upscaled = F.interpolate(
                mask_2d.unsqueeze(0).unsqueeze(0),
                size=(output_h, output_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)

            print(f"[DiffDiff] Mask upscaled: {mask_2d.shape} -> {mask_upscaled.shape}")

            diff_diff = DifferentialDiffusionAdvanced()
            diff_diff.multiplier = multiplier
            model = model.clone()
            model.set_model_denoise_mask_function(diff_diff.forward)

            print(f"[DiffDiff] Model patched with multiplier={multiplier}")
            denoise_mask_upscaled = mask_upscaled

        # ===== UPSCALE CONTROL IMAGE TO OUTPUT SIZE =====
        control_image_upscaled = None
        control_mask_upscaled = None
        if controlnet_enabled:
            control_image_upscaled = F.interpolate(
                control_image.movedim(-1, 1),
                size=(output_h, output_w),
                mode='bilinear',
                align_corners=False
            ).movedim(1, -1)
            print(f"[ControlNet] Control image upscaled: {control_image.shape} -> {control_image_upscaled.shape}")

            if control_mask is not None:
                if control_mask.dim() == 3:
                    ctrl_mask_2d = control_mask[0]
                else:
                    ctrl_mask_2d = control_mask

                control_mask_upscaled = F.interpolate(
                    ctrl_mask_2d.unsqueeze(0).unsqueeze(0),
                    size=(output_h, output_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)
                print(f"[ControlNet] Control mask upscaled: {ctrl_mask_2d.shape} -> {control_mask_upscaled.shape}")

        # Set up A1111 patches
        shared.sd_upscalers[0] = UpscalerData()
        shared.actual_upscaler = None  # No upscaler needed - image is pre-upscaled

        # Set the batch of images
        shared.batch = [tensor_to_pil(upscaled_image, i) for i in range(len(upscaled_image))]
        shared.batch_as_tensor = upscaled_image

        # Processing - upscale_by = 1.0 since image is already upscaled
        # force_uniform_tiles = True (hardcoded - Matrix Search already calculated uniform tiles)
        sdprocessing = StableDiffusionProcessing(
            shared.batch[0], model, conditionings, negative, vae,
            seed, steps, cfg, sampler_name, scheduler, denoise, 1.0, True, tiled_decode,
            tile_width, tile_height, MODES[self.mode_type], SEAM_FIX_MODES[self.seam_fix_mode],
            None, None,  # custom_sampler, custom_sigmas
            denoise_mask_tensor=denoise_mask_upscaled,
            model_patch=model_patch if controlnet_enabled else None,
            control_image_tensor=control_image_upscaled if controlnet_enabled else None,
            control_strength=control_strength if controlnet_enabled else 1.0,
            control_mask_tensor=control_mask_upscaled if controlnet_enabled else None,
        )

        # Disable logging
        logger = logging.getLogger()
        old_level = logger.getEffectiveLevel()
        logger.setLevel(logging.CRITICAL + 1)
        try:
            script = usdu.Script()
            processed = script.run(p=sdprocessing, _=None, tile_width=self.tile_width, tile_height=self.tile_height,
                               mask_blur=self.mask_blur, padding=self.tile_padding, seams_fix_width=self.seam_fix_width,
                               seams_fix_denoise=self.seam_fix_denoise, seams_fix_padding=self.seam_fix_padding,
                               upscaler_index=0, save_upscaled_image=False, redraw_mode=MODES[self.mode_type],
                               save_seams_fix_image=False, seams_fix_mask_blur=self.seam_fix_mask_blur,
                               seams_fix_type=SEAM_FIX_MODES[self.seam_fix_mode], target_size_type=2,
                               custom_width=None, custom_height=None, custom_scale=1.0)

            images = [pil_to_tensor(img) for img in shared.batch]
            tensor = torch.cat(images, dim=0)
            return (tensor,)
        finally:
            logger.setLevel(old_level)


class ArchAi3D_Smart_USDU_Universal_NoUpscale_V2:
    """
    Smart USDU Universal (No Upscale) V2 - Enhanced Debug Output.

    Same as NoUpscale but with comprehensive debug showing:
    - Pixel dimensions (tile_width, tile_height, tile_padding, mask_blur)
    - Latent dimensions (pixel / 8 for VAE)
    - Final image size
    - Coverage verification
    """

    @classmethod
    def INPUT_TYPES(s):
        required, optional = USDU_noupscale_inputs()
        return prepare_inputs(required, optional)

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "debug_info")
    FUNCTION = "upscale"
    CATEGORY = "ArchAi3d/Upscaling"

    def upscale(self, upscaled_image, output_width, output_height, tiles_x, tiles_y, safe_guard,
                enable_diffdiff, enable_controlnet, model, conditionings, negative, vae, seed,
                steps, cfg, sampler_name, scheduler, denoise,
                mode_type, tile_width, tile_height, mask_blur, tile_padding,
                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding, tiled_decode,
                denoise_mask=None, multiplier=1.0,
                model_patch=None, control_image=None, control_strength=1.0, control_mask=None):
        """
        Process upscaled image using values from Matrix Search.
        V2: Enhanced debug output with latent dimensions.
        """
        # Store params
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.mask_blur = mask_blur
        self.tile_padding = tile_padding
        self.seam_fix_width = seam_fix_width
        self.seam_fix_denoise = seam_fix_denoise
        self.seam_fix_padding = seam_fix_padding
        self.seam_fix_mode = seam_fix_mode
        self.mode_type = mode_type
        self.seam_fix_mask_blur = seam_fix_mask_blur

        # ===== SAFEGUARD VALIDATION =====
        if safe_guard:
            validate_safeguard(
                upscaled_image, output_width, output_height,
                tile_width, tile_height, tile_padding,
                tiles_x, tiles_y
            )

        # ===== USE INPUT VALUES DIRECTLY - NO CALCULATIONS =====
        img_h, img_w = upscaled_image.shape[1], upscaled_image.shape[2]
        output_w = output_width   # FROM INPUT - not calculated
        output_h = output_height  # FROM INPUT - not calculated
        cols = tiles_x            # FROM INPUT - not calculated
        rows = tiles_y            # FROM INPUT - not calculated
        total_tiles = tiles_x * tiles_y  # FROM INPUT values

        num_conditionings = len(conditionings) if isinstance(conditionings, list) else 1

        # Actual sampling size (rounded up to 64 for VAE)
        sampling_width = math.ceil((tile_width + tile_padding) / 64) * 64
        sampling_height = math.ceil((tile_height + tile_padding) / 64) * 64

        # Coverage calculation
        if tiles_x > 1:
            coverage_w = tile_width * tiles_x - tile_padding * (tiles_x - 1)
        else:
            coverage_w = tile_width
        if tiles_y > 1:
            coverage_h = tile_height * tiles_y - tile_padding * (tiles_y - 1)
        else:
            coverage_h = tile_height

        # ===== V2 ENHANCED DEBUG OUTPUT (STRING) =====
        debug_lines = []
        debug_lines.append("=" * 60)
        debug_lines.append("Smart USDU Universal (No Upscale) V2")
        debug_lines.append("=" * 60)
        debug_lines.append(f"SAFEGUARD: {'ENABLED' if safe_guard else 'DISABLED'}")
        debug_lines.append("")
        debug_lines.append("PIXEL DIMENSIONS:")
        debug_lines.append(f"  tile_width:     {tile_width}px")
        debug_lines.append(f"  tile_height:    {tile_height}px")
        debug_lines.append(f"  tile_padding:   {tile_padding}px (overlap)")
        debug_lines.append(f"  mask_blur:      {mask_blur}px")
        debug_lines.append(f"  output_size:    {output_width} x {output_height}px")
        debug_lines.append("")
        debug_lines.append("LATENT DIMENSIONS (pixel / 8 for VAE):")
        debug_lines.append(f"  latent_tile_w:  {tile_width // 8}")
        debug_lines.append(f"  latent_tile_h:  {tile_height // 8}")
        debug_lines.append(f"  latent_padding: {tile_padding // 8}")
        debug_lines.append(f"  latent_blur:    {mask_blur // 8}")
        debug_lines.append(f"  latent_output:  {output_width // 8} x {output_height // 8}")
        debug_lines.append("")
        debug_lines.append("GRID LAYOUT (from Matrix Search):")
        debug_lines.append(f"  tiles_x:        {tiles_x}")
        debug_lines.append(f"  tiles_y:        {tiles_y}")
        debug_lines.append(f"  total_tiles:    {total_tiles}")
        debug_lines.append(f"  sampling_size:  {sampling_width} x {sampling_height}px")
        debug_lines.append("")
        debug_lines.append("COVERAGE CHECK:")
        debug_lines.append(f"  coverage:       {coverage_w} x {coverage_h}px")
        debug_lines.append(f"  output_size:    {output_width} x {output_height}px")
        coverage_ok = coverage_w >= output_width and coverage_h >= output_height
        debug_lines.append(f"  status:         {'OK' if coverage_ok else 'INSUFFICIENT'}")
        debug_lines.append("")
        debug_lines.append("INPUT IMAGE (pre-upscaled):")
        debug_lines.append(f"  actual_size:    {img_w} x {img_h}px")
        debug_lines.append("")
        debug_lines.append("SAMPLING:")
        debug_lines.append(f"  mode:           {mode_type}")
        debug_lines.append(f"  steps:          {steps}")
        debug_lines.append(f"  cfg:            {cfg}")
        debug_lines.append(f"  denoise:        {denoise}")
        debug_lines.append(f"  sampler:        {sampler_name}")
        debug_lines.append(f"  scheduler:      {scheduler}")
        debug_lines.append(f"  seed:           {seed}")
        debug_lines.append("")
        debug_lines.append("CONDITIONING:")
        debug_lines.append(f"  count:          {num_conditionings}")
        debug_lines.append(f"  expected:       {total_tiles}")
        if num_conditionings != total_tiles:
            debug_lines.append(f"  WARNING: count ({num_conditionings}) != tiles ({total_tiles})")
        debug_lines.append("=" * 60)

        debug_info = "\n".join(debug_lines)

        # Also print to console
        print("\n" + debug_info + "\n")

        # ===== APPLY DIFFERENTIAL DIFFUSION IF ENABLED =====
        diff_diff_enabled = enable_diffdiff and denoise_mask is not None
        controlnet_enabled = enable_controlnet and model_patch is not None and control_image is not None
        denoise_mask_upscaled = None
        if diff_diff_enabled:
            if denoise_mask.dim() == 3:
                mask_2d = denoise_mask[0]
            else:
                mask_2d = denoise_mask

            mask_upscaled = F.interpolate(
                mask_2d.unsqueeze(0).unsqueeze(0),
                size=(output_h, output_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)

            print(f"[DiffDiff] Mask upscaled: {mask_2d.shape} -> {mask_upscaled.shape}")

            diff_diff = DifferentialDiffusionAdvanced()
            diff_diff.multiplier = multiplier
            model = model.clone()
            model.set_model_denoise_mask_function(diff_diff.forward)

            print(f"[DiffDiff] Model patched with multiplier={multiplier}")
            denoise_mask_upscaled = mask_upscaled

        # ===== UPSCALE CONTROL IMAGE TO OUTPUT SIZE =====
        control_image_upscaled = None
        control_mask_upscaled = None
        if controlnet_enabled:
            control_image_upscaled = F.interpolate(
                control_image.movedim(-1, 1),
                size=(output_h, output_w),
                mode='bilinear',
                align_corners=False
            ).movedim(1, -1)
            print(f"[ControlNet] Control image upscaled: {control_image.shape} -> {control_image_upscaled.shape}")

            if control_mask is not None:
                if control_mask.dim() == 3:
                    ctrl_mask_2d = control_mask[0]
                else:
                    ctrl_mask_2d = control_mask

                control_mask_upscaled = F.interpolate(
                    ctrl_mask_2d.unsqueeze(0).unsqueeze(0),
                    size=(output_h, output_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)
                print(f"[ControlNet] Control mask upscaled: {ctrl_mask_2d.shape} -> {control_mask_upscaled.shape}")

        # Set up A1111 patches
        shared.sd_upscalers[0] = UpscalerData()
        shared.actual_upscaler = None  # No upscaler needed - image is pre-upscaled

        # Set the batch of images
        shared.batch = [tensor_to_pil(upscaled_image, i) for i in range(len(upscaled_image))]
        shared.batch_as_tensor = upscaled_image

        # Processing - upscale_by = 1.0 since image is already upscaled
        # force_uniform_tiles = True (hardcoded - Matrix Search already calculated uniform tiles)
        sdprocessing = StableDiffusionProcessing(
            shared.batch[0], model, conditionings, negative, vae,
            seed, steps, cfg, sampler_name, scheduler, denoise, 1.0, True, tiled_decode,
            tile_width, tile_height, MODES[self.mode_type], SEAM_FIX_MODES[self.seam_fix_mode],
            None, None,  # custom_sampler, custom_sigmas
            denoise_mask_tensor=denoise_mask_upscaled,
            model_patch=model_patch if controlnet_enabled else None,
            control_image_tensor=control_image_upscaled if controlnet_enabled else None,
            control_strength=control_strength if controlnet_enabled else 1.0,
            control_mask_tensor=control_mask_upscaled if controlnet_enabled else None,
        )

        # Disable logging
        logger = logging.getLogger()
        old_level = logger.getEffectiveLevel()
        logger.setLevel(logging.CRITICAL + 1)
        try:
            script = usdu.Script()
            processed = script.run(p=sdprocessing, _=None, tile_width=self.tile_width, tile_height=self.tile_height,
                               mask_blur=self.mask_blur, padding=self.tile_padding, seams_fix_width=self.seam_fix_width,
                               seams_fix_denoise=self.seam_fix_denoise, seams_fix_padding=self.seam_fix_padding,
                               upscaler_index=0, save_upscaled_image=False, redraw_mode=MODES[self.mode_type],
                               save_seams_fix_image=False, seams_fix_mask_blur=self.seam_fix_mask_blur,
                               seams_fix_type=SEAM_FIX_MODES[self.seam_fix_mode], target_size_type=2,
                               custom_width=None, custom_height=None, custom_scale=1.0)

            images = [pil_to_tensor(img) for img in shared.batch]
            tensor = torch.cat(images, dim=0)
            return (tensor, debug_info)
        finally:
            logger.setLevel(old_level)


class ArchAi3D_Smart_USDU_Universal_CustomSample(ArchAi3D_Smart_USDU_Universal):
    """Universal variant with custom sampler and sigmas support."""

    @classmethod
    def INPUT_TYPES(s):
        required, optional = USDU_universal_inputs()
        remove_input(required, "upscale_model")
        optional.append(("upscale_model", ("UPSCALE_MODEL",)))
        optional.append(("custom_sampler", ("SAMPLER",)))
        optional.append(("custom_sigmas", ("SIGMAS",)))
        return prepare_inputs(required, optional)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "ArchAi3d/Upscaling"

    def upscale(self, image, enable_diffdiff, enable_controlnet, model, conditionings, negative, vae, upscale_by, seed,
                steps, cfg, sampler_name, scheduler, denoise,
                mode_type, tile_width, tile_height, mask_blur, tile_padding,
                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode,
                upscale_model=None,
                custom_sampler=None, custom_sigmas=None,
                denoise_mask=None, multiplier=1.0,
                model_patch=None, control_image=None, control_strength=1.0, control_mask=None):
        return super().upscale(image, enable_diffdiff, enable_controlnet, model, conditionings, negative, vae, upscale_by, seed,
                steps, cfg, sampler_name, scheduler, denoise, upscale_model,
                mode_type, tile_width, tile_height, mask_blur, tile_padding,
                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode,
                custom_sampler, custom_sigmas,
                denoise_mask=denoise_mask, multiplier=multiplier,
                model_patch=model_patch, control_image=control_image,
                control_strength=control_strength, control_mask=control_mask)


class ArchAi3D_Smart_USDU_Universal_NoUpscale_V3:
    """
    Smart USDU Universal (No Upscale) V3 - Tile Feather Mask + Debug Outputs.

    NEW in V3:
    - Tile Feather Mask: Creates gradient at tile edges for color consistency
    - Debug Outputs: Visual inspection of tiles and feather masks
      - debug_tiles: Grid of all cropped tile images
      - debug_feather_masks: Grid of all feather masks (grayscale)
      - debug_combined: Tiles with masks overlaid (red tint on feather zones)

    Feather mask is applied DURING sampling via latent["noise_mask"].
    This reduces denoise at tile edges, allowing better color blending.
    """

    @classmethod
    def INPUT_TYPES(s):
        required, optional = USDU_noupscale_v3_inputs()
        return prepare_inputs(required, optional)

    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "debug_info", "debug_tiles", "debug_feather_masks", "debug_combined")
    FUNCTION = "upscale"
    CATEGORY = "ArchAi3d/Upscaling"

    def upscale(self, upscaled_image, output_width, output_height, tiles_x, tiles_y, safe_guard,
                enable_diffdiff, enable_controlnet,
                feather_enabled, feather_width, feather_blur,
                model, conditionings, negative, vae, seed,
                steps, cfg, sampler_name, scheduler, denoise,
                mode_type, tile_width, tile_height, mask_blur, tile_padding,
                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding, tiled_decode,
                denoise_mask=None, multiplier=1.0,
                model_patch=None, control_image=None, control_strength=1.0, control_mask=None):
        """
        Process upscaled image with Tile Feather Mask for color consistency.
        V3: Adds debug outputs for visual inspection.
        """
        # Store params
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.mask_blur = mask_blur
        self.tile_padding = tile_padding
        self.seam_fix_width = seam_fix_width
        self.seam_fix_denoise = seam_fix_denoise
        self.seam_fix_padding = seam_fix_padding
        self.seam_fix_mode = seam_fix_mode
        self.mode_type = mode_type
        self.seam_fix_mask_blur = seam_fix_mask_blur

        # ===== SAFEGUARD VALIDATION =====
        if safe_guard:
            validate_safeguard(
                upscaled_image, output_width, output_height,
                tile_width, tile_height, tile_padding,
                tiles_x, tiles_y
            )

        # ===== USE INPUT VALUES DIRECTLY =====
        img_h, img_w = upscaled_image.shape[1], upscaled_image.shape[2]
        output_w = output_width
        output_h = output_height
        cols = tiles_x
        rows = tiles_y
        total_tiles = tiles_x * tiles_y

        num_conditionings = len(conditionings) if isinstance(conditionings, list) else 1
        sampling_width = math.ceil((tile_width + tile_padding) / 64) * 64
        sampling_height = math.ceil((tile_height + tile_padding) / 64) * 64

        # Coverage calculation
        if tiles_x > 1:
            coverage_w = tile_width * tiles_x - tile_padding * (tiles_x - 1)
        else:
            coverage_w = tile_width
        if tiles_y > 1:
            coverage_h = tile_height * tiles_y - tile_padding * (tiles_y - 1)
        else:
            coverage_h = tile_height

        # ===== V3: CREATE DEBUG OUTPUTS =====
        print("\n" + "=" * 60)
        print("🎛️ Smart USDU Universal (No Upscale) V3 - Feather Mask Edition")
        print("=" * 60)

        # Calculate tile positions and create feather masks
        debug_tiles_list = []
        debug_masks_list = []

        # Calculate step size (for overlapping tiles)
        if tiles_x > 1:
            step_x = (output_w - tile_width) // (tiles_x - 1) if tiles_x > 1 else 0
        else:
            step_x = 0

        if tiles_y > 1:
            step_y = (output_h - tile_height) // (tiles_y - 1) if tiles_y > 1 else 0
        else:
            step_y = 0

        print(f"\nFEATHER MASK (v3.1):")
        print(f"  Enabled: {'YES' if feather_enabled else 'NO'}")
        if feather_enabled:
            print(f"  feather_width: {feather_width}px")
            print(f"  feather_blur: {feather_blur}px")
            print(f"  tile_padding: {tile_padding}px")
            if feather_width >= tile_padding:
                print(f"  ⚠️ WARNING: feather_width ({feather_width}) >= tile_padding ({tile_padding})")
                print(f"              This may cause issues. User is responsible for correct values.")

        print(f"\nDEBUG OUTPUTS:")
        print(f"  Generating tile crops and feather masks for visualization...")

        # Get image as tensor [H, W, C]
        img_tensor = upscaled_image[0]  # Remove batch dimension

        for yi in range(rows):
            for xi in range(cols):
                # Calculate tile position
                if tiles_x > 1:
                    x1 = xi * step_x
                else:
                    x1 = 0
                if tiles_y > 1:
                    y1 = yi * step_y
                else:
                    y1 = 0

                x2 = min(x1 + tile_width, output_w)
                y2 = min(y1 + tile_height, output_h)

                # Crop tile from image
                tile_crop = img_tensor[y1:y2, x1:x2, :].clone()
                debug_tiles_list.append(tile_crop)

                # Create feather mask for this tile
                is_left_edge = (xi == 0)
                is_right_edge = (xi == cols - 1)
                is_top_edge = (yi == 0)
                is_bottom_edge = (yi == rows - 1)

                if feather_enabled:
                    feather_mask = create_tile_feather_mask(
                        tile_h=tile_crop.shape[0],
                        tile_w=tile_crop.shape[1],
                        feather_width=feather_width,
                        feather_blur=feather_blur,
                        is_left_edge=is_left_edge,
                        is_right_edge=is_right_edge,
                        is_top_edge=is_top_edge,
                        is_bottom_edge=is_bottom_edge
                    )
                else:
                    # All white mask (no feathering)
                    feather_mask = torch.ones((tile_crop.shape[0], tile_crop.shape[1]))

                debug_masks_list.append(feather_mask)

        # Create debug grid images
        debug_tiles_grid = create_debug_grid(debug_tiles_list, rows, cols, border_width=2, border_color=(1.0, 0.0, 0.0))
        debug_masks_grid = create_debug_grid(debug_masks_list, rows, cols, border_width=2, border_color=(0.0, 0.0, 1.0))
        debug_combined_grid = create_combined_debug(debug_tiles_list, debug_masks_list, rows, cols, border_width=2)

        print(f"  debug_tiles: {debug_tiles_grid.shape}")
        print(f"  debug_feather_masks: {debug_masks_grid.shape}")
        print(f"  debug_combined: {debug_combined_grid.shape}")

        # ===== V3 DEBUG INFO STRING =====
        debug_lines = []
        debug_lines.append("=" * 60)
        debug_lines.append("Smart USDU Universal (No Upscale) V3 - Feather Mask Edition")
        debug_lines.append("=" * 60)
        debug_lines.append(f"SAFEGUARD: {'ENABLED' if safe_guard else 'DISABLED'}")
        debug_lines.append("")
        debug_lines.append("FEATHER MASK (v3.1):")
        debug_lines.append(f"  enabled:        {feather_enabled}")
        debug_lines.append(f"  feather_width:  {feather_width}px")
        debug_lines.append(f"  feather_blur:   {feather_blur}px")
        debug_lines.append(f"  tile_padding:   {tile_padding}px")
        debug_lines.append("")
        debug_lines.append("PIXEL DIMENSIONS:")
        debug_lines.append(f"  tile_width:     {tile_width}px")
        debug_lines.append(f"  tile_height:    {tile_height}px")
        debug_lines.append(f"  mask_blur:      {mask_blur}px")
        debug_lines.append(f"  output_size:    {output_width} x {output_height}px")
        debug_lines.append("")
        debug_lines.append("LATENT DIMENSIONS (pixel / 8):")
        debug_lines.append(f"  latent_tile_w:  {tile_width // 8}")
        debug_lines.append(f"  latent_tile_h:  {tile_height // 8}")
        debug_lines.append(f"  latent_padding: {tile_padding // 8}")
        debug_lines.append(f"  latent_feather: {feather_width // 8}")
        debug_lines.append("")
        debug_lines.append("GRID LAYOUT:")
        debug_lines.append(f"  tiles_x:        {tiles_x}")
        debug_lines.append(f"  tiles_y:        {tiles_y}")
        debug_lines.append(f"  total_tiles:    {total_tiles}")
        debug_lines.append(f"  step_x:         {step_x}px")
        debug_lines.append(f"  step_y:         {step_y}px")
        debug_lines.append("")
        debug_lines.append("COVERAGE CHECK:")
        debug_lines.append(f"  coverage:       {coverage_w} x {coverage_h}px")
        debug_lines.append(f"  output_size:    {output_width} x {output_height}px")
        coverage_ok = coverage_w >= output_width and coverage_h >= output_height
        debug_lines.append(f"  status:         {'OK' if coverage_ok else 'INSUFFICIENT'}")
        debug_lines.append("")
        debug_lines.append("SAMPLING:")
        debug_lines.append(f"  mode:           {mode_type}")
        debug_lines.append(f"  steps:          {steps}")
        debug_lines.append(f"  cfg:            {cfg}")
        debug_lines.append(f"  denoise:        {denoise}")
        debug_lines.append(f"  seed:           {seed}")
        debug_lines.append("")
        debug_lines.append("CONDITIONING:")
        debug_lines.append(f"  count:          {num_conditionings}")
        debug_lines.append(f"  expected:       {total_tiles}")
        if num_conditionings != total_tiles:
            debug_lines.append(f"  ⚠️ WARNING: count != tiles")
        debug_lines.append("=" * 60)

        debug_info = "\n".join(debug_lines)
        print(debug_info)

        # ===== APPLY DIFFERENTIAL DIFFUSION IF ENABLED =====
        diff_diff_enabled = enable_diffdiff and denoise_mask is not None
        controlnet_enabled = enable_controlnet and model_patch is not None and control_image is not None
        denoise_mask_upscaled = None

        # ===== CREATE FULL-IMAGE FEATHER MASK (for DiffDiff integration) =====
        # This creates a full-image mask from all tile feather masks
        feather_mask_full = None
        if feather_enabled:
            # Start with ones
            feather_mask_full = torch.ones((output_h, output_w), dtype=torch.float32)

            # Apply each tile's feather mask to its region
            for yi in range(rows):
                for xi in range(cols):
                    idx = yi * cols + xi
                    if idx < len(debug_masks_list):
                        # Calculate tile position
                        if tiles_x > 1:
                            x1 = xi * step_x
                        else:
                            x1 = 0
                        if tiles_y > 1:
                            y1 = yi * step_y
                        else:
                            y1 = 0

                        tile_mask = debug_masks_list[idx]
                        h, w = tile_mask.shape

                        # Blend feather mask into full mask (use minimum for overlapping regions)
                        y2 = min(y1 + h, output_h)
                        x2 = min(x1 + w, output_w)
                        actual_h = y2 - y1
                        actual_w = x2 - x1

                        feather_mask_full[y1:y2, x1:x2] = torch.minimum(
                            feather_mask_full[y1:y2, x1:x2],
                            tile_mask[:actual_h, :actual_w]
                        )

            print(f"[Feather] Full-image feather mask created: {feather_mask_full.shape}")

        if diff_diff_enabled or feather_enabled:
            if denoise_mask is not None:
                if denoise_mask.dim() == 3:
                    mask_2d = denoise_mask[0]
                else:
                    mask_2d = denoise_mask

                mask_upscaled = F.interpolate(
                    mask_2d.unsqueeze(0).unsqueeze(0),
                    size=(output_h, output_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)
            else:
                # No user mask - create all-ones
                mask_upscaled = torch.ones((output_h, output_w), dtype=torch.float32)

            # Merge with feather mask if enabled
            if feather_enabled and feather_mask_full is not None:
                mask_upscaled = mask_upscaled * feather_mask_full
                print(f"[Feather] Feather mask merged with denoise mask")

            print(f"[DiffDiff] Final mask shape: {mask_upscaled.shape}")

            if enable_diffdiff:
                diff_diff = DifferentialDiffusionAdvanced()
                diff_diff.multiplier = multiplier
                model = model.clone()
                model.set_model_denoise_mask_function(diff_diff.forward)
                print(f"[DiffDiff] Model patched with multiplier={multiplier}")

            denoise_mask_upscaled = mask_upscaled

        # ===== UPSCALE CONTROL IMAGE =====
        control_image_upscaled = None
        control_mask_upscaled = None
        if controlnet_enabled:
            control_image_upscaled = F.interpolate(
                control_image.movedim(-1, 1),
                size=(output_h, output_w),
                mode='bilinear',
                align_corners=False
            ).movedim(1, -1)
            print(f"[ControlNet] Control image upscaled: {control_image.shape} -> {control_image_upscaled.shape}")

            if control_mask is not None:
                if control_mask.dim() == 3:
                    ctrl_mask_2d = control_mask[0]
                else:
                    ctrl_mask_2d = control_mask

                control_mask_upscaled = F.interpolate(
                    ctrl_mask_2d.unsqueeze(0).unsqueeze(0),
                    size=(output_h, output_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)

        # Set up A1111 patches
        shared.sd_upscalers[0] = UpscalerData()
        shared.actual_upscaler = None

        shared.batch = [tensor_to_pil(upscaled_image, i) for i in range(len(upscaled_image))]
        shared.batch_as_tensor = upscaled_image

        # Processing
        sdprocessing = StableDiffusionProcessing(
            shared.batch[0], model, conditionings, negative, vae,
            seed, steps, cfg, sampler_name, scheduler, denoise, 1.0, True, tiled_decode,
            tile_width, tile_height, MODES[self.mode_type], SEAM_FIX_MODES[self.seam_fix_mode],
            None, None,
            denoise_mask_tensor=denoise_mask_upscaled,
            model_patch=model_patch if controlnet_enabled else None,
            control_image_tensor=control_image_upscaled if controlnet_enabled else None,
            control_strength=control_strength if controlnet_enabled else 1.0,
            control_mask_tensor=control_mask_upscaled if controlnet_enabled else None,
        )

        # Disable logging
        logger = logging.getLogger()
        old_level = logger.getEffectiveLevel()
        logger.setLevel(logging.CRITICAL + 1)
        try:
            script = usdu.Script()
            processed = script.run(p=sdprocessing, _=None, tile_width=self.tile_width, tile_height=self.tile_height,
                               mask_blur=self.mask_blur, padding=self.tile_padding, seams_fix_width=self.seam_fix_width,
                               seams_fix_denoise=self.seam_fix_denoise, seams_fix_padding=self.seam_fix_padding,
                               upscaler_index=0, save_upscaled_image=False, redraw_mode=MODES[self.mode_type],
                               save_seams_fix_image=False, seams_fix_mask_blur=self.seam_fix_mask_blur,
                               seams_fix_type=SEAM_FIX_MODES[self.seam_fix_mode], target_size_type=2,
                               custom_width=None, custom_height=None, custom_scale=1.0)

            images = [pil_to_tensor(img) for img in shared.batch]
            tensor = torch.cat(images, dim=0)

            print(f"\n[Smart USDU V3] Done! Feather mask {'applied' if feather_enabled else 'disabled'}")

            return (tensor, debug_info, debug_tiles_grid, debug_masks_grid, debug_combined_grid)
        finally:
            logger.setLevel(old_level)


# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "ArchAi3D_Smart_USDU_Universal": ArchAi3D_Smart_USDU_Universal,
    "ArchAi3D_Smart_USDU_Universal_NoUpscale": ArchAi3D_Smart_USDU_Universal_NoUpscale,
    "ArchAi3D_Smart_USDU_Universal_NoUpscale_V2": ArchAi3D_Smart_USDU_Universal_NoUpscale_V2,
    "ArchAi3D_Smart_USDU_Universal_NoUpscale_V3": ArchAi3D_Smart_USDU_Universal_NoUpscale_V3,
    "ArchAi3D_Smart_USDU_Universal_CustomSample": ArchAi3D_Smart_USDU_Universal_CustomSample
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_Smart_USDU_Universal": "🎛️ Smart USDU Universal",
    "ArchAi3D_Smart_USDU_Universal_NoUpscale": "🎛️ Smart USDU Universal (No Upscale)",
    "ArchAi3D_Smart_USDU_Universal_NoUpscale_V2": "🎛️ Smart USDU Universal (No Upscale) V2",
    "ArchAi3D_Smart_USDU_Universal_NoUpscale_V3": "🎛️ Smart USDU Universal (No Upscale) V3",
    "ArchAi3D_Smart_USDU_Universal_CustomSample": "🎛️ Smart USDU Universal (Custom Sample)"
}
