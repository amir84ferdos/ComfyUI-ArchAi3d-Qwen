"""
Smart USDU Universal V3 - Tile Feather Mask (STANDALONE)
========================================================

This is a STANDALONE file - no imports from processing.py, usdu_patch.py, or ultimate_upscale.py.
All logic is contained in this single file for easy debugging and discussion.

================================================================================
PROBLEM: COLOR INCONSISTENCY AT TILE SEAMS
================================================================================

When using tiled upscaling (Ultimate SD Upscale), each tile is processed
independently by the sampler. Adjacent tiles often have slight color/tone
differences, creating visible seams when stitched together.

    ┌─────────┬─────────┐
    │  Tile A │  Tile B │
    │  (warm) │  (cool) │  ← Color mismatch at seam!
    └─────────┴─────────┘

Current approaches (NOT working well):
1. SEGS Mask Blur - Applied AFTER sampling (doesn't affect actual sampling)
2. Simple overlap blending - Just blends pixels, doesn't fix root cause

================================================================================
PROPOSED SOLUTION: FEATHER MASK DURING SAMPLING
================================================================================

Apply a gradient mask to `latent["noise_mask"]` DURING sampling, not after.

    ┌─────────────────────────────────┐
    │  tile_padding (overlap area)    │  ← Preserved from neighbor tile
    │  ┌───────────────────────────┐  │
    │  │  feather_mask (gradient)  │  │  ← Gradient zone
    │  │  ┌─────────────────────┐  │  │
    │  │  │                     │  │  │
    │  │  │  FULL DENOISE (1.0) │  │  │  ← Center: full sampling
    │  │  │                     │  │  │
    │  │  └─────────────────────┘  │  │
    │  │      gradient 1.0→0.0     │  │  ← Edges: reduced sampling
    │  └───────────────────────────┘  │
    │      (0.0 = no change)          │
    └─────────────────────────────────┘

Key concepts:
- feather_width MUST be < tile_padding (so there's unmodified overlap to blend)
- At interior tile edges: gradient from 1.0 (inside) to 0.0 (at edge)
- At image boundaries: keep 1.0 (no feathering needed at image edges)
- This mask goes to latent["noise_mask"] so sampler respects it

================================================================================
QUESTIONS FOR DISCUSSION (Please help answer these!)
================================================================================

Q1: Is `latent["noise_mask"]` the right mechanism?
    - The noise_mask controls "where to apply denoising"
    - Value 1.0 = full denoise, 0.0 = no denoise (preserve original)
    - Is this correct for tile blending? Or is there a better approach?

Q2: How to merge feather_mask with user's denoise_mask?
    - If user provides a denoise_mask (for DiffDiff), we currently multiply:
      final_mask = feather_mask * user_denoise_mask
    - Is multiplication correct? Should it be min()? max()? something else?

Q3: What's the right relationship between feather_width and tile_padding?
    - Example: tile_padding=64px, feather_width=16px
    - Gradient covers 16px, remaining 48px is at 0.0 (preserved)
    - Is this correct? Should feather_width = tile_padding?

    Tile A:                          Tile B:
    [...content...][gradient 16px]   [gradient 16px][...content...]
                   ↑ 0.0 here        ↑ 0.0 here

    Both tiles have 0.0 at their edges = neither changes the overlap much

Q4: How does the sampler actually use noise_mask?
    - Does it affect noise injection? Denoising strength? Both?
    - What happens at noise_mask=0.5? Is it 50% change or something else?

Q5: Is there a better blending strategy?
    - Current: Both adjacent tiles have low denoise at their shared edge
    - Alternative: One tile does full denoise, neighbor preserves?
    - What about inpainting-style approach?

================================================================================
Author: Amir Ferdos (ArchAi3d)
Email: Amir84ferdos@gmail.com
Version: 3.1.0 - Standalone Feather Mask Implementation
License: Dual License (Free for personal use, Commercial license required)
================================================================================
"""

import math
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

# ComfyUI imports
from nodes import common_ksampler, VAEEncode, VAEDecode
import comfy.samplers
import comfy.sample

MAX_RESOLUTION = 8192


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def tensor_to_pil(tensor: torch.Tensor, index: int = 0) -> Image.Image:
    """Convert tensor [B, H, W, C] to PIL Image."""
    if tensor.dim() == 4:
        img = tensor[index]
    else:
        img = tensor
    img = img.cpu().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to tensor [1, H, W, C]."""
    img = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(img).unsqueeze(0)


# ============================================================================
# FEATHER MASK FUNCTION
# ============================================================================

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
    Create feather mask for one tile.

    The mask controls where denoising is applied:
    - 1.0 = full denoise (center of tile)
    - 0.0 = no denoise (preserve original - at tile edges with neighbors)

    At IMAGE BOUNDARIES (edges of the full image), we keep 1.0 because
    there's no neighbor tile to blend with.

    At INTERIOR EDGES (edges shared with neighbor tiles), we create a
    gradient from 1.0 (inside) to 0.0 (at edge).

    Args:
        tile_h: Tile height in pixels
        tile_w: Tile width in pixels
        feather_width: Width of gradient zone in pixels
        feather_blur: Gaussian blur radius to smooth the gradient
        is_left_edge: True if this tile is at the LEFT edge of the full image
        is_right_edge: True if this tile is at the RIGHT edge of the full image
        is_top_edge: True if this tile is at the TOP edge of the full image
        is_bottom_edge: True if this tile is at the BOTTOM edge of the full image

    Returns:
        Tensor shape [H, W] with values 0.0-1.0

    Example for a 3x2 grid:

        Tile (0,0)          Tile (1,0)          Tile (2,0)
        is_left=True        is_left=False       is_left=False
        is_right=False      is_right=False      is_right=True
        is_top=True         is_top=True         is_top=True
        is_bottom=False     is_bottom=False     is_bottom=False

        [1111111▼]          [▼111111▼]          [▼1111111]
        [1111111▼]          [▼111111▼]          [▼1111111]
        [1111111▼]          [▼111111▼]          [▼1111111]
        [▼▼▼▼▼▼▼▼]          [▼▼▼▼▼▼▼▼]          [▼▼▼▼▼▼▼▼]

        1 = 1.0 (full denoise)
        ▼ = gradient 1.0→0.0 (feather zone)
    """
    # Start with all 1.0 (full denoise everywhere)
    mask = torch.ones((tile_h, tile_w), dtype=torch.float32)

    if feather_width <= 0:
        return mask

    # Create gradient for INTERIOR edges only (not at image boundaries)

    # LEFT edge gradient (only if NOT at image left boundary)
    if not is_left_edge:
        for x in range(min(feather_width, tile_w)):
            # Gradient from 0.0 at edge to 1.0 at feather_width
            value = x / feather_width
            mask[:, x] = torch.minimum(mask[:, x], torch.tensor(value))

    # RIGHT edge gradient (only if NOT at image right boundary)
    if not is_right_edge:
        for x in range(min(feather_width, tile_w)):
            # Gradient from 0.0 at edge to 1.0 at feather_width
            value = x / feather_width
            mask[:, tile_w - 1 - x] = torch.minimum(mask[:, tile_w - 1 - x], torch.tensor(value))

    # TOP edge gradient (only if NOT at image top boundary)
    if not is_top_edge:
        for y in range(min(feather_width, tile_h)):
            # Gradient from 0.0 at edge to 1.0 at feather_width
            value = y / feather_width
            mask[y, :] = torch.minimum(mask[y, :], torch.tensor(value))

    # BOTTOM edge gradient (only if NOT at image bottom boundary)
    if not is_bottom_edge:
        for y in range(min(feather_width, tile_h)):
            # Gradient from 0.0 at edge to 1.0 at feather_width
            value = y / feather_width
            mask[tile_h - 1 - y, :] = torch.minimum(mask[tile_h - 1 - y, :], torch.tensor(value))

    # Apply Gaussian blur to smooth the gradient
    if feather_blur > 0:
        # Convert to PIL for blur (simpler than torch gaussian)
        mask_np = (mask.numpy() * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_np, mode='L')
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=feather_blur))
        mask_np = np.array(mask_pil).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask_np)

    # Ensure values are in [0, 1]
    mask = torch.clamp(mask, 0.0, 1.0)

    return mask


# ============================================================================
# DEBUG VISUALIZATION FUNCTIONS
# ============================================================================

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
        border_width: Width of border between tiles (pixels)
        border_color: RGB color for border (0.0-1.0)

    Returns:
        Single tensor [1, H*rows + border*(rows-1), W*cols + border*(cols-1), C]
    """
    if len(tiles) == 0:
        return torch.zeros((1, 64, 64, 3))

    # Get tile dimensions from first tile
    tile_h, tile_w = tiles[0].shape[0], tiles[0].shape[1]

    # Handle grayscale vs RGB
    if len(tiles[0].shape) == 2:
        # Convert grayscale to RGB for visualization
        tiles = [t.unsqueeze(-1).repeat(1, 1, 3) for t in tiles]

    channels = tiles[0].shape[2]

    # Calculate grid dimensions
    grid_h = tile_h * rows + border_width * (rows - 1)
    grid_w = tile_w * cols + border_width * (cols - 1)

    # Create grid filled with border color
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

    # Add batch dimension [1, H, W, C]
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

    Red tint shows where feathering reduces denoise strength.

    Args:
        tiles: List of tile images [H, W, C]
        masks: List of feather masks [H, W]
        rows, cols: Grid dimensions
        border_width: Border between tiles

    Returns:
        Grid tensor with masks overlaid as red tint
    """
    combined_tiles = []

    for tile, mask in zip(tiles, masks):
        # Ensure tile is RGB
        if len(tile.shape) == 2:
            tile = tile.unsqueeze(-1).repeat(1, 1, 3)

        # Create overlay: red where mask < 1.0
        overlay = tile.clone()

        # Red channel: increase where mask is low (feathered areas)
        # This makes feathered areas appear reddish
        feather_intensity = 1.0 - mask  # 0 where full denoise, 1 where no denoise
        overlay[:, :, 0] = torch.clamp(overlay[:, :, 0] + feather_intensity * 0.5, 0, 1)

        # Dim green/blue slightly in feathered areas
        overlay[:, :, 1] = overlay[:, :, 1] * (0.7 + 0.3 * mask)
        overlay[:, :, 2] = overlay[:, :, 2] * (0.7 + 0.3 * mask)

        combined_tiles.append(overlay)

    return create_debug_grid(combined_tiles, rows, cols, border_width, (0.0, 1.0, 0.0))


# ============================================================================
# SIMPLE TILE PROCESSOR (No external dependencies)
# ============================================================================

class SimpleTileProcessor:
    """
    Minimal tile processing with feather mask support.

    This class implements tile-by-tile processing WITHOUT using:
    - processing.py
    - usdu_patch.py
    - ultimate_upscale.py

    The tile loop is implemented directly here for clarity and debugging.

    TILE PROCESSING FLOW:
    1. Calculate tile positions (with overlap/padding)
    2. For each tile:
       a. Crop tile region from input image
       b. Create feather mask (if enabled)
       c. Merge feather mask with user denoise_mask (if provided)
       d. Encode tile to latent space (VAE encode)
       e. Set latent["noise_mask"] for sampling
       f. Run sampler (KSampler)
       g. Decode latent back to pixels (VAE decode)
       h. Paste tile back to output (with blending in overlap)
    3. Return final composed image
    """

    def __init__(self):
        self.vae_encoder = VAEEncode()
        self.vae_decoder = VAEDecode()

    def calculate_tile_positions(
        self,
        image_w: int,
        image_h: int,
        tiles_x: int,
        tiles_y: int,
        tile_width: int,
        tile_height: int
    ) -> List[Tuple[int, int, int, int]]:
        """
        Calculate tile crop positions with overlap.

        Returns list of (x1, y1, x2, y2) for each tile.
        Tiles overlap to ensure full coverage.
        """
        positions = []

        # Calculate step size (distance between tile starts)
        if tiles_x > 1:
            step_x = (image_w - tile_width) // (tiles_x - 1)
        else:
            step_x = 0

        if tiles_y > 1:
            step_y = (image_h - tile_height) // (tiles_y - 1)
        else:
            step_y = 0

        for yi in range(tiles_y):
            for xi in range(tiles_x):
                # Calculate tile position
                if tiles_x > 1:
                    x1 = xi * step_x
                else:
                    x1 = (image_w - tile_width) // 2  # Center if single tile

                if tiles_y > 1:
                    y1 = yi * step_y
                else:
                    y1 = (image_h - tile_height) // 2  # Center if single tile

                # Ensure tile stays within image bounds
                x1 = max(0, min(x1, image_w - tile_width))
                y1 = max(0, min(y1, image_h - tile_height))

                x2 = x1 + tile_width
                y2 = y1 + tile_height

                positions.append((x1, y1, x2, y2))

        return positions

    def process_tiles(
        self,
        image: torch.Tensor,
        tiles_x: int,
        tiles_y: int,
        tile_width: int,
        tile_height: int,
        feather_enabled: bool,
        feather_width: int,
        feather_blur: int,
        model,
        positive,
        negative,
        vae,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,
        denoise_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], str]:
        """
        Process image tile by tile with optional feather mask.

        Args:
            image: Input image tensor [B, H, W, C]
            tiles_x, tiles_y: Number of tiles in each dimension
            tile_width, tile_height: Size of each tile in pixels
            feather_enabled: Whether to apply feather mask
            feather_width: Width of feather gradient in pixels
            feather_blur: Blur radius for feather mask
            model: ComfyUI model
            positive, negative: Conditioning
            vae: VAE for encode/decode
            seed: Random seed
            steps: Sampling steps
            cfg: CFG scale
            sampler_name: Name of sampler
            scheduler: Name of scheduler
            denoise: Denoise strength (0.0-1.0)
            denoise_mask: Optional user-provided denoise mask [H, W]

        Returns:
            Tuple of:
            - output_image: Processed image tensor [B, H, W, C]
            - debug_tiles: List of tile crops (before processing)
            - debug_masks: List of feather masks
            - debug_info: String with debug information
        """
        # Get image dimensions
        batch_size = image.shape[0]
        image_h = image.shape[1]
        image_w = image.shape[2]

        # Calculate tile positions
        positions = self.calculate_tile_positions(
            image_w, image_h, tiles_x, tiles_y, tile_width, tile_height
        )

        # Initialize output image (copy of input)
        output = image.clone()

        # Track tiles processed for blending weights
        blend_weights = torch.zeros((image_h, image_w), dtype=torch.float32)
        accumulated = torch.zeros((batch_size, image_h, image_w, image.shape[3]), dtype=torch.float32)

        # Debug collections
        debug_input_tiles = []   # Input crops BEFORE sampling
        debug_output_tiles = []  # Processed tiles AFTER sampling, BEFORE blending
        debug_masks = []
        debug_lines = []

        debug_lines.append("=" * 60)
        debug_lines.append("SimpleTileProcessor - Tile Processing Log")
        debug_lines.append("=" * 60)
        debug_lines.append(f"Image size: {image_w} x {image_h}")
        debug_lines.append(f"Tiles: {tiles_x} x {tiles_y} = {tiles_x * tiles_y}")
        debug_lines.append(f"Tile size: {tile_width} x {tile_height}")
        debug_lines.append(f"Feather: {'ENABLED' if feather_enabled else 'DISABLED'}")
        if feather_enabled:
            debug_lines.append(f"  feather_width: {feather_width}px")
            debug_lines.append(f"  feather_blur: {feather_blur}px")
        debug_lines.append(f"Denoise mask: {'PROVIDED' if denoise_mask is not None else 'NONE'}")
        debug_lines.append("")

        # Process each tile
        for tile_idx, (x1, y1, x2, y2) in enumerate(positions):
            yi = tile_idx // tiles_x
            xi = tile_idx % tiles_x

            debug_lines.append(f"Tile {tile_idx + 1}/{len(positions)} at ({xi}, {yi})")
            debug_lines.append(f"  Crop region: ({x1}, {y1}) to ({x2}, {y2})")

            # Determine edge status
            is_left_edge = (x1 == 0)
            is_right_edge = (x2 >= image_w)
            is_top_edge = (y1 == 0)
            is_bottom_edge = (y2 >= image_h)

            debug_lines.append(f"  Edges: L={is_left_edge}, R={is_right_edge}, T={is_top_edge}, B={is_bottom_edge}")

            # Crop tile from image
            tile_crop = image[:, y1:y2, x1:x2, :].clone()
            tile_h = tile_crop.shape[1]
            tile_w = tile_crop.shape[2]

            # Store INPUT tile for debug output (BEFORE sampling)
            debug_input_tiles.append(tile_crop[0])  # First batch item

            # Create feather mask for this tile
            if feather_enabled:
                feather_mask = create_tile_feather_mask(
                    tile_h=tile_h,
                    tile_w=tile_w,
                    feather_width=feather_width,
                    feather_blur=feather_blur,
                    is_left_edge=is_left_edge,
                    is_right_edge=is_right_edge,
                    is_top_edge=is_top_edge,
                    is_bottom_edge=is_bottom_edge
                )
                debug_lines.append(f"  Feather mask: min={feather_mask.min():.2f}, max={feather_mask.max():.2f}")
            else:
                # All 1.0 = full denoise everywhere
                feather_mask = torch.ones((tile_h, tile_w), dtype=torch.float32)

            debug_masks.append(feather_mask)

            # Merge with user denoise_mask if provided
            if denoise_mask is not None:
                # Crop user mask to tile region
                if denoise_mask.dim() == 3:
                    user_mask_crop = denoise_mask[0, y1:y2, x1:x2]
                else:
                    user_mask_crop = denoise_mask[y1:y2, x1:x2]

                # Multiply masks together
                # Q2: Is multiplication the right operation?
                final_mask = feather_mask * user_mask_crop
                debug_lines.append(f"  Merged with user mask: min={final_mask.min():.2f}, max={final_mask.max():.2f}")
            else:
                final_mask = feather_mask

            # Encode tile to latent
            tile_pil = tensor_to_pil(tile_crop)
            tile_tensor_for_vae = pil_to_tensor(tile_pil)
            (latent,) = self.vae_encoder.encode(vae, tile_tensor_for_vae)

            # Get latent dimensions
            latent_h = latent["samples"].shape[2]
            latent_w = latent["samples"].shape[3]

            # Resize mask to latent dimensions
            mask_latent = F.interpolate(
                final_mask.unsqueeze(0).unsqueeze(0),
                size=(latent_h, latent_w),
                mode='bilinear',
                align_corners=False
            )

            # Set noise_mask for sampling
            # THIS IS THE KEY PART - the mask tells the sampler where to denoise
            latent["noise_mask"] = mask_latent

            debug_lines.append(f"  Latent size: {latent_w} x {latent_h}")
            debug_lines.append(f"  noise_mask set: shape={mask_latent.shape}")

            # Run sampler
            # Using common_ksampler from ComfyUI
            (sampled_latent,) = common_ksampler(
                model,
                seed + tile_idx,  # Different seed per tile for variety
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive,
                negative,
                latent,
                denoise=denoise
            )

            # Decode back to pixels
            (decoded,) = self.vae_decoder.decode(vae, sampled_latent)

            # Convert to same format as input
            tile_result = decoded

            # Store OUTPUT tile for debug (AFTER sampling, BEFORE blending)
            debug_output_tiles.append(tile_result[0])  # First batch item

            # Paste tile back to output with blending
            # Use the feather mask as blend weight
            for b in range(batch_size):
                tile_data = tile_result[0]  # [H, W, C]

                # Add to accumulated with mask weighting
                for c in range(tile_data.shape[2]):
                    accumulated[b, y1:y2, x1:x2, c] += tile_data[:, :, c] * final_mask

            blend_weights[y1:y2, x1:x2] += final_mask

            debug_lines.append(f"  Tile processed and blended")
            debug_lines.append("")

        # Normalize by blend weights
        blend_weights = torch.clamp(blend_weights, min=1e-8)  # Avoid division by zero
        for b in range(batch_size):
            for c in range(accumulated.shape[3]):
                output[b, :, :, c] = accumulated[b, :, :, c] / blend_weights

        debug_lines.append("=" * 60)
        debug_lines.append("Processing complete!")
        debug_lines.append(f"Output shape: {output.shape}")
        debug_lines.append("=" * 60)

        debug_info = "\n".join(debug_lines)

        return output, debug_input_tiles, debug_output_tiles, debug_masks, debug_info


# ============================================================================
# COMFYUI NODE CLASS
# ============================================================================

class ArchAi3D_Smart_USDU_Universal_NoUpscale_V3:
    """
    Smart USDU Universal (No Upscale) V3 - Tile Feather Mask Edition

    This is a STANDALONE implementation for debugging the tile feather mask concept.

    NEW IN V3:
    - Tile Feather Mask: Gradient at tile edges for color consistency
    - Debug Outputs: Visual inspection of tiles and masks
    - Standalone: No dependencies on processing.py or ultimate_upscale.py

    INPUTS:
    - feather_enabled: Turn feather mask on/off
    - feather_width: Width of gradient zone (pixels)
    - feather_blur: Gaussian blur radius for smooth gradients

    OUTPUTS:
    - image: Processed image
    - debug_info: Text log of processing
    - debug_tiles: Grid of input tile crops
    - debug_feather_masks: Grid of feather masks (grayscale)
    - debug_combined: Tiles with mask overlay (red = feathered areas)
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "tiles_x": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
                "tiles_y": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
                "tile_width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 64}),
                "tile_height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 64}),

                # Feather mask controls
                "feather_enabled": ("BOOLEAN", {"default": True,
                    "tooltip": "Enable tile feather mask for color consistency"}),
                "feather_width": ("INT", {"default": 32, "min": 0, "max": 256, "step": 8,
                    "tooltip": "Width of gradient zone at tile edges (pixels)"}),
                "feather_blur": ("INT", {"default": 8, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Gaussian blur radius for smooth gradients"}),

                # Sampling parameters
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.5}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "denoise_mask": ("MASK", {"tooltip": "Optional per-pixel denoise mask (for DiffDiff)"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "debug_info", "input_tiles", "output_tiles", "feather_masks", "combined_view")
    FUNCTION = "process"
    CATEGORY = "ArchAi3d/Upscaling"

    def process(
        self,
        image: torch.Tensor,
        tiles_x: int,
        tiles_y: int,
        tile_width: int,
        tile_height: int,
        feather_enabled: bool,
        feather_width: int,
        feather_blur: int,
        model,
        positive,
        negative,
        vae,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,
        denoise_mask: Optional[torch.Tensor] = None
    ):
        """Main processing function."""

        print("\n" + "=" * 60)
        print("Smart USDU Universal V3 - Tile Feather Mask (STANDALONE)")
        print("=" * 60)
        print(f"Image: {image.shape}")
        print(f"Tiles: {tiles_x} x {tiles_y}")
        print(f"Tile size: {tile_width} x {tile_height}")
        print(f"Feather: {'ENABLED' if feather_enabled else 'DISABLED'}")
        if feather_enabled:
            print(f"  width={feather_width}, blur={feather_blur}")
        print(f"Denoise: {denoise}")
        print("=" * 60)

        # Create processor
        processor = SimpleTileProcessor()

        # Process tiles
        output_image, input_tiles_list, output_tiles_list, masks_list, debug_info = processor.process_tiles(
            image=image,
            tiles_x=tiles_x,
            tiles_y=tiles_y,
            tile_width=tile_width,
            tile_height=tile_height,
            feather_enabled=feather_enabled,
            feather_width=feather_width,
            feather_blur=feather_blur,
            model=model,
            positive=positive,
            negative=negative,
            vae=vae,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            denoise=denoise,
            denoise_mask=denoise_mask
        )

        # Create debug outputs as INDIVIDUAL TILES (batch tensor)
        # Each output is [N, H, W, C] where N = number of tiles
        # Connect to Preview Image node to view tiles one by one

        # input_tiles: Original crops BEFORE sampling [N, H, W, C]
        input_tiles_batch = torch.stack(input_tiles_list, dim=0)

        # output_tiles: Processed tiles AFTER sampling, BEFORE blending [N, H, W, C]
        output_tiles_batch = torch.stack(output_tiles_list, dim=0)

        # feather_masks: Grayscale masks converted to RGB [N, H, W, 3]
        masks_rgb = [m.unsqueeze(-1).repeat(1, 1, 3) for m in masks_list]
        masks_batch = torch.stack(masks_rgb, dim=0)

        # combined_view: Input tiles with feather mask overlay (red = feathered areas) [N, H, W, C]
        combined_tiles = []
        for tile, mask in zip(input_tiles_list, masks_list):
            overlay = tile.clone()
            feather_intensity = 1.0 - mask  # 1.0 where feathered, 0.0 where full denoise
            overlay[:, :, 0] = torch.clamp(overlay[:, :, 0] + feather_intensity * 0.5, 0, 1)
            overlay[:, :, 1] = overlay[:, :, 1] * (0.7 + 0.3 * mask)
            overlay[:, :, 2] = overlay[:, :, 2] * (0.7 + 0.3 * mask)
            combined_tiles.append(overlay)
        combined_batch = torch.stack(combined_tiles, dim=0)

        num_tiles = len(input_tiles_list)
        print(f"\nDebug outputs created ({num_tiles} individual tiles as batch):")
        print(f"  input_tiles:   {input_tiles_batch.shape}  (BEFORE sampling)")
        print(f"  output_tiles:  {output_tiles_batch.shape}  (AFTER sampling, BEFORE blend)")
        print(f"  feather_masks: {masks_batch.shape}")
        print(f"  combined_view: {combined_batch.shape}  (input + red feather overlay)")
        print("=" * 60 + "\n")

        return (output_image, debug_info, input_tiles_batch, output_tiles_batch, masks_batch, combined_batch)


# ============================================================================
# NODE REGISTRATION
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "ArchAi3D_Smart_USDU_Universal_NoUpscale_V3_Standalone": ArchAi3D_Smart_USDU_Universal_NoUpscale_V3
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_Smart_USDU_Universal_NoUpscale_V3_Standalone": "🎛️ Smart USDU V3 (Standalone - Feather Debug)"
}
