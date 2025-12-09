# ArchAi3D Smart Tile SEGS Blur
#
# Combined node: Creates SEGS from Smart Tile Calculator + applies mask blur
# One node does everything: SEGS creation + mask blurring + bundle pass-through
#
# Author: Amir Ferdos (ArchAi3d)
# Version: 1.6.0 - Use tile_padding for proper tile overlap (fixes dark cross at seams)
#                  v1.5.2: Simple full-strip fills (not just padding area) for all edge alignments
#                  v1.5.1: ARCHIVED - padding-based fill failed with latent_divisor=64
#                  v1.5.0: Remove caching, apply Gaussian blur AFTER organic circles
#                  v1.4.0: ARCHIVED - caching caused dark bands from margins
#                  v1.3.0: ARCHIVED - wave_scale created artifacts
#                  v1.2.1: Crop regions now divisible by 8/32 for sampler
#                  v1.2.0: Noise-modulated gradients for invisible seams
#                  v1.1.0: Perfect alignment + blur only on interior seams
#                  v1.0.0: Initial combined node
# License: Dual License (Free for personal use, Commercial license required for business use)

import numpy as np
import torch
from collections import namedtuple
from PIL import Image, ImageFilter

# Define SEG namedtuple (compatible with Impact Pack)
SEG = namedtuple("SEG",
                 ['cropped_image', 'cropped_mask', 'confidence', 'crop_region', 'bbox', 'label', 'control_net_wrapper'],
                 defaults=[None])


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def normalize_region(limit, startp, size):
    """Normalize region to fit within image bounds."""
    if startp < 0:
        new_endp = min(limit, size)
        new_startp = 0
    elif startp + size > limit:
        new_startp = max(0, limit - size)
        new_endp = limit
    else:
        new_startp = startp
        new_endp = min(limit, startp + size)
    return int(new_startp), int(new_endp)


def make_crop_region(w, h, bbox, crop_factor, divisor=8):
    """
    Create expanded crop region around bbox with dimensions divisible by divisor.

    The crop_region goes to the sampler, so dimensions MUST be divisible by:
    - 8 minimum (for VAE compatibility)
    - 32 recommended (for optimal model performance)

    Args:
        w, h: Image dimensions
        bbox: (x1, y1, x2, y2) bounding box
        crop_factor: How much to expand (1.0 = no expansion, 2.0 = 2x size)
        divisor: Dimensions will be rounded to nearest multiple (8 or 32)

    Returns:
        (cx1, cy1, cx2, cy2) - expanded crop region with aligned dimensions
    """
    x1, y1, x2, y2 = bbox

    bbox_w = x2 - x1
    bbox_h = y2 - y1

    # Calculate expanded size and round to divisor
    crop_w = int(bbox_w * crop_factor)
    crop_h = int(bbox_h * crop_factor)

    # Round UP to nearest divisor (ensures we don't cut off content)
    crop_w = ((crop_w + divisor - 1) // divisor) * divisor
    crop_h = ((crop_h + divisor - 1) // divisor) * divisor

    # Ensure minimum size (at least divisor * 8 = 64 or 256)
    min_size = divisor * 8
    crop_w = max(min_size, crop_w)
    crop_h = max(min_size, crop_h)

    # Center the crop around the bbox
    cx1 = x1 - (crop_w - bbox_w) // 2
    cy1 = y1 - (crop_h - bbox_h) // 2

    # Align start positions to divisor as well (helps with tiling)
    # Round DOWN to nearest divisor
    cx1 = (cx1 // divisor) * divisor
    cy1 = (cy1 // divisor) * divisor

    # Normalize to fit within image
    cx1, cx2 = normalize_region(w, cx1, crop_w)
    cy1, cy2 = normalize_region(h, cy1, crop_h)

    # Final check: ensure dimensions are still divisible after normalization
    actual_w = cx2 - cx1
    actual_h = cy2 - cy1
    if actual_w % divisor != 0:
        cx2 = cx1 + ((actual_w // divisor) + 1) * divisor
        cx2 = min(cx2, w)
    if actual_h % divisor != 0:
        cy2 = cy1 + ((actual_h // divisor) + 1) * divisor
        cy2 = min(cy2, h)

    return (cx1, cy1, cx2, cy2)


def calculate_tile_positions(image_w, image_h, tile_w, tile_h, tiles_x, tiles_y, tile_padding):
    """
    Calculate tile positions with proper overlap using tile_padding.

    v1.6.0: Uses tile_padding to create proper overlap between tiles.
    This ensures blur gradients at interior seams actually meet and blend.

    Step formula: step = tile_size - tile_padding
    Overlap: tile_padding pixels between adjacent tiles

    Args:
        image_w, image_h: Image dimensions
        tile_w, tile_h: Tile dimensions
        tiles_x, tiles_y: Number of tiles in each dimension
        tile_padding: Overlap between tiles in pixels

    Returns:
        List of position dicts with bbox and edge info
    """
    positions = []

    # Calculate step sizes using tile_padding for proper overlap
    # step = tile_size - tile_padding creates overlap of tile_padding pixels
    # For 1 tile, step is 0 (tile fills entire dimension)
    step_w = tile_w - tile_padding if tiles_x > 1 else 0
    step_h = tile_h - tile_padding if tiles_y > 1 else 0

    for y in range(tiles_y):
        for x in range(tiles_x):
            # Calculate position using perfect alignment
            x1 = int(round(x * step_w))
            y1 = int(round(y * step_h))
            x2 = x1 + tile_w
            y2 = y1 + tile_h

            # NO CLAMPING - with perfect alignment, tiles should fit exactly
            # But safety clamp in case of rounding
            x2 = min(x2, image_w)
            y2 = min(y2, image_h)

            positions.append({
                'bbox': (x1, y1, x2, y2),
                'row': y,
                'col': x,
                'is_left_edge': x == 0,
                'is_right_edge': x == tiles_x - 1,
                'is_top_edge': y == 0,
                'is_bottom_edge': y == tiles_y - 1,
            })

    return positions


# ============================================================================
# GAUSSIAN BLUR FALLBACK (for irregularity=0)
# ============================================================================

def create_tile_mask_gaussian(mask_h, mask_w, bbox, crop_region, blur_radius,
                               is_left_edge, is_right_edge, is_top_edge, is_bottom_edge):
    """
    Create mask using simple Gaussian blur on binary edges.
    This is the standard approach - used when irregularity=0.

    Args:
        mask_h, mask_w: Target mask dimensions
        bbox: Tile bounding box in image coordinates
        crop_region: Expanded crop region in image coordinates
        blur_radius: Blur amount for feathering
        is_*_edge: Whether tile is at image boundary (keep sharp)

    Returns:
        numpy array mask with Gaussian-blurred interior edges
    """
    x1, y1, x2, y2 = bbox
    cx1, cy1, cx2, cy2 = crop_region

    # Calculate relative positions within crop region
    rel_x1 = x1 - cx1
    rel_y1 = y1 - cy1
    rel_x2 = x2 - cx1
    rel_y2 = y2 - cy1

    # Clamp to mask bounds
    rel_x1 = max(0, min(rel_x1, mask_w))
    rel_y1 = max(0, min(rel_y1, mask_h))
    rel_x2 = max(0, min(rel_x2, mask_w))
    rel_y2 = max(0, min(rel_y2, mask_h))

    # Create binary mask (1 inside bbox, 0 outside)
    mask = np.zeros((mask_h, mask_w), dtype=np.float32)
    mask[rel_y1:rel_y2, rel_x1:rel_x2] = 1.0

    if blur_radius <= 0:
        return mask

    # Apply Gaussian blur
    mask_uint8 = (mask * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_uint8, mode='L')
    blurred_pil = mask_pil.filter(ImageFilter.GaussianBlur(blur_radius))
    mask = np.array(blurred_pil).astype(np.float32) / 255.0

    # RESTORE sharp outer edges (image boundaries should stay sharp)
    edge_restore = blur_radius * 2

    if is_left_edge and rel_x1 < edge_restore:
        mask[:, :edge_restore] = np.maximum(mask[:, :edge_restore],
                                            np.linspace(1.0, mask[:, edge_restore].mean(), edge_restore))
    if is_right_edge and rel_x2 > mask_w - edge_restore:
        mask[:, -edge_restore:] = np.maximum(mask[:, -edge_restore:],
                                              np.linspace(mask[:, -edge_restore-1].mean(), 1.0, edge_restore))
    if is_top_edge and rel_y1 < edge_restore:
        for i in range(edge_restore):
            mask[i, :] = np.maximum(mask[i, :], 1.0 - i / edge_restore * 0.5)
    if is_bottom_edge and rel_y2 > mask_h - edge_restore:
        for i in range(edge_restore):
            mask[mask_h - 1 - i, :] = np.maximum(mask[mask_h - 1 - i, :], 1.0 - i / edge_restore * 0.5)

    return mask


# ============================================================================
# ORGANIC CIRCLE ALGORITHM (v1.5.0 - direct per-tile, no caching)
# ============================================================================

def draw_irregular_edges_inplace(mask, bbox, factor, is_left_edge, is_right_edge,
                                  is_top_edge, is_bottom_edge, seed=None):
    """
    Draw overlapping circles OUTSIDE the bbox on INTERIOR edges.
    This creates organic variation that gets smoothed by subsequent Gaussian blur.

    v1.5.0: Draw directly on actual mask (no caching/scaling).

    Args:
        mask: numpy array to draw on (modified in place)
        bbox: (x1, y1, x2, y2) in mask coordinates
        factor: Controls circle size (0.1 = small circles, 1.0 = large circles)
        is_*_edge: Whether edge is at image boundary (keep sharp)
        seed: Random seed for reproducibility
    """
    rng = np.random.RandomState(seed)
    x1, y1, x2, y2 = bbox
    h, w = mask.shape

    bbox_w = x2 - x1
    bbox_h = y2 - y1

    if bbox_w <= 0 or bbox_h <= 0:
        return

    # Circle size based on factor (0.1 = small circles, 1.0 = large circles)
    # Smaller circles = more variation, larger circles = smoother edges
    circle_factor = max(6, int(min(bbox_w, bbox_h) * factor / 4))

    def draw_circle_add(cx, cy, radius):
        """Draw filled circle, adding 1.0 to mask."""
        # Use vectorized approach for speed
        y_coords, x_coords = np.ogrid[-radius:radius+1, -radius:radius+1]
        circle_mask = x_coords**2 + y_coords**2 <= radius**2

        y_start = max(0, int(cy) - radius)
        y_end = min(h, int(cy) + radius + 1)
        x_start = max(0, int(cx) - radius)
        x_end = min(w, int(cx) + radius + 1)

        cm_y_start = y_start - (int(cy) - radius)
        cm_y_end = cm_y_start + (y_end - y_start)
        cm_x_start = x_start - (int(cx) - radius)
        cm_x_end = cm_x_start + (x_end - x_start)

        if y_end > y_start and x_end > x_start:
            mask[y_start:y_end, x_start:x_end] = np.where(
                circle_mask[cm_y_start:cm_y_end, cm_x_start:cm_x_end],
                1.0,
                mask[y_start:y_end, x_start:x_end]
            )

    def draw_irregular_line(start, end, pivot, is_vertical):
        """Draw circles along an edge for organic variation."""
        i = start
        while i < end:
            radius = rng.randint(5, max(6, circle_factor))
            if is_vertical:
                draw_circle_add(pivot, i, radius)
            else:
                draw_circle_add(i, pivot, radius)
            # Step BY radius - creates natural overlap
            i += radius

    # Draw circles OUTSIDE bbox on INTERIOR edges only
    # This creates organic "bulges" that extend past the rectangular edge

    if not is_top_edge:
        # Top interior edge - draw circles above y1
        draw_irregular_line(x1, x2, y1 - circle_factor // 2, is_vertical=False)

    if not is_bottom_edge:
        # Bottom interior edge - draw circles below y2
        draw_irregular_line(x1, x2, y2 + circle_factor // 2, is_vertical=False)

    if not is_left_edge:
        # Left interior edge - draw circles left of x1
        draw_irregular_line(y1, y2, x1 - circle_factor // 2, is_vertical=True)

    if not is_right_edge:
        # Right interior edge - draw circles right of x2
        draw_irregular_line(y1, y2, x2 + circle_factor // 2, is_vertical=True)


def create_tile_mask_organic(mask_h, mask_w, bbox, crop_region, blur_radius,
                              is_left_edge, is_right_edge, is_top_edge, is_bottom_edge,
                              irregularity=0.0, seed=None):
    """
    Create mask with organic edges + Gaussian blur feathering.

    v1.5.0 Algorithm:
    1. Fill bbox rectangle with 1.0
    2. Draw organic circles on INTERIOR edges (if irregularity > 0)
    3. Apply Gaussian blur for feathering (ALWAYS when blur_radius > 0)
    4. Restore sharp OUTER edges (image boundaries)

    This ensures blur ALWAYS works regardless of irregularity value.

    Args:
        mask_h, mask_w: Target mask dimensions
        bbox: Tile bounding box in image coordinates
        crop_region: Expanded crop region in image coordinates
        blur_radius: Blur amount for feathering
        is_*_edge: Whether tile is at image boundary (keep sharp)
        irregularity: 0.0=rectangular, 0.5=natural organic, 1.0=very organic
        seed: Random seed for reproducibility

    Returns:
        numpy array mask with Gaussian-blurred edges and organic variation
    """
    x1, y1, x2, y2 = bbox
    cx1, cy1, cx2, cy2 = crop_region

    # Calculate relative positions within crop region (mask coordinates)
    rel_x1 = x1 - cx1
    rel_y1 = y1 - cy1
    rel_x2 = x2 - cx1
    rel_y2 = y2 - cy1

    # Clamp to mask bounds
    rel_x1 = max(0, min(rel_x1, mask_w))
    rel_y1 = max(0, min(rel_y1, mask_h))
    rel_x2 = max(0, min(rel_x2, mask_w))
    rel_y2 = max(0, min(rel_y2, mask_h))

    # Start with zeros
    mask = np.zeros((mask_h, mask_w), dtype=np.float32)

    # STEP 1: Fill interior with 1.0
    mask[rel_y1:rel_y2, rel_x1:rel_x2] = 1.0

    # STEP 1.5: For OUTER edges, fill FULL strips to mask boundary BEFORE blur
    # v1.5.2: Simple approach - fill generous strips at outer edges regardless of bbox position
    # This works with any latent_divisor/crop_factor alignment because it doesn't depend on rel_* values
    outer_margin = max(blur_radius * 4, 32)  # Generous margin for outer edges

    if is_left_edge:
        # Fill left strip (full height) - prevents any gradient at left image boundary
        mask[:, :min(outer_margin, mask_w)] = 1.0
    if is_right_edge:
        # Fill right strip (full height) - prevents any gradient at right image boundary
        mask[:, max(0, mask_w - outer_margin):] = 1.0
    if is_top_edge:
        # Fill top strip (full width) - prevents any gradient at top image boundary
        mask[:min(outer_margin, mask_h), :] = 1.0
    if is_bottom_edge:
        # Fill bottom strip (full width) - prevents any gradient at bottom image boundary
        mask[max(0, mask_h - outer_margin):, :] = 1.0

    # STEP 2: Draw organic circles on INTERIOR edges only (if irregularity > 0)
    if irregularity > 0 and blur_radius > 0:
        # Generate seed from tile position for consistency
        tile_seed = seed if seed is not None else ((x1 * 1000 + y1) % 100000)
        draw_irregular_edges_inplace(
            mask,
            (rel_x1, rel_y1, rel_x2, rel_y2),  # bbox in mask coordinates
            irregularity,
            is_left_edge, is_right_edge, is_top_edge, is_bottom_edge,
            seed=tile_seed
        )

    # STEP 3: Apply Gaussian blur for feathering (ALWAYS, if blur_radius > 0)
    if blur_radius > 0:
        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_uint8, mode='L')
        blurred_pil = mask_pil.filter(ImageFilter.GaussianBlur(blur_radius))
        mask = np.array(blurred_pil).astype(np.float32) / 255.0

    # STEP 4: Safety net - restore outer edges after blur (redundant with v1.5.2 but harmless)
    # v1.5.2: STEP 1.5 now fills strips BEFORE blur, so this is just a safety net
    # Keeping smaller margin since STEP 1.5 already did the heavy lifting
    edge_margin = max(1, blur_radius * 2) if blur_radius > 0 else 0

    if edge_margin > 0:
        if is_left_edge:
            mask[:, :edge_margin] = 1.0
        if is_right_edge:
            mask[:, -edge_margin:] = 1.0
        if is_top_edge:
            mask[:edge_margin, :] = 1.0
        if is_bottom_edge:
            mask[-edge_margin:, :] = 1.0

    return mask


# ============================================================================
# NODE CLASS
# ============================================================================

class ArchAi3D_Smart_Tile_SEGS_Blur:
    """
    Combined node: Creates SEGS from tile dimensions and applies mask blur.

    This node combines Smart Tile SEGS + SEGS Mask Blur into one streamlined node.
    Connect bundle from Smart Tile Calculator for auto-configuration.

    v1.6.0 Features:
    - Uses tile_padding for proper tile overlap (fixes dark cross at seams)
    - Step = tile_size - tile_padding creates overlap for blur blending

    v1.5.2 Features:
    - Robust edge filling: full strips at outer edges (works with any latent_divisor)
    - Gaussian blur ALWAYS applied when blur_radius > 0
    - Optional organic circle variation for invisible seams
    - Direct per-tile mask generation (no caching artifacts)
    - Crop regions divisible by 8/32 for VAE/sampler compatibility
    - Perfect tile alignment matching Calculator's formula
    - Selective blur: only interior seams, outer edges stay sharp

    Parameters:
    - mask_blur: Gaussian blur radius (ALWAYS applied for feathering)
    - irregularity: 0=rectangular edges, 0.5=natural organic, 1.0=very organic
    - Latent divisor ensures crop regions are sampler-compatible (8 or 32)
    """

    CATEGORY = "ArchAi3d/Utils"
    RETURN_TYPES = ("SEGS", "SMART_TILE_BUNDLE")
    RETURN_NAMES = ("segs", "bundle")
    FUNCTION = "create_segs_with_blur"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Image to create SEGS from (or use bundle)"
                }),
            },
            "optional": {
                "bundle": ("SMART_TILE_BUNDLE", {
                    "tooltip": "Connect from Smart Tile Calculator - provides all settings automatically"
                }),
                "tile_width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 8,
                    "tooltip": "Width of each tile (overridden by bundle if connected)"
                }),
                "tile_height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 8,
                    "tooltip": "Height of each tile (overridden by bundle if connected)"
                }),
                "tiles_x": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 16,
                    "tooltip": "Number of horizontal tiles (overridden by bundle if connected)"
                }),
                "tiles_y": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 16,
                    "tooltip": "Number of vertical tiles (overridden by bundle if connected)"
                }),
                "tile_padding": ("INT", {
                    "default": 32,
                    "min": 0,
                    "max": 512,
                    "step": 8,
                    "tooltip": "Tile overlap/padding - NOT used for positioning (bundle handles that)"
                }),
                "crop_factor": ("FLOAT", {
                    "default": 1.5,
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Context expansion around tiles (overridden by bundle if connected)"
                }),
                "mask_blur": ("INT", {
                    "default": 8,
                    "min": 0,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Blur radius for interior seams (outer edges stay sharp)"
                }),
                "irregularity": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Organic edge variation (0=rectangular, 0.5=natural organic, 1.0=very organic)"
                }),
                "latent_divisor": ("INT", {
                    "default": 8,
                    "min": 8,
                    "max": 64,
                    "step": 8,
                    "tooltip": "Crop region dimensions divisible by this (8=VAE min, 32=recommended for sampler)"
                }),
            }
        }

    def create_segs_with_blur(self, image, bundle=None, tile_width=512, tile_height=512,
                               tiles_x=2, tiles_y=2, tile_padding=32, crop_factor=1.5,
                               mask_blur=8, irregularity=0.0, latent_divisor=8):
        """
        Create SEGS from tile grid with perfect alignment and organic blur.

        v1.5.2: Robust edge filling - full strips at outer edges (any latent_divisor).

        Args:
            image: Input image tensor (B, H, W, C)
            bundle: Optional SMART_TILE_BUNDLE with all settings
            tile_width, tile_height: Tile dimensions
            tiles_x, tiles_y: Grid dimensions
            tile_padding: Overlap between tiles (for reference, not positioning)
            crop_factor: Context expansion factor
            mask_blur: Blur radius for interior seams only (ALWAYS applied)
            irregularity: 0=rectangular edges, 0.5=natural organic, 1.0=very organic
            latent_divisor: Crop region dimensions divisible by this (8 or 32)

        Returns:
            (segs, bundle) - SEGS with Gaussian-blurred masks and bundle pass-through
        """
        # Extract from bundle if provided (overrides individual inputs)
        if bundle is not None:
            image = bundle.get("scaled_image", image)
            tile_width = bundle.get("tile_width", tile_width)
            tile_height = bundle.get("tile_height", tile_height)
            tiles_x = bundle.get("tiles_x", tiles_x)
            tiles_y = bundle.get("tiles_y", tiles_y)
            tile_padding = bundle.get("tile_padding", tile_padding)
            crop_factor = bundle.get("crop_factor", crop_factor)
            mask_blur = bundle.get("mask_blur", mask_blur)
            irregularity = bundle.get("irregularity", irregularity)
            latent_divisor = bundle.get("latent_divisor", latent_divisor)
            print(f"[Smart Tile SEGS Blur v1.6.0] Using bundle: {tiles_x}x{tiles_y} tiles, blur={mask_blur}, padding={tile_padding}")
        else:
            print(f"[Smart Tile SEGS Blur v1.6.0] Manual config: {tiles_x}x{tiles_y} tiles, blur={mask_blur}, padding={tile_padding}")

        # Get image dimensions (B, H, W, C)
        _, ih, iw, _ = image.shape

        total_tiles = tiles_x * tiles_y

        # Calculate tile positions with proper overlap using tile_padding
        # v1.6.0: Step = tile_size - tile_padding for proper blur blending
        positions = calculate_tile_positions(iw, ih, tile_width, tile_height, tiles_x, tiles_y, tile_padding)

        # Debug: show calculated step sizes (using tile_padding formula)
        step_w = tile_width - tile_padding if tiles_x > 1 else 0
        step_h = tile_height - tile_padding if tiles_y > 1 else 0
        overlap_w = tile_width - step_w if tiles_x > 1 else 0
        overlap_h = tile_height - step_h if tiles_y > 1 else 0

        print(f"  Image: {iw}x{ih}, Tile: {tile_width}x{tile_height}")
        print(f"  Step: {step_w}x{step_h} (overlap={tile_padding}px on each seam)")
        print(f"  Crop factor: {crop_factor}, Mask blur: {mask_blur}")
        print(f"  Irregularity: {irregularity}")
        print(f"  Latent divisor: {latent_divisor} (crop regions divisible by {latent_divisor})")
        mode_desc = "rectangular + Gaussian blur" if irregularity <= 0 else f"organic circles + Gaussian blur"
        print(f"  Feathering: {mode_desc}, interior seams only")

        segs = []

        for pos in positions:
            bbox = pos['bbox']
            x1, y1, x2, y2 = bbox

            # Verify tile size consistency
            actual_w = x2 - x1
            actual_h = y2 - y1

            # Create expanded crop region for context (aligned to latent_divisor)
            crop_region = make_crop_region(iw, ih, bbox, crop_factor, latent_divisor)
            cx1, cy1, cx2, cy2 = crop_region

            # Mask dimensions match crop region
            mask_h = cy2 - cy1
            mask_w = cx2 - cx1

            # Create mask with Gaussian blur (and organic variation if irregularity > 0)
            # v1.5.0: Blur ALWAYS applied, organic circles add variation before blur
            mask = create_tile_mask_organic(
                mask_h, mask_w,
                bbox, crop_region,
                mask_blur,
                is_left_edge=pos['is_left_edge'],
                is_right_edge=pos['is_right_edge'],
                is_top_edge=pos['is_top_edge'],
                is_bottom_edge=pos['is_bottom_edge'],
                irregularity=irregularity
            )

            # Create label
            label = f"tile_{pos['row']}_{pos['col']}"

            # Create SEG
            seg = SEG(
                cropped_image=None,
                cropped_mask=mask,
                confidence=1.0,
                crop_region=crop_region,
                bbox=bbox,
                label=label,
                control_net_wrapper=None
            )
            segs.append(seg)

        # SEGS format: ((height, width), [list of SEG])
        result = ((ih, iw), segs)

        # Debug: show tile positions and sizes
        print(f"  Tile positions (perfect alignment):")
        for i, pos in enumerate(positions):
            bbox = pos['bbox']
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            edges = []
            if pos['is_left_edge']: edges.append('L')
            if pos['is_right_edge']: edges.append('R')
            if pos['is_top_edge']: edges.append('T')
            if pos['is_bottom_edge']: edges.append('B')
            edge_str = ''.join(edges) if edges else 'interior'
            print(f"    [{i}] row={pos['row']}, col={pos['col']}: ({bbox[0]},{bbox[1]})-({bbox[2]},{bbox[3]}) size={w}x{h} edges={edge_str}")

        # Debug: show mask stats for corner and center tiles
        if segs and mask_blur > 0:
            # First tile (top-left corner)
            first_mask = segs[0].cropped_mask
            h, w = first_mask.shape
            print(f"  Mask[0] (top-left): top={first_mask[0, w//2]:.2f}, left={first_mask[h//2, 0]:.2f}, center={first_mask[h//2, w//2]:.2f}")

            # Last tile (bottom-right corner)
            if len(segs) > 1:
                last_mask = segs[-1].cropped_mask
                h, w = last_mask.shape
                print(f"  Mask[-1] (bot-right): bottom={last_mask[h-1, w//2]:.2f}, right={last_mask[h//2, w-1]:.2f}, center={last_mask[h//2, w//2]:.2f}")

        blur_type = "organic + Gaussian" if irregularity > 0 else "Gaussian"
        print(f"[Smart Tile SEGS Blur v1.6.0] Created {len(segs)} SEGS with {blur_type} blur, overlap={tile_padding}px")

        # Return segs + bundle pass-through
        return (result, bundle)


# ============================================================================
# NODE REGISTRATION
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "ArchAi3D_Smart_Tile_SEGS_Blur": ArchAi3D_Smart_Tile_SEGS_Blur,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_Smart_Tile_SEGS_Blur": "🧱 Smart Tile SEGS Blur",
}
