# ArchAi3D Smart Tile Merger
#
# Composites processed tiles using normalized weight blending
# Eliminates seam artifacts by ensuring weights sum to 1.0 in overlap regions
#
# Author: Amir Ferdos (ArchAi3d)
# Version: 1.0.0 - Initial release with normalized weight blending
# License: Dual License (Free for personal use, Commercial license required for business use)

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter


# ============================================================================
# BLENDING ALGORITHMS
# ============================================================================

def create_linear_blend_mask(tile_h, tile_w, overlap, is_left, is_right, is_top, is_bottom):
    """
    Create a linear blend mask that guarantees mask_A + mask_B = 1.0 in overlap.

    Unlike Gaussian blur which creates arbitrary falloffs, linear ramps
    ensure perfect blending when tiles overlap.

    Args:
        tile_h, tile_w: Tile dimensions
        overlap: Overlap size in pixels
        is_left, is_right, is_top, is_bottom: Whether tile is at image boundary

    Returns:
        numpy array (H, W) with values 0.0-1.0
    """
    mask = np.ones((tile_h, tile_w), dtype=np.float32)

    if overlap <= 0:
        return mask

    # Create linear ramps for interior edges
    # Left edge (if not at image boundary)
    if not is_left and overlap > 0:
        ramp = np.linspace(0.0, 1.0, overlap)
        for i in range(min(overlap, tile_w)):
            mask[:, i] *= ramp[i]

    # Right edge (if not at image boundary)
    if not is_right and overlap > 0:
        ramp = np.linspace(1.0, 0.0, overlap)
        for i in range(min(overlap, tile_w)):
            mask[:, tile_w - overlap + i] *= ramp[i]

    # Top edge (if not at image boundary)
    if not is_top and overlap > 0:
        ramp = np.linspace(0.0, 1.0, overlap)
        for i in range(min(overlap, tile_h)):
            mask[i, :] *= ramp[i]

    # Bottom edge (if not at image boundary)
    if not is_bottom and overlap > 0:
        ramp = np.linspace(1.0, 0.0, overlap)
        for i in range(min(overlap, tile_h)):
            mask[tile_h - overlap + i, :] *= ramp[i]

    return mask


def create_smooth_blend_mask(tile_h, tile_w, overlap, blur_radius,
                              is_left, is_right, is_top, is_bottom):
    """
    Create a smooth blend mask with Gaussian blur, but normalized for proper blending.

    Args:
        tile_h, tile_w: Tile dimensions
        overlap: Overlap size in pixels
        blur_radius: Gaussian blur radius
        is_left, is_right, is_top, is_bottom: Whether tile is at image boundary

    Returns:
        numpy array (H, W) with values 0.0-1.0
    """
    # Start with linear mask
    mask = create_linear_blend_mask(tile_h, tile_w, overlap,
                                     is_left, is_right, is_top, is_bottom)

    if blur_radius > 0:
        # Apply Gaussian blur for smoother transitions
        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_uint8, mode='L')
        blurred_pil = mask_pil.filter(ImageFilter.GaussianBlur(blur_radius))
        mask = np.array(blurred_pil).astype(np.float32) / 255.0

        # Restore sharp outer edges
        edge_margin = blur_radius * 2
        if is_left:
            mask[:, :edge_margin] = 1.0
        if is_right:
            mask[:, -edge_margin:] = 1.0
        if is_top:
            mask[:edge_margin, :] = 1.0
        if is_bottom:
            mask[-edge_margin:, :] = 1.0

    return mask


# ============================================================================
# NODE CLASS
# ============================================================================

class ArchAi3D_Smart_Tile_Merger:
    """
    Merge processed tiles using normalized weight blending.

    This node solves the seam problem by:
    1. Accumulating all tiles with their weights
    2. Dividing by total weight to normalize

    This ensures that in overlap regions, the weights always sum to 1.0,
    eliminating dark or bright seams.

    Features:
    - Normalized weight blending (no seams)
    - Linear or smooth blend modes
    - Works with SEGS from Smart Tile Detailer
    - Configurable overlap and blur
    """

    CATEGORY = "ArchAi3d/Utils"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "merge_tiles"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE", {
                    "tooltip": "Base image (same size as output, used for fallback areas)"
                }),
                "segs": ("SEGS", {
                    "tooltip": "SEGS with processed tiles (from Smart Tile Detailer)"
                }),
            },
            "optional": {
                "bundle": ("SMART_TILE_BUNDLE", {
                    "tooltip": "Bundle from Smart Tile Calculator (provides overlap settings)"
                }),
                "blend_mode": (["normalized", "linear", "smooth", "use_seg_mask"], {
                    "default": "normalized",
                    "tooltip": "Blending mode: normalized=best quality, linear=fast, smooth=Gaussian, use_seg_mask=use masks from SEGS"
                }),
                "overlap": ("INT", {
                    "default": 32,
                    "min": 0,
                    "max": 256,
                    "step": 8,
                    "tooltip": "Overlap between tiles (overridden by bundle if connected)"
                }),
                "blur_radius": ("INT", {
                    "default": 8,
                    "min": 0,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Blur radius for smooth mode (overridden by bundle if connected)"
                }),
            }
        }

    def merge_tiles(self, base_image, segs, bundle=None, blend_mode="normalized",
                    overlap=32, blur_radius=8):
        """
        Merge tiles using normalized weight blending.

        Args:
            base_image: Base image tensor (B, H, W, C)
            segs: SEGS tuple ((h, w), [list of SEG with cropped_image])
            bundle: Optional bundle with overlap settings
            blend_mode: Blending algorithm to use
            overlap: Overlap between tiles
            blur_radius: Blur radius for smooth mode

        Returns:
            Merged image with no seams
        """
        # Extract from bundle if provided
        if bundle is not None:
            overlap = bundle.get("tile_padding", overlap)
            blur_radius = bundle.get("mask_blur", blur_radius)
            print(f"[Smart Tile Merger v1.0] Using bundle: overlap={overlap}, blur={blur_radius}")

        # Unpack SEGS
        (img_h, img_w), seg_list = segs

        if not seg_list:
            print("[Smart Tile Merger v1.0] No segments to merge")
            return (base_image,)

        # Get device from base_image
        device = base_image.device

        # Initialize accumulators
        # result_accum: sum of (tile * weight)
        # weight_accum: sum of weights
        result_accum = torch.zeros_like(base_image, dtype=torch.float32, device=device)
        weight_accum = torch.zeros((1, img_h, img_w, 1), dtype=torch.float32, device=device)

        print(f"\n[Smart Tile Merger v1.0] Merging {len(seg_list)} tiles...")
        print(f"  Image: {img_w}x{img_h}")
        print(f"  Blend mode: {blend_mode}, Overlap: {overlap}, Blur: {blur_radius}")

        # Determine grid size from labels
        tiles_x = 1
        tiles_y = 1
        for seg in seg_list:
            if seg.label and seg.label.startswith("tile_"):
                parts = seg.label.split("_")
                if len(parts) == 3:
                    try:
                        row = int(parts[1])
                        col = int(parts[2])
                        tiles_y = max(tiles_y, row + 1)
                        tiles_x = max(tiles_x, col + 1)
                    except ValueError:
                        pass

        print(f"  Grid: {tiles_x}x{tiles_y}")

        # Process each SEG
        for i, seg in enumerate(seg_list):
            # Get tile image
            if seg.cropped_image is None:
                print(f"  Warning: SEG {i} has no cropped_image, skipping")
                continue

            # Convert to tensor if needed
            if isinstance(seg.cropped_image, np.ndarray):
                tile_img = torch.from_numpy(seg.cropped_image).to(device)
            else:
                tile_img = seg.cropped_image.to(device)

            # Ensure 4D
            if tile_img.dim() == 3:
                tile_img = tile_img.unsqueeze(0)

            # Get bbox
            bbox = seg.bbox
            x1, y1, x2, y2 = bbox
            tile_w = x2 - x1
            tile_h = y2 - y1

            # Parse tile position
            row, col = 0, 0
            if seg.label and seg.label.startswith("tile_"):
                parts = seg.label.split("_")
                if len(parts) == 3:
                    try:
                        row = int(parts[1])
                        col = int(parts[2])
                    except ValueError:
                        pass

            # Determine edge positions
            is_left = (col == 0)
            is_right = (col == tiles_x - 1)
            is_top = (row == 0)
            is_bottom = (row == tiles_y - 1)

            # Create blend mask based on mode
            if blend_mode == "use_seg_mask" and seg.cropped_mask is not None:
                # Use mask from SEGS (already blurred)
                if isinstance(seg.cropped_mask, np.ndarray):
                    blend_mask = torch.from_numpy(seg.cropped_mask).float()
                else:
                    blend_mask = seg.cropped_mask.float()
            elif blend_mode == "linear":
                # Linear ramp masks
                mask_np = create_linear_blend_mask(tile_h, tile_w, overlap,
                                                    is_left, is_right, is_top, is_bottom)
                blend_mask = torch.from_numpy(mask_np)
            elif blend_mode == "smooth":
                # Gaussian-smoothed masks
                mask_np = create_smooth_blend_mask(tile_h, tile_w, overlap, blur_radius,
                                                    is_left, is_right, is_top, is_bottom)
                blend_mask = torch.from_numpy(mask_np)
            else:  # "normalized" (default)
                # For normalized mode, we use linear masks and rely on weight normalization
                mask_np = create_linear_blend_mask(tile_h, tile_w, overlap,
                                                    is_left, is_right, is_top, is_bottom)
                blend_mask = torch.from_numpy(mask_np)

            blend_mask = blend_mask.to(device)

            # Ensure mask matches tile size
            if blend_mask.shape[0] != tile_h or blend_mask.shape[1] != tile_w:
                blend_mask = F.interpolate(
                    blend_mask.unsqueeze(0).unsqueeze(0),
                    size=(tile_h, tile_w),
                    mode='bilinear', align_corners=False
                ).squeeze()

            # Reshape mask for broadcasting (1, H, W, 1)
            mask_4d = blend_mask.unsqueeze(0).unsqueeze(-1)

            # Ensure tile image matches mask size
            if tile_img.shape[1] != tile_h or tile_img.shape[2] != tile_w:
                tile_img = F.interpolate(
                    tile_img.permute(0, 3, 1, 2),
                    size=(tile_h, tile_w),
                    mode='bilinear', align_corners=False
                ).permute(0, 2, 3, 1)

            # Accumulate weighted tile
            result_accum[:, y1:y2, x1:x2, :] += tile_img * mask_4d
            weight_accum[:, y1:y2, x1:x2, :] += mask_4d

            print(f"  Tile {i+1}/{len(seg_list)} ({seg.label}): bbox=({x1},{y1})-({x2},{y2})")

        # Normalize by weights
        # Avoid division by zero - use base_image where weight is 0
        weight_accum = torch.clamp(weight_accum, min=1e-8)
        result = result_accum / weight_accum

        # Fill areas with no tile coverage using base_image
        # (This shouldn't happen with proper tiling, but just in case)
        no_coverage = (weight_accum < 0.001).squeeze(-1).unsqueeze(-1).expand_as(base_image)
        result = torch.where(no_coverage, base_image, result)

        # Ensure proper range [0, 1]
        result = torch.clamp(result, 0.0, 1.0)

        print(f"\n[Smart Tile Merger v1.0] Merge complete!")
        print(f"  Weight range: {weight_accum.min():.3f} - {weight_accum.max():.3f}")

        return (result,)


# ============================================================================
# NODE REGISTRATION
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "ArchAi3D_Smart_Tile_Merger": ArchAi3D_Smart_Tile_Merger,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_Smart_Tile_Merger": "🔀 Smart Tile Merger",
}
