"""
Simple USDU - Blend Mask Generator
===================================

Creates blend masks for tile compositing using distance-based masking.

KEY FEATURE: Distance-based masking creates UNIFORM border thickness on ALL edges,
including corners. This fixes the diagonal-dark-corner problem in the old system.
"""

import numpy as np
import torch
from PIL import Image
from .utils import get_tile_position, pil_to_tensor


class ArchAi3D_Simple_Blend_Mask:
    """
    Create blend mask for a single tile using distance-based masking.

    The mask determines how this tile blends with neighbors:
    - White (1.0) = full contribution from this tile
    - Black (0.0) = no contribution (overlap zone)
    - Gradient = smooth transition

    KEY: Only creates gradients on INTERNAL edges (where tiles meet).
    Boundary edges (at image borders) stay sharp white.

    This gives UNIFORM border thickness everywhere - no dark diagonal corners.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tile_idx": ("INT", {"default": 0, "min": 0, "max": 1000}),
                "tile_width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "tile_height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "tiles_x": ("INT", {"default": 2, "min": 1, "max": 100}),
                "tiles_y": ("INT", {"default": 2, "min": 1, "max": 100}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 512,
                                    "tooltip": "Overlap between tiles (same as tile_padding)"}),
                "mask_blur": ("INT", {"default": 16, "min": 0, "max": 256,
                                      "tooltip": "Gradient width in pixels (blend zone)"}),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("blend_mask", "blend_mask_rgb")
    FUNCTION = "create_mask"
    CATEGORY = "ArchAi3d/Upscaling/Simple USDU"

    def create_mask(self, tile_idx, tile_width, tile_height, tiles_x, tiles_y,
                    overlap, mask_blur):
        """
        Create distance-based blend mask for a single tile.

        The mask uses minimum distance to nearest internal edge.
        This gives uniform gradient thickness on edges AND corners.

        Args:
            tile_idx: Which tile (0-indexed)
            tile_width: Width of each tile
            tile_height: Height of each tile
            tiles_x: Number of tiles horizontally
            tiles_y: Number of tiles vertically
            overlap: Overlap between tiles
            mask_blur: Gradient width (blend zone width)

        Returns:
            blend_mask: MASK tensor [H, W]
            blend_mask_rgb: IMAGE tensor [1, H, W, 3] for preview
        """
        xi, yi = get_tile_position(tile_idx, tiles_x, tiles_y)

        # Calculate output image size
        step_x = tile_width - overlap
        step_y = tile_height - overlap
        output_w = step_x * (tiles_x - 1) + tile_width
        output_h = step_y * (tiles_y - 1) + tile_height

        # This tile's output rectangle (non-overlapping region)
        x1 = xi * step_x
        y1 = yi * step_y
        x2 = x1 + tile_width
        y2 = y1 + tile_height

        # Clamp to image bounds
        x2 = min(x2, output_w)
        y2 = min(y2, output_h)

        tile_h = y2 - y1
        tile_w = x2 - x1

        # Handle no blur case
        if mask_blur <= 0:
            # White tile on black canvas
            result = np.zeros((output_h, output_w), dtype=np.float32)
            result[y1:y2, x1:x2] = 1.0
            mask_pil = Image.fromarray((result * 255).astype(np.uint8), mode='L')
            return self._to_outputs(mask_pil, output_h, output_w)

        # === DISTANCE-BASED MASKING ===
        # For each pixel, find distance to nearest INTERNAL edge.
        # Internal edge = edge where this tile has a neighbor.
        # Boundary edge (no neighbor) = no gradient, stays white.

        # Create coordinate grids for the tile area (vectorized)
        ys = np.arange(tile_h).reshape(-1, 1)  # Column vector [tile_h, 1]
        xs = np.arange(tile_w).reshape(1, -1)  # Row vector [1, tile_w]

        # Start with maximum possible distance (all white = 1.0)
        min_dist = np.full((tile_h, tile_w), float('inf'), dtype=np.float32)

        # Calculate distance to each INTERNAL edge (only edges with neighbors)
        # Distance is measured in pixels from the edge

        if xi > 0:  # Has left neighbor - gradient on left edge
            min_dist = np.minimum(min_dist, xs)  # Distance from left edge

        if xi < tiles_x - 1:  # Has right neighbor - gradient on right edge
            min_dist = np.minimum(min_dist, tile_w - 1 - xs)  # Distance from right edge

        if yi > 0:  # Has top neighbor - gradient on top edge
            min_dist = np.minimum(min_dist, ys)  # Distance from top edge

        if yi < tiles_y - 1:  # Has bottom neighbor - gradient on bottom edge
            min_dist = np.minimum(min_dist, tile_h - 1 - ys)  # Distance from bottom edge

        # Convert distance to gradient value:
        # - At edge (distance=0): value=0 (black, no contribution)
        # - At distance>=mask_blur: value=1 (white, full contribution)
        # - If no internal edges (inf distance): value=1 (white)
        tile_mask = np.clip(min_dist / mask_blur, 0.0, 1.0)

        # Place tile mask on canvas (black background)
        result = np.zeros((output_h, output_w), dtype=np.float32)
        result[y1:y2, x1:x2] = tile_mask

        mask_pil = Image.fromarray((result * 255).astype(np.uint8), mode='L')
        return self._to_outputs(mask_pil, output_h, output_w)

    def _to_outputs(self, mask_pil, height, width):
        """Convert PIL mask to output tensors."""
        # MASK format: [H, W] or [1, H, W]
        mask_np = np.array(mask_pil).astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_np)  # [H, W]

        # IMAGE format for preview: [1, H, W, 3]
        rgb_pil = mask_pil.convert('RGB')
        rgb_tensor = pil_to_tensor(rgb_pil)

        return (mask_tensor, rgb_tensor)


class ArchAi3D_Simple_Blend_Mask_Batch:
    """
    Create blend masks for ALL tiles at once.

    This is more efficient than calling Simple_Blend_Mask in a loop.
    Returns a batch of masks that can be used with Simple_Tile_Compositor.

    Accepts either individual parameters OR a tile_params bundle from Smart Tile Solver V6.2.
    When tile_params is connected, it overrides the individual INT inputs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_blur": ("INT", {"default": 16, "min": 0, "max": 256}),
            },
            "optional": {
                "tile_params": ("TILE_PARAMS", {"tooltip": "Bundle from Smart Tile Solver V6.2 (overrides individual inputs)"}),
                "tile_width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "tile_height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "tiles_x": ("INT", {"default": 2, "min": 1, "max": 100}),
                "tiles_y": ("INT", {"default": 2, "min": 1, "max": 100}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 512}),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE", "INT")
    RETURN_NAMES = ("blend_masks", "blend_masks_rgb", "total_tiles")
    FUNCTION = "create_masks"
    CATEGORY = "ArchAi3d/Upscaling/Simple USDU"

    def create_masks(self, mask_blur, tile_params=None, tile_width=512, tile_height=512, tiles_x=2, tiles_y=2, overlap=64):
        """Create blend masks for all tiles."""
        # Extract from bundle if provided
        if tile_params is not None:
            tile_width = tile_params.get("tile_width", tile_width)
            tile_height = tile_params.get("tile_height", tile_height)
            tiles_x = tile_params.get("tiles_x", tiles_x)
            tiles_y = tile_params.get("tiles_y", tiles_y)
            overlap = tile_params.get("overlap", overlap)

        total_tiles = tiles_x * tiles_y
        single_mask_node = ArchAi3D_Simple_Blend_Mask()

        masks = []
        masks_rgb = []

        for tile_idx in range(total_tiles):
            mask, mask_rgb = single_mask_node.create_mask(
                tile_idx, tile_width, tile_height, tiles_x, tiles_y, overlap, mask_blur
            )
            masks.append(mask.unsqueeze(0))  # Add batch dim
            masks_rgb.append(mask_rgb)

        # Stack into batches
        masks_batch = torch.cat(masks, dim=0)  # [N, H, W]
        masks_rgb_batch = torch.cat(masks_rgb, dim=0)  # [N, H, W, 3]

        return (masks_batch, masks_rgb_batch, total_tiles)


# Node registration
NODE_CLASS_MAPPINGS = {
    "ArchAi3D_Simple_Blend_Mask": ArchAi3D_Simple_Blend_Mask,
    "ArchAi3D_Simple_Blend_Mask_Batch": ArchAi3D_Simple_Blend_Mask_Batch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_Simple_Blend_Mask": "Simple Blend Mask",
    "ArchAi3D_Simple_Blend_Mask_Batch": "Simple Blend Mask (Batch)",
}
