"""
USDU Edge Repair - Preview Mode
================================

Generate preview images using TileGeometry for all calculations.
This ensures preview shows EXACT same data used in actual processing.
"""

import torch
from .utils import tensor_to_pil, pil_to_tensor


def generate_tile_previews(geometry, upscaled_image, edge_mask_width, edge_mask_feather, debug_info):
    """
    Generate preview images using TileGeometry.

    All geometry calculations come from the same TileGeometry instance
    used by the actual processing, guaranteeing consistency.

    Returns 4 separate batches:
    1. tiles_original: Exact tile rectangle (NO padding, NO overlap)
    2. tiles_padded: Tile WITH padding context (what gets cropped for processing)
    3. tiles_blend_mask: The ACTUAL blending mask used in compositing
    4. tiles_edge_mask: Edge masks showing border zones for DiffDiff

    Args:
        geometry: TileGeometry instance (single source of truth)
        upscaled_image: Input image tensor [B, H, W, C] - already padded by geometry
        edge_mask_width: Width of edge border in pixels
        edge_mask_feather: Feather amount for edge mask
        debug_info: Debug string to append to

    Returns:
        Tuple of (image, tiles_original, tiles_padded, tiles_blend_mask, tiles_edge_mask, debug_info)
    """
    pil_image = tensor_to_pil(upscaled_image, 0)

    tiles_original = []
    tiles_padded = []
    tiles_blend_mask = []
    tiles_edge_mask = []

    print(f"\n[Preview Mode] Generating tile previews using TileGeometry...")
    print(geometry.get_debug_info())

    for i in range(geometry.total_tiles):
        xi, yi = geometry.get_tile_coords(i)

        # 1. tiles_original: Exact tile rectangle (from geometry)
        original = geometry.get_tile_original(pil_image, i)
        tiles_original.append(pil_to_tensor(original))

        # 2. tiles_padded: Tile WITH padding context (from geometry)
        padded = geometry.get_tile_crop(pil_image, i)
        tiles_padded.append(pil_to_tensor(padded))

        # 3. tiles_blend_mask: ACTUAL blending mask (from geometry)
        # Crop to tile_rect (not padded_rect) to show only the blend zone
        # This avoids black strips on edge tiles where canvas padding extends
        # beyond the tile_rect (that padding area is cropped away in final output)
        blend_mask = geometry.get_blend_mask(i)
        tile_rect = geometry.get_tile_rect(i)
        blend_mask_crop = blend_mask.crop(tile_rect)
        # Resize to padded_tile_size for consistent display with other previews
        blend_mask_crop = blend_mask_crop.resize(geometry.padded_tile_size)
        blend_mask_rgb = blend_mask_crop.convert('RGB')
        tiles_blend_mask.append(pil_to_tensor(blend_mask_rgb))

        # 4. tiles_edge_mask: Edge mask for DiffDiff (from geometry)
        edge_mask = geometry.create_edge_mask(i, edge_mask_width, edge_mask_feather)
        edge_mask_rgb = edge_mask.convert('RGB')
        tiles_edge_mask.append(pil_to_tensor(edge_mask_rgb))

        tile_rect = geometry.get_tile_rect(i)
        print(f"  Tile [{i}] ({xi},{yi}): rect={tile_rect}, padded={padded.size}")

    # Stack into batch tensors
    batch_original = torch.cat(tiles_original, dim=0)
    batch_padded = torch.cat(tiles_padded, dim=0)
    batch_blend_mask = torch.cat(tiles_blend_mask, dim=0)
    batch_edge_mask = torch.cat(tiles_edge_mask, dim=0)

    # Update debug info
    padded_w, padded_h = geometry.padded_tile_size
    preview_debug = debug_info + f"\n\n[PREVIEW MODE - Using TileGeometry]"
    preview_debug += f"\nOutputs: 4 batches x {geometry.total_tiles} tiles"
    preview_debug += f"\n  tiles_original:   {geometry.tile_width}x{geometry.tile_height} (exact tile rect)"
    preview_debug += f"\n  tiles_padded:     {padded_w}x{padded_h} (tile + {geometry.tile_padding}px padding)"
    preview_debug += f"\n  tiles_blend_mask: From TileGeometry (blur={geometry.mask_blur})"
    preview_debug += f"\n  tiles_edge_mask:  Edge mask (width={edge_mask_width}, feather={edge_mask_feather})"

    print(f"[Preview Mode] Generated {geometry.total_tiles} tiles using TileGeometry")

    return (upscaled_image, batch_original, batch_padded, batch_blend_mask, batch_edge_mask, preview_debug)
