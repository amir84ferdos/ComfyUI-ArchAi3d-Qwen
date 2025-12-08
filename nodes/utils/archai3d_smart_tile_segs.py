# ArchAi3D Smart Tile SEGS
#
# Create SEGS from Smart Tile Calculator outputs
# Compatible with Impact Pack's DetailerForEach and SEGSLabelAssign
#
# Author: Amir Ferdos (ArchAi3d)
# Version: 1.3.0 - Added SMART_TILE_BUNDLE input for one-wire connections
# License: Dual License (Free for personal use, Commercial license required for business use)

import numpy as np
import torch
from collections import namedtuple

# Define SEG namedtuple (compatible with Impact Pack)
SEG = namedtuple("SEG",
                 ['cropped_image', 'cropped_mask', 'confidence', 'crop_region', 'bbox', 'label', 'control_net_wrapper'],
                 defaults=[None])


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def bbox_overlaps(bbox1, bbox2):
    """
    Check if two bounding boxes overlap.

    Args:
        bbox1, bbox2: (x1, y1, x2, y2) bounding boxes

    Returns:
        True if boxes overlap, False otherwise
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Check if boxes don't overlap (any separation)
    if x2_1 <= x1_2 or x2_2 <= x1_1:  # Horizontal separation
        return False
    if y2_1 <= y1_2 or y2_2 <= y1_1:  # Vertical separation
        return False

    return True


def check_segs_overlap(tile_bbox, segs_list):
    """
    Check if tile bbox overlaps with any SEG in the list.

    Args:
        tile_bbox: (x1, y1, x2, y2) tile bounding box
        segs_list: List of SEG namedtuples

    Returns:
        True if tile overlaps with any SEG
    """
    for seg in segs_list:
        if bbox_overlaps(tile_bbox, seg.bbox):
            return True
    return False


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


def make_crop_region(w, h, bbox, crop_factor):
    """
    Create expanded crop region around bbox.

    Args:
        w, h: Image dimensions
        bbox: (x1, y1, x2, y2) bounding box
        crop_factor: How much to expand (1.0 = no expansion, 2.0 = 2x size)

    Returns:
        (cx1, cy1, cx2, cy2) - expanded crop region
    """
    x1, y1, x2, y2 = bbox

    bbox_w = x2 - x1
    bbox_h = y2 - y1

    # Calculate expanded size
    crop_w = int(bbox_w * crop_factor)
    crop_h = int(bbox_h * crop_factor)

    # Center the crop around the bbox
    cx1 = x1 - (crop_w - bbox_w) // 2
    cy1 = y1 - (crop_h - bbox_h) // 2

    # Normalize to fit within image
    cx1, cx2 = normalize_region(w, cx1, crop_w)
    cy1, cy2 = normalize_region(h, cy1, crop_h)

    return (cx1, cy1, cx2, cy2)


def create_tile_mask(mask_h, mask_w, bbox, crop_region, irregularity=0.0):
    """
    Create a mask for the tile within the crop region.

    Args:
        mask_h, mask_w: Mask dimensions (crop region size)
        bbox: (x1, y1, x2, y2) - tile bounding box in image coordinates
        crop_region: (cx1, cy1, cx2, cy2) - crop region in image coordinates
        irregularity: Amount of edge irregularity (0-1)

    Returns:
        numpy array mask
    """
    mask = np.zeros((mask_h, mask_w), dtype=np.float32)

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

    if irregularity > 0:
        # Add some noise to edges for better blending
        noise_mask = np.random.rand(mask_h, mask_w).astype(np.float32) * irregularity
        mask[rel_y1:rel_y2, rel_x1:rel_x2] = 1.0

        # Apply edge feathering
        edge_size = max(4, int((rel_x2 - rel_x1) * 0.1))

        # Horizontal feathering
        for i in range(edge_size):
            alpha = i / edge_size
            if rel_x1 + i < rel_x2:
                mask[rel_y1:rel_y2, rel_x1 + i] *= alpha
            if rel_x2 - i - 1 >= rel_x1:
                mask[rel_y1:rel_y2, rel_x2 - i - 1] *= alpha

        # Vertical feathering
        for i in range(edge_size):
            alpha = i / edge_size
            if rel_y1 + i < rel_y2:
                mask[rel_y1 + i, rel_x1:rel_x2] *= alpha
            if rel_y2 - i - 1 >= rel_y1:
                mask[rel_y2 - i - 1, rel_x1:rel_x2] *= alpha
    else:
        # Simple rectangular mask
        mask[rel_y1:rel_y2, rel_x1:rel_x2] = 1.0

    return mask


# ============================================================================
# NODE CLASS
# ============================================================================

class ArchAi3D_Smart_Tile_SEGS:
    """
    Create SEGS from Smart Tile Calculator outputs.

    Compatible with Impact Pack's DetailerForEach, SEGSLabelAssign, and other SEGS nodes.

    This node takes explicit tile dimensions (tile_width, tile_height, tiles_x, tiles_y)
    from Smart Tile Calculator and creates SEGS segments for each tile.

    Unlike MakeTileSEGS which only supports square tiles and calculates grid internally,
    this node supports rectangular tiles and explicit grid dimensions.
    """

    CATEGORY = "ArchAi3d/Utils"
    RETURN_TYPES = ("SEGS",)
    RETURN_NAMES = ("segs",)
    FUNCTION = "create_segs"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Image to create SEGS from (use upscaled image)"
                }),
                "tile_width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 8,
                    "tooltip": "Width of each tile (from Smart Tile Calculator)"
                }),
                "tile_height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 8,
                    "tooltip": "Height of each tile (from Smart Tile Calculator)"
                }),
                "tiles_x": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 16,
                    "tooltip": "Number of horizontal tiles (from Smart Tile Calculator)"
                }),
                "tiles_y": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 16,
                    "tooltip": "Number of vertical tiles (from Smart Tile Calculator)"
                }),
                "tile_padding": ("INT", {
                    "default": 32,
                    "min": 0,
                    "max": 512,
                    "step": 8,
                    "tooltip": "Tile padding/overlap (from Smart Tile Calculator)"
                }),
            },
            "optional": {
                "bundle": ("SMART_TILE_BUNDLE", {
                    "tooltip": "Connect bundle from Smart Tile Calculator (overrides individual inputs)"
                }),
                "filter_in_segs_opt": ("SEGS", {
                    "tooltip": "Only include tiles that overlap with these SEGS (e.g., face detection)"
                }),
                "filter_out_segs_opt": ("SEGS", {
                    "tooltip": "Exclude tiles that overlap with these SEGS"
                }),
                "crop_factor": ("FLOAT", {
                    "default": 1.5,
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Extra context around each tile for better blending (from Smart Tile Calculator)"
                }),
                "mask_irregularity": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "How irregular the mask edges are (0=sharp, 1=very irregular)"
                }),
            }
        }

    def create_segs(self, image, tile_width, tile_height, tiles_x, tiles_y,
                    tile_padding, bundle=None, filter_in_segs_opt=None, filter_out_segs_opt=None,
                    crop_factor=1.5, mask_irregularity=0.0):
        """
        Create SEGS from explicit tile grid.

        The SEGS format is: ((image_height, image_width), [list of SEG items])
        Each SEG has: cropped_image, cropped_mask, confidence, crop_region, bbox, label, control_net_wrapper

        Filtering:
        - filter_in_segs_opt: Only include tiles that overlap with these SEGS
        - filter_out_segs_opt: Exclude tiles that overlap with these SEGS
        """
        # If bundle provided, extract values (overrides individual inputs)
        if bundle is not None:
            image = bundle.get("scaled_image", image)
            tile_width = bundle.get("tile_width", tile_width)
            tile_height = bundle.get("tile_height", tile_height)
            tiles_x = bundle.get("tiles_x", tiles_x)
            tiles_y = bundle.get("tiles_y", tiles_y)
            tile_padding = bundle.get("tile_padding", tile_padding)
            crop_factor = bundle.get("crop_factor", crop_factor)
            print(f"[Smart Tile SEGS v1.3] Using bundle: {tiles_x}x{tiles_y} tiles, {tile_width}x{tile_height}px")

        # Get image dimensions (B, H, W, C)
        _, ih, iw, _ = image.shape

        # Extract SEG lists from SEGS tuples
        filter_in_list = None
        filter_out_list = None

        if filter_in_segs_opt is not None:
            _, filter_in_list = filter_in_segs_opt
            print(f"  Filter IN: {len(filter_in_list)} SEGS (only include overlapping tiles)")

        if filter_out_segs_opt is not None:
            _, filter_out_list = filter_out_segs_opt
            print(f"  Filter OUT: {len(filter_out_list)} SEGS (exclude overlapping tiles)")

        total_tiles = tiles_x * tiles_y

        # Calculate step sizes (tile size minus overlap/padding)
        step_w = tile_width - tile_padding
        step_h = tile_height - tile_padding

        print(f"\n[Smart Tile SEGS v1.3] Creating SEGS for {tiles_x}x{tiles_y} = {total_tiles} tiles")
        print(f"  Image: {iw}x{ih}, Tile: {tile_width}x{tile_height}, Padding: {tile_padding}")
        print(f"  Step: {step_w}x{step_h}, Crop factor: {crop_factor}")

        segs = []
        tile_num = 0
        filtered_count = 0

        for y in range(tiles_y):
            for x in range(tiles_x):
                tile_num += 1

                # Calculate tile boundaries using step size for overlapping tiles
                # Formula: x1 = x * step_w (where step_w = tile_width - tile_padding)
                x1 = x * step_w
                y1 = y * step_h
                x2 = x1 + tile_width
                y2 = y1 + tile_height

                # Clamp to image bounds (handles edge tiles automatically)
                x1 = max(0, min(x1, iw))
                y1 = max(0, min(y1, ih))
                x2 = max(0, min(x2, iw))
                y2 = max(0, min(y2, ih))

                bbox = (x1, y1, x2, y2)

                # Apply filtering
                # filter_in: Only include if tile overlaps with filter_in SEGS
                if filter_in_list is not None:
                    if not check_segs_overlap(bbox, filter_in_list):
                        filtered_count += 1
                        continue  # Skip this tile

                # filter_out: Exclude if tile overlaps with filter_out SEGS
                if filter_out_list is not None:
                    if check_segs_overlap(bbox, filter_out_list):
                        filtered_count += 1
                        continue  # Skip this tile

                # Create expanded crop region
                crop_region = make_crop_region(iw, ih, bbox, crop_factor)
                cx1, cy1, cx2, cy2 = crop_region

                # Create mask
                mask_h = cy2 - cy1
                mask_w = cx2 - cx1
                mask = create_tile_mask(mask_h, mask_w, bbox, crop_region, mask_irregularity)

                # Create label
                label = f"tile_{y}_{x}"

                # Create SEG (cropped_image is None, will be filled by DetailerForEach)
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

        if filtered_count > 0:
            print(f"[Smart Tile SEGS v1.3] Created {len(segs)} SEGS segments ({filtered_count} filtered out)")
        else:
            print(f"[Smart Tile SEGS v1.3] Created {len(segs)} SEGS segments")

        return (result,)


# ============================================================================
# NODE REGISTRATION
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "ArchAi3D_Smart_Tile_SEGS": ArchAi3D_Smart_Tile_SEGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_Smart_Tile_SEGS": "🧱 Smart Tile SEGS",
}
