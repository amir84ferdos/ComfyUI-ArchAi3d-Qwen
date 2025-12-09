# ArchAi3D SEGS To Bundle
#
# Bridge node: Convert Impact Pack's Make Tile SEGS output to SMART_TILE_BUNDLE
# Allows using Impact Pack's tiling with Smart Tile Prompter Turbo
#
# Author: Amir Ferdos (ArchAi3d)
# Version: 1.1.0 - Auto-detect tiles_x, tiles_y, tile_padding from SEGS bboxes
#                  Only mask_blur needs manual input
# License: Dual License (Free for personal use, Commercial license required for business use)


class ArchAi3D_SEGS_To_Bundle:
    """
    Bridge node: Convert SEGS to SMART_TILE_BUNDLE.

    Use this to connect Impact Pack's Make Tile SEGS to Smart Tile Prompter Turbo.

    How it works:
    - Auto-detects grid dimensions (tiles_x, tiles_y) from bbox positions
    - Auto-detects tile size from first SEG's bbox
    - Auto-detects tile_padding from overlap between adjacent tiles
    - Only mask_blur needs to be set manually (match your SEGS Mask Blur setting)
    """

    CATEGORY = "ArchAi3d/Utils"
    RETURN_TYPES = ("SMART_TILE_BUNDLE",)
    RETURN_NAMES = ("bundle",)
    FUNCTION = "convert_segs"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segs": ("SEGS", {
                    "tooltip": "SEGS from Impact Pack's Make Tile SEGS"
                }),
                "image": ("IMAGE", {
                    "tooltip": "The image that SEGS were created from"
                }),
            },
            "optional": {
                "mask_blur": ("INT", {
                    "default": 8,
                    "min": 0,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Mask blur for downstream nodes (match your SEGS Mask Blur setting)"
                }),
            }
        }

    def convert_segs(self, segs, image, mask_blur=8):
        """
        Convert SEGS to SMART_TILE_BUNDLE with auto-detection.

        Args:
            segs: SEGS tuple ((h, w), [list of SEG])
            image: Input image tensor
            mask_blur: Blur value for bundle (match SEGS Mask Blur)

        Returns:
            SMART_TILE_BUNDLE dict
        """
        # Unpack SEGS
        (img_h, img_w), seg_list = segs
        num_segs = len(seg_list)

        if num_segs == 0:
            print("[SEGS To Bundle v1.1] Warning: No segments in SEGS, creating 1x1 bundle")
            return ({
                "scaled_image": image,
                "tile_width": img_w,
                "tile_height": img_h,
                "tiles_x": 1,
                "tiles_y": 1,
                "tile_padding": 0,
                "crop_factor": 1.0,
                "mask_blur": mask_blur,
                "latent_divisor": 32,
            },)

        # Collect all bbox positions to determine grid
        x1_values = set()
        y1_values = set()
        for seg in seg_list:
            x1, y1, x2, y2 = seg.bbox
            x1_values.add(x1)
            y1_values.add(y1)

        # Grid dimensions from unique positions
        tiles_x = len(x1_values)
        tiles_y = len(y1_values)

        # Tile size from first SEG's bbox
        first_bbox = seg_list[0].bbox
        tile_width = first_bbox[2] - first_bbox[0]   # x2 - x1
        tile_height = first_bbox[3] - first_bbox[1]  # y2 - y1

        # Calculate overlap from adjacent tiles
        sorted_x = sorted(x1_values)
        sorted_y = sorted(y1_values)

        h_overlap = 0
        v_overlap = 0

        # Horizontal overlap: find tile at first x position, get its x2, subtract next x1
        if len(sorted_x) > 1:
            for seg in seg_list:
                if seg.bbox[0] == sorted_x[0]:
                    h_overlap = seg.bbox[2] - sorted_x[1]
                    break

        # Vertical overlap: find tile at first y position, get its y2, subtract next y1
        if len(sorted_y) > 1:
            for seg in seg_list:
                if seg.bbox[1] == sorted_y[0]:
                    v_overlap = seg.bbox[3] - sorted_y[1]
                    break

        # Use minimum overlap as tile_padding (more conservative)
        if h_overlap > 0 and v_overlap > 0:
            tile_padding = min(h_overlap, v_overlap)
        else:
            tile_padding = max(h_overlap, v_overlap)

        # Ensure non-negative
        tile_padding = max(0, tile_padding)

        print(f"\n[SEGS To Bundle v1.1] Auto-detected from {num_segs} SEGS:")
        print(f"  Image: {img_w}x{img_h}")
        print(f"  Grid: {tiles_x}x{tiles_y}")
        print(f"  Tile size: {tile_width}x{tile_height}")
        print(f"  Overlap: h={h_overlap}, v={v_overlap} -> padding={tile_padding}")
        print(f"  Manual: mask_blur={mask_blur}")

        # Create bundle compatible with Smart Tile Prompter Turbo
        bundle = {
            "scaled_image": image,
            "tile_width": tile_width,
            "tile_height": tile_height,
            "tiles_x": tiles_x,
            "tiles_y": tiles_y,
            "tile_padding": tile_padding,
            "crop_factor": 1.5,  # Default
            "mask_blur": mask_blur,
            "latent_divisor": 32,
        }

        return (bundle,)


# ============================================================================
# NODE REGISTRATION
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "ArchAi3D_SEGS_To_Bundle": ArchAi3D_SEGS_To_Bundle,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_SEGS_To_Bundle": "🔗 SEGS To Bundle",
}
