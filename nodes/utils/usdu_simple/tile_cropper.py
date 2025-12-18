"""
Simple USDU - Tile Cropper
===========================

Extracts individual tiles from an image with proper padding context.
"""

import json
import torch
from PIL import Image
from .utils import (
    tensor_to_pil, pil_to_tensor, get_tile_position,
    get_padded_rect
)


class ArchAi3D_Simple_Tile_Cropper:
    """
    Crop a single tile from an image.

    The tile is cropped with its full size (which includes overlap).
    Edge tiles may be smaller if they extend past the image boundary.

    Outputs JSON strings with tile geometry for use by the compositor.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_idx": ("INT", {"default": 0, "min": 0, "max": 1000}),
                "tile_width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "tile_height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "tiles_x": ("INT", {"default": 2, "min": 1, "max": 100}),
                "tiles_y": ("INT", {"default": 2, "min": 1, "max": 100}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 512,
                                    "tooltip": "Overlap between tiles (context padding)"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "INT")
    RETURN_NAMES = ("cropped_tile", "tile_info", "crop_width", "crop_height")
    FUNCTION = "crop_tile"
    CATEGORY = "ArchAi3d/Upscaling/Simple USDU"

    def crop_tile(self, image, tile_idx, tile_width, tile_height, tiles_x, tiles_y, overlap):
        """
        Crop a single tile from the image.

        Args:
            image: Input IMAGE tensor [B, H, W, C]
            tile_idx: Which tile to crop (0-indexed)
            tile_width: Width of each tile
            tile_height: Height of each tile
            tiles_x: Number of tiles horizontally
            tiles_y: Number of tiles vertically
            overlap: Overlap between tiles

        Returns:
            cropped_tile: IMAGE tensor of the cropped tile
            tile_info: JSON string with tile geometry
            crop_width: Actual width of cropped tile
            crop_height: Actual height of cropped tile
        """
        batch_size, img_h, img_w, channels = image.shape

        # Get crop rectangle for this tile
        x1, y1, x2, y2 = get_padded_rect(
            tile_idx, tile_width, tile_height, tiles_x, tiles_y, overlap,
            img_w, img_h
        )

        # Crop the tile
        cropped = image[:, y1:y2, x1:x2, :]

        crop_width = x2 - x1
        crop_height = y2 - y1

        # Store tile info as JSON for the compositor
        xi, yi = get_tile_position(tile_idx, tiles_x, tiles_y)
        step_x = tile_width - overlap
        step_y = tile_height - overlap

        tile_info = json.dumps({
            "tile_idx": tile_idx,
            "xi": xi,
            "yi": yi,
            "crop_rect": [x1, y1, x2, y2],  # Where this tile came from
            "tile_width": tile_width,
            "tile_height": tile_height,
            "tiles_x": tiles_x,
            "tiles_y": tiles_y,
            "overlap": overlap,
            "step_x": step_x,
            "step_y": step_y,
        })

        return (cropped, tile_info, crop_width, crop_height)


class ArchAi3D_Simple_Tile_Cropper_Batch:
    """
    Crop ALL tiles from an image at once.

    This is more efficient than calling Simple_Tile_Cropper in a loop.
    Returns a batch of tiles for processing.

    Accepts either individual parameters OR a tile_params bundle from Smart Tile Solver V6.2.
    When tile_params is connected, it overrides the individual INT inputs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
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

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("cropped_tiles", "tiles_info", "total_tiles")
    FUNCTION = "crop_all_tiles"
    CATEGORY = "ArchAi3d/Upscaling/Simple USDU"

    def crop_all_tiles(self, image, tile_params=None, tile_width=512, tile_height=512, tiles_x=2, tiles_y=2, overlap=64):
        """
        Crop all tiles from the image.

        Note: Since tiles may have different sizes at edges, we resize them
        all to the expected tile size for batch processing.

        Args:
            image: Input IMAGE tensor [B, H, W, C] (assumes B=1)
            tile_params: Optional TILE_PARAMS bundle (overrides individual inputs)
            tile_width: Width of each tile
            tile_height: Height of each tile
            tiles_x: Number of tiles horizontally
            tiles_y: Number of tiles vertically
            overlap: Overlap between tiles

        Returns:
            cropped_tiles: IMAGE tensor [total_tiles, H, W, C]
            tiles_info: JSON string with all tile geometries
            total_tiles: Number of tiles
        """
        # Extract from bundle if provided
        if tile_params is not None:
            tile_width = tile_params.get("tile_width", tile_width)
            tile_height = tile_params.get("tile_height", tile_height)
            tiles_x = tile_params.get("tiles_x", tiles_x)
            tiles_y = tile_params.get("tiles_y", tiles_y)
            overlap = tile_params.get("overlap", overlap)

        total_tiles = tiles_x * tiles_y
        batch_size, img_h, img_w, channels = image.shape

        cropper = ArchAi3D_Simple_Tile_Cropper()
        tiles = []
        infos = []

        for tile_idx in range(total_tiles):
            cropped, info, crop_w, crop_h = cropper.crop_tile(
                image, tile_idx, tile_width, tile_height, tiles_x, tiles_y, overlap
            )

            # Resize to expected tile size if needed (edge tiles may be smaller)
            if crop_w != tile_width or crop_h != tile_height:
                # Use PIL for resizing
                pil_img = tensor_to_pil(cropped, 0)
                pil_img = pil_img.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
                cropped = pil_to_tensor(pil_img)

            tiles.append(cropped)
            infos.append(json.loads(info))

        # Stack into batch
        tiles_batch = torch.cat(tiles, dim=0)  # [total_tiles, H, W, C]

        # Combine info
        tiles_info = json.dumps({
            "tiles": infos,
            "tile_width": tile_width,
            "tile_height": tile_height,
            "tiles_x": tiles_x,
            "tiles_y": tiles_y,
            "overlap": overlap,
            "image_width": img_w,
            "image_height": img_h,
        })

        return (tiles_batch, tiles_info, total_tiles)


# Node registration
NODE_CLASS_MAPPINGS = {
    "ArchAi3D_Simple_Tile_Cropper": ArchAi3D_Simple_Tile_Cropper,
    "ArchAi3D_Simple_Tile_Cropper_Batch": ArchAi3D_Simple_Tile_Cropper_Batch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_Simple_Tile_Cropper": "Simple Tile Cropper",
    "ArchAi3D_Simple_Tile_Cropper_Batch": "Simple Tile Cropper (Batch)",
}
