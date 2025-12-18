"""
Simple USDU - Tile Compositor
==============================

Composites processed tiles back into a final image using weighted accumulation.

KEY FEATURE: Proper weighted averaging prevents seam artifacts.
Formula: result[x,y] = sum(tile[x,y] * mask[x,y]) / sum(mask[x,y])
"""

import json
import numpy as np
import torch
from PIL import Image
from .utils import tensor_to_pil, pil_to_tensor, get_tile_position


class ArchAi3D_Simple_Tile_Compositor:
    """
    Composite processed tiles into a final image using weighted accumulation.

    Takes a batch of processed tiles and their blend masks,
    composites them using proper weighted averaging.

    Formula:
        result[x,y] = sum(tile[x,y] * mask[x,y]) / sum(mask[x,y])

    This prevents seam artifacts because overlapping regions get
    properly blended based on the mask weights.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "processed_tiles": ("IMAGE",),  # Batch of processed tiles
                "blend_masks": ("MASK",),  # Batch of blend masks (full canvas size)
                "tiles_info": ("STRING", {"multiline": True}),  # JSON from cropper
            },
            "optional": {
                "original_image": ("IMAGE",),  # Fallback for unprocessed areas
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("final_image", "debug_weights")
    FUNCTION = "composite"
    CATEGORY = "ArchAi3d/Upscaling/Simple USDU"

    def composite(self, processed_tiles, blend_masks, tiles_info, original_image=None):
        """
        Composite all tiles into final image.

        Args:
            processed_tiles: IMAGE tensor [N, H, W, C] - processed tiles
            blend_masks: MASK tensor [N, H, W] - blend masks at canvas size
            tiles_info: JSON string with tile geometry
            original_image: Optional IMAGE tensor to use for unprocessed areas

        Returns:
            final_image: IMAGE tensor [1, H, W, C]
            debug_weights: IMAGE tensor showing weight accumulation
        """
        info = json.loads(tiles_info)
        tiles_x = info["tiles_x"]
        tiles_y = info["tiles_y"]
        tile_width = info["tile_width"]
        tile_height = info["tile_height"]
        overlap = info["overlap"]
        output_w = info.get("image_width")
        output_h = info.get("image_height")

        # Calculate output size if not provided
        if output_w is None or output_h is None:
            step_x = tile_width - overlap
            step_y = tile_height - overlap
            output_w = step_x * (tiles_x - 1) + tile_width
            output_h = step_y * (tiles_y - 1) + tile_height

        total_tiles = tiles_x * tiles_y

        # Initialize accumulators (float32 for precision)
        tile_sum = np.zeros((output_h, output_w, 3), dtype=np.float32)
        weight_sum = np.zeros((output_h, output_w), dtype=np.float32)

        # Get step size
        step_x = tile_width - overlap
        step_y = tile_height - overlap

        print(f"[Compositor] Compositing {total_tiles} tiles into {output_w}x{output_h} image")

        for tile_idx in range(total_tiles):
            xi, yi = get_tile_position(tile_idx, tiles_x, tiles_y)

            # Get tile position
            x1 = xi * step_x
            y1 = yi * step_y
            x2 = min(x1 + tile_width, output_w)
            y2 = min(y1 + tile_height, output_h)

            actual_w = x2 - x1
            actual_h = y2 - y1

            # Get processed tile (resize if needed)
            tile_tensor = processed_tiles[tile_idx:tile_idx+1]  # [1, H, W, C]
            tile_pil = tensor_to_pil(tile_tensor, 0)

            if tile_pil.width != actual_w or tile_pil.height != actual_h:
                tile_pil = tile_pil.resize((actual_w, actual_h), Image.Resampling.LANCZOS)

            tile_np = np.array(tile_pil).astype(np.float32) / 255.0

            # Get blend mask for this tile
            # Masks are at canvas size - extract the relevant region
            if len(blend_masks.shape) == 3:  # [N, H, W]
                mask_tensor = blend_masks[tile_idx]  # [H, W]
            else:  # [H, W]
                mask_tensor = blend_masks

            mask_np = mask_tensor.cpu().numpy().astype(np.float32)

            # Extract mask region for this tile
            mask_region = mask_np[y1:y2, x1:x2]

            # Accumulate weighted contributions
            tile_sum[y1:y2, x1:x2] += tile_np * mask_region[..., np.newaxis]
            weight_sum[y1:y2, x1:x2] += mask_region

        # Compute final result using weighted average
        # Avoid division by zero
        weight_safe = np.maximum(weight_sum, 1e-6)
        result = tile_sum / weight_safe[..., np.newaxis]

        # For areas with zero weight, use original image if provided
        if original_image is not None:
            original_pil = tensor_to_pil(original_image, 0)
            if original_pil.width != output_w or original_pil.height != output_h:
                original_pil = original_pil.resize((output_w, output_h), Image.Resampling.LANCZOS)
            original_np = np.array(original_pil).astype(np.float32) / 255.0

            # Where weight is zero, use original
            zero_weight_mask = weight_sum < 1e-6
            result[zero_weight_mask] = original_np[zero_weight_mask]

        # Clamp result to valid range
        result = np.clip(result, 0.0, 1.0)

        # Convert to tensor
        result_pil = Image.fromarray((result * 255).astype(np.uint8))
        result_tensor = pil_to_tensor(result_pil)

        # Create debug visualization of weights
        weight_viz = np.clip(weight_sum / weight_sum.max() if weight_sum.max() > 0 else weight_sum, 0, 1)
        weight_pil = Image.fromarray((weight_viz * 255).astype(np.uint8), mode='L').convert('RGB')
        weight_tensor = pil_to_tensor(weight_pil)

        print(f"[Compositor] Done. Weight range: {weight_sum.min():.3f} - {weight_sum.max():.3f}")

        return (result_tensor, weight_tensor)


class ArchAi3D_Simple_Tile_Compositor_Single:
    """
    Composite tiles ONE AT A TIME into an accumulator.

    Use this when processing tiles in a loop rather than as a batch.
    Connect the output back to the input for the next iteration.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "processed_tile": ("IMAGE",),  # Single processed tile [1, H, W, C]
                "blend_mask": ("MASK",),  # Single blend mask at canvas size
                "tile_idx": ("INT", {"default": 0, "min": 0, "max": 1000}),
                "tile_width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "tile_height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "tiles_x": ("INT", {"default": 2, "min": 1, "max": 100}),
                "tiles_y": ("INT", {"default": 2, "min": 1, "max": 100}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 512}),
            },
            "optional": {
                "accumulator": ("IMAGE",),  # Previous result to accumulate into
                "weight_accumulator": ("MASK",),  # Previous weight accumulator
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("accumulator", "weight_accumulator", "current_result")
    FUNCTION = "accumulate"
    CATEGORY = "ArchAi3d/Upscaling/Simple USDU"

    def accumulate(self, processed_tile, blend_mask, tile_idx, tile_width, tile_height,
                   tiles_x, tiles_y, overlap, accumulator=None, weight_accumulator=None):
        """
        Add one tile to the accumulator.

        This allows iterative compositing - call once per tile,
        connecting outputs to inputs for the next iteration.
        """
        # Calculate output size
        step_x = tile_width - overlap
        step_y = tile_height - overlap
        output_w = step_x * (tiles_x - 1) + tile_width
        output_h = step_y * (tiles_y - 1) + tile_height

        # Initialize or use existing accumulators
        if accumulator is None:
            tile_sum = np.zeros((output_h, output_w, 3), dtype=np.float32)
        else:
            tile_sum = accumulator[0].cpu().numpy().astype(np.float32)

        if weight_accumulator is None:
            weight_sum = np.zeros((output_h, output_w), dtype=np.float32)
        else:
            weight_sum = weight_accumulator.cpu().numpy().astype(np.float32)

        # Get tile position
        xi, yi = get_tile_position(tile_idx, tiles_x, tiles_y)
        x1 = xi * step_x
        y1 = yi * step_y
        x2 = min(x1 + tile_width, output_w)
        y2 = min(y1 + tile_height, output_h)

        actual_w = x2 - x1
        actual_h = y2 - y1

        # Get tile and resize if needed
        tile_pil = tensor_to_pil(processed_tile, 0)
        if tile_pil.width != actual_w or tile_pil.height != actual_h:
            tile_pil = tile_pil.resize((actual_w, actual_h), Image.Resampling.LANCZOS)
        tile_np = np.array(tile_pil).astype(np.float32) / 255.0

        # Get mask region
        mask_np = blend_mask.cpu().numpy().astype(np.float32)
        if len(mask_np.shape) == 3:
            mask_np = mask_np[0]  # [H, W]
        mask_region = mask_np[y1:y2, x1:x2]

        # Accumulate
        tile_sum[y1:y2, x1:x2] += tile_np * mask_region[..., np.newaxis]
        weight_sum[y1:y2, x1:x2] += mask_region

        # Compute current result
        weight_safe = np.maximum(weight_sum, 1e-6)
        result = tile_sum / weight_safe[..., np.newaxis]
        result = np.clip(result, 0.0, 1.0)

        # Convert to tensors
        # Store tile_sum as accumulator (not normalized result)
        acc_tensor = torch.from_numpy(tile_sum).unsqueeze(0)  # [1, H, W, 3]
        weight_tensor = torch.from_numpy(weight_sum)  # [H, W]

        result_pil = Image.fromarray((result * 255).astype(np.uint8))
        result_tensor = pil_to_tensor(result_pil)

        return (acc_tensor, weight_tensor, result_tensor)


# Node registration
NODE_CLASS_MAPPINGS = {
    "ArchAi3D_Simple_Tile_Compositor": ArchAi3D_Simple_Tile_Compositor,
    "ArchAi3D_Simple_Tile_Compositor_Single": ArchAi3D_Simple_Tile_Compositor_Single,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_Simple_Tile_Compositor": "Simple Tile Compositor",
    "ArchAi3D_Simple_Tile_Compositor_Single": "Simple Tile Compositor (Single)",
}
