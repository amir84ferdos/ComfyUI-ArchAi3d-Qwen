# -*- coding: utf-8 -*-
"""
ArchAi3D Mask Uncrop (Stitch-Back) Node
Author: Amir Ferdos (ArchAi3d)
Email: Amir84ferdos@gmail.com
LinkedIn: https://www.linkedin.com/in/archai3d/

Description:
    Companion to ArchAi3D_Mask_Crop_Rotate. Takes a processed cropped region
    and stitches it back into the original image at the correct position.
    Handles reverse rotation and optional feathered blending.

Usage:
    1. Use Mask Crop & Rotate to extract a region
    2. Process the crop (inpaint, upscale, etc.)
    3. Use this node to paste the result back into the original image

Version: 1.0.0
Created: 2025-10-18
"""

import numpy as np
import torch
from PIL import Image, ImageFilter
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

# Reuse helpers from the crop node
from .archai3d_mask_crop_rotate import image_to_pil, pil_to_tensor, mask_to_pil


# ============================================================================
# NODE DEFINITION
# ============================================================================

class ArchAi3D_Mask_Uncrop(io.ComfyNode):
    """Mask Uncrop: Stitch a processed crop back into the original image.

    Companion node to Mask Crop & Rotate. Reverses the crop (and optional
    rotation) to paste the processed region back at its original position.

    Workflow:
    1. Parse crop_bbox to get paste position
    2. Reverse rotation if angle was applied
    3. Resize crop to match bbox dimensions
    4. Paste into original with optional mask feathering
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ArchAi3D_Mask_Uncrop",
            category="ArchAi3d/Utils",
            inputs=[
                io.Image.Input(
                    "original_image",
                    tooltip="The full original image (before cropping)"
                ),
                io.Image.Input(
                    "processed_crop",
                    tooltip="The processed cropped region to paste back"
                ),
                io.String.Input(
                    "crop_bbox",
                    default="[0, 0, 512, 512]",
                    tooltip="Bounding box from Mask Crop & Rotate: [x1, y1, x2, y2]"
                ),
                io.Float.Input(
                    "rotation_angle",
                    default=0.0,
                    min=-360.0,
                    max=360.0,
                    step=1.0,
                    tooltip="Same rotation angle used in Mask Crop & Rotate (will be reversed)"
                ),
                io.Combo.Input(
                    "expand_canvas",
                    options=["yes", "no"],
                    default="yes",
                    tooltip="Must match the expand_canvas setting used in Mask Crop & Rotate"
                ),
                io.Int.Input(
                    "blend_feather",
                    default=8,
                    min=0,
                    max=100,
                    step=1,
                    tooltip="Feather radius for edge blending (0 = hard paste, higher = smoother)"
                ),
                io.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip="Optional mask for feathered blending (from the original crop operation)"
                ),
            ],
            outputs=[
                io.Image.Output(
                    "stitched_image",
                    tooltip="Original image with processed crop pasted back"
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        original_image,
        processed_crop,
        crop_bbox,
        rotation_angle,
        expand_canvas,
        blend_feather,
        mask=None,
    ) -> io.NodeOutput:
        """Execute the uncrop/stitch-back operation."""

        # Step 1: Parse bbox string
        try:
            bbox_str = crop_bbox.strip().strip('[]')
            parts = [int(x.strip()) for x in bbox_str.split(',')]
            x1, y1, x2, y2 = parts
        except (ValueError, IndexError):
            print(f"Warning: Could not parse crop_bbox '{crop_bbox}'. Returning original image.")
            return io.NodeOutput(original_image)

        crop_width = x2 - x1
        crop_height = y2 - y1

        if crop_width <= 0 or crop_height <= 0:
            print(f"Warning: Invalid bbox dimensions {crop_width}x{crop_height}. Returning original.")
            return io.NodeOutput(original_image)

        expand = (expand_canvas == "yes")
        orig_batch = original_image.shape[0]
        crop_batch = processed_crop.shape[0]
        batch_size = max(orig_batch, crop_batch)

        # Prepare mask-based feather (shared across batch)
        paste_mask = None
        if mask is not None and blend_feather > 0:
            mask_pil = mask_to_pil(mask)
            mask_crop = mask_pil.crop((x1, y1, x2, y2))
            if mask_crop.size != (crop_width, crop_height):
                mask_crop = mask_crop.resize((crop_width, crop_height), Image.LANCZOS)
            paste_mask = mask_crop.filter(
                ImageFilter.GaussianBlur(radius=blend_feather)
            )
        elif blend_feather > 0 and mask is None:
            # Build rectangular feather mask once
            feather_np = np.full((crop_height, crop_width), 255.0, dtype=np.float32)
            for i in range(blend_feather):
                alpha = int(255 * (i + 1) / blend_feather)
                if i < crop_height:
                    feather_np[i, :] = np.minimum(feather_np[i, :], alpha)
                if crop_height - 1 - i >= 0:
                    feather_np[crop_height - 1 - i, :] = np.minimum(
                        feather_np[crop_height - 1 - i, :], alpha
                    )
                if i < crop_width:
                    feather_np[:, i] = np.minimum(feather_np[:, i], alpha)
                if crop_width - 1 - i >= 0:
                    feather_np[:, crop_width - 1 - i] = np.minimum(
                        feather_np[:, crop_width - 1 - i], alpha
                    )
            paste_mask = Image.fromarray(feather_np.astype(np.uint8), mode='L')

        # Step 2: Process each image in the batch
        result_tensors = []
        for b in range(batch_size):
            # Get original image (clamp index for mismatched batch sizes)
            orig_idx = min(b, orig_batch - 1)
            orig_np = (original_image[orig_idx].cpu().numpy() * 255).astype(np.uint8)
            orig_pil = Image.fromarray(orig_np, mode='RGB')

            # Get processed crop
            crop_idx = min(b, crop_batch - 1)
            crop_np = (processed_crop[crop_idx].cpu().numpy() * 255).astype(np.uint8)
            crop_pil = Image.fromarray(crop_np, mode='RGB')

            # Reverse rotation if needed
            if rotation_angle != 0:
                crop_pil = crop_pil.rotate(
                    -rotation_angle, expand=expand,
                    fillcolor=(0, 0, 0), resample=Image.BICUBIC
                )
                cw, ch = crop_pil.size
                if cw != crop_width or ch != crop_height:
                    left = (cw - crop_width) // 2
                    top = (ch - crop_height) // 2
                    crop_pil = crop_pil.crop((
                        left, top, left + crop_width, top + crop_height
                    ))

            # Resize to exact bbox dimensions if needed
            if crop_pil.size != (crop_width, crop_height):
                crop_pil = crop_pil.resize((crop_width, crop_height), Image.LANCZOS)

            # Paste into original
            result_pil = orig_pil.copy()
            if paste_mask is not None:
                result_pil.paste(crop_pil, (x1, y1), paste_mask)
            else:
                result_pil.paste(crop_pil, (x1, y1))

            result_tensors.append(pil_to_tensor(result_pil))

        # Stack batch: (B, H, W, 3)
        output_tensor = torch.cat(result_tensors, dim=0)

        # Debug output
        print("\n" + "=" * 70)
        print("🔙 ArchAi3D Mask Uncrop - v1.1.0")
        print("=" * 70)
        print(f"Batch size: {batch_size}")
        print(f"Original image size: {original_image.shape[2]}x{original_image.shape[1]}")
        print(f"Crop bbox: [{x1}, {y1}, {x2}, {y2}]")
        print(f"Crop region size: {crop_width}x{crop_height}")
        print(f"Rotation angle (reversed): {rotation_angle}")
        print(f"Blend feather: {blend_feather}px")
        print(f"Mask provided: {'Yes' if mask is not None else 'No'}")
        print(f"Output shape: {output_tensor.shape}")
        print("=" * 70 + "\n")

        return io.NodeOutput(output_tensor)


# ============================================================================
# EXTENSION REGISTRATION
# ============================================================================

class MaskUncropExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [ArchAi3D_Mask_Uncrop]


async def comfy_entrypoint():
    return MaskUncropExtension()
