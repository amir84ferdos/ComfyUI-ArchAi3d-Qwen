"""
ArchAi3D Qwen Image Stitch
Version: 1.1.0

Re-composites a processed cropped region back into the original image.
Works with STITCH_DATA from Qwen Image Scale V2.

Features:
- Optional mask input for precise blending control
- Seamless blending with feather/gaussian/hard/mask_only modes
- Automatic scaling back to original crop size
- Preserves original image outside processed region

Author: Amir Ferdos (ArchAi3d)
Email: Amir84ferdos@gmail.com
LinkedIn: https://www.linkedin.com/in/archai3d/
GitHub: https://github.com/amir84ferdos
Category: ArchAi3d/Qwen
License: MIT
"""

import torch
import numpy as np
import comfy.utils
from typing import Tuple, Dict, Any, Optional
from scipy.ndimage import gaussian_filter


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_feather_mask(height: int, width: int, feather_pixels: int) -> torch.Tensor:
    """
    Create a feathered blend mask with smooth falloff at edges.

    Args:
        height: Mask height
        width: Mask width
        feather_pixels: Width of feather zone at edges

    Returns:
        Tensor of shape (H, W) with values 0-1
    """
    if feather_pixels <= 0:
        return torch.ones((height, width), dtype=torch.float32)

    mask = torch.ones((height, width), dtype=torch.float32)

    # Create linear ramps for each edge
    for i in range(min(feather_pixels, height // 2)):
        alpha = (i + 1) / feather_pixels
        mask[i, :] *= alpha          # Top edge
        mask[-(i + 1), :] *= alpha   # Bottom edge

    for i in range(min(feather_pixels, width // 2)):
        alpha = (i + 1) / feather_pixels
        mask[:, i] *= alpha          # Left edge
        mask[:, -(i + 1)] *= alpha   # Right edge

    return mask


def create_gaussian_mask(height: int, width: int, feather_pixels: int) -> torch.Tensor:
    """
    Create a gaussian-blurred blend mask.

    Args:
        height: Mask height
        width: Mask width
        feather_pixels: Sigma for gaussian blur

    Returns:
        Tensor of shape (H, W) with values 0-1
    """
    if feather_pixels <= 0:
        return torch.ones((height, width), dtype=torch.float32)

    # Create hard edge mask
    mask = np.ones((height, width), dtype=np.float32)

    # Set edges to 0
    edge_width = max(1, feather_pixels // 2)
    mask[:edge_width, :] = 0
    mask[-edge_width:, :] = 0
    mask[:, :edge_width] = 0
    mask[:, -edge_width:] = 0

    # Apply gaussian blur
    sigma = feather_pixels / 3.0
    mask = gaussian_filter(mask, sigma=sigma)

    # Normalize to 0-1
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

    return torch.from_numpy(mask)


# ============================================================================
# MAIN NODE CLASS
# ============================================================================

class ArchAi3D_Qwen_Image_Stitch:
    """
    Re-composites a processed cropped region back into the original image.

    Takes STITCH_DATA from Qwen Image Scale V2 and seamlessly blends
    the processed region back into the original image.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "processed_image": ("IMAGE", {
                    "tooltip": "The processed/inpainted cropped region from your pipeline."
                }),
                "stitch_data": ("STITCH_DATA", {
                    "tooltip": "Stitch data from Qwen Image Scale V2 containing original image and bbox info."
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Optional mask for blending. White=use processed, Black=use original. Will be scaled to crop size. If provided, overrides blend_mode feathering."
                }),
                "blend_override": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 128,
                    "step": 1,
                    "tooltip": "Override blend pixels (-1 = use value from stitch_data). Only used when no mask is provided."
                }),
                "blend_mode": (["feather", "gaussian", "hard", "mask_only"], {
                    "default": "feather",
                    "tooltip": "Blend mode: feather/gaussian/hard (edge blending), mask_only (use mask directly without edge feather)."
                }),
                "debug": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show debug info."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "stitch"
    CATEGORY = "ArchAi3d/Qwen"

    def stitch(
        self,
        processed_image: torch.Tensor,
        stitch_data: Dict[str, Any],
        mask: Optional[torch.Tensor] = None,
        blend_override: int = -1,
        blend_mode: str = "feather",
        debug: bool = True
    ) -> Tuple[torch.Tensor]:
        """
        Stitch processed region back into original image.

        Args:
            processed_image: Processed cropped region (B, H, W, C)
            stitch_data: Dict containing original_image, crop_bbox, etc.
            mask: Optional mask for blending (white=processed, black=original)
            blend_override: Override blend pixels (-1 = use stitch_data value)
            blend_mode: feather, gaussian, hard, or mask_only
            debug: Print debug info

        Returns:
            Final composited image
        """

        # Handle None stitch_data (V2 was in disabled mode)
        if stitch_data is None:
            if debug:
                print("🧵 Stitch: No stitch_data (crop mode was disabled). Returning processed image as-is.")
            return (processed_image,)

        # Extract stitch data
        original_image = stitch_data["original_image"]
        crop_bbox = stitch_data["crop_bbox"]
        latent_size = stitch_data["latent_size"]
        blend_pixels = stitch_data["blend_pixels"]
        crop_mask = stitch_data.get("crop_mask", None)

        # Override blend pixels if specified
        if blend_override >= 0:
            blend_pixels = blend_override

        x1, y1, x2, y2 = crop_bbox
        crop_width = x2 - x1
        crop_height = y2 - y1

        # Get batch size
        batch_size = processed_image.shape[0]

        # =========================================================================
        # STEP 1: Resize processed image back to crop bbox size
        # =========================================================================

        # Convert to BCHW for scaling
        processed_bchw = processed_image.movedim(-1, 1).contiguous()

        # Scale back to original crop size using bilinear with align_corners=True
        # This ensures pixel-perfect alignment without shift
        resized_bchw = torch.nn.functional.interpolate(
            processed_bchw,
            size=(crop_height, crop_width),
            mode='bilinear',
            align_corners=True  # Critical: prevents pixel shift
        )

        # Convert back to BHWC
        resized_processed = resized_bchw.movedim(1, -1).contiguous()

        # =========================================================================
        # STEP 2: Create blend mask
        # =========================================================================

        mask_source = "generated"

        if mask is not None:
            # Use provided mask - scale to crop size
            mask_source = "input mask"

            # Ensure proper shape for interpolate
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)  # (H, W) -> (1, H, W)

            # Scale mask to crop size
            mask_for_scale = mask.unsqueeze(1)  # (B, 1, H, W)
            scaled_mask = torch.nn.functional.interpolate(
                mask_for_scale,
                size=(crop_height, crop_width),
                mode='bilinear',
                align_corners=True  # Match image scaling for pixel-perfect alignment
            )
            blend_mask = scaled_mask.squeeze(1)  # (B, H, W)

            # If not mask_only mode, multiply with edge feathering
            if blend_mode != "mask_only":
                if blend_mode == "gaussian":
                    edge_mask = create_gaussian_mask(crop_height, crop_width, blend_pixels)
                elif blend_mode == "hard":
                    edge_mask = torch.ones((crop_height, crop_width), dtype=torch.float32)
                else:  # feather
                    edge_mask = create_feather_mask(crop_height, crop_width, blend_pixels)

                edge_mask = edge_mask.to(processed_image.device)
                # Combine: input mask * edge feathering
                blend_mask = blend_mask * edge_mask.unsqueeze(0)
                mask_source = f"input mask + {blend_mode} edges"

            # Move to device and expand for broadcasting
            blend_mask = blend_mask.to(processed_image.device)
            blend_mask_expanded = blend_mask.unsqueeze(-1).expand(-1, -1, -1, 1)

        else:
            # Generate mask from blend mode
            if blend_mode == "hard" or blend_mode == "mask_only":
                blend_mask = torch.ones((crop_height, crop_width), dtype=torch.float32)
            elif blend_mode == "gaussian":
                blend_mask = create_gaussian_mask(crop_height, crop_width, blend_pixels)
            else:  # feather
                blend_mask = create_feather_mask(crop_height, crop_width, blend_pixels)

            # Move to same device as images
            blend_mask = blend_mask.to(processed_image.device)

            # Expand mask for broadcasting (H, W) -> (B, H, W, 1)
            blend_mask_expanded = blend_mask.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, 1)

        # =========================================================================
        # STEP 3: Composite into original image
        # =========================================================================

        # Clone original to avoid modifying it
        result = original_image.clone()

        # Ensure result has same number of channels as processed
        if result.shape[-1] != resized_processed.shape[-1]:
            if result.shape[-1] == 4 and resized_processed.shape[-1] == 3:
                # Add alpha channel to processed
                alpha = torch.ones((*resized_processed.shape[:-1], 1),
                                  device=resized_processed.device, dtype=resized_processed.dtype)
                resized_processed = torch.cat([resized_processed, alpha], dim=-1)
            elif result.shape[-1] == 3 and resized_processed.shape[-1] == 4:
                # Remove alpha from processed
                resized_processed = resized_processed[:, :, :, :3]

        # Extract the crop region from original
        original_crop = result[:, y1:y2, x1:x2, :].clone()

        # Blend: result = original * (1 - mask) + processed * mask
        blended_crop = original_crop * (1 - blend_mask_expanded) + resized_processed * blend_mask_expanded

        # Place blended region back
        result[:, y1:y2, x1:x2, :] = blended_crop

        # =========================================================================
        # DEBUG INFO
        # =========================================================================

        if debug:
            orig_h, orig_w = original_image.shape[1], original_image.shape[2]
            proc_h, proc_w = processed_image.shape[1], processed_image.shape[2]

            print(f"""
🧵 Qwen Image Stitch Debug Info
━━━━━━━━━━━━━━━━━━━━━━━━━━━
📥 INPUT:
   Original image: {orig_w}x{orig_h}
   Processed image: {proc_w}x{proc_h}
   Crop bbox: ({x1}, {y1}, {x2}, {y2})
   Crop size: {crop_width}x{crop_height}
   Mask input: {"✅ Provided" if mask is not None else "❌ None"}

🔧 PROCESSING:
   Resized processed: {proc_w}x{proc_h} → {crop_width}x{crop_height}
   Blend mode: {blend_mode}
   Blend pixels: {blend_pixels}
   Mask source: {mask_source}

📤 OUTPUT:
   Final image: {orig_w}x{orig_h}
   Stitched region: ({x1}, {y1}) to ({x2}, {y2})
━━━━━━━━━━━━━━━━━━━━━━━━━━━""")

        return (result,)


# ============================================================================
# NODE REGISTRATION
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "ArchAi3D_Qwen_Image_Stitch": ArchAi3D_Qwen_Image_Stitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_Qwen_Image_Stitch": "🧵 Qwen Image Stitch",
}
