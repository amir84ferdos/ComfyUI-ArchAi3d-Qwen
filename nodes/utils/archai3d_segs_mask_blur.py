# ArchAi3D SEGS Mask Blur
#
# Apply Gaussian blur to SEGS masks for soft feathering
# Use before DetailerForEach or Smart Tile Detailer for seamless blending
#
# Author: Amir Ferdos (ArchAi3d)
# Version: 1.0.0
# License: Dual License (Free for personal use, Commercial license required for business use)

import numpy as np
import torch
from collections import namedtuple
from PIL import Image, ImageFilter

# Define SEG namedtuple (compatible with Impact Pack)
SEG = namedtuple("SEG",
    ['cropped_image', 'cropped_mask', 'confidence', 'crop_region', 'bbox', 'label', 'control_net_wrapper'],
    defaults=[None]
)


class ArchAi3D_SEGS_Mask_Blur:
    """
    Apply Gaussian blur to SEGS masks for soft edge feathering.

    This creates smooth transitions at tile/segment edges when compositing.
    Use this before DetailerForEach or any node that composites SEGS back to image.

    Based on Ultimate SD Upscale's mask_blur approach.
    """

    CATEGORY = "ArchAi3d/Utils"
    RETURN_TYPES = ("SEGS",)
    RETURN_NAMES = ("segs",)
    FUNCTION = "blur_masks"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segs": ("SEGS", {
                    "tooltip": "SEGS to apply mask blur to"
                }),
                "mask_blur": ("INT", {
                    "default": 8,
                    "min": 0,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Gaussian blur radius for mask edges (0=sharp, 8=soft, 32=very soft)"
                }),
            },
            "optional": {
                "bundle": ("SMART_TILE_BUNDLE", {
                    "tooltip": "If connected, uses mask_blur from bundle"
                }),
            }
        }

    def blur_masks(self, segs, mask_blur, bundle=None):
        """
        Apply Gaussian blur to each SEG's mask.

        Args:
            segs: SEGS tuple ((h, w), [list of SEG])
            mask_blur: Blur radius (pixels)
            bundle: Optional bundle to get mask_blur from

        Returns:
            Modified SEGS with blurred masks
        """
        # Extract mask_blur from bundle if provided
        if bundle is not None:
            mask_blur = bundle.get("mask_blur", mask_blur)
            print(f"[SEGS Mask Blur v1.0] Using bundle mask_blur: {mask_blur}")

        # Unpack SEGS
        segs_header, seg_list = segs

        if mask_blur == 0:
            print(f"[SEGS Mask Blur v1.0] mask_blur=0, passing through unchanged")
            return (segs,)

        print(f"\n[SEGS Mask Blur v1.0] Applying blur={mask_blur} to {len(seg_list)} segments")

        new_segs = []
        for i, seg in enumerate(seg_list):
            # Get the mask (numpy array or tensor)
            mask = seg.cropped_mask

            # Convert to PIL Image for blur
            if isinstance(mask, torch.Tensor):
                mask_np = mask.cpu().numpy()
            else:
                mask_np = mask

            # Ensure it's 2D
            if mask_np.ndim == 3:
                mask_np = mask_np.squeeze()

            # Convert to uint8 for PIL
            mask_uint8 = (mask_np * 255).astype(np.uint8)
            mask_pil = Image.fromarray(mask_uint8, mode='L')

            # Apply Gaussian blur
            blurred_pil = mask_pil.filter(ImageFilter.GaussianBlur(mask_blur))

            # Convert back to numpy float
            blurred_np = np.array(blurred_pil).astype(np.float32) / 255.0

            # Create new SEG with blurred mask
            new_seg = SEG(
                cropped_image=seg.cropped_image,
                cropped_mask=blurred_np,
                confidence=seg.confidence,
                crop_region=seg.crop_region,
                bbox=seg.bbox,
                label=seg.label,
                control_net_wrapper=seg.control_net_wrapper
            )
            new_segs.append(new_seg)

        result = (segs_header, new_segs)
        print(f"[SEGS Mask Blur v1.0] Done! Blurred {len(new_segs)} masks with radius {mask_blur}")

        return (result,)


# ============================================================================
# NODE REGISTRATION
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "ArchAi3D_SEGS_Mask_Blur": ArchAi3D_SEGS_Mask_Blur,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_SEGS_Mask_Blur": "🌫️ SEGS Mask Blur",
}
