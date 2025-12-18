"""
ArchAi3D Qwen Image Scale V2
Version: 2.0.1

Intelligent image scaling for QwenVL with preferred aspect ratios + mask-based cropping.
- All V1 features: Aspect selection, VL/Latent scaling, pixel-perfect alignment
- NEW: Mask-based crop mode with padding and context factor
- NEW: Outputs STITCH_DATA for seamless re-compositing
- FIX: Forces stretch mode in mask_crop to prevent pixel shift when stitching

Author: Amir Ferdos (ArchAi3d)
Email: Amir84ferdos@gmail.com
LinkedIn: https://www.linkedin.com/in/archai3d/
GitHub: https://github.com/amir84ferdos
Category: ArchAi3d/Qwen
License: MIT
"""

import torch
import math
import numpy as np
import comfy.utils
from PIL import Image
from typing import Tuple, Optional, Dict, Any


# ============================================================================
# PREFERRED ASPECT RATIOS (Optimized for QwenVL)
# ============================================================================

PREFERRED_ASPECT_RATIOS = {
    "1:1 (Perfect Square)": (1, 1),
    "2:1 (Panorama 360)": (2, 1),
    "2:3 (Classic Portrait)": (2, 3),
    "3:4 (Golden Ratio)": (3, 4),
    "3:5 (Elegant Vertical)": (3, 5),
    "4:5 (Artistic Frame)": (4, 5),
    "5:7 (Balanced Portrait)": (5, 7),
    "5:8 (Tall Portrait)": (5, 8),
    "7:9 (Modern Portrait)": (7, 9),
    "9:16 (Slim Vertical)": (9, 16),
    "9:19 (Tall Slim)": (9, 19),
    "9:21 (Ultra Tall)": (9, 21),
    "9:32 (Skyline)": (9, 32),
    "3:2 (Golden Landscape)": (3, 2),
    "4:3 (Classic Landscape)": (4, 3),
    "5:3 (Wide Horizon)": (5, 3),
    "5:4 (Balanced Frame)": (5, 4),
    "7:5 (Elegant Landscape)": (7, 5),
    "8:5 (Cinematic View)": (8, 5),
    "9:7 (Artful Horizon)": (9, 7),
    "16:9 (Panorama)": (16, 9),
    "19:9 (Cinematic Ultrawide)": (19, 9),
    "21:9 (Epic Ultrawide)": (21, 9),
    "32:9 (Extreme Ultrawide)": (32, 9),
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def round_to_multiple(value: float, multiple: int) -> int:
    """Round a value to the nearest multiple of a given number."""
    return int(round(value / multiple) * multiple)


def find_closest_preferred_aspect_ratio(
    input_aspect: float,
    preferred_ratios: dict = PREFERRED_ASPECT_RATIOS
) -> Tuple[int, int, str]:
    """Find the closest preferred aspect ratio to the input aspect ratio."""
    best_match = None
    best_diff = float('inf')
    best_name = ""

    for name, (w_ratio, h_ratio) in preferred_ratios.items():
        ratio_aspect = w_ratio / h_ratio
        diff = abs(ratio_aspect - input_aspect)

        if diff < best_diff:
            best_diff = diff
            best_match = (w_ratio, h_ratio)
            best_name = name

    return best_match[0], best_match[1], best_name


def find_optimal_dimensions_preferred_aspect(
    preferred_ratio: Tuple[int, int],
    target_area: int = 147456,
    divisible_by: int = 32
) -> Tuple[int, int]:
    """Calculate dimensions using a preferred aspect ratio."""
    w_ratio, h_ratio = preferred_ratio
    aspect_ratio = w_ratio / h_ratio

    ideal_height = math.sqrt(target_area / aspect_ratio)
    ideal_width = ideal_height * aspect_ratio

    best_width, best_height = None, None
    best_error = float('inf')

    for height_mult in range(
        max(1, int(ideal_height / divisible_by) - 3),
        int(ideal_height / divisible_by) + 4
    ):
        height_test = height_mult * divisible_by
        if height_test < divisible_by:
            continue

        width_exact = height_test * aspect_ratio
        width_test = round_to_multiple(width_exact, divisible_by)
        width_test = max(divisible_by, width_test)

        test_aspect = width_test / height_test
        aspect_error = abs(test_aspect - aspect_ratio)
        if aspect_error / aspect_ratio > 0.01:
            continue

        test_area = width_test * height_test
        area_error = abs(test_area - target_area)
        total_error = area_error + (aspect_error * 100000000)

        if total_error < best_error:
            best_error = total_error
            best_width = width_test
            best_height = height_test

    if best_width is None:
        best_height = round_to_multiple(ideal_height, divisible_by)
        best_height = max(divisible_by, best_height)
        best_width = round_to_multiple(ideal_height * aspect_ratio, divisible_by)
        best_width = max(divisible_by, best_width)

    return best_width, best_height


def find_optimal_dimensions_latent_aspect_lock(
    vl_width: int,
    vl_height: int,
    target_area: int = 1048576,
    tolerance: float = 0.3,
    multiple: int = 32
) -> Tuple[int, int]:
    """Find latent dimensions that EXACTLY match VL aspect ratio."""
    aspect_ratio = vl_width / vl_height

    area_min = int(target_area * (1 - tolerance))
    area_max = int(target_area * (1 + tolerance))

    ideal_height = math.sqrt(target_area / aspect_ratio)
    ideal_width = ideal_height * aspect_ratio

    best_width, best_height = None, None
    best_error = float('inf')

    height_min = int(math.sqrt(area_min / aspect_ratio))
    height_max = int(math.sqrt(area_max / aspect_ratio))
    height_min = max(multiple, height_min)
    height_max = max(height_min + multiple * 10, height_max)

    for height_test in range(
        round_to_multiple(height_min, multiple),
        round_to_multiple(height_max, multiple) + multiple,
        multiple
    ):
        if height_test < multiple:
            continue

        width_exact = height_test * aspect_ratio
        width_test = round_to_multiple(width_exact, multiple)
        width_test = max(multiple, width_test)

        test_area = width_test * height_test

        if area_min <= test_area <= area_max:
            area_error = abs(test_area - target_area)
            test_aspect = width_test / height_test
            aspect_error = abs(test_aspect - aspect_ratio) * 10000000
            total_error = area_error + aspect_error

            if total_error < best_error:
                best_error = total_error
                best_width = width_test
                best_height = height_test

    if best_width is None:
        best_height = round_to_multiple(ideal_height, multiple)
        best_height = max(multiple, best_height)
        best_width = round_to_multiple(best_height * aspect_ratio, multiple)
        best_width = max(multiple, best_width)

    return best_width, best_height


def mask_to_pil(mask_tensor: torch.Tensor) -> Image.Image:
    """Convert ComfyUI mask tensor to PIL Image."""
    if len(mask_tensor.shape) == 3:
        mask_tensor = mask_tensor[0]
    mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(mask_np, mode='L')


def get_mask_bbox(mask_pil: Image.Image, padding: int = 0) -> Optional[Tuple[int, int, int, int]]:
    """Get bounding box of mask region using PIL's getbbox."""
    bbox = mask_pil.getbbox()
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    width, height = mask_pil.size

    # Apply padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)

    return (x1, y1, x2, y2)


def expand_bbox_by_factor(
    bbox: Tuple[int, int, int, int],
    factor: float,
    img_width: int,
    img_height: int
) -> Tuple[int, int, int, int]:
    """Expand bounding box by a context factor (e.g., 1.5 = 50% larger)."""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    # Calculate expansion
    expand_w = int(w * (factor - 1) / 2)
    expand_h = int(h * (factor - 1) / 2)

    # Apply expansion and clamp to image bounds
    new_x1 = max(0, x1 - expand_w)
    new_y1 = max(0, y1 - expand_h)
    new_x2 = min(img_width, x2 + expand_w)
    new_y2 = min(img_height, y2 + expand_h)

    return (new_x1, new_y1, new_x2, new_y2)


# ============================================================================
# MAIN NODE CLASS
# ============================================================================

class ArchAi3D_Qwen_Image_Scale_V2:
    """
    Qwen Image Scale V2 - Intelligent scaling with mask-based cropping.

    Features:
    - All V1 features (aspect ratio snapping, VL/latent scaling)
    - Mask-based crop mode with padding and context factor
    - Outputs STITCH_DATA for seamless re-compositing
    """

    def __init__(self):
        pass

    def _letterbox_scale(self, image_bchw: torch.Tensor, target_width: int, target_height: int,
                        upscale_method: str = "area") -> torch.Tensor:
        """Scale image to fit within target dimensions and pad with black bars."""
        _, _, orig_height, orig_width = image_bchw.shape

        scale_w = target_width / orig_width
        scale_h = target_height / orig_height
        scale = min(scale_w, scale_h)

        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)

        scaled_image = comfy.utils.common_upscale(
            image_bchw, new_width, new_height,
            upscale_method=upscale_method, crop="disabled"
        )

        pad_left = (target_width - new_width) // 2
        pad_right = target_width - new_width - pad_left
        pad_top = (target_height - new_height) // 2
        pad_bottom = target_height - new_height - pad_top

        letterboxed = torch.nn.functional.pad(
            scaled_image,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode='constant',
            value=0
        )

        return letterboxed

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                # ===== CROP MODE SETTINGS (NEW in V2) =====
                "crop_mode": (["disabled", "mask_crop"], {
                    "default": "disabled",
                    "tooltip": "disabled: Use full image (V1 behavior). mask_crop: Crop to mask region with padding/context."
                }),
                "crop_padding": ("INT", {
                    "default": 64,
                    "min": 0,
                    "max": 512,
                    "step": 8,
                    "tooltip": "Fixed pixel padding around mask bounding box."
                }),
                "context_factor": ("FLOAT", {
                    "default": 1.0,
                    "min": 1.0,
                    "max": 3.0,
                    "step": 0.1,
                    "tooltip": "Context expansion factor (1.0 = no expansion, 1.5 = 50% extra context). Applied after padding."
                }),
                "blend_pixels": ("INT", {
                    "default": 16,
                    "min": 0,
                    "max": 128,
                    "step": 4,
                    "tooltip": "Feather width for seamless blending in stitch node."
                }),

                # ===== ASPECT RATIO SETTINGS =====
                "aspect_ratio_mode": (["auto", "manual"], {
                    "default": "auto",
                    "tooltip": "Auto: Find closest preferred aspect ratio. Manual: Use selected ratio."
                }),
                "preferred_aspect_ratio": (list(PREFERRED_ASPECT_RATIOS.keys()), {
                    "default": "16:9 (Panorama)",
                    "tooltip": "Manually select preferred aspect ratio (only used in manual mode)."
                }),

                # ===== VL SETTINGS =====
                "vl_target_area": ("INT", {
                    "default": 147456,
                    "min": 3000,
                    "max": 500000,
                    "step": 1000,
                    "tooltip": "Target pixel area for QwenVL output (~147K optimal)."
                }),
                "vl_divisible_by": ("INT", {
                    "default": 32,
                    "min": 1,
                    "max": 256,
                    "step": 1,
                    "tooltip": "VL dimensions divisible by this number."
                }),

                # ===== LATENT SETTINGS =====
                "latent_target_area": ("INT", {
                    "default": 1763584,
                    "min": 3000,
                    "max": 4500000,
                    "step": 1000,
                    "tooltip": "Target pixel area for latent output."
                }),
                "latent_divisible_by": ("INT", {
                    "default": 32,
                    "min": 1,
                    "max": 256,
                    "step": 1,
                    "tooltip": "Latent dimensions divisible by this number."
                }),
                "latent_area_tolerance": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.1,
                    "max": 0.8,
                    "step": 0.05,
                    "tooltip": "Flexibility for latent pixel count (±%)."
                }),

                # ===== VL SOURCE SETTINGS =====
                "vl_use_latent_source": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "True: Use Latent as VL source (pixel-perfect). False: Use original input."
                }),
                "vl_ignore_latent_letterbox": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "When using Latent source, ignore vl_letterbox setting."
                }),
                "vl_ignore_latent_crop": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "When using Latent source, ignore vl_crop setting."
                }),

                # ===== VL SCALING =====
                "vl_upscale_method": (["area", "bicubic", "bilinear", "nearest", "lanczos"], {
                    "default": "area",
                    "tooltip": "Resampling algorithm for VL."
                }),
                "vl_crop": (["disabled", "center"], {
                    "default": "disabled",
                    "tooltip": "Crop mode for VL output."
                }),
                "vl_letterbox": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Add black bars to preserve aspect ratio."
                }),

                # ===== LATENT SCALING =====
                "latent_upscale_method": (["lanczos", "area", "bicubic", "bilinear", "nearest"], {
                    "default": "lanczos",
                    "tooltip": "Resampling algorithm for latent."
                }),
                "latent_crop": (["disabled", "center"], {
                    "default": "center",
                    "tooltip": "Crop mode for latent output."
                }),
                "latent_letterbox": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Add black bars to latent."
                }),

                # ===== MASK INPUT =====
                "mask": ("MASK", {
                    "tooltip": "Mask for crop mode (white = region to crop) and output scaling."
                }),

                # ===== DEBUG =====
                "debug": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show detailed debug info."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "INT", "INT", "STITCH_DATA", "STRING")
    RETURN_NAMES = ("image_vl", "image_latent", "mask", "latent_width", "latent_height", "stitch_data", "debug_text")
    FUNCTION = "process"
    CATEGORY = "ArchAi3d/Qwen"

    def process(self, image: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, Optional[Dict], str]:
        """Process image with optional mask-based cropping."""

        # Extract parameters
        crop_mode = kwargs.get("crop_mode", "disabled")
        crop_padding = kwargs.get("crop_padding", 64)
        context_factor = kwargs.get("context_factor", 1.0)
        blend_pixels = kwargs.get("blend_pixels", 16)

        aspect_ratio_mode = kwargs.get("aspect_ratio_mode", "auto")
        preferred_aspect_ratio = kwargs.get("preferred_aspect_ratio", "16:9 (Panorama)")
        vl_target_area = kwargs.get("vl_target_area", 147456)
        vl_divisible_by = kwargs.get("vl_divisible_by", 32)
        vl_use_latent_source = kwargs.get("vl_use_latent_source", True)
        vl_ignore_latent_letterbox = kwargs.get("vl_ignore_latent_letterbox", True)
        vl_ignore_latent_crop = kwargs.get("vl_ignore_latent_crop", True)
        latent_target_area = kwargs.get("latent_target_area", 1048576)
        latent_divisible_by = kwargs.get("latent_divisible_by", 32)
        latent_area_tolerance = kwargs.get("latent_area_tolerance", 0.3)
        vl_upscale_method = kwargs.get("vl_upscale_method", "area")
        vl_crop = kwargs.get("vl_crop", "disabled")
        vl_letterbox = kwargs.get("vl_letterbox", False)
        latent_upscale_method = kwargs.get("latent_upscale_method", "lanczos")
        latent_crop = kwargs.get("latent_crop", "center")
        latent_letterbox = kwargs.get("latent_letterbox", False)
        debug = kwargs.get("debug", True)
        input_mask = kwargs.get("mask", None)

        # Get original dimensions
        batch_size, orig_height, orig_width, channels = image.shape

        # Store original image for stitch_data
        original_image = image.clone()

        # Ensure RGB
        if channels == 4:
            image = image[:, :, :, :3]
        elif channels == 1:
            image = image.repeat(1, 1, 1, 3)

        # =========================================================================
        # PHASE 1: MASK-BASED CROPPING (if enabled)
        # =========================================================================

        crop_bbox = None
        working_image = image
        crop_mask_tensor = None

        if crop_mode == "mask_crop" and input_mask is not None:
            # Convert mask to PIL for bbox detection
            mask_pil = mask_to_pil(input_mask)

            # Get bounding box with padding
            bbox = get_mask_bbox(mask_pil, padding=crop_padding)

            if bbox is not None:
                # Apply context factor
                bbox = expand_bbox_by_factor(bbox, context_factor, orig_width, orig_height)
                crop_bbox = bbox
                x1, y1, x2, y2 = bbox

                # Crop image to bbox
                working_image = image[:, y1:y2, x1:x2, :]

                # Crop mask to same region
                crop_mask_tensor = input_mask[:, y1:y2, x1:x2] if len(input_mask.shape) == 3 else input_mask[y1:y2, x1:x2]

                if debug:
                    print(f"\n📏 V2 CROP MODE: Cropped to bbox ({x1}, {y1}, {x2}, {y2})")
                    print(f"   Original: {orig_width}x{orig_height} → Cropped: {x2-x1}x{y2-y1}")

        # Get working image dimensions
        _, height, width, _ = working_image.shape

        # Convert to BCHW for scaling
        image_bchw = working_image.movedim(-1, 1).contiguous()

        # =========================================================================
        # PHASE 2: SELECT PREFERRED ASPECT RATIO
        # =========================================================================

        input_aspect = width / height

        if aspect_ratio_mode == "auto":
            w_ratio, h_ratio, ratio_name = find_closest_preferred_aspect_ratio(
                input_aspect, PREFERRED_ASPECT_RATIOS
            )
        else:
            w_ratio, h_ratio = PREFERRED_ASPECT_RATIOS[preferred_aspect_ratio]
            ratio_name = preferred_aspect_ratio

        # =========================================================================
        # PHASE 3: CALCULATE VL DIMENSIONS
        # =========================================================================

        vl_width, vl_height = find_optimal_dimensions_preferred_aspect(
            (w_ratio, h_ratio),
            target_area=vl_target_area,
            divisible_by=vl_divisible_by
        )

        # =========================================================================
        # PHASE 4: CALCULATE LATENT DIMENSIONS
        # =========================================================================

        latent_width, latent_height = find_optimal_dimensions_latent_aspect_lock(
            vl_width, vl_height,
            target_area=latent_target_area,
            tolerance=latent_area_tolerance,
            multiple=latent_divisible_by
        )

        # =========================================================================
        # PHASE 5: PROCESS IMAGES
        # =========================================================================

        # Force stretch mode when in mask_crop to preserve ALL content for proper stitching
        # Center crop would lose content, causing pixel shift when stitching back
        effective_latent_crop = latent_crop
        if crop_mode == "mask_crop" and crop_bbox is not None:
            effective_latent_crop = "disabled"  # Override: preserve all content
            if debug and latent_crop != "disabled":
                print(f"   ⚠️ Overriding latent_crop from '{latent_crop}' to 'disabled' for mask_crop mode")

        # Process LATENT
        if latent_letterbox:
            image_latent_bchw = self._letterbox_scale(
                image_bchw, latent_width, latent_height, latent_upscale_method
            )
        else:
            image_latent_bchw = comfy.utils.common_upscale(
                image_bchw, latent_width, latent_height,
                upscale_method=latent_upscale_method, crop=effective_latent_crop
            )

        # Process VL
        if vl_use_latent_source:
            vl_source_bchw = image_latent_bchw
            should_apply_letterbox = vl_letterbox and not vl_ignore_latent_letterbox

            if should_apply_letterbox:
                image_vl_bchw = self._letterbox_scale(
                    vl_source_bchw, vl_width, vl_height, vl_upscale_method
                )
            else:
                effective_crop = vl_crop if not vl_ignore_latent_crop else "disabled"
                image_vl_bchw = comfy.utils.common_upscale(
                    vl_source_bchw, vl_width, vl_height,
                    upscale_method=vl_upscale_method, crop=effective_crop
                )
        else:
            if vl_letterbox:
                image_vl_bchw = self._letterbox_scale(
                    image_bchw, vl_width, vl_height, vl_upscale_method
                )
            else:
                image_vl_bchw = comfy.utils.common_upscale(
                    image_bchw, vl_width, vl_height,
                    upscale_method=vl_upscale_method, crop=vl_crop
                )

        # Convert back to BHWC
        image_vl = image_vl_bchw.movedim(1, -1).contiguous()
        image_latent = image_latent_bchw.movedim(1, -1).contiguous()

        # =========================================================================
        # PHASE 6: PROCESS MASK
        # =========================================================================

        if input_mask is not None:
            # Use cropped mask if in crop mode, otherwise use full mask
            mask_to_scale = crop_mask_tensor if crop_mask_tensor is not None else input_mask

            # Ensure proper shape for interpolate
            if len(mask_to_scale.shape) == 2:
                mask_to_scale = mask_to_scale.unsqueeze(0)

            mask_for_scale = mask_to_scale.unsqueeze(1)
            scaled_mask = torch.nn.functional.interpolate(
                mask_for_scale,
                size=(latent_height, latent_width),
                mode='bilinear',
                align_corners=True  # Pixel-perfect alignment
            )
            output_mask = scaled_mask.squeeze(1)
        else:
            output_mask = torch.ones((batch_size, latent_height, latent_width),
                                    dtype=torch.float32, device=image.device)

        # =========================================================================
        # PHASE 7: CREATE STITCH_DATA
        # =========================================================================

        if crop_mode == "mask_crop" and crop_bbox is not None:
            crop_w = crop_bbox[2] - crop_bbox[0]
            crop_h = crop_bbox[3] - crop_bbox[1]
            stitch_data = {
                "original_size": (orig_width, orig_height),
                "crop_bbox": crop_bbox,
                "crop_size": (crop_w, crop_h),  # For clarity in stitch node
                "latent_size": (latent_width, latent_height),
                "scale_factors": (latent_width / crop_w, latent_height / crop_h),
                "blend_pixels": blend_pixels,
                "crop_mode": crop_mode,
                "original_image": original_image,
                "crop_mask": crop_mask_tensor,
            }
        else:
            stitch_data = None

        # =========================================================================
        # PHASE 8: DEBUG INFO
        # =========================================================================

        debug_text = ""
        if debug:
            input_area = width * height
            vl_area = vl_width * vl_height
            latent_area = latent_width * latent_height
            vl_aspect = vl_width / vl_height
            latent_aspect = latent_width / latent_height

            crop_info = ""
            if crop_mode == "mask_crop" and crop_bbox is not None:
                x1, y1, x2, y2 = crop_bbox
                crop_info = f"""
🔲 CROP MODE: mask_crop
   Original: {orig_width}x{orig_height}
   Crop bbox: ({x1}, {y1}, {x2}, {y2})
   Cropped size: {x2-x1}x{y2-y1}
   Padding: {crop_padding}px
   Context factor: {context_factor}x
   Blend pixels: {blend_pixels}
   STITCH_DATA: ✅ Generated"""
            else:
                crop_info = f"""
🔲 CROP MODE: disabled (full image)
   STITCH_DATA: None"""

            debug_text = f"""📏 Qwen Image Scale V2 Debug Info
━━━━━━━━━━━━━━━━━━━━━━━━━━━{crop_info}

🎨 ASPECT RATIO:
   Mode: {aspect_ratio_mode.upper()}
   Selected: {ratio_name} ({w_ratio}:{h_ratio})
   Input aspect: {input_aspect:.4f}

📐 DIMENSIONS:
   Working: {width}x{height} = {input_area:,}px
   VL:      {vl_width}x{vl_height} = {vl_area:,}px | aspect: {vl_aspect:.4f}
   LATENT:  {latent_width}x{latent_height} = {latent_area:,}px | aspect: {latent_aspect:.4f}

📊 SCALE FACTORS:
   VL:     {vl_width/width:.3f}x width | {vl_height/height:.3f}x height
   LATENT: {latent_width/width:.3f}x width | {latent_height/height:.3f}x height
━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

            print(debug_text)

        return (image_vl, image_latent, output_mask, latent_width, latent_height, stitch_data, debug_text)


# ============================================================================
# NODE REGISTRATION
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "ArchAi3D_Qwen_Image_Scale_V2": ArchAi3D_Qwen_Image_Scale_V2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_Qwen_Image_Scale_V2": "📏 Qwen Image Scale V2",
}
