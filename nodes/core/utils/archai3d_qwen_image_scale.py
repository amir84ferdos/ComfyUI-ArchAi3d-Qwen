"""
ArchAi3D Qwen Image Scale
Version: 1.0.0

Intelligent image scaling for QwenVL with preferred aspect ratios.
- Aspect Selection: Snaps to preferred ratios (1:1, 16:9, 3:4, etc.) optimized for QwenVL
- Calculation: VL uses preferred ratio → Latent matches VL aspect EXACTLY
- Processing: Latent first (crop by default) → VL from Latent (pixel-perfect)

Author: Amir Ferdos (ArchAi3d)
Email: Amir84ferdos@gmail.com
LinkedIn: https://www.linkedin.com/in/archai3d/
GitHub: https://github.com/amir84ferdos
Category: ArchAi3d/Qwen
License: MIT
"""

import torch
import math
import comfy.utils
from typing import Tuple, Optional


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
    """
    Find the closest preferred aspect ratio to the input aspect ratio.

    Args:
        input_aspect: Input aspect ratio (width/height)
        preferred_ratios: Dictionary of preferred aspect ratios

    Returns:
        Tuple of (width_ratio, height_ratio, ratio_name)
        Example: (16, 9, "16:9 (Panorama)")
    """
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
    """
    Calculate dimensions using a preferred aspect ratio.
    Priority: Preferred ratio > divisibility > target area

    Args:
        preferred_ratio: (width_ratio, height_ratio) e.g., (16, 9)
        target_area: Target pixel area (default ~147K for QwenVL)
        divisible_by: Both dimensions must be divisible by this

    Returns:
        Tuple of (width, height) matching preferred ratio

    Example:
        preferred_ratio=(16, 9), target_area=147456, divisible_by=32
        Returns (512, 288) which is exactly 16:9 and ~147K pixels
    """
    w_ratio, h_ratio = preferred_ratio
    aspect_ratio = w_ratio / h_ratio

    # Calculate ideal dimensions from preferred aspect ratio
    ideal_height = math.sqrt(target_area / aspect_ratio)
    ideal_width = ideal_height * aspect_ratio

    # Search for best dimensions that maintain exact preferred ratio
    best_width, best_height = None, None
    best_error = float('inf')

    # Try different multiples of divisible_by around ideal dimensions
    for height_mult in range(
        max(1, int(ideal_height / divisible_by) - 3),
        int(ideal_height / divisible_by) + 4
    ):
        height_test = height_mult * divisible_by
        if height_test < divisible_by:
            continue

        # Calculate width from EXACT preferred ratio
        width_exact = height_test * aspect_ratio
        width_test = round_to_multiple(width_exact, divisible_by)
        width_test = max(divisible_by, width_test)

        # Verify aspect ratio is maintained (allow 1% tolerance for rounding)
        test_aspect = width_test / height_test
        aspect_error = abs(test_aspect - aspect_ratio)
        if aspect_error / aspect_ratio > 0.01:
            continue  # Skip if aspect drifted too much

        # Calculate area error
        test_area = width_test * height_test
        area_error = abs(test_area - target_area)

        # Total error (heavily penalize aspect ratio drift)
        total_error = area_error + (aspect_error * 100000000)

        if total_error < best_error:
            best_error = total_error
            best_width = width_test
            best_height = height_test

    # Fallback: force aspect match even if not perfect divisibility
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
    """
    Find latent dimensions that EXACTLY match VL aspect ratio.
    Allows pixel count flexibility within tolerance range.

    Args:
        vl_width: VL width (defines aspect ratio)
        vl_height: VL height (defines aspect ratio)
        target_area: Target pixel area for latent (default ~1M)
        tolerance: ±% flexibility for area (e.g., 0.3 = ±30%)
        multiple: Both dimensions divisible by this

    Returns:
        (width, height) that EXACTLY match VL aspect ratio

    Priority: EXACT aspect ratio > divisibility > target area
    """
    # Get exact aspect ratio from VL
    aspect_ratio = vl_width / vl_height

    # Calculate area bounds
    area_min = int(target_area * (1 - tolerance))
    area_max = int(target_area * (1 + tolerance))

    # Calculate ideal dimensions
    ideal_height = math.sqrt(target_area / aspect_ratio)
    ideal_width = ideal_height * aspect_ratio

    # Search for best dimensions
    best_width, best_height = None, None
    best_error = float('inf')

    # Search range (within tolerance bounds)
    height_min = int(math.sqrt(area_min / aspect_ratio))
    height_max = int(math.sqrt(area_max / aspect_ratio))

    # Ensure minimum search range
    height_min = max(multiple, height_min)
    height_max = max(height_min + multiple * 10, height_max)

    # Try all possible heights (divisible by multiple)
    for height_test in range(
        round_to_multiple(height_min, multiple),
        round_to_multiple(height_max, multiple) + multiple,
        multiple
    ):
        if height_test < multiple:
            continue

        # Calculate width from EXACT aspect ratio
        width_exact = height_test * aspect_ratio
        width_test = round_to_multiple(width_exact, multiple)
        width_test = max(multiple, width_test)

        # Calculate resulting area
        test_area = width_test * height_test

        # Check if within tolerance
        if area_min <= test_area <= area_max:
            # Calculate errors
            area_error = abs(test_area - target_area)

            # Aspect error (penalized heavily)
            test_aspect = width_test / height_test
            aspect_error = abs(test_aspect - aspect_ratio) * 10000000

            total_error = area_error + aspect_error

            if total_error < best_error:
                best_error = total_error
                best_width = width_test
                best_height = height_test

    # Fallback: force aspect match even if outside tolerance
    if best_width is None:
        best_height = round_to_multiple(ideal_height, multiple)
        best_height = max(multiple, best_height)
        best_width = round_to_multiple(best_height * aspect_ratio, multiple)
        best_width = max(multiple, best_width)

    return best_width, best_height


# ============================================================================
# MAIN NODE CLASS
# ============================================================================

class ArchAi3D_Qwen_Image_Scale:
    """
    Intelligent image scaling for QwenVL and latent processing.
    - Snaps to preferred aspect ratios optimized for QwenVL (1:1, 16:9, 3:4, etc.)
    - VL calculated using preferred ratio → Latent matches VL aspect EXACTLY
    - Latent processed FIRST (crop by default) → VL from Latent (pixel-perfect)
    """

    def __init__(self):
        pass

    def _letterbox_scale(self, image_bchw: torch.Tensor, target_width: int, target_height: int,
                        upscale_method: str = "area") -> torch.Tensor:
        """
        Scale image to fit within target dimensions and pad with black bars (letterbox).
        Preserves aspect ratio without cropping.
        """
        _, _, orig_height, orig_width = image_bchw.shape

        # Calculate scale to fit inside target dimensions
        scale_w = target_width / orig_width
        scale_h = target_height / orig_height
        scale = min(scale_w, scale_h)

        # Calculate new dimensions (maintaining aspect ratio)
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)

        # Scale the image
        scaled_image = comfy.utils.common_upscale(
            image_bchw, new_width, new_height,
            upscale_method=upscale_method, crop="disabled"
        )

        # Calculate padding
        pad_left = (target_width - new_width) // 2
        pad_right = target_width - new_width - pad_left
        pad_top = (target_height - new_height) // 2
        pad_bottom = target_height - new_height - pad_top

        # Pad with black (0)
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
                # Aspect ratio selection
                "aspect_ratio_mode": (["auto", "manual"], {
                    "default": "auto",
                    "tooltip": "Auto: Automatically find closest preferred aspect ratio to input image. Manual: Choose specific preferred ratio (ignores input aspect)."
                }),
                "preferred_aspect_ratio": (list(PREFERRED_ASPECT_RATIOS.keys()), {
                    "default": "16:9 (Panorama)",
                    "tooltip": "Manually select preferred aspect ratio (only used in manual mode). These ratios are optimized for QwenVL's training data."
                }),

                # QwenVL settings (calculated using preferred aspect)
                "vl_target_area": ("INT", {
                    "default": 147456,  # 384×384
                    "min": 3000,
                    "max": 500000,
                    "step": 1000,
                    "tooltip": "Target pixel area for QwenVL output (e.g., 147456 = ~147K = 384×384). VL dimensions are calculated using preferred aspect ratio to match QwenVL's optimal training ratios. QwenVL works best with ~150K pixels."
                }),
                "vl_divisible_by": ("INT", {
                    "default": 32,
                    "min": 1,
                    "max": 256,
                    "step": 1,
                    "tooltip": "Ensures VL width and height are divisible by this number. Recommended: 32 for optimal AI processing."
                }),

                # Latent settings (matches VL preferred aspect)
                "latent_target_area": ("INT", {
                    "default": 1763584,  # 1.76 pixels
                    "min": 3000,
                    "max": 4500000,
                    "step": 1000,
                    "tooltip": "Target pixel area for latent output (e.g., 1048576 = ~1M = ~1024×1024). Latent dimensions will EXACTLY match VL's preferred aspect ratio while staying within tolerance range. Higher = better quality but slower."
                }),
                "latent_divisible_by": ("INT", {
                    "default": 32,
                    "min": 1,
                    "max": 256,
                    "step": 1,
                    "tooltip": "Ensures latent width and height are divisible by this number. Recommended: 32 for VAE compatibility. Must achieve exact aspect match with VL."
                }),
                "latent_area_tolerance": ("FLOAT", {
                    "default": 0.3,  # ±30%
                    "min": 0.1,
                    "max": 0.8,
                    "step": 0.05,
                    "tooltip": "Flexibility for latent pixel count (±%). Default 0.3 = ±30%, allowing 700K-1.3M when target is 1M. Gives algorithm room to find dimensions that EXACTLY match VL aspect ratio while meeting divisibility constraints. Higher tolerance = easier to find perfect aspect match."
                }),

                # VL source selection (HYBRID approach)
                "vl_use_latent_source": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "VL source selection. True (recommended): Use processed Latent as source for pixel-perfect alignment and guaranteed aspect match. False: Use original Input for traditional independent processing."
                }),
                "vl_ignore_latent_letterbox": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "When using Latent source, ignore vl_letterbox setting. Latent is already letterboxed if needed, prevents double padding."
                }),
                "vl_ignore_latent_crop": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "When using Latent source, ignore vl_crop setting. Latent is already cropped if needed, prevents double cropping."
                }),

                # VL scaling settings
                "vl_upscale_method": (["area", "bicubic", "bilinear", "nearest", "lanczos"], {
                    "default": "area",
                    "tooltip": "Resampling algorithm for VL. Area=best for downscaling (sharp), Lanczos=best for upscaling (smooth). Default 'area' optimal since VL is typically downscaled."
                }),
                "vl_crop": (["disabled", "center"], {
                    "default": "disabled",
                    "tooltip": "How to handle aspect ratio mismatch. Usually keep disabled as VL aspect matches Latent exactly. NOTE: Ignored when vl_use_latent_source=True and vl_ignore_latent_crop=True."
                }),
                "vl_letterbox": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Add black bars to preserve exact aspect ratio. Usually keep disabled as VL aspect matches Latent exactly. NOTE: Ignored when vl_use_latent_source=True and vl_ignore_latent_letterbox=True."
                }),

                # Latent scaling settings
                "latent_upscale_method": (["lanczos", "area", "bicubic", "bilinear", "nearest"], {
                    "default": "lanczos",
                    "tooltip": "Resampling algorithm for latent. Lanczos=best for upscaling (smooth, high quality), Area=best for downscaling (sharp). Default 'lanczos' optimal since latent is usually upscaled."
                }),
                "latent_crop": (["disabled", "center"], {
                    "default": "center",
                    "tooltip": "How to handle aspect ratio mismatch for latent. Center (recommended): Crops input image to fill latent frame completely with preferred aspect ratio. Disabled: Stretches to fit (may distort)."
                }),
                "latent_letterbox": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Add black bars to latent instead of cropping. Enable to preserve entire input image with preferred aspect ratio."
                }),

                # Debug
                "debug": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show detailed info: aspect ratio selection (auto/manual), preferred ratio match quality, dimensions, tolerance checks, and pixel alignment status."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("image_vl", "image_latent", "debug_text")
    FUNCTION = "process"
    CATEGORY = "ArchAi3d/Qwen"

    def process(self, image: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Process image using preferred aspect ratios optimized for QwenVL.
        VL calculated with preferred ratio, Latent matches VL aspect.
        Latent processed first (crop), VL from Latent (pixel-perfect).
        """
        # Extract parameters
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

        # Get input dimensions
        batch_size, height, width, channels = image.shape

        # Ensure RGB (3 channels)
        if channels == 4:
            image = image[:, :, :, :3]
        elif channels == 1:
            image = image.repeat(1, 1, 1, 3)

        # Convert to BCHW for scaling
        image_bchw = image.movedim(-1, 1).contiguous()

        # =========================================================================
        # PHASE 1: SELECT PREFERRED ASPECT RATIO
        # =========================================================================

        input_aspect = width / height

        if aspect_ratio_mode == "auto":
            # Automatically find closest preferred aspect ratio
            w_ratio, h_ratio, ratio_name = find_closest_preferred_aspect_ratio(
                input_aspect,
                PREFERRED_ASPECT_RATIOS
            )
        else:  # manual
            # Use user-selected preferred ratio
            w_ratio, h_ratio = PREFERRED_ASPECT_RATIOS[preferred_aspect_ratio]
            ratio_name = preferred_aspect_ratio

        preferred_aspect = w_ratio / h_ratio

        # =========================================================================
        # PHASE 2: CALCULATE VL DIMENSIONS (Using preferred aspect ratio)
        # =========================================================================

        vl_width, vl_height = find_optimal_dimensions_preferred_aspect(
            (w_ratio, h_ratio),
            target_area=vl_target_area,
            divisible_by=vl_divisible_by
        )

        # =========================================================================
        # PHASE 3: CALCULATE LATENT DIMENSIONS (Match VL aspect exactly)
        # =========================================================================

        latent_width, latent_height = find_optimal_dimensions_latent_aspect_lock(
            vl_width, vl_height,  # VL aspect ratio
            target_area=latent_target_area,
            tolerance=latent_area_tolerance,
            multiple=latent_divisible_by
        )

        # =========================================================================
        # PHASE 4: PROCESS IMAGES
        # =========================================================================

        # STEP 1: Process LATENT first (from original input, crop by default)
        if latent_letterbox:
            image_latent_bchw = self._letterbox_scale(
                image_bchw, latent_width, latent_height, latent_upscale_method
            )
        else:
            image_latent_bchw = comfy.utils.common_upscale(
                image_bchw, latent_width, latent_height,
                upscale_method=latent_upscale_method, crop=latent_crop
            )

        # STEP 2: Process VL from Latent OR Input (hybrid approach)
        if vl_use_latent_source:
            # Use processed Latent as source (pixel-perfect)
            vl_source_bchw = image_latent_bchw

            # Apply letterbox/crop only if not ignoring
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
            # Use original Input as source (traditional)
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

        # Calculate debug information
        debug_text = ""
        if debug:
            input_area = width * height
            vl_area = vl_width * vl_height
            latent_area = latent_width * latent_height

            vl_aspect = vl_width / vl_height
            latent_aspect = latent_width / latent_height

            # Aspect ratio differences
            aspect_diff_input_preferred = abs(input_aspect - preferred_aspect) / input_aspect * 100
            aspect_diff_vl_preferred = abs(vl_aspect - preferred_aspect) / preferred_aspect * 100
            aspect_diff_vl_latent = abs(vl_aspect - latent_aspect) / vl_aspect * 100

            # Aspect ratio selection quality
            if aspect_diff_input_preferred < 0.1:
                match_quality = "✅ PERFECT (0.1%)"
            elif aspect_diff_input_preferred < 1.0:
                match_quality = "✅ EXCELLENT (<1%)"
            elif aspect_diff_input_preferred < 5.0:
                match_quality = "✅ GOOD (<5%)"
            else:
                match_quality = f"⚠️ ADJUSTED ({aspect_diff_input_preferred:.1f}%)"

            # Check divisibility
            vl_w_div = "✅" if vl_width % vl_divisible_by == 0 else "❌"
            vl_h_div = "✅" if vl_height % vl_divisible_by == 0 else "❌"
            lat_w_div = "✅" if latent_width % latent_divisible_by == 0 else "❌"
            lat_h_div = "✅" if latent_height % latent_divisible_by == 0 else "❌"

            vl_div_perfect = vl_width % vl_divisible_by == 0 and vl_height % vl_divisible_by == 0
            lat_div_perfect = latent_width % latent_divisible_by == 0 and latent_height % latent_divisible_by == 0
            div_status = "✅ PERFECT" if (vl_div_perfect and lat_div_perfect) else "⚠️ CHECK"

            # Aspect ratio lock quality
            if aspect_diff_vl_latent < 0.01:
                aspect_lock_status = "✅ PERFECT - Locked at <0.01%"
            elif aspect_diff_vl_latent < 0.1:
                aspect_lock_status = "✅ EXCELLENT - Locked at <0.1%"
            elif aspect_diff_vl_latent < 0.5:
                aspect_lock_status = "✅ GOOD - Locked at <0.5%"
            else:
                aspect_lock_status = "⚠️ ACCEPTABLE - Check tolerance"

            # VL source display
            if vl_use_latent_source:
                vl_source_display = "🔗 LATENT (pixel-perfect alignment)"
                if vl_ignore_latent_letterbox and vl_ignore_latent_crop:
                    vl_source_note = "Inheriting Latent letterbox/crop (no double processing)"
                elif vl_ignore_latent_letterbox:
                    vl_source_note = "Inheriting Latent letterbox, applying VL crop"
                elif vl_ignore_latent_crop:
                    vl_source_note = "Inheriting Latent crop, applying VL letterbox"
                else:
                    vl_source_note = "Applying additional VL letterbox/crop"
            else:
                vl_source_display = "📥 INPUT (traditional mode)"
                vl_source_note = "VL processed independently from Input image"

            # Pixel alignment status
            if vl_use_latent_source:
                if aspect_diff_vl_latent < 0.01:
                    pixel_alignment = "✅ PERFECT - Zero pixel shift guaranteed"
                elif aspect_diff_vl_latent < 0.1:
                    pixel_alignment = "✅ EXCELLENT - Minimal pixel shift"
                else:
                    pixel_alignment = "⚠️ GOOD - Minor aspect adjustment"
            else:
                pixel_alignment = "⚠️ Independent processing - Pixel shifts may occur"

            # Area tolerance check
            area_min = int(latent_target_area * (1 - latent_area_tolerance))
            area_max = int(latent_target_area * (1 + latent_area_tolerance))
            area_delta = latent_area - latent_target_area
            delta_percent = (area_delta / latent_target_area) * 100

            if area_min <= latent_area <= area_max:
                tolerance_status = f"✅ Within ±{int(latent_area_tolerance*100)}% tolerance"
            else:
                tolerance_status = f"⚠️ Outside ±{int(latent_area_tolerance*100)}% tolerance"

            # Processing strategy display
            latent_process_mode = "LETTERBOX (add black bars)" if latent_letterbox else f"CROP ({latent_crop})" if latent_crop != "disabled" else "STRETCH (disabled)"

            debug_text = f"""🎯 Qwen Preferred-Aspect Scale Debug Info
━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎨 ASPECT RATIO SELECTION:
  Mode:      {aspect_ratio_mode.upper()}
  Input:     {width} × {height} | aspect: {input_aspect:.4f}
  Selected:  {ratio_name} | aspect: {preferred_aspect:.4f}
  Match:     {match_quality} ({aspect_diff_input_preferred:.2f}% difference)

🔧 CALCULATION ORDER:
  1️⃣ VL (using preferred {w_ratio}:{h_ratio}): {vl_target_area//1000}K px target
  2️⃣ LATENT (matches VL {w_ratio}:{h_ratio}): {latent_target_area//1000}K px target ±{int(latent_area_tolerance*100)}%

🔧 PROCESSING STRATEGY:
  1️⃣ LATENT: {latent_process_mode} (fill frame with preferred aspect)
  2️⃣ VL: Source={vl_source_display}
     {vl_source_note}

🔒 ASPECT RATIO LOCK: {aspect_lock_status}
🎯 PIXEL ALIGNMENT: {pixel_alignment}
━━━━━━━━━━━━━━━━━━━━━━━━━━━
📐 DIMENSIONS:
  INPUT:  {width:4d} × {height:4d} = {input_area:8,d} px ({input_area//1000}K) | aspect: {input_aspect:.4f}
  VL:     {vl_width:4d} × {vl_height:4d} = {vl_area:8,d} px ({vl_area//1000}K) | aspect: {vl_aspect:.4f} ({w_ratio}:{h_ratio}) ✅
  LATENT: {latent_width:4d} × {latent_height:4d} = {latent_area:8,d} px ({latent_area//1000}K) | aspect: {latent_aspect:.4f} ({w_ratio}:{h_ratio}) ✅

📊 AREA TOLERANCE CHECK:
  Target:  {latent_target_area:,d} px ({latent_target_area//1000}K)
  Range:   {area_min:,d} - {area_max:,d} px ({area_min//1000}K - {area_max//1000}K)
  Actual:  {latent_area:,d} px ({latent_area//1000}K)
  Delta:   {area_delta:+,d} px ({delta_percent:+.1f}%) {tolerance_status}

🔢 DIVISIBILITY CHECK (VL÷{vl_divisible_by}, Latent÷{latent_divisible_by}): {div_status}
  VL:     Width {vl_w_div} ({vl_width}÷{vl_divisible_by}={vl_width/vl_divisible_by:.1f})  |  Height {vl_h_div} ({vl_height}÷{vl_divisible_by}={vl_height/vl_divisible_by:.1f})
  LATENT: Width {lat_w_div} ({latent_width}÷{latent_divisible_by}={latent_width/latent_divisible_by:.1f})  |  Height {lat_h_div} ({latent_height}÷{latent_divisible_by}={latent_height/latent_divisible_by:.1f})

⭐ PREFERRED RATIO QUALITY: {"✅ EXACT MATCH" if aspect_diff_vl_preferred < 0.01 else "✅ EXCELLENT" if aspect_diff_vl_preferred < 0.1 else "⚠️ CHECK"} ({w_ratio}:{h_ratio})
  VL aspect:     {vl_aspect:.6f}
  Latent aspect: {latent_aspect:.6f}
  Difference:    {aspect_diff_vl_latent:.6f}% {"✅ LOCKED!" if aspect_diff_vl_latent < 0.1 else "⚠️ CHECK"}

📊 SCALE FACTORS:
  VL:     {vl_width/width:.3f}× width  |  {vl_height/height:.3f}× height
  LATENT: {latent_width/width:.3f}× width  |  {latent_height/height:.3f}× height
━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

            print(debug_text)

        return (image_vl, image_latent, debug_text)


# ============================================================================
# NODE REGISTRATION
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "ArchAi3D_Qwen_Image_Scale": ArchAi3D_Qwen_Image_Scale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_Qwen_Image_Scale": "ArchAi3D Qwen Image Scale",
}
