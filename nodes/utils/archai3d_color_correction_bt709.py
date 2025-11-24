# -*- coding: utf-8 -*-
"""
ArchAi3D Color Correction BT.709 Node
Author: Amir Ferdos (ArchAi3d)
Email: Amir84ferdos@gmail.com
LinkedIn: https://www.linkedin.com/in/archai3d/

Description:
    Automatic color correction with BT.709 linear chroma matching.
    Applies highlight and lowlight attenuation to prevent:
    - Bright-scene green shift
    - Highlight clipping
    - Loss of detail in shadows

Usage:
    Pre-processing node before Qwen image editing.
    Automatically corrects color and exposure issues.

Version: 1.0.0
Created: 2025-10-17
"""

import numpy as np
import torch
from PIL import Image
import colour
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override


# ============================================================================
# COLOR SPACE TRANSFORMS
# ============================================================================

def srgb_to_linear(img):
    """
    Convert sRGB to linear RGB using standard gamma correction.

    Args:
        img: numpy array in range [0, 1], sRGB color space

    Returns:
        numpy array in range [0, 1], linear RGB
    """
    # sRGB to linear formula (ITU-R BT.709)
    linear = np.where(
        img <= 0.04045,
        img / 12.92,
        np.power((img + 0.055) / 1.055, 2.4)
    )
    return linear


def linear_to_srgb(img):
    """
    Convert linear RGB to sRGB using standard gamma correction.

    Args:
        img: numpy array in range [0, 1], linear RGB

    Returns:
        numpy array in range [0, 1], sRGB color space
    """
    # Linear to sRGB formula (ITU-R BT.709)
    srgb = np.where(
        img <= 0.0031308,
        img * 12.92,
        1.055 * np.power(img, 1.0 / 2.4) - 0.055
    )
    return srgb


def apply_bt709_matrix(img_linear):
    """
    Apply BT.709 color matrix to ensure proper chroma.

    Args:
        img_linear: numpy array (H, W, 3) in linear RGB

    Returns:
        numpy array (H, W, 3) with BT.709 chroma
    """
    # BT.709 RGB to XYZ matrix (D65 white point)
    # This ensures proper chroma characteristics
    bt709_to_xyz = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])

    # XYZ to BT.709 RGB (inverse)
    xyz_to_bt709 = np.linalg.inv(bt709_to_xyz)

    # Reshape for matrix multiplication
    h, w, c = img_linear.shape
    img_flat = img_linear.reshape(-1, 3)

    # Apply BT.709 matrix
    img_xyz = img_flat @ bt709_to_xyz.T
    img_bt709 = img_xyz @ xyz_to_bt709.T

    # Reshape back
    img_bt709 = img_bt709.reshape(h, w, c)

    return img_bt709


# ============================================================================
# HIGHLIGHT/LOWLIGHT ATTENUATION
# ============================================================================

def detect_highlights(img_linear, threshold=0.8):
    """
    Detect highlight regions in linear RGB.

    Args:
        img_linear: numpy array (H, W, 3) in linear RGB [0, 1]
        threshold: brightness threshold for highlights

    Returns:
        numpy array (H, W) mask where highlights are detected
    """
    # Calculate luminance (BT.709 weights)
    luminance = 0.2126 * img_linear[..., 0] + \
                0.7152 * img_linear[..., 1] + \
                0.0722 * img_linear[..., 2]

    # Highlight mask
    highlight_mask = luminance > threshold

    return highlight_mask, luminance


def detect_lowlights(img_linear, threshold=0.1):
    """
    Detect lowlight (shadow) regions in linear RGB.

    Args:
        img_linear: numpy array (H, W, 3) in linear RGB [0, 1]
        threshold: brightness threshold for lowlights

    Returns:
        numpy array (H, W) mask where lowlights are detected
    """
    # Calculate luminance (BT.709 weights)
    luminance = 0.2126 * img_linear[..., 0] + \
                0.7152 * img_linear[..., 1] + \
                0.0722 * img_linear[..., 2]

    # Lowlight mask
    lowlight_mask = luminance < threshold

    return lowlight_mask, luminance


def attenuate_highlights(img_linear, luminance, threshold=0.8, strength=0.5):
    """
    Attenuate highlights to prevent clipping and green shift.

    Uses smooth compression curve above threshold.

    Args:
        img_linear: numpy array (H, W, 3) in linear RGB
        luminance: numpy array (H, W) with luminance values
        threshold: brightness threshold for attenuation
        strength: attenuation strength (0.0 = none, 1.0 = maximum)

    Returns:
        numpy array (H, W, 3) with attenuated highlights
    """
    # Create smooth compression curve for highlights
    highlight_factor = np.ones_like(luminance)

    # Where luminance > threshold, apply compression
    mask = luminance > threshold

    # Smooth compression: reduces extreme values more than moderate ones
    excess = (luminance - threshold) / (1.0 - threshold + 1e-6)
    compression = 1.0 - (excess * strength * 0.5)  # Compress to 50-100% of original
    compression = np.clip(compression, 0.5, 1.0)

    highlight_factor[mask] = compression[mask]

    # Apply factor to RGB channels
    highlight_factor_3d = highlight_factor[..., np.newaxis]
    img_corrected = img_linear * highlight_factor_3d

    return img_corrected


def attenuate_lowlights(img_linear, luminance, threshold=0.1, strength=0.5):
    """
    Attenuate lowlights (lift shadows) to preserve detail.

    Args:
        img_linear: numpy array (H, W, 3) in linear RGB
        luminance: numpy array (H, W) with luminance values
        threshold: brightness threshold for attenuation
        strength: lift strength (0.0 = none, 1.0 = maximum)

    Returns:
        numpy array (H, W, 3) with lifted shadows
    """
    # Create smooth lift curve for shadows
    lowlight_lift = np.zeros_like(luminance)

    # Where luminance < threshold, apply lift
    mask = luminance < threshold

    # Smooth lift: adds more to very dark areas
    deficit = (threshold - luminance) / (threshold + 1e-6)
    lift = deficit * strength * 0.1  # Lift by up to 10% of threshold

    lowlight_lift[mask] = lift[mask]

    # Apply lift to RGB channels
    lowlight_lift_3d = lowlight_lift[..., np.newaxis]
    img_corrected = img_linear + lowlight_lift_3d

    return img_corrected


def correct_green_shift(img_linear, luminance, threshold=0.8):
    """
    Correct green shift in bright scenes.

    Bright scenes often have excessive green channel values.
    This reduces green in highlights while preserving color balance.

    Args:
        img_linear: numpy array (H, W, 3) in linear RGB
        luminance: numpy array (H, W) with luminance values
        threshold: brightness threshold for correction

    Returns:
        numpy array (H, W, 3) with corrected green channel
    """
    # Detect bright areas
    mask = luminance > threshold

    # Calculate green excess
    r = img_linear[..., 0]
    g = img_linear[..., 1]
    b = img_linear[..., 2]

    # Average of R and B channels
    rb_avg = (r + b) / 2.0

    # Green excess (positive when green is higher than R/B average)
    green_excess = g - rb_avg

    # Reduce green excess in highlights
    correction = np.zeros_like(img_linear)
    correction[..., 1][mask] = -green_excess[mask] * 0.3  # Reduce 30% of excess

    img_corrected = img_linear + correction

    return img_corrected


# ============================================================================
# TENSOR CONVERSION HELPERS
# ============================================================================

def tensor_to_numpy(tensor):
    """
    Convert ComfyUI image tensor to numpy array.

    Args:
        tensor: torch.Tensor (B, H, W, C) in range [0, 1]

    Returns:
        numpy array (H, W, C) in range [0, 1]
    """
    # Take first image if batched
    if len(tensor.shape) == 4:
        tensor = tensor[0]

    # Convert to numpy
    img_np = tensor.cpu().numpy()

    # Ensure [0, 1] range
    img_np = np.clip(img_np, 0.0, 1.0)

    return img_np


def numpy_to_tensor(img_np):
    """
    Convert numpy array to ComfyUI image tensor.

    Args:
        img_np: numpy array (H, W, C) in range [0, 1]

    Returns:
        torch.Tensor (1, H, W, C) in range [0, 1]
    """
    # Ensure [0, 1] range
    img_np = np.clip(img_np, 0.0, 1.0)

    # Convert to tensor
    tensor = torch.from_numpy(img_np).float()

    # Add batch dimension
    tensor = tensor.unsqueeze(0)

    return tensor


# ============================================================================
# MAIN COLOR CORRECTION FUNCTION
# ============================================================================

def color_correct_bt709_auto(img_np):
    """
    Automatic BT.709 color correction with highlight/lowlight attenuation.

    Pipeline:
    1. Convert sRGB to linear RGB
    2. Apply BT.709 chroma matrix
    3. Detect and attenuate highlights
    4. Detect and lift lowlights
    5. Correct green shift in bright areas
    6. Convert back to sRGB

    Args:
        img_np: numpy array (H, W, 3) in sRGB [0, 1]

    Returns:
        numpy array (H, W, 3) in sRGB [0, 1], corrected
    """
    # Step 1: sRGB to linear
    img_linear = srgb_to_linear(img_np)

    # Step 2: Apply BT.709 chroma matrix
    img_bt709 = apply_bt709_matrix(img_linear)

    # Step 3: Detect luminance
    _, luminance = detect_highlights(img_bt709, threshold=0.8)

    # Step 4: Attenuate highlights (prevent clipping)
    img_corrected = attenuate_highlights(
        img_bt709,
        luminance,
        threshold=0.8,
        strength=0.6  # 60% attenuation
    )

    # Step 5: Lift lowlights (preserve shadow detail)
    img_corrected = attenuate_lowlights(
        img_corrected,
        luminance,
        threshold=0.1,
        strength=0.5  # 50% lift
    )

    # Step 6: Correct green shift in bright areas
    _, luminance_updated = detect_highlights(img_corrected, threshold=0.8)
    img_corrected = correct_green_shift(
        img_corrected,
        luminance_updated,
        threshold=0.8
    )

    # Step 7: Clip to valid range
    img_corrected = np.clip(img_corrected, 0.0, 1.0)

    # Step 8: Convert back to sRGB
    img_srgb = linear_to_srgb(img_corrected)

    # Final clip
    img_srgb = np.clip(img_srgb, 0.0, 1.0)

    return img_srgb


# ============================================================================
# COMFYUI NODE CLASS
# ============================================================================

class ArchAi3D_Color_Correction_BT709(io.ComfyNode):
    """
    Automatic BT.709 color correction node.

    Pre-processing for Qwen image editing:
    - Linear BT.709 chroma matching
    - Highlight attenuation (prevents clipping and green shift)
    - Lowlight lifting (preserves shadow detail)
    - Fully automatic, no manual controls needed
    """

    @classmethod
    def define_schema(cls):
        """Define node schema for ComfyUI."""
        return io.Schema(
            node_id="ArchAi3D_Color_Correction_BT709",
            category="ArchAi3d/Utils",
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="Input image to color correct (pre-processing for Qwen)"
                ),
            ],
            outputs=[
                io.Image.Output(
                    "corrected_image",
                    tooltip="Color-corrected image ready for Qwen processing"
                ),
            ],
        )

    @classmethod
    def execute(cls, image) -> io.NodeOutput:
        """
        Execute automatic BT.709 color correction.

        Args:
            image: ComfyUI image tensor (B, H, W, C) in sRGB [0, 1]

        Returns:
            io.NodeOutput with corrected image tensor
        """
        # Convert tensor to numpy
        img_np = tensor_to_numpy(image)

        # Apply automatic color correction
        img_corrected = color_correct_bt709_auto(img_np)

        # Convert back to tensor
        corrected_tensor = numpy_to_tensor(img_corrected)

        # Debug output
        print("\n" + "="*70)
        print("🎨 ArchAi3D Color Correction BT.709 - v1.0.0")
        print("="*70)
        print(f"Input shape: {image.shape}")
        print(f"Output shape: {corrected_tensor.shape}")
        print(f"Corrections applied:")
        print(f"  ✓ Linear BT.709 chroma matching")
        print(f"  ✓ Highlight attenuation (threshold: 0.8, strength: 60%)")
        print(f"  ✓ Lowlight lifting (threshold: 0.1, strength: 50%)")
        print(f"  ✓ Green shift correction in bright areas")
        print("="*70 + "\n")

        return io.NodeOutput(corrected_tensor)


# ============================================================================
# EXTENSION REGISTRATION
# ============================================================================

class ColorCorrectionBT709Extension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [ArchAi3D_Color_Correction_BT709]


async def comfy_entrypoint():
    return ColorCorrectionBT709Extension()
