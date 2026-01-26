# -*- coding: utf-8 -*-
"""
ArchAi3D High-Pass Filter + Blend Node
Author: Amir Ferdos (ArchAi3d)
Email: Amir84ferdos@gmail.com
LinkedIn: https://www.linkedin.com/in/archai3d/

Description:
    Detail-preserving image enhancement using high-pass filtering.
    Extracts high frequency details from ORIGINAL and transfers to PROCESSED image.

    Workflow:
    1. Extract high frequency details from ORIGINAL image
    2. Apply those details to RELIGHTED/PROCESSED image
    3. Result preserves fine textures while keeping new lighting

    Formula:
    high_freq = original - gaussian_blur(original, radius)
    result = relighted + (high_freq * strength)  # Linear mode
    result = overlay(relighted, high_freq)       # Overlay mode

Usage:
    - Restore details lost during relighting (IC-Light, etc.)
    - Transfer texture details to processed images
    - Preserve material textures after color/light edits

Version: 1.1.0
Created: 2025-01-20
Updated: 2025-01-20 - Added dual image input for detail transfer
"""

import torch
import torch.nn.functional as F
from comfy_api.latest import io


# ============================================================================
# GAUSSIAN BLUR
# ============================================================================

def gaussian_kernel_1d(sigma: float, kernel_size: int, device: torch.device) -> torch.Tensor:
    """
    Create a 1D Gaussian kernel.

    Args:
        sigma: Standard deviation of the Gaussian
        kernel_size: Size of the kernel (should be odd)
        device: Torch device

    Returns:
        1D Gaussian kernel tensor
    """
    x = torch.arange(kernel_size, device=device, dtype=torch.float32) - (kernel_size - 1) / 2
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel


def gaussian_blur(image: torch.Tensor, radius: float) -> torch.Tensor:
    """
    Apply Gaussian blur using separable convolution for efficiency.

    Args:
        image: Input image tensor (B, H, W, C) in range [0, 1]
        radius: Blur radius (sigma = radius / 2)

    Returns:
        Blurred image tensor (B, H, W, C)
    """
    if radius <= 0:
        return image

    # Convert to BCHW
    bchw = image.movedim(-1, 1).contiguous()
    B, C, H, W = bchw.shape

    # Calculate kernel parameters
    sigma = radius / 2.0
    kernel_size = int(2 * radius + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(3, min(kernel_size, 31))  # Clamp to reasonable range

    # Create 1D Gaussian kernel
    kernel_1d = gaussian_kernel_1d(sigma, kernel_size, bchw.device)

    # Reshape for separable convolution
    kernel_h = kernel_1d.view(1, 1, kernel_size, 1).repeat(C, 1, 1, 1)
    kernel_w = kernel_1d.view(1, 1, 1, kernel_size).repeat(C, 1, 1, 1)

    # Padding
    pad_h = kernel_size // 2
    pad_w = kernel_size // 2

    # Apply separable blur (horizontal then vertical)
    blurred = F.pad(bchw, (pad_w, pad_w, 0, 0), mode='reflect')
    blurred = F.conv2d(blurred, kernel_w, groups=C)

    blurred = F.pad(blurred, (0, 0, pad_h, pad_h), mode='reflect')
    blurred = F.conv2d(blurred, kernel_h, groups=C)

    # Convert back to BHWC
    return blurred.movedim(1, -1).clamp(0, 1)


# ============================================================================
# BLEND MODES
# ============================================================================

def blend_linear(base: torch.Tensor, detail: torch.Tensor, strength: float) -> torch.Tensor:
    """
    Simple linear blend: base + detail * strength

    Args:
        base: Base image (B, H, W, C)
        detail: High frequency detail layer (B, H, W, C)
        strength: Blend strength

    Returns:
        Blended image
    """
    return (base + detail * strength).clamp(0, 1)


def blend_overlay(base: torch.Tensor, detail: torch.Tensor, strength: float) -> torch.Tensor:
    """
    Overlay blend mode (like Photoshop).
    Increases contrast in midtones while preserving shadows and highlights.

    Args:
        base: Base image (B, H, W, C)
        detail: High frequency detail layer, normalized to 0.5 center
        strength: Blend strength

    Returns:
        Blended image
    """
    # Normalize detail to 0-1 range centered at 0.5
    detail_norm = (detail + 1.0) / 2.0  # From [-1, 1] to [0, 1]

    # Overlay formula
    result = torch.where(
        base < 0.5,
        2 * base * detail_norm,
        1 - 2 * (1 - base) * (1 - detail_norm)
    )

    # Blend with strength
    return (base * (1 - strength) + result * strength).clamp(0, 1)


def blend_soft_light(base: torch.Tensor, detail: torch.Tensor, strength: float) -> torch.Tensor:
    """
    Soft Light blend mode (gentler than Overlay).
    Good for subtle detail enhancement.

    Args:
        base: Base image (B, H, W, C)
        detail: High frequency detail layer, normalized to 0.5 center
        strength: Blend strength

    Returns:
        Blended image
    """
    # Normalize detail to 0-1 range centered at 0.5
    detail_norm = (detail + 1.0) / 2.0  # From [-1, 1] to [0, 1]

    # Soft light formula (W3C version)
    result = torch.where(
        detail_norm <= 0.5,
        base - (1 - 2 * detail_norm) * base * (1 - base),
        base + (2 * detail_norm - 1) * (torch.where(
            base <= 0.25,
            ((16 * base - 12) * base + 4) * base,
            torch.sqrt(base)
        ) - base)
    )

    # Blend with strength
    return (base * (1 - strength) + result * strength).clamp(0, 1)


def blend_vivid_light(base: torch.Tensor, detail: torch.Tensor, strength: float) -> torch.Tensor:
    """
    Vivid Light blend mode (stronger contrast effect).
    Good for dramatic detail enhancement.

    Args:
        base: Base image (B, H, W, C)
        detail: High frequency detail layer
        strength: Blend strength

    Returns:
        Blended image
    """
    # Normalize detail to 0-1 range centered at 0.5
    detail_norm = (detail + 1.0) / 2.0

    # Vivid light = color burn for dark, color dodge for light
    eps = 1e-6

    # Color burn: 1 - (1 - base) / detail
    burn = torch.where(
        detail_norm > eps,
        1 - (1 - base) / (2 * detail_norm + eps),
        torch.zeros_like(base)
    )

    # Color dodge: base / (1 - detail)
    dodge = torch.where(
        detail_norm < 1 - eps,
        base / (2 * (1 - detail_norm) + eps),
        torch.ones_like(base)
    )

    result = torch.where(detail_norm <= 0.5, burn, dodge)

    # Blend with strength
    return (base * (1 - strength) + result.clamp(0, 1) * strength).clamp(0, 1)


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def highpass_transfer(
    original: torch.Tensor,
    target: torch.Tensor,
    radius: float = 3.0,
    strength: float = 1.0,
    blend_mode: str = "linear"
) -> torch.Tensor:
    """
    Extract high-pass details from original and transfer to target image.

    Args:
        original: Source image to extract details from (B, H, W, C)
        target: Target image to apply details to (B, H, W, C)
        radius: Gaussian blur radius for extracting high frequency
        strength: Blend strength (0.0 = no effect, 1.0 = full strength)
        blend_mode: One of "linear", "overlay", "soft_light", "vivid_light"

    Returns:
        Target image with details from original (B, H, W, C)
    """
    # Extract low frequency from original (blurred)
    low_freq = gaussian_blur(original, radius)

    # Extract high frequency (details) from original
    high_freq = original - low_freq  # Range approximately [-1, 1]

    # Apply blend mode to TARGET image
    if blend_mode == "linear":
        result = blend_linear(target, high_freq, strength)
    elif blend_mode == "overlay":
        result = blend_overlay(target, high_freq, strength)
    elif blend_mode == "soft_light":
        result = blend_soft_light(target, high_freq, strength)
    elif blend_mode == "vivid_light":
        result = blend_vivid_light(target, high_freq, strength)
    else:
        result = blend_linear(target, high_freq, strength)

    return result


# ============================================================================
# COMFYUI NODE CLASS
# ============================================================================

class ArchAi3D_HighPass_Blend(io.ComfyNode):
    """
    High-Pass Detail Transfer for relighting/processing workflows.

    Extracts high frequency details from the ORIGINAL image and transfers
    them to the RELIGHTED/PROCESSED image. This restores fine details
    (textures, edges, material patterns) that may be lost during:
    - IC-Light relighting
    - Style transfer
    - Color grading
    - Any image processing that smooths details

    Workflow:
    Original Image --[extract details]--> High Frequency Layer
    Relighted Image + High Frequency --[blend]--> Final Result

    Blend Modes:
    - Linear: Direct addition (result = relighted + detail * strength)
    - Overlay: Photoshop-style, increases midtone contrast
    - Soft Light: Gentler effect, good for subtle enhancement
    - Vivid Light: Stronger contrast, dramatic effect
    """

    @classmethod
    def define_schema(cls):
        """Define node schema for ComfyUI."""
        return io.Schema(
            node_id="ArchAi3D_HighPass_Blend",
            category="ArchAi3d/Utils",
            inputs=[
                io.Image.Input(
                    "original",
                    tooltip="Original image to extract details FROM"
                ),
                io.Image.Input(
                    "relighted",
                    tooltip="Relighted/processed image to apply details TO"
                ),
                io.Float.Input(
                    "radius",
                    default=3.0,
                    min=0.5,
                    max=20.0,
                    step=0.5,
                    tooltip="Blur radius for high-pass filter (higher = extract larger details)"
                ),
                io.Float.Input(
                    "strength",
                    default=1.0,
                    min=0.0,
                    max=3.0,
                    step=0.1,
                    tooltip="Blend strength (0 = no effect, 1 = normal, >1 = stronger)"
                ),
                io.Combo.Input(
                    "blend_mode",
                    options=["linear", "overlay", "soft_light", "vivid_light"],
                    default="linear",
                    tooltip="Blend mode for combining detail layer"
                ),
            ],
            outputs=[
                io.Image.Output(
                    "result",
                    tooltip="Relighted image with restored details"
                ),
                io.Image.Output(
                    "high_freq",
                    tooltip="Extracted high frequency detail layer (for debugging/manual use)"
                ),
            ],
        )

    @classmethod
    def execute(cls, original, relighted, radius, strength, blend_mode) -> io.NodeOutput:
        """
        Execute high-pass detail transfer.

        Args:
            original: Original image to extract details from (B, H, W, C)
            relighted: Relighted image to apply details to (B, H, W, C)
            radius: Blur radius for high-pass extraction
            strength: Blend strength
            blend_mode: Blend mode selection

        Returns:
            io.NodeOutput with enhanced image and detail layer
        """
        # Ensure float tensors
        original = original.float()
        relighted = relighted.float()

        # Handle size mismatch - resize original's details to match relighted
        if original.shape[1:3] != relighted.shape[1:3]:
            # Resize original to match relighted dimensions
            orig_bchw = original.movedim(-1, 1)
            target_h, target_w = relighted.shape[1], relighted.shape[2]
            orig_resized = F.interpolate(orig_bchw, size=(target_h, target_w), mode='bilinear', align_corners=False)
            original = orig_resized.movedim(1, -1)
            print(f"  Resized original from {orig_bchw.shape[2:]} to {(target_h, target_w)}")

        # Extract high frequency from original for output
        low_freq = gaussian_blur(original, float(radius))
        high_freq = original - low_freq

        # Apply detail transfer
        result = highpass_transfer(
            original,
            relighted,
            radius=float(radius),
            strength=float(strength),
            blend_mode=str(blend_mode)
        )

        # Normalize high_freq for visualization (shift from [-1,1] to [0,1])
        high_freq_vis = ((high_freq + 1.0) / 2.0).clamp(0, 1)

        # Debug output
        print("\n" + "="*70)
        print("High-Pass Detail Transfer - v1.1.0")
        print("="*70)
        print(f"Original shape: {original.shape}")
        print(f"Relighted shape: {relighted.shape}")
        print(f"Blur radius: {radius}")
        print(f"Strength: {strength}")
        print(f"Blend mode: {blend_mode}")
        print(f"Detail range: [{high_freq.min():.3f}, {high_freq.max():.3f}]")
        print("="*70 + "\n")

        return io.NodeOutput(result, high_freq_vis)


# ============================================================================
# EXPORTS
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "ArchAi3D_HighPass_Blend": ArchAi3D_HighPass_Blend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_HighPass_Blend": "High-Pass Detail Transfer",
}
