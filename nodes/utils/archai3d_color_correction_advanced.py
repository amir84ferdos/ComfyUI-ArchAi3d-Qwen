# -*- coding: utf-8 -*-
"""
ArchAi3D Advanced Color Correction Node
Author: Amir Ferdos (ArchAi3d)
Email: Amir84ferdos@gmail.com
LinkedIn: https://www.linkedin.com/in/archai3d/

Description:
    Professional-grade color correction with chroma matching.
    Based on advanced techniques from qi_refedit by wallen0322.

    Features:
    - YCbCr chroma statistics matching
    - Adaptive highlight/lowlight attenuation with smoothstep
    - Brightness-dependent processing
    - Ultra-conservative corrections (prevents artifacts)
    - Optional reference image matching

Usage:
    Pre-processing node before Qwen image editing.
    More sophisticated than basic BT.709 node.
    Use when you need professional-grade color matching.

Version: 1.2.0
Created: 2025-10-17
Updated: 2025-10-18 - Fixed autocolor using lowpass reference (strength now works correctly)
Based on: qi_refedit.py by wallen0322
"""

import torch
import torch.nn.functional as F
import numpy as np
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override


# ============================================================================
# COLOR SPACE TRANSFORMS
# ============================================================================

def srgb_to_linear(t: torch.Tensor) -> torch.Tensor:
    """
    Convert sRGB to linear RGB using ITU-R BT.709 standard.

    Args:
        t: torch.Tensor in range [0, 1], sRGB color space

    Returns:
        torch.Tensor in range [0, 1], linear RGB
    """
    t = t.clamp(0.0, 1.0)
    return torch.where(t <= 0.04045, t / 12.92, ((t + 0.055)/1.055) ** 2.4)


def linear_to_srgb(t: torch.Tensor) -> torch.Tensor:
    """
    Convert linear RGB to sRGB using ITU-R BT.709 standard.

    Args:
        t: torch.Tensor in range [0, 1], linear RGB

    Returns:
        torch.Tensor in range [0, 1], sRGB color space
    """
    t = t.clamp(0.0, 1.0)
    return torch.where(t <= 0.0031308, t*12.92, 1.055*torch.pow(t, 1.0/2.4) - 0.055)


def smoothstep(a: float, b: float, x: torch.Tensor) -> torch.Tensor:
    """
    Smooth Hermite interpolation between a and b.

    Args:
        a: Lower bound
        b: Upper bound
        x: Input values

    Returns:
        Smoothly interpolated values [0, 1]
    """
    t = (x - a) / max(1e-6, (b - a))
    t = t.clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ============================================================================
# CHOLESKY DECOMPOSITION FOR 2x2 COVARIANCE MATCHING
# ============================================================================

def cholesky_2x2(a, b, c, eps=1e-8):
    """
    Cholesky decomposition for 2x2 covariance matrix.

    Matrix: [[a, b],
             [b, c]]

    Returns L and L^-1 where L is lower triangular.

    Args:
        a, b, c: Covariance matrix elements
        eps: Small value for numerical stability

    Returns:
        Tuple of (l11, l21, l22, inv_l11, inv_l21, inv_l22)
    """
    a = torch.clamp(a, min=eps)
    l11 = torch.sqrt(a)
    l21 = b / l11
    t = torch.clamp(c - l21*l21, min=eps)
    l22 = torch.sqrt(t)

    inv_l11 = 1.0 / l11
    inv_l22 = 1.0 / l22
    inv_l21 = -l21 * inv_l11 * inv_l22

    return l11, l21, l22, inv_l11, inv_l21, inv_l22


# ============================================================================
# CHROMA STATISTICS MATCHING
# ============================================================================

def apply_chroma_stats(Cbx, Crx, Cbr, Crr, Yx):
    """
    Match chroma statistics from source (x) to reference (r).

    Uses professional-grade techniques:
    - Adaptive masking based on luminance and chroma magnitude
    - Covariance matching with Cholesky decomposition
    - Brightness-dependent attenuation
    - Ultra-conservative clamping (1.5% max change)

    Args:
        Cbx, Crx: Source Cb/Cr channels (B, 1, H, W)
        Cbr, Crr: Reference Cb/Cr channels (B, 1, H, W)
        Yx: Source luminance (B, 1, H, W)

    Returns:
        Tuple of (transform_matrix, mean_shift, source_mean)
    """
    B, _, H, W = Yx.shape

    # Calculate luminance-based mask (focus on middle tones)
    yv = Yx.reshape(B, 1, H*W)
    lo = torch.quantile(yv, 0.06, dim=-1, keepdim=True).reshape(B, 1, 1, 1)
    hi = torch.quantile(yv, 0.94, dim=-1, keepdim=True).reshape(B, 1, 1, 1)
    mask = (Yx >= lo) & (Yx <= hi)

    # Adaptive chroma threshold (brightness-dependent)
    meanY = Yx.mean(dim=(2,3), keepdim=True)
    base_thr = 1.6e-3
    hi_thr = 2.0e-3
    thr = torch.full_like(Yx, base_thr)
    thr = torch.where(meanY > 0.72, torch.full_like(thr, hi_thr), thr)

    # Chroma magnitude mask (only consider pixels with enough color)
    chroma2 = Cbx*Cbx + Crx*Crx
    mask = mask & (chroma2 > thr)

    m = mask.float()
    denom = m.sum(dim=(2,3), keepdim=True).clamp_min(1.0)

    def compute_stats(Cb, Cr):
        """Compute mean and covariance of chroma channels."""
        mu_cb = (Cb * m).sum(dim=(2,3), keepdim=True) / denom
        mu_cr = (Cr * m).sum(dim=(2,3), keepdim=True) / denom

        dcb = (Cb - mu_cb) * m
        dcr = (Cr - mu_cr) * m

        # Covariance matrix elements
        a = (dcb * (Cb - mu_cb)).sum(dim=(2,3), keepdim=True) / denom + 1e-6
        c = (dcr * (Cr - mu_cr)).sum(dim=(2,3), keepdim=True) / denom + 1e-6
        b = (dcb * (Cr - mu_cr)).sum(dim=(2,3), keepdim=True) / denom

        return mu_cb, mu_cr, a.squeeze(), b.squeeze(), c.squeeze()

    # Compute statistics for both source and reference
    mu_x_cb, mu_x_cr, ax, bx, cx = compute_stats(Cbx, Crx)
    mu_r_cb, mu_r_cr, ar, br, cr = compute_stats(Cbr, Crr)

    # Cholesky decomposition for covariance matching
    l11x, l21x, l22x, inv11x, inv21x, inv22x = cholesky_2x2(ax, bx, cx)
    l11r, l21r, l22r, _, _, _ = cholesky_2x2(ar, br, cr)

    # Compute transform matrix T = L_ref @ L_src^-1
    t00 = l11r * inv11x + 0.0 * inv21x
    t01 = l11r * 0.0    + 0.0 * inv22x
    t10 = l21r * inv11x + l22r * inv21x
    t11 = l21r * 0.0    + l22r * inv22x

    # CRITICAL: Ultra-conservative clamping (prevents artifacts)
    t00 = t00.clamp(0.985, 1.015)  # Only ±1.5% change
    t11 = t11.clamp(0.985, 1.015)
    t01 = t01.clamp(-0.015, 0.015)
    t10 = t10.clamp(-0.015, 0.015)

    # Mean shift with adaptive limits
    dmu_cb = (mu_r_cb - mu_x_cb)
    dmu_cr = (mu_r_cr - mu_x_cr)

    base_mu = 0.002
    strict_mu = 0.0015
    mu_lim = torch.where(meanY > 0.72,
                         torch.full_like(meanY, strict_mu),
                         torch.full_like(meanY, base_mu))

    dmu_cb = dmu_cb.clamp(-mu_lim, mu_lim)
    dmu_cr = dmu_cr.clamp(-mu_lim, mu_lim)

    # Brightness-dependent attenuation (reduce correction in very bright scenes)
    delta = torch.clamp(meanY - 0.72, min=0.0, max=0.20)
    k = 1.0 - delta * 0.55  # Up to 11% reduction for bright scenes
    k = torch.clamp(k, 0.75, 1.0)

    # Apply attenuation
    t00 = 1.0 + (t00 - 1.0) * k
    t11 = 1.0 + (t11 - 1.0) * k
    t01 = t01 * k
    t10 = t10 * k
    dmu_cb = dmu_cb * k
    dmu_cr = dmu_cr * k

    return (t00, t01, t10, t11), (dmu_cb, dmu_cr), (mu_x_cb, mu_x_cr)


# ============================================================================
# LOWPASS REFERENCE GENERATION
# ============================================================================

def lowpass_ref(bhwc: torch.Tensor, size: int = 64) -> torch.Tensor:
    """
    Create a lowpass-filtered reference by downsampling and upsampling.

    This creates a blurred version of the image that can be used as a
    reference for color correction. The downsampled version has smoothed
    color statistics.

    Args:
        bhwc: Input image (B, H, W, C) in range [0, 1]
        size: Intermediate downsampling size (default: 64x64)

    Returns:
        Lowpass-filtered image (B, H, W, C) in range [0, 1]
    """
    # Convert to BCHW
    bchw = bhwc.movedim(-1, 1).contiguous()
    B, C, H, W = bchw.shape

    # Downsample to low resolution (creates smooth color statistics)
    bchw_low = F.interpolate(bchw, size=(size, size), mode="area")

    # Upsample back to original resolution
    bchw_restored = F.interpolate(bchw_low, size=(H, W), mode="bilinear", align_corners=False)

    # Convert back to BHWC
    return bchw_restored.movedim(1, -1).clamp(0, 1)


# ============================================================================
# MAIN COLOR CORRECTION FUNCTION
# ============================================================================

def color_correct_advanced(x_bhwc: torch.Tensor, ref_bhwc: torch.Tensor = None, mix: float = 0.97, saturation: float = 1.0) -> torch.Tensor:
    """
    Advanced color correction with chroma matching.

    Professional-grade algorithm:
    1. Convert to linear RGB
    2. Transform to YCbCr color space
    3. Match chroma statistics between source and reference
    4. Apply adaptive highlight/lowlight attenuation
    5. Convert back to sRGB

    Args:
        x_bhwc: Source image (B, H, W, C) in sRGB [0, 1]
        ref_bhwc: Reference image (B, H, W, C) in sRGB [0, 1], or None to use self
        mix: Correction strength (0.0 = none, 1.0 = full)
        saturation: Saturation adjustment (0.0 = grayscale, 1.0 = original, 2.0 = 2x saturation)

    Returns:
        Corrected image (B, H, W, C) in sRGB [0, 1]
    """
    # Convert to BCHW format
    x = x_bhwc.movedim(-1, 1).contiguous()

    if ref_bhwc is None:
        ref_bhwc = x_bhwc  # Self-correction mode

    r = ref_bhwc.movedim(-1, 1).contiguous()

    # Convert to linear color space
    x_lin = srgb_to_linear(x)
    r_lin = srgb_to_linear(r)

    # Extract RGB channels
    R, G, B = x_lin[:, 0:1], x_lin[:, 1:2], x_lin[:, 2:3]
    Rr, Gr, Br = r_lin[:, 0:1], r_lin[:, 1:2], r_lin[:, 2:3]

    # Convert to YCbCr (ITU-R BT.709)
    Yx = 0.2126*R + 0.7152*G + 0.0722*B
    Yr = 0.2126*Rr + 0.7152*Gr + 0.0722*Br

    cb_s = 0.5 / (1.0 - 0.0722)
    cr_s = 0.5 / (1.0 - 0.2126)

    Cbx = (B - Yx) * cb_s
    Crx = (R - Yx) * cr_s
    Cbr = (Br - Yr) * cb_s
    Crr = (Rr - Yr) * cr_s

    # Apply chroma statistics matching
    (t00, t01, t10, t11), (dmu_cb, dmu_cr), (mu_x_cb, mu_x_cr) = apply_chroma_stats(
        Cbx, Crx, Cbr, Crr, Yx
    )

    # Transform chroma channels
    cb = Cbx - mu_x_cb
    cr = Crx - mu_x_cr

    Cb_aligned = t00*cb + t01*cr + (mu_x_cb + dmu_cb)
    Cr_aligned = t10*cb + t11*cr + (mu_x_cr + dmu_cr)

    # Convert back to RGB
    inv_cb = 1.0 / cb_s
    inv_cr = 1.0 / cr_s

    R2 = Yx + Cr_aligned * inv_cr
    B2 = Yx + Cb_aligned * inv_cb
    G2 = (Yx - 0.2126*R2 - 0.0722*B2) / 0.7152

    aligned = torch.cat([R2, G2, B2], dim=1).clamp(0, 1)

    # Adaptive blending with smoothstep attenuation
    # Reduce correction in very bright and very dark areas
    w_hi = 1.0 - smoothstep(0.74, 0.93, Yx)  # Fade in highlights
    w_lo = smoothstep(0.05, 0.12, Yx)        # Fade in shadows
    w = (0.9 * mix * w_hi * w_lo).clamp(0.0, 1.0)

    # Blend corrected and original
    out_lin = w * aligned + (1.0 - w) * x_lin

    # Apply saturation adjustment (if not 1.0)
    if saturation != 1.0:
        # Calculate luminance in linear space
        Y_out = 0.2126*out_lin[:, 0:1] + 0.7152*out_lin[:, 1:2] + 0.0722*out_lin[:, 2:3]

        # Separate luminance and chroma
        # Saturation adjustment: interpolate between grayscale and full color
        # sat = 0.0: pure grayscale (Y, Y, Y)
        # sat = 1.0: original colors
        # sat > 1.0: enhanced saturation
        grayscale = Y_out.repeat(1, 3, 1, 1)

        if saturation < 1.0:
            # Desaturate: blend towards grayscale
            out_lin = out_lin * saturation + grayscale * (1.0 - saturation)
        else:
            # Enhance saturation: push away from grayscale
            sat_factor = saturation
            out_lin = grayscale + (out_lin - grayscale) * sat_factor

        out_lin = out_lin.clamp(0, 1)

    # Convert back to sRGB
    out = linear_to_srgb(out_lin).movedim(1, -1)

    return out.clamp(0, 1)


# ============================================================================
# TENSOR CONVERSION HELPERS
# ============================================================================

def ensure_bhwc(tensor):
    """
    Convert tensor to BHWC format.

    Args:
        tensor: Input tensor (various formats)

    Returns:
        torch.Tensor (B, H, W, C) in range [0, 1]
    """
    if len(tensor.shape) == 4:
        if tensor.shape[-1] in (1, 3, 4):
            # Already BHWC
            pass
        elif tensor.shape[1] in (1, 3, 4):
            # BCHW -> BHWC
            tensor = tensor.movedim(1, -1)

    # Ensure RGB (3 channels)
    if tensor.shape[-1] == 1:
        tensor = tensor.repeat(1, 1, 1, 3)
    elif tensor.shape[-1] == 4:
        tensor = tensor[..., :3]

    return tensor.float().clamp(0, 1)


# ============================================================================
# COMFYUI NODE CLASS
# ============================================================================

class ArchAi3D_Color_Correction_Advanced(io.ComfyNode):
    """
    Advanced color correction with professional-grade chroma matching.

    Based on techniques from qi_refedit by wallen0322.

    Features:
    - YCbCr chroma statistics matching with covariance alignment
    - Adaptive highlight/lowlight attenuation using smoothstep
    - Brightness-dependent processing (conservative in bright scenes)
    - Ultra-conservative corrections (max 1.5% transform change)
    - Optional reference image for color matching

    Use cases:
    - Pre-processing before Qwen for consistent colors
    - Matching generated images to reference colors
    - Professional color grading workflows
    """

    @classmethod
    def define_schema(cls):
        """Define node schema for ComfyUI."""
        return io.Schema(
            node_id="ArchAi3D_Color_Correction_Advanced",
            category="ArchAi3d/Utils",
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="Input image to color correct"
                ),
                io.Float.Input(
                    "strength",
                    default=0.97,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Correction strength (0.0 = no correction, 1.0 = full correction)"
                ),
                io.Float.Input(
                    "saturation",
                    default=1.0,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    tooltip="Saturation adjustment (0.0 = grayscale, 1.0 = original, 2.0 = 2x saturation)"
                ),
            ],
            outputs=[
                io.Image.Output(
                    "corrected_image",
                    tooltip="Color-corrected image with professional chroma matching"
                ),
            ],
        )

    @classmethod
    def execute(cls, image, strength, saturation) -> io.NodeOutput:
        """
        Execute advanced color correction.

        Args:
            image: ComfyUI image tensor (B, H, W, C) in sRGB [0, 1]
            strength: Correction strength (0.0-1.0)
            saturation: Saturation adjustment (0.0-2.0)

        Returns:
            io.NodeOutput with corrected image tensor
        """
        # Ensure BHWC format
        img_bhwc = ensure_bhwc(image)

        # Create lowpass-filtered reference (64x64 downsampled)
        # This provides smoothed color statistics for the correction
        ref_bhwc = lowpass_ref(img_bhwc, size=64)

        # Apply advanced color correction using lowpass reference
        img_corrected = color_correct_advanced(
            img_bhwc,
            ref_bhwc=ref_bhwc,  # Use lowpass reference
            mix=float(strength),
            saturation=float(saturation)
        )

        # Debug output
        print("\n" + "="*70)
        print("🎨 ArchAi3D Advanced Color Correction - v1.2.0")
        print("="*70)
        print(f"Input shape: {image.shape}")
        print(f"Output shape: {img_corrected.shape}")
        print(f"Correction strength: {strength:.2f}")
        print(f"Saturation: {saturation:.2f}")
        if saturation < 1.0:
            print(f"  → Desaturated ({(1.0-saturation)*100:.0f}% towards grayscale)")
        elif saturation > 1.0:
            print(f"  → Enhanced ({(saturation-1.0)*100:.0f}% more saturation)")
        print(f"\nTechniques applied:")
        print(f"  ✓ Lowpass reference generation (64x64 downsampled)")
        print(f"  ✓ YCbCr chroma statistics matching")
        print(f"  ✓ 2x2 Cholesky covariance alignment")
        print(f"  ✓ Adaptive highlight attenuation (smoothstep 0.74-0.93)")
        print(f"  ✓ Adaptive lowlight attenuation (smoothstep 0.05-0.12)")
        print(f"  ✓ Brightness-dependent processing")
        print(f"  ✓ Ultra-conservative clamping (±1.5% max)")
        if saturation != 1.0:
            print(f"  ✓ Manual saturation adjustment")
        print("="*70 + "\n")

        return io.NodeOutput(img_corrected)


# ============================================================================
# EXTENSION REGISTRATION
# ============================================================================

class ColorCorrectionAdvancedExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [ArchAi3D_Color_Correction_Advanced]


async def comfy_entrypoint():
    return ColorCorrectionAdvancedExtension()
