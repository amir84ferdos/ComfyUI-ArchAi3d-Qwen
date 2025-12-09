# -*- coding: utf-8 -*-
"""
ArchAi3D Smart Tile Calculator

Calculates optimal tile size, upscale factor, and blur/padding values
for Ultimate SD Upscale to minimize wasted overlap and processing costs.

Author: Amir Ferdos (ArchAi3d)
Email: Amir84ferdos@gmail.com
LinkedIn: https://www.linkedin.com/in/archai3d/
GitHub: https://github.com/amir84ferdos

Version: 2.5.0 - Changed tile divisor from 8 to 32 for VAE compatibility
              v2.4.0: Added SMART_TILE_BUNDLE output for one-wire connections
License: Dual License (Free for personal use, Commercial license required for business use)
"""

import math
import torch
import torch.nn.functional as F


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def find_optimal_tile_size(aspect_ratio: float, target_mp: float = 2.0, divisor: int = 32) -> tuple:
    """
    Find tile dimensions matching aspect ratio, ~target_mp, divisible by divisor.

    Args:
        aspect_ratio: Width / Height ratio of the image
        target_mp: Target megapixels per tile (default 2.0 for z-image-turbo)
        divisor: Tile dimensions must be divisible by this (32 for VAE compatibility)

    Returns:
        Tuple of (tile_width, tile_height)
    """
    target_area = target_mp * 1_000_000

    # Calculate dimensions preserving aspect ratio
    # tile_w * tile_h = target_area
    # tile_w / tile_h = aspect_ratio
    # Therefore: tile_h = sqrt(target_area / aspect_ratio)
    tile_h = math.sqrt(target_area / aspect_ratio)
    tile_w = tile_h * aspect_ratio

    # Round to nearest divisor (32 for VAE/latent compatibility)
    tile_h = round(tile_h / divisor) * divisor
    tile_w = round(tile_w / divisor) * divisor

    # Ensure minimum tile size
    tile_w = max(64, int(tile_w))
    tile_h = max(64, int(tile_h))

    return tile_w, tile_h


def scale_overlap_params(tile_size: int, base_tile: int = 512, mode: str = "Proportional") -> dict:
    """
    Scale blur/padding proportionally with tile size, respecting USDU limits.

    Reference: USDU defaults are tuned for 512x512 tiles.
    - mask_blur: 8 (1.56% of 512)
    - tile_padding: 32 (6.25% of 512)
    - seam_fix_width: 64 (12.5% of 512)
    - seam_fix_mask_blur: 8 (1.56% of 512)
    - seam_fix_padding: 16 (3.125% of 512)

    Args:
        tile_size: The calculated tile size (use smaller dimension)
        base_tile: Reference tile size (512 for USDU defaults)
        mode: "Fixed" for USDU defaults, "Proportional" for scaled values

    Returns:
        Dict with mask_blur, tile_padding, seam_fix_width, seam_fix_mask_blur, seam_fix_padding
    """
    if mode == "Fixed":
        return {
            "mask_blur": 8,
            "tile_padding": 32,
            "seam_fix_width": 64,
            "seam_fix_mask_blur": 8,
            "seam_fix_padding": 16
        }

    # Proportional scaling
    scale = tile_size / base_tile

    # Scale proportionally, but respect USDU max limits
    mask_blur = min(64, max(4, int(8 * scale)))           # max 64, min 4
    tile_padding = max(8, int(32 * scale))                 # min 8
    seam_fix_width = max(8, int(64 * scale))               # min 8
    seam_fix_mask_blur = min(64, max(4, int(8 * scale)))  # max 64, min 4
    seam_fix_padding = max(8, int(16 * scale))             # min 8

    # Round padding/width to step size 8
    tile_padding = round(tile_padding / 8) * 8
    seam_fix_width = round(seam_fix_width / 8) * 8
    seam_fix_padding = round(seam_fix_padding / 8) * 8

    return {
        "mask_blur": mask_blur,
        "tile_padding": tile_padding,
        "seam_fix_width": seam_fix_width,
        "seam_fix_mask_blur": seam_fix_mask_blur,
        "seam_fix_padding": seam_fix_padding
    }


def calculate_efficiency(out_w: int, out_h: int, tile_w: int, tile_h: int,
                          tile_padding: int, mask_blur: int, seam_fix_width: int) -> tuple:
    """
    Calculate efficiency and tile count for given parameters.

    Args:
        out_w, out_h: Output image dimensions
        tile_w, tile_h: Tile dimensions
        tile_padding, mask_blur, seam_fix_width: Overlap parameters

    Returns:
        Tuple of (efficiency, tiles_x, tiles_y, overlap_waste_mp)
    """
    # Total overlap per edge = padding (provides context, overlaps between tiles)
    # Note: mask_blur and seam_fix happen within the tile, not extra overlap
    overlap = tile_padding

    # Effective tile size (step size between tiles)
    effective_tile_w = tile_w - overlap
    effective_tile_h = tile_h - overlap

    # Ensure positive effective size
    effective_tile_w = max(64, effective_tile_w)
    effective_tile_h = max(64, effective_tile_h)

    # Number of tiles needed
    tiles_x = max(1, math.ceil(out_w / effective_tile_w))
    tiles_y = max(1, math.ceil(out_h / effective_tile_h))

    # Total processed area (all tiles at full size)
    processed_area = tiles_x * tiles_y * tile_w * tile_h

    # Useful area (final output image)
    useful_area = out_w * out_h

    # Efficiency = useful / processed (higher = less waste)
    efficiency = useful_area / processed_area if processed_area > 0 else 0

    # Overlap waste in MP
    overlap_waste_mp = (processed_area - useful_area) / 1_000_000

    return efficiency, tiles_x, tiles_y, overlap_waste_mp


def find_optimal_upscale(input_w: int, input_h: int, tile_w: int, tile_h: int,
                          target_upscale: float, tolerance: float,
                          tile_padding: int, mask_blur: int, seam_fix_width: int) -> tuple:
    """
    Find upscale factor that minimizes overlap waste within tolerance.

    Args:
        input_w, input_h: Input image dimensions
        tile_w, tile_h: Tile dimensions
        target_upscale: User's desired upscale factor
        tolerance: How much upscale can deviate (e.g., 0.3 = ±30%)
        tile_padding, mask_blur, seam_fix_width: Overlap parameters

    Returns:
        Tuple of (best_upscale, best_efficiency, tiles_x, tiles_y)
    """
    best_upscale = target_upscale
    best_efficiency = 0
    best_tiles_x = 1
    best_tiles_y = 1

    # Search range
    min_upscale = max(0.05, target_upscale * (1 - tolerance))
    max_upscale = min(4.0, target_upscale * (1 + tolerance))

    # Step through possible upscales (0.05 increments as per USDU)
    upscale = min_upscale
    while upscale <= max_upscale:
        out_w = int(input_w * upscale)
        out_h = int(input_h * upscale)

        efficiency, tiles_x, tiles_y, _ = calculate_efficiency(
            out_w, out_h, tile_w, tile_h,
            tile_padding, mask_blur, seam_fix_width
        )

        if efficiency > best_efficiency:
            best_efficiency = efficiency
            best_upscale = upscale
            best_tiles_x = tiles_x
            best_tiles_y = tiles_y

        upscale += 0.05

    return round(best_upscale, 2), best_efficiency, best_tiles_x, best_tiles_y


def get_overlap_params_for_tile(tile_w: int, tile_h: int, scaling_mode: str, overlap_scale: float) -> dict:
    """Calculate overlap parameters for a given tile size."""
    min_tile_dim = min(tile_w, tile_h)
    overlap_params = scale_overlap_params(min_tile_dim, base_tile=512, mode=scaling_mode)

    # Apply overlap_scale
    mask_blur = max(1, int(overlap_params["mask_blur"] * overlap_scale))
    tile_padding = max(8, int(overlap_params["tile_padding"] * overlap_scale))
    seam_fix_width = max(8, int(overlap_params["seam_fix_width"] * overlap_scale))
    seam_fix_mask_blur = max(1, int(overlap_params["seam_fix_mask_blur"] * overlap_scale))
    seam_fix_padding = max(8, int(overlap_params["seam_fix_padding"] * overlap_scale))

    # Round to step size 8
    tile_padding = max(8, round(tile_padding / 8) * 8)
    seam_fix_width = max(8, round(seam_fix_width / 8) * 8)
    seam_fix_padding = max(8, round(seam_fix_padding / 8) * 8)

    return {
        "mask_blur": mask_blur,
        "tile_padding": tile_padding,
        "seam_fix_width": seam_fix_width,
        "seam_fix_mask_blur": seam_fix_mask_blur,
        "seam_fix_padding": seam_fix_padding,
    }


def find_perfect_alignments(input_w: int, input_h: int, aspect_ratio: float,
                             target_upscale: float, tolerance: float,
                             min_tile_mp: float, max_tile_mp: float,
                             scaling_mode: str, overlap_scale: float) -> list:
    """
    Phase 1: Find ALL configurations where tiles fit output PERFECTLY.

    Mathematical formula for perfect fit:
    output_w = tile_w + (tiles_x - 1) × step_w
    step_w = tile_w - padding

    This function calculates the EXACT upscale that gives perfect alignment
    for each tile size and tile count combination.

    Args:
        input_w, input_h: Input image dimensions
        aspect_ratio: Width / Height ratio
        target_upscale: User's desired upscale factor
        tolerance: How much upscale can deviate (e.g., 0.5 = ±50%)
        min_tile_mp, max_tile_mp: Tile size range to search
        scaling_mode, overlap_scale: Overlap parameters

    Returns:
        List of perfect-fit candidate configurations
    """
    candidates = []

    # Search tile sizes in fine steps (0.1 MP)
    tile_mp = min_tile_mp
    while tile_mp <= max_tile_mp + 0.001:
        # Calculate tile dimensions for this MP
        tile_w, tile_h = find_optimal_tile_size(aspect_ratio, tile_mp, divisor=32)
        actual_mp = (tile_w * tile_h) / 1_000_000

        # Get overlap params for this tile size
        params = get_overlap_params_for_tile(tile_w, tile_h, scaling_mode, overlap_scale)
        padding = params["tile_padding"]

        # Step sizes
        step_w = tile_w - padding
        step_h = tile_h - padding

        if step_w <= 0 or step_h <= 0:
            tile_mp += 0.1
            continue

        # Try different tile counts (1 to 10 tiles per dimension)
        for tiles_x in range(1, 11):
            for tiles_y in range(1, 11):
                # Calculate the EXACT output size for perfect fit
                # output = tile + (tiles - 1) * step
                perfect_out_w = tile_w + (tiles_x - 1) * step_w
                perfect_out_h = tile_h + (tiles_y - 1) * step_h

                # Calculate required upscales
                upscale_w = perfect_out_w / input_w
                upscale_h = perfect_out_h / input_h

                # Skip if upscales are too different (aspect ratio not preserved)
                if max(upscale_w, upscale_h) > 0:
                    aspect_deviation = abs(upscale_w - upscale_h) / max(upscale_w, upscale_h)
                    if aspect_deviation > 0.03:  # 3% tolerance
                        continue

                # Average upscale
                upscale = (upscale_w + upscale_h) / 2

                # Check if within tolerance of target
                if target_upscale > 0:
                    deviation = abs(upscale - target_upscale) / target_upscale
                    if deviation > tolerance:
                        continue

                # Skip if upscale is out of valid range
                if upscale < 0.1 or upscale > 4.0:
                    continue

                # PERFECT FIT FOUND!
                candidates.append({
                    "tile_w": tile_w,
                    "tile_h": tile_h,
                    "tile_mp": actual_mp,
                    "tiles_x": tiles_x,
                    "tiles_y": tiles_y,
                    "upscale": round(upscale, 3),
                    "output_w": int(input_w * upscale),
                    "output_h": int(input_h * upscale),
                    "efficiency": 1.0,  # Perfect fit = 100% efficiency
                    "total_tiles": tiles_x * tiles_y,
                    "waste_mp": 0.0,
                    "is_perfect": True,
                    **params,
                })

        tile_mp += 0.1

    return candidates


def find_best_near_perfect(input_w: int, input_h: int, aspect_ratio: float,
                            target_upscale: float, tolerance: float,
                            min_tile_mp: float, max_tile_mp: float,
                            scaling_mode: str, overlap_scale: float) -> dict:
    """
    Phase 2 (Fallback): Fine-grained brute-force search for best near-perfect efficiency.

    Used when no perfect alignment exists within tolerance.
    Uses finer search steps: 0.01 upscale and 0.1 MP tile steps.

    Args:
        Same as find_perfect_alignments

    Returns:
        Best near-perfect configuration dict
    """
    best_result = None
    best_score = -1

    # Fine upscale steps (0.01 instead of 0.05)
    min_upscale = max(0.1, target_upscale * (1 - tolerance))
    max_upscale = min(4.0, target_upscale * (1 + tolerance))

    # Search tile sizes in 0.1 MP steps
    tile_mp = min_tile_mp
    while tile_mp <= max_tile_mp + 0.001:
        tile_w, tile_h = find_optimal_tile_size(aspect_ratio, tile_mp, divisor=32)
        actual_mp = (tile_w * tile_h) / 1_000_000

        params = get_overlap_params_for_tile(tile_w, tile_h, scaling_mode, overlap_scale)

        # Search upscales in 0.01 steps
        upscale = min_upscale
        while upscale <= max_upscale + 0.001:
            out_w = int(input_w * upscale)
            out_h = int(input_h * upscale)

            efficiency, tiles_x, tiles_y, waste_mp = calculate_efficiency(
                out_w, out_h, tile_w, tile_h,
                params["tile_padding"], params["mask_blur"], params["seam_fix_width"]
            )

            # Score: efficiency is primary, with small penalties for deviation and tile count
            deviation_penalty = abs(upscale - target_upscale) * 0.05
            tile_penalty = (tiles_x * tiles_y) * 0.005
            score = efficiency - deviation_penalty - tile_penalty

            if score > best_score:
                best_score = score
                best_result = {
                    "tile_w": tile_w,
                    "tile_h": tile_h,
                    "tile_mp": actual_mp,
                    "tiles_x": tiles_x,
                    "tiles_y": tiles_y,
                    "upscale": round(upscale, 2),
                    "output_w": out_w,
                    "output_h": out_h,
                    "efficiency": efficiency,
                    "total_tiles": tiles_x * tiles_y,
                    "waste_mp": waste_mp,
                    "is_perfect": False,
                    **params,
                }

            upscale += 0.01

        tile_mp += 0.1

    return best_result


def find_optimal_tile_and_upscale(input_w: int, input_h: int, aspect_ratio: float,
                                   target_upscale: float, upscale_tolerance: float,
                                   min_tile_mp: float, max_tile_mp: float,
                                   scaling_mode: str, overlap_scale: float) -> dict:
    """
    v2.0 Two-Phase Algorithm: find optimal tile size AND upscale factor.

    Phase 1: Mathematical perfect alignment search
    Phase 2: Fine-grained brute-force fallback (if no perfect fit found)

    Args:
        input_w, input_h: Input image dimensions
        aspect_ratio: Width / Height ratio
        target_upscale: User's desired upscale factor
        upscale_tolerance: How much upscale can deviate (e.g., 0.5 = ±50%)
        min_tile_mp: Minimum megapixels per tile to search
        max_tile_mp: Maximum megapixels per tile (e.g., 2.0 for z-image-turbo)
        scaling_mode: "Proportional" or "Fixed" for blur/padding
        overlap_scale: Global multiplier for overlap parameters

    Returns:
        Dict with best tile size, upscale, overlap params, efficiency, etc.
    """
    # Phase 1: Try to find PERFECT alignment (100% efficiency)
    perfect_candidates = find_perfect_alignments(
        input_w=input_w,
        input_h=input_h,
        aspect_ratio=aspect_ratio,
        target_upscale=target_upscale,
        tolerance=upscale_tolerance,
        min_tile_mp=min_tile_mp,
        max_tile_mp=max_tile_mp,
        scaling_mode=scaling_mode,
        overlap_scale=overlap_scale
    )

    if perfect_candidates:
        # Sort by: (1) closest to target upscale, (2) fewest tiles, (3) largest tile MP
        best = min(perfect_candidates, key=lambda c: (
            abs(c["upscale"] - target_upscale),  # Primary: closest to target
            c["total_tiles"],                      # Secondary: fewer tiles
            -c["tile_mp"]                          # Tertiary: prefer larger tiles
        ))
        return best

    # Phase 2: No perfect fit found, use fine-grained brute-force
    near_perfect = find_best_near_perfect(
        input_w=input_w,
        input_h=input_h,
        aspect_ratio=aspect_ratio,
        target_upscale=target_upscale,
        tolerance=upscale_tolerance,
        min_tile_mp=min_tile_mp,
        max_tile_mp=max_tile_mp,
        scaling_mode=scaling_mode,
        overlap_scale=overlap_scale
    )

    if near_perfect:
        return near_perfect

    # Fallback if nothing found (shouldn't happen with reasonable params)
    tile_w, tile_h = find_optimal_tile_size(aspect_ratio, max_tile_mp, divisor=32)
    params = get_overlap_params_for_tile(tile_w, tile_h, scaling_mode, overlap_scale)
    return {
        "tile_w": tile_w,
        "tile_h": tile_h,
        "tile_mp": (tile_w * tile_h) / 1_000_000,
        "upscale": target_upscale,
        "efficiency": 0.5,
        "tiles_x": 1,
        "tiles_y": 1,
        "total_tiles": 1,
        "output_w": int(input_w * target_upscale),
        "output_h": int(input_h * target_upscale),
        "waste_mp": 0,
        "is_perfect": False,
        **params,
    }


# ============================================================================
# NODE CLASS
# ============================================================================

class ArchAi3D_Smart_Tile_Calculator:
    """
    Calculate optimal tile size and upscale for Ultimate SD Upscale.

    This node analyzes the input image and calculates:
    - Optimal tile size matching the image's aspect ratio (~2MP for z-image-turbo)
    - Optimal upscale factor within tolerance to minimize overlap waste
    - Proportionally scaled blur/padding values for the tile size

    All outputs can be directly connected to Ultimate SD Upscale node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Input image to analyze for optimal tile/upscale calculation"
                }),
                "target_upscale": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.5,
                    "max": 4.0,
                    "step": 0.05,
                    "tooltip": "Desired upscale factor (will be optimized within tolerance)"
                }),
                "min_tile_mp": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 4.0,
                    "step": 0.25,
                    "tooltip": "MINIMUM tile megapixels to search. Set equal to max for fixed tile size."
                }),
                "max_tile_mp": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.5,
                    "max": 4.0,
                    "step": 0.25,
                    "tooltip": "MAXIMUM tile megapixels (2.0 for z-image-turbo). Node searches min to max."
                }),
                "upscale_tolerance": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 0.75,
                    "step": 0.05,
                    "tooltip": "How much upscale can deviate from target (0.5 = ±50%). More tolerance = better optimization."
                }),
                "scaling_mode": (["Proportional", "Fixed"], {
                    "default": "Proportional",
                    "tooltip": "Proportional: scale blur/padding with tile size. Fixed: use USDU defaults"
                }),
                "overlap_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.25,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Global overlap multiplier: 1.0=100%, 0.5=50% less overlap, 1.5=50% more overlap"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "INT", "INT", "INT", "INT", "INT", "INT", "INT",
                    "INT", "INT", "INT", "FLOAT", "FLOAT", "INT", "INT", "STRING", "INT",
                    "SMART_TILE_BUNDLE")
    RETURN_NAMES = ("scaled_image", "upscale_by", "tile_width", "tile_height",
                    "mask_blur", "tile_padding", "seam_fix_width",
                    "seam_fix_mask_blur", "seam_fix_padding",
                    "tiles_x", "tiles_y", "total_tiles",
                    "efficiency", "crop_factor", "output_width", "output_height", "debug_info", "guide_size",
                    "bundle")
    FUNCTION = "calculate"
    CATEGORY = "ArchAi3d/Utils"

    def calculate(self, image, target_upscale, min_tile_mp, max_tile_mp, upscale_tolerance, scaling_mode, overlap_scale):
        """
        v2.0 Two-Phase Algorithm for optimal tile size and upscale factor.

        Phase 1: Mathematical perfect alignment search (100% efficiency)
        Phase 2: Fine-grained brute-force fallback (95%+ efficiency)

        Returns all values needed for Ultimate SD Upscale.
        """
        # Get image dimensions from tensor (B, H, W, C)
        _, height, width, _ = image.shape
        aspect_ratio = width / height

        # Ensure min <= max
        if min_tile_mp > max_tile_mp:
            min_tile_mp, max_tile_mp = max_tile_mp, min_tile_mp

        # Comprehensive 2D search: find best tile size AND upscale combo
        result = find_optimal_tile_and_upscale(
            input_w=width,
            input_h=height,
            aspect_ratio=aspect_ratio,
            target_upscale=target_upscale,
            upscale_tolerance=upscale_tolerance,
            min_tile_mp=min_tile_mp,
            max_tile_mp=max_tile_mp,
            scaling_mode=scaling_mode,
            overlap_scale=overlap_scale
        )

        # Extract results
        tile_w = result["tile_w"]
        tile_h = result["tile_h"]
        actual_tile_mp = result["tile_mp"]
        best_upscale = result["upscale"]
        efficiency = result["efficiency"]
        tiles_x = result["tiles_x"]
        tiles_y = result["tiles_y"]
        total_tiles = result["total_tiles"]
        output_width = result["output_w"]
        output_height = result["output_h"]
        waste_mp = result.get("waste_mp", 0)
        mask_blur = result["mask_blur"]
        tile_padding = result["tile_padding"]
        seam_fix_width = result["seam_fix_width"]
        seam_fix_mask_blur = result["seam_fix_mask_blur"]
        seam_fix_padding = result["seam_fix_padding"]
        is_perfect = result.get("is_perfect", False)

        # Calculate crop_factor for Smart Tile SEGS
        # Formula: (tile_width + 2 * tile_padding) / tile_width
        # This gives the expansion ratio around each tile for context
        crop_factor = (tile_w + 2 * tile_padding) / tile_w

        # Calculate guide_size for Detailer (SEGS) - use larger tile dimension
        guide_size = max(tile_w, tile_h)

        # Determine algorithm used
        if is_perfect:
            algo_status = "PERFECT ALIGNMENT (100% efficiency)"
        else:
            algo_status = f"Near-Perfect ({efficiency*100:.1f}% efficiency)"

        # Build debug info
        debug_lines = [
            "=" * 50,
            "Smart Tile Calculator v2.5 (Bundle Output)",
            "=" * 50,
            f"Input: {width}x{height} ({aspect_ratio:.3f} aspect ratio)",
            f"Target: {target_upscale}x upscale, tiles {min_tile_mp}-{max_tile_mp}MP",
            f"Search: tiles {min_tile_mp}-{max_tile_mp}MP, upscale ±{upscale_tolerance*100:.0f}%",
            "",
            f"--- Result: {algo_status} ---",
            f"Tile Size: {tile_w}x{tile_h} ({actual_tile_mp:.2f}MP)",
            f"Upscale: {best_upscale}x (target was {target_upscale}x)",
            f"Output: {output_width}x{output_height}",
            "",
            "--- Overlap Parameters ({}, scale={:.0%}) ---".format(scaling_mode, overlap_scale),
            f"mask_blur: {mask_blur}",
            f"tile_padding: {tile_padding}",
            f"seam_fix_width: {seam_fix_width}",
            f"seam_fix_mask_blur: {seam_fix_mask_blur}",
            f"seam_fix_padding: {seam_fix_padding}",
            f"crop_factor: {crop_factor:.3f} (for Smart Tile SEGS)",
            f"guide_size: {guide_size} (for Detailer SEGS)",
            "",
            "--- Efficiency ---",
            f"Tiles: {tiles_x}x{tiles_y} = {total_tiles} total",
            f"Efficiency: {efficiency*100:.1f}%",
            f"Wasted: {waste_mp:.2f}MP",
            "=" * 50,
        ]
        debug_info = "\n".join(debug_lines)

        # Log to console
        perfect_tag = " [PERFECT]" if is_perfect else ""
        print(f"\n[Smart Tile Calculator v2.5]{perfect_tag}")
        print(f"  Input: {width}x{height} → Tile: {tile_w}x{tile_h} ({actual_tile_mp:.2f}MP)")
        print(f"  Upscale: {best_upscale}x → Output: {output_width}x{output_height}")
        print(f"  Overlap: {scaling_mode}, scale={overlap_scale:.0%} (blur={mask_blur}, pad={tile_padding})")
        print(f"  Tiles: {tiles_x}x{tiles_y}={total_tiles}, Efficiency: {efficiency*100:.1f}%, Waste: {waste_mp:.2f}MP")

        # Scale the image to output dimensions
        # Image tensor is (B, H, W, C), need to convert to (B, C, H, W) for interpolate
        img_permuted = image.permute(0, 3, 1, 2)  # (B, C, H, W)
        scaled = F.interpolate(img_permuted, size=(output_height, output_width), mode='bilinear', align_corners=False)
        scaled_image = scaled.permute(0, 2, 3, 1)  # Back to (B, H, W, C)
        print(f"  Scaled image: {width}x{height} → {output_width}x{output_height}")

        # Create bundle with all tile data (for one-wire connections)
        bundle = {
            "scaled_image": scaled_image,
            "tile_width": tile_w,
            "tile_height": tile_h,
            "tiles_x": tiles_x,
            "tiles_y": tiles_y,
            "total_tiles": total_tiles,
            "tile_padding": tile_padding,
            "mask_blur": mask_blur,
            "crop_factor": round(crop_factor, 3),
            "guide_size": guide_size,
            "seam_fix_width": seam_fix_width,
            "seam_fix_mask_blur": seam_fix_mask_blur,
            "seam_fix_padding": seam_fix_padding,
            "upscale_by": best_upscale,
            "output_width": output_width,
            "output_height": output_height,
            "efficiency": round(efficiency, 4),
        }

        return (
            scaled_image,
            best_upscale,
            tile_w,
            tile_h,
            mask_blur,
            tile_padding,
            seam_fix_width,
            seam_fix_mask_blur,
            seam_fix_padding,
            tiles_x,
            tiles_y,
            total_tiles,
            round(efficiency, 4),
            round(crop_factor, 3),
            output_width,
            output_height,
            debug_info,
            guide_size,
            bundle
        )


# ============================================================================
# NODE REGISTRATION
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "ArchAi3D_Smart_Tile_Calculator": ArchAi3D_Smart_Tile_Calculator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_Smart_Tile_Calculator": "🧮 Smart Tile Calculator",
}
