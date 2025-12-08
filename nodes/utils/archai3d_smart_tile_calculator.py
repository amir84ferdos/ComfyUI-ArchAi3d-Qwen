# -*- coding: utf-8 -*-
"""
ArchAi3D Smart Tile Calculator

Calculates optimal tile size, upscale factor, and blur/padding values
for Ultimate SD Upscale to minimize wasted overlap and processing costs.

Author: Amir Ferdos (ArchAi3d)
Email: Amir84ferdos@gmail.com
LinkedIn: https://www.linkedin.com/in/archai3d/
GitHub: https://github.com/amir84ferdos

Version: 1.1.0 - Added overlap_scale for global overlap control
License: Dual License (Free for personal use, Commercial license required for business use)
"""

import math


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def find_optimal_tile_size(aspect_ratio: float, target_mp: float = 2.0, divisor: int = 8) -> tuple:
    """
    Find tile dimensions matching aspect ratio, ~target_mp, divisible by divisor.

    Args:
        aspect_ratio: Width / Height ratio of the image
        target_mp: Target megapixels per tile (default 2.0 for z-image-turbo)
        divisor: Tile dimensions must be divisible by this (8 for USDU)

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

    # Round to nearest divisor (8 for Ultimate SD Upscale compatibility)
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
                "target_tile_mp": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.5,
                    "max": 4.0,
                    "step": 0.1,
                    "tooltip": "Target megapixels per tile (2.0 for z-image-turbo)"
                }),
                "upscale_tolerance": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.05,
                    "tooltip": "How much upscale can deviate from target (0.3 = ±30%)"
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

    RETURN_TYPES = ("FLOAT", "INT", "INT", "INT", "INT", "INT", "INT", "INT",
                    "INT", "INT", "INT", "FLOAT", "INT", "INT", "STRING")
    RETURN_NAMES = ("upscale_by", "tile_width", "tile_height",
                    "mask_blur", "tile_padding", "seam_fix_width",
                    "seam_fix_mask_blur", "seam_fix_padding",
                    "tiles_x", "tiles_y", "total_tiles",
                    "efficiency", "output_width", "output_height", "debug_info")
    FUNCTION = "calculate"
    CATEGORY = "ArchAi3d/Utils"

    def calculate(self, image, target_upscale, target_tile_mp, upscale_tolerance, scaling_mode, overlap_scale):
        """
        Calculate optimal tile size and upscale factor.

        Returns all values needed for Ultimate SD Upscale.
        """
        # Get image dimensions from tensor (B, H, W, C)
        _, height, width, _ = image.shape
        aspect_ratio = width / height

        # Step 1: Calculate optimal tile size
        tile_w, tile_h = find_optimal_tile_size(aspect_ratio, target_tile_mp, divisor=8)
        actual_tile_mp = (tile_w * tile_h) / 1_000_000

        # Step 2: Calculate blur/padding values
        min_tile_dim = min(tile_w, tile_h)
        overlap_params = scale_overlap_params(min_tile_dim, base_tile=512, mode=scaling_mode)

        # Step 2.5: Apply global overlap_scale multiplier
        # This allows user to test with less/more overlap (e.g., 0.5 = 50% less)
        mask_blur = max(1, int(overlap_params["mask_blur"] * overlap_scale))
        tile_padding = max(8, int(overlap_params["tile_padding"] * overlap_scale))
        seam_fix_width = max(8, int(overlap_params["seam_fix_width"] * overlap_scale))
        seam_fix_mask_blur = max(1, int(overlap_params["seam_fix_mask_blur"] * overlap_scale))
        seam_fix_padding = max(8, int(overlap_params["seam_fix_padding"] * overlap_scale))

        # Round padding/width to step size 8 (USDU requirement)
        tile_padding = round(tile_padding / 8) * 8
        seam_fix_width = round(seam_fix_width / 8) * 8
        seam_fix_padding = round(seam_fix_padding / 8) * 8

        # Ensure minimum values
        tile_padding = max(8, tile_padding)
        seam_fix_width = max(8, seam_fix_width)
        seam_fix_padding = max(8, seam_fix_padding)

        # Step 3: Find optimal upscale factor
        best_upscale, efficiency, tiles_x, tiles_y = find_optimal_upscale(
            width, height, tile_w, tile_h,
            target_upscale, upscale_tolerance,
            tile_padding, mask_blur, seam_fix_width
        )

        # Calculate final output dimensions
        output_width = int(width * best_upscale)
        output_height = int(height * best_upscale)
        total_tiles = tiles_x * tiles_y

        # Build debug info
        debug_lines = [
            "=" * 50,
            "Smart Tile Calculator Results",
            "=" * 50,
            f"Input: {width}x{height} ({aspect_ratio:.3f} aspect ratio)",
            f"Target: {target_upscale}x upscale, {target_tile_mp}MP tiles",
            "",
            "--- Calculated Values ---",
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
            "",
            "--- Efficiency ---",
            f"Tiles: {tiles_x}x{tiles_y} = {total_tiles} total",
            f"Efficiency: {efficiency*100:.1f}%",
            "=" * 50,
        ]
        debug_info = "\n".join(debug_lines)

        # Log to console
        print(f"\n[Smart Tile Calculator]")
        print(f"  Input: {width}x{height} → Tile: {tile_w}x{tile_h}")
        print(f"  Upscale: {best_upscale}x → Output: {output_width}x{output_height}")
        print(f"  Overlap: {scaling_mode}, scale={overlap_scale:.0%} (blur={mask_blur}, pad={tile_padding})")
        print(f"  Tiles: {tiles_x}x{tiles_y}={total_tiles}, Efficiency: {efficiency*100:.1f}%")

        return (
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
            output_width,
            output_height,
            debug_info
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
