"""
Smart Tile Calculator V3 (Flux/Qwen Edition)

A mathematical node that calculates optimal tile parameters for Flux/Qwen models.
Ensures 64-pixel alignment which is critical for transformer-based models.

Key features:
- User-selectable tile aspect ratios (1:1, 16:9, 9:16, 3:2, 2:3, etc.)
- Enforces 64-pixel alignment for tile dimensions
- "Even Overlap" strategy for uniform seam distribution
- Auto aspect ratio detection based on image orientation
- Image upscaling with selectable interpolation methods
- Comprehensive debug output
"""

import math
import torch
import torch.nn.functional as F


# Upscale method mapping
UPSCALE_METHODS = {
    "nearest": "nearest",
    "bilinear": "bilinear",
    "bicubic": "bicubic",
    "area": "area",
    "nearest-exact": "nearest-exact",
    "lanczos": "bicubic",  # PyTorch doesn't have lanczos, use bicubic with antialiasing
}


class ArchAi3D_Smart_Tile_Calculator_V3:
    """
    Smart Tile Calculator V3 for Flux/Qwen models.

    Calculates optimal tile dimensions with 64-pixel alignment,
    which is critical for modern transformer-based diffusion models.
    Also upscales the image and provides comprehensive debug info.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 8.0,
                    "step": 0.1,
                    "tooltip": "Upscale factor (e.g. 2.0 = 2x upscale)"
                }),
                "upscale_method": (["lanczos", "bicubic", "bilinear", "nearest", "nearest-exact", "area"], {
                    "default": "lanczos",
                    "tooltip": "Interpolation method for upscaling"
                }),
                "tile_aspect_ratio": (["auto", "1:1", "16:9", "9:16", "3:2", "2:3", "4:3", "3:4", "2:1", "1:2"], {
                    "default": "auto",
                    "tooltip": "Tile aspect ratio. 'auto' matches image orientation."
                }),
                "target_tile_mp": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.25,
                    "max": 4.0,
                    "step": 0.1,
                    "tooltip": "Target Megapixels per tile (e.g. 1.0 = ~1024x1024)"
                }),
                "padding_pixels": ("INT", {
                    "default": 64,
                    "min": 0,
                    "max": 512,
                    "step": 8,
                    "tooltip": "Minimum overlap between tiles for blending"
                }),
                "optimize_efficiency": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Search for tile size that maximizes efficiency (less wasted pixels)"
                }),
                "mp_tolerance": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.5,
                    "step": 0.1,
                    "tooltip": "How much tile MP can deviate from target. 0=exact, 0.5=±50%, 1.0=±100%"
                }),
                "aspect_freedom": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "How much aspect ratio can change. 0=exact, 0.5=±50%, 1.0=any aspect"
                }),
                "upscale_tolerance": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 0.2,
                    "step": 0.01,
                    "tooltip": "Allow output size to vary for better tile fit. 0=exact, 0.05=±5%, 0.1=±10%"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT", "INT", "INT", "INT", "INT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("upscaled_image", "tile_width", "tile_height", "overlap", "output_width", "output_height", "tiles_x", "tiles_y", "total_tiles", "upscale", "efficiency", "debug_info")
    FUNCTION = "calculate"
    CATEGORY = "ArchAi3d/Upscaling"

    def calculate(self, image, upscale, upscale_method, tile_aspect_ratio, target_tile_mp, padding_pixels, optimize_efficiency, mp_tolerance, aspect_freedom, upscale_tolerance):
        # ============================================================
        # 1. GET INPUT DIMENSIONS
        # ============================================================
        # image is [B, H, W, C]
        batch_size = image.shape[0]
        input_h, input_w = image.shape[1], image.shape[2]
        channels = image.shape[3]

        # Calculate base output dimensions (may be adjusted by upscale_tolerance)
        base_output_w = int(input_w * upscale)
        base_output_h = int(input_h * upscale)

        # These will be updated if upscale_tolerance optimization finds better values
        output_w = base_output_w
        output_h = base_output_h
        actual_upscale = upscale

        # Input stats
        input_mp = (input_w * input_h) / (1024 * 1024)
        output_mp = (output_w * output_h) / (1024 * 1024)

        # ============================================================
        # 2. UPSCALE THE IMAGE (done later after optimization)
        # ============================================================
        # Get interpolation mode
        mode = UPSCALE_METHODS.get(upscale_method, "bicubic")

        # Use antialiasing for downscaling or lanczos-like quality
        antialias = upscale_method in ["lanczos", "bicubic", "bilinear"] and upscale >= 1.0

        # ============================================================
        # 3. DETERMINE TILE ASPECT RATIO
        # ============================================================
        ratio = 1.0
        ratio_name = "1:1"

        if tile_aspect_ratio == "auto":
            # Simple Heuristic: Match OUTPUT image orientation
            if output_w > 1.2 * output_h:
                ratio = 16 / 9  # Wide
                ratio_name = "16:9 (auto-wide)"
            elif output_h > 1.2 * output_w:
                ratio = 9 / 16  # Tall
                ratio_name = "9:16 (auto-tall)"
            else:
                ratio = 1.0  # Square
                ratio_name = "1:1 (auto-square)"
        else:
            # Parse "16:9" -> 1.777
            try:
                w_str, h_str = tile_aspect_ratio.split(":")
                ratio = float(w_str) / float(h_str)
                ratio_name = tile_aspect_ratio
            except:
                ratio = 1.0
                ratio_name = "1:1 (fallback)"

        # ============================================================
        # 4. CALCULATE TILE DIMENSIONS (64-ALIGNED)
        # ============================================================
        # Area = MP * 1,048,576 (2^20) for strict binary MP
        # w * h = Area
        # w = r * h  ->  r * h^2 = Area  -> h = sqrt(Area/r)
        target_area = target_tile_mp * 1024 * 1024

        raw_h = math.sqrt(target_area / ratio)
        raw_w = raw_h * ratio

        # Enforce 64-Pixel Alignment (CRITICAL STEP)
        base_tile_width = math.ceil(raw_w / 64) * 64
        base_tile_height = math.ceil(raw_h / 64) * 64

        # Sanity Checks
        base_tile_width = max(64, base_tile_width)
        base_tile_height = max(64, base_tile_height)

        # ============================================================
        # 4b. OPTIMIZE TILE SIZE FOR EFFICIENCY (OPTIONAL)
        # ============================================================
        def calc_efficiency_for_tile(tw, th, out_w, out_h, min_overlap):
            """Calculate efficiency for a given tile size."""
            if tw <= min_overlap or th <= min_overlap:
                return 0, 999, 999, 0  # Invalid

            # Calculate tiles needed
            def tiles_needed(total, tile, ovl):
                if total <= tile:
                    return 1
                step = tile - ovl
                if step <= 0:
                    return max(1, math.ceil(total / tile))
                return 1 + math.ceil((total - tile) / step)

            tx = tiles_needed(out_w, tw, min_overlap)
            ty = tiles_needed(out_h, th, min_overlap)
            total = tx * ty
            processed = total * tw * th
            eff = (out_w * out_h) / processed * 100 if processed > 0 else 0
            return eff, tx, ty, total

        optimization_info = ""
        upscale_adjusted = False

        if optimize_efficiency and (mp_tolerance > 0 or aspect_freedom > 0 or upscale_tolerance > 0):
            best_eff = 0
            best_tw, best_th = base_tile_width, base_tile_height
            best_out_w, best_out_h = output_w, output_h
            candidates_tested = 0

            # Calculate bounds based on user tolerance settings
            min_mp = target_tile_mp * (1.0 - mp_tolerance)
            max_mp = target_tile_mp * (1.0 + mp_tolerance)
            min_mp = max(0.1, min_mp)  # Absolute minimum 0.1 MP

            # Aspect ratio bounds
            min_ratio = ratio * (1.0 - aspect_freedom) if aspect_freedom < 1.0 else 0.1
            max_ratio = ratio * (1.0 + aspect_freedom) if aspect_freedom < 1.0 else 10.0

            # Output size range (for upscale tolerance)
            min_out_w = int(base_output_w * (1.0 - upscale_tolerance))
            max_out_w = int(base_output_w * (1.0 + upscale_tolerance))
            min_out_h = int(base_output_h * (1.0 - upscale_tolerance))
            max_out_h = int(base_output_h * (1.0 + upscale_tolerance))

            # Round to 64 for cleaner dimensions
            min_out_w = (min_out_w // 64) * 64
            max_out_w = ((max_out_w + 63) // 64) * 64
            min_out_h = (min_out_h // 64) * 64
            max_out_h = ((max_out_h + 63) // 64) * 64

            # Generate output size candidates
            if upscale_tolerance > 0:
                out_w_candidates = list(range(min_out_w, max_out_w + 1, 64))
                out_h_candidates = list(range(min_out_h, max_out_h + 1, 64))
            else:
                out_w_candidates = [output_w]
                out_h_candidates = [output_h]

            # ========== PHASE 1: Search tile sizes for each output candidate ==========
            search_range = max(1, int(5 * max(mp_tolerance, 0.3)))

            for out_w_cand in out_w_candidates:
                for out_h_cand in out_h_candidates:
                    # Maintain aspect ratio within tolerance
                    if upscale_tolerance > 0:
                        cand_ratio = out_w_cand / out_h_cand
                        orig_ratio = base_output_w / base_output_h
                        if abs(cand_ratio - orig_ratio) / orig_ratio > upscale_tolerance:
                            continue

                    for dw in range(-search_range, search_range + 1):
                        for dh in range(-search_range, search_range + 1):
                            tw = base_tile_width + dw * 64
                            th = base_tile_height + dh * 64

                            if tw < 256 or th < 256:
                                continue

                            mp = (tw * th) / (1024 * 1024)
                            if mp < min_mp or mp > max_mp:
                                continue

                            if aspect_freedom < 1.0:
                                candidate_ratio = tw / th
                                if candidate_ratio < min_ratio or candidate_ratio > max_ratio:
                                    continue

                            candidates_tested += 1
                            eff, _, _, _ = calc_efficiency_for_tile(tw, th, out_w_cand, out_h_cand, padding_pixels)

                            if eff > best_eff:
                                best_eff = eff
                                best_tw, best_th = tw, th
                                best_out_w, best_out_h = out_w_cand, out_h_cand

            # ========== PHASE 2: Wide search (only if freedom allows) ==========
            if aspect_freedom >= 0.5 and mp_tolerance >= 0.3:
                for out_w_cand in out_w_candidates:
                    for out_h_cand in out_h_candidates:
                        if upscale_tolerance > 0:
                            cand_ratio = out_w_cand / out_h_cand
                            orig_ratio = base_output_w / base_output_h
                            if abs(cand_ratio - orig_ratio) / orig_ratio > upscale_tolerance:
                                continue

                        for tw in range(256, min(2048, out_w_cand) + 1, 64):
                            for th in range(256, min(2048, out_h_cand) + 1, 64):
                                mp = (tw * th) / (1024 * 1024)
                                if mp < min_mp or mp > max_mp:
                                    continue

                                if aspect_freedom < 1.0:
                                    candidate_ratio = tw / th
                                    if candidate_ratio < min_ratio or candidate_ratio > max_ratio:
                                        continue

                                candidates_tested += 1
                                eff, _, _, _ = calc_efficiency_for_tile(tw, th, out_w_cand, out_h_cand, padding_pixels)

                                # Bonus for perfect fit
                                step_w = tw - padding_pixels
                                step_h = th - padding_pixels
                                if step_w > 0 and step_h > 0:
                                    remainder_w = (out_w_cand - tw) % step_w if out_w_cand > tw else 0
                                    remainder_h = (out_h_cand - th) % step_h if out_h_cand > th else 0
                                    if remainder_w == 0 and remainder_h == 0:
                                        eff += 10  # Big bonus for perfect fit
                                    elif remainder_w < 32 and remainder_h < 32:
                                        eff += 5

                                if eff > best_eff:
                                    best_eff = eff
                                    best_tw, best_th = tw, th
                                    best_out_w, best_out_h = out_w_cand, out_h_cand

            tile_width = best_tw
            tile_height = best_th

            # Update output dimensions if changed
            if best_out_w != base_output_w or best_out_h != base_output_h:
                output_w = best_out_w
                output_h = best_out_h
                actual_upscale = output_w / input_w  # Recalculate actual upscale
                upscale_adjusted = True
                output_mp = (output_w * output_h) / (1024 * 1024)

            upscale_info = f", Upscale±{upscale_tolerance*100:.0f}%" if upscale_tolerance > 0 else ""
            adjusted_info = f" [ADJUSTED to {actual_upscale:.3f}x]" if upscale_adjusted else ""
            optimization_info = f"  Optimization: ON (MP±{mp_tolerance*100:.0f}%, Aspect±{aspect_freedom*100:.0f}%{upscale_info}, {candidates_tested} candidates){adjusted_info}"
        else:
            tile_width = base_tile_width
            tile_height = base_tile_height
            optimization_info = f"  Optimization: OFF (using exact {target_tile_mp:.1f}MP @ {ratio_name})"

        actual_tile_mp = (tile_width * tile_height) / (1024 * 1024)

        # ============================================================
        # 5b. PERFORM ACTUAL UPSCALING (with optimized dimensions)
        # ============================================================
        image_bchw = image.movedim(-1, 1)

        if mode in ["nearest", "nearest-exact"]:
            upscaled_bchw = F.interpolate(image_bchw, size=(output_h, output_w), mode=mode)
        elif mode == "area":
            upscaled_bchw = F.interpolate(image_bchw, size=(output_h, output_w), mode=mode)
        else:
            upscaled_bchw = F.interpolate(
                image_bchw,
                size=(output_h, output_w),
                mode=mode,
                align_corners=False,
                antialias=antialias
            )

        upscaled_image = upscaled_bchw.movedim(1, -1)

        # Latent dimensions (VAE compression = 8x)
        latent_tile_w = tile_width // 8
        latent_tile_h = tile_height // 8

        # ============================================================
        # 5. CALCULATE GRID & OVERLAP (ELASTIC OVERLAP)
        # ============================================================
        def calculate_tiles_needed(total_len, tile_len, overlap):
            """
            Calculate how many tiles are needed to cover total_len with given overlap.
            """
            if total_len <= tile_len:
                return 1
            step = tile_len - overlap
            if step <= 0:
                return math.ceil(total_len / tile_len)
            # First tile covers tile_len, each additional covers step more
            remaining = total_len - tile_len
            additional = math.ceil(remaining / step)
            return 1 + additional

        def calculate_optimal_overlap(total_len, tile_len, num_tiles):
            """
            Given fixed tile count, calculate the exact overlap for perfect distribution.
            """
            if num_tiles <= 1:
                return 0
            # total_len = tile_len + (num_tiles - 1) * (tile_len - overlap)
            # overlap = (num_tiles * tile_len - total_len) / (num_tiles - 1)
            overlap = (num_tiles * tile_len - total_len) / (num_tiles - 1)
            return max(0, overlap)

        # Start with minimum padding as baseline overlap
        base_overlap = padding_pixels

        # Round base overlap to multiple of 8 for clean latent math
        base_overlap = max(padding_pixels, ((padding_pixels + 7) // 8) * 8)

        # Calculate tiles needed with base overlap
        tiles_x = calculate_tiles_needed(output_w, tile_width, base_overlap)
        tiles_y = calculate_tiles_needed(output_h, tile_height, base_overlap)

        # Calculate actual optimal overlap for each dimension
        overlap_x = calculate_optimal_overlap(output_w, tile_width, tiles_x)
        overlap_y = calculate_optimal_overlap(output_h, tile_height, tiles_y)

        # Use the larger overlap for both dimensions (ensures coverage)
        final_overlap = max(overlap_x, overlap_y)

        # Round UP to multiple of 8 for clean latent math
        final_overlap = math.ceil(final_overlap / 8) * 8

        # Ensure minimum padding
        final_overlap = max(final_overlap, padding_pixels)

        # Recalculate tiles with final overlap (may need more tiles now)
        tiles_x = calculate_tiles_needed(output_w, tile_width, final_overlap)
        tiles_y = calculate_tiles_needed(output_h, tile_height, final_overlap)

        total_tiles = tiles_x * tiles_y

        # ============================================================
        # 6. CALCULATE COVERAGE & EFFICIENCY
        # ============================================================
        if tiles_x > 1:
            coverage_w = tile_width + (tiles_x - 1) * (tile_width - final_overlap)
        else:
            coverage_w = tile_width

        if tiles_y > 1:
            coverage_h = tile_height + (tiles_y - 1) * (tile_height - final_overlap)
        else:
            coverage_h = tile_height

        # Total pixels processed vs output pixels (efficiency)
        total_processed = total_tiles * tile_width * tile_height
        overlap_waste = total_processed - (output_w * output_h)
        efficiency = (output_w * output_h) / total_processed * 100 if total_processed > 0 else 100
        wasted_mp = overlap_waste / (1024 * 1024)

        coverage_ok = coverage_w >= output_w and coverage_h >= output_h

        # ============================================================
        # 7. BUILD DEBUG INFO STRING
        # ============================================================
        debug_lines = [
            "=" * 60,
            "SMART TILE CALCULATOR V3 (Flux/Qwen Edition)",
            "=" * 60,
            "",
            "INPUT IMAGE:",
            f"  Dimensions: {input_w} x {input_h} px",
            f"  Megapixels: {input_mp:.2f} MP",
            f"  Batch size: {batch_size}",
            f"  Channels: {channels}",
            "",
            "UPSCALING:",
            f"  Requested: {upscale}x",
            f"  Actual: {actual_upscale:.4f}x" + (" [ADJUSTED]" if upscale_adjusted else ""),
            f"  Tolerance: ±{upscale_tolerance*100:.0f}%",
            f"  Method: {upscale_method}",
            f"  Antialiasing: {antialias}",
            "",
            "OUTPUT IMAGE:",
            f"  Base dimensions: {base_output_w} x {base_output_h} px",
            f"  Final dimensions: {output_w} x {output_h} px" + (" [ADJUSTED]" if upscale_adjusted else ""),
            f"  Megapixels: {output_mp:.2f} MP",
            "",
            "TILE CONFIGURATION:",
            f"  Aspect Ratio: {ratio_name} ({ratio:.3f})",
            f"  Target MP: {target_tile_mp} ({target_area:,.0f} px^2)",
            f"  Raw tile size: {raw_w:.1f} x {raw_h:.1f} px",
            f"  MP Tolerance: ±{mp_tolerance*100:.0f}% ({target_tile_mp*(1-mp_tolerance):.2f} - {target_tile_mp*(1+mp_tolerance):.2f} MP)",
            f"  Aspect Freedom: ±{aspect_freedom*100:.0f}%",
            "",
            "64-ALIGNED TILE (FINAL):",
            f"  Base tile size: {base_tile_width} x {base_tile_height} px",
            f"  Final tile size: {tile_width} x {tile_height} px",
            f"  Actual MP: {actual_tile_mp:.2f} MP",
            f"  Latent size: {latent_tile_w} x {latent_tile_h}",
            f"  Width divisible by 64: {tile_width % 64 == 0}",
            f"  Height divisible by 64: {tile_height % 64 == 0}",
            optimization_info,
            "",
            "GRID CALCULATION:",
            f"  Tiles X: {tiles_x} (calculated overlap: {overlap_x})",
            f"  Tiles Y: {tiles_y} (calculated overlap: {overlap_y})",
            f"  Final overlap: {final_overlap} px",
            f"  Total tiles: {total_tiles}",
            "",
            "COVERAGE ANALYSIS:",
            f"  Coverage: {coverage_w} x {coverage_h} px",
            f"  Output: {output_w} x {output_h} px",
            f"  Coverage OK: {coverage_ok}",
            "",
            "--- EFFICIENCY ---",
            f"  Tiles: {tiles_x}x{tiles_y} = {total_tiles} total",
            f"  Efficiency: {efficiency:.1f}%",
            f"  Wasted: {wasted_mp:.2f} MP",
            f"  Total pixels processed: {total_processed:,}",
            f"  Overlap waste: {overlap_waste:,} px",
            "",
            "RECOMMENDED USDU SETTINGS:",
            f"  tile_width: {tile_width}",
            f"  tile_height: {tile_height}",
            f"  tile_padding: {final_overlap}",
            f"  upscale_by: {actual_upscale:.4f}" + (f" (was {upscale})" if upscale_adjusted else ""),
            "=" * 60,
        ]

        debug_info = "\n".join(debug_lines)

        # Print to console as well
        print(debug_info)

        # ============================================================
        # 8. RETURN ALL OUTPUTS
        # ============================================================
        return (
            upscaled_image,     # IMAGE - upscaled image
            tile_width,         # INT - 64-aligned tile width
            tile_height,        # INT - 64-aligned tile height
            final_overlap,      # INT - calculated overlap/padding
            output_w,           # INT - output image width
            output_h,           # INT - output image height
            tiles_x,            # INT - tiles in X direction
            tiles_y,            # INT - tiles in Y direction
            total_tiles,        # INT - total number of tiles
            actual_upscale,     # FLOAT - actual upscale factor (may differ from requested)
            efficiency,         # FLOAT - efficiency percentage
            debug_info,         # STRING - full debug output
        )


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "ArchAi3D_Smart_Tile_Calculator_V3": ArchAi3D_Smart_Tile_Calculator_V3
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_Smart_Tile_Calculator_V3": "🧮 Smart Tile Calculator V3 (Flux)"
}
