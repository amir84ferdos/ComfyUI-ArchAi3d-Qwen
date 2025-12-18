"""
Smart Tile Solver V6 - "Grid-Lock" Engine (ORIGINAL - PRE-FIX)

A revolutionary approach to tile calculation that GUARANTEES optimal efficiency.

NOTE: This is the original V6 code before V6.1 fixes were applied.
      See archai3d_smart_tile_solver_v6.py for the current V6.1 version.

THE PARADIGM SHIFT:
===================
Old Logic (V3-V5 - Tile-First - BAD):
    1. Pick tile size (e.g., 1024)
    2. See how many fit (e.g., 3.4 tiles)
    3. Round up to 4 tiles
    4. Result: 0.6 tiles of WASTE (overhang)

New V6 Logic (Grid-First - GOOD):
    1. Estimate target grid counts (e.g., 3x4 or 4x5)
    2. Calculate exactly what tile size makes that grid fit PERFECTLY
    3. Snap to 64-alignment
    4. Adjust output image slightly to match perfect math
    5. Result: 0% waste - grid is "LOCKED" to image

KEY INSIGHT:
============
Instead of forcing tiles onto a fixed output, we let the output "breathe"
within tolerance to achieve perfect grid lock. This is mathematically
optimal and guarantees minimum VRAM usage.

FORMULA:
========
Coverage: output_w = tile_w * tiles_x - overlap * (tiles_x - 1)
Solving:  tile_w = (output_w + overlap * (tiles_x - 1)) / tiles_x
Then:     exact_output_w = tile_w * tiles_x - overlap * (tiles_x - 1)
"""

import math
import torch
import torch.nn.functional as F

UPSCALE_METHODS = {
    "nearest": "nearest",
    "bilinear": "bilinear",
    "bicubic": "bicubic",
    "area": "area",
    "nearest-exact": "nearest-exact",
    "lanczos": "bicubic",  # PyTorch uses bicubic with antialiasing as lanczos approximation
}

# Aspect ratio limits (reject thin strips)
AR_CONSTRAINTS = {
    "1:1 only": (0.95, 1.05),      # Square only
    "4:3 max": (0.75, 1.34),       # 3:4 to 4:3
    "16:9 max": (0.5625, 1.78),    # 9:16 to 16:9
    "2:1 max": (0.5, 2.0),         # 1:2 to 2:1
    "free": (0.25, 4.0),           # Nearly any shape
}


class ArchAi3D_Smart_Tile_Solver_V6_Old:
    """
    Smart Tile Solver V6 - "Grid-Lock" Engine (ORIGINAL)

    Revolutionary Grid-First approach that guarantees optimal efficiency
    by reverse-engineering the perfect tile size for each grid configuration.

    Instead of guessing tiles and forcing them to fit, V6:
    1. Tries different grid configurations (3x4, 4x5, etc.)
    2. Calculates the exact tile size needed for perfect fit
    3. Snaps to 64-alignment
    4. Adjusts output within tolerance for perfect math
    5. Picks the most efficient solution
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_by": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 8.0,
                    "step": 0.1,
                    "tooltip": "Target upscale factor"
                }),
                "upscale_tolerance": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.01,
                    "tooltip": "How much output can deviate from target (±10% = 0.1)"
                }),
                "target_mp": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.25,
                    "max": 4.0,
                    "step": 0.1,
                    "tooltip": "Target megapixels per tile"
                }),
                "mp_tolerance": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "How much tile MP can deviate from target (±30% = 0.3)"
                }),
                "min_overlap": ("INT", {
                    "default": 64,
                    "min": 0,
                    "max": 512,
                    "step": 8,
                    "tooltip": "Minimum overlap between tiles for blending"
                }),
                "divisibility": ([64, 32, 16, 8], {
                    "default": 64,
                    "tooltip": "Tile alignment (64 for Flux/Qwen, 8 for SD1.5)"
                }),
                "ar_constraint": (list(AR_CONSTRAINTS.keys()), {
                    "default": "16:9 max",
                    "tooltip": "Maximum aspect ratio deviation for tiles"
                }),
                "upscale_method": (["lanczos", "bicubic", "bilinear", "nearest", "nearest-exact", "area"], {
                    "default": "lanczos",
                    "tooltip": "Interpolation method for upscaling"
                }),
                "output_mode": (["shrink_only", "allow_grow", "force_tile_size"], {
                    "default": "shrink_only",
                    "tooltip": "shrink_only: Output never exceeds target. allow_grow: Can grow within tolerance. force_tile_size: Use exact tile from forced inputs."
                }),
            },
            "optional": {
                "forced_tile_width": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "step": 64,
                    "tooltip": "Force specific tile width (0 = auto). Only used when output_mode='force_tile_size'"
                }),
                "forced_tile_height": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "step": 64,
                    "tooltip": "Force specific tile height (0 = auto). Only used when output_mode='force_tile_size'"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT", "INT", "INT", "INT", "INT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("upscaled_image", "tile_width", "tile_height", "overlap", "output_width", "output_height",
                    "tiles_x", "tiles_y", "total_tiles", "actual_upscale", "efficiency", "debug_info")
    FUNCTION = "solve"
    CATEGORY = "ArchAi3d/Upscaling"

    def _generate_grid_candidates(self, estimated_tiles, img_ratio):
        """
        Generate grid configurations to try, sorted by likelihood of efficiency.

        Args:
            estimated_tiles: Approximate total tiles needed (e.g., 12.4)
            img_ratio: Image aspect ratio (width/height)

        Returns:
            List of (tiles_x, tiles_y) tuples to evaluate
        """
        candidates = []

        # Search range: ±50% of estimated tiles
        min_total = max(1, int(estimated_tiles * 0.5))
        max_total = int(estimated_tiles * 2.0) + 1

        for total in range(min_total, max_total + 1):
            # Find all factor pairs for this total
            for tx in range(1, total + 1):
                if total % tx == 0:
                    ty = total // tx

                    # Calculate how well this grid matches image aspect ratio
                    # Ideal: tx/ty ≈ img_ratio
                    grid_ratio = tx / ty if ty > 0 else float('inf')
                    ratio_match = abs(grid_ratio - img_ratio) / max(img_ratio, 0.1)

                    candidates.append((tx, ty, total, ratio_match))

        # Also try non-factor combinations (e.g., 3x5 for 15 area)
        for tx in range(1, max(10, int(estimated_tiles) + 3)):
            for ty in range(1, max(10, int(estimated_tiles / tx) + 3)):
                total = tx * ty
                if min_total <= total <= max_total:
                    grid_ratio = tx / ty if ty > 0 else float('inf')
                    ratio_match = abs(grid_ratio - img_ratio) / max(img_ratio, 0.1)

                    if (tx, ty, total, ratio_match) not in candidates:
                        candidates.append((tx, ty, total, ratio_match))

        # Sort by: 1) total tiles (prefer fewer), 2) ratio match (prefer matching image AR)
        candidates.sort(key=lambda x: (x[2], x[3]))

        # Return just (tiles_x, tiles_y)
        return [(c[0], c[1]) for c in candidates[:100]]  # Limit to top 100

    def _solve_grid(self, image_w, image_h, upscale_by, upscale_tolerance,
                    target_mp, mp_tolerance, min_overlap, divisibility, ar_limits,
                    allow_grow=False):
        """
        Core Grid-Lock algorithm: Find optimal grid configuration.

        KEY CONSTRAINTS (User's requirements):
        1. Output can only SHRINK within tolerance (never grow beyond target)
        2. Overlap is a MINIMUM - can be increased to achieve perfect fit
        3. Tile aspect ratio and MP can vary within tolerance

        Returns:
            dict with optimal solution, or None if no valid solution found
        """
        target_output_w = image_w * upscale_by
        target_output_h = image_h * upscale_by
        target_output_area = target_output_w * target_output_h

        # Estimate tiles needed
        target_tile_area = target_mp * 1024 * 1024
        estimated_tiles = target_output_area / target_tile_area

        # Image aspect ratio
        img_ratio = target_output_w / target_output_h if target_output_h > 0 else 1.0

        # Generate grid candidates
        candidates = self._generate_grid_candidates(estimated_tiles, img_ratio)

        # AR limits
        min_ar, max_ar = ar_limits

        # MP limits
        min_mp = target_mp * (1.0 - mp_tolerance)
        max_mp = target_mp * (1.0 + mp_tolerance)

        # Upscale limits - KEY: Output can only SHRINK, not grow!
        min_upscale = upscale_by * (1.0 - upscale_tolerance)
        max_upscale = upscale_by if not allow_grow else upscale_by * (1.0 + upscale_tolerance)

        best_solution = None
        best_efficiency = 0
        all_candidates = []

        for tiles_x, tiles_y in candidates:
            # Skip single-tile case for multi-MP images
            if tiles_x == 1 and tiles_y == 1 and estimated_tiles > 1.5:
                continue

            # ============================================================
            # CORE FORMULA: Calculate tile size for perfect grid fit
            # ============================================================
            # Coverage: output_w = tile_w * tiles_x - overlap * (tiles_x - 1)
            # Solving:  tile_w = (output_w + overlap * (tiles_x - 1)) / tiles_x

            if tiles_x > 1:
                raw_tile_w = (target_output_w + min_overlap * (tiles_x - 1)) / tiles_x
            else:
                raw_tile_w = target_output_w

            if tiles_y > 1:
                raw_tile_h = (target_output_h + min_overlap * (tiles_y - 1)) / tiles_y
            else:
                raw_tile_h = target_output_h

            # Snap to divisibility (64)
            tile_w = round(raw_tile_w / divisibility) * divisibility
            tile_h = round(raw_tile_h / divisibility) * divisibility

            # Enforce minimum tile size
            tile_w = max(divisibility * 4, tile_w)  # At least 256 for div=64
            tile_h = max(divisibility * 4, tile_h)

            # ============================================================
            # CONSTRAINT CHECKS
            # ============================================================

            # 1. Aspect ratio limits (reject strips)
            ar = tile_w / tile_h if tile_h > 0 else float('inf')
            if ar < min_ar or ar > max_ar:
                continue

            # 2. MP tolerance
            tile_mp = (tile_w * tile_h) / (1024 * 1024)
            if tile_mp < min_mp or tile_mp > max_mp:
                continue

            # ============================================================
            # CALCULATE OUTPUT SIZE & OVERLAP
            # ============================================================
            # KEY INSIGHT: We have snapped tile size. Now we need to find
            # the overlap that makes the grid fit within target output.
            #
            # Coverage = tile * n - overlap * (n-1)
            # We want Coverage <= target_output (no growth!)
            # Solving for overlap: overlap >= (tile * n - target) / (n-1)

            if tiles_x > 1:
                # Minimum overlap needed to not exceed target width
                min_overlap_w = (tile_w * tiles_x - target_output_w) / (tiles_x - 1)
                # Overlap must be at least user's min_overlap
                actual_overlap_w = max(min_overlap, min_overlap_w)
                # Round UP to 8 for latent alignment
                actual_overlap_w = math.ceil(actual_overlap_w / 8) * 8
                exact_output_w = tile_w * tiles_x - actual_overlap_w * (tiles_x - 1)
            else:
                actual_overlap_w = 0
                exact_output_w = tile_w

            if tiles_y > 1:
                min_overlap_h = (tile_h * tiles_y - target_output_h) / (tiles_y - 1)
                actual_overlap_h = max(min_overlap, min_overlap_h)
                actual_overlap_h = math.ceil(actual_overlap_h / 8) * 8
                exact_output_h = tile_h * tiles_y - actual_overlap_h * (tiles_y - 1)
            else:
                actual_overlap_h = 0
                exact_output_h = tile_h

            # Use the larger overlap for both axes (USDU requires single overlap)
            final_overlap = max(actual_overlap_w, actual_overlap_h)

            # Recalculate exact output with unified overlap
            if tiles_x > 1:
                exact_output_w = tile_w * tiles_x - final_overlap * (tiles_x - 1)
            if tiles_y > 1:
                exact_output_h = tile_h * tiles_y - final_overlap * (tiles_y - 1)

            # 3. Upscale tolerance check
            actual_upscale_w = exact_output_w / image_w if image_w > 0 else upscale_by
            actual_upscale_h = exact_output_h / image_h if image_h > 0 else upscale_by

            # KEY: Output must be <= target (or within tolerance if allow_grow)
            if exact_output_w > target_output_w and not allow_grow:
                continue
            if exact_output_h > target_output_h and not allow_grow:
                continue

            if actual_upscale_w < min_upscale:
                continue
            if actual_upscale_h < min_upscale:
                continue

            # Also check max if allow_grow is True
            if allow_grow:
                if actual_upscale_w > max_upscale or actual_upscale_h > max_upscale:
                    continue

            # ============================================================
            # CALCULATE EFFICIENCY
            # ============================================================
            total_tiles = tiles_x * tiles_y
            processed_pixels = total_tiles * tile_w * tile_h
            needed_pixels = exact_output_w * exact_output_h
            efficiency = (needed_pixels / processed_pixels * 100) if processed_pixels > 0 else 0

            # Average upscale factor
            avg_upscale = (actual_upscale_w + actual_upscale_h) / 2

            solution = {
                'tiles_x': tiles_x,
                'tiles_y': tiles_y,
                'tile_w': int(tile_w),
                'tile_h': int(tile_h),
                'output_w': int(exact_output_w),
                'output_h': int(exact_output_h),
                'overlap': int(final_overlap),
                'efficiency': efficiency,
                'total_tiles': total_tiles,
                'tile_mp': tile_mp,
                'actual_upscale': avg_upscale,
            }

            all_candidates.append(solution)

            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_solution = solution

        return best_solution, all_candidates

    def solve(self, image, upscale_by, upscale_tolerance, target_mp, mp_tolerance,
              min_overlap, divisibility, ar_constraint, upscale_method, output_mode,
              forced_tile_width=0, forced_tile_height=0):
        """
        Main entry point: Solve for optimal tile configuration.

        output_mode options:
        - "shrink_only": Output never exceeds target (default, safest)
        - "allow_grow": Output can grow within tolerance
        - "force_tile_size": Use forced_tile_width/height, calculate grid and overlap
        """
        # ============================================================
        # 1. GET INPUT DIMENSIONS
        # ============================================================
        batch_size, input_h, input_w, channels = image.shape

        input_mp = (input_w * input_h) / (1024 * 1024)
        target_output_w = int(input_w * upscale_by)
        target_output_h = int(input_h * upscale_by)
        target_output_mp = (target_output_w * target_output_h) / (1024 * 1024)

        # Get AR limits
        ar_limits = AR_CONSTRAINTS.get(ar_constraint, (0.5625, 1.78))

        # ============================================================
        # 2. HANDLE FORCED TILE SIZE MODE
        # ============================================================
        if output_mode == "force_tile_size" and forced_tile_width > 0 and forced_tile_height > 0:
            # User wants specific tile size - calculate optimal overlap and grid
            tile_width = forced_tile_width
            tile_height = forced_tile_height

            # Find optimal overlap that keeps output <= target
            best_overlap = min_overlap
            best_tiles_x = 1
            best_tiles_y = 1

            for overlap in range(min_overlap, min(tile_width, tile_height) // 2, 8):
                step_w = tile_width - overlap
                step_h = tile_height - overlap

                if step_w <= 0 or step_h <= 0:
                    continue

                # Tiles needed
                tiles_x = 1 if target_output_w <= tile_width else 1 + math.ceil((target_output_w - tile_width) / step_w)
                tiles_y = 1 if target_output_h <= tile_height else 1 + math.ceil((target_output_h - tile_height) / step_h)

                # Coverage
                cov_w = tile_width if tiles_x == 1 else tile_width + (tiles_x - 1) * step_w
                cov_h = tile_height if tiles_y == 1 else tile_height + (tiles_y - 1) * step_h

                # We want coverage >= target but as small as possible
                if cov_w >= target_output_w and cov_h >= target_output_h:
                    best_overlap = overlap
                    best_tiles_x = tiles_x
                    best_tiles_y = tiles_y
                    break

            # Use the output size that matches the grid exactly
            step_w = tile_width - best_overlap
            step_h = tile_height - best_overlap
            output_w = tile_width if best_tiles_x == 1 else tile_width + (best_tiles_x - 1) * step_w
            output_h = tile_height if best_tiles_y == 1 else tile_height + (best_tiles_y - 1) * step_h

            total_tiles = best_tiles_x * best_tiles_y
            tile_mp = (tile_width * tile_height) / (1024 * 1024)
            actual_upscale = (output_w / input_w + output_h / input_h) / 2

            processed_pixels = total_tiles * tile_width * tile_height
            efficiency = (output_w * output_h) / processed_pixels * 100 if processed_pixels > 0 else 0

            solution = {
                'tiles_x': best_tiles_x,
                'tiles_y': best_tiles_y,
                'tile_w': tile_width,
                'tile_h': tile_height,
                'output_w': output_w,
                'output_h': output_h,
                'overlap': best_overlap,
                'efficiency': efficiency,
                'total_tiles': total_tiles,
                'tile_mp': tile_mp,
                'actual_upscale': actual_upscale,
            }
            all_candidates = [solution]
            forced_mode = True
        else:
            # ============================================================
            # 2. RUN GRID-LOCK SOLVER
            # ============================================================
            allow_grow = (output_mode == "allow_grow")
            solution, all_candidates = self._solve_grid(
                input_w, input_h, upscale_by, upscale_tolerance,
                target_mp, mp_tolerance, min_overlap, divisibility, ar_limits,
                allow_grow=allow_grow
            )
            forced_mode = False

        # ============================================================
        # 3. HANDLE NO SOLUTION FOUND
        # ============================================================
        if solution is None:
            # Fallback: Use basic calculation
            tile_dim = int(math.sqrt(target_mp * 1024 * 1024))
            tile_w = max(256, round(tile_dim / divisibility) * divisibility)
            tile_h = tile_w

            output_w = target_output_w
            output_h = target_output_h

            # Calculate tiles needed
            step_w = tile_w - min_overlap
            step_h = tile_h - min_overlap

            if step_w > 0 and step_h > 0:
                tiles_x = max(1, math.ceil((output_w - min_overlap) / step_w))
                tiles_y = max(1, math.ceil((output_h - min_overlap) / step_h))
            else:
                tiles_x = math.ceil(output_w / tile_w)
                tiles_y = math.ceil(output_h / tile_h)

            total_tiles = tiles_x * tiles_y
            processed_pixels = total_tiles * tile_w * tile_h
            efficiency = (output_w * output_h) / processed_pixels * 100 if processed_pixels > 0 else 0

            solution = {
                'tiles_x': tiles_x,
                'tiles_y': tiles_y,
                'tile_w': tile_w,
                'tile_h': tile_h,
                'output_w': output_w,
                'output_h': output_h,
                'overlap': min_overlap,
                'efficiency': efficiency,
                'total_tiles': total_tiles,
                'tile_mp': (tile_w * tile_h) / (1024 * 1024),
                'actual_upscale': upscale_by,
            }

            fallback_used = True
        else:
            fallback_used = False

        # Extract solution
        tile_width = solution['tile_w']
        tile_height = solution['tile_h']
        output_w = solution['output_w']
        output_h = solution['output_h']
        tiles_x = solution['tiles_x']
        tiles_y = solution['tiles_y']
        total_tiles = solution['total_tiles']
        overlap = solution['overlap']
        efficiency = solution['efficiency']
        actual_upscale = solution['actual_upscale']
        tile_mp = solution['tile_mp']

        # ============================================================
        # 4. UPSCALE IMAGE TO EXACT OUTPUT SIZE
        # ============================================================
        image_bchw = image.movedim(-1, 1)
        mode = UPSCALE_METHODS.get(upscale_method, "bicubic")
        antialias = mode in ["bicubic", "bilinear"]

        if mode in ["nearest", "nearest-exact", "area"]:
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

        # ============================================================
        # 5. VERIFY COVERAGE (should always be exact in V6!)
        # ============================================================
        if tiles_x > 1:
            coverage_w = tile_width + (tiles_x - 1) * (tile_width - overlap)
        else:
            coverage_w = tile_width

        if tiles_y > 1:
            coverage_h = tile_height + (tiles_y - 1) * (tile_height - overlap)
        else:
            coverage_h = tile_height

        coverage_ok = coverage_w >= output_w and coverage_h >= output_h

        # ============================================================
        # 6. BUILD DEBUG INFO
        # ============================================================
        output_mp = (output_w * output_h) / (1024 * 1024)
        latent_tile_w = tile_width // 8
        latent_tile_h = tile_height // 8

        # Show top candidates
        top_candidates_str = ""
        if all_candidates:
            sorted_candidates = sorted(all_candidates, key=lambda x: -x['efficiency'])[:5]
            top_candidates_str = "\n  TOP CANDIDATES:\n"
            for i, c in enumerate(sorted_candidates):
                marker = " <-- SELECTED" if c == solution else ""
                top_candidates_str += f"    {i+1}. {c['tiles_x']}x{c['tiles_y']} grid, {c['tile_w']}x{c['tile_h']} tiles, {c['efficiency']:.1f}% eff{marker}\n"

        # Determine mode label
        if forced_mode:
            mode_label = "FORCED"
        elif fallback_used:
            mode_label = "FALLBACK"
        else:
            mode_label = "GRID-LOCKED"

        debug_lines = [
            "=" * 60,
            "SMART TILE SOLVER V6 - Grid-Lock Engine (ORIGINAL)",
            "=" * 60,
            "",
            f"MODE: {output_mode.upper()}" + (f" (tile: {forced_tile_width}x{forced_tile_height})" if forced_mode else ""),
            "",
            "INPUT IMAGE:",
            f"  Dimensions: {input_w} x {input_h} px",
            f"  Megapixels: {input_mp:.2f} MP",
            f"  Batch size: {batch_size}",
            "",
            "UPSCALING:",
            f"  Requested: {upscale_by}x",
            f"  Actual: {actual_upscale:.4f}x" + (" [ADJUSTED]" if abs(actual_upscale - upscale_by) > 0.001 else ""),
            f"  Tolerance: +/-{upscale_tolerance*100:.0f}%",
            f"  Method: {upscale_method}",
            "",
            "OUTPUT IMAGE:",
            f"  Target: {target_output_w} x {target_output_h} px",
            f"  Final: {output_w} x {output_h} px [{mode_label}]",
            f"  Megapixels: {output_mp:.2f} MP",
            f"  Size change: {((output_w * output_h) / (target_output_w * target_output_h) - 1) * 100:+.1f}%",
            "",
            "GRID-LOCK SOLUTION:",
            f"  Grid: {tiles_x} x {tiles_y} = {total_tiles} tiles",
            f"  Tile size: {tile_width} x {tile_height} px ({tile_mp:.2f} MP)",
            f"  Overlap: {overlap} px (min was {min_overlap}px)",
            f"  Divisibility: {divisibility} (aligned: W={tile_width % divisibility == 0}, H={tile_height % divisibility == 0})",
            f"  Latent size: {latent_tile_w} x {latent_tile_h}",
            "",
            "COVERAGE VERIFICATION:",
            f"  Coverage: {coverage_w} x {coverage_h} px",
            f"  Output: {output_w} x {output_h} px",
            f"  Coverage OK: {coverage_ok}",
            "",
            f"EFFICIENCY: {efficiency:.1f}%",
            "",
            "CONSTRAINTS:",
            f"  Output Mode: {output_mode}",
            f"  Target MP: {target_mp} +/-{mp_tolerance*100:.0f}%",
            f"  AR Constraint: {ar_constraint}",
            f"  Candidates evaluated: {len(all_candidates)}",
            top_candidates_str,
            "RECOMMENDED USDU SETTINGS:",
            f"  tile_width: {tile_width}",
            f"  tile_height: {tile_height}",
            f"  tile_padding: {overlap}",
            f"  upscale_by: {actual_upscale:.4f}",
            "=" * 60,
        ]

        debug_info = "\n".join(debug_lines)
        print(debug_info)

        # ============================================================
        # 7. RETURN ALL OUTPUTS
        # ============================================================
        return (
            upscaled_image,
            tile_width,
            tile_height,
            overlap,
            output_w,
            output_h,
            tiles_x,
            tiles_y,
            total_tiles,
            actual_upscale,
            efficiency,
            debug_info,
        )


# Node mappings for ComfyUI - NOT REGISTERED (backup file only)
# NODE_CLASS_MAPPINGS = {
#     "ArchAi3D_Smart_Tile_Solver_V6_Old": ArchAi3D_Smart_Tile_Solver_V6_Old
# }
#
# NODE_DISPLAY_NAME_MAPPINGS = {
#     "ArchAi3D_Smart_Tile_Solver_V6_Old": "🧩 Smart Tile Solver V6 (Original)"
# }
