"""
USDU Edge Repair - Main Node
=============================

ComfyUI node for tile-based image processing with USDU.
Uses TileGeometry for all geometry calculations.
"""

import logging
import torch
import torch.nn.functional as F

from .constants import MODES, SEAM_FIX_MODES
from .inputs import USDU_edge_repair_inputs, prepare_inputs
from .validation import validate_safeguard
from .diffdiff import DifferentialDiffusionAdvanced
from .preview import generate_tile_previews
from .tile_geometry import TileGeometry
from .utils import tensor_to_pil, pil_to_tensor
from .processing import StableDiffusionProcessing
from .usdu_patch import usdu
from . import shared
from .upscaler import UpscalerData


class ArchAi3D_USDU_EdgeRepair:
    """
    USDU Edge Repair - ComfyUI Node for tile-based image processing.

    Features:
        1. Per-tile Conditioning via CONDITIONING_LIST
        2. Differential Diffusion via mask
        3. Per-tile ControlNet
        4. Safeguard Validation
        5. Preview mode for debugging

    Uses TileGeometry for unified geometry calculations.
    """

    @classmethod
    def INPUT_TYPES(s):
        required, optional = USDU_edge_repair_inputs()
        return prepare_inputs(required, optional)

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("image", "tiles_original", "tiles_padded", "tiles_blend_mask", "tiles_edge_mask", "tiles_processed", "debug_info")
    FUNCTION = "upscale"
    CATEGORY = "ArchAi3d/Upscaling/USDU"

    def upscale(self, upscaled_image, output_width, output_height, tiles_x, tiles_y, safe_guard,
                enable_diffdiff, enable_controlnet, preview_mode, model, conditionings, negative, vae, seed,
                steps, cfg, sampler_name, scheduler, denoise,
                mode_type, tile_width, tile_height, mask_blur, tile_padding,
                edge_mask_width, edge_mask_feather, use_edge_mask_diffdiff,
                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding, tiled_decode,
                denoise_mask=None, multiplier=1.0,
                model_patch=None, control_image=None, control_strength=1.0, control_mask=None):
        """Main execution function."""

        # Store params
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.mask_blur = mask_blur
        self.tile_padding = tile_padding
        self.seam_fix_width = seam_fix_width
        self.seam_fix_denoise = seam_fix_denoise
        self.seam_fix_padding = seam_fix_padding
        self.seam_fix_mode = seam_fix_mode
        self.mode_type = mode_type
        self.seam_fix_mask_blur = seam_fix_mask_blur

        # Validation
        if safe_guard:
            validate_safeguard(upscaled_image, output_width, output_height,
                              tile_width, tile_height, tile_padding, tiles_x, tiles_y)

        # Get original dimensions
        orig_h, orig_w = upscaled_image.shape[1], upscaled_image.shape[2]

        # Create TileGeometry - single source of truth for all geometry
        geometry = TileGeometry(
            original_size=(orig_h, orig_w),
            tile_width=tile_width,
            tile_height=tile_height,
            tiles_x=tiles_x,
            tiles_y=tiles_y,
            tile_padding=tile_padding,
            mask_blur=mask_blur
        )

        # Note: pad_image() is now a no-op (mirror padding removed)
        # Kept for API compatibility but returns input unchanged
        upscaled_image = geometry.pad_image(upscaled_image)
        img_h, img_w = upscaled_image.shape[1], upscaled_image.shape[2]

        # Stats
        total_tiles = geometry.total_tiles
        num_conditionings = len(conditionings) if isinstance(conditionings, list) else 1

        # Log geometry info (mirror padding removed - edge tiles have less context)
        print(f"[TileGeometry] Canvas: {img_w}x{img_h}, Grid: {geometry.tiles_x}x{geometry.tiles_y} = {total_tiles} tiles")

        # Debug output
        debug_info = self._build_debug_info(
            geometry=geometry,
            output_width=output_width,
            output_height=output_height,
            num_conditionings=num_conditionings,
            conditionings=conditionings,
            mode_type=mode_type,
            steps=steps,
            cfg=cfg,
            denoise=denoise,
            sampler_name=sampler_name,
            scheduler=scheduler,
            seed=seed,
            safe_guard=safe_guard,
            enable_diffdiff=enable_diffdiff,
            denoise_mask=denoise_mask,
            multiplier=multiplier,
            enable_controlnet=enable_controlnet,
            model_patch=model_patch,
            control_image=control_image,
            control_strength=control_strength,
            control_mask=control_mask,
            edge_mask_width=edge_mask_width,
            edge_mask_feather=edge_mask_feather,
            preview_mode=preview_mode,
            seam_fix_mode=seam_fix_mode,
            seam_fix_denoise=seam_fix_denoise,
            seam_fix_width=seam_fix_width,
            seam_fix_padding=seam_fix_padding,
            seam_fix_mask_blur=seam_fix_mask_blur,
            tiled_decode=tiled_decode,
        )
        print("\n" + debug_info + "\n")

        # Preview mode
        preview_batches = None
        if preview_mode:
            preview_batches = generate_tile_previews(
                geometry, upscaled_image,
                edge_mask_width, edge_mask_feather, debug_info
            )

        # DiffDiff setup - scale mask to padded dimensions
        denoise_mask_upscaled = None
        if enable_diffdiff and denoise_mask is not None:
            mask_2d = denoise_mask[0] if denoise_mask.dim() == 3 else denoise_mask
            # First scale to original output size
            denoise_mask_upscaled = F.interpolate(
                mask_2d.unsqueeze(0).unsqueeze(0),
                size=(orig_h, orig_w), mode='bilinear', align_corners=False
            ).squeeze(0).squeeze(0)
            # Pad mask using geometry
            denoise_mask_upscaled = geometry.pad_mask(denoise_mask_upscaled)

            diff_diff = DifferentialDiffusionAdvanced()
            diff_diff.multiplier = multiplier
            model = model.clone()
            model.set_model_denoise_mask_function(diff_diff.forward)

        # ControlNet setup - scale to padded dimensions
        control_image_upscaled = None
        control_mask_upscaled = None
        controlnet_enabled = enable_controlnet and model_patch is not None and control_image is not None
        if controlnet_enabled:
            # First scale to original output size
            control_image_upscaled = F.interpolate(
                control_image.movedim(-1, 1),
                size=(orig_h, orig_w), mode='bilinear', align_corners=False
            ).movedim(1, -1)
            # Pad image using geometry
            control_image_upscaled = geometry.pad_image(control_image_upscaled)

            if control_mask is not None:
                ctrl_mask_2d = control_mask[0] if control_mask.dim() == 3 else control_mask
                control_mask_upscaled = F.interpolate(
                    ctrl_mask_2d.unsqueeze(0).unsqueeze(0),
                    size=(orig_h, orig_w), mode='bilinear', align_corners=False
                ).squeeze(0).squeeze(0)
                # Pad mask using geometry
                control_mask_upscaled = geometry.pad_mask(control_mask_upscaled)

        # Setup shared state
        shared.sd_upscalers[0] = UpscalerData()
        shared.actual_upscaler = None
        shared.batch = [tensor_to_pil(upscaled_image, i) for i in range(len(upscaled_image))]
        shared.batch_as_tensor = upscaled_image

        # Create processing object with geometry
        sdprocessing = StableDiffusionProcessing(
            shared.batch[0], model, conditionings, negative, vae,
            seed, steps, cfg, sampler_name, scheduler, denoise, 1.0, True, tiled_decode,
            tile_width, tile_height, MODES[self.mode_type], SEAM_FIX_MODES[self.seam_fix_mode],
            None, None,
            denoise_mask_tensor=denoise_mask_upscaled,
            model_patch=model_patch if controlnet_enabled else None,
            control_image_tensor=control_image_upscaled if controlnet_enabled else None,
            control_strength=control_strength if controlnet_enabled else 1.0,
            control_mask_tensor=control_mask_upscaled if controlnet_enabled else None,
            geometry=geometry,
            # Edge mask for DiffDiff (functional feature)
            use_edge_mask_diffdiff=use_edge_mask_diffdiff and enable_diffdiff,
            edge_mask_width=edge_mask_width,
            edge_mask_feather=edge_mask_feather,
        )

        # Run USDU
        logger = logging.getLogger()
        old_level = logger.getEffectiveLevel()
        logger.setLevel(logging.CRITICAL + 1)
        try:
            script = usdu.Script()
            processed = script.run(
                p=sdprocessing, _=None, tile_width=self.tile_width, tile_height=self.tile_height,
                mask_blur=self.mask_blur, padding=self.tile_padding, seams_fix_width=self.seam_fix_width,
                seams_fix_denoise=self.seam_fix_denoise, seams_fix_padding=self.seam_fix_padding,
                upscaler_index=0, save_upscaled_image=False, redraw_mode=MODES[self.mode_type],
                save_seams_fix_image=False, seams_fix_mask_blur=self.seam_fix_mask_blur,
                seams_fix_type=SEAM_FIX_MODES[self.seam_fix_mode], target_size_type=2,
                custom_width=None, custom_height=None, custom_scale=1.0
            )

            images = [pil_to_tensor(img) for img in shared.batch]
            tensor = torch.cat(images, dim=0)

            # Crop back to original size using geometry
            tensor = geometry.crop_to_original(tensor)

            # Get processed tiles from StableDiffusionProcessing for debug output
            if sdprocessing._processed_tiles:
                batch_processed = torch.cat(sdprocessing._processed_tiles, dim=0)
            else:
                batch_processed = tensor[:1]  # Fallback

            # Get ACTUAL data from processing (not simulated)
            # tiles_padded: ACTUAL tile crops before sampling
            if sdprocessing._actual_tiles_input:
                batch_actual_tiles = torch.cat([pil_to_tensor(t) for t in sdprocessing._actual_tiles_input], dim=0)
            else:
                batch_actual_tiles = tensor[:1]  # Fallback

            # tiles_blend_mask: ACTUAL blurred masks used in compositing
            if sdprocessing._actual_masks:
                batch_actual_masks = torch.cat([pil_to_tensor(m.convert('RGB')) for m in sdprocessing._actual_masks], dim=0)
            else:
                batch_actual_masks = tensor[:1]  # Fallback

            if preview_mode and preview_batches is not None:
                # Use ACTUAL data for tiles_padded and tiles_blend_mask
                # preview_batches[1] = tiles_original (simulated, kept for reference)
                # preview_batches[4] = tiles_edge_mask (simulated, shows edge mask config)
                return (tensor, preview_batches[1], batch_actual_tiles, batch_actual_masks, preview_batches[4], batch_processed, preview_batches[5])
            else:
                empty_batch = tensor[:1]
                return (tensor, empty_batch, batch_actual_tiles, batch_actual_masks, empty_batch, batch_processed, debug_info)
        finally:
            logger.setLevel(old_level)

    def _extract_prompt_from_cond(self, cond):
        """
        Extract prompt text from a conditioning if available.

        Returns tuple: (prompt_text, tensor_shape, dict_keys)
        """
        prompt_text = None
        tensor_shape = None
        dict_keys = []

        if cond is None:
            return None, None, []

        # Conditioning format: [[tensor, dict], ...] or just the list
        try:
            # Get the first element's dict
            if isinstance(cond, list) and len(cond) > 0:
                first_item = cond[0]
                if isinstance(first_item, (list, tuple)):
                    # Get tensor shape
                    if len(first_item) > 0 and hasattr(first_item[0], 'shape'):
                        tensor_shape = tuple(first_item[0].shape)

                    # Get dict info
                    if len(first_item) > 1:
                        cond_dict = first_item[1]
                        if isinstance(cond_dict, dict):
                            dict_keys = list(cond_dict.keys())

                            # Try common keys where prompt text might be stored
                            for key in ['prompt', 'text', 'caption', 'description', 'positive', 'negative']:
                                if key in cond_dict:
                                    val = cond_dict[key]
                                    if isinstance(val, str):
                                        prompt_text = val
                                        break

                            # Check for any string values in the dict
                            if prompt_text is None:
                                for key, value in cond_dict.items():
                                    if isinstance(value, str) and len(value) > 5 and key not in ['model_name', 'model_type']:
                                        prompt_text = f"{value}"
                                        break
        except Exception:
            pass

        return prompt_text, tensor_shape, dict_keys

    def _build_debug_info(self, geometry, output_width, output_height,
                          num_conditionings, conditionings, mode_type, steps, cfg,
                          denoise, sampler_name, scheduler, seed, safe_guard,
                          enable_diffdiff, denoise_mask, multiplier,
                          enable_controlnet, model_patch, control_image, control_strength, control_mask,
                          edge_mask_width, edge_mask_feather, preview_mode,
                          seam_fix_mode, seam_fix_denoise, seam_fix_width, seam_fix_padding, seam_fix_mask_blur,
                          tiled_decode):
        """Build comprehensive debug info string."""
        lines = [
            "=" * 70,
            "🔧 USDU Edge Repair - Debug Info",
            "=" * 70,
        ]

        # ─────────────────────────────────────────────────────────────────────
        # SAFEGUARD & MODE STATUS
        # ─────────────────────────────────────────────────────────────────────
        lines.append("")
        lines.append("┌─ STATUS ─────────────────────────────────────────────────────────┐")
        lines.append(f"│  Safeguard:    {'✓ ENABLED' if safe_guard else '✗ DISABLED':<20} Preview Mode: {'✓ ON' if preview_mode else '✗ OFF':<15} │")
        lines.append(f"│  Tiled VAE:    {'✓ ENABLED' if tiled_decode else '✗ DISABLED':<20}                              │")
        lines.append("└──────────────────────────────────────────────────────────────────┘")

        # ─────────────────────────────────────────────────────────────────────
        # GEOMETRY INFO
        # ─────────────────────────────────────────────────────────────────────
        lines.append("")
        lines.append("┌─ GEOMETRY ────────────────────────────────────────────────────────┐")
        lines.append(f"│  Original Size:     {geometry.original_w}x{geometry.original_h:<30} │")
        lines.append(f"│  Canvas Size:       {geometry.canvas_w}x{geometry.canvas_h:<30} │")
        lines.append(f"│  Output Target:     {output_width}x{output_height:<30} │")
        lines.append(f"│  Note:              Edge tiles have less context (clamped to bounds) │")
        lines.append("│                                                                    │")
        lines.append(f"│  Tile Grid:         {geometry.tiles_x}x{geometry.tiles_y} = {geometry.total_tiles} tiles{' ' * (30 - len(str(geometry.total_tiles)))}│")
        lines.append(f"│  Tile Size:         {geometry.tile_width}x{geometry.tile_height:<30} │")
        lines.append(f"│  Tile Padding:      {geometry.tile_padding}px context around each tile{' ' * 17}│")
        lines.append(f"│  Padded Tile Size:  {geometry.padded_tile_size[0]}x{geometry.padded_tile_size[1]:<30} │")
        lines.append(f"│  Mask Blur:         {geometry.mask_blur}px{' ' * 40}│")
        lines.append("└──────────────────────────────────────────────────────────────────┘")

        # ─────────────────────────────────────────────────────────────────────
        # TILE GRID DETAIL
        # ─────────────────────────────────────────────────────────────────────
        lines.append("")
        lines.append("┌─ TILE GRID DETAIL ─────────────────────────────────────────────────┐")
        for i in range(geometry.total_tiles):
            xi, yi = geometry.get_tile_coords(i)
            rect = geometry.get_tile_rect(i)
            padded = geometry.get_padded_rect(i)
            # Identify border tiles
            borders = []
            if xi == 0:
                borders.append("L")
            if xi == geometry.tiles_x - 1:
                borders.append("R")
            if yi == 0:
                borders.append("T")
            if yi == geometry.tiles_y - 1:
                borders.append("B")
            border_str = f"[{''.join(borders)}]" if borders else "   "
            lines.append(f"│  Tile [{i}] ({xi},{yi}) {border_str}: rect=({rect[0]:4},{rect[1]:4},{rect[2]:4},{rect[3]:4}) padded=({padded[0]:4},{padded[1]:4},{padded[2]:4},{padded[3]:4}) │")
        lines.append("│  Legend: [L]=Left border, [R]=Right, [T]=Top, [B]=Bottom          │")
        lines.append("└──────────────────────────────────────────────────────────────────┘")

        # ─────────────────────────────────────────────────────────────────────
        # SAMPLING PARAMETERS
        # ─────────────────────────────────────────────────────────────────────
        lines.append("")
        lines.append("┌─ SAMPLING ────────────────────────────────────────────────────────┐")
        lines.append(f"│  Mode:         {mode_type:<50} │")
        lines.append(f"│  Sampler:      {sampler_name:<50} │")
        lines.append(f"│  Scheduler:    {scheduler:<50} │")
        lines.append(f"│  Steps:        {steps:<50} │")
        lines.append(f"│  CFG:          {cfg:<50} │")
        lines.append(f"│  Denoise:      {denoise:<50} │")
        lines.append(f"│  Seed:         {seed:<50} │")
        lines.append("└──────────────────────────────────────────────────────────────────┘")

        # ─────────────────────────────────────────────────────────────────────
        # CONDITIONING
        # ─────────────────────────────────────────────────────────────────────
        lines.append("")
        lines.append("┌─ CONDITIONING ────────────────────────────────────────────────────┐")
        cond_match = "✓ MATCH" if num_conditionings == geometry.total_tiles else f"⚠ MISMATCH"
        lines.append(f"│  Provided:     {num_conditionings} conditionings{' ' * 38}│")
        lines.append(f"│  Expected:     {geometry.total_tiles} (one per tile){' ' * 36}│")
        lines.append(f"│  Status:       {cond_match:<50} │")
        if num_conditionings != geometry.total_tiles:
            if num_conditionings == 1:
                lines.append(f"│  Note:         Single conditioning will be used for ALL tiles       │")
            else:
                lines.append(f"│  Warning:      Conditioning count doesn't match tile count!         │")
        lines.append("└──────────────────────────────────────────────────────────────────┘")

        # ─────────────────────────────────────────────────────────────────────
        # PER-TILE PROMPTS
        # ─────────────────────────────────────────────────────────────────────
        lines.append("")
        lines.append("┌─ PER-TILE PROMPTS ──────────────────────────────────────────────────┐")

        # Get list of conditionings
        cond_list = conditionings if isinstance(conditionings, list) else [conditionings]

        # Show available keys from first conditioning (for debugging)
        if len(cond_list) > 0:
            _, _, first_keys = self._extract_prompt_from_cond(cond_list[0])
            if first_keys:
                keys_str = ", ".join(first_keys[:5])  # Show first 5 keys
                if len(first_keys) > 5:
                    keys_str += f", ... (+{len(first_keys)-5} more)"
                lines.append(f"│  Dict keys: [{keys_str}]{' ' * max(0, 53 - len(keys_str))}│")
                lines.append("│" + "─" * 68 + "│")

        for i in range(geometry.total_tiles):
            xi, yi = geometry.get_tile_coords(i)
            # Border info
            borders = []
            if xi == 0:
                borders.append("L")
            if xi == geometry.tiles_x - 1:
                borders.append("R")
            if yi == 0:
                borders.append("T")
            if yi == geometry.tiles_y - 1:
                borders.append("B")
            border_str = f"[{''.join(borders)}]" if borders else "   "

            # Get the conditioning for this tile
            cond_idx = i if i < len(cond_list) else 0
            cond = cond_list[cond_idx]

            # Try to extract prompt info
            prompt_text, tensor_shape, dict_keys = self._extract_prompt_from_cond(cond)

            # Build tile header
            tile_header = f"│  Tile [{i}] ({xi},{yi}) {border_str}"

            if prompt_text:
                # Show prompt text (potentially multi-line for long prompts)
                lines.append(f"{tile_header}:{' ' * (68 - len(tile_header))}│")

                # Split long prompts into multiple lines
                max_line_len = 64
                prompt_lines = []
                remaining = prompt_text
                while remaining:
                    if len(remaining) <= max_line_len:
                        prompt_lines.append(remaining)
                        break
                    else:
                        # Find a good break point
                        break_at = remaining.rfind(' ', 0, max_line_len)
                        if break_at == -1:
                            break_at = max_line_len
                        prompt_lines.append(remaining[:break_at])
                        remaining = remaining[break_at:].lstrip()

                for j, line in enumerate(prompt_lines[:3]):  # Max 3 lines per tile
                    prefix = '    "' if j == 0 else '     '
                    suffix = '"' if j == len(prompt_lines) - 1 or j == 2 else ''
                    if j == 2 and len(prompt_lines) > 3:
                        suffix = '..."'
                    display_line = f"{prefix}{line}{suffix}"
                    lines.append(f"│{display_line:<68}│")
            else:
                # Show tensor shape info
                shape_info = f"tensor {tensor_shape}" if tensor_shape else f"cond #{cond_idx}"
                lines.append(f"{tile_header}: {shape_info:<{68 - len(tile_header) - 2}}│")

        lines.append("│" + "─" * 68 + "│")
        lines.append("│  💡 To show prompts: store 'prompt' key in conditioning dict        │")
        lines.append("└──────────────────────────────────────────────────────────────────────┘")

        # ─────────────────────────────────────────────────────────────────────
        # DIFFDIFF
        # ─────────────────────────────────────────────────────────────────────
        lines.append("")
        lines.append("┌─ DIFFERENTIAL DIFFUSION ──────────────────────────────────────────┐")
        diffdiff_active = enable_diffdiff and denoise_mask is not None
        if diffdiff_active:
            mask_shape = tuple(denoise_mask.shape) if denoise_mask is not None else "N/A"
            lines.append(f"│  Status:       ✓ ENABLED{' ' * 40}│")
            lines.append(f"│  Mask Shape:   {str(mask_shape):<50} │")
            lines.append(f"│  Multiplier:   {multiplier:<50} │")
            lines.append(f"│  Effect:       White=more denoise, Black=less denoise{' ' * 11}│")
        else:
            if not enable_diffdiff:
                lines.append(f"│  Status:       ✗ DISABLED (toggle off){' ' * 26}│")
            else:
                lines.append(f"│  Status:       ✗ DISABLED (no mask provided){' ' * 21}│")
        lines.append("└──────────────────────────────────────────────────────────────────┘")

        # ─────────────────────────────────────────────────────────────────────
        # CONTROLNET
        # ─────────────────────────────────────────────────────────────────────
        lines.append("")
        lines.append("┌─ CONTROLNET ──────────────────────────────────────────────────────┐")
        controlnet_active = enable_controlnet and model_patch is not None and control_image is not None
        if controlnet_active:
            ctrl_shape = tuple(control_image.shape) if control_image is not None else "N/A"
            ctrl_mask_info = f"Shape: {tuple(control_mask.shape)}" if control_mask is not None else "None"
            lines.append(f"│  Status:       ✓ ENABLED{' ' * 40}│")
            lines.append(f"│  Image Shape:  {str(ctrl_shape):<50} │")
            lines.append(f"│  Strength:     {control_strength:<50} │")
            lines.append(f"│  Mask:         {ctrl_mask_info:<50} │")
        else:
            reasons = []
            if not enable_controlnet:
                reasons.append("toggle off")
            if model_patch is None:
                reasons.append("no model_patch")
            if control_image is None:
                reasons.append("no control_image")
            reason_str = ", ".join(reasons) if reasons else "unknown"
            lines.append(f"│  Status:       ✗ DISABLED ({reason_str}){' ' * max(0, 35 - len(reason_str))}│")
        lines.append("└──────────────────────────────────────────────────────────────────┘")

        # ─────────────────────────────────────────────────────────────────────
        # EDGE MASK (for DiffDiff tile edges)
        # ─────────────────────────────────────────────────────────────────────
        lines.append("")
        lines.append("┌─ EDGE MASK (Preview) ─────────────────────────────────────────────┐")
        lines.append(f"│  Edge Width:   {edge_mask_width}px{' ' * 45}│")
        lines.append(f"│  Edge Feather: {edge_mask_feather}px{' ' * 45}│")
        lines.append(f"│  Purpose:      Shows tile borders for DiffDiff mask design{' ' * 6}│")
        lines.append("└──────────────────────────────────────────────────────────────────┘")

        # ─────────────────────────────────────────────────────────────────────
        # SEAM FIX
        # ─────────────────────────────────────────────────────────────────────
        lines.append("")
        lines.append("┌─ SEAM FIX ────────────────────────────────────────────────────────┐")
        seam_active = seam_fix_mode != "None"
        if seam_active:
            lines.append(f"│  Status:       ✓ ENABLED{' ' * 40}│")
            lines.append(f"│  Mode:         {seam_fix_mode:<50} │")
            lines.append(f"│  Width:        {seam_fix_width}px{' ' * 45}│")
            lines.append(f"│  Padding:      {seam_fix_padding}px{' ' * 45}│")
            lines.append(f"│  Denoise:      {seam_fix_denoise:<50} │")
            lines.append(f"│  Mask Blur:    {seam_fix_mask_blur}px{' ' * 45}│")
        else:
            lines.append(f"│  Status:       ✗ DISABLED (mode=None){' ' * 27}│")
        lines.append("└──────────────────────────────────────────────────────────────────┘")

        # ─────────────────────────────────────────────────────────────────────
        # MEMORY ESTIMATES
        # ─────────────────────────────────────────────────────────────────────
        lines.append("")
        lines.append("┌─ MEMORY ESTIMATES ────────────────────────────────────────────────┐")
        # Canvas image: BHWC float32 = B * H * W * 4 * 4 bytes
        canvas_mem = geometry.canvas_w * geometry.canvas_h * 4 * 4 / (1024 * 1024)
        # Padded tile: same calculation
        pw, ph = geometry.padded_tile_size
        tile_mem = pw * ph * 4 * 4 / (1024 * 1024)
        # Latent (1/8 size, 4 channels)
        latent_mem = (pw // 8) * (ph // 8) * 4 * 4 / (1024 * 1024)
        lines.append(f"│  Canvas Image:    ~{canvas_mem:.1f} MB (float32){' ' * 30}│")
        lines.append(f"│  Per Tile Image:  ~{tile_mem:.1f} MB (float32){' ' * 30}│")
        lines.append(f"│  Per Tile Latent: ~{latent_mem:.2f} MB (float32, 1/8 scale){' ' * 20}│")
        lines.append(f"│  Note: Actual VRAM usage depends on model and batch size{' ' * 8}│")
        lines.append("└──────────────────────────────────────────────────────────────────┘")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)


# ComfyUI Node Registration
NODE_CLASS_MAPPINGS = {
    "ArchAi3D_USDU_EdgeRepair": ArchAi3D_USDU_EdgeRepair,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_USDU_EdgeRepair": "USDU Edge Repair",
}
