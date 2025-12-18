# Smart USDU Mask Denoise - ComfyUI Node
#
# A variant of Smart Ultimate SD Upscale with PIXEL-LEVEL denoise control.
# Uses Differential Diffusion for SINGLE-PASS per-pixel denoise control.
#
# HOW IT WORKS:
#   - White regions (mask=1): HIGH denoise (more regeneration)
#   - Black regions (mask=0): LOW denoise (less regeneration, but still processed!)
#   - Gray regions: Interpolated denoise between high and low
#
# Based on Smart Ultimate SD Upscale + ComfyUI's DifferentialDiffusion
# Author: Amir Ferdos (ArchAi3d)
#
# Version: 3.0.0 - Differential Diffusion single-pass pixel-level denoise control
# License: Dual License (Free for personal use, Commercial license required for business use)

import logging
import math
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import comfy
from .smart_usdu.usdu_patch import usdu
from .smart_usdu.utils import tensor_to_pil, pil_to_tensor
from .smart_usdu.processing import StableDiffusionProcessing
from .smart_usdu import shared
from .smart_usdu.upscaler import UpscalerData

MAX_RESOLUTION = 8192

# The modes available for Ultimate SD Upscale
MODES = {
    "Linear": usdu.USDUMode.LINEAR,
    "Chess": usdu.USDUMode.CHESS,
    "None": usdu.USDUMode.NONE,
}

# The seam fix modes
SEAM_FIX_MODES = {
    "None": usdu.USDUSFMode.NONE,
    "Band Pass": usdu.USDUSFMode.BAND_PASS,
    "Half Tile": usdu.USDUSFMode.HALF_TILE,
    "Half Tile + Intersections": usdu.USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS,
}


def calculate_per_tile_denoise(mask_tensor, rows, cols, tile_width, tile_height,
                               output_w, output_h, denoise_masked, denoise_unmasked):
    """
    Calculate denoise value for each tile based on mask intensity.

    Args:
        mask_tensor: ComfyUI MASK tensor [B, H, W] or [H, W]
        rows, cols: Grid dimensions
        tile_width, tile_height: Size of each tile
        output_w, output_h: Output image size (after upscale)
        denoise_masked: Denoise for white (1.0) regions
        denoise_unmasked: Denoise for black (0.0) regions

    Returns:
        List of denoise values in row-major order (same as tile processing order)
    """
    # Handle batch dimension
    if mask_tensor.dim() == 3:
        mask = mask_tensor[0]  # Take first in batch
    else:
        mask = mask_tensor

    # Resize mask to output size (after upscale)
    mask_resized = F.interpolate(
        mask.unsqueeze(0).unsqueeze(0),
        size=(output_h, output_w),
        mode='bilinear',
        align_corners=False
    ).squeeze()

    denoise_list = []

    for row in range(rows):
        for col in range(cols):
            # Calculate tile bounds (using USDU's ceil formula)
            x1 = col * tile_width
            y1 = row * tile_height
            x2 = min(x1 + tile_width, output_w)
            y2 = min(y1 + tile_height, output_h)

            # Extract tile region from mask
            tile_mask = mask_resized[y1:y2, x1:x2]

            # Calculate average intensity (0-1)
            avg_intensity = tile_mask.mean().item()

            # Interpolate denoise: 0.0 (black) = unmasked, 1.0 (white) = masked
            tile_denoise = denoise_unmasked + avg_intensity * (denoise_masked - denoise_unmasked)
            denoise_list.append(tile_denoise)

    return denoise_list


def prepare_inputs(required: list, optional: list = None):
    inputs = {}
    if required:
        inputs["required"] = {}
        for name, type in required:
            inputs["required"][name] = type
    if optional:
        inputs["optional"] = {}
        for name, type in optional:
            inputs["optional"][name] = type
    return inputs


class ArchAi3D_Smart_USDU_Mask_Denoise:
    """
    Smart USDU with PIXEL-LEVEL Denoise Control (Differential Diffusion).

    Uses ComfyUI's DifferentialDiffusion for SINGLE-PASS per-pixel control:
    - White (mask=1): Gets HIGH denoise (more regeneration)
    - Black (mask=0): Gets LOW denoise (less regeneration, but STILL PROCESSED!)
    - Gray (mask=0.5): Gets interpolated denoise between high and low

    SAME speed as regular USDU (single pass per tile)!

    Example use cases:
    - Exterior: High denoise on sky (0.6), low denoise on building (0.25)
    - Interior: High denoise on walls (0.5), low denoise on furniture (0.2)
    """

    @classmethod
    def INPUT_TYPES(s):
        required = [
            ("image", ("IMAGE",)),
            # Sampling Params
            ("model", ("MODEL",)),
            ("conditionings", ("CONDITIONING_LIST",)),
            ("negative", ("CONDITIONING",)),
            ("vae", ("VAE",)),
            ("upscale_by", ("FLOAT", {"default": 2, "min": 0.05, "max": 4, "step": 0.05})),
            ("seed", ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})),
            ("steps", ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1})),
            ("cfg", ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0})),
            ("sampler_name", (comfy.samplers.KSampler.SAMPLERS,)),
            ("scheduler", (comfy.samplers.KSampler.SCHEDULERS,)),
            # Mask-based denoise (replaces single denoise input)
            ("denoise_mask", ("MASK",)),
            ("denoise_masked", ("FLOAT", {
                "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": "Denoise for WHITE/masked regions (e.g., sky)"
            })),
            ("denoise_unmasked", ("FLOAT", {
                "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": "Denoise for BLACK/unmasked regions (e.g., building)"
            })),
            # Upscale Params
            ("upscale_model", ("UPSCALE_MODEL",)),
            ("mode_type", (list(MODES.keys()),)),
            ("tile_width", ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8})),
            ("tile_height", ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8})),
            ("mask_blur", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1})),
            ("tile_padding", ("INT", {"default": 32, "min": 0, "max": MAX_RESOLUTION, "step": 8})),
            # Seam fix params
            ("seam_fix_mode", (list(SEAM_FIX_MODES.keys()),)),
            ("seam_fix_denoise", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})),
            ("seam_fix_width", ("INT", {"default": 64, "min": 0, "max": MAX_RESOLUTION, "step": 8})),
            ("seam_fix_mask_blur", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1})),
            ("seam_fix_padding", ("INT", {"default": 16, "min": 0, "max": MAX_RESOLUTION, "step": 8})),
            # Misc
            ("force_uniform_tiles", ("BOOLEAN", {"default": True})),
            ("tiled_decode", ("BOOLEAN", {"default": False})),
        ]

        optional = []

        return prepare_inputs(required, optional)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "ArchAi3d/Upscaling"

    def upscale(self, image, model, conditionings, negative, vae, upscale_by, seed,
                steps, cfg, sampler_name, scheduler,
                denoise_mask, denoise_masked, denoise_unmasked,
                upscale_model, mode_type, tile_width, tile_height, mask_blur, tile_padding,
                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode):
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
        self.upscale_by = upscale_by
        self.seam_fix_mask_blur = seam_fix_mask_blur

        # Calculate grid dimensions
        input_h, input_w = image.shape[1], image.shape[2]
        output_w = int(input_w * upscale_by)
        output_h = int(input_h * upscale_by)
        num_conditionings = len(conditionings) if isinstance(conditionings, list) else 1

        cols = math.ceil(output_w / tile_width)
        rows = math.ceil(output_h / tile_height)
        total_tiles = rows * cols

        # Calculate per-tile denoise from mask
        denoise_list = calculate_per_tile_denoise(
            denoise_mask, rows, cols, tile_width, tile_height,
            output_w, output_h, denoise_masked, denoise_unmasked
        )

        # Use average denoise for compatibility with existing code paths
        avg_denoise = sum(denoise_list) / len(denoise_list)

        # ===== DEBUG INFO =====
        sampling_width = math.ceil((tile_width + tile_padding) / 64) * 64
        sampling_height = math.ceil((tile_height + tile_padding) / 64) * 64

        print("\n" + "=" * 60)
        print("🎭 Smart USDU Mask Denoise v3.0 - DIFFERENTIAL DIFFUSION")
        print("=" * 60)
        print(f"INPUT IMAGE:")
        print(f"  Size: {input_w}x{input_h}")
        print(f"  Upscale by: {upscale_by}x")
        print(f"  Output size: {output_w}x{output_h}")
        print(f"\nTILE SETTINGS:")
        print(f"  tile_width: {tile_width}px")
        print(f"  tile_height: {tile_height}px")
        print(f"  tile_padding: {tile_padding}px")
        print(f"  mask_blur: {mask_blur}px")
        print(f"\nCALCULATED GRID:")
        print(f"  Rows x Cols: {rows}x{cols} = {total_tiles} tiles")
        print(f"  Sampling size per tile: {sampling_width}x{sampling_height}px")
        print(f"\n⚡ DIFFERENTIAL DIFFUSION PIXEL-LEVEL DENOISE:")
        print(f"  WHITE regions (mask=1): {denoise_masked} denoise")
        print(f"  BLACK regions (mask=0): {denoise_unmasked} denoise")
        print(f"  GRAY regions: Interpolated denoise")
        print(f"  Method: Single-pass with per-pixel control (SAME speed as normal!)")
        print(f"\nSEAM FIX SETTINGS:")
        print(f"  Mode: {seam_fix_mode}")
        print(f"  seam_fix_denoise: {seam_fix_denoise}")
        print(f"\nSAMPLING:")
        print(f"  Mode: {mode_type}")
        print(f"  Steps: {steps}, CFG: {cfg}")
        print(f"  Sampler: {sampler_name}, Scheduler: {scheduler}")
        print(f"\nCONDITIONING:")
        print(f"  Number of conditionings: {num_conditionings}")
        print(f"  Expected tiles: {total_tiles}")
        if num_conditionings != total_tiles:
            print(f"  Warning: Conditioning count != tile count")
        print("=" * 60 + "\n")

        #
        # Set up A1111 patches
        #

        # Upscaler
        shared.sd_upscalers[0] = UpscalerData()
        shared.actual_upscaler = upscale_model

        # Set the batch of images
        shared.batch = [tensor_to_pil(image, i) for i in range(len(image))]
        shared.batch_as_tensor = image

        # Convert mask tensor to PIL for DIFFERENTIAL DIFFUSION pixel-level denoise
        # Handle batch dimension
        if denoise_mask.dim() == 3:
            mask_2d = denoise_mask[0]
        else:
            mask_2d = denoise_mask

        # Convert to numpy and then PIL (grayscale)
        mask_np = (mask_2d.cpu().numpy() * 255).astype(np.uint8)
        denoise_mask_pil = Image.fromarray(mask_np, mode='L')

        # Resize to output size (after upscale) - same size as upscaled image
        denoise_mask_pil = denoise_mask_pil.resize((output_w, output_h), Image.Resampling.BILINEAR)

        print(f"  DIFF DIFFUSION MASK: {denoise_mask_pil.size}, high={denoise_masked}, low={denoise_unmasked}")

        # Processing - Pass mask + high/low denoise for Differential Diffusion pixel-level control
        sdprocessing = StableDiffusionProcessing(
            shared.batch[0], model, conditionings, negative, vae,
            seed, steps, cfg, sampler_name, scheduler, avg_denoise, upscale_by, force_uniform_tiles, tiled_decode,
            tile_width, tile_height, MODES[self.mode_type], SEAM_FIX_MODES[self.seam_fix_mode],
            None, None,  # custom_sampler, custom_sigmas
            denoise_list,  # per-tile denoise list (for compatibility)
            denoise_mask_pil,  # PIL mask for pixel-level blending
            denoise_masked,    # HIGH denoise for white regions
            denoise_unmasked,  # LOW denoise for black regions
        )

        # Disable logging
        logger = logging.getLogger()
        old_level = logger.getEffectiveLevel()
        logger.setLevel(logging.CRITICAL + 1)
        try:
            # Running the script
            script = usdu.Script()
            processed = script.run(p=sdprocessing, _=None, tile_width=self.tile_width, tile_height=self.tile_height,
                               mask_blur=self.mask_blur, padding=self.tile_padding, seams_fix_width=self.seam_fix_width,
                               seams_fix_denoise=self.seam_fix_denoise, seams_fix_padding=self.seam_fix_padding,
                               upscaler_index=0, save_upscaled_image=False, redraw_mode=MODES[self.mode_type],
                               save_seams_fix_image=False, seams_fix_mask_blur=self.seam_fix_mask_blur,
                               seams_fix_type=SEAM_FIX_MODES[self.seam_fix_mode], target_size_type=2,
                               custom_width=None, custom_height=None, custom_scale=self.upscale_by)

            # Return the resulting images
            images = [pil_to_tensor(img) for img in shared.batch]
            tensor = torch.cat(images, dim=0)
            return (tensor,)
        finally:
            # Restore the original logging level
            logger.setLevel(old_level)


# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "ArchAi3D_Smart_USDU_Mask_Denoise": ArchAi3D_Smart_USDU_Mask_Denoise,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_Smart_USDU_Mask_Denoise": "🎭 Smart USDU Mask Denoise",
}
