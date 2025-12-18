# Smart USDU Differential Diffusion - ComfyUI Node
#
# A copy of Smart Ultimate SD Upscale with Differential Diffusion support.
# Adds optional MASK + multiplier for per-pixel denoise control.
#
# KEEPS ALL FEATURES from Smart Ultimate SD Upscale!
# ADDS: Optional denoise_mask (MASK) + multiplier parameter
#
# Based on: Smart Ultimate SD Upscale + KJNodes DifferentialDiffusionAdvanced
# Modified by Amir Ferdos (ArchAi3d)
#
# Version: 1.0.0 - Initial release with Differential Diffusion support
# License: Dual License (Free for personal use, Commercial license required for business use)

import logging
import math
import torch
import torch.nn.functional as F
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


def USDU_base_inputs():
    required = [
        ("image", ("IMAGE",)),
        # Sampling Params
        ("model", ("MODEL",)),
        ("conditionings", ("CONDITIONING_LIST",)),  # CHANGED: Per-tile conditionings!
        ("negative", ("CONDITIONING",)),
        ("vae", ("VAE",)),
        ("upscale_by", ("FLOAT", {"default": 2, "min": 0.05, "max": 4, "step": 0.05})),
        ("seed", ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})),
        ("steps", ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1})),
        ("cfg", ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0})),
        ("sampler_name", (comfy.samplers.KSampler.SAMPLERS,)),
        ("scheduler", (comfy.samplers.KSampler.SCHEDULERS,)),
        ("denoise", ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01})),
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

    # NEW: Optional Differential Diffusion inputs
    optional = [
        ("denoise_mask", ("MASK", {
            "tooltip": "Optional mask for per-pixel denoise. White=more denoise, Black=less"
        })),
        ("multiplier", ("FLOAT", {
            "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001,
            "tooltip": "Controls effect strength. <1=stronger, >1=weaker"
        })),
    ]

    return required, optional


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


def remove_input(inputs: list, input_name: str):
    for i, (n, _) in enumerate(inputs):
        if n == input_name:
            del inputs[i]
            break


def rename_input(inputs: list, old_name: str, new_name: str):
    for i, (n, t) in enumerate(inputs):
        if n == old_name:
            inputs[i] = (new_name, t)
            break


# ============================================================
# Copied EXACTLY from KJNodes DifferentialDiffusionAdvanced
# Source: comfyui-kjnodes/nodes/nodes.py lines 1743-1779
# DO NOT MODIFY THIS CODE
# ============================================================
class DifferentialDiffusionAdvanced():
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL", ),
                    "samples": ("LATENT",),
                    "mask": ("MASK",),
                    "multiplier": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001}),
                            }}
    RETURN_TYPES = ("MODEL", "LATENT")
    FUNCTION = "apply"
    CATEGORY = "_for_testing"
    INIT = False

    def apply(self, model, samples, mask, multiplier):
        self.multiplier = multiplier
        model = model.clone()
        model.set_model_denoise_mask_function(self.forward)
        s = samples.copy()
        s["noise_mask"] = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
        return (model, s)

    def forward(self, sigma: torch.Tensor, denoise_mask: torch.Tensor, extra_options: dict):
        model = extra_options["model"]
        step_sigmas = extra_options["sigmas"]
        sigma_to = model.inner_model.model_sampling.sigma_min
        if step_sigmas[-1] > sigma_to:
            sigma_to = step_sigmas[-1]
        sigma_from = step_sigmas[0]

        ts_from = model.inner_model.model_sampling.timestep(sigma_from)
        ts_to = model.inner_model.model_sampling.timestep(sigma_to)
        current_ts = model.inner_model.model_sampling.timestep(sigma[0])

        threshold = (current_ts - ts_to) / (ts_from - ts_to) / self.multiplier

        return (denoise_mask >= threshold).to(denoise_mask.dtype)
# ============================================================
# END OF KJNODES CODE
# ============================================================


class ArchAi3D_Smart_USDU_DiffDiffusion:
    """
    Smart USDU with Differential Diffusion support.

    KEEPS ALL FEATURES from Smart Ultimate SD Upscale!
    ADDS: Optional denoise_mask + multiplier for per-pixel denoise control.

    Without mask: Behaves exactly like Smart Ultimate SD Upscale
    With mask:
      - White (mask=1): More denoise (more change)
      - Black (mask=0): Less denoise (preserved)
      - Multiplier < 1: Stronger effect
      - Multiplier > 1: Weaker effect

    Tile order: row-major (left-to-right, top-to-bottom)
    Same order as Smart Tile Prompter output.
    """

    @classmethod
    def INPUT_TYPES(s):
        required, optional = USDU_base_inputs()
        return prepare_inputs(required, optional)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "ArchAi3d/Upscaling"

    def upscale(self, image, model, conditionings, negative, vae, upscale_by, seed,
                steps, cfg, sampler_name, scheduler, denoise, upscale_model,
                mode_type, tile_width, tile_height, mask_blur, tile_padding,
                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode,
                custom_sampler=None, custom_sigmas=None,
                denoise_mask=None, multiplier=1.0):  # NEW: Differential Diffusion parameters
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

        # ===== DEBUG INFO =====
        input_h, input_w = image.shape[1], image.shape[2]
        output_w = int(input_w * upscale_by)
        output_h = int(input_h * upscale_by)
        num_conditionings = len(conditionings) if isinstance(conditionings, list) else 1

        # USDU formula: rows = ceil(height / tile_height), cols = ceil(width / tile_width)
        # padding is EXTRA CONTEXT around each tile, not spacing between tiles
        cols = math.ceil(output_w / tile_width)
        rows = math.ceil(output_h / tile_height)
        total_tiles = rows * cols

        # Actual sampling size (rounded up to 64 for VAE)
        sampling_width = math.ceil((tile_width + tile_padding) / 64) * 64
        sampling_height = math.ceil((tile_height + tile_padding) / 64) * 64

        print("\n" + "=" * 60)
        print("🎭 Smart USDU Differential Diffusion - DEBUG INFO")
        print("=" * 60)
        print(f"INPUT IMAGE:")
        print(f"  Size: {input_w}x{input_h}")
        print(f"  Upscale by: {upscale_by}x")
        print(f"  Output size: {output_w}x{output_h}")
        print(f"\nTILE SETTINGS:")
        print(f"  tile_width: {tile_width}px (visible area per tile)")
        print(f"  tile_height: {tile_height}px (visible area per tile)")
        print(f"  tile_padding: {tile_padding}px (extra context around each tile)")
        print(f"  mask_blur: {mask_blur}px (blending at tile edges)")
        print(f"\nCALCULATED GRID (USDU formula: ceil(size/tile_size)):")
        print(f"  Rows x Cols: {rows}x{cols} = {total_tiles} tiles")
        print(f"  Sampling size per tile: {sampling_width}x{sampling_height}px (rounded to 64)")
        print(f"  Context per tile: {tile_width}+{tile_padding}={tile_width+tile_padding}px")
        print(f"\nTILE OVERLAP EXPLANATION:")
        print(f"  - Tiles are placed {tile_width}px apart (NO overlap in placement)")
        print(f"  - Each tile SAMPLES {tile_padding}px extra context for blending")
        print(f"  - mask_blur={mask_blur}px feathers the edges for seamless blending")
        print(f"\nSEAM FIX SETTINGS:")
        print(f"  Mode: {seam_fix_mode}")
        print(f"  seam_fix_width: {seam_fix_width}px")
        print(f"  seam_fix_padding: {seam_fix_padding}px")
        print(f"  seam_fix_denoise: {seam_fix_denoise}")
        print(f"  seam_fix_mask_blur: {seam_fix_mask_blur}px")
        print(f"\nSAMPLING:")
        print(f"  Mode: {mode_type}")
        print(f"  Steps: {steps}, CFG: {cfg}, Denoise: {denoise}")
        print(f"  Sampler: {sampler_name}, Scheduler: {scheduler}")
        print(f"  Seed: {seed}")
        print(f"\nCONDITIONING:")
        print(f"  Number of conditionings: {num_conditionings}")
        print(f"  Expected tiles: {total_tiles}")
        if num_conditionings != total_tiles:
            print(f"  ⚠️ WARNING: Conditioning count ({num_conditionings}) != tile count ({total_tiles})")
            print(f"  (Fallback: will reuse last conditioning for extra tiles)")

        # ===== DIFFERENTIAL DIFFUSION INFO =====
        diff_diff_enabled = denoise_mask is not None
        print(f"\nDIFFERENTIAL DIFFUSION:")
        if diff_diff_enabled:
            print(f"  Status: ENABLED")
            print(f"  Multiplier: {multiplier}")
            print(f"  Mask shape: {denoise_mask.shape}")
            print(f"  Effect: White=more denoise, Black=less denoise")
        else:
            print(f"  Status: DISABLED (no mask provided)")
            print(f"  Behavior: Standard Smart USDU")
        print("=" * 60 + "\n")

        # ===== APPLY DIFFERENTIAL DIFFUSION IF MASK PROVIDED =====
        denoise_mask_upscaled = None
        if diff_diff_enabled:
            # Upscale mask to match output image size
            if denoise_mask.dim() == 3:
                mask_2d = denoise_mask[0]
            else:
                mask_2d = denoise_mask

            # Resize mask to output size (same as upscaled image)
            mask_upscaled = F.interpolate(
                mask_2d.unsqueeze(0).unsqueeze(0),
                size=(output_h, output_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)

            print(f"[DiffDiff] Mask upscaled: {mask_2d.shape} -> {mask_upscaled.shape}")

            # Use DifferentialDiffusionAdvanced to patch model
            diff_diff = DifferentialDiffusionAdvanced()
            diff_diff.multiplier = multiplier
            model = model.clone()
            model.set_model_denoise_mask_function(diff_diff.forward)

            print(f"[DiffDiff] Model patched with multiplier={multiplier}")

            # Store upscaled mask for per-tile processing
            denoise_mask_upscaled = mask_upscaled

        #
        # Set up A1111 patches
        #

        # Upscaler
        # An object that the script works with
        shared.sd_upscalers[0] = UpscalerData()
        # Where the actual upscaler is stored, will be used when the script upscales using the Upscaler in UpscalerData
        shared.actual_upscaler = upscale_model

        # Set the batch of images
        shared.batch = [tensor_to_pil(image, i) for i in range(len(image))]
        shared.batch_as_tensor = image

        # Processing - Pass conditionings list + optional denoise mask for Differential Diffusion
        sdprocessing = StableDiffusionProcessing(
            shared.batch[0], model, conditionings, negative, vae,
            seed, steps, cfg, sampler_name, scheduler, denoise, upscale_by, force_uniform_tiles, tiled_decode,
            tile_width, tile_height, MODES[self.mode_type], SEAM_FIX_MODES[self.seam_fix_mode],
            custom_sampler, custom_sigmas,
            denoise_mask_tensor=denoise_mask_upscaled,  # NEW: Pass upscaled mask for per-tile processing
        )

        # Disable logging
        logger = logging.getLogger()
        old_level = logger.getEffectiveLevel()
        logger.setLevel(logging.CRITICAL + 1)
        try:
            #
            # Running the script
            #
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


class ArchAi3D_Smart_USDU_DiffDiffusion_NoUpscale(ArchAi3D_Smart_USDU_DiffDiffusion):
    """Smart USDU DiffDiff variant that skips the initial upscale step."""

    @classmethod
    def INPUT_TYPES(s):
        required, optional = USDU_base_inputs()
        remove_input(required, "upscale_model")
        remove_input(required, "upscale_by")
        rename_input(required, "image", "upscaled_image")
        return prepare_inputs(required, optional)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "ArchAi3d/Upscaling"

    def upscale(self, upscaled_image, model, conditionings, negative, vae, seed,
                steps, cfg, sampler_name, scheduler, denoise,
                mode_type, tile_width, tile_height, mask_blur, tile_padding,
                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode,
                denoise_mask=None, multiplier=1.0):  # NEW: Differential Diffusion parameters
        upscale_by = 1.0
        return super().upscale(upscaled_image, model, conditionings, negative, vae, upscale_by, seed,
                               steps, cfg, sampler_name, scheduler, denoise, None,
                               mode_type, tile_width, tile_height, mask_blur, tile_padding,
                               seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                               seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode,
                               denoise_mask=denoise_mask, multiplier=multiplier)


class ArchAi3D_Smart_USDU_DiffDiffusion_CustomSample(ArchAi3D_Smart_USDU_DiffDiffusion):
    """Smart USDU DiffDiff variant with custom sampler and sigmas support."""

    @classmethod
    def INPUT_TYPES(s):
        required, optional = USDU_base_inputs()
        remove_input(required, "upscale_model")
        optional.append(("upscale_model", ("UPSCALE_MODEL",)))
        optional.append(("custom_sampler", ("SAMPLER",)))
        optional.append(("custom_sigmas", ("SIGMAS",)))
        return prepare_inputs(required, optional)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "ArchAi3d/Upscaling"

    def upscale(self, image, model, conditionings, negative, vae, upscale_by, seed,
                steps, cfg, sampler_name, scheduler, denoise,
                mode_type, tile_width, tile_height, mask_blur, tile_padding,
                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode,
                upscale_model=None,
                custom_sampler=None, custom_sigmas=None,
                denoise_mask=None, multiplier=1.0):  # NEW: Differential Diffusion parameters
        return super().upscale(image, model, conditionings, negative, vae, upscale_by, seed,
                steps, cfg, sampler_name, scheduler, denoise, upscale_model,
                mode_type, tile_width, tile_height, mask_blur, tile_padding,
                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode,
                custom_sampler, custom_sigmas,
                denoise_mask=denoise_mask, multiplier=multiplier)


# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "ArchAi3D_Smart_USDU_DiffDiffusion": ArchAi3D_Smart_USDU_DiffDiffusion,
    "ArchAi3D_Smart_USDU_DiffDiffusion_NoUpscale": ArchAi3D_Smart_USDU_DiffDiffusion_NoUpscale,
    "ArchAi3D_Smart_USDU_DiffDiffusion_CustomSample": ArchAi3D_Smart_USDU_DiffDiffusion_CustomSample
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_Smart_USDU_DiffDiffusion": "🎭 Smart USDU Differential Diffusion",
    "ArchAi3D_Smart_USDU_DiffDiffusion_NoUpscale": "🎭 Smart USDU DiffDiff (No Upscale)",
    "ArchAi3D_Smart_USDU_DiffDiffusion_CustomSample": "🎭 Smart USDU DiffDiff (Custom Sample)"
}
