# Smart USDU Split-Latent - Per-Pixel Denoise Control
#
# A standalone node implementing the Split-Latent workflow for per-pixel denoise control.
# Uses manual noise injection with mask-based blending before sampling.
#
# Author: Amir Ferdos (ArchAi3d)
# Version: 1.0.0
# License: Dual License (Free for personal use, Commercial license required for business use)

import logging
import math
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter
import comfy
from nodes import common_ksampler, VAEEncode, VAEDecode, VAEDecodeTiled
from comfy.samplers import calculate_sigmas
from .smart_usdu.usdu_patch import usdu
from .smart_usdu.utils import tensor_to_pil, pil_to_tensor, get_crop_region, expand_crop, crop_cond
from .smart_usdu import shared
from .smart_usdu.upscaler import UpscalerData
from tqdm import tqdm

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


def get_sigma_at_denoise(denoise_val, total_steps, sigmas):
    """Get sigma and start_index for a denoise value from the ACTUAL schedule.

    Args:
        denoise_val: Denoise strength (0.0 to 1.0)
        total_steps: Total sampling steps
        sigmas: The sigma schedule tensor

    Returns:
        Tuple of (sigma_value, start_index)
    """
    if denoise_val <= 0.0:
        return 0.0, total_steps
    if denoise_val >= 1.0:
        return sigmas[0].item(), 0

    # Calculate how many steps correspond to this denoise amount
    # e.g. 20 steps * 0.6 denoise = 12 steps of processing needed
    # So we start at step: 20 - 12 = 8
    remaining_steps = int(total_steps * denoise_val)
    start_index = total_steps - remaining_steps

    # Clamp index to be safe
    start_index = max(0, min(start_index, total_steps - 1))

    # Get the sigma at that exact step index
    return sigmas[start_index].item(), start_index


class ArchAi3D_Smart_USDU_Split_Latent:
    """
    Smart USDU Split-Latent - Per-Pixel Denoise Control

    Uses the Split-Latent workflow:
    1. VAE Encode tile to clean latent
    2. Generate ONE noise tensor (same seed for both streams)
    3. Create two noised latents: low (black mask) and high (white mask)
    4. Blend via mask before sampling
    5. Sample with add_noise=False and start_step based on max denoise

    Known limitation: Low-denoise regions may be "over-processed" (burn risk)
    because sampler starts at the high-denoise step.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("MODEL",),
                "conditionings": ("CONDITIONING_LIST",),  # Per-tile prompts from Smart Tile Conditioning
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "mask": ("MASK",),
                "denoise_high": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "denoise_low": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "upscale_by": ("FLOAT", {"default": 2, "min": 0.05, "max": 4, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "upscale_model": ("UPSCALE_MODEL",),
                "mode_type": (list(MODES.keys()),),
                "tile_width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "tile_height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "mask_blur": ("INT", {"default": 8, "min": 0, "max": 64, "step": 1}),
                "tile_padding": ("INT", {"default": 32, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "seam_fix_mode": (list(SEAM_FIX_MODES.keys()),),
                "seam_fix_denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seam_fix_width": ("INT", {"default": 64, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "seam_fix_mask_blur": ("INT", {"default": 8, "min": 0, "max": 64, "step": 1}),
                "seam_fix_padding": ("INT", {"default": 16, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "force_uniform_tiles": ("BOOLEAN", {"default": True}),
                "tiled_decode": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "ArchAi3d/Upscaling"

    def upscale(self, image, model, conditionings, negative, vae, mask,
                denoise_high, denoise_low, upscale_by, seed, steps, cfg,
                sampler_name, scheduler, upscale_model, mode_type,
                tile_width, tile_height, mask_blur, tile_padding,
                seam_fix_mode, seam_fix_denoise, seam_fix_width,
                seam_fix_mask_blur, seam_fix_padding, force_uniform_tiles, tiled_decode):

        # Convert mask tensor to PIL for cropping per tile
        # mask shape: [H, W] or [1, H, W] or [B, H, W]
        if mask.dim() == 2:
            mask_np = mask.cpu().numpy()
        elif mask.dim() == 3:
            mask_np = mask[0].cpu().numpy()
        else:
            mask_np = mask[0, 0].cpu().numpy()

        # Convert to 0-255 range PIL image
        mask_pil_input = Image.fromarray((mask_np * 255).astype(np.uint8), mode='L')

        # IMPORTANT: Resize mask to match OUTPUT (upscaled) dimensions!
        # USDU calculates crop regions at output resolution, so mask must match
        input_h, input_w = image.shape[1], image.shape[2]
        output_w = int(input_w * upscale_by)
        output_h = int(input_h * upscale_by)
        mask_pil = mask_pil_input.resize((output_w, output_h), Image.Resampling.BILINEAR)

        # ===== DEBUG INFO =====
        cols = math.ceil(output_w / tile_width)
        rows = math.ceil(output_h / tile_height)
        total_tiles = rows * cols

        print("\n" + "=" * 60)
        print("🚀 Smart USDU Split-Latent - DEBUG INFO")
        print("=" * 60)
        print(f"INPUT IMAGE:")
        print(f"  Size: {input_w}x{input_h}")
        print(f"  Upscale by: {upscale_by}x")
        print(f"  Output size: {output_w}x{output_h}")
        print(f"\nSPLIT-LATENT SETTINGS:")
        print(f"  denoise_high (white mask): {denoise_high}")
        print(f"  denoise_low (black mask): {denoise_low}")
        print(f"  mask size: {mask_pil.size}")
        print(f"\nTILE SETTINGS:")
        print(f"  tile_width: {tile_width}px")
        print(f"  tile_height: {tile_height}px")
        print(f"  Rows x Cols: {rows}x{cols} = {total_tiles} tiles")
        print(f"\nSAMPLING:")
        print(f"  Steps: {steps}, CFG: {cfg}")
        print(f"  Sampler: {sampler_name}, Scheduler: {scheduler}")
        print(f"  Seed: {seed}")
        print(f"\nCONDITIONING:")
        num_conditionings = len(conditionings) if isinstance(conditionings, list) else 1
        print(f"  Number of conditionings: {num_conditionings}")
        print(f"  Expected tiles: {total_tiles}")
        if num_conditionings != total_tiles:
            print(f"  WARNING: Conditioning count ({num_conditionings}) != tile count ({total_tiles})")
            print(f"  (Fallback: will reuse last conditioning for extra tiles)")
        print("=" * 60 + "\n")

        # Set up A1111 patches
        shared.sd_upscalers[0] = UpscalerData()
        shared.actual_upscaler = upscale_model

        # Set the batch of images
        shared.batch = [tensor_to_pil(image, i) for i in range(len(image))]
        shared.batch_as_tensor = image

        # Create processing object with split-latent params
        sdprocessing = SplitLatentProcessing(
            shared.batch[0], model, conditionings, negative, vae,
            seed, steps, cfg, sampler_name, scheduler,
            denoise_high,  # Use max denoise for base processing
            upscale_by, force_uniform_tiles, tiled_decode,
            tile_width, tile_height, MODES[mode_type], SEAM_FIX_MODES[seam_fix_mode],
            mask_pil=mask_pil,
            denoise_high_val=denoise_high,
            denoise_low_val=denoise_low,
        )

        # Patch the processing module to use our split-latent process_images
        from .smart_usdu import processing as proc_module
        original_process_images = proc_module.process_images
        proc_module.process_images = process_images

        # Disable logging
        logger = logging.getLogger()
        old_level = logger.getEffectiveLevel()
        logger.setLevel(logging.CRITICAL + 1)
        try:
            # Running the script
            script = usdu.Script()
            processed = script.run(
                p=sdprocessing, _=None, tile_width=tile_width, tile_height=tile_height,
                mask_blur=mask_blur, padding=tile_padding, seams_fix_width=seam_fix_width,
                seams_fix_denoise=seam_fix_denoise, seams_fix_padding=seam_fix_padding,
                upscaler_index=0, save_upscaled_image=False, redraw_mode=MODES[mode_type],
                save_seams_fix_image=False, seams_fix_mask_blur=seam_fix_mask_blur,
                seams_fix_type=SEAM_FIX_MODES[seam_fix_mode], target_size_type=2,
                custom_width=None, custom_height=None, custom_scale=upscale_by
            )

            # Return the resulting images
            images = [pil_to_tensor(img) for img in shared.batch]
            tensor = torch.cat(images, dim=0)
            return (tensor,)
        finally:
            # Restore the original processing function
            proc_module.process_images = original_process_images
            # Restore the original logging level
            logger.setLevel(old_level)


class SplitLatentProcessing:
    """
    Processing class that implements the Split-Latent workflow.

    This replaces the standard processing logic with:
    1. Per-tile mask cropping
    2. Dual noise injection (low/high)
    3. Mask-based latent blending
    4. Sampling with add_noise=False and proper start_step

    Also supports per-tile conditioning from Smart Tile Conditioning.
    """

    def __init__(self, init_img, model, positive_list, negative, vae,
                 seed, steps, cfg, sampler_name, scheduler, denoise,
                 upscale_by, uniform_tile_mode, tiled_decode,
                 tile_width, tile_height, redraw_mode, seam_fix_mode,
                 custom_sampler=None, custom_sigmas=None,
                 mask_pil=None, denoise_high_val=0.6, denoise_low_val=0.2):

        # Variables used by the USDU script
        self.init_images = [init_img]
        self.image_mask = None
        self.mask_blur = 0
        self.inpaint_full_res_padding = 0
        self.width = init_img.width * upscale_by
        self.height = init_img.height * upscale_by
        self.rows = round(self.height / tile_height)
        self.cols = round(self.width / tile_width)

        # Per-tile conditioning support (same pattern as original Smart USDU)
        self._positive_list = positive_list  # Store the list of conditionings
        self._current_tile_index = 0         # Track current tile
        self._negative = negative            # Store negative conditioning

        # Standard processing attributes
        self.model = model
        self.vae = vae
        self.seed = seed
        self.steps = steps
        self.cfg = cfg
        self.sampler_name = sampler_name
        self.scheduler = scheduler
        self.denoise = denoise

        # Split-latent specific
        self.mask_pil = mask_pil
        self.denoise_high = denoise_high_val
        self.denoise_low = denoise_low_val

        # Optional custom sampler
        self.custom_sampler = custom_sampler
        self.custom_sigmas = custom_sigmas

        # Variables used only by this script
        self.init_size = init_img.width, init_img.height
        self.upscale_by = upscale_by
        self.uniform_tile_mode = uniform_tile_mode
        self.tiled_decode = tiled_decode
        self.vae_decoder = VAEDecode()
        self.vae_encoder = VAEEncode()
        self.vae_decoder_tiled = VAEDecodeTiled()

        # Other required A1111 variables
        self.extra_generation_params = {}

        # Progress bar
        self.progress_bar_enabled = False
        if comfy.utils.PROGRESS_BAR_ENABLED:
            self.progress_bar_enabled = True
            comfy.utils.PROGRESS_BAR_ENABLED = False
            self.tiles = self.rows * self.cols
            self.pbar = None

        # ===== FIX: GENERATE GLOBAL NOISE ONCE =====
        # Generate noise for the ENTIRE upscaled resolution here.
        # This ensures tiles share the same noise structure at seams (no striping artifacts).
        latent_h = math.ceil(self.height / 8)
        latent_w = math.ceil(self.width / 8)

        # Detect latent channels from model (SD/SDXL=4, Flux=16)
        try:
            channels = model.get_model_object("latent_format").latent_channels
        except:
            channels = 4  # Fallback to SD/SDXL

        print(f"[Smart USDU Split-Latent] Generating Global Noise: {latent_w}x{latent_h} latent, {channels} channels (Seed: {seed})")

        # Generate on CPU to save VRAM, move to GPU only when slicing per tile
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.global_noise_tensor = torch.randn((channels, latent_h, latent_w), generator=generator, dtype=torch.float32)

        # Log conditioning info
        print(f"[Smart USDU Split-Latent] Tile conditionings: {len(self._positive_list)}, Grid: {self.cols}x{self.rows} = {self.rows * self.cols} tiles")

    @property
    def positive(self):
        """Return conditioning for current tile."""
        if self._current_tile_index < len(self._positive_list):
            return self._positive_list[self._current_tile_index]
        # Fallback to last conditioning if index exceeds list
        return self._positive_list[-1] if self._positive_list else None

    @positive.setter
    def positive(self, value):
        """Allow setting (for compatibility), but we ignore it."""
        pass

    @property
    def negative(self):
        """Return negative conditioning."""
        return self._negative

    @negative.setter
    def negative(self, value):
        """Allow setting negative conditioning."""
        self._negative = value

    def advance_tile(self):
        """Move to next tile's conditioning (called by USDU after each tile)."""
        self._current_tile_index += 1

    def reset_tile_index(self):
        """Reset tile index (for seams fix pass)."""
        self._current_tile_index = 0

    def __del__(self):
        if self.progress_bar_enabled:
            comfy.utils.PROGRESS_BAR_ENABLED = True


class Processed:
    def __init__(self, p, images, seed, info):
        self.images = images
        self.seed = seed
        self.info = info

    def infotext(self, p, index):
        return None


def fix_seed(p):
    pass


def process_images(p) -> Processed:
    """
    Main processing function with Split-Latent workflow.
    """
    # Show the progress bar
    if p.progress_bar_enabled and p.pbar is None:
        p.pbar = tqdm(total=p.tiles, desc='Smart USDU Split-Latent', unit='tile')

    # Setup
    image_mask = p.image_mask.convert('L')
    init_image = p.init_images[0]

    # Locate the white region of the mask outlining the tile
    crop_region = get_crop_region(image_mask, p.inpaint_full_res_padding)

    if p.uniform_tile_mode:
        x1, y1, x2, y2 = crop_region
        crop_width = x2 - x1
        crop_height = y2 - y1
        crop_ratio = crop_width / crop_height
        p_ratio = p.width / p.height
        if crop_ratio > p_ratio:
            target_width = crop_width
            target_height = round(crop_width / p_ratio)
        else:
            target_width = round(crop_height * p_ratio)
            target_height = crop_height
        crop_region, _ = expand_crop(crop_region, image_mask.width, image_mask.height, target_width, target_height)
        tile_size = p.width, p.height
    else:
        x1, y1, x2, y2 = crop_region
        crop_width = x2 - x1
        crop_height = y2 - y1
        target_width = math.ceil(crop_width / 8) * 8
        target_height = math.ceil(crop_height / 8) * 8
        crop_region, tile_size = expand_crop(crop_region, image_mask.width,
                                             image_mask.height, target_width, target_height)

    # Blur the mask
    if p.mask_blur > 0:
        image_mask = image_mask.filter(ImageFilter.GaussianBlur(p.mask_blur))

    # Crop the images to get the tiles
    tiles = [img.crop(crop_region) for img in shared.batch]
    initial_tile_size = tiles[0].size

    # Resize if necessary
    for i, tile in enumerate(tiles):
        if tile.size != tile_size:
            tiles[i] = tile.resize(tile_size, Image.Resampling.LANCZOS)

    # Crop conditioning
    positive_cropped = crop_cond(p.positive, crop_region, p.init_size, init_image.size, tile_size)
    negative_cropped = crop_cond(p.negative, crop_region, p.init_size, init_image.size, tile_size)

    # Encode the image
    batched_tiles = torch.cat([pil_to_tensor(tile) for tile in tiles], dim=0)
    (latent,) = p.vae_encoder.encode(p.vae, batched_tiles)

    # ===== SPLIT-LATENT WORKFLOW =====
    if p.mask_pil is not None:
        # Step 1: Crop and resize mask to tile/latent dimensions
        mask_cropped = p.mask_pil.crop(crop_region)
        if mask_cropped.size != tile_size:
            mask_cropped = mask_cropped.resize(tile_size, Image.Resampling.BILINEAR)

        mask_np = np.array(mask_cropped).astype(np.float32) / 255.0
        mask_tensor = torch.tensor(mask_np, dtype=torch.float32)

        latent_h, latent_w = latent["samples"].shape[-2], latent["samples"].shape[-1]
        mask_latent = F.interpolate(
            mask_tensor.unsqueeze(0).unsqueeze(0),
            size=(latent_h, latent_w),
            mode='bilinear',
            align_corners=False
        )

        device = latent["samples"].device
        dtype = latent["samples"].dtype
        mask_latent = mask_latent.to(device=device, dtype=dtype)
        mask_4d = mask_latent.expand_as(latent["samples"])

        # Step 2: Get clean latent
        clean_latent = latent["samples"].clone()

        # Step 3: Slice noise from GLOBAL tensor (FIX for tile seam artifacts)
        # Get tile position in latent space (pixels / 8)
        x1, y1, x2, y2 = crop_region
        lx, ly = int(x1 / 8), int(y1 / 8)
        lh, lw = latent["samples"].shape[-2], latent["samples"].shape[-1]

        # Slice from global noise tensor (generated once in __init__)
        noise_slice = p.global_noise_tensor[:, ly:ly+lh, lx:lx+lw]

        # Handle edge cases where slice might be slightly off due to rounding
        if noise_slice.shape[-1] != lw or noise_slice.shape[-2] != lh:
            noise_slice = F.interpolate(
                noise_slice.unsqueeze(0),
                size=(lh, lw),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        # Add batch dimension and move to device
        noise = noise_slice.unsqueeze(0).to(device=device, dtype=dtype)

        # Step 4: Calculate sigmas from ACTUAL schedule
        model_sampling = p.model.get_model_object("model_sampling")
        sigmas = calculate_sigmas(model_sampling, p.scheduler, p.steps)

        sigma_low, start_index_low = get_sigma_at_denoise(p.denoise_low, p.steps, sigmas)
        sigma_high, start_index_high = get_sigma_at_denoise(p.denoise_high, p.steps, sigmas)

        # Step 5: EPS formula - latent_noisy = clean + noise * sigma
        latent_low = clean_latent + noise * sigma_low
        latent_high = clean_latent + noise * sigma_high

        # Step 6: Composite with mask
        # mask=0 (black) -> latent_low, mask=1 (white) -> latent_high
        blended_latent = latent_low * (1.0 - mask_4d) + latent_high * mask_4d

        # Step 7: Determine Global Start Step
        # Must start at the earlier step (higher denoise = lower index)
        global_start_step = min(start_index_low, start_index_high)

        # Debug output
        print(f"  [SplitLatent] TILE {p._current_tile_index + 1} DEBUG:")
        print(f"    - Using conditioning #{p._current_tile_index + 1} of {len(p._positive_list)}")
        print(f"    - denoise_low={p.denoise_low:.3f} -> sigma={sigma_low:.4f}, start_idx={start_index_low}")
        print(f"    - denoise_high={p.denoise_high:.3f} -> sigma={sigma_high:.4f}, start_idx={start_index_high}")
        print(f"    - global_start_step={global_start_step}, steps={p.steps}")

        latent["samples"] = blended_latent

        # Step 8: Sample with add_noise=False and start_step
        (samples,) = common_ksampler(
            p.model, p.seed, p.steps, p.cfg, p.sampler_name, p.scheduler,
            positive_cropped, negative_cropped, latent,
            denoise=1.0, disable_noise=True, start_step=global_start_step
        )
    else:
        # Fallback: standard sampling
        (samples,) = common_ksampler(
            p.model, p.seed, p.steps, p.cfg, p.sampler_name, p.scheduler,
            positive_cropped, negative_cropped, latent, denoise=p.denoise
        )

    # Update the progress bar
    if p.progress_bar_enabled:
        p.pbar.update(1)

    # Decode the sample
    if not p.tiled_decode:
        (decoded,) = p.vae_decoder.decode(p.vae, samples)
    else:
        (decoded,) = p.vae_decoder_tiled.decode(p.vae, samples, 512)

    # Convert the sample to a PIL image
    tiles_sampled = [tensor_to_pil(decoded, i) for i in range(len(decoded))]

    for i, tile_sampled in enumerate(tiles_sampled):
        init_image = shared.batch[i]

        # Resize back to the original size
        if tile_sampled.size != initial_tile_size:
            tile_sampled = tile_sampled.resize(initial_tile_size, Image.Resampling.LANCZOS)

        # Put the tile into position
        image_tile_only = Image.new('RGBA', init_image.size)
        image_tile_only.paste(tile_sampled, crop_region[:2])

        # Add the mask as an alpha channel
        temp = image_tile_only.copy()
        temp.putalpha(image_mask)
        image_tile_only.paste(temp, image_tile_only)

        # Add back the tile to the initial image
        result = init_image.convert('RGBA')
        result.alpha_composite(image_tile_only)

        # Convert back to RGB
        result = result.convert('RGB')

        shared.batch[i] = result

    processed = Processed(p, [shared.batch[0]], p.seed, None)
    return processed


# Node mappings
NODE_CLASS_MAPPINGS = {
    "ArchAi3D_Smart_USDU_Split_Latent": ArchAi3D_Smart_USDU_Split_Latent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_Smart_USDU_Split_Latent": "🚀 Smart USDU Split-Latent",
}
