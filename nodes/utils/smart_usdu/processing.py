from PIL import Image, ImageFilter
import torch
import torch.nn.functional as F
import numpy as np
import math
from nodes import common_ksampler, VAEEncode, VAEDecode, VAEDecodeTiled
from comfy_extras.nodes_custom_sampler import SamplerCustom
from comfy_extras.nodes_differential_diffusion import DifferentialDiffusion
from .utils import pil_to_tensor, tensor_to_pil, get_crop_region, expand_crop, crop_cond
from . import shared
from tqdm import tqdm
import comfy
import comfy.utils
import comfy.model_management
from enum import Enum
import json
import os


# ============================================================
# ControlNet Patches - Copied from ComfyUI nodes_model_patch.py
# For per-tile ControlNet application (avoids circular imports)
# Supports both DiffSynth and Z-Image ControlNet types
# ============================================================
import comfy.latent_formats

class DiffSynthCnetPatch:
    """
    ControlNet patch for DiffSynth/Qwen models (QwenImageBlockWiseControlNet).
    Encodes control image and applies control at each diffusion block.
    """
    def __init__(self, model_patch, vae, image, strength, mask=None):
        self.model_patch = model_patch
        self.vae = vae
        self.image = image
        self.strength = strength
        self.mask = mask
        self.encoded_image = model_patch.model.process_input_latent_image(self.encode_latent_cond(image))
        self.encoded_image_size = (image.shape[1], image.shape[2])

    def encode_latent_cond(self, image):
        latent_image = self.vae.encode(image)
        if self.model_patch.model.additional_in_dim > 0:
            if self.mask is None:
                mask_ = torch.ones_like(latent_image)[:, :self.model_patch.model.additional_in_dim // 4]
            else:
                mask_ = comfy.utils.common_upscale(self.mask.mean(dim=1, keepdim=True), latent_image.shape[-1], latent_image.shape[-2], "bilinear", "none")
            return torch.cat([latent_image, mask_], dim=1)
        else:
            return latent_image

    def __call__(self, kwargs):
        x = kwargs.get("x")
        img = kwargs.get("img")
        block_index = kwargs.get("block_index")
        spacial_compression = self.vae.spacial_compression_encode()
        if self.encoded_image is None or self.encoded_image_size != (x.shape[-2] * spacial_compression, x.shape[-1] * spacial_compression):
            image_scaled = comfy.utils.common_upscale(self.image.movedim(-1, 1), x.shape[-1] * spacial_compression, x.shape[-2] * spacial_compression, "area", "center")
            loaded_models = comfy.model_management.loaded_models(only_currently_used=True)
            self.encoded_image = self.model_patch.model.process_input_latent_image(self.encode_latent_cond(image_scaled.movedim(1, -1)))
            self.encoded_image_size = (image_scaled.shape[-2], image_scaled.shape[-1])
            comfy.model_management.load_models_gpu(loaded_models)

        img[:, :self.encoded_image.shape[1]] += (self.model_patch.model.control_block(img[:, :self.encoded_image.shape[1]], self.encoded_image.to(img.dtype), block_index) * self.strength)
        kwargs['img'] = img
        return kwargs

    def to(self, device_or_dtype):
        if isinstance(device_or_dtype, torch.device):
            self.encoded_image = self.encoded_image.to(device_or_dtype)
        return self

    def models(self):
        return [self.model_patch]


class ZImageControlPatch:
    """
    ControlNet patch for Z-Image ControlNet (ZImage_Control).
    Different encoding and control application than DiffSynth.
    """
    def __init__(self, model_patch, vae, image, strength):
        self.model_patch = model_patch
        self.vae = vae
        self.image = image
        self.strength = strength
        self.encoded_image = self.encode_latent_cond(image)
        self.encoded_image_size = (image.shape[1], image.shape[2])
        self.temp_data = None

    def encode_latent_cond(self, image):
        latent_image = comfy.latent_formats.Flux().process_in(self.vae.encode(image))
        return latent_image

    def __call__(self, kwargs):
        x = kwargs.get("x")
        img = kwargs.get("img")
        txt = kwargs.get("txt")
        pe = kwargs.get("pe")
        vec = kwargs.get("vec")
        block_index = kwargs.get("block_index")
        spacial_compression = self.vae.spacial_compression_encode()
        if self.encoded_image is None or self.encoded_image_size != (x.shape[-2] * spacial_compression, x.shape[-1] * spacial_compression):
            image_scaled = comfy.utils.common_upscale(self.image.movedim(-1, 1), x.shape[-1] * spacial_compression, x.shape[-2] * spacial_compression, "area", "center")
            loaded_models = comfy.model_management.loaded_models(only_currently_used=True)
            self.encoded_image = self.encode_latent_cond(image_scaled.movedim(1, -1))
            self.encoded_image_size = (image_scaled.shape[-2], image_scaled.shape[-1])
            comfy.model_management.load_models_gpu(loaded_models)

        cnet_index = (block_index // 5)
        cnet_index_float = (block_index / 5)

        kwargs.pop("img", None)  # we do ops in place
        kwargs.pop("txt", None)

        cnet_blocks = self.model_patch.model.n_control_layers
        if cnet_index_float > (cnet_blocks - 1):
            self.temp_data = None
            return kwargs

        if self.temp_data is None or self.temp_data[0] > cnet_index:
            self.temp_data = (-1, (None, self.model_patch.model(txt, self.encoded_image.to(img.dtype), pe, vec)))

        while self.temp_data[0] < cnet_index and (self.temp_data[0] + 1) < cnet_blocks:
            next_layer = self.temp_data[0] + 1
            self.temp_data = (next_layer, self.model_patch.model.forward_control_block(next_layer, self.temp_data[1][1], img[:, :self.temp_data[1][1].shape[1]], None, pe, vec))

        if cnet_index_float == self.temp_data[0]:
            img[:, :self.temp_data[1][0].shape[1]] += (self.temp_data[1][0] * self.strength)
            if cnet_blocks == self.temp_data[0] + 1:
                self.temp_data = None

        return kwargs

    def to(self, device_or_dtype):
        if isinstance(device_or_dtype, torch.device):
            self.encoded_image = self.encoded_image.to(device_or_dtype)
            self.temp_data = None
        return self

    def models(self):
        return [self.model_patch]


def is_zimage_control(model_patch):
    """Check if model_patch is a Z-Image ControlNet."""
    try:
        import comfy.ldm.lumina.controlnet
        return isinstance(model_patch.model, comfy.ldm.lumina.controlnet.ZImage_Control)
    except (ImportError, AttributeError):
        # Fallback: check for n_control_layers attribute (Z-Image specific)
        return hasattr(model_patch.model, 'n_control_layers')
# ============================================================
# END OF ControlNet Patches
# ============================================================

if (not hasattr(Image, 'Resampling')):  # For older versions of Pillow
    Image.Resampling = Image

# Taken from the USDU script
class USDUMode(Enum):
    LINEAR = 0
    CHESS = 1
    NONE = 2

class USDUSFMode(Enum):
    NONE = 0
    BAND_PASS = 1
    HALF_TILE = 2
    HALF_TILE_PLUS_INTERSECTIONS = 3

class StableDiffusionProcessing:
    """
    Modified StableDiffusionProcessing with per-tile conditioning support.

    Instead of a single `positive` conditioning, this class accepts a list
    of conditionings (one per tile) and returns the appropriate one based
    on the current tile index.
    """

    def __init__(
        self,
        init_img,
        model,
        positive_list,  # Changed: now accepts CONDITIONING_LIST
        negative,
        vae,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        upscale_by,
        uniform_tile_mode,
        tiled_decode,
        tile_width,
        tile_height,
        redraw_mode,
        seam_fix_mode,
        custom_sampler=None,
        custom_sigmas=None,
        denoise_list=None,  # Optional per-tile denoise values
        denoise_mask_pil=None,  # PIL Image mask for pixel-level denoise (Two-Latent Blend)
        denoise_high=None,  # High denoise for white mask regions (Two-Latent Blend)
        denoise_low=None,   # Low denoise for black mask regions (Two-Latent Blend)
        denoise_mask_tensor=None,  # Tensor mask for Differential Diffusion per-tile
        # ControlNet per-tile parameters
        model_patch=None,  # MODEL_PATCH from ModelPatchLoader
        control_image_tensor=None,  # Upscaled control image [B, H, W, C]
        control_strength=1.0,  # ControlNet strength
        control_mask_tensor=None,  # Upscaled control mask [H, W] (separate from denoise_mask)
    ):
        # Variables used by the USDU script
        self.init_images = [init_img]
        self.image_mask = None
        self.mask_blur = 0
        self.inpaint_full_res_padding = 0
        self.width = init_img.width * upscale_by
        self.height = init_img.height * upscale_by
        self.rows = round(self.height / tile_height)
        self.cols = round(self.width / tile_width)

        # Per-tile conditioning support
        self._positive_list = positive_list  # Store the list of conditionings
        self._current_tile_index = 0         # Track current tile
        self._negative = negative            # Store negative conditioning

        # ComfyUI Sampler inputs
        self.model = model
        self.vae = vae
        self.seed = seed
        self.steps = steps
        self.cfg = cfg
        self.sampler_name = sampler_name
        self.scheduler = scheduler

        # Per-tile denoise support (similar to per-tile conditioning)
        self._denoise_global = denoise
        self._denoise_list = denoise_list

        # Pixel-level denoise control (Two-Latent Blend method)
        self._denoise_mask_pil = denoise_mask_pil  # PIL Image
        self._denoise_high = denoise_high  # Float for white regions
        self._denoise_low = denoise_low    # Float for black regions

        # Differential Diffusion per-tile mask (tensor, upscaled to output size)
        self._denoise_mask_tensor = denoise_mask_tensor  # Torch tensor [H, W]

        # ControlNet per-tile parameters
        self._model_patch = model_patch  # MODEL_PATCH from ModelPatchLoader
        self._control_image_tensor = control_image_tensor  # Upscaled control image [B, H, W, C]
        self._control_strength = control_strength  # ControlNet strength
        self._control_mask_tensor = control_mask_tensor  # Upscaled control mask [H, W] (separate from denoise_mask)
        self._original_model = model  # Store unpatched model for per-tile patching

        # Optional custom sampler and sigmas
        self.custom_sampler = custom_sampler
        self.custom_sigmas = custom_sigmas

        if (custom_sampler is not None) ^ (custom_sigmas is not None):
            print("[Smart USDU] Both custom sampler and custom sigmas must be provided, defaulting to widget sampler and sigmas")

        # Variables used only by this script
        self.init_size = init_img.width, init_img.height
        self.upscale_by = upscale_by
        self.uniform_tile_mode = uniform_tile_mode
        self.tiled_decode = tiled_decode
        self.vae_decoder = VAEDecode()
        self.vae_encoder = VAEEncode()
        self.vae_decoder_tiled = VAEDecodeTiled()

        if self.tiled_decode:
            print("[Smart USDU] Using tiled decode")

        # Other required A1111 variables for the USDU script that is currently unused in this script
        self.extra_generation_params = {}

        # Load config file for USDU
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)

        # Progress bar for the entire process instead of per tile
        self.progress_bar_enabled = False
        if comfy.utils.PROGRESS_BAR_ENABLED:
            self.progress_bar_enabled = True
            comfy.utils.PROGRESS_BAR_ENABLED = config.get('per_tile_progress', True)
            self.tiles = 0
            if redraw_mode.value != USDUMode.NONE.value:
                self.tiles += self.rows * self.cols
            if seam_fix_mode.value == USDUSFMode.BAND_PASS.value:
                self.tiles += (self.rows - 1) + (self.cols - 1)
            elif seam_fix_mode.value == USDUSFMode.HALF_TILE.value:
                self.tiles += (self.rows - 1) * self.cols + (self.cols - 1) * self.rows
            elif seam_fix_mode.value == USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS.value:
                self.tiles += (self.rows - 1) * self.cols + (self.cols - 1) * self.rows + (self.rows - 1) * (self.cols - 1)
            self.pbar = None

        # Log conditioning info
        print(f"[Smart USDU] Tile conditionings: {len(self._positive_list)}, Grid: {self.cols}x{self.rows} = {self.rows * self.cols} tiles")

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

    @property
    def denoise(self):
        """Return denoise for current tile (or global if no per-tile list)."""
        if self._denoise_list and self._current_tile_index < len(self._denoise_list):
            return self._denoise_list[self._current_tile_index]
        return self._denoise_global

    @denoise.setter
    def denoise(self, value):
        """Allow setting global denoise (for compatibility)."""
        self._denoise_global = value

    def advance_tile(self):
        """Move to next tile's conditioning."""
        self._current_tile_index += 1

    def reset_tile_index(self):
        """Reset tile index (for seams fix pass)."""
        self._current_tile_index = 0

    def __del__(self):
        # Undo changes to progress bar flag when node is done or cancelled
        if self.progress_bar_enabled:
            comfy.utils.PROGRESS_BAR_ENABLED = True

class Processed:

    def __init__(self, p: StableDiffusionProcessing, images: list, seed: int, info: str):
        self.images = images
        self.seed = seed
        self.info = info

    def infotext(self, p: StableDiffusionProcessing, index):
        return None


def fix_seed(p: StableDiffusionProcessing):
    pass


def sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise, custom_sampler, custom_sigmas):
    # Choose way to sample based on given inputs

    # Custom sampler and sigmas
    if custom_sampler is not None and custom_sigmas is not None:
        kwargs = dict(
            model=model,
            add_noise=True,
            noise_seed=seed,
            cfg=cfg,
            positive=positive,
            negative=negative,
            sampler=custom_sampler,
            sigmas=custom_sigmas,
            latent_image=latent
        )
        if "execute" in dir(SamplerCustom):
            (samples, _) = SamplerCustom.execute(**kwargs)
        else:
            custom_sample = SamplerCustom()
            (samples, _) = getattr(custom_sample, custom_sample.FUNCTION)(**kwargs)
        return samples

    # Default
    (samples,) = common_ksampler(model, seed, steps, cfg, sampler_name,
                                 scheduler, positive, negative, latent, denoise=denoise)
    return samples


def sample_no_add_noise(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise, custom_sampler, custom_sigmas):
    """Sample WITHOUT adding noise - for use with manually pre-noised latents."""

    # Custom sampler and sigmas
    if custom_sampler is not None and custom_sigmas is not None:
        kwargs = dict(
            model=model,
            add_noise=False,  # IMPORTANT: No noise addition - we already added it manually
            noise_seed=seed,
            cfg=cfg,
            positive=positive,
            negative=negative,
            sampler=custom_sampler,
            sigmas=custom_sigmas,
            latent_image=latent
        )
        if "execute" in dir(SamplerCustom):
            (samples, _) = SamplerCustom.execute(**kwargs)
        else:
            custom_sample = SamplerCustom()
            (samples, _) = getattr(custom_sample, custom_sample.FUNCTION)(**kwargs)
        return samples

    # Default - use common_ksampler with disable_noise=True
    (samples,) = common_ksampler(model, seed, steps, cfg, sampler_name,
                                 scheduler, positive, negative, latent,
                                 denoise=denoise, disable_noise=True)
    return samples


def process_images(p: StableDiffusionProcessing) -> Processed:
    # Where the main image generation happens in A1111

    # Show the progress bar
    if p.progress_bar_enabled and p.pbar is None:
        p.pbar = tqdm(total=p.tiles, desc='Smart USDU', unit='tile')

    # Setup
    image_mask = p.image_mask.convert('L')
    init_image = p.init_images[0]

    # Locate the white region of the mask outlining the tile and add padding
    crop_region = get_crop_region(image_mask, p.inpaint_full_res_padding)

    if p.uniform_tile_mode:
        # Expand the crop region to match the processing size ratio and then resize it to the processing size
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
        # Ensure tile_size is 64-aligned for Flux/Qwen models
        aligned_w = math.ceil(p.width / 64) * 64
        aligned_h = math.ceil(p.height / 64) * 64
        tile_size = aligned_w, aligned_h
    else:
        # Uses the minimal size that can fit the mask, minimizes tile size but may lead to image sizes that the model is not trained on
        x1, y1, x2, y2 = crop_region
        crop_width = x2 - x1
        crop_height = y2 - y1
        # Enforce 64-pixel alignment for Flux/Qwen models (was /8 * 8)
        target_width = math.ceil(crop_width / 64) * 64
        target_height = math.ceil(crop_height / 64) * 64
        crop_region, tile_size = expand_crop(crop_region, image_mask.width,
                                             image_mask.height, target_width, target_height)

    # Blur the mask
    if p.mask_blur > 0:
        image_mask = image_mask.filter(ImageFilter.GaussianBlur(p.mask_blur))

    # Crop the images to get the tiles that will be used for generation
    tiles = [img.crop(crop_region) for img in shared.batch]

    # Assume the same size for all images in the batch
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

    # ===== DIFFERENTIAL DIFFUSION PER-TILE MASK =====
    # If denoise_mask_tensor is provided, crop it to tile and set as noise_mask
    if p._denoise_mask_tensor is not None:
        # Crop mask to tile region (same as image)
        x1, y1, x2, y2 = crop_region
        mask_cropped = p._denoise_mask_tensor[y1:y2, x1:x2]

        # Resize to tile_size if needed (match the tile processing)
        if mask_cropped.shape[0] != tile_size[1] or mask_cropped.shape[1] != tile_size[0]:
            mask_cropped = F.interpolate(
                mask_cropped.unsqueeze(0).unsqueeze(0),
                size=(tile_size[1], tile_size[0]),  # (H, W)
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)

        # Resize to latent dimensions
        latent_h, latent_w = latent["samples"].shape[-2], latent["samples"].shape[-1]
        mask_latent = F.interpolate(
            mask_cropped.unsqueeze(0).unsqueeze(0),
            size=(latent_h, latent_w),
            mode='bilinear',
            align_corners=False
        )  # Shape: [1, 1, H, W]

        # Set noise_mask for Differential Diffusion
        # The model's denoise_mask_function will use this
        latent["noise_mask"] = mask_latent

        print(f"  [DiffDiff] Tile mask: crop_region={crop_region}, mask_shape={mask_latent.shape}")

    # ===== CONTROLNET PER-TILE =====
    # If ControlNet params are provided, crop control image and apply patch
    model_for_sampling = p.model
    if p._model_patch is not None and p._control_image_tensor is not None:
        # Crop control image to tile region (same crop as main image)
        x1, y1, x2, y2 = crop_region
        control_cropped = p._control_image_tensor[:, y1:y2, x1:x2, :]

        # Resize to tile_size if needed (match the tile processing)
        if control_cropped.shape[1] != tile_size[1] or control_cropped.shape[2] != tile_size[0]:
            control_cropped = F.interpolate(
                control_cropped.movedim(-1, 1),  # BHWC -> BCHW
                size=(tile_size[1], tile_size[0]),
                mode='bilinear',
                align_corners=False
            ).movedim(1, -1)  # BCHW -> BHWC

        # Crop control mask if provided (separate from denoise_mask)
        control_mask_cropped = None
        if p._control_mask_tensor is not None:
            control_mask_cropped = p._control_mask_tensor[y1:y2, x1:x2]

            # Resize to tile_size if needed
            if control_mask_cropped.shape[0] != tile_size[1] or control_mask_cropped.shape[1] != tile_size[0]:
                control_mask_cropped = F.interpolate(
                    control_mask_cropped.unsqueeze(0).unsqueeze(0),
                    size=(tile_size[1], tile_size[0]),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)

            # DiffSynthCnetPatch expects mask as [B, C, H, W] format (4D)
            # encode_latent_cond uses mask.mean(dim=1, keepdim=True) which needs 4D input
            control_mask_cropped = control_mask_cropped.unsqueeze(0).unsqueeze(0)  # [H, W] -> [1, 1, H, W]

        # Clone the model so the patch only affects THIS tile
        model_for_sampling = p._original_model.clone()

        # Create and apply ControlNet patch with cropped control image
        # Use the correct patch class based on model type
        if is_zimage_control(p._model_patch):
            # Z-Image ControlNet (doesn't support mask)
            cnet_patch = ZImageControlPatch(p._model_patch, p.vae, control_cropped, p._control_strength)
            patch_type = "ZImage"
        else:
            # DiffSynth/Qwen ControlNet (supports mask)
            cnet_patch = DiffSynthCnetPatch(p._model_patch, p.vae, control_cropped, p._control_strength, control_mask_cropped)
            patch_type = "DiffSynth"
        model_for_sampling.set_model_double_block_patch(cnet_patch)

        mask_info = f", mask_shape={control_mask_cropped.shape}" if control_mask_cropped is not None and patch_type == "DiffSynth" else ""
        print(f"  [ControlNet:{patch_type}] Tile control: crop_region={crop_region}, control_shape={control_cropped.shape}{mask_info}")

    # ===== TWO-LATENT BLEND - PROPER SIGMA-BASED NOISE INJECTION =====
    # Creates two noised latents (low/high) with SAME seed, blends via mask, samples with add_noise=False
    if p._denoise_mask_pil is not None and p._denoise_high is not None and p._denoise_low is not None:
        from comfy.samplers import calculate_sigmas

        # Step 1: Crop mask to tile region (same as image crop)
        mask_cropped = p._denoise_mask_pil.crop(crop_region)

        # Resize to tile_size if needed
        if mask_cropped.size != tile_size:
            mask_cropped = mask_cropped.resize(tile_size, Image.Resampling.BILINEAR)

        # Convert mask to tensor and resize to latent size
        mask_np = np.array(mask_cropped).astype(np.float32) / 255.0
        mask_tensor = torch.tensor(mask_np, dtype=torch.float32)

        latent_h, latent_w = latent["samples"].shape[-2], latent["samples"].shape[-1]
        mask_latent = F.interpolate(
            mask_tensor.unsqueeze(0).unsqueeze(0),
            size=(latent_h, latent_w),
            mode='bilinear',
            align_corners=False
        )  # Shape: [1, 1, H, W]

        device = latent["samples"].device
        dtype = latent["samples"].dtype
        mask_latent = mask_latent.to(device=device, dtype=dtype)

        # Expand mask to match latent channels [1, 1, H, W] -> [B, C, H, W]
        mask_4d = mask_latent.expand_as(latent["samples"])

        # Step 2: Get the clean latent
        clean_latent = latent["samples"].clone()

        # Step 3: Generate noise with FIXED SEED (critical: same noise for both latents!)
        generator = torch.Generator(device=device).manual_seed(p.seed)
        noise = torch.randn(clean_latent.shape, generator=generator, device=device, dtype=dtype)

        # Step 4: Get the model's sampling object for proper noise scaling
        model_sampling = p.model.get_model_object("model_sampling")

        denoise_low = p._denoise_low
        denoise_high = p._denoise_high
        max_denoise = max(denoise_low, denoise_high)

        # Calculate sigmas for each denoise level
        # For denoise < 1.0: new_steps = steps/denoise, take last (steps+1) sigmas
        def get_start_sigma(denoise_val):
            if denoise_val <= 0.0:
                return 0.0
            if denoise_val >= 0.9999:
                new_steps = p.steps
            else:
                new_steps = int(p.steps / denoise_val)
            sigmas = calculate_sigmas(model_sampling, p.scheduler, new_steps)
            # Take the starting sigma (corresponds to the noise level)
            start_sigma = sigmas[-(p.steps + 1)].item() if denoise_val < 0.9999 else sigmas[0].item()
            return start_sigma

        sigma_low = get_start_sigma(denoise_low)
        sigma_high = get_start_sigma(denoise_high)
        sigma_max = get_start_sigma(max_denoise)

        # Step 5: Apply proper noise scaling using the model's method
        # For EPS: noisy = latent + noise * sigma
        # For CONST/Flow: noisy = sigma * noise + (1 - sigma) * latent
        # Use the model's noise_scaling function for correctness
        latent_a = model_sampling.noise_scaling(
            torch.tensor([sigma_low], device=device, dtype=dtype),
            noise.clone(), clean_latent.clone(), max_denoise=False
        )
        latent_b = model_sampling.noise_scaling(
            torch.tensor([sigma_high], device=device, dtype=dtype),
            noise.clone(), clean_latent.clone(), max_denoise=False
        )

        # Step 6: Blend latents via mask
        # mask=0 (black) → use latent_a (low noise)
        # mask=1 (white) → use latent_b (high noise)
        blended_latent = latent_a * (1.0 - mask_4d) + latent_b * mask_4d

        # Debug output
        print(f"  [TwoLatentBlend] TILE DEBUG:")
        print(f"    - Input mask range: min={mask_np.min():.3f}, max={mask_np.max():.3f}")
        print(f"    - denoise_low={denoise_low}, denoise_high={denoise_high}")
        print(f"    - sigma_low={sigma_low:.4f}, sigma_high={sigma_high:.4f}, sigma_max={sigma_max:.4f}")
        print(f"    - clean_latent std={clean_latent.std():.4f}")
        print(f"    - latent_a (low) std={latent_a.std():.4f}")
        print(f"    - latent_b (high) std={latent_b.std():.4f}")
        print(f"    - blended_latent std={blended_latent.std():.4f}")

        # Replace latent with blended version
        latent["samples"] = blended_latent

        # Step 7: Sample with add_noise=DISABLE (we already added noise!)
        print(f"    - Sampling with add_noise=False, denoise={max_denoise}")

        samples = sample_no_add_noise(model_for_sampling, p.seed, p.steps, p.cfg, p.sampler_name, p.scheduler,
                                       positive_cropped, negative_cropped, latent,
                                       max_denoise, p.custom_sampler, p.custom_sigmas)
    else:
        # Single pass (original behavior)
        samples = sample(model_for_sampling, p.seed, p.steps, p.cfg, p.sampler_name, p.scheduler,
                         positive_cropped, negative_cropped, latent, p.denoise,
                         p.custom_sampler, p.custom_sigmas)

    # Update the progress bar
    if p.progress_bar_enabled:
        p.pbar.update(1)

    # Decode the sample
    if not p.tiled_decode:
        (decoded,) = p.vae_decoder.decode(p.vae, samples)
    else:
        (decoded,) = p.vae_decoder_tiled.decode(p.vae, samples, 512)  # Default tile size is 512

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
        # Must make a copy due to the possibility of an edge becoming black
        temp = image_tile_only.copy()
        temp.putalpha(image_mask)
        image_tile_only.paste(temp, image_tile_only)

        # Add back the tile to the initial image according to the mask in the alpha channel
        result = init_image.convert('RGBA')
        result.alpha_composite(image_tile_only)

        # Convert back to RGB
        result = result.convert('RGB')

        shared.batch[i] = result

    processed = Processed(p, [shared.batch[0]], p.seed, None)
    return processed
