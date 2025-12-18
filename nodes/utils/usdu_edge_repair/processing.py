"""
USDU Edge Repair - Processing Module (Core Sampling & ControlNet)
==================================================================

This is the core processing module that handles tile-based image generation.
It includes StableDiffusionProcessing (A1111 compatibility layer), ControlNet
patches for per-tile control, and the main process_images function.

TREE MAP:
---------
processing.py
│
├── CONTROLNET PATCHES (imported from controlnet.py)
│   ├── DiffSynthCnetPatch               - ControlNet patch for DiffSynth/Qwen models
│   ├── ZImageControlPatch               - ControlNet patch for Z-Image models
│   └── is_zimage_control()              - Check if model is Z-Image ControlNet
│
├── ENUMS (shared with ultimate_upscale.py)
│   ├── USDUMode                          - Tile processing modes
│   │   ├── LINEAR = 0                   - Row-by-row processing
│   │   ├── CHESS = 1                    - Checkerboard processing
│   │   └── NONE = 2                     - Skip redraw
│   │
│   └── USDUSFMode                        - Seam fix modes
│       ├── NONE = 0                     - No seam fix
│       ├── BAND_PASS = 1                - Band pass method
│       ├── HALF_TILE = 2                - Half tile overlap
│       └── HALF_TILE_PLUS_INTERSECTIONS - Half tile + corners
│
├── StableDiffusionProcessing (class)    - Main processing container
│   ├── __init__(...)                    - Initialize with all parameters
│   ├── positive (property)              - Get conditioning for current tile
│   ├── negative (property)              - Get negative conditioning
│   ├── denoise (property)               - Get denoise for current tile
│   ├── advance_tile()                   - Move to next tile index
│   └── reset_tile_index()               - Reset tile index (for seam pass)
│
├── Processed (class)                    - Result container
│   ├── __init__(p, images, seed, info)
│   └── infotext(p, index)               - Get info text (returns None)
│
├── fix_seed(p)                          - A1111 compatibility stub
│
├── SAMPLING FUNCTIONS
│   ├── sample(...)                      - Sample with noise addition
│   └── sample_no_add_noise(...)         - Sample without noise (for pre-noised)
│
└── process_images(p)                    - MAIN FUNCTION: Process one tile
    ├── 1. Locate tile region from mask
    ├── 2. Crop conditioning to tile
    ├── 3. Encode tile to latent
    ├── 4. Apply Differential Diffusion mask (optional)
    ├── 5. Apply ControlNet patch (optional)
    ├── 6. Apply Two-Latent Blend (optional)
    ├── 7. Sample with diffusion model
    ├── 8. Decode latent to image
    └── 9. Composite tile back into image

DATA FLOW:
----------
1. USDU calls process_images(p) for each tile
2. process_images:
   a. Reads p.image_mask to find tile region
   b. Crops images from shared.batch to tile size
   c. Crops conditioning with crop_cond()
   d. Encodes tile to latent
   e. (Optional) Applies Differential Diffusion mask
   f. (Optional) Applies ControlNet patch to model
   g. (Optional) Two-Latent Blend for variable denoise
   h. Calls sample() or sample_no_add_noise()
   i. Decodes latent back to image
   j. Composites tile back into shared.batch

PER-TILE FEATURES:
------------------
1. Per-tile Conditioning:
   - StableDiffusionProcessing accepts positive_list (one per tile)
   - advance_tile() increments _current_tile_index
   - positive property returns conditioning for current tile

2. Per-tile Denoise:
   - denoise_list parameter for different denoise per tile
   - denoise property returns value for current tile

3. Differential Diffusion:
   - denoise_mask_tensor: Grayscale mask for per-pixel denoise
   - Cropped to tile, set as latent["noise_mask"]

4. Two-Latent Blend:
   - denoise_mask_pil + denoise_high + denoise_low
   - Creates two noised latents, blends via mask
   - Samples with add_noise=False

5. Per-tile ControlNet:
   - model_patch + control_image_tensor + control_strength
   - Crops control image to tile
   - Applies DiffSynthCnetPatch or ZImageControlPatch
"""

from PIL import Image, ImageFilter, ImageDraw
import torch
import torch.nn.functional as F
import numpy as np
import math
from nodes import common_ksampler, VAEEncode, VAEDecode, VAEDecodeTiled
from comfy_extras.nodes_custom_sampler import SamplerCustom
# UNUSED: from comfy_extras.nodes_differential_diffusion import DifferentialDiffusion
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
# CONTROLNET PATCHES - Imported from controlnet.py
# ============================================================
from .controlnet import DiffSynthCnetPatch, ZImageControlPatch, is_zimage_control


# ============================================================
# ENUMS - Shared with ultimate_upscale.py
# ============================================================

# Pillow compatibility for older versions
if (not hasattr(Image, 'Resampling')):
    Image.Resampling = Image


class USDUMode(Enum):
    """
    Tile redraw processing modes.

    LINEAR: Process tiles row by row (left-right, top-bottom)
    CHESS: Process tiles in checkerboard pattern (reduces seams)
    NONE: Skip the redraw pass entirely
    """
    LINEAR = 0
    CHESS = 1
    NONE = 2


class USDUSFMode(Enum):
    """
    Seam fix processing modes.

    NONE: No seam fixing
    BAND_PASS: Process narrow bands along seam lines
    HALF_TILE: Overlap tiles at seam positions
    HALF_TILE_PLUS_INTERSECTIONS: Half tile + corner intersections
    """
    NONE = 0
    BAND_PASS = 1
    HALF_TILE = 2
    HALF_TILE_PLUS_INTERSECTIONS = 3


# ============================================================
# STABLE DIFFUSION PROCESSING CLASS
# ============================================================

class StableDiffusionProcessing:
    """
    A1111-compatible processing class with per-tile conditioning support.

    This class mimics A1111's StableDiffusionProcessing but adds support for:
    - Per-tile conditioning (different prompts per tile)
    - Per-tile denoise values
    - Differential Diffusion (per-pixel denoise via mask)
    - Two-Latent Blend (variable denoise via mask)
    - Per-tile ControlNet

    Key Modification:
        Instead of a single `positive` conditioning, this accepts a LIST
        of conditionings (one per tile). The `positive` property returns
        the conditioning for the current tile based on _current_tile_index.

    Attributes:
        init_images: List of init images (from A1111)
        image_mask: PIL mask for current tile
        mask_blur: Blur amount for mask
        width, height: Target canvas size
        model: ComfyUI model for sampling
        vae: VAE for encode/decode
        seed, steps, cfg: Sampling parameters
        sampler_name, scheduler: Sampler configuration
        _positive_list: List of conditionings (one per tile)
        _current_tile_index: Current tile being processed
    """

    def __init__(
        self,
        init_img,
        model,
        positive_list,  # List of conditionings, one per tile
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
        geometry=None,  # TileGeometry instance for unified calculations
        # Edge mask for DiffDiff (functional, not just preview)
        use_edge_mask_diffdiff=False,  # Whether to apply edge mask to DiffDiff
        edge_mask_width=20,  # Width of edge border
        edge_mask_feather=8,  # Feather amount for edge mask
    ):
        """
        Initialize StableDiffusionProcessing with all parameters.

        Args:
            init_img: Initial PIL Image
            model: ComfyUI MODEL
            positive_list: List of CONDITIONING (one per tile)
            negative: CONDITIONING for negative prompt
            vae: VAE model
            seed: Random seed
            steps: Number of sampling steps
            cfg: CFG scale
            sampler_name: Name of sampler
            scheduler: Scheduler name
            denoise: Global denoise strength
            upscale_by: Upscale factor
            uniform_tile_mode: Use uniform tile sizes
            tiled_decode: Use tiled VAE decoding
            tile_width, tile_height: Tile dimensions
            redraw_mode: USDUMode enum
            seam_fix_mode: USDUSFMode enum
            custom_sampler: Optional custom sampler
            custom_sigmas: Optional custom sigmas
            denoise_list: Optional per-tile denoise values
            denoise_mask_pil: PIL mask for Two-Latent Blend
            denoise_high: High denoise for white mask regions
            denoise_low: Low denoise for black mask regions
            denoise_mask_tensor: Tensor mask for Differential Diffusion
            model_patch: MODEL_PATCH for ControlNet
            control_image_tensor: Control image for ControlNet
            control_strength: ControlNet strength
            control_mask_tensor: Control mask (separate from denoise mask)
        """
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

        # Per-tile denoise support
        self._denoise_global = denoise
        self._denoise_list = denoise_list

        # Pixel-level denoise control (Two-Latent Blend method)
        self._denoise_mask_pil = denoise_mask_pil  # PIL Image
        self._denoise_high = denoise_high  # Float for white regions
        self._denoise_low = denoise_low    # Float for black regions

        # Differential Diffusion per-tile mask
        self._denoise_mask_tensor = denoise_mask_tensor  # Torch tensor [H, W]

        # ControlNet per-tile parameters
        self._model_patch = model_patch
        self._control_image_tensor = control_image_tensor
        self._control_strength = control_strength
        self._control_mask_tensor = control_mask_tensor
        self._original_model = model  # Store unpatched model for per-tile patching

        # TileGeometry for unified geometry calculations
        self._geometry = geometry

        # Weighted accumulation for proper tile blending (fixes boundary artifacts)
        self._original_images = None  # Store originals before any tiles
        self._tile_sum = None         # Accumulated: tile * mask
        self._weight_sum = None       # Accumulated: mask
        self._processed_tiles = []    # Store processed tiles for debug output

        # ACTUAL data capture for preview (not simulated)
        self._actual_masks = []           # Actual blurred masks used in compositing
        self._actual_tiles_input = []     # Actual tile crops before sampling

        # Edge mask for DiffDiff (functional feature)
        self._use_edge_mask_diffdiff = use_edge_mask_diffdiff
        self._edge_mask_width = edge_mask_width
        self._edge_mask_feather = edge_mask_feather

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

        # Other required A1111 variables (unused in this script)
        self.extra_generation_params = {}

        # Load config file for USDU
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)

        # Progress bar for the entire process
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
        """
        Return conditioning for the current tile.

        Uses _current_tile_index to select from _positive_list.
        Falls back to last conditioning if index exceeds list length.
        """
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
        """
        Return denoise for current tile.

        If denoise_list is provided, returns value for current tile.
        Otherwise returns global denoise value.
        """
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

    def reset_tile_index(self, reset_accumulators=True):
        """Reset tile index. Optionally keep accumulators for chess mode."""
        self._current_tile_index = 0
        if reset_accumulators:
            # Reset accumulators for seam fix pass (but NOT between chess passes)
            self._original_images = None
            self._tile_sum = None
            self._weight_sum = None
            self._processed_tiles = []

    def __del__(self):
        """Cleanup: restore progress bar flag when done."""
        if self.progress_bar_enabled:
            comfy.utils.PROGRESS_BAR_ENABLED = True


# ============================================================
# RESULT CONTAINER
# ============================================================

class Processed:
    """
    Container for processing results (A1111 compatibility).

    Attributes:
        images: List of result PIL Images
        seed: Random seed used
        info: Info string (unused)
    """

    def __init__(self, p: StableDiffusionProcessing, images: list, seed: int, info: str):
        """
        Initialize result container.

        Args:
            p: The StableDiffusionProcessing instance
            images: List of result images
            seed: Random seed used
            info: Info string
        """
        self.images = images
        self.seed = seed
        self.info = info

    def infotext(self, p: StableDiffusionProcessing, index):
        """Return info text (returns None for ComfyUI)."""
        return None


# ============================================================
# A1111 COMPATIBILITY
# ============================================================

def fix_seed(p: StableDiffusionProcessing):
    """
    A1111 compatibility stub.

    In A1111, this fixes random seed. In ComfyUI, seed is already fixed.
    Required by ultimate_upscale.py Script.run().
    """
    pass


# ============================================================
# SAMPLING FUNCTIONS
# ============================================================

def sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise, custom_sampler, custom_sigmas):
    """
    Run diffusion sampling with noise addition.

    This is the standard sampling path where noise is added by the sampler.

    Args:
        model: ComfyUI MODEL
        seed: Random seed
        steps: Number of steps
        cfg: CFG scale
        sampler_name: Sampler name
        scheduler: Scheduler name
        positive: Positive conditioning
        negative: Negative conditioning
        latent: Latent dict with 'samples' key
        denoise: Denoise strength
        custom_sampler: Optional custom sampler
        custom_sigmas: Optional custom sigmas

    Returns:
        Latent dict with sampled 'samples'
    """
    # Custom sampler path
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

    # Default common_ksampler path
    (samples,) = common_ksampler(model, seed, steps, cfg, sampler_name,
                                 scheduler, positive, negative, latent, denoise=denoise)
    return samples


def sample_no_add_noise(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise, custom_sampler, custom_sigmas):
    """
    Run diffusion sampling WITHOUT adding noise.

    Used with Two-Latent Blend where noise is manually pre-added.

    Args:
        model: ComfyUI MODEL
        seed: Random seed
        steps: Number of steps
        cfg: CFG scale
        sampler_name: Sampler name
        scheduler: Scheduler name
        positive: Positive conditioning
        negative: Negative conditioning
        latent: Latent dict with pre-noised 'samples'
        denoise: Denoise strength
        custom_sampler: Optional custom sampler
        custom_sigmas: Optional custom sigmas

    Returns:
        Latent dict with sampled 'samples'
    """
    # Custom sampler path
    if custom_sampler is not None and custom_sigmas is not None:
        kwargs = dict(
            model=model,
            add_noise=False,  # IMPORTANT: No noise addition
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

    # Default common_ksampler path with disable_noise=True
    (samples,) = common_ksampler(model, seed, steps, cfg, sampler_name,
                                 scheduler, positive, negative, latent,
                                 denoise=denoise, disable_noise=True)
    return samples


# ============================================================
# MAIN PROCESSING FUNCTION
# ============================================================

def process_images(p: StableDiffusionProcessing) -> Processed:
    """
    Process a single tile - the main image generation function.

    This is called by USDURedraw/USDUSeamsFix for each tile. It:
    1. Locates the tile region from the mask
    2. Crops images and conditioning to tile
    3. Applies optional features (DiffDiff, ControlNet, Two-Latent)
    4. Samples with the diffusion model
    5. Decodes and composites back into the image

    Args:
        p: StableDiffusionProcessing instance with all parameters

    Returns:
        Processed: Result containing the processed image

    Flow:
        1. Mask parsing -> get tile crop region
        2. Expand region to match processing size
        3. Blur mask for smooth blending
        4. Crop tiles from shared.batch
        5. Crop conditioning for tile
        6. Encode tiles to latent
        7. (Optional) Apply Differential Diffusion mask
        8. (Optional) Apply ControlNet patch
        9. (Optional) Apply Two-Latent Blend
        10. Sample with model
        11. Decode latent to image
        12. Resize and composite back
    """
    # === PROGRESS BAR ===
    if p.progress_bar_enabled and p.pbar is None:
        p.pbar = tqdm(total=p.tiles, desc='Smart USDU', unit='tile')

    # === SETUP ===
    image_mask = p.image_mask.convert('L')
    init_image = p.init_images[0]

    # Locate the white region of the mask (tile location) and add padding
    crop_region = get_crop_region(image_mask, p.inpaint_full_res_padding)

    # === DETERMINE TILE SIZE ===
    if p.uniform_tile_mode:
        # Expand crop to match processing size ratio, then resize to processing size
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
        # Ensure 64-pixel alignment for Flux/Qwen models
        aligned_w = math.ceil(p.width / 64) * 64
        aligned_h = math.ceil(p.height / 64) * 64
        tile_size = aligned_w, aligned_h
    else:
        # Use minimal size that fits the mask (may vary per tile)
        x1, y1, x2, y2 = crop_region
        crop_width = x2 - x1
        crop_height = y2 - y1
        # Enforce 64-pixel alignment for Flux/Qwen models
        target_width = math.ceil(crop_width / 64) * 64
        target_height = math.ceil(crop_height / 64) * 64
        crop_region, tile_size = expand_crop(crop_region, image_mask.width,
                                             image_mask.height, target_width, target_height)

    # === BLUR MASK FOR SMOOTH BLENDING ===
    if p.mask_blur > 0:
        image_mask = image_mask.filter(ImageFilter.GaussianBlur(p.mask_blur))

        # === RESTORE SHARP EDGES AT IMAGE BOUNDARIES ===
        # Blur should only affect INTERNAL edges (where tiles meet).
        # Boundary edges (at image edges) should stay sharp (full white).
        # This gives consistent border thickness across all tiles.
        if hasattr(p, '_geometry') and p._geometry is not None:
            tile_idx = p._current_tile_index
            xi, yi = p._geometry.get_tile_coords(tile_idx)
            draw = ImageDraw.Draw(image_mask)

            # Get tile boundaries
            x1, y1, x2, y2 = p._geometry.get_tile_rect(tile_idx)
            blur_width = p.mask_blur * 2  # Blur spreads approximately this far

            # Restore sharp white edge on IMAGE BOUNDARIES (where no neighbor exists)
            if xi == 0:  # Left boundary
                draw.rectangle((x1, y1, x1 + blur_width, y2), fill=255)
            if xi == p._geometry.tiles_x - 1:  # Right boundary
                draw.rectangle((x2 - blur_width, y1, x2, y2), fill=255)
            if yi == 0:  # Top boundary
                draw.rectangle((x1, y1, x2, y1 + blur_width), fill=255)
            if yi == p._geometry.tiles_y - 1:  # Bottom boundary
                draw.rectangle((x1, y2 - blur_width, x2, y2), fill=255)

    # === CAPTURE ACTUAL MASK (for preview - shows real blurred mask used) ===
    # Crop to tile region for display
    actual_mask_crop = image_mask.crop(crop_region)
    if actual_mask_crop.size != tile_size:
        actual_mask_crop = actual_mask_crop.resize(tile_size, Image.Resampling.LANCZOS)
    p._actual_masks.append(actual_mask_crop)

    # === CROP TILES FROM BATCH ===
    tiles = [img.crop(crop_region) for img in shared.batch]
    initial_tile_size = tiles[0].size

    # Resize tiles to target size if needed
    for i, tile in enumerate(tiles):
        if tile.size != tile_size:
            tiles[i] = tile.resize(tile_size, Image.Resampling.LANCZOS)

    # === CAPTURE ACTUAL INPUT TILE (for preview - shows real tile before sampling) ===
    p._actual_tiles_input.append(tiles[0].copy())

    # === CROP CONDITIONING TO TILE ===
    positive_cropped = crop_cond(p.positive, crop_region, p.init_size, init_image.size, tile_size)
    negative_cropped = crop_cond(p.negative, crop_region, p.init_size, init_image.size, tile_size)

    # === ENCODE TO LATENT ===
    batched_tiles = torch.cat([pil_to_tensor(tile) for tile in tiles], dim=0)
    (latent,) = p.vae_encoder.encode(p.vae, batched_tiles)

    # ===== DIFFERENTIAL DIFFUSION PER-TILE MASK =====
    # If denoise_mask_tensor is provided, crop and apply as noise_mask
    if p._denoise_mask_tensor is not None:
        # Crop mask to tile region
        x1, y1, x2, y2 = crop_region
        mask_cropped = p._denoise_mask_tensor[y1:y2, x1:x2]

        # Resize to tile_size if needed
        if mask_cropped.shape[0] != tile_size[1] or mask_cropped.shape[1] != tile_size[0]:
            mask_cropped = F.interpolate(
                mask_cropped.unsqueeze(0).unsqueeze(0),
                size=(tile_size[1], tile_size[0]),
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
        latent["noise_mask"] = mask_latent

        print(f"  [DiffDiff] Tile mask: crop_region={crop_region}, mask_shape={mask_latent.shape}")

    # ===== EDGE MASK FOR DIFFDIFF (FUNCTIONAL) =====
    # If edge mask is enabled, combine with existing noise_mask or create new one
    # This applies more denoising at tile edges for smoother blending
    if p._use_edge_mask_diffdiff and p._geometry is not None:
        # Get edge mask for current tile from geometry
        tile_idx = p._current_tile_index
        edge_mask_pil = p._geometry.create_edge_mask(tile_idx, p._edge_mask_width, p._edge_mask_feather)

        # Convert PIL to tensor and normalize to 0-1
        edge_tensor = torch.from_numpy(np.array(edge_mask_pil)).float() / 255.0

        # Resize edge mask to tile_size if needed
        if edge_tensor.shape[0] != tile_size[1] or edge_tensor.shape[1] != tile_size[0]:
            edge_tensor = F.interpolate(
                edge_tensor.unsqueeze(0).unsqueeze(0),
                size=(tile_size[1], tile_size[0]),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)

        # Resize to latent dimensions
        latent_h, latent_w = latent["samples"].shape[-2], latent["samples"].shape[-1]
        edge_latent = F.interpolate(
            edge_tensor.unsqueeze(0).unsqueeze(0),
            size=(latent_h, latent_w),
            mode='bilinear',
            align_corners=False
        )  # Shape: [1, 1, H, W]

        # Combine with existing noise_mask or use edge mask alone
        if "noise_mask" in latent:
            # Combine: max(existing_mask, edge_mask) - edges always get denoised
            existing_mask = latent["noise_mask"]
            combined_mask = torch.max(existing_mask, edge_latent)
            latent["noise_mask"] = combined_mask
            print(f"  [EdgeMask] Combined with DiffDiff: tile_idx={tile_idx}, edge_shape={edge_latent.shape}")
        else:
            # No existing mask - use edge mask alone
            latent["noise_mask"] = edge_latent
            print(f"  [EdgeMask] Applied alone: tile_idx={tile_idx}, edge_shape={edge_latent.shape}")

    # ===== CONTROLNET PER-TILE =====
    # If ControlNet params are provided, crop control image and apply patch
    model_for_sampling = p.model
    if p._model_patch is not None and p._control_image_tensor is not None:
        # Crop control image to tile region
        x1, y1, x2, y2 = crop_region
        control_cropped = p._control_image_tensor[:, y1:y2, x1:x2, :]

        # Resize to tile_size if needed
        if control_cropped.shape[1] != tile_size[1] or control_cropped.shape[2] != tile_size[0]:
            control_cropped = F.interpolate(
                control_cropped.movedim(-1, 1),  # BHWC -> BCHW
                size=(tile_size[1], tile_size[0]),
                mode='bilinear',
                align_corners=False
            ).movedim(1, -1)  # BCHW -> BHWC

        # Crop control mask if provided
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

            # DiffSynthCnetPatch expects mask as [B, C, H, W] format
            control_mask_cropped = control_mask_cropped.unsqueeze(0).unsqueeze(0)

        # Clone model so patch only affects THIS tile
        model_for_sampling = p._original_model.clone()

        # Create and apply ControlNet patch
        if is_zimage_control(p._model_patch):
            cnet_patch = ZImageControlPatch(p._model_patch, p.vae, control_cropped, p._control_strength)
            patch_type = "ZImage"
        else:
            cnet_patch = DiffSynthCnetPatch(p._model_patch, p.vae, control_cropped, p._control_strength, control_mask_cropped)
            patch_type = "DiffSynth"
        model_for_sampling.set_model_double_block_patch(cnet_patch)

        mask_info = f", mask_shape={control_mask_cropped.shape}" if control_mask_cropped is not None and patch_type == "DiffSynth" else ""
        print(f"  [ControlNet:{patch_type}] Tile control: crop_region={crop_region}, control_shape={control_cropped.shape}{mask_info}")

    # ===== TWO-LATENT BLEND =====
    # Creates two noised latents (low/high), blends via mask, samples without noise
    if p._denoise_mask_pil is not None and p._denoise_high is not None and p._denoise_low is not None:
        from comfy.samplers import calculate_sigmas

        # Step 1: Crop mask to tile region
        mask_cropped = p._denoise_mask_pil.crop(crop_region)
        if mask_cropped.size != tile_size:
            mask_cropped = mask_cropped.resize(tile_size, Image.Resampling.BILINEAR)

        # Convert mask to tensor
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

        # Step 2: Get clean latent and generate noise
        clean_latent = latent["samples"].clone()
        generator = torch.Generator(device=device).manual_seed(p.seed)
        noise = torch.randn(clean_latent.shape, generator=generator, device=device, dtype=dtype)

        # Step 3: Get model sampling object
        model_sampling = p.model.get_model_object("model_sampling")

        denoise_low = p._denoise_low
        denoise_high = p._denoise_high
        max_denoise = max(denoise_low, denoise_high)

        # Calculate sigmas for each denoise level
        def get_start_sigma(denoise_val):
            if denoise_val <= 0.0:
                return 0.0
            if denoise_val >= 0.9999:
                new_steps = p.steps
            else:
                new_steps = int(p.steps / denoise_val)
            sigmas = calculate_sigmas(model_sampling, p.scheduler, new_steps)
            start_sigma = sigmas[-(p.steps + 1)].item() if denoise_val < 0.9999 else sigmas[0].item()
            return start_sigma

        sigma_low = get_start_sigma(denoise_low)
        sigma_high = get_start_sigma(denoise_high)
        sigma_max = get_start_sigma(max_denoise)

        # Step 4: Create noised latents
        latent_a = model_sampling.noise_scaling(
            torch.tensor([sigma_low], device=device, dtype=dtype),
            noise.clone(), clean_latent.clone(), max_denoise=False
        )
        latent_b = model_sampling.noise_scaling(
            torch.tensor([sigma_high], device=device, dtype=dtype),
            noise.clone(), clean_latent.clone(), max_denoise=False
        )

        # Step 5: Blend latents (mask=0 -> low, mask=1 -> high)
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

        latent["samples"] = blended_latent

        # Step 6: Sample WITHOUT adding noise
        print(f"    - Sampling with add_noise=False, denoise={max_denoise}")
        samples = sample_no_add_noise(model_for_sampling, p.seed, p.steps, p.cfg, p.sampler_name, p.scheduler,
                                       positive_cropped, negative_cropped, latent,
                                       max_denoise, p.custom_sampler, p.custom_sigmas)
    else:
        # === STANDARD SAMPLING ===
        samples = sample(model_for_sampling, p.seed, p.steps, p.cfg, p.sampler_name, p.scheduler,
                         positive_cropped, negative_cropped, latent, p.denoise,
                         p.custom_sampler, p.custom_sigmas)

    # === UPDATE PROGRESS BAR ===
    if p.progress_bar_enabled:
        p.pbar.update(1)

    # === DECODE LATENT TO IMAGE ===
    if not p.tiled_decode:
        (decoded,) = p.vae_decoder.decode(p.vae, samples)
    else:
        (decoded,) = p.vae_decoder_tiled.decode(p.vae, samples, 512)

    # === CONVERT TO PIL AND COMPOSITE (Weighted Accumulation) ===
    tiles_sampled = [tensor_to_pil(decoded, i) for i in range(len(decoded))]

    for i, tile_sampled in enumerate(tiles_sampled):
        init_image = shared.batch[i]

        # Store processed tile for debug output at consistent tile_size
        debug_tile = tile_sampled
        if debug_tile.size != tile_size:
            debug_tile = debug_tile.resize(tile_size, Image.Resampling.LANCZOS)
        p._processed_tiles.append(pil_to_tensor(debug_tile))

        # Resize back to original tile size (varies per tile due to crop region)
        if tile_sampled.size != initial_tile_size:
            tile_sampled = tile_sampled.resize(initial_tile_size, Image.Resampling.LANCZOS)

        # Initialize accumulators on first tile
        if p._original_images is None:
            p._original_images = [img.copy() for img in shared.batch]
            p._tile_sum = [np.zeros((img.height, img.width, 3), dtype=np.float32)
                           for img in shared.batch]
            p._weight_sum = [np.zeros((img.height, img.width), dtype=np.float32)
                             for img in shared.batch]

        # Get mask at full canvas size (already blurred)
        mask_np = np.array(image_mask).astype(np.float32) / 255.0

        # Create full-canvas tile image
        tile_canvas = np.zeros((init_image.height, init_image.width, 3), dtype=np.float32)
        x1, y1 = crop_region[:2]
        x2 = x1 + tile_sampled.width
        y2 = y1 + tile_sampled.height
        tile_canvas[y1:y2, x1:x2] = np.array(tile_sampled).astype(np.float32)

        # Accumulate weighted contributions
        p._tile_sum[i] += tile_canvas * mask_np[..., np.newaxis]
        p._weight_sum[i] += mask_np

        # Compute blended result using weighted average
        original_np = np.array(p._original_images[i]).astype(np.float32)
        weight = p._weight_sum[i]

        # Where weight > 0: use accumulated tiles; else: use original
        weight_safe = np.maximum(weight, 1e-6)
        tile_result = p._tile_sum[i] / weight_safe[..., np.newaxis]

        # Blend factor: how much tile vs original (clamped to 0-1)
        blend = np.clip(weight, 0, 1)
        result_np = original_np * (1 - blend[..., np.newaxis]) + tile_result * blend[..., np.newaxis]

        # Update shared.batch - CLAMP to [0, 255] to prevent uint8 overflow/underflow
        shared.batch[i] = Image.fromarray(np.clip(result_np, 0, 255).astype(np.uint8))

    processed = Processed(p, [shared.batch[0]], p.seed, None)
    return processed
