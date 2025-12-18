"""
Simple USDU - Tile Sampler
===========================

Process a single tile with diffusion sampling.
Optionally supports DiffDiff (per-pixel denoise) and ControlNet.
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import comfy.samplers

# Import from ComfyUI's nodes module (not our local nodes package)
# Use sys.modules to get the already-loaded ComfyUI nodes module
_comfy_nodes = sys.modules.get('nodes')
if _comfy_nodes is None:
    # Fallback: force absolute import
    import importlib
    _comfy_nodes = importlib.import_module('nodes')

common_ksampler = _comfy_nodes.common_ksampler
VAEEncode = _comfy_nodes.VAEEncode
VAEDecode = _comfy_nodes.VAEDecode

from .utils import tensor_to_pil, pil_to_tensor


class ArchAi3D_Simple_Tile_Sampler:
    """
    Process a single tile with diffusion sampling.

    This is a simplified KSampler that:
    1. Encodes tile to latent
    2. (Optional) Applies DiffDiff noise mask
    3. (Optional) Applies ControlNet
    4. Samples with model
    5. Decodes back to image

    Use this with Simple_Tile_Cropper and Simple_Tile_Compositor for a
    complete tile-based upscaling pipeline.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tile": ("IMAGE",),  # Cropped tile from Tile Cropper
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "denoise_mask": ("MASK",),  # For DiffDiff - white=more denoise
                "control_image": ("IMAGE",),  # For ControlNet
                "controlnet": ("CONTROL_NET",),  # ControlNet model
                "control_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("processed_tile",)
    FUNCTION = "sample_tile"
    CATEGORY = "ArchAi3d/Upscaling/Simple USDU"

    def sample_tile(self, tile, model, positive, negative, vae, seed, steps, cfg,
                    sampler_name, scheduler, denoise,
                    denoise_mask=None, control_image=None, controlnet=None, control_strength=1.0):
        """
        Process a single tile with diffusion sampling.

        Args:
            tile: IMAGE tensor [1, H, W, C] - cropped tile
            model: ComfyUI MODEL
            positive: CONDITIONING
            negative: CONDITIONING
            vae: VAE for encode/decode
            seed: Random seed
            steps: Number of sampling steps
            cfg: CFG scale
            sampler_name: Sampler name
            scheduler: Scheduler name
            denoise: Denoise strength

        Optional:
            denoise_mask: MASK for DiffDiff (white=more denoise, black=less)
            control_image: IMAGE for ControlNet
            controlnet: CONTROL_NET model
            control_strength: ControlNet strength

        Returns:
            processed_tile: IMAGE tensor [1, H, W, C]
        """
        # === ENCODE TO LATENT ===
        vae_encoder = VAEEncode()
        (latent,) = vae_encoder.encode(vae, tile)

        # === DIFFDIFF: Apply noise mask if provided ===
        if denoise_mask is not None:
            # Resize mask to latent size
            latent_h = latent["samples"].shape[-2]
            latent_w = latent["samples"].shape[-1]

            # Ensure mask is 2D or 3D
            if len(denoise_mask.shape) == 2:
                mask_2d = denoise_mask
            else:
                mask_2d = denoise_mask[0]  # Take first if batched

            mask_latent = F.interpolate(
                mask_2d.unsqueeze(0).unsqueeze(0),
                size=(latent_h, latent_w),
                mode='bilinear',
                align_corners=False
            )  # [1, 1, H, W]

            # Set noise_mask for Differential Diffusion
            latent["noise_mask"] = mask_latent
            print(f"  [DiffDiff] Applied mask: latent_size={latent_w}x{latent_h}")

        # === CONTROLNET: Apply if provided ===
        model_for_sampling = model
        if controlnet is not None and control_image is not None:
            # Apply ControlNet to conditioning
            from comfy_extras.nodes_controlnet import ControlNetApplyAdvanced

            cnet_apply = ControlNetApplyAdvanced()
            (positive_cnet, negative_cnet) = cnet_apply.apply_controlnet(
                positive=positive,
                negative=negative,
                control_net=controlnet,
                image=control_image,
                strength=control_strength,
                vae=vae
            )
            positive = positive_cnet
            negative = negative_cnet
            print(f"  [ControlNet] Applied with strength={control_strength}")

        # === SAMPLE ===
        (samples,) = common_ksampler(
            model_for_sampling, seed, steps, cfg, sampler_name, scheduler,
            positive, negative, latent, denoise=denoise
        )

        # === DECODE ===
        vae_decoder = VAEDecode()
        (decoded,) = vae_decoder.decode(vae, samples)

        return (decoded,)


class ArchAi3D_Simple_Edge_Mask:
    """
    Create an edge mask for DiffDiff per-tile processing.

    This creates a mask where:
    - White center = MORE denoising (regenerate content)
    - Black edges = LESS denoising (preserve for blending)

    Use this to create masks for tiles that need edge preservation
    when blending with neighbors.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tile_idx": ("INT", {"default": 0, "min": 0, "max": 1000}),
                "tile_width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "tile_height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "tiles_x": ("INT", {"default": 2, "min": 1, "max": 100}),
                "tiles_y": ("INT", {"default": 2, "min": 1, "max": 100}),
                "edge_width": ("INT", {"default": 32, "min": 0, "max": 256,
                                       "tooltip": "Width of edge border in pixels"}),
                "edge_feather": ("INT", {"default": 8, "min": 0, "max": 64,
                                         "tooltip": "Feather amount for soft edges"}),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("edge_mask", "edge_mask_rgb")
    FUNCTION = "create_edge_mask"
    CATEGORY = "ArchAi3d/Upscaling/Simple USDU"

    def create_edge_mask(self, tile_idx, tile_width, tile_height, tiles_x, tiles_y,
                         edge_width, edge_feather):
        """
        Create edge mask for DiffDiff.

        The mask has:
        - White (255) in CENTER = full denoising
        - Black (0) at EDGES with neighbors = preserve for blending
        - No black edge on image boundaries (no neighbor to blend with)

        Args:
            tile_idx: Which tile (0-indexed)
            tile_width: Width of tile
            tile_height: Height of tile
            tiles_x: Number of tiles horizontally
            tiles_y: Number of tiles vertically
            edge_width: Width of black edge border
            edge_feather: Feather amount for soft transition

        Returns:
            edge_mask: MASK tensor [tile_height, tile_width]
            edge_mask_rgb: IMAGE tensor for preview
        """
        from PIL import ImageDraw, ImageFilter
        from .utils import get_tile_position

        xi, yi = get_tile_position(tile_idx, tiles_x, tiles_y)

        # Start with white (full denoise in center)
        mask = Image.new('L', (tile_width, tile_height), 255)
        draw = ImageDraw.Draw(mask)

        # Draw BLACK edges only where tile has neighbors
        # (preserve edges for blending with neighbors)
        if xi > 0:  # Has left neighbor
            draw.rectangle((0, 0, edge_width, tile_height), fill=0)
        if xi < tiles_x - 1:  # Has right neighbor
            draw.rectangle((tile_width - edge_width, 0, tile_width, tile_height), fill=0)
        if yi > 0:  # Has top neighbor
            draw.rectangle((0, 0, tile_width, edge_width), fill=0)
        if yi < tiles_y - 1:  # Has bottom neighbor
            draw.rectangle((0, tile_height - edge_width, tile_width, tile_height), fill=0)

        # Apply feather/blur
        if edge_feather > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(edge_feather))

        # Convert to tensors
        mask_np = np.array(mask).astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_np)  # [H, W]

        # RGB preview
        mask_rgb = mask.convert('RGB')
        mask_rgb_tensor = pil_to_tensor(mask_rgb)

        return (mask_tensor, mask_rgb_tensor)


# Node registration
NODE_CLASS_MAPPINGS = {
    "ArchAi3D_Simple_Tile_Sampler": ArchAi3D_Simple_Tile_Sampler,
    "ArchAi3D_Simple_Edge_Mask": ArchAi3D_Simple_Edge_Mask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_Simple_Tile_Sampler": "Simple Tile Sampler",
    "ArchAi3D_Simple_Edge_Mask": "Simple Edge Mask (DiffDiff)",
}
