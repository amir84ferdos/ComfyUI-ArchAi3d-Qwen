# ArchAi3D Smart Tile Detailer
#
# Process each SEG with its own conditioning from Smart Tile Conditioning
# Optimized pipeline for per-tile prompts
#
# Author: Amir Ferdos (ArchAi3d)
# Version: 2.0.1 - Fixed mask dimension bug in F.interpolate
#                  v2.0.0: Simplified - removed guide_size and feather
#                  Tiles processed at original size
#                  Use SEGS Mask Blur node for mask feathering
# License: Dual License (Free for personal use, Commercial license required for business use)

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import comfy.samplers
import comfy.sample
import latent_preview


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def tensor_to_pil(tensor):
    """Convert tensor (B,H,W,C) to PIL Image."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    np_img = (tensor.cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(np_img)


def pil_to_tensor(pil_img):
    """Convert PIL Image to tensor (1,H,W,C)."""
    np_img = np.array(pil_img).astype(np.float32) / 255.0
    return torch.from_numpy(np_img).unsqueeze(0)


def tensor_resize(tensor, width, height):
    """Resize tensor (B,H,W,C) to new dimensions."""
    # Convert to (B,C,H,W) for interpolate
    t = tensor.permute(0, 3, 1, 2)
    t = F.interpolate(t, size=(height, width), mode='bilinear', align_corners=False)
    # Convert back to (B,H,W,C)
    return t.permute(0, 2, 3, 1)


def tensor_crop(tensor, crop_region):
    """Crop tensor (B,H,W,C) using crop_region (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = crop_region
    return tensor[:, y1:y2, x1:x2, :]


def composite_images(base, overlay, mask, position):
    """Composite overlay onto base at position using mask."""
    x1, y1 = position
    h, w = overlay.shape[1], overlay.shape[2]

    # Ensure mask is 3D (H, W) -> (H, W, 1) for broadcasting
    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)

    # Get the region from base
    base_region = base[:, y1:y1+h, x1:x1+w, :]

    # Blend
    blended = base_region * (1 - mask) + overlay * mask

    # Put back
    result = base.clone()
    result[:, y1:y1+h, x1:x1+w, :] = blended

    return result


def parse_tile_position(label):
    """
    Parse tile position from SEG label.

    Labels are in format "tile_Y_X" (e.g., "tile_0_0", "tile_1_2")

    Returns:
        (row, col) tuple or None if not parseable
    """
    if label and label.startswith("tile_"):
        parts = label.split("_")
        if len(parts) == 3:
            try:
                row = int(parts[1])
                col = int(parts[2])
                return (row, col)
            except ValueError:
                pass
    return None


# ============================================================================
# NODE CLASS
# ============================================================================

class ArchAi3D_Smart_Tile_Detailer:
    """
    Process each SEG with its own conditioning.

    Takes SEGS from Smart Tile SEGS and conditionings from Smart Tile Conditioning,
    processes each tile with its unique prompt, and composites the result.

    Features:
    - True per-tile prompts (each tile gets unique conditioning)
    - Optimized pipeline (conditioning already encoded)
    - Tiles processed at their original size (no upscaling)
    - Uses SEG masks directly for blending (pre-blur with SEGS Mask Blur node)
    """

    CATEGORY = "ArchAi3d/Utils"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_tiles"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Input image (scaled_image from Smart Tile Calculator)"
                }),
                "segs": ("SEGS", {
                    "tooltip": "SEGS from Smart Tile SEGS (optionally blurred with SEGS Mask Blur)"
                }),
                "model": ("MODEL", {
                    "tooltip": "Diffusion model for sampling"
                }),
                "vae": ("VAE", {
                    "tooltip": "VAE for encode/decode"
                }),
                "conditionings": ("CONDITIONING_LIST", {
                    "tooltip": "List of conditionings from Smart Tile Conditioning"
                }),
                "negative": ("CONDITIONING", {
                    "tooltip": "Negative conditioning from Smart Tile Conditioning"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed for sampling"
                }),
                "steps": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Number of sampling steps"
                }),
                "cfg": ("FLOAT", {
                    "default": 7.0,
                    "min": 1.0,
                    "max": 30.0,
                    "step": 0.5,
                    "tooltip": "CFG scale"
                }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {
                    "default": "euler",
                    "tooltip": "Sampler algorithm"
                }),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {
                    "default": "normal",
                    "tooltip": "Noise scheduler"
                }),
                "denoise": ("FLOAT", {
                    "default": 0.4,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Denoise strength (lower = preserve more detail)"
                }),
            },
            "optional": {
                "bundle": ("SMART_TILE_BUNDLE", {
                    "tooltip": "Connect bundle from Smart Tile Calculator (auto-fills image)"
                }),
            }
        }

    def process_tiles(self, image, segs, model, vae, conditionings, negative,
                      seed, steps, cfg, sampler_name, scheduler, denoise,
                      bundle=None):
        """
        Process each SEG with its own conditioning at original tile size.

        Args:
            image: Input image tensor (B, H, W, C)
            segs: SEGS tuple ((h, w), [list of SEG])
            model: Diffusion model
            vae: VAE model
            conditionings: List of CONDITIONING (one per SEG)
            negative: Negative CONDITIONING
            seed: Random seed
            steps: Sampling steps
            cfg: CFG scale
            sampler_name: Sampler name
            scheduler: Scheduler name
            denoise: Denoise strength
            bundle: Optional bundle from Smart Tile Calculator

        Returns:
            Processed image with all tiles enhanced
        """
        # Extract from bundle if available
        tiles_x = None
        tiles_y = None

        if bundle is not None:
            image = bundle.get("scaled_image", image)
            tiles_x = bundle.get("tiles_x", None)
            tiles_y = bundle.get("tiles_y", None)
            print(f"[Smart Tile Detailer v2.0] Using bundle: grid={tiles_x}x{tiles_y}")

        # Unpack SEGS
        (img_h, img_w), seg_list = segs

        if not seg_list:
            print("[Smart Tile Detailer v2.0] No segments to process")
            return (image,)

        num_segs = len(seg_list)
        num_conds = len(conditionings)

        # Try to infer grid size from SEG labels if not provided
        if tiles_x is None or tiles_y is None:
            if seg_list and seg_list[0].label:
                rows = set()
                cols = set()
                for seg in seg_list:
                    pos = parse_tile_position(seg.label)
                    if pos:
                        rows.add(pos[0])
                        cols.add(pos[1])
                if rows and cols:
                    tiles_y = max(rows) + 1
                    tiles_x = max(cols) + 1
                    print(f"[Smart Tile Detailer v2.0] Inferred grid: {tiles_x}x{tiles_y} from labels")

        if num_conds < num_segs:
            print(f"[Smart Tile Detailer v2.0] Warning: {num_segs} SEGs but only {num_conds} conditionings. Reusing last conditioning.")

        print(f"\n[Smart Tile Detailer v2.0] Processing {num_segs} tiles at original size...")
        print(f"  Grid: {tiles_x}x{tiles_y if tiles_x and tiles_y else 'unknown'}")
        print(f"  Sampler: {sampler_name}, Steps: {steps}, CFG: {cfg}, Denoise: {denoise}")

        # Clone image for output
        result = image.clone()

        # Process each SEG
        for i, seg in enumerate(seg_list):
            # Get conditioning for this SEG (cycle if not enough)
            cond_idx = min(i, num_conds - 1)
            positive = conditionings[cond_idx]

            # Get crop region (context area including padding)
            crop_region = seg.crop_region
            x1, y1, x2, y2 = crop_region
            crop_w = x2 - x1
            crop_h = y2 - y1

            # Get bbox (actual tile area without padding)
            bbox = seg.bbox
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
            tile_w = bbox_x2 - bbox_x1
            tile_h = bbox_y2 - bbox_y1

            # Parse tile position from label
            tile_pos = parse_tile_position(seg.label)
            if tile_pos:
                row, col = tile_pos
            else:
                row = i // (tiles_x or 1) if tiles_x else 0
                col = i % (tiles_x or 1) if tiles_x else 0

            print(f"\n  Tile {i+1}/{num_segs} (row={row}, col={col}): crop={crop_w}x{crop_h}, bbox={tile_w}x{tile_h}")

            # Crop image at the crop_region (includes padding for context)
            cropped = tensor_crop(image, crop_region)

            # Encode with VAE (process at original size)
            latent_samples = vae.encode(cropped[:, :, :, :3])
            latent = {"samples": latent_samples}

            # Use SEG mask for noise masking if available
            if seg.cropped_mask is not None:
                mask = torch.from_numpy(seg.cropped_mask).float()

                # Ensure mask is 4D (N, C, H, W) for F.interpolate
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)
                elif mask.dim() == 3:
                    mask = mask.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)

                # Resize mask to latent size (1/8 of image)
                latent_h = crop_h // 8
                latent_w = crop_w // 8
                noise_mask = F.interpolate(mask,
                                           size=(latent_h, latent_w),
                                           mode='bilinear', align_corners=False)
                latent["noise_mask"] = noise_mask.squeeze(0)

            # Sample
            print(f"    Sampling with conditioning {cond_idx}...")

            # Prepare noise
            noise = comfy.sample.prepare_noise(latent["samples"], seed + i)

            # Get noise mask if present
            noise_mask = latent.get("noise_mask", None)

            # Prepare callback
            callback = latent_preview.prepare_callback(model, steps)

            # Sample
            samples = comfy.sample.sample(
                model, noise, steps, cfg,
                sampler_name, scheduler,
                positive, negative,
                latent["samples"],
                denoise=denoise,
                noise_mask=noise_mask,
                callback=callback,
                disable_pbar=True,
                seed=seed + i
            )

            # Decode
            decoded = vae.decode(samples)

            # Calculate bbox position relative to crop_region
            rel_x1 = bbox_x1 - x1
            rel_y1 = bbox_y1 - y1
            rel_x2 = bbox_x2 - x1
            rel_y2 = bbox_y2 - y1

            # Extract just the tile portion from decoded result
            tile_result = decoded[:, rel_y1:rel_y2, rel_x1:rel_x2, :]

            # Ensure size matches (handle any rounding issues)
            if tile_result.shape[1] != tile_h or tile_result.shape[2] != tile_w:
                tile_result = tensor_resize(tile_result, tile_w, tile_h)

            # Create blend mask from SEG mask (already blurred if SEGS Mask Blur was used)
            if seg.cropped_mask is not None:
                # Extract the tile portion of the mask
                full_mask = torch.from_numpy(seg.cropped_mask).float()
                if full_mask.dim() == 2:
                    blend_mask = full_mask[rel_y1:rel_y2, rel_x1:rel_x2]
                else:
                    blend_mask = full_mask[0, rel_y1:rel_y2, rel_x1:rel_x2]

                # Ensure size matches
                if blend_mask.shape[0] != tile_h or blend_mask.shape[1] != tile_w:
                    blend_mask = F.interpolate(blend_mask.unsqueeze(0).unsqueeze(0),
                                                size=(tile_h, tile_w),
                                                mode='bilinear', align_corners=False).squeeze()
            else:
                # No mask - use full opacity
                blend_mask = torch.ones((tile_h, tile_w), dtype=torch.float32)

            blend_mask = blend_mask.unsqueeze(0).unsqueeze(-1)  # (1, H, W, 1)

            # Composite at bbox position
            result = composite_images(result, tile_result, blend_mask, (bbox_x1, bbox_y1))

            print(f"    Done! Composited tile at ({bbox_x1},{bbox_y1}) size {tile_w}x{tile_h}")

        print(f"\n[Smart Tile Detailer v2.0] Completed! Processed {num_segs} tiles.")

        return (result,)


# ============================================================================
# NODE REGISTRATION
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "ArchAi3D_Smart_Tile_Detailer": ArchAi3D_Smart_Tile_Detailer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_Smart_Tile_Detailer": "🔧 Smart Tile Detailer",
}
