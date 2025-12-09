# ArchAi3D Smart Tile Sampler
#
# Process each SEG through diffusion WITHOUT compositing
# Outputs SEGS with rendered tiles for use with Smart Tile Merger
#
# Author: Amir Ferdos (ArchAi3d)
# Version: 1.0.0 - Initial release (sampling only, no compositing)
# License: Dual License (Free for personal use, Commercial license required for business use)

import torch
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
import comfy.samplers
import comfy.sample
import latent_preview

# Define SEG namedtuple (compatible with Impact Pack)
SEG = namedtuple("SEG",
                 ['cropped_image', 'cropped_mask', 'confidence', 'crop_region', 'bbox', 'label', 'control_net_wrapper'],
                 defaults=[None])


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def tensor_crop(tensor, crop_region):
    """Crop tensor (B,H,W,C) using crop_region (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = crop_region
    return tensor[:, y1:y2, x1:x2, :]


def tensor_resize(tensor, width, height):
    """Resize tensor (B,H,W,C) to new dimensions."""
    t = tensor.permute(0, 3, 1, 2)
    t = F.interpolate(t, size=(height, width), mode='bilinear', align_corners=False)
    return t.permute(0, 2, 3, 1)


def parse_tile_position(label):
    """Parse tile position from SEG label (format: tile_Y_X)."""
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

class ArchAi3D_Smart_Tile_Sampler:
    """
    Process tiles through diffusion WITHOUT compositing.

    This node samples each tile independently and outputs SEGS containing
    the rendered tiles. Use Smart Tile Merger to composite with proper
    normalized weight blending (no seams).

    Key difference from Smart Tile Detailer:
    - NO compositing - just sampling
    - Output SEGS contains rendered tiles
    - Pass-through image for Merger's base

    Workflow:
    Calculator → SEGS Blur → Sampler → Merger → Final Image
    """

    CATEGORY = "ArchAi3d/Utils"
    RETURN_TYPES = ("IMAGE", "SEGS")
    RETURN_NAMES = ("image", "segs")
    FUNCTION = "sample_tiles"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Input image (pass-through to Merger as base)"
                }),
                "segs": ("SEGS", {
                    "tooltip": "SEGS from Smart Tile SEGS Blur"
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
                    "tooltip": "Negative conditioning"
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
                    "tooltip": "Denoise strength"
                }),
            },
            "optional": {
                "bundle": ("SMART_TILE_BUNDLE", {
                    "tooltip": "Bundle from Smart Tile Calculator"
                }),
            }
        }

    def sample_tiles(self, image, segs, model, vae, conditionings, negative,
                     seed, steps, cfg, sampler_name, scheduler, denoise,
                     bundle=None):
        """
        Sample each tile through diffusion, output SEGS with rendered tiles.

        NO COMPOSITING - use Smart Tile Merger for proper blending.
        """
        # Extract grid info from bundle
        tiles_x = None
        tiles_y = None

        if bundle is not None:
            tiles_x = bundle.get("tiles_x", None)
            tiles_y = bundle.get("tiles_y", None)
            print(f"[Smart Tile Sampler v1.0] Using bundle: grid={tiles_x}x{tiles_y}")

        # Unpack SEGS
        (img_h, img_w), seg_list = segs

        if not seg_list:
            print("[Smart Tile Sampler v1.0] No segments to process")
            return (image, segs)

        num_segs = len(seg_list)
        num_conds = len(conditionings)

        # Infer grid size from labels if not provided
        if tiles_x is None or tiles_y is None:
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

        if num_conds < num_segs:
            print(f"[Smart Tile Sampler v1.0] Warning: {num_segs} SEGs but only {num_conds} conditionings")

        print(f"\n[Smart Tile Sampler v1.0] Sampling {num_segs} tiles...")
        print(f"  Grid: {tiles_x}x{tiles_y if tiles_x and tiles_y else 'unknown'}")
        print(f"  Sampler: {sampler_name}, Steps: {steps}, CFG: {cfg}, Denoise: {denoise}")
        print(f"  NOTE: No compositing - use Smart Tile Merger for final image")

        # Collect processed SEGs
        processed_segs = []

        # Process each SEG
        for i, seg in enumerate(seg_list):
            # Get conditioning
            cond_idx = min(i, num_conds - 1)
            positive = conditionings[cond_idx]

            # Get regions
            crop_region = seg.crop_region
            x1, y1, x2, y2 = crop_region
            crop_w = x2 - x1
            crop_h = y2 - y1

            bbox = seg.bbox
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
            tile_w = bbox_x2 - bbox_x1
            tile_h = bbox_y2 - bbox_y1

            # Parse position
            tile_pos = parse_tile_position(seg.label)
            if tile_pos:
                row, col = tile_pos
            else:
                row = i // (tiles_x or 1) if tiles_x else 0
                col = i % (tiles_x or 1) if tiles_x else 0

            print(f"\n  Tile {i+1}/{num_segs} ({seg.label}): crop={crop_w}x{crop_h}, bbox={tile_w}x{tile_h}")

            # Crop image at crop_region
            cropped = tensor_crop(image, crop_region)

            # Encode with VAE
            latent_samples = vae.encode(cropped[:, :, :, :3])
            latent = {"samples": latent_samples}

            # Apply noise mask if available
            if seg.cropped_mask is not None:
                mask = torch.from_numpy(seg.cropped_mask).float()
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0).unsqueeze(0)
                elif mask.dim() == 3:
                    mask = mask.unsqueeze(0)

                latent_h = crop_h // 8
                latent_w = crop_w // 8
                noise_mask = F.interpolate(mask, size=(latent_h, latent_w),
                                           mode='bilinear', align_corners=False)
                latent["noise_mask"] = noise_mask.squeeze(0)

            # Sample
            print(f"    Sampling with conditioning {cond_idx}...")
            noise = comfy.sample.prepare_noise(latent["samples"], seed + i)
            noise_mask = latent.get("noise_mask", None)
            callback = latent_preview.prepare_callback(model, steps)

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

            # Calculate bbox relative to crop
            rel_x1 = bbox_x1 - x1
            rel_y1 = bbox_y1 - y1
            rel_x2 = bbox_x2 - x1
            rel_y2 = bbox_y2 - y1

            # Get decoded dimensions
            decoded_h = decoded.shape[1]
            decoded_w = decoded.shape[2]

            # Safe extraction bounds
            safe_rel_x1 = max(0, min(rel_x1, decoded_w))
            safe_rel_y1 = max(0, min(rel_y1, decoded_h))
            safe_rel_x2 = max(0, min(rel_x2, decoded_w))
            safe_rel_y2 = max(0, min(rel_y2, decoded_h))

            # Extract tile portion
            tile_result = decoded[:, safe_rel_y1:safe_rel_y2, safe_rel_x1:safe_rel_x2, :]

            # Resize if needed
            if tile_result.shape[1] != tile_h or tile_result.shape[2] != tile_w:
                print(f"    Resizing tile from {tile_result.shape[2]}x{tile_result.shape[1]} to {tile_w}x{tile_h}")
                tile_result = tensor_resize(tile_result, tile_w, tile_h)

            # Get blend mask from input SEG
            if seg.cropped_mask is not None:
                full_mask = torch.from_numpy(seg.cropped_mask).float()
                if full_mask.dim() == 2:
                    mask_h, mask_w = full_mask.shape
                else:
                    mask_h, mask_w = full_mask.shape[1], full_mask.shape[2]

                safe_mask_y1 = max(0, min(rel_y1, mask_h))
                safe_mask_y2 = max(0, min(rel_y2, mask_h))
                safe_mask_x1 = max(0, min(rel_x1, mask_w))
                safe_mask_x2 = max(0, min(rel_x2, mask_w))

                if full_mask.dim() == 2:
                    blend_mask = full_mask[safe_mask_y1:safe_mask_y2, safe_mask_x1:safe_mask_x2]
                else:
                    blend_mask = full_mask[0, safe_mask_y1:safe_mask_y2, safe_mask_x1:safe_mask_x2]

                if blend_mask.shape[0] != tile_h or blend_mask.shape[1] != tile_w:
                    blend_mask = F.interpolate(blend_mask.unsqueeze(0).unsqueeze(0),
                                               size=(tile_h, tile_w),
                                               mode='bilinear', align_corners=False).squeeze()
            else:
                blend_mask = torch.ones((tile_h, tile_w), dtype=torch.float32)

            # Create processed SEG with rendered tile
            # NO COMPOSITING - just store the tile for Merger
            processed_seg = SEG(
                cropped_image=tile_result.cpu().numpy(),  # Rendered tile
                cropped_mask=blend_mask.cpu().numpy(),    # Blend mask for Merger
                confidence=seg.confidence,
                crop_region=seg.crop_region,
                bbox=seg.bbox,
                label=seg.label,
                control_net_wrapper=seg.control_net_wrapper
            )
            processed_segs.append(processed_seg)

            print(f"    Done! Tile stored in SEGS (no compositing)")

        # Build output SEGS
        output_segs = ((img_h, img_w), processed_segs)

        print(f"\n[Smart Tile Sampler v1.0] Completed! {num_segs} tiles sampled.")
        print(f"  Output: pass-through IMAGE + SEGS with rendered tiles")
        print(f"  Next: Connect to Smart Tile Merger for seamless compositing")

        # Return pass-through image + processed SEGS
        return (image, output_segs)


# ============================================================================
# NODE REGISTRATION
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "ArchAi3D_Smart_Tile_Sampler": ArchAi3D_Smart_Tile_Sampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_Smart_Tile_Sampler": "🎨 Smart Tile Sampler",
}
