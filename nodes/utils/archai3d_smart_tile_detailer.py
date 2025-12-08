# ArchAi3D Smart Tile Detailer
#
# Process each SEG with its own conditioning from Smart Tile Conditioning
# Optimized pipeline for per-tile prompts
#
# Author: Amir Ferdos (ArchAi3d)
# Version: 1.3.0 - Fixed tile seams by only feathering image boundary edges
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


def create_feathered_mask(height, width, feather, feather_top=True, feather_bottom=True,
                          feather_left=True, feather_right=True):
    """
    Create a feathered mask for blending.

    Only feathers edges where feather_* is True. This allows feathering only
    at image boundaries, not between adjacent tiles.

    Args:
        height, width: Mask dimensions
        feather: Feather amount in pixels
        feather_top: Apply feathering to top edge
        feather_bottom: Apply feathering to bottom edge
        feather_left: Apply feathering to left edge
        feather_right: Apply feathering to right edge

    Returns:
        numpy array mask with feathered edges
    """
    mask = np.ones((height, width), dtype=np.float32)

    if feather > 0:
        # Create feathered edges only where specified
        for i in range(feather):
            alpha = (i + 1) / feather

            # Top edge
            if feather_top and i < height:
                mask[i, :] *= alpha

            # Bottom edge
            if feather_bottom and height - i - 1 >= 0:
                mask[height - i - 1, :] *= alpha

            # Left edge
            if feather_left and i < width:
                mask[:, i] *= alpha

            # Right edge
            if feather_right and width - i - 1 >= 0:
                mask[:, width - i - 1] *= alpha

    return mask


def tensor_gaussian_blur_mask(mask, kernel_size):
    """Apply gaussian blur to mask for feathering."""
    if kernel_size <= 0:
        return mask

    # Ensure kernel size is odd
    kernel_size = kernel_size * 2 + 1

    # Create gaussian kernel
    sigma = kernel_size / 3.0
    x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()

    # Apply separable convolution
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)

    # Pad and convolve
    pad = kernel_size // 2
    mask = F.pad(mask, (pad, pad, pad, pad), mode='reflect')

    # Horizontal pass
    kernel_h = kernel_1d.view(1, 1, 1, -1).to(mask.device)
    mask = F.conv2d(mask, kernel_h, padding=0)

    # Vertical pass
    kernel_v = kernel_1d.view(1, 1, -1, 1).to(mask.device)
    mask = F.conv2d(mask, kernel_v, padding=0)

    return mask.squeeze()


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
    - Configurable sampling parameters
    - Smart feathering: only feathers image boundary edges, not tile seams
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
                    "tooltip": "SEGS from Smart Tile SEGS"
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
                    "tooltip": "Connect bundle from Smart Tile Calculator (auto-fills image, guide_size, feather, grid info)"
                }),
                "guide_size": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 2048,
                    "step": 8,
                    "tooltip": "Target size for tile processing (from Smart Tile Calculator)"
                }),
                "feather": ("INT", {
                    "default": 64,
                    "min": 0,
                    "max": 256,
                    "step": 8,
                    "tooltip": "Feather amount for image boundary blending"
                }),
            }
        }

    def process_tiles(self, image, segs, model, vae, conditionings, negative,
                      seed, steps, cfg, sampler_name, scheduler, denoise,
                      bundle=None, guide_size=512, feather=64):
        """
        Process each SEG with its own conditioning.

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
            guide_size: Target processing size
            feather: Feather amount for blending

        Returns:
            Processed image with all tiles enhanced
        """
        # Extract grid info from bundle if available
        tiles_x = None
        tiles_y = None

        if bundle is not None:
            image = bundle.get("scaled_image", image)
            guide_size = bundle.get("guide_size", guide_size)
            feather = bundle.get("mask_blur", feather)  # Use mask_blur as feather
            tiles_x = bundle.get("tiles_x", None)
            tiles_y = bundle.get("tiles_y", None)
            print(f"[Smart Tile Detailer v1.3] Using bundle: guide_size={guide_size}, feather={feather}, grid={tiles_x}x{tiles_y}")

        # Unpack SEGS
        (img_h, img_w), seg_list = segs

        if not seg_list:
            print("[Smart Tile Detailer v1.3] No segments to process")
            return (image,)

        num_segs = len(seg_list)
        num_conds = len(conditionings)

        # Try to infer grid size from number of segments if not provided
        if tiles_x is None or tiles_y is None:
            # Try to parse from first SEG label
            if seg_list and seg_list[0].label:
                # Count unique rows and cols from labels
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
                    print(f"[Smart Tile Detailer v1.3] Inferred grid: {tiles_x}x{tiles_y} from labels")

        if num_conds < num_segs:
            print(f"[Smart Tile Detailer v1.3] Warning: {num_segs} SEGs but only {num_conds} conditionings. Reusing last conditioning.")

        print(f"\n[Smart Tile Detailer v1.3] Processing {num_segs} tiles...")
        print(f"  Guide size: {guide_size}, Feather: {feather}")
        print(f"  Grid: {tiles_x}x{tiles_y if tiles_x and tiles_y else 'unknown'}")
        print(f"  Sampler: {sampler_name}, Steps: {steps}, CFG: {cfg}, Denoise: {denoise}")

        # Clone image for output
        result = image.clone()

        # Process each SEG
        for i, seg in enumerate(seg_list):
            # Get conditioning for this SEG (cycle if not enough)
            cond_idx = min(i, num_conds - 1)
            positive = conditionings[cond_idx]

            # Get crop region
            crop_region = seg.crop_region
            x1, y1, x2, y2 = crop_region
            crop_w = x2 - x1
            crop_h = y2 - y1

            # Parse tile position from label
            tile_pos = parse_tile_position(seg.label)
            if tile_pos:
                row, col = tile_pos
            else:
                # Fallback: estimate from segment index
                row = i // (tiles_x or 1) if tiles_x else 0
                col = i % (tiles_x or 1) if tiles_x else 0

            print(f"\n  Tile {i+1}/{num_segs} (row={row}, col={col}): crop_region=({x1},{y1},{x2},{y2}), size={crop_w}x{crop_h}")

            # Crop image
            cropped = tensor_crop(image, crop_region)

            # Calculate upscale factor - only upscale if tile is SMALLER than guide_size
            min_dim = min(crop_w, crop_h)
            if min_dim < guide_size:
                upscale = guide_size / min_dim
            else:
                upscale = 1.0  # Process at original size - no wasted computation

            # Calculate processing dimensions
            new_w = int(crop_w * upscale)
            new_h = int(crop_h * upscale)

            # Resize for processing (only if needed)
            if upscale != 1.0:
                cropped_upscaled = tensor_resize(cropped, new_w, new_h)
                print(f"    Upscaling: {upscale:.2f}x -> {new_w}x{new_h}")
            else:
                cropped_upscaled = cropped
                print(f"    Processing at original size: {crop_w}x{crop_h}")

            # Encode with VAE
            latent_samples = vae.encode(cropped_upscaled[:, :, :, :3])
            latent = {"samples": latent_samples}

            # Create noise mask from SEG mask if available
            if seg.cropped_mask is not None:
                # Resize mask to match latent dimensions
                mask = torch.from_numpy(seg.cropped_mask).float()
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0)

                # Resize mask to match upscaled image
                mask = F.interpolate(mask.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False)
                mask = mask.squeeze(0)

                # Apply feathering
                if feather > 0:
                    mask = tensor_gaussian_blur_mask(mask, feather)

                # Resize to latent size (1/8 of image)
                latent_h = new_h // 8
                latent_w = new_w // 8
                noise_mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0),
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

            # Get bbox (actual tile area) vs crop_region (expanded context area)
            bbox = seg.bbox
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
            tile_w = bbox_x2 - bbox_x1
            tile_h = bbox_y2 - bbox_y1

            # Calculate bbox position relative to crop_region
            rel_x1 = bbox_x1 - x1
            rel_y1 = bbox_y1 - y1
            rel_x2 = bbox_x2 - x1
            rel_y2 = bbox_y2 - y1

            # If we upscaled, the decoded image is larger - scale relative positions
            if upscale != 1.0:
                # Positions in the upscaled decoded image
                rel_x1_scaled = int(rel_x1 * upscale)
                rel_y1_scaled = int(rel_y1 * upscale)
                rel_x2_scaled = int(rel_x2 * upscale)
                rel_y2_scaled = int(rel_y2 * upscale)

                # Extract just the tile portion from decoded result
                tile_result = decoded[:, rel_y1_scaled:rel_y2_scaled, rel_x1_scaled:rel_x2_scaled, :]

                # Resize tile_result back to original bbox size
                tile_result = tensor_resize(tile_result, tile_w, tile_h)
            else:
                # No upscaling - extract directly
                tile_result = decoded[:, rel_y1:rel_y2, rel_x1:rel_x2, :]

                # Ensure size matches (handle any rounding issues)
                if tile_result.shape[1] != tile_h or tile_result.shape[2] != tile_w:
                    tile_result = tensor_resize(tile_result, tile_w, tile_h)

            # Determine which edges to feather based on tile position
            # Only feather edges at image boundary, not between adjacent tiles
            if tiles_x is not None and tiles_y is not None:
                feather_top = (row == 0)                    # Top row
                feather_bottom = (row == tiles_y - 1)       # Bottom row
                feather_left = (col == 0)                   # Left column
                feather_right = (col == tiles_x - 1)        # Right column
                edge_info = f"feather edges: T={feather_top}, B={feather_bottom}, L={feather_left}, R={feather_right}"
            else:
                # Fallback: no grid info, don't feather any edges (safer for seamless tiles)
                feather_top = feather_bottom = feather_left = feather_right = False
                edge_info = "no grid info - no edge feathering"

            # Create feathered mask for the tile
            if feather > 0 and (feather_top or feather_bottom or feather_left or feather_right):
                blend_mask = torch.from_numpy(create_feathered_mask(
                    tile_h, tile_w, feather,
                    feather_top=feather_top,
                    feather_bottom=feather_bottom,
                    feather_left=feather_left,
                    feather_right=feather_right
                ))
                print(f"    {edge_info}")
            else:
                blend_mask = torch.ones((tile_h, tile_w), dtype=torch.float32)
                if feather > 0:
                    print(f"    {edge_info}")

            blend_mask = blend_mask.unsqueeze(0).unsqueeze(-1)  # (1, H, W, 1)

            # Composite at bbox position (not crop_region position)
            result = composite_images(result, tile_result, blend_mask, (bbox_x1, bbox_y1))

            print(f"    Done! Composited tile at bbox ({bbox_x1},{bbox_y1}) size {tile_w}x{tile_h}")

        print(f"\n[Smart Tile Detailer v1.3] Completed! Processed {num_segs} tiles.")

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
