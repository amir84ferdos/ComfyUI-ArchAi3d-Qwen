"""
USDU Edge Repair - Utility Functions Module
============================================

TREE MAP:
---------
utils.py
│
├── CONSTANTS
│   └── BLUR_KERNEL_SIZE = 15              - Kernel size for Gaussian blur
│
├── TENSOR/PIL CONVERSION
│   ├── tensor_to_pil(tensor, batch_index) - Convert tensor [B,H,W,C] to PIL
│   ├── pil_to_tensor(image)               - Convert PIL to tensor [1,H,W,C]
│   ├── controlnet_hint_to_pil(tensor)     - Convert ControlNet hint [B,C,H,W] to PIL
│   └── pil_to_controlnet_hint(img)        - Convert PIL to ControlNet hint [1,C,H,W]
│
├── TENSOR OPERATIONS
│   ├── crop_tensor(tensor, region)        - Crop tensor to region (x1,y1,x2,y2)
│   └── resize_tensor(tensor, size, mode)  - Resize tensor using interpolation
│
├── CROP REGION UTILITIES
│   ├── get_crop_region(mask, pad)         - Get bounding box of white pixels in mask
│   ├── fix_crop_region(region, size)      - Fix off-by-one in crop region
│   ├── expand_crop(region, w, h, tw, th)  - Expand crop region to target size
│   ├── resize_region(region, init, resize)- Scale crop region to new image size
│   └── region_intersection(r1, r2)        - Get intersection of two regions
│
│
├── IMAGE PADDING
│   ├── pad_image(img, l, r, t, b, fill, blur) - Pad PIL with edge fill
│   ├── pad_tensor(tensor, l, r, t, b)         - Pad tensor with edge replication
│   ├── resize_and_pad_image(img, w, h)        - Resize then pad to exact size
│   └── resize_and_pad_tensor(t, w, h)         - Resize then pad to exact size
│
├── EDGE MASK CREATION
│   ├── create_edge_mask(w, h, border, feather) - Create border mask for edge repair
│   └── create_edge_mask_tensor(...)            - Same but returns tensor
│
├── CONDITIONING CROP FUNCTIONS
│   ├── crop_controlnet(cond_dict, ...)    - Crop ControlNet hints to tile region
│   ├── crop_gligen(cond_dict, ...)        - Crop GLIGEN positions to tile region
│   ├── crop_area(cond_dict, ...)          - Crop area conditioning to tile region
│   ├── crop_mask(cond_dict, ...)          - Crop mask conditioning to tile region
│   └── crop_reference_latents(...)        - Crop Flux-Kontext reference latents
│
└── MAIN CONDITIONING CROPPER
    └── crop_cond(cond, region, ...)       - Crop all conditioning types for a tile

DATA FLOW:
----------
During tile processing:
1. get_crop_region() finds tile boundaries from mask
2. expand_crop() expands region to target tile size
3. crop_cond() crops all conditioning types to tile region:
   - Calls crop_controlnet, crop_gligen, crop_area, crop_mask
4. Tiles are processed as PIL images, converted via tensor_to_pil/pil_to_tensor

REGION FORMAT:
--------------
All regions are tuples of (x1, y1, x2, y2) where:
- (x1, y1) is the top-left corner
- (x2, y2) is the bottom-right corner (exclusive)
"""

import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import torch
import torch.nn.functional as F
# UNUSED: from torchvision.transforms import GaussianBlur
import math

# Pillow compatibility: older versions don't have Resampling enum
if (not hasattr(Image, 'Resampling')):
    Image.Resampling = Image

# ============================================================
# CONSTANTS
# ============================================================

# Kernel size for Gaussian blur when blending padded edges
BLUR_KERNEL_SIZE = 15


# ============================================================
# TENSOR/PIL CONVERSION FUNCTIONS
# ============================================================

def tensor_to_pil(img_tensor, batch_index=0):
    """
    Convert a tensor to a PIL Image.

    Args:
        img_tensor: Tensor of shape [batch_size, height, width, channels]
                   Values should be in range [0, 1]
        batch_index: Which image in the batch to convert (default: 0)

    Returns:
        PIL.Image: RGB image

    Note:
        NaN values are replaced with 0 using torch.nan_to_num()
        Values are clamped to [0, 1] to prevent uint8 overflow/underflow
    """
    safe_tensor = torch.nan_to_num(img_tensor[batch_index])
    # CLAMP to [0, 1] to prevent uint8 overflow/underflow from VAE output
    clamped = torch.clamp(safe_tensor, 0.0, 1.0)
    return Image.fromarray((255 * clamped.cpu().numpy()).astype(np.uint8))


def pil_to_tensor(image):
    """
    Convert a PIL Image to a tensor.

    Args:
        image: PIL Image (RGB or grayscale)

    Returns:
        torch.Tensor: Shape [1, height, width, channels], values in [0, 1]

    Note:
        Grayscale images get a channel dimension added.
    """
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).unsqueeze(0)
    if len(image.shape) == 3:  # If grayscale, add channel dimension
        image = image.unsqueeze(-1)
    return image


def controlnet_hint_to_pil(tensor, batch_index=0):
    """
    Convert a ControlNet hint tensor to PIL Image.

    Args:
        tensor: Shape [B, C, H, W] (ControlNet format)
        batch_index: Which image in batch (default: 0)

    Returns:
        PIL.Image: RGB image
    """
    # Move channels from dim 1 to dim -1: [B,C,H,W] -> [B,H,W,C]
    return tensor_to_pil(tensor.movedim(1, -1), batch_index)


def pil_to_controlnet_hint(img):
    """
    Convert a PIL Image to ControlNet hint tensor.

    Args:
        img: PIL Image

    Returns:
        torch.Tensor: Shape [1, C, H, W] (ControlNet format)
    """
    # Convert to [1,H,W,C] then move channels: [1,H,W,C] -> [1,C,H,W]
    return pil_to_tensor(img).movedim(-1, 1)


# ============================================================
# TENSOR OPERATIONS
# ============================================================

def crop_tensor(tensor, region):
    """
    Crop a tensor to a specified region.

    Args:
        tensor: Shape [batch_size, height, width, channels]
        region: Tuple (x1, y1, x2, y2) - crop boundaries

    Returns:
        torch.Tensor: Cropped tensor
    """
    x1, y1, x2, y2 = region
    return tensor[:, y1:y2, x1:x2, :]


def resize_tensor(tensor, size, mode="nearest-exact"):
    """
    Resize a tensor to a specified size.

    Args:
        tensor: Shape [B, C, H, W] (channels first!)
        size: Tuple (height, width) - target size
        mode: Interpolation mode (default: "nearest-exact")

    Returns:
        torch.Tensor: Resized tensor [B, C, size[0], size[1]]
    """
    return torch.nn.functional.interpolate(tensor, size=size, mode=mode)


# ============================================================
# CROP REGION UTILITIES
# ============================================================

def get_crop_region(mask, pad=0):
    """
    Get the bounding box of white pixels in a mask.

    This finds the smallest rectangle containing all white (non-zero)
    pixels in the mask, with optional padding.

    Args:
        mask: PIL Image in 'L' mode (grayscale)
        pad: Padding to add around the bounding box (default: 0)

    Returns:
        Tuple (x1, y1, x2, y2): Crop region coordinates

    Equivalent to:
        A1111's get_crop_region from modules/masking.py
    """
    coordinates = mask.getbbox()
    if coordinates is not None:
        x1, y1, x2, y2 = coordinates
    else:
        # No white pixels found - return empty region
        x1, y1, x2, y2 = mask.width, mask.height, 0, 0

    # Apply padding (clamped to image bounds)
    x1 = max(x1 - pad, 0)
    y1 = max(y1 - pad, 0)
    x2 = min(x2 + pad, mask.width)
    y2 = min(y2 + pad, mask.height)

    return fix_crop_region((x1, y1, x2, y2), (mask.width, mask.height))


def fix_crop_region(region, image_size):
    """
    Fix off-by-one error in crop region.

    PIL's getbbox() returns inclusive coordinates, but we want exclusive.
    This adjusts x2 and y2 unless they're at the image boundary.

    Args:
        region: Tuple (x1, y1, x2, y2)
        image_size: Tuple (width, height)

    Returns:
        Tuple (x1, y1, x2, y2): Fixed region
    """
    image_width, image_height = image_size
    x1, y1, x2, y2 = region
    if x2 < image_width:
        x2 -= 1
    if y2 < image_height:
        y2 -= 1
    return x1, y1, x2, y2


def expand_crop(region, width, height, target_width, target_height):
    """
    Expand a crop region to a target size, centered on the original region.

    Expands equally in both directions when possible. If the region hits
    an image boundary, the remaining expansion goes to the other side.

    Args:
        region: Tuple (x1, y1, x2, y2) - original crop region
        width: Width of the source image
        height: Height of the source image
        target_width: Desired width of expanded region
        target_height: Desired height of expanded region

    Returns:
        Tuple of:
            - (x1, y1, x2, y2): Expanded region
            - (target_width, target_height): Actual final size
    """
    x1, y1, x2, y2 = region
    actual_width = x2 - x1
    actual_height = y2 - y1

    # === Expand horizontally ===
    # Try to expand to the right by half the difference
    width_diff = target_width - actual_width
    x2 = min(x2 + width_diff // 2, width)
    # Expand left by remaining difference
    width_diff = target_width - (x2 - x1)
    x1 = max(x1 - width_diff, 0)
    # Try right again if left hit boundary
    width_diff = target_width - (x2 - x1)
    x2 = min(x2 + width_diff, width)

    # === Expand vertically ===
    # Try to expand down by half the difference
    height_diff = target_height - actual_height
    y2 = min(y2 + height_diff // 2, height)
    # Expand up by remaining difference
    height_diff = target_height - (y2 - y1)
    y1 = max(y1 - height_diff, 0)
    # Try down again if top hit boundary
    height_diff = target_height - (y2 - y1)
    y2 = min(y2 + height_diff, height)

    return (x1, y1, x2, y2), (target_width, target_height)


def resize_region(region, init_size, resize_size):
    """
    Scale a crop region to match a resized image.

    When an image is resized, this adjusts crop coordinates proportionally.

    Args:
        region: Tuple (x1, y1, x2, y2) - original coordinates
        init_size: Tuple (width, height) - original image size
        resize_size: Tuple (width, height) - new image size

    Returns:
        Tuple (x1, y1, x2, y2): Scaled coordinates
    """
    x1, y1, x2, y2 = region
    init_width, init_height = init_size
    resize_width, resize_height = resize_size

    # Scale coordinates proportionally
    # Use floor for start coords, ceil for end coords to ensure coverage
    x1 = math.floor(x1 * resize_width / init_width)
    x2 = math.ceil(x2 * resize_width / init_width)
    y1 = math.floor(y1 * resize_height / init_height)
    y2 = math.ceil(y2 * resize_height / init_height)

    return (x1, y1, x2, y2)


def region_intersection(region1, region2):
    """
    Get the intersection of two rectangular regions.

    Args:
        region1: Tuple (x1, y1, x2, y2) - first region
        region2: Tuple (x1, y1, x2, y2) - second region

    Returns:
        Tuple (x1, y1, x2, y2) if regions intersect, None otherwise
    """
    x1, y1, x2, y2 = region1
    x1_, y1_, x2_, y2_ = region2

    # Intersection is max of mins, min of maxes
    x1 = max(x1, x1_)
    y1 = max(y1, y1_)
    x2 = min(x2, x2_)
    y2 = min(y2, y2_)

    # Check if intersection exists
    if x1 >= x2 or y1 >= y2:
        return None

    return (x1, y1, x2, y2)


# ============================================================
# IMAGE PADDING FUNCTIONS
# ============================================================
# Note: Mirror padding for tile grids is now handled by TileGeometry class
# in tile_geometry.py. Use geometry.pad_image() and geometry.pad_mask() instead.

def pad_image(image, left_pad, right_pad, top_pad, bottom_pad, fill=False, blur=False):
    """
    Pad an image with edge replication.

    Fills edges with straight lines from edge pixels.

    Args:
        image: PIL Image to pad
        left_pad: Pixels to add on left
        right_pad: Pixels to add on right
        top_pad: Pixels to add on top
        bottom_pad: Pixels to add on bottom
        fill: If True, fill padding with edge data (otherwise black)
        blur: If True, blur padded areas then paste original back

    Returns:
        PIL.Image: Padded image
    """
    # Extract edge strips
    left_edge = image.crop((0, 1, 1, image.height - 1))
    right_edge = image.crop((image.width - 1, 1, image.width, image.height - 1))
    top_edge = image.crop((1, 0, image.width - 1, 1))
    bottom_edge = image.crop((1, image.height - 1, image.width - 1, image.height))

    new_width = image.width + left_pad + right_pad
    new_height = image.height + top_pad + bottom_pad
    padded_image = Image.new(image.mode, (new_width, new_height))
    padded_image.paste(image, (left_pad, top_pad))

    if fill:
        if left_pad > 0:
            padded_image.paste(left_edge.resize((left_pad, new_height), resample=Image.Resampling.NEAREST), (0, 0))
        if right_pad > 0:
            padded_image.paste(right_edge.resize((right_pad, new_height),
                               resample=Image.Resampling.NEAREST), (new_width - right_pad, 0))
        if top_pad > 0:
            padded_image.paste(top_edge.resize((new_width, top_pad), resample=Image.Resampling.NEAREST), (0, 0))
        if bottom_pad > 0:
            padded_image.paste(bottom_edge.resize((new_width, bottom_pad),
                               resample=Image.Resampling.NEAREST), (0, new_height - bottom_pad))

        # Optionally blur padding and restore original
        if blur and not (left_pad == right_pad == top_pad == bottom_pad == 0):
            padded_image = padded_image.filter(ImageFilter.GaussianBlur(BLUR_KERNEL_SIZE))
            padded_image.paste(image, (left_pad, top_pad))

    return padded_image


def pad_tensor(tensor, left_pad, right_pad, top_pad, bottom_pad):
    """
    Pad a tensor with edge replication.

    Args:
        tensor: Shape [B, C, H, W] (channels first!)
        left_pad: Pixels to add on left
        right_pad: Pixels to add on right
        top_pad: Pixels to add on top
        bottom_pad: Pixels to add on bottom

    Returns:
        torch.Tensor: Padded tensor [B, C, H+vpad, W+hpad]
    """
    batch_size, channels, height, width = tensor.shape
    h_pad = left_pad + right_pad
    v_pad = top_pad + bottom_pad
    new_width = width + h_pad
    new_height = height + v_pad

    # Create empty padded tensor
    padded = torch.zeros((batch_size, channels, new_height, new_width), dtype=tensor.dtype)

    # Copy original into center
    padded[:, :, top_pad:top_pad + height, left_pad:left_pad + width] = tensor

    # Replicate edges
    if top_pad > 0:
        padded[:, :, :top_pad, :] = padded[:, :, top_pad:top_pad + 1, :]
    if bottom_pad > 0:
        padded[:, :, -bottom_pad:, :] = padded[:, :, -bottom_pad - 1:-bottom_pad, :]
    if left_pad > 0:
        padded[:, :, :, :left_pad] = padded[:, :, :, left_pad:left_pad + 1]
    if right_pad > 0:
        padded[:, :, :, -right_pad:] = padded[:, :, :, -right_pad - 1:-right_pad]

    return padded


def resize_and_pad_image(image, width, height, fill=False, blur=False):
    """
    Resize an image maintaining aspect ratio, then pad to exact size.

    The image is scaled to fit within (width, height) while maintaining
    aspect ratio, then padded on sides to reach exact dimensions.

    Args:
        image: PIL Image
        width: Target width
        height: Target height
        fill: If True, fill padding with edge data
        blur: If True, blur padded areas

    Returns:
        Tuple of:
            - PIL.Image: Resized and padded image
            - (horizontal_pad, vertical_pad): Padding amounts used
    """
    # Calculate resize ratio to fit while maintaining aspect
    width_ratio = width / image.width
    height_ratio = height / image.height
    resize_ratio = min(width_ratio, height_ratio)

    resize_width = round(image.width * resize_ratio)
    resize_height = round(image.height * resize_ratio)
    resized = image.resize((resize_width, resize_height), resample=Image.Resampling.LANCZOS)

    # Calculate padding needed
    horizontal_pad = (width - resize_width) // 2
    vertical_pad = (height - resize_height) // 2

    # Pad and resize to exact target
    result = pad_image(resized, horizontal_pad, horizontal_pad, vertical_pad, vertical_pad, fill, blur)
    result = result.resize((width, height), resample=Image.Resampling.LANCZOS)

    return result, (horizontal_pad, vertical_pad)


def resize_and_pad_tensor(tensor, width, height):
    """
    Resize a tensor maintaining aspect ratio, then pad to exact size.

    Args:
        tensor: Shape [B, C, H, W] (channels first!)
        width: Target width
        height: Target height

    Returns:
        torch.Tensor: Resized and padded tensor [B, C, height, width]
    """
    # Calculate resize ratio to fit while maintaining aspect
    width_ratio = width / tensor.shape[3]
    height_ratio = height / tensor.shape[2]
    resize_ratio = min(width_ratio, height_ratio)

    resize_width = round(tensor.shape[3] * resize_ratio)
    resize_height = round(tensor.shape[2] * resize_ratio)
    resized = F.interpolate(tensor, size=(resize_height, resize_width), mode='nearest-exact')

    # Calculate padding needed
    horizontal_pad = (width - resize_width) // 2
    vertical_pad = (height - resize_height) // 2

    # Pad and resize to exact target
    result = pad_tensor(resized, horizontal_pad, horizontal_pad, vertical_pad, vertical_pad)
    result = F.interpolate(result, size=(height, width), mode='nearest-exact')

    return result


# ============================================================
# EDGE MASK CREATION
# ============================================================

def create_edge_mask(width, height, border_width, feather=0):
    """
    Create a border mask with feathered inner edge for edge repair.

    This creates a mask where:
    - Border area (outer ring) = WHITE (255) = less denoise
    - Center area = BLACK (0) = full denoise
    - Inner edge = feathered transition

    Used with Differential Diffusion to preserve original colors at tile edges.

    Args:
        width: Mask width in pixels
        height: Mask height in pixels
        border_width: Width of the border in pixels (the "blue zone")
        feather: Blur amount for inner edge (default: 0 = sharp edge)

    Returns:
        PIL.Image: Grayscale mask in 'L' mode
            - White (255) at borders = preserve original (less denoise)
            - Black (0) at center = full processing (more denoise)

    Example:
        # Create 576x576 mask with 20px border and 8px feather
        mask = create_edge_mask(576, 576, 20, 8)

    Visual representation (border_width=20):
        ┌──────────────────────────────┐
        │ ████████████████████████████ │ ← White border (20px)
        │ ██┌──────────────────────┐██ │
        │ ██│                      │██ │ ← Black center
        │ ██│      (full denoise)  │██ │
        │ ██│                      │██ │
        │ ██└──────────────────────┘██ │
        │ ████████████████████████████ │
        └──────────────────────────────┘
    """
    if border_width <= 0:
        # No border = all black (full denoise everywhere)
        return Image.new("L", (width, height), 0)

    # Start with white (border areas)
    mask = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(mask)

    # Draw black rectangle for center (leaves white border)
    # The inner rectangle starts at border_width and ends at (size - border_width)
    inner_x1 = border_width
    inner_y1 = border_width
    inner_x2 = width - border_width
    inner_y2 = height - border_width

    # Only draw if there's actually a center area
    if inner_x2 > inner_x1 and inner_y2 > inner_y1:
        draw.rectangle([inner_x1, inner_y1, inner_x2, inner_y2], fill=0)

    # Apply feather/blur to create soft transition at inner edge
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))

    return mask


def create_edge_mask_tensor(width, height, border_width, feather=0):
    """
    Create edge mask as a tensor for direct use in sampling.

    Same as create_edge_mask but returns a torch tensor.

    Args:
        width: Mask width in pixels
        height: Mask height in pixels
        border_width: Width of border in pixels
        feather: Blur amount for inner edge

    Returns:
        torch.Tensor: Shape [1, height, width], values 0-1
            - 1.0 at borders (preserve original)
            - 0.0 at center (full denoise)
    """
    mask_pil = create_edge_mask(width, height, border_width, feather)
    # Convert to tensor [1, H, W, 1] then squeeze to [1, H, W]
    mask_tensor = pil_to_tensor(mask_pil).squeeze(-1)
    return mask_tensor


def create_edge_mask_for_tile(width, height, border_width, feather=0,
                               has_left=True, has_right=True, has_top=True, has_bottom=True):
    """
    Create edge mask only on sides that have neighboring tiles.

    Tiles at the image border should NOT have edge mask on sides that touch
    the image edge, since there's no neighbor to blend with.

    Args:
        width: Mask width in pixels
        height: Mask height in pixels
        border_width: Width of border in pixels
        feather: Blur amount for inner edge
        has_left: True if tile has a left neighbor (not at left edge of image)
        has_right: True if tile has a right neighbor (not at right edge of image)
        has_top: True if tile has a top neighbor (not at top edge of image)
        has_bottom: True if tile has a bottom neighbor (not at bottom edge of image)

    Returns:
        PIL.Image: Grayscale mask in 'L' mode
            - White (255) on sides with neighbors = preserve original
            - Black (0) at center and edges without neighbors = full denoise

    Example for corner tile (top-left, no top/left neighbors):
        ┌──────────────────────────────┐
        │                              │ ← No border on top (no top neighbor)
        │                     ████████ │
        │                     ████████ │ ← Border on right (has right neighbor)
        │                     ████████ │
        │ ████████████████████████████ │ ← Border on bottom (has bottom neighbor)
        └──────────────────────────────┘
          ↑ No border on left (no left neighbor)
    """
    if border_width <= 0:
        return Image.new("L", (width, height), 0)

    # Start with black (full denoise)
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    # Draw white borders only on sides that have neighbors
    if has_left:
        draw.rectangle([0, 0, border_width, height], fill=255)
    if has_right:
        draw.rectangle([width - border_width, 0, width, height], fill=255)
    if has_top:
        draw.rectangle([0, 0, width, border_width], fill=255)
    if has_bottom:
        draw.rectangle([0, height - border_width, width, height], fill=255)

    # Apply feather/blur to create soft transition
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))

    return mask


# ============================================================
# CONDITIONING CROP FUNCTIONS
# ============================================================
# These functions crop various conditioning types to match a tile region.
# They all share the same signature for consistency.

def crop_controlnet(cond_dict, region, init_size, canvas_size, tile_size, w_pad, h_pad):
    """
    Crop ControlNet hints to match a tile region.

    Handles chained ControlNets (each can have a previous_controlnet).

    Args:
        cond_dict: Conditioning dictionary (modified in place)
        region: Tile region (x1, y1, x2, y2) in canvas coordinates
        init_size: Original image size before upscaling
        canvas_size: Current upscaled canvas size
        tile_size: Target tile size (width, height)
        w_pad: Horizontal padding (currently unused)
        h_pad: Vertical padding (currently unused)
    """
    if "control" not in cond_dict:
        return

    c = cond_dict["control"]
    controlnet = c.copy()
    cond_dict["control"] = controlnet

    while c is not None:
        # hint is shape (B, C, H, W)
        hint = controlnet.cond_hint_original

        # Scale crop region to match hint resolution
        resized_crop = resize_region(region, canvas_size, hint.shape[:-3:-1])

        # Crop and resize hint
        hint = crop_tensor(hint.movedim(1, -1), resized_crop).movedim(-1, 1)
        hint = resize_tensor(hint, tile_size[::-1])
        controlnet.cond_hint_original = hint

        # Process chained ControlNets
        c = c.previous_controlnet
        controlnet.set_previous_controlnet(c.copy() if c is not None else None)
        controlnet = controlnet.previous_controlnet


def crop_gligen(cond_dict, region, init_size, canvas_size, tile_size, w_pad, h_pad):
    """
    Crop GLIGEN position embeddings to match a tile region.

    GLIGEN uses position embeddings for grounded generation. This crops
    the embedding positions to match the tile region.

    Args:
        cond_dict: Conditioning dictionary (modified in place)
        region: Tile region (x1, y1, x2, y2) in canvas coordinates
        init_size: Original image size before upscaling
        canvas_size: Current upscaled canvas size
        tile_size: Target tile size (width, height)
        w_pad: Horizontal padding to add
        h_pad: Vertical padding to add
    """
    if "gligen" not in cond_dict:
        return

    type, model, cond = cond_dict["gligen"]
    if type != "position":
        from warnings import warn
        warn(f"Unknown gligen type {type}")
        return

    cropped = []
    for c in cond:
        emb, h, w, y, x = c

        # Convert latent coords to pixel coords
        x1 = x * 8
        y1 = y * 8
        x2 = x1 + w * 8
        y2 = y1 + h * 8

        # Scale to canvas size
        gligen_upscaled_box = resize_region((x1, y1, x2, y2), init_size, canvas_size)

        # Get intersection with tile region
        intersection = region_intersection(gligen_upscaled_box, region)
        if intersection is None:
            continue
        x1, y1, x2, y2 = intersection

        # Offset to tile-local coordinates
        x1 -= region[0]
        y1 -= region[1]
        x2 -= region[0]
        y2 -= region[1]

        # Add padding offset
        x1 += w_pad
        y1 += h_pad
        x2 += w_pad
        y2 += h_pad

        # Convert back to latent coords
        h = (y2 - y1) // 8
        w = (x2 - x1) // 8
        x = x1 // 8
        y = y1 // 8

        cropped.append((emb, h, w, y, x))

    cond_dict["gligen"] = (type, model, cropped)


def crop_area(cond_dict, region, init_size, canvas_size, tile_size, w_pad, h_pad):
    """
    Crop area conditioning to match a tile region.

    Area conditioning applies prompts to specific regions. This crops
    those regions to match the tile.

    Args:
        cond_dict: Conditioning dictionary (modified in place)
        region: Tile region (x1, y1, x2, y2) in canvas coordinates
        init_size: Original image size before upscaling
        canvas_size: Current upscaled canvas size
        tile_size: Target tile size (width, height)
        w_pad: Horizontal padding to add
        h_pad: Vertical padding to add
    """
    if "area" not in cond_dict:
        return

    # Get area bounds (in latent coords)
    h, w, y, x = cond_dict["area"]
    w, h, x, y = 8 * w, 8 * h, 8 * x, 8 * y

    # Scale to canvas size
    x1, y1, x2, y2 = resize_region((x, y, x + w, y + h), init_size, canvas_size)

    # Get intersection with tile region
    intersection = region_intersection((x1, y1, x2, y2), region)
    if intersection is None:
        # Area doesn't intersect tile - remove it
        del cond_dict["area"]
        del cond_dict["strength"]
        return
    x1, y1, x2, y2 = intersection

    # Offset to tile-local coordinates
    x1 -= region[0]
    y1 -= region[1]
    x2 -= region[0]
    y2 -= region[1]

    # Add padding offset
    x1 += w_pad
    y1 += h_pad
    x2 += w_pad
    y2 += h_pad

    # Convert back to latent coords
    w, h = (x2 - x1) // 8, (y2 - y1) // 8
    x, y = x1 // 8, y1 // 8

    cond_dict["area"] = (h, w, y, x)


def crop_mask(cond_dict, region, init_size, canvas_size, tile_size, w_pad, h_pad):
    """
    Crop mask conditioning to match a tile region.

    Masks control where conditioning is applied. This crops and resizes
    the mask to match the tile.

    Args:
        cond_dict: Conditioning dictionary (modified in place)
        region: Tile region (x1, y1, x2, y2) in canvas coordinates
        init_size: Original image size before upscaling
        canvas_size: Current upscaled canvas size
        tile_size: Target tile size (width, height)
        w_pad: Horizontal padding (currently unused)
        h_pad: Vertical padding (currently unused)
    """
    if "mask" not in cond_dict:
        return

    mask_tensor = cond_dict["mask"]  # (B, H, W)
    masks = []

    for i in range(mask_tensor.shape[0]):
        # Convert to PIL
        mask = tensor_to_pil(mask_tensor, i)

        # Resize to canvas size
        mask = mask.resize(canvas_size, Image.Resampling.BICUBIC)

        # Crop to tile region
        mask = mask.crop(region)

        # Resize and pad to tile size
        mask, _ = resize_and_pad_image(mask, tile_size[0], tile_size[1], fill=True)

        # Ensure exact tile size
        if tile_size != mask.size:
            mask = mask.resize(tile_size, Image.Resampling.BICUBIC)

        # Convert back to tensor
        mask = pil_to_tensor(mask)  # (1, H, W, 1)
        mask = mask.squeeze(-1)     # (1, H, W)
        masks.append(mask)

    cond_dict["mask"] = torch.cat(masks, dim=0)  # (B, H, W)


def crop_reference_latents(cond_dict, region, init_size, canvas_size, tile_size, w_pad, h_pad):
    """
    Crop Flux-Kontext reference latents to match a tile region.

    Added for Flux-Kontext support by TBG ETUR.

    This crops reference latents (used for image conditioning) to match
    the tile region. Works in latent space (8x downsampled from pixels).

    Args:
        cond_dict: Conditioning dictionary (modified in place)
        region: Tile region (x1, y1, x2, y2) in pixel coordinates
        init_size: Original image size before upscaling
        canvas_size: Current upscaled canvas size (pixels)
        tile_size: Target tile size (width, height) in pixels
        w_pad: Horizontal padding (currently unused)
        h_pad: Vertical padding (currently unused)

    Expects:
        "reference_latents" key with list of BCHW tensors
    """
    latents = cond_dict.get("reference_latents")
    if not isinstance(latents, list):
        return  # Nothing to do

    k = 8  # Pixel to latent downsample factor

    # Convert sizes to latent units
    W_can_px, H_can_px = canvas_size
    W_can_lat, H_can_lat = W_can_px // k, H_can_px // k

    W_tile_px, H_tile_px = tile_size
    W_tile_lat, H_tile_lat = max(1, W_tile_px // k), max(1, H_tile_px // k)

    x1_px, y1_px, x2_px, y2_px = region

    new_latents = []
    for t in latents:  # (B,C,H_lat_in,W_lat_in)
        has_5d = False
        if t.ndim == 5:  # (B,C,1,H_lat_in,W_lat_in)
            has_5d = True
            t = t.squeeze(2)
        if t.ndim != 4:
            raise ValueError(f"expected BCHW, got {t.shape}")

        # 1. Resize to canvas resolution in latent units
        if t.shape[-2:] != (H_can_lat, W_can_lat):
            t = F.interpolate(t, size=(H_can_lat, W_can_lat), mode="bilinear", align_corners=False)

        # 2. Convert pixel crop to latent slice
        w0_lat = int(round(x1_px / k))
        w1_lat = int(round(x2_px / k))
        h0_lat = int(round(y1_px / k))
        h1_lat = int(round(y2_px / k))

        cropped = t[:, :, h0_lat:h1_lat, w0_lat:w1_lat]

        # 3. Resize to latent-tile size
        cropped = F.interpolate(cropped, size=(H_tile_lat, W_tile_lat), mode="bilinear", align_corners=False)

        if has_5d:
            cropped = cropped.unsqueeze(2)
        new_latents.append(cropped)

    cond_dict["reference_latents"] = new_latents


# ============================================================
# MAIN CONDITIONING CROPPER
# ============================================================

def crop_cond(cond, region, init_size, canvas_size, tile_size, w_pad=0, h_pad=0):
    """
    Crop all conditioning types to match a tile region.

    This is the main entry point for conditioning cropping. It processes
    each conditioning entry and crops all supported types.

    Args:
        cond: List of (embedding, dict) tuples - the conditioning
        region: Tile region (x1, y1, x2, y2) in canvas coordinates
        init_size: Original image size before upscaling
        canvas_size: Current upscaled canvas size
        tile_size: Target tile size (width, height)
        w_pad: Horizontal padding (default: 0)
        h_pad: Vertical padding (default: 0)

    Returns:
        List of (embedding, dict) tuples with cropped conditioning
    """
    cropped = []

    for emb, x in cond:
        # Copy the conditioning dict to avoid modifying original
        cond_dict = x.copy()
        n = [emb, cond_dict]

        # Crop all supported conditioning types
        crop_controlnet(cond_dict, region, init_size, canvas_size, tile_size, w_pad, h_pad)
        crop_gligen(cond_dict, region, init_size, canvas_size, tile_size, w_pad, h_pad)
        crop_area(cond_dict, region, init_size, canvas_size, tile_size, w_pad, h_pad)
        crop_mask(cond_dict, region, init_size, canvas_size, tile_size, w_pad, h_pad)
        crop_reference_latents(cond_dict, region, init_size, canvas_size, tile_size, w_pad, h_pad)

        cropped.append(n)

    return cropped
