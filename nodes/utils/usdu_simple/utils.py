"""
Simple USDU - Utility Functions
================================

Core helper functions for tensor/PIL conversion and geometry.
"""

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# Pillow compatibility: older versions don't have Resampling enum
if not hasattr(Image, 'Resampling'):
    Image.Resampling = Image


def tensor_to_pil(img_tensor, batch_index=0):
    """
    Convert a tensor to a PIL Image.

    Args:
        img_tensor: Tensor of shape [batch_size, height, width, channels]
                   Values should be in range [0, 1]
        batch_index: Which image in the batch to convert (default: 0)

    Returns:
        PIL.Image: RGB image
    """
    safe_tensor = torch.nan_to_num(img_tensor[batch_index])
    clamped = torch.clamp(safe_tensor, 0.0, 1.0)
    return Image.fromarray((255 * clamped.cpu().numpy()).astype(np.uint8))


def pil_to_tensor(image):
    """
    Convert a PIL Image to a tensor.

    Args:
        image: PIL Image (RGB or grayscale)

    Returns:
        torch.Tensor: Shape [1, height, width, channels], values in [0, 1]
    """
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).unsqueeze(0)
    if len(image.shape) == 3:  # If grayscale, add channel dimension
        image = image.unsqueeze(-1)
    return image


def get_tile_position(tile_idx, tiles_x, tiles_y):
    """
    Convert tile index to grid coordinates (xi, yi).

    Args:
        tile_idx: Index of tile (0 to total_tiles-1)
        tiles_x: Number of tiles horizontally
        tiles_y: Number of tiles vertically (unused, for clarity)

    Returns:
        Tuple (xi, yi): Grid coordinates
    """
    xi = tile_idx % tiles_x
    yi = tile_idx // tiles_x
    return xi, yi


def get_tile_rect(tile_idx, tile_width, tile_height, tiles_x, tiles_y, overlap):
    """
    Get the NON-OVERLAPPING tile rectangle (for compositing).

    This is the actual output region - where the tile's center contributes.
    The layout uses overlapping tiles but non-overlapping output regions.

    Args:
        tile_idx: Index of tile (0-indexed)
        tile_width: Width of each tile (with overlap)
        tile_height: Height of each tile (with overlap)
        tiles_x: Number of tiles horizontally
        tiles_y: Number of tiles vertically
        overlap: Overlap between tiles in pixels

    Returns:
        Tuple (x1, y1, x2, y2): Tile output rectangle
    """
    xi, yi = get_tile_position(tile_idx, tiles_x, tiles_y)

    # Step between tile starts = tile_size - overlap
    step_x = tile_width - overlap
    step_y = tile_height - overlap

    # Tile start position
    x1 = xi * step_x
    y1 = yi * step_y

    # Tile end = start + tile_size (not step)
    # But for OUTPUT rect, we use step to avoid overlap
    x2 = x1 + step_x
    y2 = y1 + step_y

    # Last tile extends to its full size
    if xi == tiles_x - 1:
        x2 = x1 + tile_width
    if yi == tiles_y - 1:
        y2 = y1 + tile_height

    return (x1, y1, x2, y2)


def get_padded_rect(tile_idx, tile_width, tile_height, tiles_x, tiles_y, overlap,
                    image_width, image_height):
    """
    Get the padded crop rectangle for a tile (for cropping from image).

    This includes the overlap/padding context around the tile.

    Args:
        tile_idx: Index of tile (0-indexed)
        tile_width: Width of each tile
        tile_height: Height of each tile
        tiles_x: Number of tiles horizontally
        tiles_y: Number of tiles vertically
        overlap: Overlap between tiles in pixels
        image_width: Width of source image
        image_height: Height of source image

    Returns:
        Tuple (x1, y1, x2, y2): Padded crop rectangle (clamped to image bounds)
    """
    xi, yi = get_tile_position(tile_idx, tiles_x, tiles_y)

    # Step between tile starts = tile_size - overlap
    step_x = tile_width - overlap
    step_y = tile_height - overlap

    # Tile start position
    x1 = xi * step_x
    y1 = yi * step_y

    # The crop region is the full tile size
    x2 = x1 + tile_width
    y2 = y1 + tile_height

    # Clamp to image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_width, x2)
    y2 = min(image_height, y2)

    return (x1, y1, x2, y2)


def resize_region(region, from_size, to_size):
    """
    Scale a crop region to match a resized image.

    Args:
        region: Tuple (x1, y1, x2, y2) - original coordinates
        from_size: Tuple (width, height) - original image size
        to_size: Tuple (width, height) - new image size

    Returns:
        Tuple (x1, y1, x2, y2): Scaled coordinates
    """
    x1, y1, x2, y2 = region
    from_w, from_h = from_size
    to_w, to_h = to_size

    scale_x = to_w / from_w
    scale_y = to_h / from_h

    return (
        int(x1 * scale_x),
        int(y1 * scale_y),
        int(x2 * scale_x),
        int(y2 * scale_y)
    )
