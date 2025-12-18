"""
USDU Edge Repair - Monkey Patches Module
=========================================

TREE MAP:
---------
usdu_patch.py
├── round_length(length, multiple=8)     - Round to nearest multiple (default 8)
│
├── PATCH 1: USDUpscaler.__init__        - Patch canvas + tile grid
│   ├── old_init                         - Original __init__ reference
│   └── new_init                         - Rounds to 8px + uses TileGeometry grid
│
├── PATCH 2: USDURedraw.init_draw        - Patch redraw tile size
│   ├── old_setup_redraw                 - Original init_draw reference
│   └── new_setup_redraw                 - New init_draw that rounds to 8px
│
├── PATCH 3: USDUSeamsFix.init_draw      - Patch seam fix tile size
│   ├── old_setup_seams_fix              - Original init_draw reference
│   └── new_setup_seams_fix              - New init_draw that rounds to 8px
│
└── PATCH 4: USDUpscaler.upscale         - Patch batch upscaling behavior
    ├── old_upscale                      - Original upscale reference
    └── new_upscale                      - New upscale that handles batch

WHY THESE PATCHES EXIST:
------------------------
The original USDU script uses multiples of 64 for all dimensions.
These patches change it to multiples of 8, which:
1. Allows finer control over tile sizes
2. Reduces wasted pixels at image edges
3. Works better with VAE (8px alignment is minimum requirement)

PATCH DETAILS:
--------------
1. USDUpscaler.__init__:
   - Rounds canvas width/height to 8px BEFORE original init
   - Ensures upscaled canvas dimensions are 8px-aligned
   - OVERRIDES rows/cols with TileGeometry values (fixes extra tile bug)
   - Without this fix: cols = ceil(canvas_w / tile_width) creates extra tiles

2. USDURedraw.init_draw:
   - Rounds processing width/height to 8px AFTER original init_draw
   - Ensures tile processing dimensions are 8px-aligned

3. USDUSeamsFix.init_draw:
   - Same as Redraw patch but for seam fixing pass
   - Ensures seam fix tile dimensions are 8px-aligned

4. USDUpscaler.upscale:
   - After upscaling first image, resize remaining batch images
   - Ensures all batch images match the canvas dimensions

DATA FLOW:
----------
When USDU runs:
1. USDUpscaler.__init__ is called -> PATCH 1 runs
2. Upscaler.upscale() is called -> PATCH 4 runs
3. USDURedraw.init_draw() is called per tile -> PATCH 2 runs
4. USDUSeamsFix.init_draw() is called per seam -> PATCH 3 runs

IMPORT NOTE:
------------
This module is imported as `usdu` by other modules:
    from .usdu_patch import usdu
This gives access to the patched ultimate_upscale module.
"""

from . import ultimate_upscale as usdu
from . import shared
import math
from PIL import Image

# Pillow compatibility: older versions don't have Resampling enum
if (not hasattr(Image, 'Resampling')):
    Image.Resampling = Image


# ============================================================
# UTILITY FUNCTION
# ============================================================

def round_length(length, multiple=8):
    """
    Round a length to the nearest multiple.

    Args:
        length: The length to round
        multiple: The multiple to round to (default: 8)

    Returns:
        int: Rounded length

    Example:
        round_length(65, 8) -> 64
        round_length(68, 8) -> 72
    """
    return round(length / multiple) * multiple


# ============================================================
# PATCH 1: USDUpscaler.__init__ - Canvas size rounding
# ============================================================
# Rounds the upscaled canvas dimensions to 8px multiples
# Original uses 64px multiples which wastes pixels at edges

old_init = usdu.USDUpscaler.__init__


def new_init(self, p, image, upscaler_index, save_redraw, save_seams_fix, tile_width, tile_height):
    """
    Patched USDUpscaler.__init__ that:
    1. Rounds canvas to 8px multiples
    2. Uses TileGeometry for tile grid (fixes extra tile problem)

    Calculates the upscaled canvas dimensions and rounds them to 8px
    BEFORE calling the original __init__. This ensures the canvas
    dimensions are 8px-aligned from the start.

    After original init, overrides rows/cols with TileGeometry values
    to prevent USDU from calculating extra tiles due to canvas padding.
    """
    # Round canvas dimensions to 8px multiples
    p.width = round_length(image.width * p.upscale_by)
    p.height = round_length(image.height * p.upscale_by)

    # Call original __init__ with rounded dimensions
    old_init(self, p, image, upscaler_index, save_redraw, save_seams_fix, tile_width, tile_height)

    # OVERRIDE: Use TileGeometry values instead of calculated values
    # This fixes the "extra tile" problem caused by canvas padding.
    # Without this, USDU calculates: cols = ceil(canvas_w / tile_width)
    # which creates extra tiles when canvas has padding.
    if hasattr(p, '_geometry') and p._geometry is not None:
        self.rows = p._geometry.tiles_y
        self.cols = p._geometry.tiles_x
        print(f"[USDU Patch] Using TileGeometry grid: {self.cols}x{self.rows} tiles")


usdu.USDUpscaler.__init__ = new_init


# ============================================================
# PATCH 2: USDURedraw.init_draw - Redraw tile size rounding
# ============================================================
# Rounds the redraw tile processing dimensions to 8px multiples
# This affects the size of each tile during the redraw pass

old_setup_redraw = usdu.USDURedraw.init_draw


def new_setup_redraw(self, p, width, height):
    """
    Patched USDURedraw.init_draw that rounds tile size to 8px.

    Calls the original init_draw to create the mask, then overrides
    p.width and p.height with 8px-aligned dimensions.
    """
    # Call original to create mask and draw objects
    mask, draw = old_setup_redraw(self, p, width, height)

    # Override processing dimensions with 8px-aligned values
    p.width = round_length(self.tile_width + self.padding)
    p.height = round_length(self.tile_height + self.padding)

    return mask, draw


usdu.USDURedraw.init_draw = new_setup_redraw


# ============================================================
# PATCH 3: USDUSeamsFix.init_draw - Seam fix tile size rounding
# ============================================================
# Same as Redraw patch but for the seam fixing pass
# Ensures seam fix tiles are also 8px-aligned

old_setup_seams_fix = usdu.USDUSeamsFix.init_draw


def new_setup_seams_fix(self, p):
    """
    Patched USDUSeamsFix.init_draw that rounds tile size to 8px.

    Calls the original init_draw, then overrides p.width and p.height
    with 8px-aligned dimensions for seam fix processing.
    """
    # Call original to set up seam fix state
    old_setup_seams_fix(self, p)

    # Override processing dimensions with 8px-aligned values
    p.width = round_length(self.tile_width + self.padding)
    p.height = round_length(self.tile_height + self.padding)


usdu.USDUSeamsFix.init_draw = new_setup_seams_fix


# ============================================================
# PATCH 4: USDUpscaler.upscale - Batch upscaling
# ============================================================
# Makes USDU handle batches of images instead of single images
# After upscaling the first image, resizes the rest of the batch

old_upscale = usdu.USDUpscaler.upscale


def new_upscale(self):
    """
    Patched USDUpscaler.upscale that handles batch images.

    After the original upscale processes the first image, this patch
    resizes all remaining images in shared.batch to match the canvas
    dimensions.

    This is needed because:
    1. Original USDU only processes one image
    2. We want to process a batch
    3. All batch images need to match the canvas size
    """
    # Upscale the first image using original method
    old_upscale(self)

    # Resize remaining batch images to match canvas dimensions
    # First image (self.image) is already correctly sized
    shared.batch = [self.image] + \
        [img.resize((self.p.width, self.p.height), resample=Image.LANCZOS) for img in shared.batch[1:]]


usdu.USDUpscaler.upscale = new_upscale
