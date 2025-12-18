"""
TileGeometry - Single source of truth for all tile calculations.

All padding, masking, and geometry operations in ONE place.
Used by both preview.py and processing.py to ensure consistency.

Note: Mirror padding has been removed. Edge tiles now have less
context (clamped to image bounds), matching original USDU behavior.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


class TileGeometry:
    """
    Unified tile geometry calculator.

    Calculates ALL geometry ONCE at initialization:
    - Canvas size (with padding margin for edge tiles)
    - All tile rectangles
    - All padded crop regions
    - All blend masks

    Provides methods for:
    - Padding images/masks to canvas size
    - Getting tiles with correct padding
    - Creating blend masks
    - Compositing tiles back
    - Cropping final result

    Example usage:
        geometry = TileGeometry(
            original_size=(1024, 1024),
            tile_width=512, tile_height=512,
            tiles_x=2, tiles_y=2,
            tile_padding=64, mask_blur=8
        )

        # Pad image to canvas size
        padded_image = geometry.pad_image(image_tensor)

        # Get tile crop with padding context
        tile = geometry.get_tile_crop(pil_image, tile_idx=0)

        # Get blend mask for compositing
        mask = geometry.get_blend_mask(tile_idx=0)

        # Crop final result back to original size
        output = geometry.crop_to_original(result_tensor)
    """

    def __init__(self, original_size, tile_width, tile_height,
                 tiles_x, tiles_y, tile_padding, mask_blur):
        """
        Initialize geometry calculations.

        Args:
            original_size: (height, width) of input image
            tile_width: Width of each tile in pixels
            tile_height: Height of each tile in pixels
            tiles_x: Number of tiles horizontally
            tiles_y: Number of tiles vertically
            tile_padding: Context padding around each tile in pixels
            mask_blur: Gaussian blur radius for blend masks
        """
        self.original_h, self.original_w = original_size
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.tiles_x = tiles_x
        self.tiles_y = tiles_y
        self.tile_padding = tile_padding
        self.mask_blur = mask_blur

        # Canvas size = original image size (no mirror padding)
        # Edge tiles will have less context (clamped to image bounds)
        # This matches original USDU behavior
        self.canvas_w = self.original_w
        self.canvas_h = self.original_h

        # No canvas padding (mirror padding removed)
        self.pad_right = 0
        self.pad_bottom = 0

        # Pre-calculate all tile data
        self._tile_rects = []      # (x1, y1, x2, y2) for each tile
        self._padded_rects = []    # Padded crop regions
        self._blend_masks = None   # Lazy-loaded PIL masks for blending
        self._calc_all_tiles()

    @property
    def total_tiles(self):
        """Total number of tiles in grid."""
        return self.tiles_x * self.tiles_y

    @property
    def canvas_size(self):
        """Canvas size as (width, height)."""
        return (self.canvas_w, self.canvas_h)

    @property
    def original_size(self):
        """Original image size as (height, width)."""
        return (self.original_h, self.original_w)

    @property
    def padded_tile_size(self):
        """Size of tile WITH padding context as (width, height)."""
        return (self.tile_width + 2 * self.tile_padding,
                self.tile_height + 2 * self.tile_padding)

    def _calc_all_tiles(self):
        """Pre-calculate all tile rectangles."""
        for yi in range(self.tiles_y):
            for xi in range(self.tiles_x):
                # Tile rectangle (non-overlapping grid starting at 0,0)
                x1 = xi * self.tile_width
                y1 = yi * self.tile_height
                x2 = x1 + self.tile_width
                y2 = y1 + self.tile_height
                tile_rect = (x1, y1, x2, y2)
                self._tile_rects.append(tile_rect)

                # Padded crop region (clamped to canvas)
                pad_x1 = max(0, x1 - self.tile_padding)
                pad_y1 = max(0, y1 - self.tile_padding)
                pad_x2 = min(self.canvas_w, x2 + self.tile_padding)
                pad_y2 = min(self.canvas_h, y2 + self.tile_padding)
                self._padded_rects.append((pad_x1, pad_y1, pad_x2, pad_y2))

    def _ensure_blend_masks(self):
        """Lazy-load blend masks (created on first access)."""
        if self._blend_masks is None:
            self._blend_masks = []
            for tile_idx in range(len(self._tile_rects)):
                mask = self._create_blend_mask(tile_idx)
                self._blend_masks.append(mask)

    def _create_blend_mask(self, tile_idx):
        """
        Create blend mask for a tile with consistent border thickness.

        Uses DISTANCE-BASED masking for uniform thickness on ALL edges including corners:
        - White (255) in center = full contribution from this tile
        - Gradient (0→255) at edges with neighbors = blending zone
        - No gradient on edges touching image boundary (stays at 255)

        Each pixel's value = distance to nearest INTERNAL edge / blur_width.
        This gives uniform thickness everywhere, unlike np.minimum() which
        creates diagonal dark corners.
        """
        xi, yi = self.get_tile_coords(tile_idx)
        x1, y1, x2, y2 = self._tile_rects[tile_idx]
        tile_h = y2 - y1
        tile_w = x2 - x1

        border = self.mask_blur
        if border <= 0:
            # No blur - just white tile on black canvas
            result = np.zeros((self.canvas_h, self.canvas_w), dtype=np.float32)
            result[y1:y2, x1:x2] = 1.0
            return Image.fromarray((result * 255).astype(np.uint8))

        # Create coordinate grids for the tile area (vectorized, no slow loops)
        ys = np.arange(tile_h).reshape(-1, 1)  # Column vector [tile_h, 1]
        xs = np.arange(tile_w).reshape(1, -1)  # Row vector [1, tile_w]

        # Start with maximum possible distance (all white = 1.0)
        min_dist = np.full((tile_h, tile_w), float('inf'), dtype=np.float32)

        # Calculate distance to each INTERNAL edge (only edges with neighbors)
        # Distance is measured in pixels from the edge

        if xi > 0:  # Left edge has neighbor
            min_dist = np.minimum(min_dist, xs)  # Distance from left edge

        if xi < self.tiles_x - 1:  # Right edge has neighbor
            min_dist = np.minimum(min_dist, tile_w - 1 - xs)  # Distance from right edge

        if yi > 0:  # Top edge has neighbor
            min_dist = np.minimum(min_dist, ys)  # Distance from top edge

        if yi < self.tiles_y - 1:  # Bottom edge has neighbor
            min_dist = np.minimum(min_dist, tile_h - 1 - ys)  # Distance from bottom edge

        # Convert distance to gradient value:
        # - At edge (distance=0): value=0 (black)
        # - At distance>=border: value=1 (white)
        # - If no internal edges (inf distance): value=1 (white)
        tile_mask = np.clip(min_dist / border, 0.0, 1.0)

        # Place tile mask on canvas
        result = np.zeros((self.canvas_h, self.canvas_w), dtype=np.float32)
        result[y1:y2, x1:x2] = tile_mask

        return Image.fromarray((result * 255).astype(np.uint8))

    # =========================================================================
    # PADDING METHODS
    # =========================================================================

    def pad_image(self, image_tensor):
        """
        No-op: Mirror padding removed.

        Previously mirror-padded image to canvas size, but this was only
        needed for debug output consistency. Original USDU doesn't use
        mirror padding - edge tiles just have less context.

        Args:
            image_tensor: [B, H, W, C] tensor (ComfyUI IMAGE format)

        Returns:
            Input tensor unchanged
        """
        return image_tensor

    def pad_mask(self, mask_tensor):
        """
        No-op: Mirror padding removed.

        Previously mirror-padded mask to canvas size, but this was only
        needed for debug output consistency. Original USDU doesn't use
        mirror padding - edge tiles just have less context.

        Args:
            mask_tensor: [H, W] or [B, H, W] tensor

        Returns:
            Input tensor unchanged
        """
        return mask_tensor

    # =========================================================================
    # TILE ACCESS METHODS
    # =========================================================================

    def get_tile_index(self, xi, yi):
        """Convert grid coordinates to tile index."""
        return yi * self.tiles_x + xi

    def get_tile_coords(self, tile_idx):
        """Convert tile index to grid coordinates (xi, yi)."""
        xi = tile_idx % self.tiles_x
        yi = tile_idx // self.tiles_x
        return xi, yi

    def get_tile_rect(self, tile_idx):
        """
        Get tile rectangle (non-overlapping grid position).

        Args:
            tile_idx: Index of tile (0 to total_tiles-1)

        Returns:
            Tuple (x1, y1, x2, y2) - tile boundaries
        """
        return self._tile_rects[tile_idx]

    def get_padded_rect(self, tile_idx):
        """
        Get padded crop region for tile.

        This is the region that includes context padding around the tile.

        Args:
            tile_idx: Index of tile

        Returns:
            Tuple (x1, y1, x2, y2) - padded crop boundaries
        """
        return self._padded_rects[tile_idx]

    def get_blend_mask(self, tile_idx):
        """
        Get blend mask for tile (full canvas size).

        Args:
            tile_idx: Index of tile

        Returns:
            PIL Image (mode 'L') with blurred tile rectangle
        """
        self._ensure_blend_masks()
        return self._blend_masks[tile_idx]

    def get_blend_mask_np(self, tile_idx):
        """
        Get blend mask as normalized numpy array.

        Args:
            tile_idx: Index of tile

        Returns:
            numpy array [H, W] with values 0.0 to 1.0
        """
        mask = self.get_blend_mask(tile_idx)
        return np.array(mask).astype(np.float32) / 255.0

    def get_tile_crop(self, pil_image, tile_idx):
        """
        Crop tile from image with padding context (clamped at edges).

        Mirror padding has been removed. Border tiles will be SMALLER
        than center tiles because they can't extend past image bounds.
        This matches original USDU behavior.

        Args:
            pil_image: PIL Image at original size
            tile_idx: Index of tile

        Returns:
            PIL Image - size varies based on tile position
            (border tiles are smaller, center tiles are padded_tile_size)
        """
        rect = self._padded_rects[tile_idx]
        return pil_image.crop(rect)

    def get_tile_original(self, pil_image, tile_idx):
        """
        Get original tile rectangle (no padding).

        Args:
            pil_image: PIL Image (should be padded to canvas size)
            tile_idx: Index of tile

        Returns:
            PIL Image at (tile_width, tile_height)
        """
        rect = self._tile_rects[tile_idx]
        return pil_image.crop(rect)

    # =========================================================================
    # EDGE MASK FOR DIFFDIFF
    # =========================================================================

    def create_edge_mask(self, tile_idx, edge_width, edge_feather):
        """
        Create edge mask for DiffDiff.

        - White center (255) = MORE denoising = regenerate tile content
        - Black borders (0) = LESS denoising = preserve original for blending

        Args:
            tile_idx: Index of tile
            edge_width: Width of edge border in pixels
            edge_feather: Gaussian blur amount for feathering

        Returns:
            PIL Image (mode 'L') at padded_tile_size
        """
        xi, yi = self.get_tile_coords(tile_idx)

        w, h = self.padded_tile_size
        # Start with white (full denoise)
        mask = Image.new('L', (w, h), 255)
        draw = ImageDraw.Draw(mask)

        # Draw BLACK edges where tiles have neighbors (preserve for blending)
        if xi > 0:  # Has left neighbor
            draw.rectangle((0, 0, edge_width, h), fill=0)
        if xi < self.tiles_x - 1:  # Has right neighbor
            draw.rectangle((w - edge_width, 0, w, h), fill=0)
        if yi > 0:  # Has top neighbor
            draw.rectangle((0, 0, w, edge_width), fill=0)
        if yi < self.tiles_y - 1:  # Has bottom neighbor
            draw.rectangle((0, h - edge_width, w, h), fill=0)

        if edge_feather > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(edge_feather))

        return mask

    # =========================================================================
    # FINAL OUTPUT
    # =========================================================================

    def crop_to_original(self, tensor):
        """
        Crop tensor back to original size (removes right/bottom padding).

        Args:
            tensor: [B, H, W, C] tensor at canvas size

        Returns:
            [B, original_h, original_w, C] tensor
        """
        # Original content starts at (0, 0), just crop to original size
        return tensor[:, :self.original_h, :self.original_w, :]

    # =========================================================================
    # PREVIEW HELPERS
    # =========================================================================

    def get_all_tile_crops(self, pil_image):
        """Get all tiles as a list of PIL Images."""
        return [self.get_tile_crop(pil_image, i) for i in range(self.total_tiles)]

    def get_all_tile_originals(self, pil_image):
        """Get all original tile rectangles as a list of PIL Images."""
        return [self.get_tile_original(pil_image, i) for i in range(self.total_tiles)]

    def get_all_blend_masks(self):
        """Get all blend masks as a list of PIL Images."""
        self._ensure_blend_masks()
        return self._blend_masks.copy()

    def get_all_edge_masks(self, edge_width, edge_feather):
        """Get all edge masks as a list of PIL Images."""
        return [self.create_edge_mask(i, edge_width, edge_feather)
                for i in range(self.total_tiles)]

    # =========================================================================
    # DEBUG INFO
    # =========================================================================

    def get_debug_info(self):
        """Get debug information string."""
        lines = [
            f"TileGeometry:",
            f"  Original: {self.original_w}x{self.original_h}",
            f"  Canvas: {self.canvas_w}x{self.canvas_h}",
            f"  Grid: {self.tiles_x}x{self.tiles_y} = {self.total_tiles} tiles",
            f"  Tile size: {self.tile_width}x{self.tile_height}",
            f"  Tile padding: {self.tile_padding}px context",
            f"  Padded tile: {self.padded_tile_size[0]}x{self.padded_tile_size[1]} (max size)",
            f"  Mask blur: {self.mask_blur}",
            f"  Note: Edge tiles have less context (clamped to image bounds)",
        ]
        return "\n".join(lines)

    def __repr__(self):
        return (f"TileGeometry(original={self.original_w}x{self.original_h}, "
                f"canvas={self.canvas_w}x{self.canvas_h}, "
                f"grid={self.tiles_x}x{self.tiles_y}, "
                f"tile={self.tile_width}x{self.tile_height}, "
                f"padding={self.tile_padding})")
