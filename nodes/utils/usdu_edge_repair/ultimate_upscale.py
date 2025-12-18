"""
USDU Edge Repair - Ultimate SD Upscale Core Module
===================================================

This is the core upscaling logic from Ultimate SD Upscale (USDU).
It handles tile-based image processing with configurable redraw and seam fixing.

TREE MAP:
---------
ultimate_upscale.py
│
├── USDUpscaler (class)                  - Main orchestrator for upscaling
│   ├── __init__(p, image, ...)          - Initialize upscaler with settings
│   ├── get_factor(num)                  - Get optimal scale factor (2, 3, or 4)
│   ├── get_factors()                    - Build list of scale factors
│   ├── upscale()                        - Perform the image upscaling
│   ├── setup_redraw(mode, padding, blur)- Configure redraw settings
│   ├── setup_seams_fix(...)             - Configure seam fix settings
│   ├── save_image()                     - Save image (stub for ComfyUI)
│   ├── calc_jobs_count()                - Calculate total job count
│   ├── print_info()                     - Print debug info
│   ├── add_extra_info()                 - Add metadata to p.extra_generation_params
│   └── process()                        - Run redraw and seam fix passes
│
├── USDURedraw (class)                   - Tile redraw processor
│   ├── init_draw(p, width, height)      - Initialize mask for redraw
│   ├── calc_rectangle(xi, yi)           - Calculate tile rectangle coords
│   ├── linear_process(p, image, r, c)   - Process tiles row-by-row
│   ├── chess_process(p, image, r, c)    - Process tiles in checkerboard pattern
│   └── start(p, image, rows, cols)      - Start redraw processing
│
├── USDUSeamsFix (class)                 - Seam repair processor
│   ├── init_draw(p)                     - Initialize for seam fixing
│   ├── half_tile_process(...)           - Fix seams with half-tile overlap
│   ├── half_tile_process_corners(...)   - Fix seams + corner intersections
│   ├── band_pass_process(...)           - Fix seams with band pass method
│   └── start(p, image, rows, cols)      - Start seam fix processing
│
└── Script (class)                       - Entry point (A1111 compatibility)
    ├── title()                          - Return script title
    ├── show(is_img2img)                 - Show script in UI
    └── run(p, ...)                       - Main execution entry point

PROCESSING FLOW:
----------------
1. Script.run() is called with all parameters
2. USDUpscaler is created with the image and settings
3. upscaler.upscale() scales the image to target size
4. upscaler.setup_redraw() configures tile redrawing
5. upscaler.setup_seams_fix() configures seam repair
6. upscaler.process() runs:
   a. USDURedraw.start() processes all tiles (LINEAR or CHESS mode)
   b. USDUSeamsFix.start() fixes seams between tiles

REDRAW MODES:
-------------
- LINEAR: Process tiles row by row, left to right, top to bottom
- CHESS: Process tiles in checkerboard pattern (white squares, then black)
- NONE: Skip redraw pass

SEAM FIX MODES:
---------------
- NONE: Skip seam fixing
- BAND_PASS: Process seam lines as bands
- HALF_TILE: Overlap tiles at seams
- HALF_TILE_PLUS_INTERSECTIONS: Half tile + corner intersection repair

IMPORTANT MODIFICATIONS:
------------------------
This version has been modified from the original USDU to support:
- Per-tile conditioning (advance_tile() / reset_tile_index())
- ComfyUI integration (no A1111 save functions)
- Enums imported from processing.py (avoid duplication)
"""

import math
from PIL import Image, ImageDraw, ImageOps
from . import processing
from .processing import StableDiffusionProcessing
from .processing import Processed
from .processing import USDUMode, USDUSFMode
from .shared import opts, state
from . import shared


class USDUpscaler():
    """
    Main orchestrator for Ultimate SD Upscale operations.

    Coordinates the upscaling, tile redrawing, and seam fixing processes.
    Creates USDURedraw and USDUSeamsFix instances to handle those passes.

    Attributes:
        p: StableDiffusionProcessing instance with sampling params
        image: Current PIL Image being processed
        scale_factor: How much to upscale (e.g., 2 for 2x)
        upscaler: UpscalerData instance for neural upscaling
        redraw: USDURedraw instance for tile processing
        seams_fix: USDUSeamsFix instance for seam repair
        rows, cols: Grid dimensions (number of tiles)
        result_images: List of output images
    """

    def __init__(self, p, image, upscaler_index: int, save_redraw, save_seams_fix, tile_width, tile_height) -> None:
        """
        Initialize the upscaler with all settings.

        Args:
            p: StableDiffusionProcessing instance
            image: Input PIL Image
            upscaler_index: Index into shared.sd_upscalers (usually 0)
            save_redraw: Whether to save after redraw (unused in ComfyUI)
            save_seams_fix: Whether to save after seam fix (unused in ComfyUI)
            tile_width: Width of each tile in pixels
            tile_height: Height of each tile in pixels
        """
        self.p: StableDiffusionProcessing = p
        self.image: Image = image

        # Calculate scale factor needed to reach target size
        self.scale_factor = math.ceil(max(p.width, p.height) / max(image.width, image.height))

        # Get upscaler from shared state
        self.upscaler = shared.sd_upscalers[upscaler_index]

        # Initialize redraw processor
        self.redraw = USDURedraw()
        self.redraw.save = save_redraw
        self.redraw.tile_width = tile_width if tile_width > 0 else tile_height
        self.redraw.tile_height = tile_height if tile_height > 0 else tile_width

        # Initialize seam fix processor
        self.seams_fix = USDUSeamsFix()
        self.seams_fix.save = save_seams_fix
        self.seams_fix.tile_width = tile_width if tile_width > 0 else tile_height
        self.seams_fix.tile_height = tile_height if tile_height > 0 else tile_width

        self.initial_info = None

        # Calculate grid dimensions
        self.rows = math.ceil(self.p.height / self.redraw.tile_height)
        self.cols = math.ceil(self.p.width / self.redraw.tile_width)

    def get_factor(self, num):
        """
        Get an optimal scale factor for the given number.

        Prefers larger factors (4 > 3 > 2) when possible.

        Args:
            num: Number to factor

        Returns:
            int: 4, 3, 2, or 0 (0 means no good factor)
        """
        if num == 1:
            return 2
        if num % 4 == 0:
            return 4
        if num % 3 == 0:
            return 3
        if num % 2 == 0:
            return 2
        return 0

    def get_factors(self):
        """
        Build a list of scale factors to reach the target scale.

        For example, scale_factor=8 might produce [4, 2] (4x then 2x = 8x).
        Stores result in self.scales as enumerate iterator.
        """
        scales = []
        current_scale = 1
        current_scale_factor = self.get_factor(self.scale_factor)

        # If no good factor, increment until we find one
        while current_scale_factor == 0:
            self.scale_factor += 1
            current_scale_factor = self.get_factor(self.scale_factor)

        # Build list of factors
        while current_scale < self.scale_factor:
            current_scale_factor = self.get_factor(self.scale_factor // current_scale)
            scales.append(current_scale_factor)
            current_scale = current_scale * current_scale_factor
            if current_scale_factor == 0:
                break

        self.scales = enumerate(scales)

    def upscale(self):
        """
        Upscale the image to target dimensions.

        Uses the neural upscaler model if available, otherwise LANCZOS.
        Applies scale factors iteratively (e.g., 4x then 2x for 8x total).
        """
        # Log info
        print(f"Canvas size: {self.p.width}x{self.p.height}")
        print(f"Image size: {self.image.width}x{self.image.height}")
        print(f"Scale factor: {self.scale_factor}")

        # Check if upscaler is disabled
        if self.upscaler.name == "None":
            self.image = self.image.resize((self.p.width, self.p.height), resample=Image.LANCZOS)
            return

        # Get list of scale factors
        self.get_factors()

        # Apply each scale factor
        for index, value in self.scales:
            print(f"Upscaling iteration {index + 1} with scale factor {value}")
            self.image = self.upscaler.scaler.upscale(self.image, value, self.upscaler.data_path)

        # Final resize to exact target dimensions
        self.image = self.image.resize((self.p.width, self.p.height), resample=Image.LANCZOS)

    def setup_redraw(self, redraw_mode, padding, mask_blur):
        """
        Configure the redraw pass settings.

        Args:
            redraw_mode: USDUMode value (LINEAR, CHESS, or NONE)
            padding: Pixels to add around each tile for context
            mask_blur: Blur amount for tile edges (smoother blending)
        """
        self.redraw.mode = USDUMode(redraw_mode)
        self.redraw.enabled = self.redraw.mode != USDUMode.NONE
        self.redraw.padding = padding
        self.p.mask_blur = mask_blur

    def setup_seams_fix(self, padding, denoise, mask_blur, width, mode):
        """
        Configure the seam fix pass settings.

        Args:
            padding: Pixels of context around seam region
            denoise: Denoise strength for seam fix (usually high, ~1.0)
            mask_blur: Blur amount for seam mask edges
            width: Width of the seam band
            mode: USDUSFMode value (NONE, BAND_PASS, HALF_TILE, etc.)
        """
        self.seams_fix.padding = padding
        self.seams_fix.denoise = denoise
        self.seams_fix.mask_blur = mask_blur
        self.seams_fix.width = width
        self.seams_fix.mode = USDUSFMode(mode)
        self.seams_fix.enabled = self.seams_fix.mode != USDUSFMode.NONE

    def save_image(self):
        """
        Save the current image.

        NOTE: This is a stub for ComfyUI compatibility.
        Original A1111 code saved images here; ComfyUI handles output differently.
        """
        if type(self.p.prompt) != list:
            pass  # images.save_image removed for ComfyUI
        else:
            pass

    def calc_jobs_count(self):
        """
        Calculate the total number of jobs (tile operations).

        Updates state.job_count for progress tracking.
        """
        redraw_job_count = (self.rows * self.cols) if self.redraw.enabled else 0
        seams_job_count = 0

        if self.seams_fix.mode == USDUSFMode.BAND_PASS:
            seams_job_count = self.rows + self.cols - 2
        elif self.seams_fix.mode == USDUSFMode.HALF_TILE:
            seams_job_count = self.rows * (self.cols - 1) + (self.rows - 1) * self.cols
        elif self.seams_fix.mode == USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS:
            seams_job_count = self.rows * (self.cols - 1) + (self.rows - 1) * self.cols + (self.rows - 1) * (self.cols - 1)

        state.job_count = redraw_job_count + seams_job_count

    def print_info(self):
        """Print debug information about the upscaling configuration."""
        print(f"Tile size: {self.redraw.tile_width}x{self.redraw.tile_height}")
        print(f"Tiles amount: {self.rows * self.cols}")
        print(f"Grid: {self.rows}x{self.cols}")
        print(f"Redraw enabled: {self.redraw.enabled}")
        print(f"Seams fix mode: {self.seams_fix.mode.name}")

    def add_extra_info(self):
        """Add upscaling metadata to p.extra_generation_params."""
        self.p.extra_generation_params["Ultimate SD upscale upscaler"] = self.upscaler.name
        self.p.extra_generation_params["Ultimate SD upscale tile_width"] = self.redraw.tile_width
        self.p.extra_generation_params["Ultimate SD upscale tile_height"] = self.redraw.tile_height
        self.p.extra_generation_params["Ultimate SD upscale mask_blur"] = self.p.mask_blur
        self.p.extra_generation_params["Ultimate SD upscale padding"] = self.redraw.padding

    def process(self):
        """
        Run the main processing pipeline.

        1. Runs the redraw pass (if enabled)
        2. Runs the seam fix pass (if enabled)
        3. Collects result images
        """
        state.begin()
        self.calc_jobs_count()
        self.result_images = []

        # === REDRAW PASS ===
        if self.redraw.enabled:
            self.image = self.redraw.start(self.p, self.image, self.rows, self.cols)
            self.initial_info = self.redraw.initial_info
        self.result_images.append(self.image)
        if self.redraw.save:
            self.save_image()

        # === SEAM FIX PASS ===
        if self.seams_fix.enabled:
            # Reset tile index for seams fix pass (for per-tile conditioning)
            if hasattr(self.p, 'reset_tile_index'):
                self.p.reset_tile_index()
            self.image = self.seams_fix.start(self.p, self.image, self.rows, self.cols)
            self.initial_info = self.seams_fix.initial_info
            self.result_images.append(self.image)
            if self.seams_fix.save:
                self.save_image()

        state.end()


class USDURedraw():
    """
    Handles the tile redraw pass.

    Processes the image tile by tile, redrawing each with the diffusion model.
    Supports LINEAR (row-by-row) and CHESS (checkerboard) patterns.

    Attributes:
        mode: USDUMode (LINEAR, CHESS, NONE)
        enabled: Whether redraw is enabled
        save: Whether to save after redraw
        tile_width, tile_height: Tile dimensions
        padding: Context padding around tiles
        initial_info: Result info string
    """

    def init_draw(self, p, width, height):
        """
        Initialize the mask and drawing context for redrawing.

        Args:
            p: StableDiffusionProcessing instance
            width: Canvas width
            height: Canvas height

        Returns:
            Tuple of (mask, draw) where:
                mask: PIL Image in 'L' mode (black background)
                draw: ImageDraw object for drawing on mask
        """
        p.inpaint_full_res = True
        p.inpaint_full_res_padding = self.padding

        # Calculate processing dimensions (rounded to 64px for VAE)
        p.width = math.ceil((self.tile_width + self.padding) / 64) * 64
        p.height = math.ceil((self.tile_height + self.padding) / 64) * 64

        # Create black mask (white regions will be processed)
        mask = Image.new("L", (width, height), "black")
        draw = ImageDraw.Draw(mask)
        return mask, draw

    def calc_rectangle(self, xi, yi):
        """
        Calculate the pixel coordinates for a tile.

        Args:
            xi: Tile column index (0-based)
            yi: Tile row index (0-based)

        Returns:
            Tuple (x1, y1, x2, y2): Tile bounding box
        """
        x1 = xi * self.tile_width
        y1 = yi * self.tile_height
        x2 = xi * self.tile_width + self.tile_width
        y2 = yi * self.tile_height + self.tile_height

        return x1, y1, x2, y2

    def linear_process(self, p, image, rows, cols):
        """
        Process tiles in linear order (row by row).

        Iterates through tiles left-to-right, top-to-bottom.

        Args:
            p: StableDiffusionProcessing instance
            image: PIL Image to process
            rows: Number of tile rows
            cols: Number of tile columns

        Returns:
            PIL Image: Processed image with all tiles redrawn
        """
        mask, draw = self.init_draw(p, image.width, image.height)

        for yi in range(rows):
            for xi in range(cols):
                if state.interrupted:
                    break

                # Draw white rectangle on mask for this tile
                draw.rectangle(self.calc_rectangle(xi, yi), fill="white")

                # Process this tile
                p.init_images = [image]
                p.image_mask = mask
                processed = processing.process_images(p)

                # Clear mask for next tile
                draw.rectangle(self.calc_rectangle(xi, yi), fill="black")

                if (len(processed.images) > 0):
                    image = processed.images[0]

                # ADDED: Advance to next tile's conditioning (per-tile support)
                if hasattr(p, 'advance_tile'):
                    p.advance_tile()

        p.width = image.width
        p.height = image.height
        self.initial_info = processed.infotext(p, 0)

        return image

    def chess_process(self, p, image, rows, cols):
        """
        Process tiles in checkerboard pattern.

        First processes all "white" squares (alternating), then all "black" squares.
        This prevents adjacent tiles from affecting each other in the same pass.

        Args:
            p: StableDiffusionProcessing instance
            image: PIL Image to process
            rows: Number of tile rows
            cols: Number of tile columns

        Returns:
            PIL Image: Processed image with all tiles redrawn
        """
        mask, draw = self.init_draw(p, image.width, image.height)
        tiles = []

        # Build checkerboard pattern (True = process in pass 1, False = pass 2)
        for yi in range(rows):
            for xi in range(cols):
                if state.interrupted:
                    break
                if xi == 0:
                    tiles.append([])
                color = xi % 2 == 0
                if yi > 0 and yi % 2 != 0:
                    color = not color
                tiles[yi].append(color)

        # === PASS 1: Process "white" squares ===
        for yi in range(len(tiles)):
            for xi in range(len(tiles[yi])):
                if state.interrupted:
                    break
                if not tiles[yi][xi]:
                    tiles[yi][xi] = not tiles[yi][xi]
                    # ADDED: Still need to advance for skipped tiles
                    if hasattr(p, 'advance_tile'):
                        p.advance_tile()
                    continue
                tiles[yi][xi] = not tiles[yi][xi]

                draw.rectangle(self.calc_rectangle(xi, yi), fill="white")
                p.init_images = [image]
                p.image_mask = mask
                processed = processing.process_images(p)
                draw.rectangle(self.calc_rectangle(xi, yi), fill="black")
                if (len(processed.images) > 0):
                    image = processed.images[0]

                # ADDED: Advance to next tile's conditioning
                if hasattr(p, 'advance_tile'):
                    p.advance_tile()

        # Reset tile index for second pass (keep accumulators for proper blending)
        if hasattr(p, 'reset_tile_index'):
            p.reset_tile_index(reset_accumulators=False)

        # === PASS 2: Process "black" squares ===
        for yi in range(len(tiles)):
            for xi in range(len(tiles[yi])):
                if state.interrupted:
                    break
                if not tiles[yi][xi]:
                    # ADDED: Still need to advance for skipped tiles
                    if hasattr(p, 'advance_tile'):
                        p.advance_tile()
                    continue

                draw.rectangle(self.calc_rectangle(xi, yi), fill="white")
                p.init_images = [image]
                p.image_mask = mask
                processed = processing.process_images(p)
                draw.rectangle(self.calc_rectangle(xi, yi), fill="black")
                if (len(processed.images) > 0):
                    image = processed.images[0]

                # ADDED: Advance to next tile's conditioning
                if hasattr(p, 'advance_tile'):
                    p.advance_tile()

        p.width = image.width
        p.height = image.height
        self.initial_info = processed.infotext(p, 0)

        return image

    def start(self, p, image, rows, cols):
        """
        Start the redraw process with the configured mode.

        Args:
            p: StableDiffusionProcessing instance
            image: PIL Image to process
            rows: Number of tile rows
            cols: Number of tile columns

        Returns:
            PIL Image: Processed image
        """
        self.initial_info = None
        if self.mode == USDUMode.LINEAR:
            return self.linear_process(p, image, rows, cols)
        if self.mode == USDUMode.CHESS:
            return self.chess_process(p, image, rows, cols)


class USDUSeamsFix():
    """
    Handles seam repair between tiles.

    After the redraw pass, visible seams may appear between tiles.
    This class provides several methods to fix these seams.

    Attributes:
        mode: USDUSFMode (NONE, BAND_PASS, HALF_TILE, etc.)
        enabled: Whether seam fix is enabled
        save: Whether to save after seam fix
        tile_width, tile_height: Tile dimensions
        padding: Context padding
        denoise: Denoise strength for seam repair
        mask_blur: Blur amount for seam masks
        width: Width of seam bands
        initial_info: Result info string
    """

    def init_draw(self, p):
        """
        Initialize settings for seam fixing.

        Args:
            p: StableDiffusionProcessing instance
        """
        self.initial_info = None
        p.width = math.ceil((self.tile_width + self.padding) / 64) * 64
        p.height = math.ceil((self.tile_height + self.padding) / 64) * 64

    def half_tile_process(self, p, image, rows, cols):
        """
        Fix seams using half-tile overlap method.

        Creates gradient masks at tile boundaries and reprocesses those regions.
        Processes horizontal seams first, then vertical seams.

        Args:
            p: StableDiffusionProcessing instance
            image: PIL Image to process
            rows: Number of tile rows
            cols: Number of tile columns

        Returns:
            PIL Image: Image with fixed seams
        """
        self.init_draw(p)
        processed = None

        # Create vertical gradient for row seams
        gradient = Image.linear_gradient("L")
        row_gradient = Image.new("L", (self.tile_width, self.tile_height), "black")
        row_gradient.paste(gradient.resize(
            (self.tile_width, self.tile_height // 2), resample=Image.BICUBIC), (0, 0))
        row_gradient.paste(gradient.rotate(180).resize(
            (self.tile_width, self.tile_height // 2), resample=Image.BICUBIC),
            (0, self.tile_height // 2))

        # Create horizontal gradient for column seams
        col_gradient = Image.new("L", (self.tile_width, self.tile_height), "black")
        col_gradient.paste(gradient.rotate(90).resize(
            (self.tile_width // 2, self.tile_height), resample=Image.BICUBIC), (0, 0))
        col_gradient.paste(gradient.rotate(270).resize(
            (self.tile_width // 2, self.tile_height), resample=Image.BICUBIC), (self.tile_width // 2, 0))

        p.denoising_strength = self.denoise
        p.mask_blur = self.mask_blur

        # === Fix horizontal seams (between rows) ===
        for yi in range(rows - 1):
            for xi in range(cols):
                if state.interrupted:
                    break
                p.width = self.tile_width
                p.height = self.tile_height
                p.inpaint_full_res = True
                p.inpaint_full_res_padding = self.padding
                mask = Image.new("L", (image.width, image.height), "black")
                mask.paste(row_gradient, (xi * self.tile_width, yi * self.tile_height + self.tile_height // 2))

                p.init_images = [image]
                p.image_mask = mask
                processed = processing.process_images(p)
                if (len(processed.images) > 0):
                    image = processed.images[0]

        # === Fix vertical seams (between columns) ===
        for yi in range(rows):
            for xi in range(cols - 1):
                if state.interrupted:
                    break
                p.width = self.tile_width
                p.height = self.tile_height
                p.inpaint_full_res = True
                p.inpaint_full_res_padding = self.padding
                mask = Image.new("L", (image.width, image.height), "black")
                mask.paste(col_gradient, (xi * self.tile_width + self.tile_width // 2, yi * self.tile_height))

                p.init_images = [image]
                p.image_mask = mask
                processed = processing.process_images(p)
                if (len(processed.images) > 0):
                    image = processed.images[0]

        p.width = image.width
        p.height = image.height
        if processed is not None:
            self.initial_info = processed.infotext(p, 0)

        return image

    def half_tile_process_corners(self, p, image, rows, cols):
        """
        Fix seams including corner intersections.

        First runs half_tile_process, then additionally processes the
        corner intersections where four tiles meet.

        Args:
            p: StableDiffusionProcessing instance
            image: PIL Image to process
            rows: Number of tile rows
            cols: Number of tile columns

        Returns:
            PIL Image: Image with fixed seams and corners
        """
        # First do regular half-tile seam fix
        fixed_image = self.half_tile_process(p, image, rows, cols)
        processed = None
        self.init_draw(p)

        # Create radial gradient for corner intersections (inverted)
        gradient = Image.radial_gradient("L").resize(
            (self.tile_width, self.tile_height), resample=Image.BICUBIC)
        gradient = ImageOps.invert(gradient)

        p.denoising_strength = self.denoise
        p.mask_blur = self.mask_blur

        # === Fix corner intersections ===
        for yi in range(rows - 1):
            for xi in range(cols - 1):
                if state.interrupted:
                    break
                p.width = self.tile_width
                p.height = self.tile_height
                p.inpaint_full_res = True
                p.inpaint_full_res_padding = 0
                mask = Image.new("L", (fixed_image.width, fixed_image.height), "black")
                mask.paste(gradient, (xi * self.tile_width + self.tile_width // 2,
                                      yi * self.tile_height + self.tile_height // 2))

                p.init_images = [fixed_image]
                p.image_mask = mask
                processed = processing.process_images(p)
                if (len(processed.images) > 0):
                    fixed_image = processed.images[0]

        p.width = fixed_image.width
        p.height = fixed_image.height
        if processed is not None:
            self.initial_info = processed.infotext(p, 0)

        return fixed_image

    def band_pass_process(self, p, image, cols, rows):
        """
        Fix seams using band pass method.

        Processes narrow bands along seam lines. Uses gradient masks
        that fade to zero at the edges for smooth blending.

        Args:
            p: StableDiffusionProcessing instance
            image: PIL Image to process
            cols: Number of tile columns
            rows: Number of tile rows

        Returns:
            PIL Image: Image with fixed seams
        """
        self.init_draw(p)
        processed = None

        p.denoising_strength = self.denoise
        p.mask_blur = 0

        # Create mirror gradient for band masks
        gradient = Image.linear_gradient("L")
        mirror_gradient = Image.new("L", (256, 256), "black")
        mirror_gradient.paste(gradient.resize((256, 128), resample=Image.BICUBIC), (0, 0))
        mirror_gradient.paste(gradient.rotate(180).resize((256, 128), resample=Image.BICUBIC), (0, 128))

        # Create gradients sized to image
        row_gradient = mirror_gradient.resize((image.width, self.width), resample=Image.BICUBIC)
        col_gradient = mirror_gradient.rotate(90).resize((self.width, image.height), resample=Image.BICUBIC)

        # === Fix vertical seams ===
        for xi in range(1, rows):
            if state.interrupted:
                break
            p.width = self.width + self.padding * 2
            p.height = image.height
            p.inpaint_full_res = True
            p.inpaint_full_res_padding = self.padding
            mask = Image.new("L", (image.width, image.height), "black")
            mask.paste(col_gradient, (xi * self.tile_width - self.width // 2, 0))

            p.init_images = [image]
            p.image_mask = mask
            processed = processing.process_images(p)
            if (len(processed.images) > 0):
                image = processed.images[0]

        # === Fix horizontal seams ===
        for yi in range(1, cols):
            if state.interrupted:
                break
            p.width = image.width
            p.height = self.width + self.padding * 2
            p.inpaint_full_res = True
            p.inpaint_full_res_padding = self.padding
            mask = Image.new("L", (image.width, image.height), "black")
            mask.paste(row_gradient, (0, yi * self.tile_height - self.width // 2))

            p.init_images = [image]
            p.image_mask = mask
            processed = processing.process_images(p)
            if (len(processed.images) > 0):
                image = processed.images[0]

        p.width = image.width
        p.height = image.height
        if processed is not None:
            self.initial_info = processed.infotext(p, 0)

        return image

    def start(self, p, image, rows, cols):
        """
        Start seam fixing with the configured mode.

        Args:
            p: StableDiffusionProcessing instance
            image: PIL Image to process
            rows: Number of tile rows
            cols: Number of tile columns

        Returns:
            PIL Image: Image with fixed seams
        """
        if USDUSFMode(self.mode) == USDUSFMode.BAND_PASS:
            return self.band_pass_process(p, image, rows, cols)
        elif USDUSFMode(self.mode) == USDUSFMode.HALF_TILE:
            return self.half_tile_process(p, image, rows, cols)
        elif USDUSFMode(self.mode) == USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS:
            return self.half_tile_process_corners(p, image, rows, cols)
        else:
            return image


class Script():
    """
    A1111-style script entry point.

    This class provides the run() method which is the main entry point
    for the USDU script. It orchestrates the entire upscaling process.
    """

    def title(self):
        """Return the script title for UI display."""
        return "Smart Ultimate SD upscale"

    def show(self, is_img2img):
        """Return whether to show this script (only for img2img)."""
        return is_img2img

    def run(self, p, _, tile_width, tile_height, mask_blur, padding, seams_fix_width, seams_fix_denoise, seams_fix_padding,
            upscaler_index, save_upscaled_image, redraw_mode, save_seams_fix_image, seams_fix_mask_blur,
            seams_fix_type, target_size_type, custom_width, custom_height, custom_scale):
        """
        Main entry point for the USDU script.

        Args:
            p: StableDiffusionProcessing instance
            _: Unused (A1111 compatibility)
            tile_width: Width of each tile
            tile_height: Height of each tile
            mask_blur: Blur for tile masks
            padding: Context padding around tiles
            seams_fix_width: Width of seam fix bands
            seams_fix_denoise: Denoise strength for seam fix
            seams_fix_padding: Padding for seam fix
            upscaler_index: Index into sd_upscalers
            save_upscaled_image: Whether to save after upscale
            redraw_mode: USDUMode value
            save_seams_fix_image: Whether to save after seam fix
            seams_fix_mask_blur: Blur for seam fix masks
            seams_fix_type: USDUSFMode value
            target_size_type: 0=from p.width/height, 1=custom, 2=scale
            custom_width: Custom width (if target_size_type=1)
            custom_height: Custom height (if target_size_type=1)
            custom_scale: Scale factor (if target_size_type=2)

        Returns:
            Processed: Result containing images and info
        """
        # Init
        processing.fix_seed(p)

        p.do_not_save_grid = True
        p.do_not_save_samples = True
        p.inpaint_full_res = False

        p.inpainting_fill = 1
        p.n_iter = 1
        p.batch_size = 1

        seed = p.seed

        # Get init image
        init_img = p.init_images[0]
        if init_img == None:
            return Processed(p, [], seed, "Empty image")

        # Override target size based on target_size_type
        if target_size_type == 1:
            # Custom width/height
            p.width = custom_width
            p.height = custom_height
        if target_size_type == 2:
            # Scale from input image
            p.width = math.ceil((init_img.width * custom_scale) / 64) * 64
            p.height = math.ceil((init_img.height * custom_scale) / 64) * 64

        # Create upscaler and run
        upscaler = USDUpscaler(p, init_img, upscaler_index, save_upscaled_image, save_seams_fix_image, tile_width, tile_height)
        upscaler.upscale()

        # Configure and run processing
        upscaler.setup_redraw(redraw_mode, padding, mask_blur)
        upscaler.setup_seams_fix(seams_fix_padding, seams_fix_denoise, seams_fix_mask_blur, seams_fix_width, seams_fix_type)
        upscaler.print_info()
        upscaler.add_extra_info()
        upscaler.process()
        result_images = upscaler.result_images

        return Processed(p, result_images, seed, upscaler.initial_info if upscaler.initial_info is not None else "")
