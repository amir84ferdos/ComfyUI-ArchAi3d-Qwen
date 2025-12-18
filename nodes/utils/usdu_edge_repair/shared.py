"""
USDU Edge Repair - Shared State Module
=======================================

TREE MAP:
---------
shared.py
├── Options (class)              - Configuration settings (A1111 compatibility)
│   └── img2img_background_color - Background color for img2img operations
│
├── State (class)                - Processing state tracker (A1111 compatibility)
│   ├── interrupted              - Flag to stop processing
│   ├── begin()                  - Called when processing starts
│   └── end()                    - Called when processing ends
│
├── opts (instance)              - Global Options instance
├── state (instance)             - Global State instance
│
├── sd_upscalers (list)          - List of available upscaler models
├── actual_upscaler              - Currently active upscaler model
│
├── batch (list[PIL.Image])      - Batch of images being processed
└── batch_as_tensor (torch.Tensor) - Same batch as tensor [B, H, W, C]

DATA FLOW:
----------
1. Main node sets batch/batch_as_tensor with input images
2. Upscaler reads batch_as_tensor for model-based upscaling
3. Processing reads/writes batch for tile-by-tile operations
4. Main node reads batch to get final results

USAGE:
------
This module provides global state that is shared between:
- archai3d_usdu_edge_repair.py (sets batch, reads results)
- processing.py (reads/writes batch during tile processing)
- upscaler.py (reads actual_upscaler and batch_as_tensor)
- usdu_patch.py (reads/writes batch for upscale patches)
"""


class Options:
    """
    A1111-compatible options class.

    Provides configuration settings that the original Ultimate SD Upscale
    script expected from Automatic1111's WebUI. In ComfyUI context, most
    of these are hardcoded since they're handled differently.
    """
    img2img_background_color = "#ffffff"  # Background color for img2img (white)


class State:
    """
    A1111-compatible state tracking class.

    Tracks processing state (interrupted, job count, etc.) that the original
    USDU script used. In ComfyUI context, interruption is handled by the
    UI's cancel button instead.

    Attributes:
        interrupted (bool): Set to True to stop processing mid-tile
        job_count (int): Total number of jobs (tiles) to process
    """
    interrupted = False

    def begin(self):
        """Called when USDU processing starts. Resets state."""
        pass

    def end(self):
        """Called when USDU processing ends. Cleanup if needed."""
        pass


# ============================================================
# GLOBAL INSTANCES - Used throughout the module
# ============================================================

# Global options and state instances (A1111 compatibility)
opts = Options()
state = State()

# ============================================================
# UPSCALER STATE
# ============================================================

# List of available upscalers - will only ever hold 1 upscaler in ComfyUI
# Index 0 is used by the USDU script
sd_upscalers = [None]

# The actual upscaler model usable by ComfyUI nodes
# Set to None when using pre-upscaled images (no model upscaling needed)
actual_upscaler = None

# ============================================================
# BATCH STATE - The images being processed
# ============================================================

# Batch of PIL Images being processed
# Modified by processing.py as each tile is completed
batch = None

# Same batch as a PyTorch tensor in [B, H, W, C] format
# Used by upscaler.py for model-based upscaling
batch_as_tensor = None
