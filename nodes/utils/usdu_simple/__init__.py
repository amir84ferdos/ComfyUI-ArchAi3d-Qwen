"""
Simple USDU - Modular Tile Processing Nodes
============================================

A simpler, more modular approach to tile-based upscaling.
Each node does ONE thing well:

1. Simple Tile Cropper - Extract tiles from image
2. Simple Blend Mask - Create distance-based blend masks
3. Simple Tile Sampler - Process tiles with diffusion
4. Simple Tile Compositor - Blend tiles back together

Also includes batch versions and utility nodes:
- Simple Tile Cropper (Batch)
- Simple Blend Mask (Batch)
- Simple Tile Compositor (Single) - For iterative compositing
- Simple Edge Mask (DiffDiff) - Create edge preservation masks

These nodes are designed to work with Smart Tile Solver V6.2 outputs.
"""

from .blend_mask import NODE_CLASS_MAPPINGS as BLEND_MASK_NODES
from .blend_mask import NODE_DISPLAY_NAME_MAPPINGS as BLEND_MASK_NAMES

from .tile_cropper import NODE_CLASS_MAPPINGS as CROPPER_NODES
from .tile_cropper import NODE_DISPLAY_NAME_MAPPINGS as CROPPER_NAMES

from .tile_compositor import NODE_CLASS_MAPPINGS as COMPOSITOR_NODES
from .tile_compositor import NODE_DISPLAY_NAME_MAPPINGS as COMPOSITOR_NAMES

from .tile_sampler import NODE_CLASS_MAPPINGS as SAMPLER_NODES
from .tile_sampler import NODE_DISPLAY_NAME_MAPPINGS as SAMPLER_NAMES


# Combine all node mappings
NODE_CLASS_MAPPINGS = {
    **BLEND_MASK_NODES,
    **CROPPER_NODES,
    **COMPOSITOR_NODES,
    **SAMPLER_NODES,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **BLEND_MASK_NAMES,
    **CROPPER_NAMES,
    **COMPOSITOR_NAMES,
    **SAMPLER_NAMES,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
