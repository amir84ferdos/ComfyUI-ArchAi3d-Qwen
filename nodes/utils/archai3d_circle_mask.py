"""
Shape Mask Generator
====================
Generates a black and white image with a circle or rectangle.
Configurable size, aspect, position, blur, and invert.
"""

import torch
import numpy as np


class ArchAi3D_CircleMask:
    """Generate a shape mask with configurable size, position, aspect ratio, blur, and invert."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shape": (["circle", "rectangle"],),
                "width": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 8,
                    "tooltip": "Image width in pixels"
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 8,
                    "tooltip": "Image height in pixels"
                }),
                "shape_width_percent": ("FLOAT", {
                    "default": 10.0,
                    "min": 0.1,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Shape width as % of image width"
                }),
                "shape_height_percent": ("FLOAT", {
                    "default": 10.0,
                    "min": 0.1,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Shape height as % of image height"
                }),
                "position_x_percent": ("FLOAT", {
                    "default": 50.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Horizontal position of shape center (0=left, 50=center, 100=right)"
                }),
                "position_y_percent": ("FLOAT", {
                    "default": 50.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Vertical position of shape center (0=top, 50=center, 100=bottom)"
                }),
                "edge_blur": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.5,
                    "tooltip": "Blur width at shape edge in pixels. 0 = hard edge."
                }),
                "invert": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert: white background with black shape"
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Number of copies in the output batch"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "generate"
    CATEGORY = "ArchAi3d/Panorama"
    DESCRIPTION = "Generates a shape mask (circle or rectangle). Separate width/height control and positionable center."

    def generate(self, shape, width, height, shape_width_percent, shape_height_percent,
                 position_x_percent, position_y_percent, edge_blur, invert, batch_size):
        # Shape center position
        cx = (position_x_percent / 100.0) * width
        cy = (position_y_percent / 100.0) * height

        # Shape half-dimensions
        half_w = (shape_width_percent / 100.0) * width / 2.0
        half_h = (shape_height_percent / 100.0) * height / 2.0

        y = np.arange(height, dtype=np.float32)
        x = np.arange(width, dtype=np.float32)
        yy, xx = np.meshgrid(y, x, indexing='ij')

        if shape == "circle":
            # Ellipse: normalize distance by radii
            dist = np.sqrt(((xx - cx) / (half_w + 1e-6)) ** 2 + ((yy - cy) / (half_h + 1e-6)) ** 2)

            if edge_blur > 0:
                # Blur in normalized space
                blur_norm = edge_blur / (min(half_w, half_h) + 1e-6)
                inner = 1.0 - blur_norm / 2.0
                outer = 1.0 + blur_norm / 2.0
                mask = np.clip((outer - dist) / (outer - inner + 1e-6), 0.0, 1.0)
            else:
                mask = (dist <= 1.0).astype(np.float32)

        else:  # rectangle
            dx = half_w - np.abs(xx - cx)
            dy = half_h - np.abs(yy - cy)
            dist_from_edge = np.minimum(dx, dy)

            if edge_blur > 0:
                mask = np.clip((dist_from_edge + edge_blur / 2.0) / (edge_blur + 1e-6), 0.0, 1.0)
            else:
                mask = (dist_from_edge >= 0).astype(np.float32)

        if invert:
            mask = 1.0 - mask

        # Create 3-channel image (1, H, W, C)
        img = np.stack([mask, mask, mask], axis=-1)
        img_tensor = torch.from_numpy(img).unsqueeze(0)

        # Create single-channel mask (1, H, W)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        if batch_size > 1:
            img_tensor = img_tensor.repeat(batch_size, 1, 1, 1)
            mask_tensor = mask_tensor.repeat(batch_size, 1, 1)

        return (img_tensor, mask_tensor)
