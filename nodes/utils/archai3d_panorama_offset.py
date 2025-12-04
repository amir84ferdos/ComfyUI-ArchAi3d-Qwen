# -*- coding: utf-8 -*-
"""
ArchAi3D Panorama Offset Node
Author: Amir Ferdos (ArchAi3d)
Email: Amir84ferdos@gmail.com
LinkedIn: https://www.linkedin.com/in/archai3d/

Description:
    Offsets an image horizontally by a percentage. Perfect for creating seamless
    panoramas by shifting the image so seams meet in the middle where they can
    be easily edited.

Usage:
    1. Connect input image
    2. Set offset percentage (default 50% = half image width)
    3. Output shows shifted image with edges meeting in center

Use Cases:
    - Creating seamless panoramas (offset 50% to check/fix seams)
    - Panorama seam editing workflow
    - Texture tiling preparation
    - Cyclic image manipulation

Version: 1.0.0
"""

import torch


class ArchAi3D_Panorama_Offset:
    """Panorama Offset: Shift image horizontally for seamless panorama creation.

    This node offsets an image horizontally by a percentage of its width.
    At 50% offset, the left and right edges meet in the center, making it
    easy to identify and fix seams for seamless panoramas.

    Workflow:
    1. Generate/load panorama image
    2. Offset by 50% to bring seams to center
    3. Fix seams in center (inpaint, manual edit, etc.)
    4. Offset back by 50% (or use original) for final result
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "offset_percent": ("FLOAT", {
                    "default": 50.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    "tooltip": "Horizontal offset as percentage of image width. 50% brings edges to center."
                }),
                "direction": (["right", "left"], {
                    "default": "right",
                    "tooltip": "Direction to shift the image. 'right' moves content to the right."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("offset_image",)
    FUNCTION = "execute"
    CATEGORY = "ArchAi3d/Utils"

    def execute(self, image, offset_percent=50.0, direction="right"):
        """Execute the Panorama Offset."""

        batch, height, width, channels = image.shape

        # Calculate pixel offset
        offset_pixels = int((offset_percent / 100.0) * width)
        offset_pixels = offset_pixels % width

        if direction == "left":
            offset_pixels = width - offset_pixels

        # Apply horizontal offset (roll)
        offset_image = torch.roll(image, shifts=offset_pixels, dims=2)

        return (offset_image,)
