"""
Batch Split & Merge Nodes
=========================
Split a 6-image batch into 6 individual outputs, or merge 6 images into a batch.
Designed for cubemap face workflows but works with any 6-image batch.
"""

import torch


class ArchAi3D_BatchToSixImages:
    """Split a batch of 6 images into 6 individual image outputs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("image_1", "image_2", "image_3", "image_4", "image_5", "image_6")
    FUNCTION = "split"
    CATEGORY = "ArchAi3d/Panorama"
    DESCRIPTION = "Splits a batch of 6 images into 6 individual outputs. For cubemap: Front, Right, Back, Left, Top, Bottom."

    def split(self, batch):
        if batch.shape[0] < 6:
            raise ValueError(
                f"Expected batch of at least 6 images, got {batch.shape[0]}."
            )

        return (
            batch[0:1],
            batch[1:2],
            batch[2:3],
            batch[3:4],
            batch[4:5],
            batch[5:6],
        )


class ArchAi3D_SixImagesToBatch:
    """Merge 6 individual images into a single batch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("batch",)
    FUNCTION = "merge"
    CATEGORY = "ArchAi3d/Panorama"
    DESCRIPTION = "Merges 6 individual images into a batch of 6. For cubemap: connect Front, Right, Back, Left, Top, Bottom in order."

    def merge(self, image_1, image_2, image_3, image_4, image_5, image_6):
        return (torch.cat([
            image_1[0:1],
            image_2[0:1],
            image_3[0:1],
            image_4[0:1],
            image_5[0:1],
            image_6[0:1],
        ], dim=0),)
