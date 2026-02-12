"""
Panorama Conversion Nodes
=========================
Convert between equirectangular (spherical) 360° panoramas and cubemap face batches.
Uses py360convert for the projection math.
"""

import torch
import numpy as np

try:
    import py360convert
    PY360_AVAILABLE = True
except ImportError:
    PY360_AVAILABLE = False
    print("[ArchAi3d] py360convert not installed. Panorama nodes disabled. Install with: pip install py360convert")


class ArchAi3D_PanoramaToCubemap:
    """Convert a 360° equirectangular panorama into 6 cubemap faces as a batch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "face_size": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 4096,
                    "step": 64,
                    "tooltip": "Size of each cube face in pixels (square)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("cubemap_faces",)
    FUNCTION = "convert"
    CATEGORY = "ArchAi3d/Panorama"
    DESCRIPTION = "Converts a 360° equirectangular panorama into 6 cubemap faces (Front, Right, Back, Left, Top, Bottom) as a batch."

    def convert(self, image, face_size):
        if not PY360_AVAILABLE:
            raise ImportError(
                "py360convert is required for panorama conversion. "
                "Install with: pip install py360convert"
            )

        # ComfyUI tensor: (B, H, W, C) float32 [0,1] — take first image
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)

        # Equirectangular to cubemap dict
        cube_faces = py360convert.e2c(
            img_np,
            face_w=face_size,
            mode='bilinear',
            cube_format='dict'
        )

        # Stack 6 faces into batch: Front, Right, Back, Left, Top, Bottom
        face_order = ['F', 'R', 'B', 'L', 'U', 'D']
        faces = [cube_faces[k].astype(np.float32) / 255.0 for k in face_order]
        batch = torch.from_numpy(np.stack(faces, axis=0))  # (6, H, W, C)

        return (batch,)


class ArchAi3D_CubemapToPanorama:
    """Convert 6 cubemap face images (batch) back into a 360° equirectangular panorama."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cubemap_faces": ("IMAGE",),
                "output_width": ("INT", {
                    "default": 2048,
                    "min": 512,
                    "max": 8192,
                    "step": 64,
                    "tooltip": "Width of the output panorama"
                }),
                "output_height": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 4096,
                    "step": 64,
                    "tooltip": "Height of the output panorama (should be half of width)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("panorama",)
    FUNCTION = "convert"
    CATEGORY = "ArchAi3d/Panorama"
    DESCRIPTION = "Converts 6 cubemap faces (batch of 6: Front, Right, Back, Left, Top, Bottom) back into a 360° equirectangular panorama."

    def convert(self, cubemap_faces, output_width, output_height):
        if not PY360_AVAILABLE:
            raise ImportError(
                "py360convert is required for panorama conversion. "
                "Install with: pip install py360convert"
            )

        if cubemap_faces.shape[0] < 6:
            raise ValueError(
                f"Expected batch of 6 cubemap faces, got {cubemap_faces.shape[0]}. "
                "Face order should be: Front, Right, Back, Left, Top, Bottom."
            )

        # Extract 6 faces from batch, convert to uint8 numpy
        face_order = ['F', 'R', 'B', 'L', 'U', 'D']
        cube_dict = {}
        for i, key in enumerate(face_order):
            cube_dict[key] = (cubemap_faces[i].cpu().numpy() * 255).astype(np.uint8)

        # Cubemap to equirectangular
        equi_img = py360convert.c2e(
            cube_dict,
            h=output_height,
            w=output_width,
            mode='bilinear',
            cube_format='dict'
        )

        # Back to ComfyUI tensor: (1, H, W, C)
        equi_tensor = torch.from_numpy(equi_img.astype(np.float32) / 255.0).unsqueeze(0)

        return (equi_tensor,)
