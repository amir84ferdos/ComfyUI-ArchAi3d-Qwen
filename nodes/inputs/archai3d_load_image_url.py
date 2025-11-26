"""
ArchAi3D Load Image From URL Node
Loads an image from a URL with a name field for web interface integration.
"""

import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO


class ArchAi3D_Load_Image_URL:
    """
    Load an image from a URL with a configurable name for web interface integration.

    The 'name' field identifies this input in web interfaces, allowing
    dynamic HTML form generation based on workflow inputs.

    Supports RGB, RGBA, and grayscale image modes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {
                    "default": "image_url",
                    "multiline": False,
                    "tooltip": "Identifier name for this input (used by web interface)"
                }),
                "url": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "URL of the image to load"
                }),
                "return_image_mode": (["RGB", "RGBA", "L"], {
                    "default": "RGB",
                    "tooltip": "Output image mode: RGB (color), RGBA (with alpha), L (grayscale)"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "MASK")
    FUNCTION = "execute"
    CATEGORY = "ArchAi3d/Inputs"

    def execute(self, name, url, return_image_mode):
        if not url or url.strip() == "":
            # Return empty tensors if no URL provided
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return (empty_image, empty_mask)

        try:
            # Fetch image from URL
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Open image with PIL
            image = Image.open(BytesIO(response.content))

            # Extract alpha channel for mask before conversion
            if image.mode == "RGBA":
                alpha = image.split()[3]
                mask = np.array(alpha).astype(np.float32) / 255.0
            else:
                # No alpha channel, create full white mask (no transparency)
                mask = np.ones((image.height, image.width), dtype=np.float32)

            # Convert to requested mode
            if return_image_mode == "RGB":
                image = image.convert("RGB")
            elif return_image_mode == "RGBA":
                image = image.convert("RGBA")
            elif return_image_mode == "L":
                image = image.convert("L")

            # Convert to numpy array
            image_np = np.array(image).astype(np.float32) / 255.0

            # Handle grayscale (add channel dimension and expand to 3 channels for ComfyUI)
            if return_image_mode == "L":
                image_np = np.stack([image_np, image_np, image_np], axis=-1)

            # Ensure 3 channels for RGB mode (some images might be RGBA after conversion issues)
            if image_np.ndim == 2:
                image_np = np.stack([image_np, image_np, image_np], axis=-1)

            # For RGBA, only use RGB channels for IMAGE output
            if return_image_mode == "RGBA" and image_np.shape[-1] == 4:
                image_np = image_np[..., :3]

            # Convert to torch tensor with batch dimension [B, H, W, C]
            image_tensor = torch.from_numpy(image_np).unsqueeze(0)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0)

            return (image_tensor, mask_tensor)

        except Exception as e:
            print(f"[ArchAi3D_Load_Image_URL] Error loading image from URL: {e}")
            # Return empty tensors on error
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return (empty_image, empty_mask)
