"""
ArchAi3D Load Image From URL Node
Loads an image from a URL with a name field for web interface integration.
Includes preview functionality like the default Load Image node.
"""

import os
import hashlib
import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import folder_paths


class ArchAi3D_Load_Image_URL:
    """
    Load an image from a URL with a configurable name for web interface integration.

    The 'name' field identifies this input in web interfaces, allowing
    dynamic HTML form generation based on workflow inputs.

    Supports RGB, RGBA, and grayscale image modes.
    Includes image preview in the node.
    """

    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(chr(ord('a') + i) for i in [0, 1, 2])
        self.compress_level = 1

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
    OUTPUT_NODE = True

    def execute(self, name, url, return_image_mode):
        if not url or url.strip() == "":
            # Return empty tensors if no URL provided
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return {"ui": {"images": []}, "result": (empty_image, empty_mask)}

        try:
            # Fetch image from URL
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Open image with PIL
            image = Image.open(BytesIO(response.content))
            original_image = image.copy()

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

            # Save preview image to temp folder
            preview_results = self._save_preview(original_image, url)

            return {"ui": {"images": preview_results}, "result": (image_tensor, mask_tensor)}

        except Exception as e:
            print(f"[ArchAi3D_Load_Image_URL] Error loading image from URL: {e}")
            # Return empty tensors on error
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return {"ui": {"images": []}, "result": (empty_image, empty_mask)}

    def _save_preview(self, image, url):
        """Save a preview image to the temp directory for display in the node."""
        results = []

        try:
            # Create a unique filename based on URL hash
            url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
            filename = f"preview_{url_hash}.png"

            # Ensure temp directory exists
            os.makedirs(self.output_dir, exist_ok=True)

            filepath = os.path.join(self.output_dir, filename)

            # Convert to RGB for preview if needed
            if image.mode in ["RGBA", "LA"]:
                preview_image = Image.new("RGB", image.size, (0, 0, 0))
                preview_image.paste(image, mask=image.split()[-1] if image.mode == "RGBA" else None)
            elif image.mode != "RGB":
                preview_image = image.convert("RGB")
            else:
                preview_image = image

            preview_image.save(filepath, compress_level=self.compress_level)

            results.append({
                "filename": filename,
                "subfolder": "",
                "type": self.type
            })

        except Exception as e:
            print(f"[ArchAi3D_Load_Image_URL] Error saving preview: {e}")

        return results

    @classmethod
    def IS_CHANGED(cls, name, url, return_image_mode):
        """Return a hash that changes when the URL changes, forcing re-execution."""
        if not url or url.strip() == "":
            return ""
        return hashlib.md5(url.encode()).hexdigest()
