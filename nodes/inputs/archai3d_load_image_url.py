"""
ArchAi3D Load Image From URL Node
Loads an image from a URL, local file path, or drag-and-drop upload.
Includes name field for web interface integration and preview functionality.
"""

import os
import hashlib
import torch
import numpy as np
from PIL import Image, ImageOps, ImageSequence
import requests
from io import BytesIO
import folder_paths


class ArchAi3D_Load_Image_URL:
    """
    Load an image from URL, local path, or drag-and-drop upload.

    Features:
    - Drag & drop images directly onto the node
    - Click to browse and select local files
    - Paste URLs to load remote images
    - Name field for web interface integration

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
        input_dir = folder_paths.get_input_directory()
        files = []
        for f in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, f)):
                files.append(f)
        files = folder_paths.filter_files_content_types(files, ["image"])

        return {
            "required": {
                "name": ("STRING", {
                    "default": "image_input",
                    "multiline": False,
                    "tooltip": "Identifier name for this input (used by web interface)"
                }),
                "image": (sorted(files), {
                    "image_upload": True,
                    "tooltip": "Drag & drop image here, or click to browse"
                }),
            },
            "optional": {
                "url": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "URL to load image from (overrides uploaded image if provided)"
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

    def _is_local_path(self, path):
        """Check if the path is a local file path."""
        # Check for common local path patterns
        if path.startswith('/') or path.startswith('\\'):
            return True
        if len(path) > 1 and path[1] == ':':  # Windows path like C:\
            return True
        if path.startswith('file://'):
            return True
        if os.path.exists(path):
            return True
        return False

    def _load_image(self, url):
        """Load image from URL or local file path."""
        url = url.strip()

        # Handle file:// protocol
        if url.startswith('file://'):
            url = url[7:]  # Remove file:// prefix

        # Check if it's a local file path
        if self._is_local_path(url):
            if os.path.exists(url):
                return Image.open(url)
            else:
                raise FileNotFoundError(f"Local file not found: {url}")
        else:
            # It's a URL, fetch it
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))

    def execute(self, name, image, url=None, return_image_mode="RGB"):
        """
        Load image from uploaded file or URL.
        Priority: URL (if provided) > Uploaded image
        """
        pil_image = None
        source_info = None

        try:
            # Priority 1: Use URL if provided
            if url and url.strip():
                pil_image = self._load_image(url)
                source_info = url

            # Priority 2: Use uploaded/dropped image
            elif image and image.strip():
                image_path = folder_paths.get_annotated_filepath(image)
                pil_image = Image.open(image_path)
                # Handle EXIF orientation
                pil_image = ImageOps.exif_transpose(pil_image)
                source_info = image_path

            # No input provided
            if pil_image is None:
                empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
                empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
                return {"ui": {"images": []}, "result": (empty_image, empty_mask)}

            original_image = pil_image.copy()

            # Extract alpha channel for mask before conversion
            if pil_image.mode == "RGBA":
                alpha = pil_image.split()[3]
                mask = np.array(alpha).astype(np.float32) / 255.0
            else:
                # No alpha channel, create full white mask (no transparency)
                mask = np.ones((pil_image.height, pil_image.width), dtype=np.float32)

            # Convert to requested mode
            if return_image_mode == "RGB":
                pil_image = pil_image.convert("RGB")
            elif return_image_mode == "RGBA":
                pil_image = pil_image.convert("RGBA")
            elif return_image_mode == "L":
                pil_image = pil_image.convert("L")

            # Convert to numpy array
            image_np = np.array(pil_image).astype(np.float32) / 255.0

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
            preview_results = self._save_preview(original_image, source_info)

            return {"ui": {"images": preview_results}, "result": (image_tensor, mask_tensor)}

        except Exception as e:
            print(f"[ArchAi3D_Load_Image_URL] Error loading image: {e}")
            # Return empty tensors on error
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return {"ui": {"images": []}, "result": (empty_image, empty_mask)}

    def _save_preview(self, image, url):
        """Save a preview image to the temp directory for display in the node."""
        results = []

        try:
            # Create a unique filename based on URL/path hash
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
    def IS_CHANGED(cls, name, image, url=None, return_image_mode="RGB"):
        """Return a hash that changes when the image/URL changes, forcing re-execution."""
        # Check URL first
        if url and url.strip():
            url = url.strip()
            if url.startswith('file://'):
                url = url[7:]
            if os.path.exists(url):
                mtime = os.path.getmtime(url)
                return hashlib.md5(f"{url}_{mtime}".encode()).hexdigest()
            return hashlib.md5(url.encode()).hexdigest()

        # Check uploaded image
        if image and image.strip():
            image_path = folder_paths.get_annotated_filepath(image)
            if os.path.exists(image_path):
                mtime = os.path.getmtime(image_path)
                return hashlib.md5(f"{image_path}_{mtime}".encode()).hexdigest()
            return hashlib.md5(image.encode()).hexdigest()

        return ""

    @classmethod
    def VALIDATE_INPUTS(cls, name, image, url=None, return_image_mode="RGB"):
        """Validate that the image file exists."""
        if url and url.strip():
            # URL validation is handled at runtime
            return True

        if image and image.strip():
            if not folder_paths.exists_annotated_filepath(image):
                return f"Invalid image file: {image}"

        return True
