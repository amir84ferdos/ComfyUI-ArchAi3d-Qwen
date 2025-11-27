"""
ArchAi3D Save Image Node
Saves an image with a name field for web interface integration.
"""

import os
import json
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import folder_paths


class ArchAi3D_Save_Image:
    """
    Save an image with a configurable name for web interface integration.

    The 'output_name' field identifies this output in web interfaces, allowing
    the web app to reference saved images by name.

    The 'save' boolean toggle controls whether the image is actually saved to disk.
    """

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "save": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable/disable saving the image to disk"
                }),
                "filename_prefix": ("STRING", {
                    "default": "ComfyUI",
                    "tooltip": "Prefix for the saved filename"
                }),
                "output_name": ("STRING", {
                    "default": "output_image",
                    "multiline": False,
                    "tooltip": "Identifier name for this output (used by web interface)"
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = "ArchAi3d/Inputs"

    def _get_next_counter(self, folder, filename_prefix):
        """Find the next available counter to avoid overwriting files."""
        counter = 1
        while True:
            # Check if file with this counter exists
            test_file = os.path.join(folder, f"{filename_prefix}_{counter:05d}.png")
            if not os.path.exists(test_file):
                return counter
            counter += 1

    def execute(self, images, save=True, filename_prefix="ComfyUI", output_name="output_image", prompt=None, extra_pnginfo=None):
        # If save is False, return empty results (no saving)
        if not save:
            return {"ui": {"images": []}}

        filename_prefix += self.prefix_append

        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
        )

        # Ensure we don't overwrite existing files
        counter = self._get_next_counter(full_output_folder, filename)

        results = []

        for batch_number, image in enumerate(images):
            # Convert from tensor to numpy
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            # Prepare metadata
            metadata = PngInfo()

            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            # Add output_name to metadata for web interface
            metadata.add_text("output_name", output_name)

            # Generate filename with batch number if multiple images
            if len(images) > 1:
                file = f"{filename}_{batch_number:05d}_{counter:05d}.png"
            else:
                file = f"{filename}_{counter:05d}.png"

            filepath = os.path.join(full_output_folder, file)

            # Extra safety: check if file exists and increment counter
            while os.path.exists(filepath):
                counter += 1
                if len(images) > 1:
                    file = f"{filename}_{batch_number:05d}_{counter:05d}.png"
                else:
                    file = f"{filename}_{counter:05d}.png"
                filepath = os.path.join(full_output_folder, file)

            img.save(filepath, pnginfo=metadata, compress_level=self.compress_level)

            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type,
                "output_name": output_name
            })

            counter += 1

        return {"ui": {"images": results}}
