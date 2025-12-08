"""
ArchAi3D Save Image Node
Saves an image with format options and workflow embedding control.

Version: 3.0.0 - Fixed API/history registration for RunPod compatibility
- Changed FUNCTION to "save_images" to match standard SaveImage
- Fixed filename format with trailing underscore
- Added verbose debug logging for API troubleshooting
"""

import os
import json
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import folder_paths


class ArchAi3D_Save_Image:
    """
    Save an image with configurable format and workflow embedding options.

    Features:
    - Multiple formats: PNG, JPG, WebP
    - Quality control for JPG/WebP
    - Option to embed or not embed workflow metadata
    - Option to save workflow as separate JSON file
    - Web interface integration via output_name
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
                "format": (["PNG", "JPG", "WebP"], {
                    "default": "PNG",
                    "tooltip": "Image format (PNG supports metadata embedding)"
                }),
            },
            "optional": {
                "quality": ("INT", {
                    "default": 95,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Quality for JPG/WebP (1-100). For WebP, 100 = lossless"
                }),
                "embed_workflow": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Embed workflow metadata in image (PNG only)"
                }),
                "save_workflow_json": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Save workflow as separate .json file"
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
    FUNCTION = "save_images"  # Match standard SaveImage for proper history registration
    OUTPUT_NODE = True
    CATEGORY = "ArchAi3d/Inputs"

    def _get_next_counter(self, folder, filename_prefix, ext):
        """Find the next available counter to avoid overwriting files."""
        counter = 1
        while True:
            # Match standard SaveImage format: filename_00001_.ext
            test_file = os.path.join(folder, f"{filename_prefix}_{counter:05}_.{ext}")
            if not os.path.exists(test_file):
                return counter
            counter += 1

    def save_images(self, images, save=True, filename_prefix="ComfyUI", format="PNG",
                    quality=95, embed_workflow=True, save_workflow_json=False,
                    output_name="output_image", prompt=None, extra_pnginfo=None):
        """
        Save images with proper ComfyUI history registration.

        Returns {"ui": {"images": results}} for history registration.
        """
        # If save is False, return empty results (no saving)
        if not save:
            print("[ArchAi3D Save Image v3.0] save=False, skipping save")
            return { "ui": { "images": [] } }

        filename_prefix += self.prefix_append

        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
        )

        # Determine file extension
        ext_map = {"PNG": "png", "JPG": "jpg", "WebP": "webp"}
        ext = ext_map.get(format, "png")

        # Ensure we don't overwrite existing files
        counter = self._get_next_counter(full_output_folder, filename, ext)

        results = []

        for batch_number, image in enumerate(images):
            # Convert from tensor to numpy
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            # Generate filename with batch number if multiple images
            # Format matches standard SaveImage: filename_00001_.ext (trailing underscore)
            if len(images) > 1:
                file = f"{filename}_{batch_number:05}_{counter:05}_.{ext}"
            else:
                file = f"{filename}_{counter:05}_.{ext}"

            filepath = os.path.join(full_output_folder, file)

            # Extra safety: check if file exists and increment counter
            while os.path.exists(filepath):
                counter += 1
                if len(images) > 1:
                    file = f"{filename}_{batch_number:05}_{counter:05}_.{ext}"
                else:
                    file = f"{filename}_{counter:05}_.{ext}"
                filepath = os.path.join(full_output_folder, file)

            # Save based on format
            if format == "PNG":
                # Prepare metadata (only for PNG and if embed_workflow is True)
                metadata = None
                if embed_workflow:
                    metadata = PngInfo()
                    if prompt is not None:
                        metadata.add_text("prompt", json.dumps(prompt))
                    if extra_pnginfo is not None:
                        for x in extra_pnginfo:
                            metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                    metadata.add_text("output_name", output_name)

                img.save(filepath, pnginfo=metadata, compress_level=self.compress_level)

            elif format == "JPG":
                # JPG doesn't support alpha channel
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                img.save(filepath, quality=quality, optimize=True)

            elif format == "WebP":
                # WebP with quality (100 = lossless)
                lossless = (quality == 100)
                img.save(filepath, quality=quality, lossless=lossless)

            # Save workflow as separate JSON file if requested
            if save_workflow_json and extra_pnginfo:
                json_filepath = filepath.rsplit('.', 1)[0] + '.json'
                # Save the raw workflow (what ComfyUI expects when you drag & drop)
                workflow = extra_pnginfo.get("workflow")
                if workflow:
                    with open(json_filepath, 'w') as f:
                        json.dump(workflow, f, indent=2)

            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })

            counter += 1

        # Verbose debug output for API/history troubleshooting
        print(f"\n[ArchAi3D Save Image v3.0] Returning to ComfyUI history:")
        print(f"  Node output: {{'ui': {{'images': {len(results)} items}}}}")
        for i, r in enumerate(results):
            print(f"    [{i}] filename={r['filename']}, subfolder='{r['subfolder']}', type={r['type']}")

        # Return format MUST match standard SaveImage exactly for history registration
        return { "ui": { "images": results } }
