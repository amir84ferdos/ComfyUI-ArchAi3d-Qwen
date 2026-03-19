"""
ArchAi3D Save Image Node
Saves an image with format options and workflow embedding control.

Version: 3.1.0 - Added advanced compression settings
- WebP: method (0-6), lossless, exact, OpenCV fast path
- PNG: compress_level (0-9)
- JPG: optimize, subsampling
- All new options at end of optional inputs (backwards compatible)

Version: 3.0.0 - Fixed API/history registration for RunPod compatibility
- Changed FUNCTION to "save_images" to match standard SaveImage
- Fixed filename format with trailing underscore
- Added verbose debug logging for API troubleshooting
"""

import os
import json
import time
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
                # --- Advanced WebP Settings ---
                "webp_method": (["fastest (0)", "fast (1)", "medium (2)", "default (4)", "best (6)"], {
                    "default": "default (4)",
                    "tooltip": "WebP compression effort. 0=fastest encoding, 6=best compression/slowest. Only affects WebP.",
                }),
                "webp_lossless": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Force lossless WebP (overrides quality). Perfect quality but slower and larger files.",
                }),
                "webp_exact": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Preserve transparent area RGB values in WebP. Slightly larger files.",
                }),
                # --- Advanced PNG Settings ---
                "png_compress_level": ("INT", {
                    "default": 4,
                    "min": 0, "max": 9, "step": 1,
                    "tooltip": "PNG compression level. 0=no compression (fastest/largest), 9=max compression (slowest/smallest).",
                }),
                # --- Advanced JPG Settings ---
                "jpg_optimize": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "JPG optimize flag. Slightly slower but better compression.",
                }),
                "jpg_subsampling": (["4:2:0 (default)", "4:2:2", "4:4:4 (best quality)"], {
                    "default": "4:2:0 (default)",
                    "tooltip": "JPG chroma subsampling. 4:4:4=best quality/larger, 4:2:0=smaller/default.",
                }),
                # --- Performance ---
                "use_opencv": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use OpenCV for WebP saving (5-10x faster on 4K+ images). Falls back to Pillow if unavailable.",
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
    CATEGORY = "ArchAi3d/Image"

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
                    output_name="output_image",
                    webp_method="default (4)", webp_lossless=False, webp_exact=False,
                    png_compress_level=4, jpg_optimize=True,
                    jpg_subsampling="4:2:0 (default)", use_opencv=False,
                    prompt=None, extra_pnginfo=None):
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
            t_save_start = time.perf_counter()
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

                img.save(filepath, pnginfo=metadata, compress_level=png_compress_level)

            elif format == "JPG":
                # JPG doesn't support alpha channel
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                # Parse subsampling: "4:2:0 (default)" -> 2, "4:2:2" -> 1, "4:4:4 ..." -> 0
                subsample_map = {"4:2:0 (default)": 2, "4:2:2": 1, "4:4:4 (best quality)": 0}
                subsample_val = subsample_map.get(jpg_subsampling, 2)
                img.save(filepath, quality=quality, optimize=jpg_optimize,
                         subsampling=subsample_val)

            elif format == "WebP":
                # Parse method: "fastest (0)" -> 0, "default (4)" -> 4, etc.
                method_map = {"fastest (0)": 0, "fast (1)": 1, "medium (2)": 2,
                              "default (4)": 4, "best (6)": 6}
                method_val = method_map.get(webp_method, 4)
                lossless = webp_lossless or (quality == 100)
                save_backend = "Pillow"

                if use_opencv and not lossless:
                    # OpenCV fast path — go straight from tensor numpy, skip PIL round-trip
                    try:
                        import cv2
                        cv_img = np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8)
                        # Tensor is RGB, OpenCV needs BGR
                        cv_img = cv_img[:, :, ::-1].copy()
                        cv2.imwrite(filepath, cv_img, [cv2.IMWRITE_WEBP_QUALITY, quality])
                        save_backend = "OpenCV"
                    except ImportError:
                        print("[ArchAi3D Save Image] OpenCV not available, falling back to Pillow")
                        img.save(filepath, quality=quality, lossless=False,
                                 method=method_val, exact=webp_exact)
                else:
                    img.save(filepath, quality=quality, lossless=lossless,
                             method=method_val, exact=webp_exact)

            t_save_end = time.perf_counter()
            save_ms = (t_save_end - t_save_start) * 1000
            backend_info = ""
            if format == "WebP":
                backend_info = f" [{save_backend}]" if 'save_backend' in dir() else ""
                backend_info = f" [{save_backend}, method={method_val}, q={quality}{'  LOSSLESS' if lossless else ''}]"
            elif format == "PNG":
                backend_info = f" [compress={png_compress_level}]"
            elif format == "JPG":
                backend_info = f" [q={quality}, sub={jpg_subsampling}]"
            print(f"[ArchAi3D Save Image] {format} {img.size[0]}x{img.size[1]} saved in {save_ms:.0f}ms{backend_info}")

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
