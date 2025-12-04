# -*- coding: utf-8 -*-
"""
ArchAi3D CivitAI Download Node
Author: Amir Ferdos (ArchAi3d)
Email: Amir84ferdos@gmail.com
LinkedIn: https://www.linkedin.com/in/archai3d/

Description:
    Download models from CivitAI with custom rename option.
    Solves the problem of badly named files on CivitAI.

Usage:
    1. Enter model_id (the number from CivitAI URL)
    2. Enter your CivitAI API token
    3. Optionally set custom_name to rename the downloaded file
    4. Select save directory
    5. Run the node

Version: 1.0.0
"""

import os
import requests
import shutil
from tqdm import tqdm
import re
import folder_paths

try:
    from server import PromptServer
    HAS_SERVER = True
except ImportError:
    HAS_SERVER = False


def get_model_dirs():
    """Get available model directories from ComfyUI."""
    try:
        models_dir = folder_paths.models_dir
        if os.path.exists(models_dir):
            dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
            return sorted(dirs) if dirs else ["checkpoints"]
        return ["checkpoints"]
    except Exception:
        return ["checkpoints", "loras", "vae", "clip", "unet", "diffusion_models"]


class ArchAi3D_CivitAI_Download:
    """Download models from CivitAI with custom rename option.

    Features:
    - Download models using CivitAI model ID
    - Rename files during download (fix bad filenames)
    - Progress indicator
    - Overwrite protection
    - API token support for restricted models
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": ("STRING", {
                    "multiline": False,
                    "default": "360292",
                    "tooltip": "CivitAI model version ID (number from URL after /models/xxx?modelVersionId=THIS)"
                }),
                "api_token": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "Your CivitAI API token (get from civitai.com/user/account)"
                }),
                "save_dir": (get_model_dirs(), {
                    "tooltip": "Directory to save the model (relative to ComfyUI/models/)"
                }),
            },
            "optional": {
                "custom_name": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "Custom filename for the downloaded file (leave empty to keep original name)"
                }),
                "overwrite": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Overwrite if file already exists"
                }),
                "save_dir_override": ("STRING", {
                    "default": "",
                    "tooltip": "Custom save path (overrides save_dir dropdown)"
                }),
            },
            "hidden": {
                "node_id": "UNIQUE_ID"
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "download"
    CATEGORY = "ArchAi3d/Download"
    OUTPUT_NODE = True

    def __init__(self):
        self.node_id = None
        self.progress = 0.0

    def set_progress(self, percentage):
        """Update download progress."""
        self.progress = percentage
        if HAS_SERVER and self.node_id:
            PromptServer.instance.send_sync("progress", {
                "node": self.node_id,
                "value": percentage,
                "max": 100
            })

    def _get_filename_from_response(self, response, url):
        """Extract filename from response headers."""
        cd = response.headers.get('content-disposition')
        if cd:
            # Try different patterns
            patterns = [
                r'filename="(.+)"',
                r"filename='(.+)'",
                r'filename=([^\s;]+)',
            ]
            for pattern in patterns:
                matches = re.findall(pattern, cd)
                if matches:
                    return matches[0].strip('"\'')
        return f"model_{url.split('/')[-1]}.safetensors"

    def download(self, model_id, api_token, save_dir, node_id=None,
                 custom_name="", overwrite=False, save_dir_override=""):
        """Download file from CivitAI."""

        self.node_id = node_id

        if not model_id:
            return ("ERROR: Missing model_id",)

        # Determine save path
        if save_dir_override:
            full_save_dir = save_dir_override
        else:
            full_save_dir = os.path.join(folder_paths.models_dir, save_dir)

        # Create directory if needed
        os.makedirs(full_save_dir, exist_ok=True)

        # Build URL
        url = f"https://civitai.com/api/download/models/{model_id}"
        params = {}
        if api_token:
            params['token'] = api_token

        print(f"[ArchAi3D CivitAI Download] Downloading model ID: {model_id}")

        try:
            # First request to get filename from headers
            response = requests.get(url, stream=True, params=params)
            response.raise_for_status()

            # Get original filename
            original_filename = self._get_filename_from_response(response, url)

            # Determine final filename
            if custom_name.strip():
                final_filename = custom_name.strip()
                # Ensure extension matches
                orig_ext = os.path.splitext(original_filename)[1]
                if orig_ext and not final_filename.endswith(orig_ext):
                    final_filename += orig_ext
            else:
                final_filename = original_filename

            full_path = os.path.join(full_save_dir, final_filename)

            print(f"[ArchAi3D CivitAI Download] Original name: {original_filename}")
            print(f"[ArchAi3D CivitAI Download] Saving as: {final_filename}")

            # Check if already exists
            if os.path.exists(full_path) and not overwrite:
                return (f"File already exists: {final_filename}\nEnable 'overwrite' to replace.",)

            total_size = int(response.headers.get('content-length', 0))
            temp_path = full_path + '.tmp'
            downloaded = 0

            with open(temp_path, 'wb') as f:
                with tqdm(total=total_size, unit='iB', unit_scale=True, desc=final_filename) as pbar:
                    for chunk in response.iter_content(chunk_size=1024*1024):
                        size = f.write(chunk)
                        downloaded += size
                        pbar.update(size)

                        if total_size > 0:
                            progress = (downloaded / total_size) * 100.0
                            self.set_progress(progress)

            # Move temp file to final location
            shutil.move(temp_path, full_path)

            # Get file size
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            size_str = f"{size_mb:.1f} MB" if size_mb < 1024 else f"{size_mb/1024:.2f} GB"

            status = f"✅ Download complete!\n\nFile: {final_filename}\nSize: {size_str}\nPath: {full_path}"
            if original_filename != final_filename:
                status += f"\n\n(Renamed from: {original_filename})"

            print(f"[ArchAi3D CivitAI Download] {status}")

            return (status,)

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                return ("❌ Unauthorized: Check your API token",)
            if e.response.status_code == 404:
                return (f"❌ Model not found: ID {model_id}",)
            return (f"❌ HTTP Error: {e}",)
        except Exception as e:
            # Clean up temp file
            temp_path = os.path.join(full_save_dir, "*.tmp")
            import glob
            for f in glob.glob(temp_path):
                os.remove(f)
            return (f"❌ Download failed: {str(e)}",)
