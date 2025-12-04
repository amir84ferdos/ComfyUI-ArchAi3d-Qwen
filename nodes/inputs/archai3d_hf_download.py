# -*- coding: utf-8 -*-
"""
ArchAi3D HuggingFace Download Node
Author: Amir Ferdos (ArchAi3d)
Email: Amir84ferdos@gmail.com
LinkedIn: https://www.linkedin.com/in/archai3d/

Description:
    Download models from HuggingFace with custom rename option.
    Solves the problem of badly named files on HuggingFace.

Usage:
    1. Enter repo_id (e.g., "Qwen/Qwen3-VL-4B-Instruct-GGUF")
    2. Enter filename from repo (e.g., "Qwen3VL-4B-Instruct-Q4_K_M.gguf")
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
        # Get ComfyUI models directory
        models_dir = folder_paths.models_dir
        if os.path.exists(models_dir):
            dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
            return sorted(dirs) if dirs else ["checkpoints"]
        return ["checkpoints"]
    except Exception:
        return ["checkpoints", "loras", "vae", "clip", "unet", "diffusion_models"]


class ArchAi3D_HF_Download:
    """Download models from HuggingFace with custom rename option.

    Features:
    - Download any file from HuggingFace repos
    - Rename files during download (fix bad filenames)
    - Progress indicator
    - Overwrite protection
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_id": ("STRING", {
                    "multiline": False,
                    "default": "Qwen/Qwen3-VL-4B-Instruct-GGUF",
                    "tooltip": "HuggingFace repository ID (e.g., 'username/model-name')"
                }),
                "filename": ("STRING", {
                    "multiline": False,
                    "default": "Qwen3VL-4B-Instruct-Q4_K_M.gguf",
                    "tooltip": "Exact filename from the repository"
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

    def download(self, repo_id, filename, save_dir, node_id=None,
                 custom_name="", overwrite=False, save_dir_override=""):
        """Download file from HuggingFace."""

        self.node_id = node_id

        if not repo_id or not filename:
            return ("ERROR: Missing repo_id or filename",)

        # Determine save path
        if save_dir_override:
            full_save_dir = save_dir_override
        else:
            full_save_dir = os.path.join(folder_paths.models_dir, save_dir)

        # Create directory if needed
        os.makedirs(full_save_dir, exist_ok=True)

        # Determine final filename
        final_filename = custom_name.strip() if custom_name.strip() else filename

        # Ensure extension matches
        orig_ext = os.path.splitext(filename)[1]
        if custom_name.strip() and not final_filename.endswith(orig_ext):
            final_filename += orig_ext

        full_path = os.path.join(full_save_dir, final_filename)

        # Check if already exists
        if os.path.exists(full_path) and not overwrite:
            return (f"File already exists: {final_filename}\nEnable 'overwrite' to replace.",)

        # Build URL
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"

        print(f"[ArchAi3D HF Download] Downloading: {url}")
        print(f"[ArchAi3D HF Download] Saving as: {final_filename}")

        try:
            # Download with progress
            response = requests.get(url, stream=True)
            response.raise_for_status()

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
            print(f"[ArchAi3D HF Download] {status}")

            return (status,)

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return (f"❌ File not found: {filename}\nCheck repo_id and filename",)
            return (f"❌ HTTP Error: {e}",)
        except Exception as e:
            # Clean up temp file
            temp_path = full_path + '.tmp'
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return (f"❌ Download failed: {str(e)}",)
