# -*- coding: utf-8 -*-
"""
ArchAi3D HuggingFace Download Node (High Speed)
Author: Amir Ferdos (ArchAi3d)
Email: Amir84ferdos@gmail.com
LinkedIn: https://www.linkedin.com/in/archai3d/

Description:
    Download models from HuggingFace with maximum speed using huggingface_hub.
    Features parallel downloads, resume support, and custom rename option.

Version: 2.0.0
"""

import os
import shutil
import folder_paths

try:
    from server import PromptServer
    HAS_SERVER = True
except ImportError:
    HAS_SERVER = False

# Try to import huggingface_hub for fast downloads
try:
    from huggingface_hub import hf_hub_download, snapshot_download
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False


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


class ArchAi3D_HF_Download:
    """Download models from HuggingFace with maximum speed.

    Features:
    - Uses huggingface_hub for optimized parallel downloads
    - Resume interrupted downloads
    - Custom rename option
    - HF token support for gated models
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_id": ("STRING", {
                    "multiline": False,
                    "default": "Comfy-Org/z_image_turbo",
                    "tooltip": "HuggingFace repository ID (e.g., 'username/model-name')"
                }),
                "filename": ("STRING", {
                    "multiline": False,
                    "default": "split_files/vae/ae.safetensors",
                    "tooltip": "Filename from the repository (can include subdirectories)"
                }),
                "save_dir": (get_model_dirs(), {
                    "tooltip": "Directory to save the model (relative to ComfyUI/models/)"
                }),
            },
            "optional": {
                "custom_name": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "Custom filename (leave empty to keep original name)"
                }),
                "hf_token": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "HuggingFace token for gated models (optional)"
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

    def download(self, repo_id, filename, save_dir, node_id=None,
                 custom_name="", hf_token="", overwrite=False, save_dir_override=""):
        """Download file from HuggingFace using huggingface_hub."""

        self.node_id = node_id

        if not repo_id or not filename:
            return ("ERROR: Missing repo_id or filename",)

        if not HAS_HF_HUB:
            return ("ERROR: huggingface_hub not installed.\nRun: pip install huggingface_hub",)

        # Determine save path
        if save_dir_override:
            full_save_dir = save_dir_override
        else:
            full_save_dir = os.path.join(folder_paths.models_dir, save_dir)

        # Extract just the filename (ignore any subdirectories in the HF path)
        base_filename = os.path.basename(filename)

        os.makedirs(full_save_dir, exist_ok=True)

        # Determine final filename
        final_filename = custom_name.strip() if custom_name.strip() else base_filename
        orig_ext = os.path.splitext(base_filename)[1]
        if custom_name.strip() and not final_filename.endswith(orig_ext):
            final_filename += orig_ext

        full_path = os.path.join(full_save_dir, final_filename)

        # Check if already exists
        if os.path.exists(full_path) and not overwrite:
            return (f"File already exists: {final_filename}\nEnable 'overwrite' to replace.",)

        print(f"[ArchAi3D HF Download] Repo: {repo_id}")
        print(f"[ArchAi3D HF Download] File: {filename}")
        print(f"[ArchAi3D HF Download] Saving as: {final_filename}")
        print(f"[ArchAi3D HF Download] Using huggingface_hub for maximum speed...")

        try:
            # Use huggingface_hub for optimized download
            token = hf_token.strip() if hf_token.strip() else None

            # Download to cache first (fast, parallel, resumable)
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                token=token,
                resume_download=True,
                force_download=overwrite,
            )

            # Copy/move to final destination with custom name
            if custom_name.strip():
                shutil.copy2(downloaded_path, full_path)
            else:
                # If no custom name, copy to destination
                shutil.copy2(downloaded_path, full_path)

            # Get file size
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            size_str = f"{size_mb:.1f} MB" if size_mb < 1024 else f"{size_mb/1024:.2f} GB"

            status = f"✅ Download complete!\n\nFile: {final_filename}\nSize: {size_str}\nPath: {full_path}"
            print(f"[ArchAi3D HF Download] {status}")

            return (status,)

        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "403" in error_msg:
                return (f"❌ Access denied. This may be a gated model.\nProvide HF token or accept license at: https://huggingface.co/{repo_id}",)
            if "404" in error_msg:
                return (f"❌ File not found: {filename}\nCheck repo_id and filename",)
            return (f"❌ Download failed: {error_msg}",)
