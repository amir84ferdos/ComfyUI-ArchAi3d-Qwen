# -*- coding: utf-8 -*-
"""
ArchAi3D Google Drive Download Node
Author: Amir Ferdos (ArchAi3d)
Email: Amir84ferdos@gmail.com
LinkedIn: https://www.linkedin.com/in/archai3d/

Description:
    Download models from Google Drive with custom rename option.
    Supports both public shared links and file IDs.
    Uses gdown library for reliable large file downloads.

Usage:
    1. Enter Google Drive file ID or share link
    2. Set custom filename
    3. Select save directory
    4. Run the node

Version: 2.0.0 (High Speed)
"""

import os
import shutil
import re
import folder_paths

try:
    from server import PromptServer
    HAS_SERVER = True
except ImportError:
    HAS_SERVER = False

# Try to import gdown
try:
    import gdown
    HAS_GDOWN = True
except ImportError:
    HAS_GDOWN = False


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


class ArchAi3D_GDrive_Download:
    """Download models from Google Drive with custom rename option.

    Features:
    - Download from Google Drive using file ID or share link
    - Rename files during download
    - Progress indicator
    - Overwrite protection
    - Handles large file confirmation automatically via gdown
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_id_or_link": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "Google Drive file ID or share link (e.g., '1ABC...xyz' or 'https://drive.google.com/file/d/1ABC...xyz/view')"
                }),
                "filename": ("STRING", {
                    "multiline": False,
                    "default": "model.safetensors",
                    "tooltip": "Filename to save as (Google Drive doesn't always provide filename)"
                }),
                "save_dir": (get_model_dirs(), {
                    "tooltip": "Directory to save the model (relative to ComfyUI/models/)"
                }),
            },
            "optional": {
                "overwrite": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Overwrite if file already exists"
                }),
                "save_dir_override": ("STRING", {
                    "default": "",
                    "tooltip": "Custom save path (overrides save_dir dropdown)"
                }),
                "fuzzy": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use fuzzy matching for file extraction (recommended)"
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

    def _extract_file_id(self, file_id_or_link):
        """Extract file ID from various Google Drive URL formats."""
        # If it's already just an ID (no slashes, no http)
        if '/' not in file_id_or_link and 'http' not in file_id_or_link:
            return file_id_or_link.strip()

        # Try various URL patterns
        patterns = [
            r'/file/d/([a-zA-Z0-9_-]+)',  # /file/d/FILE_ID/
            r'id=([a-zA-Z0-9_-]+)',        # ?id=FILE_ID
            r'/d/([a-zA-Z0-9_-]+)',        # /d/FILE_ID
            r'open\?id=([a-zA-Z0-9_-]+)',  # open?id=FILE_ID
        ]

        for pattern in patterns:
            match = re.search(pattern, file_id_or_link)
            if match:
                return match.group(1)

        # Last resort: return as-is (might be just the ID)
        return file_id_or_link.strip()

    def _download_with_gdown(self, file_id, output_path, fuzzy=True):
        """Download using gdown library."""
        url = f"https://drive.google.com/uc?id={file_id}"

        print(f"[ArchAi3D GDrive Download] Using gdown to download...")
        print(f"[ArchAi3D GDrive Download] URL: {url}")

        # gdown handles large file confirmation automatically
        result = gdown.download(url, output_path, quiet=False, fuzzy=fuzzy)

        return result

    def _download_with_requests(self, file_id, output_path):
        """Fallback download using requests with multiple confirmation methods."""
        import requests
        from tqdm import tqdm

        session = requests.Session()

        # Try multiple download approaches
        urls_to_try = [
            f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t",
            f"https://drive.google.com/uc?export=download&id={file_id}&confirm=yes",
            f"https://drive.google.com/uc?export=download&id={file_id}&confirm=1",
            f"https://drive.usercontent.google.com/download?id={file_id}&confirm=t",
        ]

        for url in urls_to_try:
            print(f"[ArchAi3D GDrive Download] Trying: {url}")

            try:
                response = session.get(url, stream=True, allow_redirects=True)
                response.raise_for_status()

                content_type = response.headers.get('Content-Type', '')
                content_length = int(response.headers.get('content-length', 0))

                # Check if we got actual file content (not HTML error page)
                if 'text/html' not in content_type or content_length > 100000:
                    # This looks like the actual file
                    temp_path = output_path + '.tmp'
                    downloaded = 0

                    with open(temp_path, 'wb') as f:
                        with tqdm(total=content_length, unit='iB', unit_scale=True) as pbar:
                            for chunk in response.iter_content(chunk_size=1024*1024):
                                if chunk:
                                    size = f.write(chunk)
                                    downloaded += size
                                    pbar.update(size)

                                    if content_length > 0:
                                        progress = (downloaded / content_length) * 100.0
                                        self.set_progress(progress)

                    # Move to final location
                    shutil.move(temp_path, output_path)
                    return output_path

            except Exception as e:
                print(f"[ArchAi3D GDrive Download] Failed with this URL: {e}")
                continue

        return None

    def download(self, file_id_or_link, filename, save_dir, node_id=None,
                 overwrite=False, save_dir_override="", fuzzy=True):
        """Download file from Google Drive."""

        self.node_id = node_id

        if not file_id_or_link:
            return ("ERROR: Missing file ID or link",)

        if not filename:
            return ("ERROR: Filename is required for Google Drive downloads",)

        # Extract file ID
        file_id = self._extract_file_id(file_id_or_link)

        if not file_id:
            return ("ERROR: Could not extract file ID from input",)

        # Determine save path
        if save_dir_override:
            full_save_dir = save_dir_override
        else:
            full_save_dir = os.path.join(folder_paths.models_dir, save_dir)

        # Create directory if needed
        os.makedirs(full_save_dir, exist_ok=True)

        full_path = os.path.join(full_save_dir, filename)

        # Check if already exists
        if os.path.exists(full_path) and not overwrite:
            return (f"File already exists: {filename}\nEnable 'overwrite' to replace.",)

        print(f"[ArchAi3D GDrive Download] File ID: {file_id}")
        print(f"[ArchAi3D GDrive Download] Saving as: {filename}")

        try:
            result = None

            # Try gdown first if available
            if HAS_GDOWN:
                try:
                    result = self._download_with_gdown(file_id, full_path, fuzzy)
                except Exception as e:
                    print(f"[ArchAi3D GDrive Download] gdown failed: {e}")
                    print("[ArchAi3D GDrive Download] Falling back to requests...")

            # Fallback to requests if gdown failed or not available
            if not result or not os.path.exists(full_path):
                if not HAS_GDOWN:
                    print("[ArchAi3D GDrive Download] gdown not installed, using requests...")
                    print("[ArchAi3D GDrive Download] For better reliability, install: pip install gdown")

                result = self._download_with_requests(file_id, full_path)

            # Check if download succeeded
            if result and os.path.exists(full_path):
                size_mb = os.path.getsize(full_path) / (1024 * 1024)
                size_str = f"{size_mb:.1f} MB" if size_mb < 1024 else f"{size_mb/1024:.2f} GB"

                status = f"✅ Download complete!\n\nFile: {filename}\nSize: {size_str}\nPath: {full_path}"
                print(f"[ArchAi3D GDrive Download] {status}")
                return (status,)
            else:
                return ("❌ Download failed. Please check:\n"
                        "1. File is set to 'Anyone with the link can view'\n"
                        "2. Link/ID is correct\n"
                        "3. Try installing gdown: pip install gdown",)

        except Exception as e:
            # Clean up temp file
            temp_path = full_path + '.tmp'
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return (f"❌ Download failed: {str(e)}\n\nTip: Install gdown for better reliability:\npip install gdown",)
