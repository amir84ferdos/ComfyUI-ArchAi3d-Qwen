# -*- coding: utf-8 -*-
"""
ArchAi3D CivitAI Download Node (High Speed)
Author: Amir Ferdos (ArchAi3d)
Email: Amir84ferdos@gmail.com
LinkedIn: https://www.linkedin.com/in/archai3d/

Description:
    Download models from CivitAI with maximum speed using parallel connections.
    Features custom rename option and progress indicator.

Version: 2.0.0
"""

import os
import requests
import shutil
from tqdm import tqdm
import re
import subprocess
import folder_paths
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def check_aria2():
    """Check if aria2c is available."""
    try:
        result = subprocess.run(['aria2c', '--version'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


HAS_ARIA2 = check_aria2()


class ArchAi3D_CivitAI_Download:
    """Download models from CivitAI with maximum speed.

    Features:
    - Multi-connection parallel download (16 connections)
    - Uses aria2c if available for fastest speeds
    - Rename files during download
    - Progress indicator
    - API token support
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": ("STRING", {
                    "multiline": False,
                    "default": "360292",
                    "tooltip": "CivitAI model version ID (number from URL)"
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
                    "tooltip": "Custom filename (leave empty to keep original name)"
                }),
                "overwrite": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Overwrite if file already exists"
                }),
                "save_dir_override": ("STRING", {
                    "default": "",
                    "tooltip": "Custom save path (overrides save_dir dropdown)"
                }),
                "connections": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 64,
                    "tooltip": "Number of parallel connections (more = faster, default 16)"
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

    def _download_with_aria2(self, url, output_path, connections=16, token=None):
        """Download using aria2c for maximum speed."""
        cmd = [
            'aria2c',
            '-x', str(connections),  # Max connections per server
            '-s', str(connections),  # Split file into N parts
            '-k', '1M',              # Min split size
            '-o', os.path.basename(output_path),
            '-d', os.path.dirname(output_path),
            '--file-allocation=none',
            '--console-log-level=error',
            '--summary-interval=1',
        ]

        if token:
            cmd.extend(['--header', f'Authorization: Bearer {token}'])

        cmd.append(url)

        print(f"[ArchAi3D CivitAI] Using aria2c with {connections} connections...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0 and os.path.exists(output_path):
            return output_path
        else:
            raise Exception(f"aria2c failed: {result.stderr}")

    def _download_chunk(self, url, start, end, temp_file, chunk_id, token=None):
        """Download a chunk of the file."""
        headers = {'Range': f'bytes={start}-{end}'}
        if token:
            headers['Authorization'] = f'Bearer {token}'

        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()

        with open(temp_file, 'r+b') as f:
            f.seek(start)
            for chunk in response.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)

        return chunk_id

    def _download_parallel(self, url, output_path, connections=16, token=None):
        """Download using parallel connections."""
        # Get file size
        headers = {}
        if token:
            headers['Authorization'] = f'Bearer {token}'

        response = requests.head(url, headers=headers, allow_redirects=True)
        total_size = int(response.headers.get('content-length', 0))

        if total_size == 0:
            # Fallback to single connection
            return self._download_single(url, output_path, token)

        # Create empty file
        with open(output_path, 'wb') as f:
            f.seek(total_size - 1)
            f.write(b'\0')

        # Calculate chunk sizes
        chunk_size = total_size // connections
        chunks = []
        for i in range(connections):
            start = i * chunk_size
            end = start + chunk_size - 1 if i < connections - 1 else total_size - 1
            chunks.append((start, end, i))

        print(f"[ArchAi3D CivitAI] Downloading with {connections} parallel connections...")

        # Download chunks in parallel
        downloaded_chunks = 0
        with ThreadPoolExecutor(max_workers=connections) as executor:
            futures = {
                executor.submit(
                    self._download_chunk, url, start, end, output_path, chunk_id, token
                ): chunk_id for start, end, chunk_id in chunks
            }

            for future in as_completed(futures):
                downloaded_chunks += 1
                progress = (downloaded_chunks / connections) * 100
                self.set_progress(progress)

        return output_path

    def _download_single(self, url, output_path, token=None):
        """Single connection download fallback."""
        headers = {}
        if token:
            headers['Authorization'] = f'Bearer {token}'

        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        self.set_progress((downloaded / total_size) * 100)

        return output_path

    def download(self, model_id, api_token, save_dir, node_id=None,
                 custom_name="", overwrite=False, save_dir_override="", connections=16):
        """Download file from CivitAI."""

        self.node_id = node_id

        if not model_id:
            return ("ERROR: Missing model_id",)

        # Determine save path
        if save_dir_override:
            full_save_dir = save_dir_override
        else:
            full_save_dir = os.path.join(folder_paths.models_dir, save_dir)

        os.makedirs(full_save_dir, exist_ok=True)

        # Build URL
        url = f"https://civitai.com/api/download/models/{model_id}"
        token = api_token.strip() if api_token.strip() else None

        print(f"[ArchAi3D CivitAI Download] Model ID: {model_id}")

        try:
            # First get the filename from headers
            headers = {}
            if token:
                headers['Authorization'] = f'Bearer {token}'

            response = requests.head(url, headers=headers, allow_redirects=True, params={'token': token} if token else None)

            # Get filename
            original_filename = self._get_filename_from_response(response, url)

            # Determine final filename
            if custom_name.strip():
                final_filename = custom_name.strip()
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

            # Download URL with token
            download_url = f"{url}?token={token}" if token else url

            # Try aria2 first (fastest), then parallel, then single
            if HAS_ARIA2:
                try:
                    self._download_with_aria2(download_url, full_path, connections, token)
                except Exception as e:
                    print(f"[ArchAi3D CivitAI] aria2c failed, trying parallel: {e}")
                    self._download_parallel(download_url, full_path, connections, token)
            else:
                self._download_parallel(download_url, full_path, connections, token)

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
            return (f"❌ Download failed: {str(e)}",)
