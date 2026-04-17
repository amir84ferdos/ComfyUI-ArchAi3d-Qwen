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
import threading
import time
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


# Module-level lock — serializes CivitAI downloads so concurrent or rapid-fire
# nodes don't trigger CivitAI rate limits (429/503). One download at a time.
_CIVITAI_DOWNLOAD_LOCK = threading.Lock()


def _request_with_retry(method, url, *, max_retries=4, backoff_base=2.0, **kwargs):
    """GET/HEAD with retry on 429/5xx/timeout. Returns final Response."""
    last_exc = None
    for attempt in range(max_retries):
        try:
            response = requests.request(method, url, **kwargs)
            if response.status_code in (429, 500, 502, 503, 504):
                # Respect Retry-After header if present
                retry_after = response.headers.get('Retry-After')
                wait = float(retry_after) if retry_after else backoff_base ** attempt
                print(f"[ArchAi3D CivitAI] Got {response.status_code}, retrying in {wait:.1f}s "
                      f"(attempt {attempt + 1}/{max_retries})")
                response.close()
                time.sleep(wait)
                continue
            return response
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_exc = e
            wait = backoff_base ** attempt
            print(f"[ArchAi3D CivitAI] {type(e).__name__}: retrying in {wait:.1f}s "
                  f"(attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)
    if last_exc:
        raise last_exc
    raise Exception(f"Failed after {max_retries} retries with 429/5xx responses")


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
        result = subprocess.run(['aria2c', '--version'], capture_output=True,
                                text=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
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
            '-x', str(connections),
            '-s', str(connections),
            '-k', '1M',
            '-o', os.path.basename(output_path),
            '-d', os.path.dirname(output_path),
            '--file-allocation=none',
            '--console-log-level=error',
            '--summary-interval=1',
            '--allow-overwrite=true',
            '--auto-file-renaming=false',
        ]

        if token:
            cmd.extend(['--header', f'Authorization: Bearer {token}'])

        cmd.append(url)

        print(f"[ArchAi3D CivitAI] Using aria2c with {connections} connections...")
        # 1 hour max — large models can take 10+ minutes on slow links
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        if result.returncode != 0:
            raise Exception(f"aria2c failed (exit {result.returncode}): {result.stderr}")
        if not os.path.exists(output_path):
            raise Exception("aria2c reported success but output file missing")
        if os.path.getsize(output_path) == 0:
            os.remove(output_path)
            raise Exception("aria2c produced 0-byte file (likely auth/redirect issue)")
        return output_path

    def _download_chunk(self, url, start, end, temp_file, chunk_id):
        """Download a chunk of the file. `url` is the resolved CDN URL (no auth needed)."""
        headers = {'Range': f'bytes={start}-{end}'}

        response = _request_with_retry('GET', url, headers=headers, stream=True, timeout=120)
        response.raise_for_status()

        written = 0
        try:
            with open(temp_file, 'r+b') as f:
                f.seek(start)
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        written += len(chunk)
        finally:
            response.close()

        expected = end - start + 1
        if written < expected:
            raise Exception(f"Chunk {chunk_id} short read: got {written}/{expected} bytes")
        return chunk_id

    def _resolve_download(self, url, token=None):
        """Resolve redirects and return (final_url, content_length).

        CivitAI's /api/download/models/{id}?token=XXX redirects to a presigned
        CDN URL that does NOT need Authorization. We follow the redirect once
        here so parallel chunks can hit the CDN directly without auth issues.
        """
        headers = {}
        if token:
            headers['Authorization'] = f'Bearer {token}'

        # Use streaming GET so we can see the final URL after redirects without
        # actually downloading the body. `with` guarantees the connection returns
        # to the pool even if we raise — otherwise the next run hangs on connection reuse.
        with requests.get(url, headers=headers, stream=True,
                          allow_redirects=True, timeout=60) as response:
            response.raise_for_status()
            final_url = response.url
            total_size = int(response.headers.get('content-length', 0))
            content_type = response.headers.get('content-type', '')

        # CivitAI returns text/html or application/json for auth errors
        if total_size == 0 or 'html' in content_type.lower() or \
                (content_type.startswith('application/json') and 'octet-stream' not in content_type):
            raise Exception(
                f"Server did not return a binary download (content-type: {content_type}, "
                f"length: {total_size}). Check your API token or model ID."
            )

        return final_url, total_size

    def _download_parallel(self, url, output_path, connections=16, token=None):
        """Download using parallel Range requests to the resolved CDN URL."""
        final_url, total_size = self._resolve_download(url, token)

        # Pre-allocate file (sparse is fine — chunks fill it)
        with open(output_path, 'wb') as f:
            f.seek(total_size - 1)
            f.write(b'\0')

        chunk_size = total_size // connections
        chunks = []
        for i in range(connections):
            start = i * chunk_size
            end = start + chunk_size - 1 if i < connections - 1 else total_size - 1
            chunks.append((start, end, i))

        print(f"[ArchAi3D CivitAI] Downloading {total_size / 1e6:.1f} MB "
              f"via {connections} parallel connections...")

        # Fail fast on any chunk error — otherwise we end up with a sparse/partial file
        downloaded_chunks = 0
        with ThreadPoolExecutor(max_workers=connections) as executor:
            futures = {
                executor.submit(
                    self._download_chunk, final_url, start, end, output_path, chunk_id
                ): chunk_id for start, end, chunk_id in chunks
            }

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    for f in futures:
                        f.cancel()
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    raise Exception(f"Parallel chunk failed: {e}")
                downloaded_chunks += 1
                self.set_progress((downloaded_chunks / connections) * 100)

        actual = os.path.getsize(output_path)
        if actual != total_size:
            os.remove(output_path)
            raise Exception(f"Size mismatch after parallel download: got {actual}, expected {total_size}")

        return output_path

    def _download_single(self, url, output_path, token=None):
        """Single connection streaming download with content-type and size verification."""
        headers = {}
        if token:
            headers['Authorization'] = f'Bearer {token}'

        with requests.get(url, headers=headers, stream=True,
                          allow_redirects=True, timeout=120) as response:
            response.raise_for_status()

            content_type = response.headers.get('content-type', '').lower()
            if 'html' in content_type or content_type.startswith('application/json'):
                raise Exception(
                    f"Server returned {content_type} instead of binary file — "
                    f"likely auth or model-ID error."
                )

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            self.set_progress((downloaded / total_size) * 100)

        actual = os.path.getsize(output_path)
        if actual == 0:
            os.remove(output_path)
            raise Exception("Download produced 0-byte file")
        if total_size > 0 and actual != total_size:
            os.remove(output_path)
            raise Exception(f"Size mismatch: got {actual}, expected {total_size}")
        return output_path

    def download(self, model_id, api_token, save_dir, node_id=None,
                 custom_name="", overwrite=False, save_dir_override="", connections=16):
        """Download file from CivitAI.

        Serialized via module-level lock so multiple CivitAI Download nodes in
        the same workflow run one at a time. Prevents CivitAI rate-limiting
        when the workflow has several downloads queued back-to-back.
        """
        with _CIVITAI_DOWNLOAD_LOCK:
            return self._download_impl(model_id, api_token, save_dir, node_id,
                                       custom_name, overwrite, save_dir_override, connections)

    def _download_impl(self, model_id, api_token, save_dir, node_id=None,
                       custom_name="", overwrite=False, save_dir_override="", connections=16):
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
            # First get the filename from headers.
            # Use GET with stream=True instead of HEAD — CivitAI is more reliable
            # with GET, and closing the response before reading the body avoids
            # downloading content. Always use a timeout so we never hang.
            headers = {}
            if token:
                headers['Authorization'] = f'Bearer {token}'

            # Retry through _request_with_retry for 429/5xx resilience
            with _request_with_retry(
                'GET', url, headers=headers, stream=True, allow_redirects=True,
                params={'token': token} if token else None, timeout=60,
            ) as response:
                if response.status_code == 401:
                    msg = "❌ Unauthorized: Check your API token"
                    print(f"[ArchAi3D CivitAI Download] {msg}")
                    return (msg,)
                if response.status_code == 404:
                    msg = (f"❌ Model not found (404): version ID '{model_id}' does not exist "
                           f"or is not downloadable. Use the modelVersionId from the CivitAI URL.")
                    print(f"[ArchAi3D CivitAI Download] {msg}")
                    return (msg,)
                response.raise_for_status()
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
                    try:
                        self._download_parallel(download_url, full_path, connections, token)
                    except Exception as e2:
                        print(f"[ArchAi3D CivitAI] parallel failed, trying single: {e2}")
                        self._download_single(download_url, full_path, token)
            else:
                try:
                    self._download_parallel(download_url, full_path, connections, token)
                except Exception as e:
                    print(f"[ArchAi3D CivitAI] parallel failed, trying single: {e}")
                    self._download_single(download_url, full_path, token)

            # Final verification — never let a 0-byte file through
            if not os.path.exists(full_path):
                return ("❌ Download failed: output file not created",)
            if os.path.getsize(full_path) == 0:
                os.remove(full_path)
                return ("❌ Download failed: 0-byte file (check API token and model ID)",)

            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            size_str = f"{size_mb:.1f} MB" if size_mb < 1024 else f"{size_mb/1024:.2f} GB"

            status = f"✅ Download complete!\n\nFile: {final_filename}\nSize: {size_str}\nPath: {full_path}"
            if original_filename != final_filename:
                status += f"\n\n(Renamed from: {original_filename})"

            print(f"[ArchAi3D CivitAI Download] {status}")
            return (status,)

        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            if code == 401:
                msg = "❌ Unauthorized: Check your API token"
            elif code == 404:
                msg = f"❌ Model not found: ID {model_id}"
            else:
                msg = f"❌ HTTP Error: {e}"
            print(f"[ArchAi3D CivitAI Download] {msg}")
            return (msg,)
        except Exception as e:
            msg = f"❌ Download failed: {type(e).__name__}: {str(e)}"
            print(f"[ArchAi3D CivitAI Download] {msg}")
            return (msg,)
