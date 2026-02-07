"""
Local Model Cache — RunPod SSD Optimization

On RunPod, /workspace/ is a slow network FUSE mount. This module copies model
files to the local SSD (/tmp/) for faster loading. On local PCs, this is a
complete no-op.

Author: Amir Ferdos (ArchAi3d)
"""

import os
import shutil
import time

LOCAL_CACHE_DIR = "/tmp/comfyui-local-models"


def copy_to_local(source_path, enabled=True):
    """Copy a model file from network to local SSD if on RunPod.

    Returns local path if copied, or original path if:
    - enabled=False (user disabled the toggle)
    - Not on RunPod (path doesn't start with /workspace/)
    - Already cached locally
    - Not enough disk space

    Args:
        source_path: Full path to the model file
        enabled: Whether the feature is enabled (from node toggle)

    Returns:
        Path to use for loading (local cache or original)
    """
    if not enabled:
        return source_path

    if not os.path.isfile(source_path):
        return source_path

    if not source_path.startswith("/workspace/"):
        return source_path  # Not on RunPod network drive, nothing to do

    # Build local cache path preserving directory structure
    rel_path = os.path.relpath(source_path, "/workspace/")
    local_path = os.path.join(LOCAL_CACHE_DIR, rel_path)

    # Already cached? (check by file size)
    if os.path.isfile(local_path):
        if os.path.getsize(source_path) == os.path.getsize(local_path):
            print(f"[LocalCache] Using cached: {os.path.basename(local_path)}")
            return local_path

    # Check free disk space (keep 20GB headroom)
    try:
        stat = os.statvfs("/tmp")
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        file_gb = os.path.getsize(source_path) / (1024**3)
        if free_gb - file_gb < 20:
            print(f"[LocalCache] Not enough space ({free_gb:.1f}GB free, need {file_gb:.1f}GB + 20GB headroom), loading from network")
            return source_path
    except OSError:
        return source_path

    # Copy to local SSD
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        filename = os.path.basename(source_path)
        print(f"[LocalCache] Copying {filename} ({file_gb:.1f}GB) to local SSD...")
        start = time.time()
        shutil.copy2(source_path, local_path)
        elapsed = time.time() - start
        speed = file_gb / elapsed * 1024 if elapsed > 0 else 0
        print(f"[LocalCache] Done in {elapsed:.1f}s ({speed:.0f} MB/s)")
        return local_path
    except (OSError, IOError) as e:
        print(f"[LocalCache] Copy failed: {e}, loading from network")
        # Clean up partial copy
        if os.path.isfile(local_path):
            try:
                os.remove(local_path)
            except OSError:
                pass
        return source_path
