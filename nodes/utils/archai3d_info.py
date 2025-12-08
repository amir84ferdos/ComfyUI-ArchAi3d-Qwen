# ArchAi3D Info Node
#
# Displays version and commit information for the ArchAi3D node pack
#
# Author: Amir Ferdos (ArchAi3d)
# Version: 1.0.0
# License: Dual License (Free for personal use, Commercial license required for business use)

import os
import subprocess
from pathlib import Path


def get_git_info():
    """Get git commit hash and date from the node pack directory."""
    node_dir = Path(__file__).parent.parent.parent  # Go up to comfyui-archai3d-qwen

    try:
        # Get commit hash
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=str(node_dir),
            capture_output=True,
            text=True,
            timeout=5
        )
        commit_hash = result.stdout.strip() if result.returncode == 0 else "unknown"

        # Get full commit hash
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=str(node_dir),
            capture_output=True,
            text=True,
            timeout=5
        )
        full_hash = result.stdout.strip() if result.returncode == 0 else "unknown"

        # Get commit date
        result = subprocess.run(
            ['git', 'log', '-1', '--format=%ci'],
            cwd=str(node_dir),
            capture_output=True,
            text=True,
            timeout=5
        )
        commit_date = result.stdout.strip() if result.returncode == 0 else "unknown"

        # Get commit message (first line)
        result = subprocess.run(
            ['git', 'log', '-1', '--format=%s'],
            cwd=str(node_dir),
            capture_output=True,
            text=True,
            timeout=5
        )
        commit_msg = result.stdout.strip() if result.returncode == 0 else "unknown"

        # Get branch name
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            cwd=str(node_dir),
            capture_output=True,
            text=True,
            timeout=5
        )
        branch = result.stdout.strip() if result.returncode == 0 else "unknown"

        # Check if there are uncommitted changes
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            cwd=str(node_dir),
            capture_output=True,
            text=True,
            timeout=5
        )
        has_changes = len(result.stdout.strip()) > 0 if result.returncode == 0 else False
        dirty = " (modified)" if has_changes else ""

        return {
            "commit_short": commit_hash + dirty,
            "commit_full": full_hash,
            "commit_date": commit_date,
            "commit_msg": commit_msg,
            "branch": branch,
            "has_changes": has_changes
        }
    except Exception as e:
        return {
            "commit_short": f"error: {str(e)}",
            "commit_full": "unknown",
            "commit_date": "unknown",
            "commit_msg": "unknown",
            "branch": "unknown",
            "has_changes": False
        }


# Package version
PACKAGE_VERSION = "3.0.0"
PACKAGE_NAME = "ComfyUI-ArchAi3D-Qwen"


class ArchAi3D_Info:
    """
    Display version and commit information for ArchAi3D nodes.

    Use this node to verify you have the correct version installed.
    """

    CATEGORY = "ArchAi3d/Utils"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("version", "commit", "commit_date", "info")
    FUNCTION = "get_info"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "trigger": ("*", {"tooltip": "Connect anything here to trigger refresh"}),
            }
        }

    def get_info(self, trigger=None):
        """Get version and commit information."""
        git_info = get_git_info()

        version = PACKAGE_VERSION
        commit = git_info["commit_short"]
        commit_date = git_info["commit_date"]

        # Build detailed info string
        info_lines = [
            "=" * 50,
            f"ArchAi3D Node Pack v{PACKAGE_VERSION}",
            "=" * 50,
            "",
            f"Package: {PACKAGE_NAME}",
            f"Version: {version}",
            f"Branch: {git_info['branch']}",
            "",
            f"Commit: {git_info['commit_short']}",
            f"Full Hash: {git_info['commit_full']}",
            f"Date: {commit_date}",
            f"Message: {git_info['commit_msg']}",
            "",
            "=" * 50,
        ]

        if git_info["has_changes"]:
            info_lines.insert(-1, "⚠️ WARNING: Local modifications detected!")
            info_lines.insert(-1, "")

        info = "\n".join(info_lines)

        # Also print to console
        print(f"\n[ArchAi3D Info] v{version} | commit: {commit} | {commit_date}")

        return (version, commit, commit_date, info)


# ============================================================================
# NODE REGISTRATION
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "ArchAi3D_Info": ArchAi3D_Info,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_Info": "ℹ️ ArchAi3D Info",
}
