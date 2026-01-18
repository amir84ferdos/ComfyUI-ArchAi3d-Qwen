#!/usr/bin/env python3
"""
ArchAi3D Qwen - ComfyUI Custom Nodes Installer

This script is automatically run by ComfyUI Manager when installing/updating the node.
It ensures all dependencies are properly installed.

Author: Amir Ferdos (ArchAi3d)
"""

import subprocess
import sys
import os

def pip_install(package, extra_args=None):
    """Install a package using pip.

    Uses --no-cache-dir for cloud environments (RunPod, etc.) to avoid disk space issues.
    """
    cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir", package]
    if extra_args:
        cmd.extend(extra_args)
    try:
        subprocess.check_call(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ArchAi3D] Warning: Failed to install {package}: {e}")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    import_name = import_name or package_name
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False

def main():
    print("=" * 60)
    print("[ArchAi3D Qwen] Installing dependencies...")
    print("=" * 60)

    # Core dependencies (usually already installed with ComfyUI)
    core_deps = [
        ("numpy", "numpy"),
        ("Pillow", "PIL"),
        ("requests", "requests"),
        ("PyYAML", "yaml"),
    ]

    # Optional but recommended dependencies
    optional_deps = [
        # For Gemini API
        ("google-genai", "google.genai", "Gemini API"),
        # For Google Drive downloads
        ("gdown", "gdown", "Google Drive downloads"),
        # For fast HuggingFace downloads
        ("hf_transfer", "hf_transfer", "Fast HF downloads"),
        ("huggingface_hub", "huggingface_hub", "HuggingFace Hub"),
    ]

    # Install core dependencies
    print("\n[1/3] Checking core dependencies...")
    for package, import_name in core_deps:
        if not check_package(package, import_name):
            print(f"  Installing {package}...")
            pip_install(package)
        else:
            print(f"  {package} - OK")

    # Install optional dependencies
    print("\n[2/3] Installing optional dependencies...")
    for item in optional_deps:
        package, import_name, description = item
        if not check_package(package, import_name):
            print(f"  Installing {package} ({description})...")
            pip_install(package)
        else:
            print(f"  {package} ({description}) - OK")

    # Special handling for SPZ support (3D Gaussian Splat compression)
    print("\n[3/3] Setting up SPZ support (for SaveSplatScene node)...")

    # Try Niantic SPZ library first (C++ - fastest, ~1-5 seconds)
    spz_installed = False
    try:
        import spz
        print("  Niantic SPZ library - OK (fastest C++ converter)")
        spz_installed = True
    except ImportError:
        print("  Installing Niantic SPZ library (C++ - fastest)...")
        # Niantic SPZ requires C++ compiler. On RunPod/cloud, this is usually available.
        success = pip_install("git+https://github.com/nianticlabs/spz.git")
        if success:
            # Verify it actually works
            try:
                import spz
                print("  Niantic SPZ library installed successfully!")
                spz_installed = True
            except ImportError:
                print("  Niantic SPZ compiled but import failed")
        else:
            print("  Niantic SPZ library not installed (requires C++ compiler)")

    # Install gsconverter as fallback (Python/Taichi - slower but always works)
    gsconverter_installed = False
    if check_package("gsconverter", "gsconverter"):
        print("  gsconverter - OK (Python fallback)")
        gsconverter_installed = True
    else:
        print("  Installing gsconverter (Python fallback)...")
        success = pip_install("git+https://github.com/francescofugazzi/3dgsconverter.git")
        if success:
            print("  gsconverter installed successfully!")
            gsconverter_installed = True
        else:
            print("  Warning: gsconverter installation failed.")

    # Summary
    if spz_installed:
        print("  SPZ support: Niantic C++ (fast, ~1-5 seconds)")
    elif gsconverter_installed:
        print("  SPZ support: gsconverter Python (slower, ~30-100 seconds)")
    else:
        print("  Warning: No SPZ converter installed!")
        print("  SaveSplatScene node will only output PLY format")
        print("  To fix, manually run:")
        print(f"    {sys.executable} -m pip install git+https://github.com/nianticlabs/spz.git")

    print("\n" + "=" * 60)
    print("[ArchAi3D Qwen] Installation complete!")
    print("=" * 60)
    print("\nPlease restart ComfyUI to load the new nodes.")

if __name__ == "__main__":
    main()
