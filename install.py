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
import platform

def install_system_deps():
    """Try to install system dependencies required for SPZ compilation."""
    if platform.system() != "Linux":
        return False

    print("  Checking system dependencies for SPZ...")

    # Check if zlib is already available
    try:
        result = subprocess.run(
            ["pkg-config", "--exists", "zlib"],
            capture_output=True, timeout=10
        )
        if result.returncode == 0:
            print("  ZLIB development headers - OK")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Try to install zlib1g-dev
    print("  Installing ZLIB development headers (required for SPZ)...")

    # Try different methods
    apt_cmds = [
        ["sudo", "apt-get", "install", "-y", "zlib1g-dev"],  # With sudo
        ["apt-get", "install", "-y", "zlib1g-dev"],          # As root (RunPod)
    ]

    for cmd in apt_cmds:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=120,
                text=True
            )
            if result.returncode == 0:
                print("  ZLIB installed successfully!")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
            continue

    print("  Warning: Could not install ZLIB (may need root/sudo)")
    return False

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
    print("  Note: Built-in SPZ v3 converter always available as fallback")

    # Try Niantic SPZ library first (C++ - fastest, ~1-5 seconds)
    spz_installed = False
    try:
        import spz
        print("  Niantic SPZ library - OK (fastest C++ converter)")
        spz_installed = True
    except ImportError:
        # First, try to install system dependencies (ZLIB)
        install_system_deps()

        print("  Installing Niantic SPZ library (C++ - fastest)...")
        # Niantic SPZ requires C++ compiler and ZLIB
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
            print("  Niantic SPZ library not installed (requires C++ compiler + ZLIB)")

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
    print("\n  SPZ converter priority:")
    if spz_installed:
        print("  1. Niantic C++ (fast, ~1-5 seconds) - INSTALLED")
    else:
        print("  1. Niantic C++ (fast) - not available")
    if gsconverter_installed:
        print("  2. gsconverter Python (~30-100 seconds) - INSTALLED")
    else:
        print("  2. gsconverter Python - not available")
    print("  3. Built-in SPZ v3 converter (always available) - OK")

    print("\n" + "=" * 60)
    print("[ArchAi3D Qwen] Installation complete!")
    print("=" * 60)
    print("\nPlease restart ComfyUI to load the new nodes.")

if __name__ == "__main__":
    main()
