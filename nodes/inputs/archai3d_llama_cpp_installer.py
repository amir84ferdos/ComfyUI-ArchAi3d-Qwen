# -*- coding: utf-8 -*-
"""
ArchAi3D Llama.cpp Installer Node
Author: Amir Ferdos (ArchAi3d)
Email: Amir84ferdos@gmail.com
LinkedIn: https://www.linkedin.com/in/archai3d/

Description:
    Install llama.cpp and download QwenVL GGUF models directly from ComfyUI.
    Perfect for RunPod, cloud instances, or fresh installations.

Usage:
    1. Add node to workflow
    2. Select model size (2B, 4B, or 8B)
    3. Run the node
    4. Wait for installation to complete

Version: 1.0.0
"""

import os
import subprocess
import shutil


class ArchAi3D_LlamaCpp_Installer:
    """Install llama.cpp and QwenVL GGUF models.

    This node automates the installation of:
    1. Build dependencies (cmake, build-essential)
    2. llama.cpp with CUDA support
    3. QwenVL GGUF models from HuggingFace

    Perfect for RunPod and cloud instances.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (["check_status", "install_llama_cpp", "download_model", "full_install"], {
                    "default": "check_status",
                    "tooltip": "check_status: Check what's installed. install_llama_cpp: Build llama.cpp. download_model: Download model only. full_install: Everything."
                }),
                "model_size": (["4B (Recommended)", "2B (Fast/Low VRAM)", "8B (Best Quality)"], {
                    "default": "4B (Recommended)",
                    "tooltip": "2B: ~4GB VRAM, 4B: ~7GB VRAM, 8B: ~12GB VRAM"
                }),
            },
            "optional": {
                "force_rebuild": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Force rebuild llama.cpp even if already installed"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "execute"
    CATEGORY = "ArchAi3d/Setup"
    OUTPUT_NODE = True

    def get_paths(self):
        """Get installation paths."""
        home = os.path.expanduser("~")
        return {
            "llama_cpp_dir": os.path.join(home, "llama.cpp"),
            "llama_server": os.path.join(home, ".local", "bin", "llama-server"),
            "models_dir": os.path.join(home, ".cache", "llama-models"),
        }

    def run_command(self, cmd, cwd=None):
        """Run a shell command and return output."""
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 min timeout for builds
            )
            return result.returncode == 0, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, str(e)

    def check_status(self):
        """Check installation status."""
        paths = self.get_paths()
        status_lines = ["=" * 50, "🔍 LLAMA.CPP INSTALLATION STATUS", "=" * 50]

        # Check llama-server
        if os.path.exists(paths["llama_server"]):
            status_lines.append(f"✅ llama-server: {paths['llama_server']}")
        else:
            status_lines.append("❌ llama-server: NOT INSTALLED")

        # Check CUDA
        success, output = self.run_command("nvcc --version")
        if success:
            cuda_ver = [l for l in output.split('\n') if 'release' in l.lower()]
            status_lines.append(f"✅ CUDA: {cuda_ver[0] if cuda_ver else 'Available'}")
        else:
            status_lines.append("⚠️ CUDA: Not found (will try to build anyway)")

        # Check models
        status_lines.append("\n📦 MODELS:")
        for model in ["qwen3-vl-2b", "qwen3-vl-4b", "qwen3-vl-8b"]:
            model_dir = os.path.join(paths["models_dir"], model)
            if os.path.exists(model_dir):
                files = os.listdir(model_dir)
                gguf_files = [f for f in files if f.endswith('.gguf')]
                if len(gguf_files) >= 2:
                    status_lines.append(f"  ✅ {model}: Ready ({len(gguf_files)} files)")
                else:
                    status_lines.append(f"  ⚠️ {model}: Incomplete ({len(gguf_files)} files)")
            else:
                status_lines.append(f"  ❌ {model}: Not downloaded")

        # Check if server is running
        success, _ = self.run_command("curl -s http://localhost:8033/health")
        if success:
            status_lines.append("\n🟢 Server: RUNNING on port 8033")
        else:
            status_lines.append("\n🔴 Server: NOT RUNNING")

        status_lines.append("=" * 50)
        return "\n".join(status_lines)

    def install_llama_cpp(self, force_rebuild=False):
        """Install llama.cpp with CUDA support."""
        paths = self.get_paths()
        status_lines = ["=" * 50, "🔧 INSTALLING LLAMA.CPP", "=" * 50]

        # Check if already installed
        if os.path.exists(paths["llama_server"]) and not force_rebuild:
            status_lines.append("✅ llama-server already installed!")
            status_lines.append(f"   Location: {paths['llama_server']}")
            status_lines.append("   Use force_rebuild=True to reinstall")
            return "\n".join(status_lines)

        # Install dependencies
        status_lines.append("\n📦 Installing build dependencies...")
        success, output = self.run_command("apt update && apt install -y build-essential cmake git")
        if not success:
            # Try without sudo (RunPod usually runs as root)
            success, output = self.run_command("sudo apt update && sudo apt install -y build-essential cmake git")

        if success:
            status_lines.append("✅ Dependencies installed")
        else:
            status_lines.append(f"⚠️ Dependencies warning: {output[:200]}")

        # Clone or update llama.cpp
        if os.path.exists(paths["llama_cpp_dir"]):
            status_lines.append("\n📥 Updating llama.cpp...")
            success, output = self.run_command("git pull", cwd=paths["llama_cpp_dir"])
        else:
            status_lines.append("\n📥 Cloning llama.cpp...")
            success, output = self.run_command(f"git clone https://github.com/ggml-org/llama.cpp {paths['llama_cpp_dir']}")

        if not success:
            status_lines.append(f"❌ Git error: {output[:200]}")
            return "\n".join(status_lines)
        status_lines.append("✅ Repository ready")

        # Build with CUDA
        status_lines.append("\n🔨 Building with CUDA support (this may take 5-10 minutes)...")
        build_dir = os.path.join(paths["llama_cpp_dir"], "build")

        # Clean build directory
        if os.path.exists(build_dir):
            shutil.rmtree(build_dir)

        # Configure
        success, output = self.run_command(
            "cmake -B build -DGGML_CUDA=ON",
            cwd=paths["llama_cpp_dir"]
        )
        if not success:
            status_lines.append(f"❌ CMake configure error: {output[:300]}")
            return "\n".join(status_lines)

        # Build
        success, output = self.run_command(
            "cmake --build build --config Release -j$(nproc)",
            cwd=paths["llama_cpp_dir"]
        )
        if not success:
            status_lines.append(f"❌ Build error: {output[:300]}")
            return "\n".join(status_lines)
        status_lines.append("✅ Build complete")

        # Install
        status_lines.append("\n📦 Installing llama-server...")
        os.makedirs(os.path.dirname(paths["llama_server"]), exist_ok=True)

        src_server = os.path.join(paths["llama_cpp_dir"], "build", "bin", "llama-server")
        if os.path.exists(src_server):
            shutil.copy2(src_server, paths["llama_server"])
            os.chmod(paths["llama_server"], 0o755)
            status_lines.append(f"✅ Installed to: {paths['llama_server']}")
        else:
            status_lines.append(f"❌ llama-server not found in build output")
            return "\n".join(status_lines)

        status_lines.append("\n" + "=" * 50)
        status_lines.append("✅ INSTALLATION COMPLETE!")
        status_lines.append("=" * 50)
        return "\n".join(status_lines)

    def download_model(self, model_size):
        """Download QwenVL GGUF model."""
        paths = self.get_paths()
        status_lines = ["=" * 50, f"📥 DOWNLOADING MODEL: {model_size}", "=" * 50]

        # Parse model size
        if "2B" in model_size:
            model_name = "qwen3-vl-2b"
            repo = "Qwen/Qwen3-VL-2B-Instruct-GGUF"
            model_file = "Qwen3VL-2B-Instruct-Q4_K_M.gguf"
            mmproj_file = "mmproj-Qwen3VL-2B-Instruct-F16.gguf"
        elif "8B" in model_size:
            model_name = "qwen3-vl-8b"
            repo = "Qwen/Qwen3-VL-8B-Instruct-GGUF"
            model_file = "Qwen3VL-8B-Instruct-Q4_K_M.gguf"
            mmproj_file = "mmproj-Qwen3VL-8B-Instruct-F16.gguf"
        else:  # 4B default
            model_name = "qwen3-vl-4b"
            repo = "Qwen/Qwen3-VL-4B-Instruct-GGUF"
            model_file = "Qwen3VL-4B-Instruct-Q4_K_M.gguf"
            mmproj_file = "mmproj-Qwen3VL-4B-Instruct-F16.gguf"

        model_dir = os.path.join(paths["models_dir"], model_name)
        os.makedirs(model_dir, exist_ok=True)

        # Check if already downloaded
        model_path = os.path.join(model_dir, model_file)
        mmproj_path = os.path.join(model_dir, mmproj_file)

        if os.path.exists(model_path) and os.path.exists(mmproj_path):
            status_lines.append(f"✅ Model already downloaded!")
            status_lines.append(f"   Model: {model_path}")
            status_lines.append(f"   MMProj: {mmproj_path}")
            return "\n".join(status_lines)

        # Download using huggingface_hub
        status_lines.append(f"\n📦 Repository: {repo}")
        status_lines.append(f"📂 Destination: {model_dir}")
        status_lines.append("\n⏳ Downloading (this may take several minutes)...")

        download_script = f'''
import os
from huggingface_hub import hf_hub_download

local_dir = "{model_dir}"
os.makedirs(local_dir, exist_ok=True)

print("Downloading model file...")
hf_hub_download("{repo}", "{model_file}", local_dir=local_dir)

print("Downloading mmproj file...")
hf_hub_download("{repo}", "{mmproj_file}", local_dir=local_dir)

print("Download complete!")
'''

        success, output = self.run_command(f'python3 -c \'{download_script}\'')

        if success and os.path.exists(model_path):
            # Get file sizes
            model_size_gb = os.path.getsize(model_path) / (1024**3)
            mmproj_size_gb = os.path.getsize(mmproj_path) / (1024**3) if os.path.exists(mmproj_path) else 0

            status_lines.append(f"\n✅ Model downloaded: {model_size_gb:.2f} GB")
            status_lines.append(f"✅ MMProj downloaded: {mmproj_size_gb:.2f} GB")
        else:
            status_lines.append(f"❌ Download error: {output[:300]}")
            return "\n".join(status_lines)

        status_lines.append("\n" + "=" * 50)
        status_lines.append("✅ DOWNLOAD COMPLETE!")
        status_lines.append("=" * 50)
        return "\n".join(status_lines)

    def execute(self, action, model_size, force_rebuild=False):
        """Execute the installer action."""

        if action == "check_status":
            status = self.check_status()

        elif action == "install_llama_cpp":
            status = self.install_llama_cpp(force_rebuild)

        elif action == "download_model":
            status = self.download_model(model_size)

        elif action == "full_install":
            # Do everything
            status_parts = []
            status_parts.append(self.install_llama_cpp(force_rebuild))
            status_parts.append("\n\n")
            status_parts.append(self.download_model(model_size))
            status_parts.append("\n\n")
            status_parts.append(self.check_status())
            status = "".join(status_parts)

        else:
            status = f"Unknown action: {action}"

        print(status)
        return (status,)
