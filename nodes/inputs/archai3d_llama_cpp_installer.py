# -*- coding: utf-8 -*-
"""
ArchAi3D Llama.cpp Installer Node
Author: Amir Ferdos (ArchAi3d)
Email: Amir84ferdos@gmail.com
LinkedIn: https://www.linkedin.com/in/archai3d/

Description:
    Install llama.cpp and download QwenVL GGUF models directly from ComfyUI.
    Perfect for RunPod, cloud instances, or fresh installations.
    Supports RTX 5090/Blackwell (sm_120) GPUs.

Usage:
    1. Add node to workflow
    2. Select model size (2B, 4B, or 8B)
    3. Run the node
    4. Wait for installation to complete

Version: 1.9.0 - Persistent CUDA libs for RunPod
"""

import os
import subprocess
import shutil

# GPU architecture mapping for CUDA build
GPU_ARCHITECTURES = {
    # Turing (sm_75)
    "RTX 2080": "75", "RTX 2070": "75", "RTX 2060": "75",
    # Ampere (sm_86)
    "RTX 3090": "86", "RTX 3080": "86", "RTX 3070": "86", "RTX 3060": "86",
    "A6000": "86", "A100": "80",
    # Ada (sm_89)
    "RTX 4090": "89", "RTX 4080": "89", "RTX 4070": "89", "RTX 4060": "89",
    "L40": "89",
    # Blackwell (sm_120)
    "RTX 5090": "120", "RTX 5080": "120", "RTX 5070": "120",
    "B100": "120", "B200": "120",
}


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
                "action": (["check_status", "install_llama_cpp", "download_model", "full_install", "install_cublas"], {
                    "default": "check_status",
                    "tooltip": "check_status: Check what's installed. install_llama_cpp: Build llama.cpp. download_model: Download model only. full_install: Everything. install_cublas: Install CUDA cuBLAS libraries."
                }),
                "model_size": (["4B (Recommended)", "2B (Fast/Low VRAM)", "8B (Best Quality)"], {
                    "default": "4B (Recommended)",
                    "tooltip": "2B: ~4GB VRAM, 4B: ~7GB VRAM, 8B: ~12GB VRAM"
                }),
            },
            "optional": {
                "quantization": (["Q4_K_M (Smaller, Faster)", "Q8_0 (Best Quality)"], {
                    "default": "Q4_K_M (Smaller, Faster)",
                    "tooltip": "Model quantization. Q4_K_M=~5GB, Q8_0=~9GB (best quality, needs more VRAM)"
                }),
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
        """Get installation paths.

        Saves everything within ComfyUI folder for RunPod persistence:
        - llama.cpp source: ComfyUI/custom_nodes/comfyui-archai3d-qwen/llama_cpp/
        - llama-server binary: ComfyUI/custom_nodes/comfyui-archai3d-qwen/bin/
        - Models: ComfyUI/models/llama-models/
        """
        # Get this node's directory (comfyui-archai3d-qwen)
        node_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # Get ComfyUI base directory
        comfyui_dir = os.path.dirname(os.path.dirname(node_dir))

        # Paths within ComfyUI folder (persistent on RunPod)
        return {
            "llama_cpp_dir": os.path.join(node_dir, "llama_cpp"),
            "llama_server": os.path.join(node_dir, "bin", "llama-server"),
            "models_dir": os.path.join(comfyui_dir, "models", "LLM", "GGUF"),
            "node_dir": node_dir,
            "comfyui_dir": comfyui_dir,
        }

    def get_cuda_env(self):
        """Get environment variables for CUDA compilation."""
        env = os.environ.copy()

        # Common CUDA paths to check
        cuda_paths = [
            "/usr/local/cuda",
            "/usr/local/cuda-12",
            "/usr/local/cuda-12.8",
            "/usr/local/cuda-12.4",
            "/usr/local/cuda-12.1",
            "/usr/local/cuda-11.8",
        ]

        cuda_home = None
        for path in cuda_paths:
            if os.path.exists(path):
                cuda_home = path
                break

        if cuda_home:
            env["CUDA_HOME"] = cuda_home
            env["CUDA_PATH"] = cuda_home
            env["PATH"] = f"{cuda_home}/bin:" + env.get("PATH", "")
            env["LD_LIBRARY_PATH"] = f"{cuda_home}/lib64:" + env.get("LD_LIBRARY_PATH", "")

        return env, cuda_home

    def get_gpu_info(self):
        """Detect GPU name and architecture."""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)

                # Detect architecture from GPU name
                cuda_arch = None
                for gpu_key, arch in GPU_ARCHITECTURES.items():
                    if gpu_key.lower() in gpu_name.lower():
                        cuda_arch = arch
                        break

                # If not found by name, try compute capability
                if cuda_arch is None:
                    major, minor = torch.cuda.get_device_capability(0)
                    arch_map = {
                        (7, 5): "75", (8, 0): "80", (8, 6): "86",
                        (8, 9): "89", (9, 0): "90", (12, 0): "120"
                    }
                    cuda_arch = arch_map.get((major, minor), f"{major}{minor}")

                vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                is_blackwell = cuda_arch == "120"

                return {
                    "name": gpu_name,
                    "cuda_arch": cuda_arch,
                    "vram_gb": round(vram, 1),
                    "available": True,
                    "is_blackwell": is_blackwell
                }
        except Exception:
            pass

        return {"name": "Unknown", "cuda_arch": None, "vram_gb": 0, "available": False, "is_blackwell": False}

    def check_cublas_available(self, cuda_home):
        """Check if cuBLAS header and library are available."""
        if not cuda_home:
            cuda_home = "/usr/local/cuda"

        # Check for cublas_v2.h header
        header_paths = [
            f"{cuda_home}/include/cublas_v2.h",
            f"{cuda_home}/targets/x86_64-linux/include/cublas_v2.h",
            "/usr/include/cublas_v2.h",
            "/usr/local/include/cublas_v2.h",
        ]

        header_found = None
        for path in header_paths:
            if os.path.exists(path):
                header_found = path
                break

        # Check for libcublas library
        lib_paths = [
            f"{cuda_home}/lib64/libcublas.so",
            f"{cuda_home}/targets/x86_64-linux/lib/libcublas.so",
            "/usr/lib/x86_64-linux-gnu/libcublas.so",
            "/usr/local/lib/libcublas.so",
        ]

        lib_found = None
        for path in lib_paths:
            if os.path.exists(path):
                lib_found = path
                break

        return header_found is not None and lib_found is not None, header_found, lib_found

    def copy_cuda_libs_to_persistent(self, cuda_home=None):
        """Copy CUDA libraries to persistent lib/ folder."""
        import glob

        paths = self.get_paths()
        lib_dir = os.path.join(paths["node_dir"], "lib")
        os.makedirs(lib_dir, exist_ok=True)

        if not cuda_home:
            _, cuda_home = self.get_cuda_env()

        # Find and copy cuBLAS and related libraries
        cuda_lib_paths = [
            f"{cuda_home}/lib64" if cuda_home else "/usr/local/cuda/lib64",
            "/usr/lib/x86_64-linux-gnu",
            f"{cuda_home}/targets/x86_64-linux/lib" if cuda_home else "",
        ]

        libs_to_copy = [
            "libcublas.so*",
            "libcublasLt.so*",
            "libcudart.so*",
        ]

        copied_libs = []
        for lib_path in cuda_lib_paths:
            if not lib_path or not os.path.exists(lib_path):
                continue
            for lib_pattern in libs_to_copy:
                for lib_file in glob.glob(os.path.join(lib_path, lib_pattern)):
                    if os.path.isfile(lib_file) and not os.path.islink(lib_file):
                        dest = os.path.join(lib_dir, os.path.basename(lib_file))
                        if not os.path.exists(dest):
                            try:
                                shutil.copy2(lib_file, dest)
                                copied_libs.append(os.path.basename(lib_file))
                            except Exception:
                                pass

        # Create symlinks (e.g., libcublas.so.12 -> libcublas.so.12.8.4.1)
        for lib_base in ["libcublas", "libcublasLt", "libcudart"]:
            lib_files = glob.glob(os.path.join(lib_dir, f"{lib_base}.so.*"))
            # Find full versioned file (not .so.12 symlink)
            for lib_file in lib_files:
                basename = os.path.basename(lib_file)
                if not basename.endswith(".so.12"):
                    symlink = os.path.join(lib_dir, f"{lib_base}.so.12")
                    if not os.path.exists(symlink):
                        try:
                            os.symlink(basename, symlink)
                        except Exception:
                            pass
                    break

        return lib_dir, copied_libs

    def install_cublas(self, gpu_info):
        """Install cuBLAS and copy to persistent location."""
        status_lines = ["=" * 50, "📦 INSTALLING CUBLAS", "=" * 50]
        env, cuda_home = self.get_cuda_env()
        paths = self.get_paths()

        # Check if libs already exist in persistent location
        lib_dir = os.path.join(paths["node_dir"], "lib")
        if os.path.exists(lib_dir):
            import glob
            existing_libs = glob.glob(os.path.join(lib_dir, "libcublas*.so*"))
            if existing_libs:
                status_lines.append(f"\n✅ CUDA libraries already in persistent location!")
                status_lines.append(f"   Location: {lib_dir}")
                status_lines.append(f"   Libraries: {len(existing_libs)} files")
                status_lines.append("\n" + "=" * 50)
                return "\n".join(status_lines), True

        # Determine CUDA version to install
        if gpu_info.get('is_blackwell'):
            # Blackwell needs CUDA 12.8
            cuda_ver = "12-8"
            status_lines.append("\n⚡ Installing cuBLAS for Blackwell (CUDA 12.8)...")
        else:
            # Try to detect CUDA version
            cuda_ver = "12-4"  # Default
            success, output = self.run_command("nvcc --version", env=env)
            if success:
                import re
                match = re.search(r'release (\d+)\.(\d+)', output)
                if match:
                    cuda_ver = f"{match.group(1)}-{match.group(2)}"
            status_lines.append(f"\n📦 Installing cuBLAS for CUDA {cuda_ver.replace('-', '.')}...")

        # Try multiple installation methods
        install_cmds = [
            # Method 1: Specific version
            f"apt-get update && apt-get install -y libcublas-{cuda_ver} libcublas-dev-{cuda_ver}",
            # Method 2: CUDA libraries package
            f"apt-get install -y cuda-libraries-{cuda_ver} cuda-libraries-dev-{cuda_ver}",
            # Method 3: Generic libcublas-dev
            "apt-get install -y libcublas-dev",
            # Method 4: Full CUDA toolkit
            "apt-get install -y nvidia-cuda-toolkit",
        ]

        installed = False
        for cmd in install_cmds:
            status_lines.append(f"\n🔧 Trying: {cmd[:60]}...")
            success, output = self.run_command(cmd, env=env)
            if success:
                # Verify installation
                cublas_ok, header, lib = self.check_cublas_available(cuda_home)
                if cublas_ok:
                    status_lines.append(f"✅ cuBLAS installed successfully!")
                    status_lines.append(f"   Header: {header}")
                    status_lines.append(f"   Library: {lib}")
                    installed = True
                    break
            else:
                status_lines.append(f"   ⚠️ Command failed, trying next...")

        if not installed:
            status_lines.append("\n❌ Could not install cuBLAS automatically")
            status_lines.append("\nManual installation options:")
            status_lines.append(f"  sudo apt-get install -y libcublas-{cuda_ver} libcublas-dev-{cuda_ver}")
            status_lines.append("  or")
            status_lines.append("  sudo apt-get install -y cuda-toolkit")
            status_lines.append("\nFor RTX 5090 (Blackwell):")
            status_lines.append("  sudo apt-get install -y libcublas-12-8 libcublas-dev-12-8")
        else:
            # Copy libraries to persistent location
            status_lines.append("\n📦 Copying CUDA libraries to persistent location...")
            lib_dir, copied_libs = self.copy_cuda_libs_to_persistent(cuda_home)
            if copied_libs:
                status_lines.append(f"✅ Copied {len(copied_libs)} libraries to {lib_dir}")
                status_lines.append("   Libraries will persist across pod restarts!")
            else:
                status_lines.append("⚠️ Could not copy libraries (they may already exist)")

        status_lines.append("\n" + "=" * 50)
        return "\n".join(status_lines), installed

    def run_command(self, cmd, cwd=None, env=None):
        """Run a shell command and return output."""
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 min timeout for builds
                env=env or os.environ.copy()
            )
            return result.returncode == 0, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, str(e)

    def check_status(self):
        """Check installation status."""
        paths = self.get_paths()
        env, cuda_home = self.get_cuda_env()
        gpu_info = self.get_gpu_info()
        status_lines = ["=" * 50, "🔍 LLAMA.CPP INSTALLATION STATUS", "=" * 50]

        # Show paths (helpful for RunPod users)
        status_lines.append("\n📂 INSTALLATION PATHS (Persistent on RunPod):")
        status_lines.append(f"   llama-server: {paths['llama_server']}")
        status_lines.append(f"   Models: {paths['models_dir']}")

        # GPU Info
        status_lines.append(f"\n🎮 GPU: {gpu_info['name']}")
        if gpu_info['cuda_arch']:
            status_lines.append(f"   Architecture: sm_{gpu_info['cuda_arch']}")
            status_lines.append(f"   VRAM: {gpu_info['vram_gb']} GB")
            if gpu_info['is_blackwell']:
                status_lines.append("   ⚡ Blackwell GPU detected!")

        # Check llama-server
        status_lines.append("")
        if os.path.exists(paths["llama_server"]):
            status_lines.append(f"✅ llama-server: {paths['llama_server']}")
        else:
            status_lines.append("❌ llama-server: NOT INSTALLED")

        # Check CUDA
        success, output = self.run_command("nvcc --version", env=env)
        if success:
            cuda_ver = [l for l in output.split('\n') if 'release' in l.lower()]
            status_lines.append(f"✅ CUDA: {cuda_ver[0] if cuda_ver else 'Available'}")
        else:
            status_lines.append("⚠️ CUDA: nvcc not found")

        if cuda_home:
            status_lines.append(f"   CUDA_HOME: {cuda_home}")

        # Check for persistent CUDA libraries (copied to lib/ folder)
        import glob
        lib_dir = os.path.join(paths["node_dir"], "lib")
        persistent_libs = glob.glob(os.path.join(lib_dir, "libcublas*.so*")) if os.path.exists(lib_dir) else []

        if persistent_libs:
            status_lines.append(f"✅ cuBLAS: Persistent ({len(persistent_libs)} libs in lib/)")
        else:
            # Check system cuBLAS (required for GPU-accelerated builds)
            cublas_ok, header, lib = self.check_cublas_available(cuda_home)
            if cublas_ok:
                status_lines.append(f"✅ cuBLAS: System (not persistent)")
                status_lines.append(f"   Run 'install_cublas' to make it persistent")
            else:
                status_lines.append(f"❌ cuBLAS: NOT FOUND")
                status_lines.append(f"   Use 'install_cublas' action to install & persist")

        # Check models - search in both new and legacy paths
        status_lines.append("\n📦 MODELS:")
        model_search_dirs = [
            paths["models_dir"],  # models/LLM/GGUF/
            os.path.join(paths["comfyui_dir"], "models", "llama-models"),  # legacy
        ]
        for size_label, prefix in [("2B", "Qwen3VL-2B"), ("4B", "Qwen3VL-4B"), ("8B", "Qwen3VL-8B")]:
            found_files = []
            for search_dir in model_search_dirs:
                if not os.path.exists(search_dir):
                    continue
                # Check flat structure and subdirectories
                for root, dirs, files in os.walk(search_dir):
                    for f in files:
                        if f.startswith(prefix) and f.endswith('.gguf'):
                            found_files.append(f)
                    break  # Only top level + one level deep
                for subdir in [f"qwen3-vl-{size_label.lower()}"]:
                    sub_path = os.path.join(search_dir, subdir)
                    if os.path.exists(sub_path):
                        for f in os.listdir(sub_path):
                            if f.startswith(prefix) and f.endswith('.gguf'):
                                found_files.append(f)
            if found_files:
                quants = [f.split('-')[-1].replace('.gguf', '') for f in found_files if 'mmproj' not in f]
                status_lines.append(f"  ✅ {size_label}: {', '.join(quants)} ({len(found_files)} files)")
            else:
                status_lines.append(f"  ❌ {size_label}: Not downloaded")

        # Check if servers are running (all ports)
        server_ports = {"2B": 8032, "4B": 8033, "8B": 8034}
        any_running = False
        for name, port in server_ports.items():
            success, _ = self.run_command(f"curl -s -o /dev/null -w '%{{http_code}}' --connect-timeout 1 http://localhost:{port}/health")
            if success:
                status_lines.append(f"\n🟢 Server {name}: RUNNING on port {port}")
                any_running = True
        if not any_running:
            status_lines.append("\n🔴 No servers running (ports 8032/8033/8034)")

        status_lines.append("=" * 50)
        return "\n".join(status_lines)

    def install_llama_cpp(self, force_rebuild=False):
        """Install llama.cpp with CUDA support."""
        paths = self.get_paths()
        env, cuda_home = self.get_cuda_env()
        gpu_info = self.get_gpu_info()
        status_lines = ["=" * 50, "🔧 INSTALLING LLAMA.CPP", "=" * 50]

        # Show GPU info
        status_lines.append(f"\n🎮 Detected GPU: {gpu_info['name']}")
        if gpu_info['cuda_arch']:
            status_lines.append(f"   Architecture: sm_{gpu_info['cuda_arch']}")
            if gpu_info['is_blackwell']:
                status_lines.append("   ⚡ Building with Blackwell (sm_120) support!")

        # Check if already installed
        if os.path.exists(paths["llama_server"]) and not force_rebuild:
            status_lines.append("✅ llama-server already installed!")
            status_lines.append(f"   Location: {paths['llama_server']}")
            status_lines.append("   Use force_rebuild=True to reinstall")
            return "\n".join(status_lines)

        # Install dependencies
        status_lines.append("\n📦 Installing build dependencies...")

        # First install basic build tools
        self.run_command("apt-get update", env=env)
        success, output = self.run_command(
            "apt-get install -y build-essential cmake git curl libcurl4-openssl-dev",
            env=env
        )
        if success:
            status_lines.append("✅ Basic build tools installed")
        else:
            status_lines.append("⚠️ Basic tools warning (may already exist)")

        # Check for cuBLAS (required for GPU-accelerated builds)
        status_lines.append("\n📦 Checking cuBLAS availability...")
        cublas_ok, header, lib = self.check_cublas_available(cuda_home)

        if cublas_ok:
            status_lines.append(f"✅ cuBLAS found!")
            status_lines.append(f"   Header: {header}")
            status_lines.append(f"   Library: {lib}")
        else:
            status_lines.append("⚠️ cuBLAS not found - attempting to install...")

            # Auto-install cuBLAS
            install_output, installed = self.install_cublas(gpu_info)
            status_lines.append(install_output)

            # Re-check after installation
            cublas_ok, header, lib = self.check_cublas_available(cuda_home)

            if cublas_ok:
                status_lines.append(f"\n✅ cuBLAS now available!")
            else:
                status_lines.append(f"\n⚠️ cuBLAS still not found - GPU build may fail")
                status_lines.append("   Build will attempt anyway, but may fall back to CPU-only")

        # Show CUDA info
        if cuda_home:
            status_lines.append(f"\n🔧 CUDA detected: {cuda_home}")
        else:
            status_lines.append("\n⚠️ CUDA_HOME not found, will try to detect automatically")

        # Clone or update llama.cpp
        if os.path.exists(paths["llama_cpp_dir"]):
            status_lines.append("\n📥 Updating llama.cpp...")
            success, output = self.run_command("git fetch && git reset --hard origin/master", cwd=paths["llama_cpp_dir"], env=env)
        else:
            status_lines.append("\n📥 Cloning llama.cpp...")
            success, output = self.run_command(f"git clone https://github.com/ggml-org/llama.cpp {paths['llama_cpp_dir']}", env=env)

        if not success:
            status_lines.append(f"❌ Git error: {output[:500]}")
            return "\n".join(status_lines)
        status_lines.append("✅ Repository ready")

        # Build with CUDA
        status_lines.append("\n🔨 Building with CUDA support (this may take 5-10 minutes)...")
        build_dir = os.path.join(paths["llama_cpp_dir"], "build")

        # Clean build directory
        if os.path.exists(build_dir):
            shutil.rmtree(build_dir)

        # Configure with explicit CUDA settings
        # Disable CURL to avoid dependency issues, disable ccache warning
        cmake_cmd = "cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_CCACHE=OFF"
        if cuda_home:
            cmake_cmd += f" -DCMAKE_CUDA_COMPILER={cuda_home}/bin/nvcc"

        # Add GPU-specific architecture flag for optimal performance
        # Use CMAKE_CUDA_ARCHITECTURES (not GGML_CUDA_ARCHITECTURES) for proper sm_120 support
        if gpu_info['cuda_arch']:
            cmake_cmd += f" -DCMAKE_CUDA_ARCHITECTURES={gpu_info['cuda_arch']}"
            status_lines.append(f"\n🎯 Targeting CUDA architecture: sm_{gpu_info['cuda_arch']}")

        status_lines.append(f"\n🔧 Running: {cmake_cmd}")
        success, output = self.run_command(cmake_cmd, cwd=paths["llama_cpp_dir"], env=env)

        if not success:
            # Show more of the error
            status_lines.append(f"\n❌ CMake configure error:")
            status_lines.append("-" * 40)
            # Show last 30 lines of output
            error_lines = output.strip().split('\n')[-30:]
            status_lines.extend(error_lines)
            status_lines.append("-" * 40)

            # Try without CUDA as fallback - must clean build dir first
            status_lines.append("\n⚠️ Trying CPU-only build as fallback...")
            if os.path.exists(build_dir):
                shutil.rmtree(build_dir)
            cmake_cmd_cpu = "cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_CCACHE=OFF -DGGML_CUDA=OFF"
            success, output = self.run_command(cmake_cmd_cpu, cwd=paths["llama_cpp_dir"], env=env)

            if not success:
                status_lines.append(f"❌ CPU build also failed: {output[-500:]}")
                return "\n".join(status_lines)
            else:
                status_lines.append("✅ CPU-only build configured (slower, but works)")
        else:
            status_lines.append("✅ CMake configure successful")

        # Build
        status_lines.append("\n🔨 Compiling (please wait)...")
        build_cmd = "cmake --build build --config Release -j$(nproc)"
        success, output = self.run_command(build_cmd, cwd=paths["llama_cpp_dir"], env=env)

        if not success:
            status_lines.append(f"\n❌ Build error:")
            status_lines.append("-" * 40)
            error_lines = output.strip().split('\n')[-30:]
            status_lines.extend(error_lines)
            status_lines.append("-" * 40)
            return "\n".join(status_lines)
        status_lines.append("✅ Build complete")

        # Install
        status_lines.append("\n📦 Installing llama-server...")
        os.makedirs(os.path.dirname(paths["llama_server"]), exist_ok=True)

        # Check multiple possible locations for llama-server
        possible_paths = [
            os.path.join(paths["llama_cpp_dir"], "build", "bin", "llama-server"),
            os.path.join(paths["llama_cpp_dir"], "build", "llama-server"),
            os.path.join(paths["llama_cpp_dir"], "llama-server"),
        ]

        src_server = None
        for p in possible_paths:
            if os.path.exists(p):
                src_server = p
                break

        if src_server:
            shutil.copy2(src_server, paths["llama_server"])
            os.chmod(paths["llama_server"], 0o755)
            status_lines.append(f"✅ Installed to: {paths['llama_server']}")

            # Copy CUDA/cuBLAS libraries to persistent location
            status_lines.append("\n📦 Copying CUDA libraries to persistent location...")
            lib_dir = os.path.join(paths["node_dir"], "lib")
            os.makedirs(lib_dir, exist_ok=True)

            # Find and copy cuBLAS and related libraries
            cuda_lib_paths = [
                f"{cuda_home}/lib64" if cuda_home else "/usr/local/cuda/lib64",
                "/usr/lib/x86_64-linux-gnu",
                f"{cuda_home}/targets/x86_64-linux/lib" if cuda_home else "",
            ]

            libs_to_copy = [
                "libcublas.so*",
                "libcublasLt.so*",
                "libcudart.so*",
            ]

            copied_libs = []
            for lib_path in cuda_lib_paths:
                if not lib_path or not os.path.exists(lib_path):
                    continue
                for lib_pattern in libs_to_copy:
                    import glob
                    for lib_file in glob.glob(os.path.join(lib_path, lib_pattern)):
                        if os.path.isfile(lib_file) and not os.path.islink(lib_file):
                            dest = os.path.join(lib_dir, os.path.basename(lib_file))
                            if not os.path.exists(dest):
                                try:
                                    shutil.copy2(lib_file, dest)
                                    copied_libs.append(os.path.basename(lib_file))
                                except Exception:
                                    pass

            if copied_libs:
                status_lines.append(f"✅ Copied {len(copied_libs)} CUDA libraries to {lib_dir}")
            else:
                status_lines.append("⚠️ Could not copy CUDA libraries (may need manual setup)")

            # Verify it runs with the local libraries
            test_env = env.copy()
            test_env["LD_LIBRARY_PATH"] = f"{lib_dir}:" + test_env.get("LD_LIBRARY_PATH", "")
            success, output = self.run_command(f"{paths['llama_server']} --version", env=test_env)
            if success:
                status_lines.append(f"✅ Version: {output.strip()[:100]}")
        else:
            # List what was built
            status_lines.append(f"❌ llama-server not found in build output")
            status_lines.append("\n📂 Build contents:")
            for root, dirs, files in os.walk(build_dir):
                for f in files:
                    if 'llama' in f.lower():
                        status_lines.append(f"   {os.path.join(root, f)}")
            return "\n".join(status_lines)

        status_lines.append("\n" + "=" * 50)
        status_lines.append("✅ INSTALLATION COMPLETE!")
        status_lines.append("=" * 50)
        return "\n".join(status_lines)

    def download_model(self, model_size, quantization="Q4_K_M (Smaller, Faster)"):
        """Download QwenVL GGUF model."""
        paths = self.get_paths()
        quant_key = quantization.split()[0]  # "Q4_K_M" or "Q8_0"
        status_lines = ["=" * 50, f"📥 DOWNLOADING MODEL: {model_size} ({quant_key})", "=" * 50]

        # Parse model size
        if "2B" in model_size:
            repo = "Qwen/Qwen3-VL-2B-Instruct-GGUF"
            model_file = f"Qwen3VL-2B-Instruct-{quant_key}.gguf"
            mmproj_file = "mmproj-Qwen3VL-2B-Instruct-F16.gguf"
        elif "8B" in model_size:
            repo = "Qwen/Qwen3-VL-8B-Instruct-GGUF"
            model_file = f"Qwen3VL-8B-Instruct-{quant_key}.gguf"
            mmproj_file = "mmproj-Qwen3VL-8B-Instruct-F16.gguf"
        else:  # 4B default
            repo = "Qwen/Qwen3-VL-4B-Instruct-GGUF"
            model_file = f"Qwen3VL-4B-Instruct-{quant_key}.gguf"
            mmproj_file = "mmproj-Qwen3VL-4B-Instruct-F16.gguf"

        model_dir = paths["models_dir"]
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
            status_lines.append(f"❌ Download error: {output[:500]}")
            return "\n".join(status_lines)

        status_lines.append("\n" + "=" * 50)
        status_lines.append("✅ DOWNLOAD COMPLETE!")
        status_lines.append("=" * 50)
        return "\n".join(status_lines)

    def execute(self, action, model_size, quantization="Q4_K_M (Smaller, Faster)",
                force_rebuild=False):
        """Execute the installer action."""

        if action == "check_status":
            status = self.check_status()

        elif action == "install_llama_cpp":
            status = self.install_llama_cpp(force_rebuild)

        elif action == "download_model":
            status = self.download_model(model_size, quantization)

        elif action == "full_install":
            # Do everything
            status_parts = []
            status_parts.append(self.install_llama_cpp(force_rebuild))
            status_parts.append("\n\n")
            status_parts.append(self.download_model(model_size, quantization))
            status_parts.append("\n\n")
            status_parts.append(self.check_status())
            status = "".join(status_parts)

        elif action == "install_cublas":
            # Install cuBLAS libraries
            gpu_info = self.get_gpu_info()
            status, _ = self.install_cublas(gpu_info)
            # Show final status
            status += "\n\n" + self.check_status()

        else:
            status = f"Unknown action: {action}"

        print(status)
        return (status,)
