"""
ArchAi3D Nunchaku Installer Node
Auto-detects GPU, CUDA, PyTorch, and Python versions to install the correct Nunchaku wheel.
"""

import os
import sys
import subprocess
import platform
import re


class ArchAi3D_Nunchaku_Installer:
    """
    Automatically detects your system configuration and installs Nunchaku.

    Features:
    - Auto-detects GPU (RTX 20/30/40/50 series, A100, etc.)
    - Auto-detects CUDA, PyTorch, and Python versions
    - Downloads and installs the correct wheel
    - Runs configuration scripts
    """

    # Wheel URLs for different configurations
    WHEEL_BASE_URL = "https://github.com/nunchaku-tech/nunchaku/releases/download"
    LATEST_VERSION = "v1.0.1"

    # GPU architecture mapping
    GPU_ARCHITECTURES = {
        # Turing (sm_75)
        "RTX 2080": "sm_75", "RTX 2080 Ti": "sm_75", "RTX 2070": "sm_75",
        "RTX 2060": "sm_75", "Quadro RTX": "sm_75", "TITAN RTX": "sm_75",
        # Ampere (sm_80/sm_86)
        "A100": "sm_80", "A30": "sm_80", "A40": "sm_86", "A10": "sm_86",
        "RTX 3090": "sm_86", "RTX 3080": "sm_86", "RTX 3070": "sm_86",
        "RTX 3060": "sm_86", "RTX A6000": "sm_86", "RTX A5000": "sm_86",
        "RTX A4000": "sm_86",
        # Ada Lovelace (sm_89)
        "RTX 4090": "sm_89", "RTX 4080": "sm_89", "RTX 4070": "sm_89",
        "RTX 4060": "sm_89", "RTX 6000 Ada": "sm_89", "L40": "sm_89",
        "L4": "sm_89",
        # Blackwell (sm_120)
        "RTX 5090": "sm_120", "RTX 5080": "sm_120", "RTX 5070": "sm_120",
        "B100": "sm_120", "B200": "sm_120",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (["Check System", "Install Nunchaku", "Update Versions File"], {
                    "default": "Check System",
                    "tooltip": "Check System: Show detected configuration\nInstall Nunchaku: Download and install\nUpdate Versions File: Refresh available models"
                }),
            },
            "optional": {
                "force_pytorch_version": (["auto", "2.5", "2.6", "2.7", "2.8"], {
                    "default": "auto",
                    "tooltip": "Override PyTorch version detection (use 'auto' for automatic)"
                }),
                "force_python_version": (["auto", "3.10", "3.11", "3.12", "3.13"], {
                    "default": "auto",
                    "tooltip": "Override Python version detection (use 'auto' for automatic)"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "execute"
    CATEGORY = "ArchAi3d/Utils"
    OUTPUT_NODE = True

    def _get_gpu_info(self):
        """Detect GPU name and architecture."""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)

                # Detect architecture from GPU name
                arch = "unknown"
                for gpu_key, sm_arch in self.GPU_ARCHITECTURES.items():
                    if gpu_key.lower() in gpu_name.lower():
                        arch = sm_arch
                        break

                # If not found by name, try to detect by compute capability
                if arch == "unknown":
                    major, minor = torch.cuda.get_device_capability(0)
                    compute_cap = f"{major}.{minor}"
                    arch_map = {
                        "7.5": "sm_75", "8.0": "sm_80", "8.6": "sm_86",
                        "8.9": "sm_89", "9.0": "sm_90", "12.0": "sm_120"
                    }
                    arch = arch_map.get(compute_cap, f"sm_{major}{minor}")

                vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                return {
                    "name": gpu_name,
                    "arch": arch,
                    "vram_gb": round(vram, 1),
                    "available": True
                }
        except Exception as e:
            pass

        return {"name": "Unknown", "arch": "unknown", "vram_gb": 0, "available": False}

    def _get_cuda_version(self):
        """Detect CUDA version."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.version.cuda
        except:
            pass

        # Try nvcc
        try:
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
            match = re.search(r'release (\d+\.\d+)', result.stdout)
            if match:
                return match.group(1)
        except:
            pass

        return "unknown"

    def _get_pytorch_version(self):
        """Detect PyTorch version."""
        try:
            import torch
            version = torch.__version__
            # Extract major.minor (e.g., "2.6" from "2.6.0+cu124")
            match = re.match(r'(\d+\.\d+)', version)
            if match:
                return match.group(1)
            return version
        except:
            return "unknown"

    def _get_python_version(self):
        """Detect Python version."""
        version = sys.version_info
        return f"{version.major}.{version.minor}"

    def _get_python_tag(self):
        """Get Python tag for wheel (e.g., cp312)."""
        version = sys.version_info
        return f"cp{version.major}{version.minor}"

    def _get_pip_path(self):
        """Get the pip executable path."""
        # Try to find pip in the same environment as the current Python
        python_path = sys.executable
        pip_path = os.path.join(os.path.dirname(python_path), 'pip')
        if os.path.exists(pip_path):
            return pip_path

        # Try pip3
        pip3_path = os.path.join(os.path.dirname(python_path), 'pip3')
        if os.path.exists(pip3_path):
            return pip3_path

        # Fall back to using python -m pip
        return f"{python_path} -m pip"

    def _check_nunchaku_installed(self):
        """Check if nunchaku is already installed."""
        try:
            import nunchaku
            # Try to get version
            if hasattr(nunchaku, '__version__'):
                return nunchaku.__version__
            # Check for key classes
            if hasattr(nunchaku, 'NunchakuFluxTransformer2DModelV2'):
                return "installed (version unknown)"
            return "installed"
        except ImportError:
            return None

    def _normalize_pytorch_version(self, pytorch_version):
        """Normalize PyTorch version to available wheel versions."""
        # Available wheel versions: 2.5, 2.6, 2.7, 2.8
        available_versions = ["2.5", "2.6", "2.7", "2.8"]

        # If exact match, use it
        if pytorch_version in available_versions:
            return pytorch_version

        # Try to parse and find closest match
        try:
            major_minor = pytorch_version.split('.')[:2]
            version_num = float(f"{major_minor[0]}.{major_minor[1]}")

            # For versions > 2.8, use 2.8 (latest available)
            if version_num > 2.8:
                return "2.8"
            # For versions < 2.5, use 2.5 (oldest available)
            elif version_num < 2.5:
                return "2.5"
            else:
                # Round to nearest available version
                return str(round(version_num, 1))
        except:
            # Default to 2.6 if parsing fails
            return "2.6"

    def _build_wheel_url(self, pytorch_version, python_tag):
        """Build the wheel URL for the detected configuration."""
        # Format: nunchaku-1.0.1+torch2.6-cp312-cp312-linux_x86_64.whl
        system = platform.system().lower()
        machine = platform.machine().lower()

        # Normalize PyTorch version to available wheel versions
        normalized_pytorch = self._normalize_pytorch_version(pytorch_version)

        if system == "linux":
            platform_tag = f"{python_tag}-linux_x86_64"
        elif system == "windows":
            platform_tag = f"{python_tag}-win_amd64"
        else:
            platform_tag = f"{python_tag}-{system}_{machine}"

        version_num = self.LATEST_VERSION.replace('v', '')
        filename = f"nunchaku-{version_num}+torch{normalized_pytorch}-{python_tag}-{platform_tag}.whl"

        return f"{self.WHEEL_BASE_URL}/{self.LATEST_VERSION}/{filename}"

    def _install_requirements(self):
        """Install nunchaku requirements."""
        pip_path = self._get_pip_path()

        # Find requirements.txt
        possible_paths = [
            "/workspace/runpod-slim/ComfyUI/custom_nodes/ComfyUI-nunchaku/requirements.txt",
            os.path.expanduser("~/ComfyUI/custom_nodes/ComfyUI-nunchaku/requirements.txt"),
        ]

        # Also check relative to this file
        this_dir = os.path.dirname(os.path.abspath(__file__))
        comfy_nodes = os.path.dirname(os.path.dirname(this_dir))
        possible_paths.append(os.path.join(comfy_nodes, "ComfyUI-nunchaku", "requirements.txt"))

        for req_path in possible_paths:
            if os.path.exists(req_path):
                cmd = f"{pip_path} install -r {req_path}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                return result.returncode == 0, result.stdout + result.stderr

        return False, "requirements.txt not found. Please install ComfyUI-nunchaku first."

    def _install_wheel(self, wheel_url):
        """Install the nunchaku wheel."""
        pip_path = self._get_pip_path()
        cmd = f"{pip_path} install {wheel_url}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout + result.stderr

    def _update_versions_file(self):
        """Run the update_versions.py script."""
        possible_paths = [
            "/workspace/runpod-slim/ComfyUI/custom_nodes/ComfyUI-nunchaku",
            os.path.expanduser("~/ComfyUI/custom_nodes/ComfyUI-nunchaku"),
        ]

        this_dir = os.path.dirname(os.path.abspath(__file__))
        comfy_nodes = os.path.dirname(os.path.dirname(this_dir))
        possible_paths.append(os.path.join(comfy_nodes, "ComfyUI-nunchaku"))

        for nunchaku_path in possible_paths:
            script_path = os.path.join(nunchaku_path, "scripts", "update_versions.py")
            if os.path.exists(script_path):
                cmd = f"{sys.executable} {script_path}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=nunchaku_path)
                return result.returncode == 0, result.stdout + result.stderr

        return False, "update_versions.py not found. Please install ComfyUI-nunchaku first."

    def execute(self, action, force_pytorch_version="auto", force_python_version="auto"):
        """Execute the selected action."""

        # Gather system info
        gpu_info = self._get_gpu_info()
        cuda_version = self._get_cuda_version()
        pytorch_version = self._get_pytorch_version() if force_pytorch_version == "auto" else force_pytorch_version
        python_version = self._get_python_version() if force_python_version == "auto" else force_python_version
        python_tag = self._get_python_tag() if force_python_version == "auto" else f"cp{force_python_version.replace('.', '')}"
        nunchaku_installed = self._check_nunchaku_installed()

        # Build status report
        status_lines = [
            "=" * 50,
            "NUNCHAKU INSTALLER - System Detection",
            "=" * 50,
            "",
            f"GPU: {gpu_info['name']}",
            f"  Architecture: {gpu_info['arch']}",
            f"  VRAM: {gpu_info['vram_gb']} GB",
            "",
            f"CUDA Version: {cuda_version}",
            f"PyTorch Version: {pytorch_version}",
            f"Python Version: {python_version}",
            f"Platform: {platform.system()} {platform.machine()}",
            "",
            f"Nunchaku Status: {nunchaku_installed if nunchaku_installed else 'Not installed'}",
            "",
        ]

        # Check compatibility
        is_compatible = True
        warnings = []

        if cuda_version != "unknown":
            cuda_major = float(cuda_version.split('.')[0])
            if cuda_major < 12:
                warnings.append(f"CUDA {cuda_version} may not be compatible. Nunchaku requires CUDA 12.2+")
                is_compatible = False

        if gpu_info['arch'] == 'sm_120':
            warnings.append("Blackwell GPU detected (RTX 50 series). Use FP4 models instead of INT4.")
            # Blackwell requires PyTorch 2.8+
            try:
                pt_version = float(pytorch_version)
                if pt_version < 2.8:
                    warnings.append(f"⚠️ Blackwell GPUs REQUIRE PyTorch 2.8+. Current: {pytorch_version}")
                    warnings.append("Please upgrade PyTorch before installing Nunchaku.")
                    is_compatible = False
            except:
                pass
            # Also check CUDA version for Blackwell
            if cuda_version != "unknown":
                try:
                    cuda_ver = float(cuda_version)
                    if cuda_ver < 12.8:
                        warnings.append(f"⚠️ Blackwell GPUs require CUDA 12.8+. Current: {cuda_version}")
                except:
                    pass

        if warnings:
            status_lines.append("WARNINGS:")
            for w in warnings:
                status_lines.append(f"  - {w}")
            status_lines.append("")

        # Execute action
        if action == "Check System":
            wheel_url = self._build_wheel_url(pytorch_version, python_tag)
            status_lines.extend([
                "Recommended wheel:",
                wheel_url,
                "",
                "To install, select 'Install Nunchaku' action."
            ])

        elif action == "Install Nunchaku":
            status_lines.append("INSTALLATION PROGRESS:")
            status_lines.append("-" * 30)

            # Step 1: Install requirements
            status_lines.append("1. Installing requirements...")
            success, output = self._install_requirements()
            if success:
                status_lines.append("   ✓ Requirements installed")
            else:
                status_lines.append(f"   ⚠ Requirements: {output[:200]}")

            # Step 2: Install wheel
            status_lines.append("2. Installing nunchaku wheel...")
            wheel_url = self._build_wheel_url(pytorch_version, python_tag)
            status_lines.append(f"   URL: {wheel_url}")

            success, output = self._install_wheel(wheel_url)
            if success:
                status_lines.append("   ✓ Nunchaku installed successfully!")
            else:
                status_lines.append(f"   ✗ Installation failed:")
                status_lines.append(f"   {output[:500]}")

            # Step 3: Update versions
            status_lines.append("3. Updating versions file...")
            success, output = self._update_versions_file()
            if success:
                status_lines.append("   ✓ Versions file updated")
            else:
                status_lines.append(f"   ⚠ {output[:200]}")

            status_lines.extend([
                "",
                "=" * 50,
                "RESTART COMFYUI TO USE NUNCHAKU NODES",
                "=" * 50,
            ])

        elif action == "Update Versions File":
            status_lines.append("Updating versions file...")
            success, output = self._update_versions_file()
            if success:
                status_lines.append("✓ Versions file updated successfully!")
                status_lines.append(output)
            else:
                status_lines.append(f"✗ Failed: {output}")

        result = "\n".join(status_lines)
        print(result)

        return (result,)


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "ArchAi3D_Nunchaku_Installer": ArchAi3D_Nunchaku_Installer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_Nunchaku_Installer": "🔧 Nunchaku Installer"
}
