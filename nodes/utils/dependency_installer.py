"""
ArchAi3D Dependency Installer Node
One-click installer for all ArchAi3D node dependencies.
Auto-detects GPU, CUDA, and PyTorch versions to install correct packages.
"""

import os
import sys
import subprocess
import platform
import re


class ArchAi3D_Dependency_Installer:
    """
    Automatically detects your system configuration and installs dependencies.

    Features:
    - Auto-detects GPU architecture (sm_75, sm_86, sm_89, sm_120)
    - Auto-detects CUDA and PyTorch versions
    - Installs Core, SAM3, Metric3D dependencies
    - Blackwell (RTX 5090) specific handling
    """

    # Dependency groups
    CORE_DEPS = [
        "PyYAML",
        "numpy",
        "Pillow",
        "requests",
        "tqdm",
    ]

    SAM3_DEPS = [
        "einops",
        "ftfy",
        "regex",
        "iopath",
        "pycocotools",
        "scikit-image",
        "scikit-learn",
        "pandas",
    ]

    METRIC3D_DEPS = [
        "huggingface_hub",
        "opencv-python",
        "matplotlib",
        "timm",
        "scipy",
        "addict",
        "yapf",
    ]

    COLOR_DEPS = [
        "colour-science",
    ]

    API_DEPS = [
        "google-genai",
        "gdown",
    ]

    # GPU architecture mapping
    GPU_ARCHITECTURES = {
        # Turing (sm_75)
        "RTX 2080": "sm_75", "RTX 2070": "sm_75", "RTX 2060": "sm_75",
        # Ampere (sm_86)
        "RTX 3090": "sm_86", "RTX 3080": "sm_86", "RTX 3070": "sm_86",
        "RTX 3060": "sm_86", "A6000": "sm_86",
        # Ada (sm_89)
        "RTX 4090": "sm_89", "RTX 4080": "sm_89", "RTX 4070": "sm_89",
        "RTX 4060": "sm_89", "L40": "sm_89",
        # Blackwell (sm_120)
        "RTX 5090": "sm_120", "RTX 5080": "sm_120", "RTX 5070": "sm_120",
        "B100": "sm_120", "B200": "sm_120",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": ([
                    "Check System",
                    "Install Core",
                    "Install SAM3",
                    "Install Metric3D",
                    "Install Color Tools",
                    "Install API Tools",
                    "Install All",
                    "Upgrade PyTorch (Blackwell)"
                ], {
                    "default": "Check System",
                    "tooltip": "Select installation action"
                }),
            },
            "optional": {
                "include_optional": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include optional dependencies"
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

                # If not found by name, try compute capability
                if arch == "unknown":
                    major, minor = torch.cuda.get_device_capability(0)
                    arch_map = {
                        "7.5": "sm_75", "8.0": "sm_80", "8.6": "sm_86",
                        "8.9": "sm_89", "9.0": "sm_90", "12.0": "sm_120"
                    }
                    compute_cap = f"{major}.{minor}"
                    arch = arch_map.get(compute_cap, f"sm_{major}{minor}")

                vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                return {
                    "name": gpu_name,
                    "arch": arch,
                    "vram_gb": round(vram, 1),
                    "available": True,
                    "is_blackwell": arch == "sm_120"
                }
        except Exception:
            pass

        return {"name": "Unknown", "arch": "unknown", "vram_gb": 0, "available": False, "is_blackwell": False}

    def _get_cuda_version(self):
        """Detect CUDA version."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.version.cuda
        except:
            pass

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

    def _get_pip_path(self):
        """Get pip executable path."""
        python_path = sys.executable
        pip_path = os.path.join(os.path.dirname(python_path), 'pip')
        if os.path.exists(pip_path):
            return pip_path
        pip3_path = os.path.join(os.path.dirname(python_path), 'pip3')
        if os.path.exists(pip3_path):
            return pip3_path
        return f"{python_path} -m pip"

    def _install_packages(self, packages, description):
        """Install a list of packages."""
        pip_path = self._get_pip_path()
        results = []

        for package in packages:
            cmd = f"{pip_path} install {package}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                results.append(f"✅ {package}")
            else:
                results.append(f"❌ {package}: {result.stderr[:100]}")

        return results

    def _upgrade_pytorch_blackwell(self):
        """Upgrade PyTorch for Blackwell GPUs."""
        pip_path = self._get_pip_path()
        results = []

        # Uninstall old PyTorch
        results.append("Uninstalling old PyTorch...")
        cmd = f"{pip_path} uninstall torch torchvision torchaudio -y"
        subprocess.run(cmd, shell=True, capture_output=True, text=True)

        # Install PyTorch cu128
        results.append("Installing PyTorch with CUDA 12.8...")
        cmd = f"{pip_path} install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode == 0:
            results.append("✅ PyTorch cu128 installed successfully!")
        else:
            results.append(f"❌ Failed: {result.stderr[:200]}")

        return results

    def _check_installed(self, package):
        """Check if a package is installed."""
        try:
            __import__(package.replace("-", "_").split(">=")[0].split("[")[0])
            return True
        except ImportError:
            return False

    def execute(self, action, include_optional=True):
        """Execute the selected action."""

        # Gather system info
        gpu_info = self._get_gpu_info()
        cuda_version = self._get_cuda_version()
        pytorch_version = self._get_pytorch_version()
        python_version = self._get_python_version()

        # Build status report
        status_lines = [
            "=" * 50,
            "ARCHAI3D DEPENDENCY INSTALLER",
            "=" * 50,
            "",
            f"GPU: {gpu_info['name']}",
            f"  Architecture: {gpu_info['arch']}",
            f"  VRAM: {gpu_info['vram_gb']} GB",
            f"  Blackwell: {'Yes' if gpu_info['is_blackwell'] else 'No'}",
            "",
            f"CUDA Version: {cuda_version}",
            f"PyTorch Version: {pytorch_version}",
            f"Python Version: {python_version}",
            f"Platform: {platform.system()} {platform.machine()}",
            "",
        ]

        # Blackwell warnings
        if gpu_info['is_blackwell']:
            status_lines.append("⚠️ BLACKWELL GPU DETECTED")
            try:
                pt_ver = float(pytorch_version)
                if pt_ver < 2.7:
                    status_lines.append(f"  ❌ PyTorch {pytorch_version} is too old for Blackwell")
                    status_lines.append("  → Use 'Upgrade PyTorch (Blackwell)' action")
                else:
                    status_lines.append(f"  ✅ PyTorch {pytorch_version} is compatible")
            except:
                pass

            try:
                cuda_ver = float(cuda_version)
                if cuda_ver < 12.8:
                    status_lines.append(f"  ⚠️ CUDA {cuda_version} - Blackwell works best with 12.8+")
            except:
                pass
            status_lines.append("")

        # Execute action
        if action == "Check System":
            status_lines.append("DEPENDENCY STATUS:")
            status_lines.append("-" * 30)

            # Check each group
            for name, deps in [
                ("Core", self.CORE_DEPS),
                ("SAM3", self.SAM3_DEPS),
                ("Metric3D", self.METRIC3D_DEPS),
                ("Color", self.COLOR_DEPS),
                ("API", self.API_DEPS),
            ]:
                installed = sum(1 for d in deps if self._check_installed(d))
                status_lines.append(f"{name}: {installed}/{len(deps)} installed")

        elif action == "Install Core":
            status_lines.append("Installing Core Dependencies...")
            status_lines.append("-" * 30)
            results = self._install_packages(self.CORE_DEPS, "Core")
            status_lines.extend(results)

        elif action == "Install SAM3":
            status_lines.append("Installing SAM3 Dependencies...")
            status_lines.append("-" * 30)
            results = self._install_packages(self.SAM3_DEPS, "SAM3")
            status_lines.extend(results)

        elif action == "Install Metric3D":
            status_lines.append("Installing Metric3D Dependencies...")
            status_lines.append("-" * 30)
            results = self._install_packages(self.METRIC3D_DEPS, "Metric3D")
            status_lines.extend(results)

        elif action == "Install Color Tools":
            status_lines.append("Installing Color Tools Dependencies...")
            status_lines.append("-" * 30)
            results = self._install_packages(self.COLOR_DEPS, "Color")
            status_lines.extend(results)

        elif action == "Install API Tools":
            status_lines.append("Installing API Tools Dependencies...")
            status_lines.append("-" * 30)
            results = self._install_packages(self.API_DEPS, "API")
            status_lines.extend(results)

        elif action == "Install All":
            status_lines.append("Installing All Dependencies...")
            status_lines.append("-" * 30)

            all_deps = (
                self.CORE_DEPS +
                self.SAM3_DEPS +
                self.METRIC3D_DEPS +
                self.COLOR_DEPS +
                self.API_DEPS
            )
            results = self._install_packages(all_deps, "All")
            status_lines.extend(results)

        elif action == "Upgrade PyTorch (Blackwell)":
            status_lines.append("Upgrading PyTorch for Blackwell...")
            status_lines.append("-" * 30)
            results = self._upgrade_pytorch_blackwell()
            status_lines.extend(results)
            status_lines.append("")
            status_lines.append("⚠️ RESTART COMFYUI after upgrade!")

        status_lines.extend([
            "",
            "=" * 50,
        ])

        result = "\n".join(status_lines)
        print(result)

        return (result,)


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "ArchAi3D_Dependency_Installer": ArchAi3D_Dependency_Installer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_Dependency_Installer": "🔧 Dependency Installer"
}
