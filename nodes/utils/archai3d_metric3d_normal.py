"""
ArchAi3D Metric3D Normal Map (Low VRAM Optimized)

An optimized version of Metric3D Normal Map preprocessor with:
- Disk-based output caching (avoids reprocessing)
- Aggressive memory cleanup (model unloaded after EVERY execution)
- torch.cuda.empty_cache() + gc.collect() for low VRAM systems
- name field for web interface integration

Based on: comfyui_controlnet_aux/node_wrappers/metric3d.py
"""

import os
import gc
import json
import hashlib

import numpy as np
import torch
from PIL import Image

import comfy.model_management as model_management

# Cache directory for this node
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "metric3d")
os.makedirs(CACHE_DIR, exist_ok=True)

# Cache index file
CACHE_INDEX_FILE = os.path.join(CACHE_DIR, "cache_index.json")

# Resolution limits
MAX_RESOLUTION = 8192


def _load_cache_index():
    """Load cache index from disk."""
    if os.path.exists(CACHE_INDEX_FILE):
        try:
            with open(CACHE_INDEX_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}


def _save_cache_index(index):
    """Save cache index to disk."""
    try:
        with open(CACHE_INDEX_FILE, 'w') as f:
            json.dump(index, f, indent=2)
    except Exception as e:
        print(f"[ArchAi3D Metric3D] Warning: Could not save cache index: {e}")


def pil2tensor(image):
    """Convert PIL image to tensor (HWC format, 0-1 range)."""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def tensor2pil(tensor):
    """Convert tensor to PIL image."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    return Image.fromarray(np.clip(255. * tensor.cpu().numpy(), 0, 255).astype(np.uint8))


def resize_image(input_image, resolution):
    """Resize image to target resolution maintaining aspect ratio."""
    # Check type first before accessing attributes
    if isinstance(input_image, np.ndarray):
        H, W = input_image.shape[:2]
    else:
        # PIL Image - use .size which returns (width, height)
        W, H = input_image.size

    k = float(resolution) / min(H, W)
    new_H = int(H * k)
    new_W = int(W * k)

    if isinstance(input_image, np.ndarray):
        pil_img = Image.fromarray(input_image)
        pil_img = pil_img.resize((new_W, new_H), Image.LANCZOS)
        return np.array(pil_img)
    else:
        return input_image.resize((new_W, new_H), Image.LANCZOS)


class ArchAi3D_Metric3D_Normal:
    """
    Optimized Metric3D Normal Map for Low VRAM Systems.

    Features:
    - Disk-based output caching (PNG files)
    - Model unloaded after EVERY execution
    - torch.cuda.empty_cache() + gc.collect()
    - name field for web interface
    - vit-small backbone recommended for low VRAM
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {
                    "default": "metric3d_normal",
                    "multiline": False,
                    "tooltip": "Identifier name for this input (used by web interface)"
                }),
                "image": ("IMAGE",),
                "backbone": (["vit-small", "vit-large", "vit-giant2"], {
                    "default": "vit-small",
                    "tooltip": "Model backbone. vit-small uses less VRAM (~1-2GB), vit-giant2 uses most (~4-6GB)"
                }),
                "resolution": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": MAX_RESOLUTION,
                    "step": 64,
                    "tooltip": "Processing resolution. Lower = faster + less VRAM"
                }),
                "use_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use disk cache to avoid reprocessing identical inputs"
                }),
            },
            "optional": {
                "fx": ("INT", {
                    "default": 1000,
                    "min": 1,
                    "max": MAX_RESOLUTION,
                    "tooltip": "Focal length X (camera intrinsic)"
                }),
                "fy": ("INT", {
                    "default": 1000,
                    "min": 1,
                    "max": MAX_RESOLUTION,
                    "tooltip": "Focal length Y (camera intrinsic)"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("normal_map",)
    FUNCTION = "execute"
    CATEGORY = "ArchAi3d/Utils"

    def _compute_cache_key(self, image, backbone, resolution, fx, fy):
        """Compute a unique cache key based on inputs."""
        # Sample image pixels for fast hashing (corners + center)
        with torch.no_grad():
            img_cpu = image.detach().cpu()
            if img_cpu.dim() == 4:
                img_cpu = img_cpu[0]  # Take first image
            h, w = img_cpu.shape[0], img_cpu.shape[1]
            samples = [
                img_cpu[0, 0].mean().item(),
                img_cpu[0, -1].mean().item(),
                img_cpu[-1, 0].mean().item(),
                img_cpu[-1, -1].mean().item(),
                img_cpu[h//2, w//2].mean().item(),
            ]
            shape_str = f"{img_cpu.shape}"
            del img_cpu

        # Combine all parameters into hash
        hash_data = f"{shape_str}_{samples}_{backbone}_{resolution}_{fx}_{fy}"
        return hashlib.md5(hash_data.encode()).hexdigest()

    def _get_cached_result(self, cache_key):
        """Check if result exists in cache and load it."""
        index = _load_cache_index()
        if cache_key not in index:
            return None

        image_path = os.path.join(CACHE_DIR, f"{cache_key}_normal.png")
        if not os.path.exists(image_path):
            return None

        try:
            # Load cached image
            normal_image = Image.open(image_path).convert("RGB")
            result_tensor = pil2tensor(normal_image)
            print(f"[ArchAi3D Metric3D] Cache hit! Loaded from disk.")
            return (result_tensor,)

        except Exception as e:
            print(f"[ArchAi3D Metric3D] Cache load error: {e}")
            return None

    def _save_to_cache(self, cache_key, result_tensor):
        """Save result to disk cache."""
        try:
            image_path = os.path.join(CACHE_DIR, f"{cache_key}_normal.png")

            # Convert tensor to PIL and save
            if result_tensor.dim() == 4:
                result_tensor = result_tensor[0]
            result_np = (result_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            result_image = Image.fromarray(result_np)
            result_image.save(image_path, "PNG")

            # Update index
            index = _load_cache_index()
            index[cache_key] = {
                "timestamp": os.path.getmtime(image_path)
            }
            _save_cache_index(index)

            print(f"[ArchAi3D Metric3D] Saved to cache: {cache_key[:8]}...")

        except Exception as e:
            print(f"[ArchAi3D Metric3D] Cache save error: {e}")

    def execute(self, name, image, backbone="vit-small", resolution=512, use_cache=True,
                fx=1000, fy=1000):
        """
        Execute Metric3D normal map generation with caching and aggressive memory cleanup.
        """
        # Check cache first (if enabled)
        cache_key = None
        if use_cache:
            cache_key = self._compute_cache_key(image, backbone, resolution, fx, fy)
            cached = self._get_cached_result(cache_key)
            if cached is not None:
                return cached

        # No cache hit - need to process
        print(f"[ArchAi3D Metric3D] Processing with {backbone}... (no cache hit)")

        model = None

        try:
            # Import Metric3D detector
            from custom_controlnet_aux.metric3d import Metric3DDetector

            # Load model
            model_filename = f"metric_depth_{backbone.replace('-', '_')}_800k.pth"
            model = Metric3DDetector.from_pretrained(filename=model_filename).to(model_management.get_torch_device())

            # Process each image in batch
            result_images = []

            for i in range(image.shape[0]):
                # Get single image as numpy
                single_img = image[i].cpu().numpy()
                single_img = (single_img * 255).clip(0, 255).astype(np.uint8)

                # Resize if needed
                if resolution != 512:
                    single_img = resize_image(single_img, resolution)

                # Run model - get normal map (index 1)
                depth_result, normal_result = model(single_img, fx=fx, fy=fy, depth_and_normal=True)

                # Convert result to tensor
                if isinstance(normal_result, np.ndarray):
                    result_tensor = torch.from_numpy(normal_result.astype(np.float32) / 255.0)
                else:
                    result_tensor = normal_result.float() / 255.0

                if result_tensor.dim() == 3:
                    result_tensor = result_tensor.unsqueeze(0)

                result_images.append(result_tensor)

            # Combine results
            output = torch.cat(result_images, dim=0)

            # Save to cache (only first image for batch)
            if use_cache and cache_key:
                self._save_to_cache(cache_key, output[0])

            return (output,)

        except ImportError as e:
            print(f"[ArchAi3D Metric3D] Error: Could not import Metric3DDetector.")
            print(f"[ArchAi3D Metric3D] Make sure comfyui_controlnet_aux is installed.")
            raise e

        finally:
            # ALWAYS cleanup - aggressive memory management
            print("[ArchAi3D Metric3D] Unloading model and freeing GPU memory...")

            if model is not None:
                # Move to CPU first, then delete
                try:
                    model.cpu()
                except:
                    pass
                del model

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Force garbage collection
            gc.collect()

            print("[ArchAi3D Metric3D] GPU memory freed!")

    @classmethod
    def IS_CHANGED(cls, name, image, backbone, resolution, use_cache, **kwargs):
        """Return unique value to control caching behavior."""
        if not use_cache:
            # Always re-run if cache disabled
            return float("nan")
        return ""


class ArchAi3D_Metric3D_Depth:
    """
    Optimized Metric3D Depth Map for Low VRAM Systems.

    Same features as Normal Map but outputs depth instead.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {
                    "default": "metric3d_depth",
                    "multiline": False,
                    "tooltip": "Identifier name for this input (used by web interface)"
                }),
                "image": ("IMAGE",),
                "backbone": (["vit-small", "vit-large", "vit-giant2"], {
                    "default": "vit-small",
                    "tooltip": "Model backbone. vit-small uses less VRAM (~1-2GB), vit-giant2 uses most (~4-6GB)"
                }),
                "resolution": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": MAX_RESOLUTION,
                    "step": 64,
                    "tooltip": "Processing resolution. Lower = faster + less VRAM"
                }),
                "use_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use disk cache to avoid reprocessing identical inputs"
                }),
            },
            "optional": {
                "fx": ("INT", {
                    "default": 1000,
                    "min": 1,
                    "max": MAX_RESOLUTION,
                    "tooltip": "Focal length X (camera intrinsic)"
                }),
                "fy": ("INT", {
                    "default": 1000,
                    "min": 1,
                    "max": MAX_RESOLUTION,
                    "tooltip": "Focal length Y (camera intrinsic)"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth_map",)
    FUNCTION = "execute"
    CATEGORY = "ArchAi3d/Utils"

    def _compute_cache_key(self, image, backbone, resolution, fx, fy):
        """Compute a unique cache key based on inputs."""
        with torch.no_grad():
            img_cpu = image.detach().cpu()
            if img_cpu.dim() == 4:
                img_cpu = img_cpu[0]
            h, w = img_cpu.shape[0], img_cpu.shape[1]
            samples = [
                img_cpu[0, 0].mean().item(),
                img_cpu[0, -1].mean().item(),
                img_cpu[-1, 0].mean().item(),
                img_cpu[-1, -1].mean().item(),
                img_cpu[h//2, w//2].mean().item(),
            ]
            shape_str = f"{img_cpu.shape}"
            del img_cpu

        hash_data = f"{shape_str}_{samples}_{backbone}_{resolution}_{fx}_{fy}_depth"
        return hashlib.md5(hash_data.encode()).hexdigest()

    def _get_cached_result(self, cache_key):
        """Check if result exists in cache and load it."""
        index = _load_cache_index()
        if cache_key not in index:
            return None

        image_path = os.path.join(CACHE_DIR, f"{cache_key}_depth.png")
        if not os.path.exists(image_path):
            return None

        try:
            depth_image = Image.open(image_path).convert("RGB")
            result_tensor = pil2tensor(depth_image)
            print(f"[ArchAi3D Metric3D Depth] Cache hit! Loaded from disk.")
            return (result_tensor,)

        except Exception as e:
            print(f"[ArchAi3D Metric3D Depth] Cache load error: {e}")
            return None

    def _save_to_cache(self, cache_key, result_tensor):
        """Save result to disk cache."""
        try:
            image_path = os.path.join(CACHE_DIR, f"{cache_key}_depth.png")

            if result_tensor.dim() == 4:
                result_tensor = result_tensor[0]
            result_np = (result_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            result_image = Image.fromarray(result_np)
            result_image.save(image_path, "PNG")

            index = _load_cache_index()
            index[cache_key] = {"timestamp": os.path.getmtime(image_path)}
            _save_cache_index(index)

            print(f"[ArchAi3D Metric3D Depth] Saved to cache: {cache_key[:8]}...")

        except Exception as e:
            print(f"[ArchAi3D Metric3D Depth] Cache save error: {e}")

    def execute(self, name, image, backbone="vit-small", resolution=512, use_cache=True,
                fx=1000, fy=1000):
        """
        Execute Metric3D depth map generation with caching and aggressive memory cleanup.
        """
        cache_key = None
        if use_cache:
            cache_key = self._compute_cache_key(image, backbone, resolution, fx, fy)
            cached = self._get_cached_result(cache_key)
            if cached is not None:
                return cached

        print(f"[ArchAi3D Metric3D Depth] Processing with {backbone}... (no cache hit)")

        model = None

        try:
            from custom_controlnet_aux.metric3d import Metric3DDetector

            model_filename = f"metric_depth_{backbone.replace('-', '_')}_800k.pth"
            model = Metric3DDetector.from_pretrained(filename=model_filename).to(model_management.get_torch_device())

            result_images = []

            for i in range(image.shape[0]):
                single_img = image[i].cpu().numpy()
                single_img = (single_img * 255).clip(0, 255).astype(np.uint8)

                if resolution != 512:
                    single_img = resize_image(single_img, resolution)

                # Get depth map (index 0)
                depth_result, normal_result = model(single_img, fx=fx, fy=fy, depth_and_normal=True)

                if isinstance(depth_result, np.ndarray):
                    result_tensor = torch.from_numpy(depth_result.astype(np.float32) / 255.0)
                else:
                    result_tensor = depth_result.float() / 255.0

                if result_tensor.dim() == 3:
                    result_tensor = result_tensor.unsqueeze(0)

                result_images.append(result_tensor)

            output = torch.cat(result_images, dim=0)

            if use_cache and cache_key:
                self._save_to_cache(cache_key, output[0])

            return (output,)

        except ImportError as e:
            print(f"[ArchAi3D Metric3D Depth] Error: Could not import Metric3DDetector.")
            print(f"[ArchAi3D Metric3D Depth] Make sure comfyui_controlnet_aux is installed.")
            raise e

        finally:
            print("[ArchAi3D Metric3D Depth] Unloading model and freeing GPU memory...")

            if model is not None:
                try:
                    model.cpu()
                except:
                    pass
                del model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            gc.collect()

            print("[ArchAi3D Metric3D Depth] GPU memory freed!")

    @classmethod
    def IS_CHANGED(cls, name, image, backbone, resolution, use_cache, **kwargs):
        """Return unique value to control caching behavior."""
        if not use_cache:
            return float("nan")
        return ""
