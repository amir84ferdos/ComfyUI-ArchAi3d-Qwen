"""
ArchAi3D SAM3 Segment (Low VRAM Optimized)

An optimized version of SAM3 Segmentation with:
- Disk-based output caching (avoids reprocessing)
- Aggressive memory cleanup (model unloaded after EVERY execution)
- torch.cuda.empty_cache() + gc.collect() for low VRAM systems
- name field for web interface integration
- Standalone: SAM3 model code bundled (no comfyui-rmbg dependency)

Based on: comfyui-rmbg/AILab_SAM3Segment.py
"""

import os
import sys
import gc
import json
import hashlib
from contextlib import nullcontext

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.hub import download_url_to_file

import folder_paths
import comfy.model_management

# SAM3 is now bundled with this node - no external dependency needed
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SAM3_LOCAL_DIR = os.path.join(CURRENT_DIR, "sam3")

# Cache directory for this node
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "sam3")
os.makedirs(CACHE_DIR, exist_ok=True)

# Cache index file
CACHE_INDEX_FILE = os.path.join(CACHE_DIR, "cache_index.json")


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
        print(f"[ArchAi3D SAM3] Warning: Could not save cache index: {e}")


def pil2tensor(image):
    """Convert PIL image to tensor."""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def tensor2pil(tensor):
    """Convert tensor to PIL image."""
    return Image.fromarray(np.clip(255. * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def process_mask(mask_image, invert_output=False, mask_blur=0, mask_offset=0):
    """Process mask with blur, offset, and inversion."""
    if invert_output:
        mask_np = np.array(mask_image)
        mask_image = Image.fromarray(255 - mask_np)
    if mask_blur > 0:
        mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=mask_blur))
    if mask_offset != 0:
        filt = ImageFilter.MaxFilter if mask_offset > 0 else ImageFilter.MinFilter
        size = abs(mask_offset) * 2 + 1
        for _ in range(abs(mask_offset)):
            mask_image = mask_image.filter(filt(size))
    return mask_image


def apply_background_color(image, mask_image, background="Alpha", background_color="#222222"):
    """Apply background color or alpha to image based on mask."""
    rgba_image = image.copy().convert("RGBA")
    rgba_image.putalpha(mask_image.convert("L"))
    if background == "Color":
        hex_color = background_color.lstrip("#")
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        bg_image = Image.new("RGBA", image.size, (r, g, b, 255))
        composite = Image.alpha_composite(bg_image, rgba_image)
        return composite.convert("RGB")
    return rgba_image


def _resolve_device(user_choice):
    """Resolve device based on user choice."""
    auto_device = comfy.model_management.get_torch_device()
    if user_choice == "CPU":
        return torch.device("cpu")
    if user_choice == "GPU":
        if auto_device.type != "cuda":
            raise RuntimeError("GPU unavailable")
        return torch.device("cuda")
    return auto_device


class ArchAi3D_SAM3_Segment:
    """
    Optimized SAM3 Segmentation for Low VRAM Systems.

    Features:
    - Disk-based output caching (PNG files)
    - Model unloaded after EVERY execution
    - torch.cuda.empty_cache() + gc.collect()
    - name field for web interface
    """

    # SAM3 model info
    SAM3_MODEL_URL = "https://huggingface.co/1038lab/sam3/resolve/main/sam3.pt"
    SAM3_FILENAME = "sam3.pt"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {
                    "default": "sam3_segment",
                    "multiline": False,
                    "tooltip": "Identifier name for this input (used by web interface)"
                }),
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Describe what to segment (e.g., 'person', 'car', 'background')"
                }),
                "confidence_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.05,
                    "max": 0.95,
                    "step": 0.01,
                    "tooltip": "Confidence threshold for segmentation"
                }),
                "use_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use disk cache to avoid reprocessing identical inputs"
                }),
            },
            "optional": {
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
                "mask_offset": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1}),
                "invert_output": ("BOOLEAN", {"default": False}),
                "background": (["Alpha", "Color"], {"default": "Alpha"}),
                "background_color": ("STRING", {"default": "#222222"}),
                "device": (["Auto", "CPU", "GPU"], {"default": "Auto"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("IMAGE", "MASK", "MASK_IMAGE")
    FUNCTION = "segment"
    CATEGORY = "ArchAi3d/Utils"

    def _compute_cache_key(self, image, prompt, confidence, mask_blur, mask_offset, invert, background, bg_color):
        """Compute a unique cache key based on inputs."""
        # Sample image pixels for fast hashing (corners + center)
        with torch.no_grad():
            img_cpu = image.detach().cpu()
            h, w = img_cpu.shape[1], img_cpu.shape[2]
            samples = [
                img_cpu[0, 0, 0].mean().item(),
                img_cpu[0, 0, -1].mean().item(),
                img_cpu[0, -1, 0].mean().item(),
                img_cpu[0, -1, -1].mean().item(),
                img_cpu[0, h//2, w//2].mean().item(),
            ]
            shape_str = f"{img_cpu.shape}"
            del img_cpu

        # Combine all parameters into hash
        hash_data = f"{shape_str}_{samples}_{prompt}_{confidence}_{mask_blur}_{mask_offset}_{invert}_{background}_{bg_color}"
        return hashlib.md5(hash_data.encode()).hexdigest()

    def _get_cached_result(self, cache_key):
        """Check if result exists in cache and load it."""
        index = _load_cache_index()
        if cache_key not in index:
            return None

        entry = index[cache_key]
        image_path = os.path.join(CACHE_DIR, f"{cache_key}_image.png")
        mask_path = os.path.join(CACHE_DIR, f"{cache_key}_mask.png")

        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            return None

        try:
            # Load cached images
            result_image = Image.open(image_path)
            mask_image = Image.open(mask_path)

            # Convert back to tensors
            if entry.get("has_alpha", False):
                result_image = result_image.convert("RGBA")
            else:
                result_image = result_image.convert("RGB")

            result_tensor = pil2tensor(result_image)
            mask_tensor = torch.from_numpy(np.array(mask_image.convert("L")).astype(np.float32) / 255.0).unsqueeze(0)

            # PIL size is (width, height), so w, h = size
            w, h = mask_image.size
            mask_rgb = mask_tensor.reshape((-1, 1, h, w)).movedim(1, -1).expand(-1, -1, -1, 3)

            print(f"[ArchAi3D SAM3] Cache hit! Loaded from disk.")
            return result_tensor, mask_tensor, mask_rgb

        except Exception as e:
            print(f"[ArchAi3D SAM3] Cache load error: {e}")
            return None

    def _save_to_cache(self, cache_key, result_image, mask_image, has_alpha):
        """Save result to disk cache."""
        try:
            image_path = os.path.join(CACHE_DIR, f"{cache_key}_image.png")
            mask_path = os.path.join(CACHE_DIR, f"{cache_key}_mask.png")

            result_image.save(image_path, "PNG")
            mask_image.save(mask_path, "PNG")

            # Update index
            index = _load_cache_index()
            index[cache_key] = {
                "has_alpha": has_alpha,
                "timestamp": os.path.getmtime(image_path)
            }
            _save_cache_index(index)

            print(f"[ArchAi3D SAM3] Saved to cache: {cache_key[:8]}...")

        except Exception as e:
            print(f"[ArchAi3D SAM3] Cache save error: {e}")

    def _get_model_path(self):
        """Get or download SAM3 model file."""
        # Check folder_paths first
        local_path = None
        if hasattr(folder_paths, "get_full_path"):
            local_path = folder_paths.get_full_path("sam3", self.SAM3_FILENAME)
        if local_path and os.path.isfile(local_path):
            return local_path

        # Fall back to models directory
        base_models_dir = getattr(folder_paths, "models_dir", os.path.join(CURRENT_DIR, "models"))
        models_dir = os.path.join(base_models_dir, "sam3")
        os.makedirs(models_dir, exist_ok=True)
        local_path = os.path.join(models_dir, self.SAM3_FILENAME)

        if not os.path.exists(local_path):
            print(f"[ArchAi3D SAM3] Downloading model from {self.SAM3_MODEL_URL}...")
            download_url_to_file(self.SAM3_MODEL_URL, local_path)

        return local_path

    def _load_model(self, device_str):
        """Load SAM3 model (will be unloaded after use)."""
        # Add SAM3 path if not already there
        if SAM3_LOCAL_DIR not in sys.path:
            sys.path.insert(0, SAM3_LOCAL_DIR)

        # Check for BPE file
        bpe_path = os.path.join(SAM3_LOCAL_DIR, "assets", "bpe_simple_vocab_16e6.txt.gz")
        if not os.path.isfile(bpe_path):
            raise RuntimeError(f"SAM3 assets missing; ensure {bpe_path} exists.")

        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        ckpt_path = self._get_model_path()

        model = build_sam3_image_model(
            bpe_path=bpe_path,
            device=device_str,
            eval_mode=True,
            checkpoint_path=ckpt_path,
            load_from_HF=False,
            enable_segmentation=True,
            enable_inst_interactivity=False,
        )

        processor = Sam3Processor(model, device=device_str)
        return model, processor

    def _empty_result(self, img_pil, background, background_color):
        """Return empty result when no segmentation found."""
        w, h = img_pil.size
        mask_image = Image.new("L", (w, h), 0)
        result_image = apply_background_color(img_pil, mask_image, background, background_color)
        if background == "Alpha":
            result_image = result_image.convert("RGBA")
        else:
            result_image = result_image.convert("RGB")
        empty_mask = torch.zeros((1, h, w), dtype=torch.float32)
        mask_rgb = empty_mask.reshape((-1, 1, h, w)).movedim(1, -1).expand(-1, -1, -1, 3)
        return result_image, mask_image, empty_mask, mask_rgb

    def segment(self, name, image, prompt, confidence_threshold=0.5, use_cache=True,
                mask_blur=0, mask_offset=0, invert_output=False,
                background="Alpha", background_color="#222222", device="Auto"):
        """
        Execute SAM3 segmentation with caching and aggressive memory cleanup.
        """
        if image.ndim == 3:
            image = image.unsqueeze(0)

        # Check cache first (if enabled)
        cache_key = None
        if use_cache:
            cache_key = self._compute_cache_key(
                image, prompt, confidence_threshold, mask_blur, mask_offset,
                invert_output, background, background_color
            )
            cached = self._get_cached_result(cache_key)
            if cached is not None:
                return cached

        # No cache hit - need to process
        print(f"[ArchAi3D SAM3] Processing... (no cache hit)")

        torch_device = _resolve_device(device)
        device_str = "cuda" if torch_device.type == "cuda" else "cpu"

        model = None
        processor = None

        try:
            # Load model
            model, processor = self._load_model(device_str)

            autocast_device = comfy.model_management.get_autocast_device(torch_device)
            autocast_enabled = torch_device.type == "cuda" and not comfy.model_management.is_device_mps(torch_device)
            ctx = torch.autocast(autocast_device, dtype=torch.bfloat16) if autocast_enabled else nullcontext()

            result_images = []
            result_masks = []
            result_mask_images = []

            with ctx:
                for tensor_img in image:
                    img_pil = tensor2pil(tensor_img)
                    text = prompt.strip() or "object"

                    state = processor.set_image(img_pil)
                    processor.reset_all_prompts(state)
                    processor.set_confidence_threshold(confidence_threshold, state)
                    state = processor.set_text_prompt(text, state)

                    masks = state.get("masks")
                    if masks is None or masks.numel() == 0:
                        result_image, mask_image, mask_tensor, mask_rgb = self._empty_result(
                            img_pil, background, background_color
                        )
                    else:
                        masks = masks.float().to("cpu")
                        if masks.ndim == 4:
                            masks = masks.squeeze(1)
                        combined = masks.amax(dim=0)
                        mask_np = (combined.clamp(0, 1).numpy() * 255).astype(np.uint8)
                        mask_image = Image.fromarray(mask_np, mode="L")
                        mask_image = process_mask(mask_image, invert_output, mask_blur, mask_offset)

                        result_image = apply_background_color(img_pil, mask_image, background, background_color)
                        if background == "Alpha":
                            result_image = result_image.convert("RGBA")
                        else:
                            result_image = result_image.convert("RGB")

                        mask_tensor = torch.from_numpy(np.array(mask_image).astype(np.float32) / 255.0).unsqueeze(0)
                        mask_rgb = mask_tensor.reshape((-1, 1, mask_image.height, mask_image.width)).movedim(1, -1).expand(-1, -1, -1, 3)

                    result_images.append(pil2tensor(result_image))
                    result_masks.append(mask_tensor)
                    result_mask_images.append(mask_rgb)

                    # Save to cache (only first image for now)
                    if use_cache and cache_key and len(result_images) == 1:
                        self._save_to_cache(
                            cache_key, result_image, mask_image,
                            has_alpha=(background == "Alpha")
                        )

            return (
                torch.cat(result_images, dim=0),
                torch.cat(result_masks, dim=0),
                torch.cat(result_mask_images, dim=0)
            )

        finally:
            # ALWAYS cleanup - aggressive memory management
            print("[ArchAi3D SAM3] Unloading model and freeing GPU memory...")

            if processor is not None:
                del processor

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

            print("[ArchAi3D SAM3] GPU memory freed!")

    @classmethod
    def IS_CHANGED(cls, name, image, prompt, confidence_threshold, use_cache, **kwargs):
        """Return unique value to control caching behavior."""
        if not use_cache:
            # Always re-run if cache disabled
            return float("nan")
        return ""
