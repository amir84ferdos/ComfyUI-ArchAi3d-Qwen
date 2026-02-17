"""
Triggered Loader Nodes for ComfyUI
Load diffusion models and CLIP with trigger inputs for execution order control.
Supports DRAM cache for fast model reload across workflow runs.
Supports VRAM pinning to keep light models on GPU permanently.

Author: Amir Ferdos (ArchAi3d)
"""

import time
import torch
import folder_paths
import comfy.sd
import comfy.model_management as mm
from ..utils.local_model_cache import copy_to_local
from ..utils.dram_cache import (
    get as dram_get, evict_previous, get_memory_stats,
    pin_model, unpin_model, is_pinned, ensure_free_memory,
)


class ArchAi3D_Load_Diffusion_Model:
    """Load diffusion model (UNET) with trigger input for execution order control.

    Connect the trigger input to another node's output to ensure this loader
    runs after that node completes. Useful for chaining after QwenVL Server Control.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],),
            },
            "optional": {
                "trigger": ("*", {
                    "tooltip": "Connect to any output to ensure this node runs after that node completes"
                }),
                "keep_on_vram": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Keep model on GPU VRAM permanently. Skips all DRAM cache logic. Good for light models that fit in VRAM."
                }),
                "use_dram": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Check DRAM cache for previously offloaded model. Much faster than disk reload."
                }),
                "replace_cached": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "When switching models, evict THIS loader's previous model from DRAM first. Saves RAM. Turn off to keep multiple models cached (needs lots of RAM)."
                }),
                "cache_to_local_ssd": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "RunPod: copy model to local SSD for faster loading. No effect on local PC."
                }),
                "auto_free_vram": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Before loading: if free VRAM is below threshold, unpin all models and let ComfyUI free VRAM."
                }),
                "min_free_vram_gb": ("FLOAT", {
                    "default": 10.0, "min": 0.0, "max": 80.0, "step": 0.5,
                    "tooltip": "Minimum free VRAM (GB) required. If below this, auto-free kicks in."
                }),
                "auto_free_dram": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Before loading: if free RAM is below threshold, clear DRAM cache to free system memory."
                }),
                "min_free_dram_gb": ("FLOAT", {
                    "default": 10.0, "min": 0.0, "max": 256.0, "step": 1.0,
                    "tooltip": "Minimum free RAM (GB) required. If below this, DRAM cache is cleared."
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "memory_stats")
    FUNCTION = "load_unet"
    CATEGORY = "ArchAi3D/Loaders"
    DESCRIPTION = """Load a diffusion model (UNET) with memory management and execution order control.

[Parameters]
trigger: Connect to any output to force execution order (e.g., run after QwenVL Server Control).
keep_on_vram: Pin model to GPU VRAM permanently. Protected from ComfyUI auto-eviction. Skips DRAM cache. Best for light models that fit in VRAM alongside other models. If you change the model name, the old one is automatically unpinned.
use_dram: Check DRAM cache (CPU RAM) for previously loaded model. Much faster than disk reload (~1s vs 10-30s). Model stays in RAM between runs.
replace_cached: When switching to a different model, evict THIS loader's previous model from DRAM first. Saves RAM. Turn off to keep multiple models cached (needs lots of RAM).
cache_to_local_ssd: RunPod only — copy model file from network drive to local NVMe for faster loading. No effect on local PC.
auto_free_vram: Safety net for serverless. Before loading from disk, checks free VRAM. If below min_free_vram_gb, unpins ALL pinned models and asks ComfyUI to free VRAM. Only triggers when loading a DIFFERENT model (same model reuses cache). Does nothing if memory is already sufficient.
min_free_vram_gb: Threshold in GB for auto_free_vram. Example: 10 means "if less than 10 GB free VRAM, free up first".
auto_free_dram: Safety net for serverless. Before loading from disk, checks free system RAM. If below min_free_dram_gb, clears ALL DRAM cache entries. Only triggers on DRAM miss (same model returns from cache without checking).
min_free_dram_gb: Threshold in GB for auto_free_dram. Example: 10 means "if less than 10 GB free RAM, clear DRAM cache first"."""

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if kwargs.get("keep_on_vram", False):
            return "v5"  # Static — let ComfyUI execution cache handle it
        if kwargs.get("use_dram", True):
            return time.time()  # Force re-execution to always check DRAM
        return "v5"

    def load_unet(self, unet_name, weight_dtype, trigger=None, keep_on_vram=False, use_dram=True, replace_cached=True, cache_to_local_ssd=True, auto_free_vram=False, min_free_vram_gb=10.0, auto_free_dram=False, min_free_dram_gb=10.0):
        # keep_on_vram: smart VRAM pinning — load to GPU, pin, detect model change
        if keep_on_vram:
            vram_key = f"unet:{unet_name}:{weight_dtype}"
            last_key = getattr(self, "_vram_key", None)
            last_model = getattr(self, "_vram_model", None)

            # Same model, already pinned → return cached (no reload)
            if last_key == vram_key and last_model is not None and is_pinned(last_model):
                print(f"[VRAM PIN] {unet_name} already pinned on GPU, reusing")
                return (last_model, get_memory_stats())

            # Model changed → unpin the old one
            if last_model is not None and last_key != vram_key:
                unpin_model(last_model)
                print(f"[VRAM PIN] Unpinned old model: {last_key}")

            # Auto-free memory if needed before loading
            ensure_free_memory(min_free_vram_gb, min_free_dram_gb, auto_free_vram, auto_free_dram)

            # Load new model
            model_options = {}
            if weight_dtype == "fp8_e4m3fn":
                model_options["dtype"] = torch.float8_e4m3fn
            elif weight_dtype == "fp8_e4m3fn_fast":
                model_options["dtype"] = torch.float8_e4m3fn
                model_options["fp8_optimizations"] = True
            elif weight_dtype == "fp8_e5m2":
                model_options["dtype"] = torch.float8_e5m2
            unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
            unet_path = copy_to_local(unet_path, enabled=cache_to_local_ssd)
            model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)

            # Pin to VRAM — protected from ComfyUI auto-eviction
            pin_model(model)
            self._vram_key = vram_key
            self._vram_model = model
            return (model, get_memory_stats())

        # If switching from keep_on_vram to DRAM mode, unpin old model
        old_vram_model = getattr(self, "_vram_model", None)
        if old_vram_model is not None:
            unpin_model(old_vram_model)
            self._vram_model = None
            self._vram_key = None
            print(f"[VRAM PIN] Switched to DRAM mode, unpinned previous model")

        # DRAM mode
        cache_key = f"unet:{unet_name}:{weight_dtype}"

        # Evict this loader's previous model if switching to a different one
        if use_dram and replace_cached:
            evict_previous(getattr(self, "_last_cache_key", None), cache_key)
        self._last_cache_key = cache_key

        # Check DRAM cache first (fast reload from CPU RAM)
        if use_dram:
            cached = dram_get(cache_key)
            if cached is not None:
                print(f"[DRAM HIT] {unet_name} loaded from DRAM | dram_id: {cache_key}")
                return (cached, get_memory_stats())
            print(f"[DRAM MISS] {unet_name} not in DRAM cache, loading from disk | dram_id: {cache_key}")

        # Auto-free memory if needed before loading from disk
        ensure_free_memory(min_free_vram_gb, min_free_dram_gb, auto_free_vram, auto_free_dram)

        # Load from disk (slow path) — SSD cache is independent
        model_options = {}

        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        unet_path = copy_to_local(unet_path, enabled=cache_to_local_ssd)
        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
        model._dram_cache_key = cache_key
        return (model, get_memory_stats())


class ArchAi3D_Load_CLIP:
    """Load CLIP text encoder with trigger input for execution order control.

    Supports all CLIP types including:
    - stable_diffusion: clip-l
    - stable_cascade: clip-g
    - sd3: t5 xxl / clip-g / clip-l
    - flux: clip-l / t5 xxl (use DualCLIPLoader for both)
    - qwen_image: Qwen VL models
    - omnigen2: qwen vl 2.5 3B
    - And many more...

    Connect the trigger input to another node's output to ensure this loader
    runs after that node completes. Useful for chaining after QwenVL Server Control.
    """

    # All supported CLIP types (matching ComfyUI's CLIPLoader)
    CLIP_TYPES = [
        "stable_diffusion",
        "stable_cascade",
        "sd3",
        "stable_audio",
        "mochi",
        "ltxv",
        "pixart",
        "cosmos",
        "lumina2",
        "wan",
        "hidream",
        "chroma",
        "ace",
        "omnigen2",
        "qwen_image",
        "hunyuan_image",
        "flux2",
        "ovis",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name": (folder_paths.get_filename_list("text_encoders"),),
                "type": (cls.CLIP_TYPES,),
            },
            "optional": {
                "trigger": ("*", {
                    "tooltip": "Connect to any output to ensure this node runs after that node completes"
                }),
                "keep_on_vram": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Keep CLIP on GPU VRAM permanently. Skips all DRAM cache logic. Good for light models that fit in VRAM."
                }),
                "use_dram": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Check DRAM cache for previously offloaded CLIP. Much faster than disk reload."
                }),
                "replace_cached": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "When switching CLIPs, evict THIS loader's previous CLIP from DRAM first. Saves RAM. Turn off to keep multiple CLIPs cached."
                }),
                "device": (["default", "cpu"], {
                    "tooltip": "Device to load CLIP on. Use 'cpu' to save VRAM."
                }),
                "cache_to_local_ssd": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "RunPod: copy model to local SSD for faster loading. No effect on local PC."
                }),
                "auto_free_vram": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Before loading: if free VRAM is below threshold, unpin all models and let ComfyUI free VRAM."
                }),
                "min_free_vram_gb": ("FLOAT", {
                    "default": 10.0, "min": 0.0, "max": 80.0, "step": 0.5,
                    "tooltip": "Minimum free VRAM (GB) required. If below this, auto-free kicks in."
                }),
                "auto_free_dram": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Before loading: if free RAM is below threshold, clear DRAM cache to free system memory."
                }),
                "min_free_dram_gb": ("FLOAT", {
                    "default": 10.0, "min": 0.0, "max": 256.0, "step": 1.0,
                    "tooltip": "Minimum free RAM (GB) required. If below this, DRAM cache is cleared."
                }),
            }
        }

    RETURN_TYPES = ("CLIP", "STRING")
    RETURN_NAMES = ("clip", "memory_stats")
    FUNCTION = "load_clip"
    CATEGORY = "ArchAi3D/Loaders"
    DESCRIPTION = """Load CLIP text encoder with memory management and execution order control.

[Parameters]
trigger: Connect to any output to force execution order.
keep_on_vram: Pin CLIP to GPU VRAM permanently. Protected from auto-eviction. If you change the CLIP, the old one is automatically unpinned.
use_dram: Check DRAM cache (CPU RAM) for previously loaded CLIP. Much faster than disk reload.
replace_cached: When switching CLIPs, evict this loader's previous CLIP from DRAM. Saves RAM.
device: Load CLIP on GPU (default) or CPU. Use CPU to save VRAM for diffusion model.
cache_to_local_ssd: RunPod only — copy to local NVMe for faster loading.
auto_free_vram: Before loading from disk, if free VRAM < threshold, unpin all models and free VRAM. Only triggers for different model (same model reuses cache).
min_free_vram_gb: VRAM threshold in GB for auto_free_vram.
auto_free_dram: Before loading from disk, if free RAM < threshold, clear DRAM cache. Only triggers on DRAM miss.
min_free_dram_gb: RAM threshold in GB for auto_free_dram.

[Recipes]
stable_diffusion: clip-l
stable_cascade: clip-g
sd3: t5 xxl / clip-g / clip-l
stable_audio: t5 base
mochi: t5 xxl
cosmos: old t5 xxl
lumina2: gemma 2 2B
wan: umt5 xxl
hidream: llama-3.1 or t5
omnigen2: qwen vl 2.5 3B
qwen_image: Qwen VL models"""

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if kwargs.get("keep_on_vram", False):
            return "v5"
        if kwargs.get("use_dram", True):
            return time.time()
        return "v5"

    def load_clip(self, clip_name, type, trigger=None, keep_on_vram=False, use_dram=True, replace_cached=True, device="default", cache_to_local_ssd=True, auto_free_vram=False, min_free_vram_gb=10.0, auto_free_dram=False, min_free_dram_gb=10.0):
        # keep_on_vram: smart VRAM pinning — load to GPU, pin, detect model change
        if keep_on_vram:
            vram_key = f"clip:{clip_name}:{type}"
            last_key = getattr(self, "_vram_key", None)
            last_clip = getattr(self, "_vram_clip", None)

            # Same CLIP, already pinned → return cached (no reload)
            if last_key == vram_key and last_clip is not None and is_pinned(last_clip.patcher):
                print(f"[VRAM PIN] {clip_name} already pinned on GPU, reusing")
                return (last_clip, get_memory_stats())

            # Model changed → unpin the old one
            if last_clip is not None and last_key != vram_key:
                unpin_model(last_clip.patcher)
                print(f"[VRAM PIN] Unpinned old CLIP: {last_key}")

            # Auto-free memory if needed before loading
            ensure_free_memory(min_free_vram_gb, min_free_dram_gb, auto_free_vram, auto_free_dram)

            # Load new CLIP
            clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
            model_options = {}
            if device == "cpu":
                model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
            clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
            clip_path = copy_to_local(clip_path, enabled=cache_to_local_ssd)
            clip = comfy.sd.load_clip(
                ckpt_paths=[clip_path],
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                clip_type=clip_type,
                model_options=model_options
            )

            # Pin to VRAM — protected from ComfyUI auto-eviction
            pin_model(clip.patcher)
            self._vram_key = vram_key
            self._vram_clip = clip
            return (clip, get_memory_stats())

        # If switching from keep_on_vram to DRAM mode, unpin old CLIP
        old_vram_clip = getattr(self, "_vram_clip", None)
        if old_vram_clip is not None:
            unpin_model(old_vram_clip.patcher)
            self._vram_clip = None
            self._vram_key = None
            print(f"[VRAM PIN] Switched to DRAM mode, unpinned previous CLIP")

        # DRAM mode
        cache_key = f"clip:{clip_name}:{type}"

        # Evict this loader's previous CLIP if switching to a different one
        if use_dram and replace_cached:
            evict_previous(getattr(self, "_last_cache_key", None), cache_key)
        self._last_cache_key = cache_key

        # Check DRAM cache first (fast reload from CPU RAM)
        if use_dram:
            cached = dram_get(cache_key)
            if cached is not None:
                print(f"[DRAM HIT] {clip_name} loaded from DRAM | dram_id: {cache_key}")
                return (cached, get_memory_stats())
            print(f"[DRAM MISS] {clip_name} not in DRAM cache, loading from disk | dram_id: {cache_key}")

        # Auto-free memory if needed before loading from disk
        ensure_free_memory(min_free_vram_gb, min_free_dram_gb, auto_free_vram, auto_free_dram)

        # Load from disk (slow path) — SSD cache is independent
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)

        model_options = {}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")

        clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
        clip_path = copy_to_local(clip_path, enabled=cache_to_local_ssd)
        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
            model_options=model_options
        )
        clip._dram_cache_key = cache_key
        return (clip, get_memory_stats())


class ArchAi3D_Load_Dual_CLIP:
    """Load two CLIP text encoders with trigger input for execution order control.

    For models that require dual CLIP (SDXL, SD3, Flux, etc.):
    - sdxl: clip-l + clip-g
    - sd3: clip-l + clip-g / clip-l + t5 / clip-g + t5
    - flux: clip-l + t5
    - hidream: t5 + llama (recommended)
    - hunyuan_image: qwen2.5vl 7b + byt5 small
    - newbie: gemma-3-4b-it + jina clip v2

    Connect the trigger input to another node's output to ensure this loader
    runs after that node completes.
    """

    # Dual CLIP types (matching ComfyUI's DualCLIPLoader)
    DUAL_CLIP_TYPES = [
        "sdxl",
        "sd3",
        "flux",
        "hunyuan_video",
        "hidream",
        "hunyuan_image",
        "hunyuan_video_15",
        "kandinsky5",
        "kandinsky5_image",
        "ltxv",
        "newbie",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name1": (folder_paths.get_filename_list("text_encoders"),),
                "clip_name2": (folder_paths.get_filename_list("text_encoders"),),
                "type": (cls.DUAL_CLIP_TYPES,),
            },
            "optional": {
                "trigger": ("*", {
                    "tooltip": "Connect to any output to ensure this node runs after that node completes"
                }),
                "keep_on_vram": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Keep CLIP on GPU VRAM permanently. Skips all DRAM cache logic. Good for light models that fit in VRAM."
                }),
                "use_dram": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Check DRAM cache for previously offloaded CLIP. Much faster than disk reload."
                }),
                "replace_cached": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "When switching CLIPs, evict THIS loader's previous CLIP from DRAM first. Saves RAM. Turn off to keep multiple CLIPs cached."
                }),
                "device": (["default", "cpu"], {
                    "tooltip": "Device to load CLIP on. Use 'cpu' to save VRAM."
                }),
                "cache_to_local_ssd": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "RunPod: copy model to local SSD for faster loading. No effect on local PC."
                }),
                "auto_free_vram": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Before loading: if free VRAM is below threshold, unpin all models and let ComfyUI free VRAM."
                }),
                "min_free_vram_gb": ("FLOAT", {
                    "default": 10.0, "min": 0.0, "max": 80.0, "step": 0.5,
                    "tooltip": "Minimum free VRAM (GB) required. If below this, auto-free kicks in."
                }),
                "auto_free_dram": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Before loading: if free RAM is below threshold, clear DRAM cache to free system memory."
                }),
                "min_free_dram_gb": ("FLOAT", {
                    "default": 10.0, "min": 0.0, "max": 256.0, "step": 1.0,
                    "tooltip": "Minimum free RAM (GB) required. If below this, DRAM cache is cleared."
                }),
            }
        }

    RETURN_TYPES = ("CLIP", "STRING")
    RETURN_NAMES = ("clip", "memory_stats")
    FUNCTION = "load_clip"
    CATEGORY = "ArchAi3D/Loaders"
    DESCRIPTION = """Load two CLIP text encoders with memory management and execution order control.

[Parameters]
trigger: Connect to any output to force execution order.
keep_on_vram: Pin dual CLIP to GPU VRAM permanently. Protected from auto-eviction. If you change either CLIP, the old one is automatically unpinned.
use_dram: Check DRAM cache (CPU RAM) for previously loaded dual CLIP. Much faster than disk reload.
replace_cached: When switching CLIPs, evict this loader's previous CLIP from DRAM. Saves RAM.
device: Load CLIP on GPU (default) or CPU. Use CPU to save VRAM for diffusion model.
cache_to_local_ssd: RunPod only — copy to local NVMe for faster loading.
auto_free_vram: Before loading from disk, if free VRAM < threshold, unpin all models and free VRAM. Only triggers for different model (same model reuses cache).
min_free_vram_gb: VRAM threshold in GB for auto_free_vram.
auto_free_dram: Before loading from disk, if free RAM < threshold, clear DRAM cache. Only triggers on DRAM miss.
min_free_dram_gb: RAM threshold in GB for auto_free_dram.

[Recipes]
sdxl: clip-l, clip-g
sd3: clip-l + clip-g / clip-l + t5 / clip-g + t5
flux: clip-l, t5
hidream: t5 + llama (recommended)
hunyuan_image: qwen2.5vl 7b, byt5 small
newbie: gemma-3-4b-it, jina clip v2"""

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if kwargs.get("keep_on_vram", False):
            return "v5"
        if kwargs.get("use_dram", True):
            return time.time()
        return "v5"

    def load_clip(self, clip_name1, clip_name2, type, trigger=None, keep_on_vram=False, use_dram=True, replace_cached=True, device="default", cache_to_local_ssd=True, auto_free_vram=False, min_free_vram_gb=10.0, auto_free_dram=False, min_free_dram_gb=10.0):
        # keep_on_vram: smart VRAM pinning — load to GPU, pin, detect model change
        if keep_on_vram:
            vram_key = f"dualclip:{clip_name1}:{clip_name2}:{type}"
            last_key = getattr(self, "_vram_key", None)
            last_clip = getattr(self, "_vram_clip", None)

            # Same CLIP, already pinned → return cached (no reload)
            if last_key == vram_key and last_clip is not None and is_pinned(last_clip.patcher):
                print(f"[VRAM PIN] {clip_name1}+{clip_name2} already pinned on GPU, reusing")
                return (last_clip, get_memory_stats())

            # Model changed → unpin the old one
            if last_clip is not None and last_key != vram_key:
                unpin_model(last_clip.patcher)
                print(f"[VRAM PIN] Unpinned old Dual CLIP: {last_key}")

            # Auto-free memory if needed before loading
            ensure_free_memory(min_free_vram_gb, min_free_dram_gb, auto_free_vram, auto_free_dram)

            # Load new Dual CLIP
            clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
            model_options = {}
            if device == "cpu":
                model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
            clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", clip_name1)
            clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", clip_name2)
            clip_path1 = copy_to_local(clip_path1, enabled=cache_to_local_ssd)
            clip_path2 = copy_to_local(clip_path2, enabled=cache_to_local_ssd)
            clip = comfy.sd.load_clip(
                ckpt_paths=[clip_path1, clip_path2],
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                clip_type=clip_type,
                model_options=model_options
            )

            # Pin to VRAM — protected from ComfyUI auto-eviction
            pin_model(clip.patcher)
            self._vram_key = vram_key
            self._vram_clip = clip
            return (clip, get_memory_stats())

        # If switching from keep_on_vram to DRAM mode, unpin old CLIP
        old_vram_clip = getattr(self, "_vram_clip", None)
        if old_vram_clip is not None:
            unpin_model(old_vram_clip.patcher)
            self._vram_clip = None
            self._vram_key = None
            print(f"[VRAM PIN] Switched to DRAM mode, unpinned previous Dual CLIP")

        # DRAM mode
        cache_key = f"dualclip:{clip_name1}:{clip_name2}:{type}"

        # Evict this loader's previous CLIP if switching to a different one
        if use_dram and replace_cached:
            evict_previous(getattr(self, "_last_cache_key", None), cache_key)
        self._last_cache_key = cache_key

        # Check DRAM cache first (fast reload from CPU RAM)
        if use_dram:
            cached = dram_get(cache_key)
            if cached is not None:
                print(f"[DRAM HIT] {clip_name1}+{clip_name2} loaded from DRAM | dram_id: {cache_key}")
                return (cached, get_memory_stats())
            print(f"[DRAM MISS] {clip_name1}+{clip_name2} not in DRAM, loading from disk | dram_id: {cache_key}")

        # Auto-free memory if needed before loading from disk
        ensure_free_memory(min_free_vram_gb, min_free_dram_gb, auto_free_vram, auto_free_dram)

        # Load from disk (slow path) — SSD cache is independent
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)

        clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", clip_name1)
        clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", clip_name2)
        clip_path1 = copy_to_local(clip_path1, enabled=cache_to_local_ssd)
        clip_path2 = copy_to_local(clip_path2, enabled=cache_to_local_ssd)

        model_options = {}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")

        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path1, clip_path2],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
            model_options=model_options
        )
        clip._dram_cache_key = cache_key
        return (clip, get_memory_stats())


# Node mappings for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "ArchAi3D_Load_Diffusion_Model": ArchAi3D_Load_Diffusion_Model,
    "ArchAi3D_Load_CLIP": ArchAi3D_Load_CLIP,
    "ArchAi3D_Load_Dual_CLIP": ArchAi3D_Load_Dual_CLIP,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_Load_Diffusion_Model": "📦 Load Diffusion Model (Triggered)",
    "ArchAi3D_Load_CLIP": "📦 Load CLIP (Triggered)",
    "ArchAi3D_Load_Dual_CLIP": "📦 Load Dual CLIP (Triggered)",
}
