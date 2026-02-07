"""
Triggered Loader Nodes for ComfyUI
Load diffusion models and CLIP with trigger inputs for execution order control.

Author: Amir Ferdos (ArchAi3d)
"""

import torch
import folder_paths
import comfy.sd
from ..utils.local_model_cache import copy_to_local


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
                "cache_to_local_ssd": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "RunPod: copy model to local SSD for faster loading. No effect on local PC."
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_unet"
    CATEGORY = "ArchAi3D/Loaders"
    DESCRIPTION = "Load a diffusion model (UNET) with optional trigger for execution order control."

    def load_unet(self, unet_name, weight_dtype, trigger=None, cache_to_local_ssd=True):
        # Trigger is ignored - it only creates execution dependency
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
        return (model,)


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
                "device": (["default", "cpu"], {
                    "tooltip": "Device to load CLIP on. Use 'cpu' to save VRAM."
                }),
                "cache_to_local_ssd": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "RunPod: copy model to local SSD for faster loading. No effect on local PC."
                }),
            }
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "load_clip"
    CATEGORY = "ArchAi3D/Loaders"
    DESCRIPTION = """Load CLIP text encoder with trigger for execution order control.

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

    def load_clip(self, clip_name, type, trigger=None, device="default", cache_to_local_ssd=True):
        # Trigger is ignored - it only creates execution dependency
        # Use getattr for dynamic enum lookup (matches original ComfyUI)
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
        return (clip,)


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
                "device": (["default", "cpu"], {
                    "tooltip": "Device to load CLIP on. Use 'cpu' to save VRAM."
                }),
                "cache_to_local_ssd": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "RunPod: copy model to local SSD for faster loading. No effect on local PC."
                }),
            }
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "load_clip"
    CATEGORY = "ArchAi3D/Loaders"
    DESCRIPTION = """Load two CLIP text encoders with trigger for execution order control.

[Recipes]
sdxl: clip-l, clip-g
sd3: clip-l + clip-g / clip-l + t5 / clip-g + t5
flux: clip-l, t5
hidream: t5 + llama (recommended)
hunyuan_image: qwen2.5vl 7b, byt5 small
newbie: gemma-3-4b-it, jina clip v2"""

    def load_clip(self, clip_name1, clip_name2, type, trigger=None, device="default", cache_to_local_ssd=True):
        # Trigger is ignored - it only creates execution dependency
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
        return (clip,)


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
