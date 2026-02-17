"""
Offload Model to DRAM — Move diffusion model weights from VRAM to CPU RAM.

Place this node after KSampler. Connect KSampler's LATENT output to 'trigger'
input — it passes through to 'passthrough' output so you can wire it to VAE Decode.
The model stays cached in DRAM for fast reload on the next workflow run.

Wiring: KSampler ──LATENT──→ [trigger] Offload Model [passthrough] ──LATENT──→ VAE Decode

Author: Amir Ferdos (ArchAi3d)
"""

import time
from . import dram_cache


class ArchAi3D_Offload_Model:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Diffusion model to offload from VRAM to DRAM"
                }),
            },
            "optional": {
                "trigger": ("*", {
                    "tooltip": "Connect KSampler LATENT output here — it passes through to 'passthrough' output"
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "*")
    RETURN_NAMES = ("memory_stats", "dram_id", "passthrough")
    OUTPUT_TOOLTIPS = (
        "Current VRAM/RAM/cache status",
        "Cache key for this model in DRAM (matches loader's key)",
        "Pass-through of trigger input (e.g. LATENT → connect to VAE Decode)"
    )
    FUNCTION = "offload"
    CATEGORY = "ArchAi3d/Memory"
    DESCRIPTION = "Move model weights from VRAM to DRAM. Passes trigger input through to passthrough output (e.g. LATENT from KSampler → VAE Decode)."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time()

    def offload(self, model, trigger=None):
        cache_key = getattr(model, "_dram_cache_key", None)
        if cache_key is None:
            cache_key = f"model:{model.model.__class__.__name__}:{id(model)}"
            model._dram_cache_key = cache_key

        dram_cache.store(cache_key, model, obj_type="model")
        stats = dram_cache.get_memory_stats()

        # Pass through trigger data (LATENT, etc.) so downstream nodes can use it
        passthrough = trigger if trigger is not None else True
        return (stats, cache_key, passthrough)
