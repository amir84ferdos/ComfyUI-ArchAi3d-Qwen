"""
Offload CLIP to DRAM — Move text encoder weights from VRAM to CPU RAM.

Place this node after CLIPTextEncode. Connect the CONDITIONING output to 'trigger'
input — it passes through to 'passthrough' output so you can wire it to KSampler.
The CLIP stays cached in DRAM for fast reload on the next workflow run.

Wiring: CLIPTextEncode ──CONDITIONING──→ [trigger] Offload CLIP [passthrough] ──CONDITIONING──→ KSampler

Author: Amir Ferdos (ArchAi3d)
"""

import time
from . import dram_cache


class ArchAi3D_Offload_CLIP:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {
                    "tooltip": "CLIP text encoder to offload from VRAM to DRAM"
                }),
            },
            "optional": {
                "trigger": ("*", {
                    "tooltip": "Connect CLIPTextEncode CONDITIONING output here — it passes through to 'passthrough' output"
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "*")
    RETURN_NAMES = ("memory_stats", "dram_id", "passthrough")
    OUTPUT_TOOLTIPS = (
        "Current VRAM/RAM/cache status",
        "Cache key for this CLIP in DRAM (matches loader's key)",
        "Pass-through of trigger input (e.g. CONDITIONING → connect to KSampler)"
    )
    FUNCTION = "offload"
    CATEGORY = "ArchAi3d/Memory"
    DESCRIPTION = "Move CLIP weights from VRAM to DRAM. Passes trigger input through to passthrough output (e.g. CONDITIONING from CLIPTextEncode → KSampler)."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time()

    def offload(self, clip, trigger=None):
        cache_key = getattr(clip, "_dram_cache_key", None)
        if cache_key is None:
            cache_key = f"clip:{clip.patcher.model.__class__.__name__}:{id(clip)}"
            clip._dram_cache_key = cache_key

        dram_cache.store(cache_key, clip, obj_type="clip")
        stats = dram_cache.get_memory_stats()

        # Pass through trigger data (CONDITIONING, etc.) so downstream nodes can use it
        passthrough = trigger if trigger is not None else True
        return (stats, cache_key, passthrough)
