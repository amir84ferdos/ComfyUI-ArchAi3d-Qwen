"""
Memory Cleanup — Clear DRAM cache, VRAM, and CUDA cache.

Place at the beginning or end of your workflow to manually manage memory.
Use this when switching between very different workflows or when system
RAM is running low.

Author: Amir Ferdos (ArchAi3d)
"""

import gc
import time
import comfy.model_management as mm
from . import dram_cache


class ArchAi3D_Memory_Cleanup:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clear_dram_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Remove all models from DRAM cache. Forces disk reload on next use."
                }),
                "clear_vram": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Unload all models from GPU VRAM."
                }),
                "clear_cuda_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Clear PyTorch CUDA cache to reclaim fragmented VRAM."
                }),
            },
        }

    RETURN_TYPES = ("STRING", "*")
    RETURN_NAMES = ("memory_stats", "trigger")
    OUTPUT_TOOLTIPS = (
        "Memory status after cleanup",
        "Connect to next node to control execution order"
    )
    FUNCTION = "cleanup"
    CATEGORY = "ArchAi3d/Memory"
    DESCRIPTION = "Clear DRAM cache, VRAM, and CUDA cache. Use when switching workflows or freeing system RAM."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time()

    def cleanup(self, clear_dram_cache, clear_vram, clear_cuda_cache):
        actions = []

        if clear_dram_cache:
            dram_cache.clear()
            actions.append("DRAM cache cleared")

        if clear_vram:
            mm.unload_all_models()
            actions.append("VRAM models unloaded")

        if clear_cuda_cache:
            mm.soft_empty_cache()
            gc.collect()
            actions.append("CUDA cache cleared")

        if actions:
            print(f"[Memory Cleanup] {', '.join(actions)}")
        else:
            print("[Memory Cleanup] No actions selected")

        stats = dram_cache.get_memory_stats()
        return (stats, True)
