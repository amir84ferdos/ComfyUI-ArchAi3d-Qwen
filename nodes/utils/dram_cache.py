"""
DRAM Cache — Persistent CPU RAM model storage for ComfyUI.

Keeps models in system DRAM (CPU RAM) between workflow runs for fast reload
to GPU. Models stored here won't be garbage collected and survive across
ComfyUI execution cycles.

Usage from other nodes:
    from ..utils.dram_cache import store, get, clear, get_memory_stats

Author: Amir Ferdos (ArchAi3d)
"""

import gc
import time
import psutil
import torch
import comfy.model_management as mm

# Module-level cache — persists across workflow runs until ComfyUI restarts
_cache = {}  # {cache_key: {"obj": model_or_clip, "stored_at": timestamp, "type": "model"|"clip"}}
_vram_checked = False


def _check_vram_state():
    """Warn once if ComfyUI's VRAM mode could interfere with manual DRAM management."""
    global _vram_checked
    if _vram_checked:
        return
    _vram_checked = True

    vram_state = mm.vram_state
    # VRAMState enum: HIGH_VRAM=0, NORMAL_VRAM=1, LOW_VRAM=2, NO_VRAM=3, SHARED=4
    state_name = vram_state.name if hasattr(vram_state, 'name') else str(vram_state)

    if vram_state == mm.VRAMState.LOW_VRAM:
        print(f"[DRAM Cache] WARNING: ComfyUI is in LOW_VRAM mode — it may move weights "
              f"to CPU on its own, conflicting with manual DRAM control. "
              f"Recommended: --normalvram --cache-classic")
    elif vram_state == mm.VRAMState.NO_VRAM:
        print(f"[DRAM Cache] WARNING: ComfyUI is in NO_VRAM mode — CPU-only processing. "
              f"DRAM cache has no effect. Recommended: --normalvram --cache-classic")
    elif vram_state == mm.VRAMState.HIGH_VRAM:
        print(f"[DRAM Cache] WARNING: ComfyUI is in HIGH_VRAM/GPU_ONLY mode — it keeps "
              f"everything on GPU and may resist offloading. "
              f"Recommended: --normalvram --cache-classic")
    else:
        print(f"[DRAM Cache] VRAM mode: {state_name} — compatible with manual memory management")


def store(key, obj, obj_type="model"):
    """Store a model/clip in DRAM cache and move its weights to CPU.

    Args:
        key: Cache key (e.g. "unet:flux-dev-fp8:fp8_e4m3fn")
        obj: MODEL (ModelPatcher) or CLIP object
        obj_type: "model" or "clip"
    """
    _check_vram_state()

    # Get the patcher (CLIP wraps it in .patcher)
    if obj_type == "clip":
        patcher = obj.patcher
    else:
        patcher = obj

    # Move all loaded weights from VRAM to CPU
    device = patcher.offload_device
    loaded = patcher.loaded_size()
    if loaded > 0:
        freed = patcher.partially_unload(device, loaded)
        print(f"[DRAM Cache] Stored '{key}': freed {freed / (1024**2):.0f} MB VRAM → CPU RAM")
    else:
        print(f"[DRAM Cache] Stored '{key}': model already on CPU")

    # Clean up fragmented VRAM
    mm.soft_empty_cache()

    # Store strong reference (prevents GC)
    _cache[key] = {
        "obj": obj,
        "stored_at": time.time(),
        "type": obj_type,
    }


def get(key):
    """Get a cached model/clip by key, or None if not cached."""
    entry = _cache.get(key)
    if entry is not None:
        return entry["obj"]
    return None


def remove(key):
    """Remove a specific model from DRAM cache."""
    entry = _cache.pop(key, None)
    if entry is not None:
        print(f"[DRAM Cache] Removed '{key}'")
        return True
    return False


def clear():
    """Clear all models from DRAM cache and run garbage collection."""
    count = len(_cache)
    _cache.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[DRAM Cache] Cleared {count} cached model(s)")


def list_cached():
    """List all cached model keys with metadata."""
    result = []
    for key, entry in _cache.items():
        obj = entry["obj"]
        obj_type = entry["type"]
        if obj_type == "clip":
            patcher = obj.patcher
        else:
            patcher = obj

        size_mb = 0
        try:
            size_mb = patcher.model_size() / (1024 ** 2)
        except Exception:
            pass

        result.append({
            "key": key,
            "type": obj_type,
            "size_mb": size_mb,
            "stored_at": entry["stored_at"],
        })
    return result


def get_memory_stats():
    """Build formatted string of current VRAM/RAM stats and cache contents."""
    device = mm.get_torch_device()

    # VRAM stats
    vram_free = mm.get_free_memory(device) / (1024 ** 3)
    vram_total = mm.get_total_memory(device) / (1024 ** 3)
    vram_used = vram_total - vram_free

    # RAM stats
    ram = psutil.virtual_memory()
    ram_free = ram.available / (1024 ** 3)
    ram_total = ram.total / (1024 ** 3)
    ram_used = (ram.total - ram.available) / (1024 ** 3)

    lines = [
        "=" * 50,
        "MEMORY STATUS",
        "=" * 50,
        f"VRAM: {vram_used:.2f} / {vram_total:.2f} GB  (free: {vram_free:.2f} GB)",
        f"RAM:  {ram_used:.1f} / {ram_total:.1f} GB  (free: {ram_free:.1f} GB)",
        "",
    ]

    # DRAM Cache contents
    cached = list_cached()
    if cached:
        lines.append(f"DRAM Cache ({len(cached)} model(s)):")
        for entry in cached:
            elapsed = time.time() - entry["stored_at"]
            mins = int(elapsed // 60)
            lines.append(f"  [{entry['type'].upper()}] {entry['key']}  "
                         f"({entry['size_mb']:.0f} MB, cached {mins}m ago)")
    else:
        lines.append("DRAM Cache: empty")

    lines.append("")

    # Currently loaded models in ComfyUI
    loaded = mm.current_loaded_models
    if loaded:
        lines.append(f"GPU Loaded Models ({len(loaded)}):")
        for i, lm in enumerate(loaded):
            patcher = lm.model
            if patcher is None:
                continue
            try:
                name = patcher.model.__class__.__name__
            except Exception:
                name = "Unknown"
            try:
                loaded_mb = lm.model_loaded_memory() / (1024 ** 2)
                offloaded_mb = lm.model_offloaded_memory() / (1024 ** 2)
            except Exception:
                loaded_mb = offloaded_mb = 0
            lines.append(f"  [{i}] {name}: {loaded_mb:.0f} MB VRAM, {offloaded_mb:.0f} MB offloaded")
    else:
        lines.append("GPU Loaded Models: none")

    lines.append("=" * 50)
    return "\n".join(lines)
