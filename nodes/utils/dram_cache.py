"""
DRAM Cache — Persistent CPU RAM model storage for ComfyUI.

Keeps models in system DRAM (CPU RAM) between workflow runs for fast reload
to GPU. Models stored here won't be garbage collected and survive across
ComfyUI execution cycles.

Also provides VRAM pinning: protect specific models from ComfyUI's auto-eviction
when running with --normalvram.

Usage from other nodes:
    from ..utils.dram_cache import store, get, clear, get_memory_stats
    from ..utils.dram_cache import pin_model, unpin_model, unpin_all, is_pinned

Author: Amir Ferdos (ArchAi3d)
"""

import gc
import time
import weakref
import psutil
import torch
import comfy.model_management as mm

# Module-level cache — persists across workflow runs until ComfyUI restarts
_cache = {}  # {cache_key: {"obj": model_or_clip, "stored_at": timestamp, "type": "model"|"clip"}}
_vram_checked = False

# ── VRAM Pin System ──────────────────────────────────────────────────────────
# Models in this set are protected from ComfyUI's auto-eviction (free_memory).
# Uses WeakSet so pins auto-clean when models are garbage collected.
_pinned_patchers = weakref.WeakSet()
_pin_patch_installed = False


def _install_vram_pin_patch():
    """Monkey-patch free_memory to protect pinned models from eviction.

    ComfyUI's free_memory() already has a keep_loaded parameter that skips
    specific models during eviction. We inject our pinned models into that list.
    Lazy-installed on first pin_model() call — zero overhead if never used.
    """
    global _pin_patch_installed
    if _pin_patch_installed:
        return
    _pin_patch_installed = True

    original_free_memory = mm.free_memory

    def _patched_free_memory(memory_required, device, keep_loaded=[], **kwargs):
        pinned_loaded = []
        for lm in mm.current_loaded_models:
            patcher = lm.model
            if patcher is not None and patcher in _pinned_patchers:
                pinned_loaded.append(lm)
        if pinned_loaded:
            keep_loaded = list(keep_loaded) + pinned_loaded
        return original_free_memory(memory_required, device, keep_loaded=keep_loaded, **kwargs)

    mm.free_memory = _patched_free_memory
    print("[VRAM PIN] Eviction protection patch installed")


def pin_model(patcher):
    """Pin a ModelPatcher to VRAM — protected from ComfyUI auto-eviction.

    Args:
        patcher: ModelPatcher object (for CLIP, pass clip.patcher)
    """
    _install_vram_pin_patch()
    _pinned_patchers.add(patcher)
    try:
        name = patcher.model.__class__.__name__
    except Exception:
        name = "Unknown"
    size_mb = patcher.model_size() / (1024 ** 2)
    print(f"[VRAM PIN] {name} pinned to GPU ({size_mb:.0f} MB, {len(_pinned_patchers)} pinned total)")


def unpin_model(patcher):
    """Unpin a ModelPatcher — allow ComfyUI to evict it normally."""
    _pinned_patchers.discard(patcher)


def is_pinned(patcher):
    """Check if a ModelPatcher is currently pinned to VRAM."""
    return patcher in _pinned_patchers


def unpin_all():
    """Remove all VRAM pins. Models become eligible for auto-eviction again."""
    count = len(_pinned_patchers)
    # WeakSet: iterate and discard to clear
    for p in list(_pinned_patchers):
        _pinned_patchers.discard(p)
    if count > 0:
        print(f"[VRAM PIN] Unpinned all {count} model(s)")


def get_pinned_count():
    """Return number of currently pinned models."""
    return len(_pinned_patchers)


def ensure_free_memory(min_free_vram_gb=0, min_free_dram_gb=0, auto_free_vram=False, auto_free_dram=False):
    """Check memory thresholds and free up if needed. Call before loading models.

    If free VRAM is below min_free_vram_gb: unpins all VRAM-pinned models and
    asks ComfyUI to evict until the threshold is met.

    If free RAM is below min_free_dram_gb: clears all DRAM cache entries.

    Args:
        min_free_vram_gb: Minimum free VRAM in GB before auto-freeing.
        min_free_dram_gb: Minimum free RAM in GB before auto-freeing.
        auto_free_vram: Enable VRAM threshold check.
        auto_free_dram: Enable RAM threshold check.

    Returns:
        List of action strings describing what was freed.
    """
    actions = []

    if auto_free_vram and min_free_vram_gb > 0:
        device = mm.get_torch_device()
        free_vram_gb = mm.get_free_memory(device) / (1024 ** 3)
        if free_vram_gb < min_free_vram_gb:
            pinned_count = get_pinned_count()
            if pinned_count > 0:
                unpin_all()
            # Ask ComfyUI to free VRAM (now that pins are removed, it can evict)
            needed_bytes = int(min_free_vram_gb * (1024 ** 3))
            mm.free_memory(needed_bytes, device)
            mm.soft_empty_cache()
            new_free_gb = mm.get_free_memory(device) / (1024 ** 3)
            actions.append(
                f"VRAM: was {free_vram_gb:.1f} GB free (need {min_free_vram_gb:.1f}), "
                f"unpinned {pinned_count}, freed → {new_free_gb:.1f} GB free"
            )

    if auto_free_dram and min_free_dram_gb > 0:
        free_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
        if free_ram_gb < min_free_dram_gb:
            cached_count = len(_cache)
            if cached_count > 0:
                clear()
            new_free_gb = psutil.virtual_memory().available / (1024 ** 3)
            actions.append(
                f"RAM: was {free_ram_gb:.1f} GB free (need {min_free_dram_gb:.1f}), "
                f"cleared {cached_count} DRAM entries → {new_free_gb:.1f} GB free"
            )

    for a in actions:
        print(f"[AUTO-FREE] {a}")

    return actions


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
        gc.collect()
        return True
    return False


def evict_previous(previous_key, current_key):
    """Evict a specific previous cache entry if it differs from current.

    Called by loaders with replace_cached=True. Each loader tracks its own
    previous key, so this only evicts what THAT loader previously loaded —
    other loaders' cached models are untouched.

    Args:
        previous_key: The cache key this loader used last time (or None)
        current_key: The cache key this loader wants now

    Returns:
        True if something was evicted, False otherwise
    """
    if previous_key is None or previous_key == current_key:
        return False

    entry = _cache.pop(previous_key, None)
    if entry is not None:
        try:
            size_mb = 0
            obj = entry["obj"]
            patcher = obj.patcher if entry["type"] == "clip" else obj
            size_mb = patcher.model_size() / (1024 ** 2)
        except Exception:
            pass
        print(f"[DRAM Cache] Replaced: evicted '{previous_key}' ({size_mb:.0f} MB) → loading '{current_key}'")
        gc.collect()
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
            pinned_tag = " [PINNED]" if is_pinned(patcher) else ""
            lines.append(f"  [{i}] {name}: {loaded_mb:.0f} MB VRAM, {offloaded_mb:.0f} MB offloaded{pinned_tag}")
    else:
        lines.append("GPU Loaded Models: none")

    pinned_count = get_pinned_count()
    if pinned_count > 0:
        lines.append(f"VRAM Pinned: {pinned_count} model(s) protected from auto-eviction")

    lines.append("=" * 50)
    return "\n".join(lines)
