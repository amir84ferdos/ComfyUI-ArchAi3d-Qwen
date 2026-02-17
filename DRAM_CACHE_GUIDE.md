# DRAM Cache System — Developer & Maintenance Guide

**Version:** 1.5.0
**Date:** 2026-02-17
**Author:** Amir Ferdos (ArchAi3d)
**ComfyUI Version Tested:** v0.14.0

---

## Table of Contents

1. [What This Is](#what-this-is)
2. [Architecture](#architecture)
3. [Node Reference](#node-reference)
4. [keep_on_vram — GPU-Only Mode](#keep_on_vram--gpu-only-mode)
5. [auto_free_vram / auto_free_dram — Automatic Memory Cleanup](#auto_free_vram--auto_free_dram--automatic-memory-cleanup)
6. [replace_cached — Smart Model Switching](#replace_cached--smart-model-switching)
7. [RunPod Serverless](#runpod-serverless)
8. [ComfyUI APIs We Depend On](#comfyui-apis-we-depend-on)
9. [How to Update for New ComfyUI Versions](#how-to-update-for-new-comfyui-versions)
10. [Startup Flags](#startup-flags)
11. [Linux System Configuration](#linux-system-configuration)
12. [Troubleshooting](#troubleshooting)
13. [File Map](#file-map)

---

## What This Is

Manual VRAM/DRAM memory management for ComfyUI. Models are explicitly offloaded
from GPU VRAM to CPU RAM (DRAM) between pipeline stages, then reloaded from DRAM
on the next run (~1s vs ~8s from disk).

**Use case:** Workflows that exceed GPU VRAM (e.g. 14.7GB CLIP + 19.6GB UNET on
a 24GB GPU). You control exactly when each model is on GPU vs CPU.

**NOT a replacement for ComfyUI's built-in memory management.** This is an
explicit, manual system for power users who want full control.

---

## Architecture

### The Pipeline

```
[Load CLIP] ──→ [CLIPTextEncode] ──CONDITIONING──→ [Offload CLIP to DRAM]
  ↑ checks DRAM first                                  │
  │                                            ┌───────┴────────┐
  │                                  passthrough(CONDITIONING)  dram_id(STRING)
  │                                            ↓                │
  │                                       [KSampler]           trigger
  │                                            │                ↓
  │                                      LATENT output    [Load Diffusion Model]
  │                                            ↓              ↑ checks DRAM first
  │                                 [Offload Model to DRAM]    │
  │                                    │              │        │
  │                             passthrough(LATENT)  dram_id   │
  │                                    ↓                       │
  │                              [VAE Decode]                  │
  │                             (full VRAM available)          │
```

### Data Flow

1. **Load CLIP** checks DRAM cache → DRAM HIT (fast) or disk load (slow)
2. **CLIPTextEncode** runs with CLIP on GPU, produces CONDITIONING
3. **Offload CLIP** moves CLIP weights to CPU RAM, passes CONDITIONING through
4. **Load UNET** checks DRAM cache → loads into now-freed VRAM
5. **KSampler** runs with UNET on GPU, produces LATENT
6. **Offload Model** moves UNET weights to CPU RAM, passes LATENT through
7. **VAE Decode** has full VRAM available for decoding

### Key Design Decisions

- **Module-level dict** (`dram_cache._cache`) holds strong references — prevents
  Python garbage collection, persists across ComfyUI workflow runs
- **Cache keys are deterministic** from loader inputs — same model + dtype always
  produces the same key, so offload node's key matches loader's key
- **`partially_unload()` keeps model tracked** in `current_loaded_models` — when
  ComfyUI needs the model again, `load_models_gpu()` auto-reloads from CPU RAM
- **Passthrough `*` type** works like ComfyUI's Reroute node — data flows through
  unchanged while the node performs its side effect (offloading)
- **SSD cache is independent** — `cache_to_local_ssd` copies from network to
  local SSD (RunPod only), DRAM cache stores in CPU RAM. Both can be active.

---

## Node Reference

### Load Diffusion Model (Triggered) — `nodes/inputs/archai3d_triggered_loaders.py`

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| unet_name | COMBO | — | Model filename from `diffusion_models/` |
| weight_dtype | COMBO | default | fp8_e4m3fn, fp8_e4m3fn_fast, fp8_e5m2 |
| trigger | * | None | Optional execution order control |
| keep_on_vram | BOOLEAN | False | Pin model on GPU permanently. Protected from auto-eviction. Skips DRAM. |
| use_dram | BOOLEAN | True | Check DRAM cache before loading from disk |
| replace_cached | BOOLEAN | True | When switching models, evict THIS loader's previous model from DRAM |
| cache_to_local_ssd | BOOLEAN | True | RunPod: copy to local SSD |
| auto_free_vram | BOOLEAN | False | Before disk load: if free VRAM < threshold, unpin all + free VRAM |
| min_free_vram_gb | FLOAT | 10.0 | VRAM threshold (GB) for auto_free_vram |
| auto_free_dram | BOOLEAN | False | Before disk load: if free RAM < threshold, clear DRAM cache |
| min_free_dram_gb | FLOAT | 10.0 | RAM threshold (GB) for auto_free_dram |

| Output | Type | Description |
|--------|------|-------------|
| model | MODEL | Loaded diffusion model |
| memory_stats | STRING | Current VRAM/RAM/cache status |

**Cache key:** `"unet:{unet_name}:{weight_dtype}"`

### Load CLIP (Triggered) — same file

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| clip_name | COMBO | — | CLIP filename from `text_encoders/` |
| type | COMBO | — | stable_diffusion, flux, sd3, qwen_image, etc. |
| trigger | * | None | Optional execution order control |
| keep_on_vram | BOOLEAN | False | Pin CLIP on GPU permanently. Protected from auto-eviction. Skips DRAM. |
| use_dram | BOOLEAN | True | Check DRAM cache before loading from disk |
| replace_cached | BOOLEAN | True | When switching CLIPs, evict THIS loader's previous CLIP from DRAM |
| device | COMBO | default | "cpu" to save VRAM |
| cache_to_local_ssd | BOOLEAN | True | RunPod: copy to local SSD |
| auto_free_vram | BOOLEAN | False | Before disk load: if free VRAM < threshold, unpin all + free VRAM |
| min_free_vram_gb | FLOAT | 10.0 | VRAM threshold (GB) for auto_free_vram |
| auto_free_dram | BOOLEAN | False | Before disk load: if free RAM < threshold, clear DRAM cache |
| min_free_dram_gb | FLOAT | 10.0 | RAM threshold (GB) for auto_free_dram |

**Cache key:** `"clip:{clip_name}:{type}"`

### Load Dual CLIP (Triggered) — same file

Same inputs as Load CLIP, with two clip_name fields instead of one.

**Cache key:** `"dualclip:{clip_name1}:{clip_name2}:{type}"`

### Offload Model to DRAM — `nodes/utils/archai3d_offload_model.py`

| Input | Type | Description |
|-------|------|-------------|
| model | MODEL | Model to offload (required) |
| trigger | * | Connect KSampler LATENT output here (optional) |

| Output | Type | Description |
|--------|------|-------------|
| memory_stats | STRING | VRAM/RAM/cache status after offload |
| dram_id | STRING | Cache key for this model |
| passthrough | * | Pass-through of trigger input (e.g. LATENT) |

### Offload CLIP to DRAM — `nodes/utils/archai3d_offload_clip.py`

Same pattern, but INPUT is `clip: CLIP`. Accesses `clip.patcher` internally.

### Memory Cleanup — `nodes/utils/archai3d_memory_cleanup.py`

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| unpin_all_vram | BOOLEAN | True | Remove all VRAM pins (models become eligible for auto-eviction) |
| clear_dram_cache | BOOLEAN | True | Remove all models from DRAM (forces disk reload) |
| clear_vram | BOOLEAN | True | Unload all models from GPU |
| clear_cuda_cache | BOOLEAN | True | Clear PyTorch CUDA allocator cache |

---

## keep_on_vram — Smart VRAM Pinning

When `keep_on_vram=True`, the loader:
1. **Loads the model to GPU** (skips all DRAM cache logic)
2. **Pins it to VRAM** — protected from ComfyUI's auto-eviction
3. **Detects model changes** — if you switch model name, it unpins the old one first
4. **Reuses on re-run** — same model + same inputs → returns cached, no reload

The pin uses ComfyUI's own `keep_loaded` mechanism in `free_memory()` — the model
is added to the eviction protection list so ComfyUI skips it when freeing VRAM
for other models.

### How Eviction Protection Works

ComfyUI's `free_memory()` has a `keep_loaded` parameter. Models in that list are
skipped during eviction. A monkey-patch (lazy-installed on first `pin_model()` call)
injects our pinned models into this list. Uses `WeakSet` so pins auto-clean when
models are garbage collected.

### Smart Detection

| Scenario | What Happens |
|----------|-------------|
| Same model, already pinned | Returns cached instantly (no disk I/O) |
| Model name changed | Unpins old model, loads + pins new one |
| Toggle switched to `use_dram` mode | Unpins previous model, enters DRAM path |
| Memory Cleanup runs | `unpin_all_vram=True` removes all pins |

### When to Use

- **Light models** that fit alongside your main model in VRAM (e.g. small VAE, ControlNet)
- **Simple workflows** where everything fits in VRAM — no need for DRAM management
- **Mixed workflows** — heavy model uses DRAM mode, light model uses `keep_on_vram=True`

### Behavior Matrix

| keep_on_vram | use_dram | Behavior |
|:---:|:---:|---|
| **True** | (ignored) | GPU mode: load to VRAM, pin, protected from eviction. |
| False | **True** | DRAM mode (default): check DRAM cache, stamp key, work with offload nodes. |
| False | False | Disk mode: always load from disk, no DRAM check. |

### Console Output

```
[VRAM PIN] Eviction protection patch installed
[VRAM PIN] Flux pinned to GPU (9800 MB, 1 pinned total)
```

On re-run (same model):
```
[VRAM PIN] flux-dev-fp8.safetensors already pinned on GPU, reusing
```

On model switch:
```
[VRAM PIN] Unpinned old model: unet:flux-dev-fp8.safetensors:fp8_e4m3fn
[VRAM PIN] SDXL pinned to GPU (6500 MB, 1 pinned total)
```

### What If VRAM Is Full?

If you pin too many models and VRAM runs out, the NEXT model that tries to load
will be loaded in lowvram mode (weights partially on CPU, slow inference). This
doesn't crash — it's ComfyUI's graceful degradation. Check memory_stats to see
how many models are pinned.

Use `auto_free_vram=True` to handle this automatically — see next section.

---

## auto_free_vram / auto_free_dram — Automatic Memory Cleanup

### The Problem

On RunPod serverless, different workflows can arrive on the same container. If
workflow A pinned a Flux model (or cached it in DRAM), and workflow B needs to
load a Qwen model, the old model is **orphaned** — still consuming VRAM/RAM but
no longer needed. Without cleanup:

- **VRAM pinned models:** ComfyUI can't evict them → new model gets partial-loaded (very slow) or CUDA OOM
- **DRAM cached models:** Old entries fill RAM → system starts swapping to disk or OOM killed

### How Auto-Free Works

Each loader has 4 inputs to handle this:

| Input | Default | What It Does |
|-------|---------|-------------|
| `auto_free_vram` | False | Before loading from disk: check free VRAM. If below threshold, unpin ALL models and ask ComfyUI to free VRAM. |
| `min_free_vram_gb` | 10.0 | VRAM threshold in GB. Example: 10 = "need at least 10 GB free" |
| `auto_free_dram` | False | Before loading from disk: check free system RAM. If below threshold, clear ALL DRAM cache entries. |
| `min_free_dram_gb` | 10.0 | RAM threshold in GB. Example: 10 = "need at least 10 GB free" |

### When It Triggers (and When It Doesn't)

Auto-free only runs **right before loading a NEW model from disk**. It does NOT
run when:

- Same model is already pinned in VRAM → returns cached (no disk load needed)
- Same model is in DRAM cache → DRAM HIT returns it (no disk load needed)
- Memory is already above the threshold → nothing to do

This means: **if the same model is already cached, auto-free never activates.**
It only kicks in when you're loading a genuinely different model AND memory is tight.

### Execution Order

```
1. Same model pinned?  → YES → return cached, skip everything
2. Same model in DRAM? → YES → DRAM HIT, skip everything
3. About to load from disk:
   a. auto_free_vram=True AND free_vram < threshold?
      → unpin_all() → mm.free_memory() → soft_empty_cache()
   b. auto_free_dram=True AND free_ram < threshold?
      → clear() (all DRAM entries removed)
4. Load model from disk
```

### Console Output

When auto-free triggers:
```
[AUTO-FREE] VRAM: was 3.2 GB free (need 10.0), unpinned 2, freed → 18.5 GB free
[AUTO-FREE] RAM: was 5.1 GB free (need 10.0), cleared 3 DRAM entries → 28.3 GB free
```

When memory is sufficient (nothing happens — no output):
```
[DRAM MISS] flux-dev not in DRAM cache, loading from disk
```

### Recommended Settings by Platform

| Platform | auto_free_vram | min_free_vram_gb | auto_free_dram | min_free_dram_gb |
|----------|:---:|---:|:---:|---:|
| Local PC (single workflow) | False | — | False | — |
| RunPod serverless (mixed workflows) | **True** | 10.0 | **True** | 10.0 |
| RunPod serverless (same workflow always) | False | — | False | — |

### Important: Auto-Free Is Destructive

When auto-free triggers, it clears **ALL** pins and **ALL** DRAM entries — not
just the ones from the previous workflow. This is intentional: we can't know
which orphaned entries belong to old workflows vs current ones. The trade-off:

- One-time cost: models reload from disk after cleanup (~8s per model)
- After that: DRAM cache repopulates, next run uses DRAM HIT (~1s)
- Without cleanup: CUDA OOM or system swap thrashing (much worse)

---

## replace_cached — Smart Model Switching

### The Problem Without It

Without `replace_cached`, switching from model_A to model_B in a loader node:
- model_A stays in DRAM (wasting RAM)
- model_B loads from disk and also gets cached in DRAM
- Now both sit in RAM — double the memory usage

Over time, switching models accumulates stale entries and fills RAM.

### How It Works

Each loader instance tracks its own `self._last_cache_key`. On every run:

1. Compare `current_key` with `self._last_cache_key`
2. If different → evict ONLY the previous entry from DRAM (`evict_previous()`)
3. If same → no-op (fast path, DRAM HIT)
4. Update `self._last_cache_key = current_key`

### Per-Instance Tracking (Multi-Loader Workflows)

Each loader node instance is independent. In a workflow with 3 Load Diffusion Model nodes:

```
Loader A:  _last_cache_key = "unet:flux-dev:fp8"
Loader B:  _last_cache_key = "unet:sd3-medium:default"
Loader C:  _last_cache_key = "unet:sdxl-base:default"
```

If you switch Loader A from `flux-dev` to `wan-fun`:
- Loader A evicts ONLY `"unet:flux-dev:fp8"` from DRAM
- Loaders B and C's cached models stay untouched
- Works because each instance has its own `self._last_cache_key`

### When to Turn Off

Set `replace_cached=False` when you want to accumulate multiple models in DRAM:
- You have lots of RAM (128GB+)
- Workflow uses multiple loaders that switch between models frequently
- You want all models pre-loaded in DRAM for fastest switching

### Console Output

When eviction happens, you'll see:
```
[DRAM Cache] Replaced: evicted 'unet:flux-dev:fp8' (19600 MB) → loading 'unet:wan-fun:fp8'
[DRAM MISS] wan-fun not in DRAM cache, loading from disk | dram_id: unet:wan-fun:fp8
```

---

## RunPod Serverless

### Architecture: Why It Works

RunPod serverless runs two processes in each container:
1. **ComfyUI** (`python main.py ...`) — background server, persistent
2. **RunPod handler** (`handler.py`) — foreground, talks to ComfyUI via HTTP/WebSocket

The handler never touches ComfyUI's Python state directly. ComfyUI runs as a
long-lived server — its module-level state (`dram_cache._cache = {}`) persists
for the entire container lifetime. This is exactly what the DRAM cache needs.

### Critical: GPU Optimizer Conflict

The `gpu_optimizer.py` auto-selects `--gpu-only` for GPUs with ≥16GB VRAM.
**This conflicts with DRAM cache** — `--gpu-only` keeps everything in VRAM and
prevents `partially_unload()` from moving weights to CPU.

**Required fix:** Set this environment variable on your RunPod endpoint:
```
COMFYUI_VRAM_MODE=normalvram
```

This overrides the gpu_optimizer and enables manual VRAM↔DRAM control.

### How DRAM Cache Behaves on RunPod

RunPod serverless containers have two states:

**Cold start** (new container or after idle timeout):
- Python process starts fresh
- `dram_cache._cache = {}` — empty
- `self._last_cache_key` doesn't exist on loader instances
- First request: DRAM MISS → loads from disk (~8s per model)
- Offload nodes store models in DRAM for next request

**Warm container** (subsequent requests, container reused):
- Same Python process, same memory
- `dram_cache._cache` still holds previous models
- `self._last_cache_key` remembers what was loaded
- DRAM HIT → reloads from CPU RAM (~1s per model)
- **This is the key performance win — 7s saved per model per request**

### Typical RunPod Request Lifecycle

```
Request 1 (cold):
  Load CLIP      → DRAM MISS → disk load (8s) → run → offload to DRAM
  Load UNET      → DRAM MISS → disk load (8s) → run → offload to DRAM
  Total overhead: ~16s model loading

Request 2 (warm):
  Load CLIP      → DRAM HIT → DRAM reload (1s)
  Load UNET      → DRAM HIT → DRAM reload (1s)
  Total overhead: ~2s model loading ← 8x faster

Request 3+ (warm, same models):
  Same as request 2 — instant DRAM reloads
```

### What Happens When the API Sends a Different Model

If your RunPod handler passes different model names per request:

**With `replace_cached=True` (default, recommended for RunPod):**
```
Request 1: Load flux-dev     → DRAM MISS → disk load → offload to DRAM
Request 2: Load flux-dev     → DRAM HIT → fast reload (1s)
Request 3: Load sd3-medium   → evict flux-dev → DRAM MISS → disk load → offload
Request 4: Load sd3-medium   → DRAM HIT → fast reload (1s)
```
- Only ONE model in DRAM at a time per loader slot
- RAM stays predictable
- No risk of OOM from accumulating stale models

**With `replace_cached=False` (for RunPod pods with lots of RAM):**
```
Request 1: Load flux-dev     → DRAM MISS → disk load → offload to DRAM
Request 2: Load sd3-medium   → DRAM MISS → disk load → offload to DRAM
Request 3: Load flux-dev     → DRAM HIT → fast reload (1s)  ← both cached!
Request 4: Load sd3-medium   → DRAM HIT → fast reload (1s)
```
- Multiple models accumulate in DRAM
- Fastest switching if you have the RAM
- Risk: RAM fills up if too many unique models are requested

### RunPod RAM Sizing

| GPU | Pod RAM | Recommended replace_cached | Reason |
|-----|---------|---------------------------|--------|
| RTX 4090 (24GB) | 48GB | True | ~20GB left for DRAM, fits one large model |
| RTX A6000 (48GB) | 96GB | True or False | ~50GB for DRAM, fits 2-3 models |
| H100 (80GB) | 200GB+ | False | Plenty of RAM for multiple models |
| RTX 5090 (32GB) | 64GB | True | ~35GB for DRAM, fits one CLIP+UNET pair |

### RunPod SSD Cache vs DRAM Cache

These two caches serve different purposes and work independently:

| Feature | SSD Cache (`cache_to_local_ssd`) | DRAM Cache (`use_dram`) |
|---------|----------------------------------|------------------------|
| What it does | Copies model from network drive to local SSD | Keeps model weights in CPU RAM |
| Speed improvement | ~2x (network → SSD I/O) | ~8x (disk → RAM) |
| Survives between requests | Yes (files persist on SSD) | Yes (Python process persists) |
| Survives container restart | No (SSD is ephemeral) | No (RAM is cleared) |
| RAM cost | None (stored on disk) | Full model size |
| Where it helps | First cold load (network → SSD) | All subsequent loads (DRAM → GPU) |

**Best practice: Enable both.** First request: network → SSD → GPU → DRAM.
Second request: DRAM → GPU (skips both disk reads entirely).

### Container Lifecycle and Cache Persistence

| Event | DRAM Cache State | Action Needed |
|-------|-----------------|---------------|
| Warm reuse (next job, same container) | **Preserved** — DRAM HIT | None |
| Idle timeout (default 5s) | **Lost** — container shuts down | Increase timeout or use Active Workers |
| FlashBoot revival | **Lost** — fast restart but RAM cleared | Cache repopulates on first request |
| Cold start (new container) | **Empty** — first load from disk | Normal — SSD cache helps |
| `REFRESH_WORKER=true` | **Lost after every job** | Never enable with DRAM cache |
| Endpoint updated | **Lost** — workers marked outdated | Expected — new code deployed |

**Recommendation for production:** Set **Active Workers ≥ 1** to keep the
container warm indefinitely. The DRAM cache persists across all requests.
For bursty workloads, increase **Idle Timeout to 60-300 seconds**.

### RunPod RAM Budget

Leave ~10GB headroom for OS, Python, ComfyUI overhead.

| GPU | System RAM | DRAM Cache Budget | Fits |
|-----|-----------|-------------------|------|
| RTX 4090 (24GB) | 41 GB | ~25-30 GB | 1 large model |
| RTX 5090 (32GB) | 35 GB | ~20-25 GB | 1 CLIP+UNET pair |
| L4 (24GB) | 50 GB | ~35-40 GB | 2 models |
| L40S (48GB) | 94 GB | ~75-80 GB | 3-4 models |
| A100 SXM (80GB) | 125 GB | ~100+ GB | Many models |
| H100/H200 | 125-276 GB | 100-250 GB | Full pipeline |

### RunPod Endpoint Environment Variables

```
COMFYUI_VRAM_MODE=normalvram     # Required — overrides gpu_optimizer's --gpu-only
COMFYUI_CACHE_MODE=classic       # Keeps node instances cached (self._last_cache_key persists)
COMFYUI_RESERVE_VRAM=0.5         # Small headroom for CUDA overhead
```

The gpu_optimizer handles everything else (attention, fast mode, etc.) automatically.

### RunPod Startup Flags

The `start.sh` script feeds env vars through `gpu_optimizer.py`, which outputs CLI flags.
With the env vars above, ComfyUI effectively runs:
```bash
python main.py --normalvram --cache-classic --use-sage-attention --fast fp16_accumulation --reserve-vram 0.5
```

No need for `vm.swappiness` — RunPod containers typically have plenty of RAM
and may not have swap enabled.

---

## ComfyUI APIs We Depend On

**This is the critical section for upgrades.** If any of these APIs change,
our nodes will break. Check these first when upgrading ComfyUI.

### 1. `ModelPatcher.partially_unload(device, amount)` — model_patcher.py

**What we use it for:** Moving model weights from GPU to CPU RAM.

```python
# In dram_cache.py store():
loaded = patcher.loaded_size()
freed = patcher.partially_unload(device, loaded)
```

**How it works internally:**
- Iterates through model modules, calls `m.to(device_to)` on each
- Stops when `current_size - target <= 0`
- Returns bytes freed
- Leaves model in `current_loaded_models` — important for auto-reload

**How to verify after upgrade:**
```python
# Check the method still exists and has same signature
import comfy.model_patcher
mp = comfy.model_patcher.ModelPatcher
print(mp.partially_unload.__doc__)
```

### 2. `ModelPatcher.loaded_size()` — model_patcher.py

**What we use it for:** Knowing how many bytes are on GPU (to unload all of them).

### 3. `ModelPatcher.offload_device` — model_patcher.py

**What we use it for:** Target device for unloading (usually `cpu`).

### 4. `ModelPatcher.model_size()` — model_patcher.py

**What we use it for:** Reporting model size in `get_memory_stats()`.

### 5. `mm.load_models_gpu([patcher])` — model_management.py

**What we use it for:** We DON'T call this directly. ComfyUI calls it automatically
when a downstream node needs the model. But this is what makes DRAM reload work —
it detects the model's weights are on CPU and copies them back to GPU.

**How to verify:** After upgrading, run a workflow with DRAM cache. If the model
reloads from DRAM (prints "[DRAM HIT]"), this API still works.

### 6. `mm.current_loaded_models` — model_management.py

**What we use it for:** `get_memory_stats()` lists currently loaded models.

```python
loaded = mm.current_loaded_models
for lm in loaded:
    patcher = lm.model
    name = patcher.model.__class__.__name__
    loaded_mb = lm.model_loaded_memory() / (1024 ** 2)
```

**Attributes we use on LoadedModel objects:**
- `lm.model` → the ModelPatcher
- `lm.model_loaded_memory()` → bytes on GPU
- `lm.model_offloaded_memory()` → bytes on CPU

### 7. `mm.soft_empty_cache()` — model_management.py

**What we use it for:** Cleaning up CUDA allocator fragmentation after offloading.

### 8. `mm.unload_all_models()` — model_management.py

**What we use it for:** Memory Cleanup node — force unload everything from GPU.

### 9. `mm.get_free_memory(device)` / `mm.get_total_memory(device)` — model_management.py

**What we use it for:** VRAM stats in `get_memory_stats()`.

### 10. `mm.vram_state` / `mm.VRAMState` — model_management.py

**What we use it for:** Detecting if ComfyUI is in a conflicting VRAM mode.

```python
# VRAMState enum values we check:
mm.VRAMState.LOW_VRAM    # Would conflict with manual control
mm.VRAMState.NO_VRAM     # CPU-only, DRAM cache has no effect
mm.VRAMState.HIGH_VRAM   # Keeps everything on GPU, resists offloading
```

### 11. `clip.patcher` — sd.py

**What we use it for:** Accessing the ModelPatcher inside a CLIP object.
CLIP wraps ModelPatcher as `self.patcher = CoreModelPatcher(...)`.

### 12. `comfy.sd.CLIPType` enum — sd.py

**What we use it for:** Converting clip type string to enum in Load CLIP.

```python
clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
```

**If new CLIP types are added:** Add them to `CLIP_TYPES` list in the loader class.

### 13. `comfy.sd.load_diffusion_model()` / `comfy.sd.load_clip()` — sd.py

**What we use it for:** Actually loading models from disk (the slow path).

### 14. `folder_paths.get_filename_list()` / `folder_paths.get_full_path_or_raise()` — folder_paths.py

**What we use it for:** Model file discovery and path resolution.

### 15. `mm.free_memory(memory_required, device, keep_loaded=[], ...)` — model_management.py

**What we use it for:** VRAM pin system monkey-patches this function to inject
pinned models into the `keep_loaded` list. Models in `keep_loaded` are skipped
during eviction (line 623: `if shift_model not in keep_loaded`).

```python
# In dram_cache.py _install_vram_pin_patch():
original_free_memory = mm.free_memory
def _patched_free_memory(memory_required, device, keep_loaded=[], **kwargs):
    # Find LoadedModels whose patcher is pinned
    pinned_loaded = [lm for lm in mm.current_loaded_models
                     if lm.model is not None and lm.model in _pinned_patchers]
    if pinned_loaded:
        keep_loaded = list(keep_loaded) + pinned_loaded
    return original_free_memory(memory_required, device, keep_loaded=keep_loaded, **kwargs)
mm.free_memory = _patched_free_memory
```

**If this API changes:** The VRAM pin system will fail to install the patch. Models
will load to GPU but won't be protected from eviction (graceful degradation).

### 16. ComfyUI Execution Cache — execution.py

**How it affects us:**
- ComfyUI caches node outputs between runs if inputs haven't changed
- With `IS_CHANGED` returning `time.time()` when `use_dram=True`, we force
  re-execution every run so the loader always checks DRAM
- With `IS_CHANGED` returning `"v4"` when `use_dram=False`, ComfyUI's execution
  cache handles caching (standard behavior)

**If execution cache behavior changes:**
Our nodes will still work but may not check DRAM on every run. Monitor if
`IS_CHANGED` contract changes.

---

## How to Update for New ComfyUI Versions

### Step-by-Step Upgrade Procedure

1. **Read the ComfyUI release notes** — look for:
   - Changes to `model_management.py`
   - Changes to `model_patcher.py` (especially `ModelPatcher` class)
   - Changes to `sd.py` (especially `CLIPType` enum, `load_clip`, `load_diffusion_model`)
   - Changes to `execution.py` (cache behavior, `IS_CHANGED` contract)
   - New VRAM management modes

2. **Check API compatibility:**
   ```bash
   cd /path/to/ComfyUI

   # Check ModelPatcher methods still exist
   grep -n "def partially_unload" comfy/model_patcher.py
   grep -n "def loaded_size" comfy/model_patcher.py
   grep -n "offload_device" comfy/model_patcher.py

   # Check model_management functions
   grep -n "def load_models_gpu" comfy/model_management.py
   grep -n "def soft_empty_cache" comfy/model_management.py
   grep -n "def unload_all_models" comfy/model_management.py
   grep -n "current_loaded_models" comfy/model_management.py
   grep -n "class VRAMState" comfy/model_management.py

   # Check CLIP structure
   grep -n "self.patcher" comfy/sd.py
   grep -n "class CLIPType" comfy/sd.py

   # Check for new CLIP types (add to CLIP_TYPES list if found)
   grep -n "class CLIPType" comfy/sd.py -A 30
   ```

3. **Quick smoke test:**
   ```bash
   # Start ComfyUI with DRAM flags
   python main.py --normalvram --cache-classic

   # Check startup log for:
   # - All archai3d nodes loaded without errors
   # - DRAM cache module initialized
   ```

4. **Run a DRAM workflow:**
   - Load the reference workflow: `workflows/DRAM_qwen_image_edit_2511.json`
   - First run: should see `[DRAM MISS]` (disk load) + offload messages
   - Second run: should see `[DRAM HIT]` (fast DRAM reload)
   - Check memory_stats output for reasonable numbers

### Common Breaking Changes to Watch For

| Change | Impact | Fix |
|--------|--------|-----|
| `partially_unload()` renamed/removed | Offload nodes break | Find new method for moving weights to CPU |
| `loaded_size()` renamed/removed | Store always reports 0 freed | Find new method for checking loaded bytes |
| `current_loaded_models` renamed | Stats reporting breaks | Find new list of tracked models |
| `VRAMState` enum changed | VRAM check warning breaks | Update enum references |
| New CLIPType values added | Can't load new CLIP types | Add to `CLIP_TYPES` and `DUAL_CLIP_TYPES` lists |
| `load_models_gpu()` behavior changed | DRAM reload stops working | Critical — may need new reload mechanism |
| `soft_empty_cache()` removed | Minor — VRAM fragmentation | Replace with `torch.cuda.empty_cache()` |
| Execution cache `IS_CHANGED` contract changed | DRAM check may not run every time | Adapt `IS_CHANGED` to new contract |
| `clip.patcher` attribute renamed | Offload CLIP breaks | Find new path to ModelPatcher inside CLIP |
| `model_loaded_memory()` / `model_offloaded_memory()` renamed | Stats inaccurate | Update attribute names in `get_memory_stats()` |

### Adding New CLIP Types

When ComfyUI adds new CLIP types (happens every few versions):

1. Check `comfy/sd.py` for new `CLIPType` enum values
2. Add to `CLIP_TYPES` list in `ArchAi3D_Load_CLIP` class (~line 104)
3. If it's a dual-CLIP type, add to `DUAL_CLIP_TYPES` in `ArchAi3D_Load_Dual_CLIP` (~line 221)
4. Bump `IS_CHANGED` version string (e.g. `"v4"` → `"v5"`) to force cache invalidation

### Version History of API Compatibility

| ComfyUI | Status | Notes |
|---------|--------|-------|
| v0.13.0 | Verified | Original development version |
| v0.14.0 | Verified | No breaking changes. FP8 LoRA fix. |

---

## Startup Flags

### Required Flags

```bash
python main.py --normalvram --cache-classic
```

| Flag | Why It's Needed |
|------|----------------|
| `--normalvram` | Prevents ComfyUI from using lowvram/highvram modes that conflict with manual offloading |
| `--cache-classic` | Uses simple execution cache (predictable behavior with `IS_CHANGED`) |

### Recommended Additional Flags

```bash
python main.py --normalvram --cache-classic --use-sage-attention --reserve-vram 0.5
```

| Flag | Why |
|------|-----|
| `--use-sage-attention` | Faster attention on supported GPUs (RTX 30xx+) |
| `--reserve-vram 0.5` | Keep 0.5GB VRAM headroom for CUDA overhead |

### Flags That Conflict (DO NOT USE with DRAM cache)

| Flag | Problem |
|------|---------|
| `--lowvram` | ComfyUI moves weights to CPU on its own, conflicts with manual control |
| `--highvram` | Keeps everything on GPU, resists offloading |
| `--gpu-only` | Same as highvram — **gpu_optimizer.py defaults to this on ≥16GB GPUs!** Override with `COMFYUI_VRAM_MODE=normalvram` |

### Start Script

See `start_comfyui_dram.sh` in the ComfyUI root directory.

---

## Linux System Configuration

### Swap Prevention (Critical)

Linux default `vm.swappiness=60` causes model memory pages to be swapped to
disk when DRAM is under pressure. This makes offloading trigger disk I/O.

**Check current value:**
```bash
cat /proc/sys/vm/swappiness
```

**Fix temporarily (until reboot):**
```bash
sudo sysctl vm.swappiness=10
```

**Fix permanently:**
```bash
echo "vm.swappiness=10" | sudo tee /etc/sysctl.d/99-comfyui-dram.conf
sudo sysctl -p /etc/sysctl.d/99-comfyui-dram.conf
```

**Why 10, not 0?**
- `0` = never swap (can cause OOM kills if RAM truly runs out)
- `10` = strongly prefer dropping page cache over swapping anonymous pages
- Models are anonymous pages — they should stay in RAM, not be swapped

### RAM Requirements

| Workflow | Minimum RAM | Recommended |
|----------|-------------|-------------|
| Single model (UNET only) | Model size + 8GB | 32GB |
| CLIP + UNET (sequential) | Larger model + 8GB | 48GB |
| Both cached in DRAM | CLIP + UNET + 8GB | 64GB |

**Example:** Qwen Image Edit workflow
- CLIP: ~14.7GB on disk (~15GB in RAM)
- UNET: ~19.6GB on disk (~20GB in RAM)
- Both in DRAM: ~35GB + system overhead → 48-64GB RAM recommended

### Check for Swap Issues

```bash
# Current swap usage
free -g

# Watch swap I/O in real-time (si/so columns should be 0)
vmstat 1

# What's using swap
cat /proc/meminfo | grep -i swap
```

---

## Troubleshooting

### RunPod: Offload has no effect / VRAM not freed

1. **Check VRAM mode:** The gpu_optimizer selects `--gpu-only` by default for ≥16GB GPUs
2. **Fix:** Set `COMFYUI_VRAM_MODE=normalvram` in RunPod endpoint env vars
3. **Verify:** Check startup logs for `[GPU-Optimizer] VRAM mode: --normalvram`

### RunPod: DRAM cache lost between requests

1. **Idle timeout too short:** Default is 5s — cache lost when container shuts down
2. **Fix:** Set Active Workers ≥ 1, or increase Idle Timeout to 60-300s
3. **Check `REFRESH_WORKER`:** Must be `false` (default) — `true` restarts container every job
4. **FlashBoot:** Doesn't preserve RAM state — cache repopulates on revival

### "DRAM HIT" never appears (always loading from disk)

1. **Check `use_dram` toggle** — must be `True` on the loader
2. **Check cache key match** — the loader and offload node must produce the same
   key. Print statements show the key: `dram_id: unet:filename:dtype`
3. **Check Memory Cleanup** — if it runs between loads, it clears the cache
4. **Check `IS_CHANGED`** — if it returns a static value when `use_dram=True`,
   ComfyUI's execution cache may skip re-executing the loader

### Disk activity during offload

1. **Check swappiness:** `cat /proc/sys/vm/swappiness` — should be ≤10
2. **Check swap usage:** `free -g` — if swap > 0, models may be swapped out
3. **Check RAM availability:** models need to fit in physical RAM
4. **First-run disk I/O is normal** — initial load always reads from disk

### VRAM not freed after offload

1. **Check memory_stats output** — does it show the model as offloaded?
2. **Check VRAM mode:** startup should show `VRAM mode: NORMAL_VRAM`
3. **Try adding `--reserve-vram 0.5`** to prevent CUDA overhead from filling gap

### Offload node has no effect

1. **Offload nodes use `IS_CHANGED = time.time()`** — they always re-execute
2. **Check the model input is connected** — model/clip is required
3. **Check console for `[DRAM Cache]` messages**

### Pinned model still gets evicted

1. **Check console for `[VRAM PIN]` messages** — should see "pinned to GPU" and "Eviction protection patch installed"
2. **If no `[VRAM PIN]` messages:** `keep_on_vram` is not reaching the pin code. Check the toggle value.
3. **If patch not installed:** The monkey-patch failed. Check for errors at startup.
4. **WeakSet cleanup:** If the model was garbage collected (node re-executed with different inputs), the pin auto-removed. This is expected.

### Too many models pinned / VRAM full

1. **Check memory_stats** — shows pinned count at the bottom
2. **Symptoms:** New models load in lowvram mode (very slow inference), high VRAM usage
3. **Fix:** Use Memory Cleanup with `unpin_all_vram=True`, or reduce number of `keep_on_vram=True` loaders
4. **Better fix:** Enable `auto_free_vram=True` on loaders — automatically cleans up when VRAM is low
5. **Rule of thumb:** Only pin models that fit alongside your main model in VRAM

### Auto-free triggers every run (thrashing)

1. **Threshold too high:** If `min_free_vram_gb` is higher than your GPU's total VRAM minus model size, auto-free triggers every time
2. **Fix:** Lower the threshold. For RTX 3090 (24GB) loading a 10GB model, set `min_free_vram_gb=12`
3. **For DRAM:** Same logic — if threshold > available RAM after model, it clears every time
4. **Signs of thrashing:** `[AUTO-FREE]` messages on every run, slow performance

### RunPod: Orphaned models from previous workflow

1. **Problem:** Different workflow arrives, old models still pinned/cached, VRAM/RAM full
2. **Fix:** Enable `auto_free_vram=True` and `auto_free_dram=True` on all loaders in your RunPod workflows
3. **How it works:** Before loading a new model from disk, checks memory. If tight, clears all old pins and DRAM entries.
4. **One-time cost:** First load after cleanup reads from disk (~8s). After that, DRAM cache repopulates.

### CLIP offload fails / no `.patcher` attribute

This would happen if ComfyUI changes how CLIP objects are structured.
Check: `grep -n "self.patcher" comfy/sd.py`

---

## File Map

```
nodes/
├── inputs/
│   └── archai3d_triggered_loaders.py    # Load Diffusion Model, Load CLIP, Load Dual CLIP
│                                         # DRAM cache check + cache_to_local_ssd + IS_CHANGED
│
├── utils/
│   ├── dram_cache.py                     # Core DRAM cache + VRAM pin + auto-free system
│   │                                     # DRAM: store(), get(), remove(), clear(), list_cached()
│   │                                     # VRAM Pin: pin_model(), unpin_model(), unpin_all(), is_pinned()
│   │                                     # Auto-free: ensure_free_memory()
│   │                                     # Stats: get_memory_stats(), _check_vram_state()
│   │
│   ├── archai3d_offload_model.py         # Offload Model to DRAM node
│   ├── archai3d_offload_clip.py          # Offload CLIP to DRAM node
│   ├── archai3d_memory_cleanup.py        # Memory Cleanup node
│   └── local_model_cache.py             # RunPod SSD cache (independent from DRAM)
│
├── __init__.py                           # Node registration (search for "Memory Management")
│
├── workflows/
│   └── DRAM_qwen_image_edit_2511.json   # Reference workflow using all DRAM nodes
│
└── start_comfyui_dram.sh                 # Start script (in ComfyUI root, not here)
    # --normalvram --cache-classic --use-sage-attention --reserve-vram 0.5
```

### Dependencies

- **Python standard library only** — no extra pip packages needed for DRAM cache
- **psutil** — used for RAM stats in `get_memory_stats()` (already in ComfyUI deps)
- **ComfyUI internals:** `comfy.model_management`, `comfy.sd`, `folder_paths`

---

## Quick Reference: Cache Key Patterns

| Loader | Key Pattern | Example |
|--------|-------------|---------|
| Load Diffusion Model | `unet:{filename}:{dtype}` | `unet:flux-dev-fp8.safetensors:fp8_e4m3fn` |
| Load CLIP | `clip:{filename}:{type}` | `clip:qwen_2.5_vl_7b.safetensors:qwen_image` |
| Load Dual CLIP | `dualclip:{file1}:{file2}:{type}` | `dualclip:clip-l.safetensors:t5xxl.safetensors:flux` |
| Offload Model (fallback) | `model:{classname}:{id}` | `model:Flux:140234567890` |
| Offload CLIP (fallback) | `clip:{classname}:{id}` | `clip:QwenVLModel:140234567891` |

The offload nodes prefer reading `._dram_cache_key` from the model (set by the
loader). The fallback key (using `id()`) only activates if the model wasn't loaded
by our triggered loaders. Fallback keys are NOT stable across runs.
