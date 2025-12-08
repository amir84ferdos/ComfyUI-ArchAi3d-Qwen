# ArchAi3D Smart Tile Prompter Turbo
#
# Z-Image-Turbo optimized prompting for tiled upscaling
# Generates dense paragraph prompts following 3-step structure:
#   1. Core (Subject & Action) - exact facts
#   2. Visual Expansion (Style, Lighting, Atmosphere) - professional aesthetics
#   3. Text Transcription - verbatim text in quotes
#
# Author: Amir Ferdos (ArchAi3d)
# Version: 1.0.3 - Added sharpness/quality cues to prevent blurry outputs
# License: Dual License (Free for personal use, Commercial license required for business use)

import base64
import hashlib
import io
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

# ============================================================================
# CONSTANTS
# ============================================================================

# Model configurations (same as QwenVL GGUF)
MODEL_CONFIGS = {
    "2B (Fast)": {"port": 8032, "name": "Qwen3-VL-2B"},
    "4B (Balanced)": {"port": 8033, "name": "Qwen3-VL-4B"},
    "8B (Best)": {"port": 8034, "name": "Qwen3-VL-8B"},
}
MODEL_SIZES = list(MODEL_CONFIGS.keys())

# Quality presets (simplified from QwenVL GGUF)
QUALITY_PRESETS = {
    "Ultra Focused": {
        "max_tokens": 1024,
        "temperature": 0.37,
        "top_p": 0.78,
        "top_k": 44,
        "min_p": 0.073,
        "repeat_penalty": 1.14,
    },
    "Balanced": {
        "max_tokens": 768,
        "temperature": 0.5,
        "top_p": 0.85,
        "top_k": 60,
        "min_p": 0.05,
        "repeat_penalty": 1.1,
    },
    "Creative": {
        "max_tokens": 1024,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 80,
        "min_p": 0.02,
        "repeat_penalty": 1.05,
    },
}
QUALITY_PRESET_NAMES = list(QUALITY_PRESETS.keys())

# Consistency levels
CONSISTENCY_LEVELS = ["Strict", "Balanced", "Creative"]

CONSISTENCY_PROMPTS = {
    "Strict": "Use ONLY the EXACT materials and colors from the style guide. Do NOT introduce ANY new terms or variations.",
    "Balanced": "Use the materials and colors from the style guide. Minor variations acceptable if they maintain consistency.",
    "Creative": "Use the style guide as reference. You may describe additional details while staying consistent with the overall style.",
}

# ============================================================================
# Z-IMAGE-TURBO OPTIMIZED PROMPTS
# ============================================================================

# Style guide extraction prompt - optimized for 3-step structure extraction
STYLE_GUIDE_PROMPT = """Analyze this image and extract a comprehensive STYLE GUIDE for consistent tile descriptions.

SUBJECTS & OBJECTS:
- List every visible subject, person, furniture, object with specific names
- Note positions and spatial relationships between elements
- Describe states and actions (person sitting, vase containing flowers)

MATERIALS & TEXTURES:
- Exact materials for each element ("walnut wood with visible grain", "brushed stainless steel", "white marble with grey veins")
- Surface qualities: smooth, rough, woven, polished, matte, glossy
- Fabric details: weave type, thread visibility, draping quality

LIGHTING ANALYSIS:
- Light source type: natural (window), artificial (lamp, overhead)
- Light direction: front, side, back, rim, overhead
- Light quality: soft/diffused vs hard/direct
- Color temperature: warm (golden), cool (blue), neutral
- Shadow characteristics: soft gradients, hard edges, absent

COLOR PALETTE:
- Exact colors for major elements ("navy blue velvet", "cream linen", "aged brass")
- Accent colors and their locations
- Overall color mood (warm neutrals, cool blues, earth tones)

ATMOSPHERE & STYLE:
- Overall aesthetic (modern minimalist, mid-century, industrial, baroque)
- Mood and feel (cozy, sterile, luxurious, rustic)
- Camera perspective if discernible (eye-level, low angle, overhead)

TEXT (if any):
- Transcribe any visible text exactly as it appears
- Note text location and style (font type, color, signage type)

Be thorough - this guide ensures all tile descriptions maintain consistency."""

# Tile prompt template - requests dense 3-step paragraph format
TILE_PROMPT_TEMPLATE = """STYLE GUIDE FOR CONSISTENCY:
{style_guide}

---

TILE POSITION: {position} (row {row}, column {col})
SPATIAL ZONE: {spatial_zone}
CONSISTENCY RULE: {consistency_rule}

CRITICAL - DESCRIBE ONLY WHAT YOU SEE:
- This tile shows the {spatial_zone} of the scene
- Do NOT place floor-level objects (tables, furniture bases, rugs) in top/ceiling tiles
- Do NOT place ceiling elements (lights, beams, ceiling fixtures) in bottom/floor tiles
- If the tile shows mostly ceiling, describe ONLY ceiling. If floor, describe ONLY floor.
- Use the style guide for materials and colors, but ONLY for objects ACTUALLY VISIBLE in this tile

Write a DENSE PARAGRAPH description of this tile region following this 3-step structure:

1. CORE (Subject & Action) - ONLY what is visible in THIS tile:
   - What subjects/objects are ACTUALLY visible (even partially cut off)
   - Their positions and relationships within THIS tile only
   - Do NOT invent objects from the style guide that aren't in THIS tile

2. VISUAL EXPANSION (Style, Lighting, Atmosphere) - Add professional details:
   - Lighting on THIS specific area (source direction, quality, shadows)
   - Textures and materials you can see (fabric weave, wood grain, surface sheen)
   - Colors matching the style guide for visible elements only
   - Depth and composition in this tile

3. TEXT (if any) - End with any visible text in "double quotes"

SHARPNESS REQUIREMENT - CRITICAL FOR IMAGE QUALITY:
- START the paragraph with "Sharp focus on" or "Crisp detail of" or "Highly detailed"
- Include at least 2-3 sharpness descriptors: "sharp edges", "crisp definition", "fine detail", "clearly defined", "precise texture", "high clarity", "distinct edges"
- Describe textures with precision: "visible grain", "individual threads", "distinct separation"
- AVOID soft/blurry language: do NOT use "softly", "gently blurred", "subtly fading", "soft gradients"

CRITICAL RULES:
- Write as a FLOWING PARAGRAPH, not bullet points or categories
- Keep it to 100-150 words (CLIP has token limits)
- Do NOT include notes, disclaimers, meta-commentary, or explanations
- Do NOT write "No text is visible" - just end the paragraph if no text exists
- If text IS visible, end with it in quotes (e.g., The sign reads "OPEN")
- Output ONLY the description paragraph, nothing else

Example format (with sharpness cues):
"Sharp focus on a leather armchair in the left foreground, its worn brown surface showing crisp creasing detail at the armrests with clearly defined stitching. Precise window light from the upper right creates distinct tonal gradients across the seat cushion, with hard-edged shadows pooling beneath the curved wooden legs. The floor beneath reveals highly detailed oak planks with sharp grain patterns running diagonally, each plank edge crisply defined..."
"""

# Result cache
TILE_PROMPTER_TURBO_CACHE = {}

# Script path for auto-starting server
SCRIPT_PATH = Path(__file__).parent.parent.parent / "start_qwenvl_server.sh"


# ============================================================================
# SERVER MANAGEMENT
# ============================================================================

def check_server_ready(port):
    """Check if server is running and responding."""
    try:
        resp = requests.get(f"http://localhost:{port}/health", timeout=2)
        return resp.status_code == 200
    except:
        return False


def start_llama_server(model_size, port, gpu_layers=99, context_size=8192):
    """Start llama-server for the specified model size."""
    model_short = model_size.split()[0]  # "4B" from "4B (Balanced)"

    if not SCRIPT_PATH.exists():
        print(f"[Smart Tile Prompter Turbo] Warning: Server script not found at {SCRIPT_PATH}")
        return None

    env = os.environ.copy()
    env["CTX"] = str(context_size)
    env["GPU_LAYERS"] = str(gpu_layers)

    process = subprocess.Popen(
        [str(SCRIPT_PATH), model_short],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        env=env
    )
    print(f"[Smart Tile Prompter Turbo] Started server for {model_short} on port {port}")
    print(f"  GPU Layers: {gpu_layers}, Context: {context_size}")
    return process


def ensure_server_running(model_size, auto_start=True, gpu_layers=99, context_size=8192):
    """Ensure the QwenVL server is running. Returns (success, error_message)."""
    config = MODEL_CONFIGS.get(model_size, MODEL_CONFIGS["4B (Balanced)"])
    port = config['port']
    model_name = config['name']
    model_short = model_size.split()[0]

    if check_server_ready(port):
        return True, None

    if not auto_start:
        error_msg = (
            f"QwenVL server not running!\n\n"
            f"Use '🎛️ QwenVL Server Control' node to start the server:\n"
            f"  - Action: Start Server\n"
            f"  - Model: {model_size}\n\n"
            f"Or run manually:\n"
            f"  ./start_qwenvl_server.sh {model_short}"
        )
        return False, error_msg

    # Auto-start the server
    print(f"[Smart Tile Prompter Turbo] Server not running, starting automatically...")
    start_llama_server(model_size, port, gpu_layers, context_size)

    # Wait for server to be ready (up to 60 seconds)
    for i in range(60):
        time.sleep(1)
        if check_server_ready(port):
            # Test with a simple request to verify model is loaded
            try:
                resp = requests.post(
                    f"http://localhost:{port}/v1/chat/completions",
                    json={"model": "test", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
                    timeout=5
                )
                if resp.status_code != 503:
                    print(f"[Smart Tile Prompter Turbo] Server ready after {i+1} seconds")
                    return True, None
            except requests.exceptions.Timeout:
                # Timeout on request = model is processing = ready
                print(f"[Smart Tile Prompter Turbo] Server ready after {i+1} seconds")
                return True, None
            except:
                pass

        if i % 10 == 0 and i > 0:
            print(f"[Smart Tile Prompter Turbo] Loading model... ({i}s)")

    return False, f"Server failed to start after 60s. Try: ./start_qwenvl_server.sh {model_short}"


def stop_server(model_size):
    """Stop the QwenVL server for the specified model size."""
    config = MODEL_CONFIGS.get(model_size, MODEL_CONFIGS["4B (Balanced)"])
    port = config['port']
    model_short = model_size.split()[0]

    if not check_server_ready(port):
        print(f"[Smart Tile Prompter Turbo] Server for {model_short} is not running")
        return False

    try:
        # Find and kill the process using the port
        # Use ss to find the PID listening on the port
        result = subprocess.run(
            ['ss', '-tlnp', f'sport = :{port}'],
            capture_output=True, text=True
        )

        # Parse the output to find PID
        # Format: LISTEN 0 128 *:8033 *:* users:(("llama-server",pid=12345,fd=3))
        output = result.stdout
        if f':{port}' in output:
            # Extract PID from output
            pid_match = re.search(r'pid=(\d+)', output)
            if pid_match:
                pid = int(pid_match.group(1))
                os.kill(pid, signal.SIGTERM)
                print(f"[Smart Tile Prompter Turbo] Stopped server for {model_short} (PID {pid})")

                # Wait for server to stop
                for _ in range(10):
                    time.sleep(0.5)
                    if not check_server_ready(port):
                        print(f"[Smart Tile Prompter Turbo] Server stopped successfully")
                        return True

        # Fallback: try pkill
        subprocess.run(['pkill', '-f', f'llama-server.*{port}'], capture_output=True)
        print(f"[Smart Tile Prompter Turbo] Sent stop signal to server on port {port}")
        return True

    except Exception as e:
        print(f"[Smart Tile Prompter Turbo] Error stopping server: {e}")
        return False


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def image_to_base64(image_tensor, index=0, max_size=1024):
    """Convert ComfyUI image tensor to base64 string with resizing."""
    idx = min(index, image_tensor.shape[0] - 1)
    img_np = (image_tensor[idx].cpu().numpy() * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)

    # Resize if needed
    w, h = pil_img.size
    if w > max_size or h > max_size:
        if w > h:
            new_w = max_size
            new_h = int(h * max_size / w)
        else:
            new_h = max_size
            new_w = int(w * max_size / h)
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def call_qwenvl_api(image_b64, prompt, model_size, quality_preset, seed=1):
    """Call QwenVL GGUF API and return response text."""
    config = MODEL_CONFIGS.get(model_size, MODEL_CONFIGS["4B (Balanced)"])
    port = config['port']
    server_url = f"http://localhost:{port}"

    preset = QUALITY_PRESETS.get(quality_preset, QUALITY_PRESETS["Ultra Focused"])

    # Build content
    content = []
    if image_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_b64}"}
        })
    content.append({"type": "text", "text": prompt})

    payload = {
        "model": "qwen3-vl",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": preset["max_tokens"],
        "temperature": preset["temperature"],
        "top_p": preset["top_p"],
        "top_k": preset["top_k"] if preset["top_k"] > 0 else -1,
        "min_p": preset["min_p"],
        "repeat_penalty": preset["repeat_penalty"],
        "seed": seed,
        "stream": False
    }

    try:
        response = requests.post(
            f"{server_url}/v1/chat/completions",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.ConnectionError:
        return f"ERROR: Cannot connect to QwenVL server on port {port}. Start with: ./start_qwenvl_server.sh"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {str(e)}"


def get_position_name(x, y, tiles_x, tiles_y):
    """Convert tile coordinates to human-readable position."""
    # Vertical position
    if tiles_y == 1:
        v_pos = ""
    elif y == 0:
        v_pos = "top"
    elif y == tiles_y - 1:
        v_pos = "bottom"
    else:
        v_pos = "middle"

    # Horizontal position
    if tiles_x == 1:
        h_pos = ""
    elif x == 0:
        h_pos = "left"
    elif x == tiles_x - 1:
        h_pos = "right"
    else:
        h_pos = "center"

    # Combine
    if v_pos and h_pos:
        return f"{v_pos}-{h_pos}"
    elif v_pos:
        return v_pos
    elif h_pos:
        return h_pos
    else:
        return "center"


def get_spatial_zone(row, col, tiles_y, tiles_x):
    """
    Determine what spatial zone this tile represents.

    This helps the LLM understand what SHOULD be visible in this tile
    based on its position in the grid, preventing hallucination of
    floor-level objects in ceiling tiles and vice versa.

    Args:
        row: Row index (0 = top)
        col: Column index (0 = left)
        tiles_y: Total rows
        tiles_x: Total columns

    Returns:
        Human-readable spatial zone description
    """
    # Vertical zones based on row position
    if tiles_y >= 4:
        # 4+ rows: top, upper-middle, lower-middle, bottom
        if row == 0:
            v_zone = "ceiling/uppermost area (lights, ceiling fixtures, top of walls)"
        elif row == tiles_y - 1:
            v_zone = "floor/ground level (floor, rug, furniture bases, lowest objects)"
        elif row < tiles_y // 2:
            v_zone = "upper area (upper walls, windows, hanging items, tall furniture tops)"
        else:
            v_zone = "lower-middle area (furniture, seated figures, tabletops, mid-height objects)"
    elif tiles_y == 3:
        # 3 rows: top, middle, bottom
        if row == 0:
            v_zone = "ceiling/upper wall area (ceiling, lights, top of tall furniture)"
        elif row == 1:
            v_zone = "mid-height area (furniture, windows, people, countertops)"
        else:
            v_zone = "floor/lower area (floor, rugs, furniture bases, low objects)"
    elif tiles_y == 2:
        # 2 rows: top half, bottom half
        if row == 0:
            v_zone = "upper half (ceiling, upper walls, hanging lights, tops of furniture)"
        else:
            v_zone = "lower half (floor, lower furniture, bases of objects)"
    else:
        # 1 row: full height
        v_zone = "full height"

    return v_zone


def crop_tile(image_tensor, tile_x, tile_y, tiles_x, tiles_y, overlap=0):
    """Extract a tile region from the image tensor.

    Args:
        image_tensor: Shape [B, H, W, C]
        tile_x, tile_y: Tile coordinates (0-indexed)
        tiles_x, tiles_y: Total number of tiles
        overlap: Overlap pixels to include from adjacent tiles

    Returns:
        Cropped image tensor [1, H', W', C]
    """
    _, height, width, channels = image_tensor.shape

    # Calculate base tile size
    tile_w = width // tiles_x
    tile_h = height // tiles_y

    # Calculate tile boundaries
    x1 = tile_x * tile_w
    y1 = tile_y * tile_h
    x2 = x1 + tile_w
    y2 = y1 + tile_h

    # Handle last tile (may be larger due to rounding)
    if tile_x == tiles_x - 1:
        x2 = width
    if tile_y == tiles_y - 1:
        y2 = height

    # Add overlap (clamped to image bounds)
    x1 = max(0, x1 - overlap)
    y1 = max(0, y1 - overlap)
    x2 = min(width, x2 + overlap)
    y2 = min(height, y2 + overlap)

    # Crop
    cropped = image_tensor[:1, y1:y2, x1:x2, :]
    return cropped


def compute_cache_key(image_tensor, tiles_x, tiles_y, model_size, quality_preset, consistency_level, user_context, seed):
    """Create unique hash for caching."""
    hasher = hashlib.md5()
    hasher.update(image_tensor.cpu().numpy().tobytes())
    hasher.update(f"{tiles_x}x{tiles_y}".encode())
    hasher.update(model_size.encode())
    hasher.update(quality_preset.encode())
    hasher.update(consistency_level.encode())
    hasher.update(user_context.encode())
    hasher.update(str(seed).encode())
    hasher.update(b"turbo")  # Distinguish from regular prompter cache
    return hasher.hexdigest()


# ============================================================================
# NODE CLASS
# ============================================================================

class ArchAi3D_Smart_Tile_Prompter_Turbo:
    """
    Smart Tile Prompter Turbo - Z-Image-Turbo optimized prompts.

    Generates dense paragraph prompts following Z-Image-Turbo's 3-step structure:
    1. Core (Subject & Action) - exact facts about subjects and objects
    2. Visual Expansion (Style, Lighting, Atmosphere) - professional aesthetics
    3. Text Transcription - verbatim text in "double quotes"

    Unlike the standard Smart Tile Prompter which generates categorized lists,
    this version produces flowing paragraphs optimized for Z-Image-Turbo's
    training format.
    """

    CATEGORY = "ArchAi3d/Utils"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "INT", "INT", "STRING", "SMART_TILE_BUNDLE")
    RETURN_NAMES = ("global_context", "tile_prompts", "tile_prompts_json",
                    "tile_prompts_labels", "tiles_x", "tiles_y", "debug_info", "bundle")
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Full image to analyze"
                }),
                "tiles_x": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 8,
                    "tooltip": "Number of horizontal tiles"
                }),
                "tiles_y": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 8,
                    "tooltip": "Number of vertical tiles"
                }),
                "model_size": (MODEL_SIZES, {
                    "default": "4B (Balanced)",
                    "tooltip": "QwenVL model size"
                }),
                "quality_preset": (QUALITY_PRESET_NAMES, {
                    "default": "Ultra Focused",
                    "tooltip": "Quality/creativity preset"
                }),
                "consistency_level": (CONSISTENCY_LEVELS, {
                    "default": "Balanced",
                    "tooltip": "How strictly to enforce style guide"
                }),
            },
            "optional": {
                "bundle": ("SMART_TILE_BUNDLE", {
                    "tooltip": "Connect bundle from Smart Tile Calculator (auto-fills tiles_x, tiles_y, image)"
                }),
                "user_context": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional: Your description to include in style guide"
                }),
                "tile_overlap": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 256,
                    "step": 8,
                    "tooltip": "Overlap pixels when cropping tiles"
                }),
                "use_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Cache results for faster re-runs"
                }),
                "auto_start_server": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically start server if not running"
                }),
                "gpu_layers": ("INT", {
                    "default": 99,
                    "min": 0,
                    "max": 99,
                    "tooltip": "GPU layers for server (99=all on GPU)"
                }),
                "context_size": ("INT", {
                    "default": 8192,
                    "min": 1024,
                    "max": 32768,
                    "step": 1024,
                    "tooltip": "Context window size in tokens"
                }),
                "stop_server_after": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Stop the QwenVL server after generating prompts (frees VRAM for diffusion)"
                }),
                "seed": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed for reproducible prompt generation"
                }),
            }
        }

    def generate(self, image, tiles_x, tiles_y, model_size, quality_preset,
                 consistency_level, bundle=None, user_context="", tile_overlap=0, use_cache=True,
                 auto_start_server=True, gpu_layers=99, context_size=8192,
                 stop_server_after=False, seed=1):
        """Generate Z-Image-Turbo optimized prompts for all tiles."""

        # If bundle provided, extract values (overrides individual inputs)
        if bundle is not None:
            image = bundle.get("scaled_image", image)
            tiles_x = bundle.get("tiles_x", tiles_x)
            tiles_y = bundle.get("tiles_y", tiles_y)
            print(f"[Smart Tile Prompter Turbo v1.0.3] Using bundle: {tiles_x}x{tiles_y} tiles")

        start_time = time.time()
        total_tiles = tiles_x * tiles_y

        print(f"\n[Smart Tile Prompter Turbo] Starting {tiles_x}x{tiles_y} = {total_tiles} tiles")
        print(f"  Model: {model_size}, Preset: {quality_preset}, Consistency: {consistency_level}")
        print(f"  Seed: {seed}")
        print(f"  Mode: Z-Image-Turbo optimized (3-step dense paragraphs)")

        # ===== CHECK SERVER =====
        server_ok, error_msg = ensure_server_running(
            model_size, auto_start_server, gpu_layers, context_size
        )
        if not server_ok:
            return (error_msg, error_msg, "{}", "", tiles_x, tiles_y, error_msg)

        # Check cache
        cache_key = None
        if use_cache:
            cache_key = compute_cache_key(
                image, tiles_x, tiles_y, model_size, quality_preset,
                consistency_level, user_context, seed
            )
            if cache_key in TILE_PROMPTER_TURBO_CACHE:
                print("[Smart Tile Prompter Turbo] Cache hit!")
                cached = TILE_PROMPTER_TURBO_CACHE[cache_key]
                return cached

        # ===== PHASE 1: GLOBAL CONTEXT EXTRACTION =====
        print("[Smart Tile Prompter Turbo] Phase 1: Extracting global context (Z-Image-Turbo format)...")

        full_image_b64 = image_to_base64(image, max_size=1536)
        style_guide = call_qwenvl_api(
            full_image_b64,
            STYLE_GUIDE_PROMPT,
            model_size,
            quality_preset,
            seed=seed
        )

        # Add user context if provided
        if user_context and user_context.strip():
            style_guide = f"USER NOTES: {user_context.strip()}\n\n{style_guide}"

        print(f"[Smart Tile Prompter Turbo] Style guide extracted ({len(style_guide)} chars)")

        # ===== PHASE 2: PER-TILE PROMPT GENERATION =====
        print(f"[Smart Tile Prompter Turbo] Phase 2: Generating {total_tiles} tile prompts (dense paragraphs)...")

        tile_prompts_list = []
        consistency_rule = CONSISTENCY_PROMPTS[consistency_level]

        for y in range(tiles_y):
            for x in range(tiles_x):
                tile_num = y * tiles_x + x + 1
                position = get_position_name(x, y, tiles_x, tiles_y)

                print(f"  Tile {tile_num}/{total_tiles}: {position}...")

                # Crop tile
                tile_image = crop_tile(image, x, y, tiles_x, tiles_y, tile_overlap)
                tile_b64 = image_to_base64(tile_image, max_size=1024)

                # Build tile prompt
                spatial_zone = get_spatial_zone(y, x, tiles_y, tiles_x)
                tile_prompt = TILE_PROMPT_TEMPLATE.format(
                    style_guide=style_guide,
                    position=position,
                    row=y + 1,
                    col=x + 1,
                    spatial_zone=spatial_zone,
                    consistency_rule=consistency_rule
                )

                # Call API (use seed + tile_num for reproducible but unique per-tile results)
                tile_description = call_qwenvl_api(
                    tile_b64,
                    tile_prompt,
                    model_size,
                    quality_preset,
                    seed=seed + tile_num
                )

                tile_prompts_list.append({
                    "tile": tile_num,
                    "position": position,
                    "x": x,
                    "y": y,
                    "prompt": tile_description
                })

        # ===== FORMAT OUTPUTS =====

        # Combined string format (human-readable)
        combined_lines = []
        for tp in tile_prompts_list:
            combined_lines.append(f"TILE {tp['tile']} ({tp['position']}):\n{tp['prompt']}")
        tile_prompts_combined = "\n\n---\n\n".join(combined_lines)

        # JSON format (for programmatic use)
        tile_prompts_json = json.dumps(tile_prompts_list, indent=2)

        # Comma-separated labels format (for SEGSLabelAssign)
        # Each prompt is one label, comma-separated
        tile_prompts_labels = ",".join([tp['prompt'].replace(",", ";") for tp in tile_prompts_list])

        # Debug info
        elapsed = time.time() - start_time
        debug_lines = [
            "=" * 50,
            "Smart Tile Prompter Turbo v1.0.3 (Z-Image-Turbo)",
            "=" * 50,
            f"Tiles: {tiles_x}x{tiles_y} = {total_tiles} total",
            f"Model: {model_size}",
            f"Preset: {quality_preset}",
            f"Consistency: {consistency_level}",
            f"Overlap: {tile_overlap}px",
            f"Seed: {seed}",
            "",
            "Prompt Format: Dense 3-step paragraphs",
            "  1. Core (Subject & Action)",
            "  2. Visual Expansion (Style, Lighting, Atmosphere)",
            "  3. Text Transcription (verbatim in quotes)",
            "",
            f"API Calls: {total_tiles + 1} (1 global + {total_tiles} tiles)",
            f"Total Time: {elapsed:.1f}s",
            f"Time per tile: {elapsed / (total_tiles + 1):.1f}s",
            "",
            "SEGS Integration:",
            f"  tile_prompts_labels: {len(tile_prompts_labels)} chars (for SEGSLabelAssign)",
            "",
            "Style Guide Preview:",
            style_guide[:300] + "..." if len(style_guide) > 300 else style_guide,
            "=" * 50,
        ]
        debug_info = "\n".join(debug_lines)

        print(f"[Smart Tile Prompter Turbo] Done in {elapsed:.1f}s")

        # Create output bundle (pass through input bundle data + add prompts)
        output_bundle = {}
        if bundle is not None:
            output_bundle.update(bundle)  # Copy all data from input bundle
        # Add/update prompt data
        output_bundle["tile_prompts_labels"] = tile_prompts_labels
        output_bundle["tile_prompts_list"] = tile_prompts_list  # Full list with positions
        output_bundle["style_guide"] = style_guide
        output_bundle["tiles_x"] = tiles_x
        output_bundle["tiles_y"] = tiles_y
        output_bundle["total_tiles"] = total_tiles

        # Cache result
        result = (style_guide, tile_prompts_combined, tile_prompts_json,
                  tile_prompts_labels, tiles_x, tiles_y, debug_info, output_bundle)
        if use_cache and cache_key:
            TILE_PROMPTER_TURBO_CACHE[cache_key] = result
            print(f"[Smart Tile Prompter Turbo] Result cached ({len(TILE_PROMPTER_TURBO_CACHE)} entries)")

        # Stop server if requested (frees VRAM for diffusion model)
        if stop_server_after:
            print(f"[Smart Tile Prompter Turbo] Stopping server to free VRAM...")
            stop_server(model_size)

        return result


# ============================================================================
# NODE REGISTRATION
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "ArchAi3D_Smart_Tile_Prompter_Turbo": ArchAi3D_Smart_Tile_Prompter_Turbo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_Smart_Tile_Prompter_Turbo": "🚀 Smart Tile Prompter Turbo",
}
