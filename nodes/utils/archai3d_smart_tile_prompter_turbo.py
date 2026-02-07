# ArchAi3D Smart Tile Prompter Turbo
#
# Z-Image-Turbo optimized prompting for tiled upscaling
# v3.0: Z-Image Turbo Texture-Focused Prompting
#   - Global style extraction (environment + lighting + color palette)
#   - Local texture-only descriptions (NO object inference)
#   - Anti-hallucination: describes surfaces only, not partially visible objects
#   - Hierarchical assembly: [Local Texture] + [Global Style]
#
# Author: Amir Ferdos (ArchAi3d)
# Version: 3.10.0 - Use bash to run script (no execute permission needed)
#                  Magazine-quality descriptions by default (Arch Digest, Dwell, Dezeen)
#                  Professional terminology for all materials and finishes
#                  v3.5.0: VLLM Guidance Instructions
#                  v3.4.0: Spatial Context for High Denoise (camera angle, depth layer)
#                  v3.3.0: Extended Detail + Use Cases (30% longer prompts, 5 photography modes)
#                  v3.2.0: Foreground First (objects > background, prevents erasing furniture)
#                  v3.1.0: Foreground Specificity + Background Texture
#                  v3.0.0: Z-Image Turbo optimized (texture focus, no hallucination)
#                  v2.1.0: Enhanced with materials + neighbor awareness
#                  v2.0.0: Scene-grounded prompting (short scene + tile details)
#                  v1.0.7: Added clear_cache option + better cache diagnostics
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
from collections import OrderedDict
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
# V3.0 Z-IMAGE TURBO OPTIMIZED PROMPTS
# ============================================================================
# Based on Z-Image Turbo prompting guide:
# - No negative prompts (all exclusions via positive description)
# - Two-pass system: Global style + Local texture
# - Texture-focused descriptions to prevent hallucination
# - Hierarchical structure: Subject + Environment + Style + Composition

# PASS 1: Global Style Extraction (run once on full image)
# Extracts: Environment context + Visual style + Color palette + Lighting
GLOBAL_STYLE_PROMPT = """Analyze this image for GLOBAL STYLE attributes only.

Extract:
1. ENVIRONMENT: Interior style (modern, Scandinavian, industrial, etc.)
2. LIGHTING: Quality and direction (soft diffused daylight, warm spotlights, etc.)
3. COLOR PALETTE: Dominant tones (muted beige, cool grays, warm earth tones)
4. PHOTOGRAPHY STYLE: Camera quality descriptors

Output format (single line, comma-separated):
"[style] interior, [lighting description], [color palette], architectural photography, photorealistic 8k"

Examples:
- "Modern Japandi interior, soft natural morning light, beige and oak color palette, architectural photography, photorealistic 8k"
- "Industrial loft space, dramatic side lighting with shadows, raw concrete and steel tones, architectural photography, photorealistic 8k"
- "Scandinavian minimalist interior, soft diffused daylight, muted white and wood tones, architectural photography, photorealistic 8k"

Output ONLY the style string, nothing else."""

# PASS 2A: Materials Grid Extraction (run once on full image)
# Identifies surface materials for consistency across tiles
MATERIALS_GRID_PROMPT = """Analyze this image divided into a {tiles_x}x{tiles_y} grid.

EXTRACT SURFACE MATERIALS (be very specific about textures):
- Floor: material + texture + finish (e.g., "light oak herringbone wood, visible grain, matte finish")
- Walls: material + texture (e.g., "white plaster, smooth finish" or "exposed red brick, rough texture")
- Ceiling: material + features (e.g., "white painted drywall, recessed spotlights")

For each tile, describe ONLY the visible SURFACES (not objects):

Grid positions (row,col starting from 0):
{grid_positions}

Output format (JSON only):
{{
  "materials": {{
    "floor": "light oak herringbone wood, visible grain, matte finish",
    "walls": "white smooth plaster with exposed brick accent",
    "ceiling": "white painted, recessed LED spotlights"
  }},
  "tiles": {{
    "0,0": "white ceiling surface, spotlight fixtures",
    "0,1": "ceiling-wall corner, white plaster meets brick",
    "1,0": "oak wood floor, furniture leg shadows",
    "1,1": "wood floor, fabric texture edge"
  }}
}}

IMPORTANT: Describe SURFACES only, not objects. Say "cylindrical black metal leg" not "chair leg".
Output ONLY valid JSON."""

# ============================================================================
# USE CASE SPECIFIC PROMPTS (v3.6 Magazine Quality)
# ============================================================================
USE_CASE_PROMPTS = {
    "Interior Design": {
        "role": "award-winning Interior Design Photographer shooting for Architectural Digest magazine",
        "objects": "Designer furniture (specify designer/style: Eames lounge, Barcelona chair, Noguchi table, B&B Italia sectional), Luxury lighting fixtures (Flos, Artemide, Louis Poulsen, custom artisan pendants), Curated decor (sculptural vases, artisan ceramics, gallery-quality artwork, designer objects)",
        "surfaces": "Premium architectural finishes (hand-troweled Venetian plaster, honed Carrara marble, wide-plank European white oak, polished concrete, Calacatta quartzite)",
        "style_suffix": "award-winning architectural interior photography, Architectural Digest quality, photorealistic 8k",
    },
    "Exterior Architecture": {
        "role": "award-winning Architectural Photographer shooting for Dwell or Dezeen magazine",
        "objects": "Architectural elements (cantilevered overhangs, floor-to-ceiling glazing, brise-soleil, steel moment frames), Professional landscaping (specimen trees with Latin names, ornamental grasses, drought-tolerant native plantings, professionally maintained lawn), Hardscape (permeable pavers, architectural concrete, natural stone coping, steel edging)",
        "surfaces": "Exterior architectural finishes (smooth stucco render, corrugated Corten steel, thermally-modified timber cladding, zinc panels), Ground treatments (decomposed granite pathways, Belgian block driveways, manicured Kentucky bluegrass, pea gravel with steel borders)",
        "style_suffix": "award-winning architectural exterior photography, Dwell magazine quality, photorealistic 8k",
    },
    "Product Photography": {
        "role": "award-winning Commercial Photographer shooting for premium brand campaigns",
        "objects": "Premium products (packaging with embossed details, luxury materials, artisan craftsmanship), Styled props (marble pedestals, linen backdrops, botanical accents), Professional lighting reflections and highlights",
        "surfaces": "Studio surfaces (Carrara marble slabs, oiled walnut, Belgian linen, seamless paper), Premium backgrounds (gradient lighting, textured plaster)",
        "style_suffix": "award-winning commercial product photography, premium brand quality, photorealistic 8k",
    },
    "Food & Beverage": {
        "role": "award-winning Food Photographer shooting for Bon Appétit or Food & Wine magazine",
        "objects": "Artisan food presentation (farm-to-table ingredients, chef-plated dishes, fresh herb garnishes), Designer tableware (Heath Ceramics, handmade artisan pottery, vintage silver), Styled props (hand-woven linen napkins, reclaimed wood boards)",
        "surfaces": "Editorial surfaces (live-edge walnut, weathered marble, hand-poured concrete, aged copper), Curated backgrounds (lime-washed plaster, natural linen)",
        "style_suffix": "award-winning editorial food photography, Bon Appétit quality, photorealistic 8k",
    },
    "Fashion & Lifestyle": {
        "role": "award-winning Fashion Photographer shooting for Vogue or Elle Decor magazine",
        "objects": "Luxury fashion details (couture fabric textures, hand-stitched details, designer hardware), Premium accessories (fine jewelry, artisan leather goods), Styled environment (designer furniture, curated art pieces)",
        "surfaces": "Editorial backgrounds (limewashed walls, polished terrazzo, draped silk), Premium flooring (herringbone parquet, honed limestone, hand-knotted rugs)",
        "style_suffix": "award-winning fashion editorial photography, Vogue quality, photorealistic 8k",
    },
}

# PASS 2B: Local Tile Description (run on each tile crop)
# v3.9: "Render-to-Real" Strategic Pivot
# - CORE PHILOSOPHY: Describe the INTENDED REALITY, not the low-quality render
# - MATERIAL UPGRADE RULE: Supply texture the render lacks
# - CROP BOUNDARY RULE: No hallucinations of off-frame elements
# - Brand names as style anchors for SD/Flux
# - Complex color vocabulary (charcoal, cognac, slate)
TEXTURE_TILE_PROMPT = """ROLE: You are an expert Architectural Visualizer.
Your job is to analyze low-quality 3D renders and write a prompt that will UPGRADE them into photorealistic, high-end photography.

CRITICAL PHILOSOPHY: You are NOT describing the render. You are describing the REALITY the render is trying to represent.
{user_guidance_section}
============================================================
THE "MATERIAL UPGRADE" RULE (CRITICAL)
============================================================
The input image may lack texture. You MUST supply it.

BAD (preserves low-quality render):
- "A grey floor"
- "A brown cabinet"
- "A blue wall"

GOOD (upgrades to reality):
- "Polished concrete flooring with subtle aggregate texture and satin finish"
- "Rift-sawn white oak cabinetry with matte oil finish"
- "Slate blue Venetian plaster with subtle trowel marks"

RULE: Always interpret the material INTENT. If a surface looks flat or noisy, upgrade it to its premium real-world equivalent. Use specific material names: Walnut, Carrara Marble, Travertine, Bouclé, Belgian Linen, Corten Steel, Honed Limestone.

============================================================
THE "CROP BOUNDARY" RULE (No Hallucinations)
============================================================
You are looking at a TILE - a small slice of a larger image.

STRICT RULES:
- Describe ONLY what is visible within this crop
- Do NOT infer roofs, ceilings, or floors if cut off by the frame
- If the top of a wall is cropped, describe the wall texture ONLY - no "visible roofline"
- Never use "in the background" or "surrounded by" if those areas are NOT visible
- If you cannot see the ground, do not mention "gravel driveway" or "lawn"
- If you cannot see the sky, do not mention "clear blue sky" or "overcast clouds"

This prevents the image generator from trying to squeeze non-existent elements into the frame.

============================================================
GLOBAL CONSISTENCY (One World Rule)
============================================================
These tiles will be stitched together. Your description MUST match:

- Scene Context: {scene_description}
- Season/Weather: (infer from scene - if winter: snow/bare trees, if summer: green/lush)
- Lighting: (infer from scene - maintain consistent direction and mood)
- Materials: Floor={floor_material}, Walls={wall_material}, Ceiling={ceiling_material}

Cross-tile rules:
- If driveway is "pea gravel" in one tile, it cannot be "concrete" in adjacent tile
- All shadows must fall in the SAME direction
- If one tile shows rain/wet surfaces, ALL tiles must show moisture

============================================================
VOCABULARY GUIDE (Premium Descriptors)
============================================================

COLORS (use complex names that imply texture):
- Instead of "grey" → use "charcoal", "slate", "concrete", "graphite", "pewter"
- Instead of "brown" → use "cognac", "umber", "walnut", "espresso", "caramel"
- Instead of "blue" → use "slate blue", "navy", "cerulean", "indigo", "cobalt"
- Instead of "white" → use "ivory", "cream", "alabaster", "bone", "eggshell"
- Instead of "green" → use "sage", "moss", "olive", "forest", "emerald"

BRAND NAMES (use as style anchors - SD/Flux trained on these):
- Furniture: "in the style of B&B Italia", "Minotti sectional", "Eames lounge chair", "Noguchi coffee table", "Herman Miller"
- Lighting: "Flos Arco lamp", "Artemide pendant", "Louis Poulsen fixture", "Tom Dixon pendant"
- Appliances: "Miele", "Sub-Zero", "Wolf range", "Gaggenau"

These brand names define SHAPE and QUALITY for the image generator.

BANNED WORDS (never use):
- "render", "3D model", "CGI", "computer generated", "digital"
- Use instead: "photograph", "built structure", "real", "photorealistic"

============================================================
DESCRIPTION STRUCTURE (Follow This Order)
============================================================

1. SUBJECT/MAIN STRUCTURE (What space is this?):
   - Architectural function: carport, living room, kitchen, terrace, office
   - Style: modern minimalist, industrial, mid-century, Scandinavian, Mediterranean
   - Example: "A contemporary open-plan living room..."

2. SPECIFIC MATERIALS & TEXTURES (Upgrade everything):
   - Apply the Material Upgrade Rule to every surface
   - Use premium material names with finish details
   - Example: "...with wide-plank European white oak flooring showing honey-gold grain variation..."

3. OBJECTS WITH STYLE ANCHORS (Brand names for quality):
   - Identify furniture, fixtures, plants with specific names
   - Use brand names to define shape and quality
   - Example: "...a low-profile sectional in the style of Minotti, upholstered in charcoal bouclé..."

4. LIGHTING & ATMOSPHERE:
   - Light source and direction (match Global Style)
   - Quality: soft diffused, harsh direct, golden hour, overcast
   - Example: "...soft northern daylight floods through floor-to-ceiling glazing, creating gentle shadows..."

5. STYLE TAGS (End with these):
   - Save magazine-quality language for the END only
   - Example: "...sharp focus, ultra detailed, professional architectural photography, 8K"

SPATIAL CONTEXT (Weave naturally):
- Camera Angle: {camera_angle}
- Depth Layer: {depth_layer}
- Frame Position: {perspective}

TILE INFO:
- Position: {position} (row {row}, col {col})
- Surface Hint: {tile_label}

OBJECTS IN THIS TILE:
{objects}

SURFACES IN THIS TILE:
{surfaces}

============================================================
DO NOT (Critical Violations)
============================================================
- Describe the "low quality" of the input - UPGRADE it instead
- Use generic colors (grey, brown, blue) - use textured colors (charcoal, cognac, slate)
- Hallucinate elements cut off by the frame (roofs, floors, sky not visible)
- Say "render", "CGI", "3D model" - say "photograph", "real", "built"
- Use "nice", "beautiful", "decoration" - use specific material names
- Contradict the Global Style or adjacent tiles
- Put magazine language in the body - save for end tags only

============================================================
EXAMPLES - "Render-to-Real" Style
============================================================

Example 1 - Carport (Exterior, Material Upgrade Focus):
"A modern residential carport with a sleek charcoal SUV parked on premium pea gravel with steel edging. The overhead structure features horizontal Western red cedar slats with clear oil finish, supported by powder-coated black steel columns. A mature Japanese maple provides dappled shade with deep burgundy palmate foliage. The gravel surface shows warm honey-beige 10mm aggregate with crisp borders against emerald lawn. Board-formed concrete walls with visible tie holes add industrial texture. Soft afternoon sunlight filters through cedar creating rhythmic shadows on stone below. Sharp focus, ultra detailed, professional architectural photography, 8K"

Example 2 - Living Room (Interior, Brand Anchors):
"A contemporary open-plan living room with double-height ceilings. A sectional in the style of B&B Italia in charcoal bouclé anchors the space, paired with a Noguchi coffee table in oiled walnut. A Flos Arco lamp in brushed stainless arcs overhead. Wide-plank European white oak flooring with matte finish shows honey-gold grain variation. The feature wall displays hand-troweled Venetian plaster in warm limestone tones. Floor-to-ceiling glazing reveals a landscaped garden. Soft northern daylight creates gentle shadows highlighting texture interplay. Sharp focus, ultra detailed, professional architectural photography, 8K"

Example 3 - Kitchen (Interior, Material Upgrade):
"A chef's kitchen with a waterfall-edge island in honed Calacatta marble with dramatic grey veining. Rift-sawn white oak cabinetry with push-latch mechanisms flanks integrated Sub-Zero refrigeration. Tom Dixon Beat pendants in brushed brass hang at varying heights, their hammered finish catching ambient light. A Wolf range with red control knobs sits within a marble alcove. Polished concrete flooring with subtle aggregate texture grounds the space. Warm afternoon light streams through French doors, highlighting natural stone patterns. Sharp focus, ultra detailed, professional architectural photography, 8K"

Example 4 - Cropped Wall Tile (Demonstrating Crop Boundary Rule):
"Textured board-formed concrete wall section with visible tie holes arranged in grid pattern. The concrete shows subtle grey variation with warm undertones and satin finish from form release. Hairline control joints create geometric divisions. Dappled sunlight from off-frame creates soft diagonal shadows across the surface. Sharp focus, ultra detailed, architectural detail photography, 8K"

(Note: This example does NOT mention roof, ground, or sky because they are not visible in the crop.)

Output ONLY the description. No explanations or preamble."""

# Phrases that indicate instruction leakage (to be cleaned from output)
# Order matters: longer phrases should come first to avoid partial matches
INSTRUCTION_LEAKAGE_PHRASES = [
    # Long phrases first (more specific)
    "explain the objects, furniture's, material and lights, explain everything very sharp, i need very very sharp image and make it to real photo",
    "explain the objects, furniture's, material and lights, explain everything very sharp",
    "furniture's, material and lights, explain everything very sharp",
    "i need very very sharp image and make it to real photo",
    "explain everything very sharp",
    "i need very very sharp image",
    "make it to real photo",
    "explain the objects",
    "furniture's, material and lights",
    "i need very sharp",
    "i need sharp",
    "real photo quality",
    "here is the description",
    "i will describe",
    "this tile shows",
    "in this tile",
    "the tile contains",
    "as requested",
    "as you asked",
    "explain everything",
    # Common trailing artifacts
    "material detail,",
    ", ,",
]

# Legacy prompts kept for backwards compatibility
SCENE_DESCRIPTION_PROMPT = GLOBAL_STYLE_PROMPT  # Alias for v2.1 compatibility
TILE_GRID_PROMPT = MATERIALS_GRID_PROMPT  # Alias for v2.1 compatibility
TILE_PROMPT_TEMPLATE_V21 = TEXTURE_TILE_PROMPT  # Alias for v2.1 compatibility

# ============================================================================
# LEGACY PROMPTS (kept for reference/advanced use)
# ============================================================================

# Style guide extraction prompt - detailed version (used internally for style_reference)
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

QUALITY REQUIREMENTS - CRITICAL FOR SHARP, REALISTIC OUTPUT:

SHARPNESS (prevents blurry output):
- START with "Sharp focus on" or "Crisp detail of" or "Highly detailed"
- Include 2-3 sharpness words: "sharp edges", "crisp", "fine detail", "clearly defined", "precise", "distinct"
- AVOID: "softly", "gently blurred", "subtly fading", "soft gradients"

PHOTOREALISTIC STYLE (ensures real photo look):
- Include "photorealistic" or "real photograph" or "DSLR quality" somewhere in the description
- End paragraph with style tags: "professional photography, 8K resolution, photorealistic"
- Use camera terms: "shallow depth of field", "natural lighting", "lens flare", "bokeh" where appropriate

CRITICAL RULES:
- Write as a FLOWING PARAGRAPH, not bullet points or categories
- Keep it to 100-150 words (CLIP has token limits)
- Do NOT include notes, disclaimers, meta-commentary, or explanations
- Do NOT write "No text is visible" - just end the paragraph if no text exists
- If text IS visible, end with it in quotes (e.g., The sign reads "OPEN")
- Output ONLY the description paragraph, nothing else

Example format (with sharpness + photorealistic style):
"Sharp focus on a leather armchair in the left foreground, its worn brown surface showing crisp creasing detail at the armrests with clearly defined stitching, photorealistic texture capturing every pore of the leather. Precise natural window light from the upper right creates distinct tonal gradients across the seat cushion, with hard-edged shadows pooling beneath the curved wooden legs. The floor beneath reveals highly detailed oak planks with sharp grain patterns, each plank edge crisply defined, professional photography, 8K resolution, photorealistic."
"""

# Result cache (LRU with max size to prevent memory leaks)
TILE_PROMPTER_TURBO_CACHE = OrderedDict()
TILE_PROMPTER_TURBO_CACHE_MAX_SIZE = 50  # Max entries (~2-5KB each)


def _cache_set(key, value):
    """Add to cache with LRU eviction."""
    global TILE_PROMPTER_TURBO_CACHE
    # Remove oldest if at capacity
    while len(TILE_PROMPTER_TURBO_CACHE) >= TILE_PROMPTER_TURBO_CACHE_MAX_SIZE:
        oldest_key = next(iter(TILE_PROMPTER_TURBO_CACHE))
        del TILE_PROMPTER_TURBO_CACHE[oldest_key]
        print(f"[Smart Tile Prompter Turbo] Cache full, evicted oldest entry")
    TILE_PROMPTER_TURBO_CACHE[key] = value


def _cache_get(key):
    """Get from cache with LRU update (move to end)."""
    global TILE_PROMPTER_TURBO_CACHE
    if key in TILE_PROMPTER_TURBO_CACHE:
        TILE_PROMPTER_TURBO_CACHE.move_to_end(key)
        return TILE_PROMPTER_TURBO_CACHE[key]
    return None

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


def start_llama_server(model_size, port, gpu_layers=99, context_size=16384):
    """Start llama-server for the specified model size.

    Returns:
        tuple: (process, log_file_path) or (None, error_message)
    """
    model_short = model_size.split()[0]  # "4B" from "4B (Balanced)"

    if not SCRIPT_PATH.exists():
        error_msg = f"Server script not found at {SCRIPT_PATH}"
        print(f"[Smart Tile Prompter Turbo] Warning: {error_msg}")
        return None, error_msg

    env = os.environ.copy()
    env["CTX"] = str(context_size)
    env["GPU_LAYERS"] = str(gpu_layers)

    # Create log file to capture output for debugging
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"qwenvl_server_{model_short}.log"

    # Open log file for writing (truncate previous content)
    # Use 'with' to ensure file handle is closed after process starts
    # The subprocess inherits the fd, so it can still write to it
    with open(log_file, 'w') as log_handle:
        process = subprocess.Popen(
            ["bash", str(SCRIPT_PATH), model_short],
            stdout=log_handle,
            stderr=subprocess.STDOUT,  # Combine stderr into stdout
            start_new_session=True,
            env=env
        )
    # File handle closed here - subprocess keeps its copy of the fd
    print(f"[Smart Tile Prompter Turbo] Started server for {model_short} on port {port}")
    print(f"  GPU Layers: {gpu_layers}, Context: {context_size}")
    print(f"  Log file: {log_file}")
    return process, str(log_file)


def ensure_server_running(model_size, auto_start=True, gpu_layers=99, context_size=16384):
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
    result = start_llama_server(model_size, port, gpu_layers, context_size)

    # Check if start_llama_server returned an error
    if result[0] is None:
        return False, result[1]

    process, log_file = result

    # Wait for server to be ready (up to 60 seconds)
    for i in range(60):
        time.sleep(1)

        # Check if process has exited (indicates startup failure)
        exit_code = process.poll()
        if exit_code is not None:
            # Process exited - read log file for error details
            error_details = ""
            try:
                with open(log_file, 'r') as f:
                    log_content = f.read()
                    # Get last 20 lines for error context
                    log_lines = log_content.strip().split('\n')
                    error_details = '\n'.join(log_lines[-20:])
            except Exception as e:
                error_details = f"Could not read log: {e}"

            # Check for common errors in log
            if "already running" in error_details.lower() or "✅" in error_details:
                # Server was already running (script exited with 0)
                print(f"[Smart Tile Prompter Turbo] Server was already running")
                time.sleep(2)  # Give it a moment
                if check_server_ready(port):
                    return True, None

            error_msg = (
                f"Server process exited with code {exit_code}\n\n"
                f"Log output:\n{error_details}\n\n"
                f"Try manually: ./start_qwenvl_server.sh {model_short}"
            )
            print(f"[Smart Tile Prompter Turbo] Server startup failed: exit code {exit_code}")
            return False, error_msg

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

    # Timeout - read log for debugging
    log_tail = ""
    try:
        with open(log_file, 'r') as f:
            log_lines = f.read().strip().split('\n')
            log_tail = '\n'.join(log_lines[-10:])
    except:
        pass

    error_msg = (
        f"Server failed to start after 60s.\n\n"
        f"Recent log:\n{log_tail}\n\n"
        f"Full log: {log_file}\n"
        f"Try manually: ./start_qwenvl_server.sh {model_short}"
    )
    return False, error_msg


def stop_server(model_size):
    """Stop the QwenVL server for the specified model size."""
    config = MODEL_CONFIGS.get(model_size, MODEL_CONFIGS["4B (Balanced)"])
    port = config['port']
    model_short = model_size.split()[0]

    if not check_server_ready(port):
        print(f"[Smart Tile Prompter Turbo] Server for {model_short} is not running on port {port}")
        return False

    print(f"[Smart Tile Prompter Turbo] Stopping server for {model_short} on port {port}...")

    try:
        # Method 1: Find PID via ss and kill it
        result = subprocess.run(
            ['ss', '-tlnp', f'sport = :{port}'],
            capture_output=True, text=True
        )

        output = result.stdout
        pid = None
        if f':{port}' in output:
            pid_match = re.search(r'pid=(\d+)', output)
            if pid_match:
                pid = int(pid_match.group(1))

        if pid:
            # Try SIGTERM first
            print(f"  Found PID {pid}, sending SIGTERM...")
            os.kill(pid, signal.SIGTERM)

            # Wait up to 5 seconds for graceful shutdown
            for i in range(10):
                time.sleep(0.5)
                if not check_server_ready(port):
                    print(f"[Smart Tile Prompter Turbo] Server stopped successfully (SIGTERM)")
                    return True

            # SIGTERM didn't work, try SIGKILL
            print(f"  SIGTERM failed, sending SIGKILL...")
            try:
                os.kill(pid, signal.SIGKILL)
                time.sleep(1)
                if not check_server_ready(port):
                    print(f"[Smart Tile Prompter Turbo] Server stopped successfully (SIGKILL)")
                    return True
            except ProcessLookupError:
                # Process already dead
                print(f"[Smart Tile Prompter Turbo] Server stopped successfully")
                return True

        # Method 2: Fallback - use fuser to kill process on port
        print(f"  Trying fuser to kill port {port}...")
        subprocess.run(['fuser', '-k', f'{port}/tcp'], capture_output=True)
        time.sleep(1)
        if not check_server_ready(port):
            print(f"[Smart Tile Prompter Turbo] Server stopped successfully (fuser)")
            return True

        # Method 3: Fallback - pkill with multiple patterns
        print(f"  Trying pkill patterns...")
        patterns = [
            f'llama-server.*--port.*{port}',
            f'llama-server.*{port}',
            f'llama.*{port}',
        ]
        for pattern in patterns:
            subprocess.run(['pkill', '-9', '-f', pattern], capture_output=True)

        time.sleep(1)
        if not check_server_ready(port):
            print(f"[Smart Tile Prompter Turbo] Server stopped successfully (pkill)")
            return True

        print(f"[Smart Tile Prompter Turbo] WARNING: Could not stop server on port {port}")
        return False

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


def generate_grid_positions(tiles_x, tiles_y):
    """Generate grid position string for TILE_GRID_PROMPT.

    Returns a string like:
    (0,0): top-left, (0,1): top-right
    (1,0): bottom-left, (1,1): bottom-right
    """
    lines = []
    for y in range(tiles_y):
        row_items = []
        for x in range(tiles_x):
            pos = get_position_name(x, y, tiles_x, tiles_y)
            row_items.append(f"({y},{x}): {pos}")
        lines.append(", ".join(row_items))
    return "\n".join(lines)


def get_neighbor_labels(tile_labels, x, y, tiles_x, tiles_y):
    """Get labels for neighboring tiles.

    Args:
        tile_labels: Dict mapping "row,col" to label string
        x, y: Current tile coordinates
        tiles_x, tiles_y: Grid dimensions

    Returns:
        Dict with keys: above, below, left, right
    """
    def get_label(row, col):
        if row < 0 or row >= tiles_y or col < 0 or col >= tiles_x:
            return "edge of image"
        return tile_labels.get(f"{row},{col}", "unknown")

    return {
        "above": get_label(y - 1, x),
        "below": get_label(y + 1, x),
        "left": get_label(y, x - 1),
        "right": get_label(y, x + 1),
    }


def parse_grid_json(response_text):
    """Parse JSON response from TILE_GRID_PROMPT.

    Handles cases where model adds extra text before/after JSON.
    Returns dict with 'materials' and 'tiles' keys.
    """
    # Try to extract JSON from response
    text = response_text.strip()

    # Find JSON block
    start = text.find('{')
    end = text.rfind('}') + 1

    if start == -1 or end == 0:
        # No JSON found, return defaults
        print("[Smart Tile Prompter Turbo v2.1] Warning: No JSON in grid response, using defaults")
        return {
            "materials": {"floor": "floor", "walls": "walls", "ceiling": "ceiling"},
            "tiles": {}
        }

    json_str = text[start:end]

    try:
        data = json.loads(json_str)
        # Ensure required keys exist
        if "materials" not in data:
            data["materials"] = {"floor": "floor", "walls": "walls", "ceiling": "ceiling"}
        if "tiles" not in data:
            data["tiles"] = {}
        return data
    except json.JSONDecodeError as e:
        print(f"[Smart Tile Prompter Turbo v3.4] Warning: JSON parse error: {e}")
        return {
            "materials": {"floor": "floor", "walls": "walls", "ceiling": "ceiling"},
            "tiles": {}
        }


def clean_prompt_output(raw_output):
    """Clean VLLM output to remove instruction leakage.

    The model sometimes echoes instructions back. This function removes
    conversational phrases that would confuse the image generator.

    Args:
        raw_output: Raw text from VLLM

    Returns:
        Cleaned prompt text safe for Z-Image Turbo
    """
    cleaned = raw_output

    # Remove known instruction leakage phrases (case-insensitive)
    for phrase in INSTRUCTION_LEAKAGE_PHRASES:
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        cleaned = pattern.sub("", cleaned)

    # Clean up punctuation artifacts using regex for robustness
    # Fix ", ." or ",  ." -> ". " (comma followed by spaces and period)
    cleaned = re.sub(r',\s*\.', '.', cleaned)
    # Fix ", ," or ",   ," -> "," (comma-spaces-comma)
    cleaned = re.sub(r',\s*,', ',', cleaned)
    # Remove double/triple periods
    cleaned = re.sub(r'\.{2,}', '.', cleaned)
    # Remove double/triple commas
    cleaned = re.sub(r',{2,}', ',', cleaned)
    # Fix multiple spaces
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)
    # Remove comma before period
    cleaned = cleaned.replace(',.', '.')
    # Remove leading/trailing commas and spaces
    cleaned = cleaned.strip().strip(",").strip()

    # Log if cleaning was needed
    if cleaned != raw_output:
        removed_len = len(raw_output) - len(cleaned)
        print(f"    [Cleaner] Removed {removed_len} chars of instruction leakage")

    return cleaned


def call_qwenvl_api(image_b64, prompt, model_size, quality_preset, seed=1, max_retries=3):
    """Call QwenVL GGUF API and return response text.

    Args:
        image_b64: Base64 encoded image
        prompt: Text prompt to send
        model_size: Model size key (e.g., "4B (Balanced)")
        quality_preset: Quality preset name
        seed: Random seed for generation
        max_retries: Number of retries for HTTP 500 errors (default 3)

    Returns:
        Response text or error message
    """
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

    # Retry loop for transient server errors (HTTP 500)
    for attempt in range(max_retries):
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
        except requests.exceptions.HTTPError as e:
            # Retry on HTTP 500 (server error) - often caused by memory pressure
            if response.status_code == 500 and attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"[Smart Tile Prompter Turbo] HTTP 500 error, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            return f"ERROR: {type(e).__name__}: {str(e)}"
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {str(e)}"

    # Should not reach here, but safety fallback
    return "ERROR: Max retries exceeded"


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


def get_spatial_context(row, col, tiles_y, tiles_x, use_case="Interior Design"):
    """
    Derive camera angle, depth layer, and perspective for a tile.

    This enables higher denoise values (0.5-0.8) by giving the diffusion model
    spatial anchoring information - it knows WHERE in 3D space it's rendering.

    Args:
        row: Row index (0 = top)
        col: Column index (0 = left)
        tiles_y, tiles_x: Total grid dimensions
        use_case: Photography mode (affects terminology)

    Returns:
        Dict with camera_angle, depth_layer, perspective, spatial_phrase
    """
    # Vertical position ratio (0.0 = top, 1.0 = bottom)
    v_ratio = row / max(tiles_y - 1, 1) if tiles_y > 1 else 0.5

    # Horizontal position ratio (0.0 = left, 1.0 = right)
    h_ratio = col / max(tiles_x - 1, 1) if tiles_x > 1 else 0.5

    # Camera angle based on vertical position
    if v_ratio < 0.25:
        camera_angle = "looking up, low camera angle"
        depth_layer = "background/sky layer"
    elif v_ratio < 0.5:
        camera_angle = "eye-level, straight-on view"
        depth_layer = "midground layer"
    elif v_ratio < 0.75:
        camera_angle = "eye-level, slight downward tilt"
        depth_layer = "midground to foreground"
    else:
        camera_angle = "looking down, high camera angle"
        depth_layer = "foreground/ground layer"

    # Perspective based on horizontal position
    if h_ratio < 0.33:
        perspective = "left side of frame, off-center composition"
    elif h_ratio > 0.67:
        perspective = "right side of frame, off-center composition"
    else:
        perspective = "center of frame, balanced composition"

    # Use case specific vocabulary for exteriors
    if use_case == "Exterior Architecture":
        if v_ratio < 0.25:
            depth_layer = "sky and roofline layer"
        elif v_ratio > 0.75:
            depth_layer = "ground plane, lawn and hardscape"

    # Build spatial phrase for prompt
    spatial_phrase = f"{camera_angle}, {depth_layer}, {perspective}"

    return {
        "camera_angle": camera_angle,
        "depth_layer": depth_layer,
        "perspective": perspective,
        "spatial_phrase": spatial_phrase,
        "v_ratio": v_ratio,
        "h_ratio": h_ratio,
    }


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


def compute_cache_key(image_tensor, tiles_x, tiles_y, model_size, quality_preset, consistency_level, use_case, user_context, vllm_guidance, seed):
    """Create unique hash for caching."""
    hasher = hashlib.md5()
    hasher.update(image_tensor.cpu().numpy().tobytes())
    hasher.update(f"{tiles_x}x{tiles_y}".encode())
    hasher.update(model_size.encode())
    hasher.update(quality_preset.encode())
    hasher.update(consistency_level.encode())
    hasher.update(use_case.encode())
    hasher.update(user_context.encode())
    hasher.update(vllm_guidance.encode())  # v3.5 VLLM guidance
    hasher.update(str(seed).encode())
    hasher.update(b"turbo_v39_render_to_real")  # v3.9 Render-to-Real Strategic Pivot
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
                "use_case": (list(USE_CASE_PROMPTS.keys()), {
                    "default": "Interior Design",
                    "tooltip": "Photography use case - adjusts vocabulary and focus areas"
                }),
            },
            "optional": {
                "bundle": ("SMART_TILE_BUNDLE", {
                    "tooltip": "Connect bundle from Smart Tile Calculator (auto-fills tiles_x, tiles_y, image)"
                }),
                "user_context": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional: Text added directly TO the output prompts (e.g., 'luxury cabin')"
                }),
                "vllm_guidance": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Instructions FOR the VLLM: mood, material preferences, style hints (e.g., 'Describe grass as Kentucky bluegrass with dew')"
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
                    "default": 16384,
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
                "clear_cache": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Clear ALL cached prompts before running (use if prompts don't match image)"
                }),
            }
        }

    def generate(self, image, tiles_x, tiles_y, model_size, quality_preset,
                 consistency_level, use_case="Interior Design", bundle=None, user_context="",
                 vllm_guidance="", tile_overlap=0, use_cache=True, auto_start_server=True,
                 gpu_layers=99, context_size=16384, stop_server_after=False, seed=1, clear_cache=False):
        """Generate Z-Image-Turbo optimized prompts for all tiles (v3.6 Magazine Quality)."""
        global TILE_PROMPTER_TURBO_CACHE

        # Clear cache if requested
        if clear_cache:
            old_count = len(TILE_PROMPTER_TURBO_CACHE)
            TILE_PROMPTER_TURBO_CACHE.clear()
            print(f"[Smart Tile Prompter Turbo] ⚠️ Cache CLEARED ({old_count} entries removed)")

        # If bundle provided, extract values (overrides individual inputs)
        if bundle is not None:
            image = bundle.get("scaled_image", image)
            tiles_x = bundle.get("tiles_x", tiles_x)
            tiles_y = bundle.get("tiles_y", tiles_y)
            print(f"[Smart Tile Prompter Turbo v2.1] Using bundle: {tiles_x}x{tiles_y} tiles")

        start_time = time.time()
        total_tiles = tiles_x * tiles_y

        print(f"\n[Smart Tile Prompter Turbo v3.9] Starting {tiles_x}x{tiles_y} = {total_tiles} tiles")
        print(f"  Model: {model_size}, Preset: {quality_preset}, Consistency: {consistency_level}")
        print(f"  Seed: {seed}")
        print(f"  Mode: Render-to-Real + Use Case: {use_case} (v3.9 Material Upgrade)")

        # ===== CHECK SERVER =====
        server_ok, error_msg = ensure_server_running(
            model_size, auto_start_server, gpu_layers, context_size
        )
        if not server_ok:
            # Return error with all 8 required outputs (including empty bundle)
            return (error_msg, error_msg, "{}", "", tiles_x, tiles_y, error_msg, {})

        # Check cache
        cache_key = None
        if use_cache:
            cache_key = compute_cache_key(
                image, tiles_x, tiles_y, model_size, quality_preset,
                consistency_level, use_case, user_context, vllm_guidance, seed
            )
            cached = _cache_get(cache_key)
            if cached is not None:
                print(f"[Smart Tile Prompter Turbo] Cache HIT (key: {cache_key[:12]}...)")
                print(f"  WARNING: Using cached prompts! Set use_cache=False if image changed.")
                return cached
            else:
                print(f"[Smart Tile Prompter Turbo] Cache MISS (key: {cache_key[:12]}..., {len(TILE_PROMPTER_TURBO_CACHE)} cached entries)")
        else:
            print(f"[Smart Tile Prompter Turbo] Cache DISABLED - analyzing image fresh")

        # ===== PHASE 1A: GLOBAL STYLE EXTRACTION =====
        print("[Smart Tile Prompter Turbo v3.9] Phase 1A: Extracting global style (environment + lighting + colors)...")

        full_image_b64 = image_to_base64(image, max_size=1536)
        scene_description = call_qwenvl_api(
            full_image_b64,
            SCENE_DESCRIPTION_PROMPT,
            model_size,
            quality_preset,
            seed=seed
        )

        # Add user context if provided
        if user_context and user_context.strip():
            scene_description = f"{user_context.strip()}. {scene_description}"

        print(f"[Smart Tile Prompter Turbo v3.9] Global Style: {scene_description[:100]}...")

        # ===== PHASE 1B: MATERIALS + SURFACE GRID =====
        print("[Smart Tile Prompter Turbo v3.9] Phase 1B: Extracting materials + surface grid...")

        grid_positions = generate_grid_positions(tiles_x, tiles_y)
        grid_prompt = TILE_GRID_PROMPT.format(
            tiles_x=tiles_x,
            tiles_y=tiles_y,
            grid_positions=grid_positions
        )

        grid_response = call_qwenvl_api(
            full_image_b64,
            grid_prompt,
            model_size,
            quality_preset,
            seed=seed + 500
        )

        # Parse grid JSON response
        grid_data = parse_grid_json(grid_response)
        materials = grid_data.get("materials", {})
        tile_labels = grid_data.get("tiles", {})

        floor_material = materials.get("floor", "floor")
        wall_material = materials.get("walls", "walls")
        ceiling_material = materials.get("ceiling", "ceiling")

        print(f"[Smart Tile Prompter Turbo v3.9] Materials: floor={floor_material}, walls={wall_material}")
        print(f"[Smart Tile Prompter Turbo v3.9] Surface hints: {len(tile_labels)} tiles mapped")

        # ===== BUILD VLLM GUIDANCE SECTION =====
        # This is instructions FOR the VLLM, not text added to output
        if vllm_guidance and vllm_guidance.strip():
            user_guidance_section = f"""
USER STYLE GUIDANCE (IMPORTANT - follow these instructions):
{vllm_guidance.strip()}
- Apply these preferences when describing materials, mood, and style
- Use the specific terms/names provided above
"""
            print(f"[Smart Tile Prompter Turbo v3.9] VLLM Guidance: {vllm_guidance[:80]}...")
        else:
            user_guidance_section = ""

        # ===== PHASE 2: LOCAL TEXTURE EXTRACTION (WITH SPATIAL CONTEXT) =====
        print(f"[Smart Tile Prompter Turbo v3.9] Phase 2: Generating {total_tiles} render-to-real prompts...")

        tile_prompts_list = []

        for y in range(tiles_y):
            for x in range(tiles_x):
                tile_num = y * tiles_x + x + 1
                position = get_position_name(x, y, tiles_x, tiles_y)

                # NEW v3.4: Get spatial context for this tile
                spatial = get_spatial_context(y, x, tiles_y, tiles_x, use_case)

                print(f"  Tile {tile_num}/{total_tiles}: {position} ({spatial['depth_layer']})...")

                # Crop tile
                tile_image = crop_tile(image, x, y, tiles_x, tiles_y, tile_overlap)
                tile_b64 = image_to_base64(tile_image, max_size=1024)

                # Get surface hint for this tile
                tile_label = tile_labels.get(f"{y},{x}", "surface detail")

                # Build tile prompt using v3.5 Spatial Context + VLLM Guidance template
                # Uses use_case specific vocabulary for role, objects, and surfaces
                use_case_config = USE_CASE_PROMPTS.get(use_case, USE_CASE_PROMPTS["Interior Design"])
                tile_prompt = TILE_PROMPT_TEMPLATE_V21.format(
                    role=use_case_config["role"],
                    objects=use_case_config["objects"],
                    surfaces=use_case_config["surfaces"],
                    scene_description=scene_description,
                    floor_material=floor_material,
                    wall_material=wall_material,
                    ceiling_material=ceiling_material,
                    position=position,
                    row=y + 1,
                    col=x + 1,
                    tile_label=tile_label,
                    # v3.4 spatial context fields:
                    camera_angle=spatial["camera_angle"],
                    depth_layer=spatial["depth_layer"],
                    perspective=spatial["perspective"],
                    spatial_phrase=spatial["spatial_phrase"],
                    # v3.5 VLLM guidance section:
                    user_guidance_section=user_guidance_section,
                )

                # Call API (use seed + tile_num for reproducible but unique per-tile results)
                local_texture_raw = call_qwenvl_api(
                    tile_b64,
                    tile_prompt,
                    model_size,
                    quality_preset,
                    seed=seed + tile_num
                )

                # Clean instruction leakage from VLLM output
                local_texture = clean_prompt_output(local_texture_raw)

                # Z-Image Turbo Assembly: [Local Texture] + [Global Style]
                # This follows the hierarchical structure: Subject + Environment + Style
                final_prompt = f"{local_texture}, {scene_description}"

                tile_prompts_list.append({
                    "tile": tile_num,
                    "position": position,
                    "x": x,
                    "y": y,
                    "prompt": final_prompt,
                    "local_texture": local_texture  # Store separately for debugging
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
        image_hash = hashlib.md5(image.cpu().numpy().tobytes()).hexdigest()[:12]
        debug_lines = [
            "=" * 50,
            "Smart Tile Prompter Turbo v3.9.0 (Render-to-Real Strategic Pivot)",
            "=" * 50,
            f"Image Hash: {image_hash} (verify correct image)",
            f"Image Shape: {list(image.shape)}",
            f"Tiles: {tiles_x}x{tiles_y} = {total_tiles} total",
            f"Model: {model_size}",
            f"Preset: {quality_preset}",
            f"Overlap: {tile_overlap}px",
            f"Seed: {seed}",
            f"Cache Key: {cache_key[:12] if cache_key else 'N/A'}...",
            "",
            f"v3.9 RENDER-TO-REAL + Use Case: {use_case}",
            f"  - Material Upgrade Rule: Supply texture the render lacks",
            f"  - Crop Boundary Rule: No hallucinations of off-frame elements",
            f"  - Brand names as style anchors (B&B Italia, Minotti, Eames)",
            f"  - Complex color vocabulary (charcoal, cognac, slate)",
            f"  - VLLM Guidance: {'[PROVIDED]' if vllm_guidance else '[NONE]'}",
            "",
        ]
        # Add VLLM guidance to debug if provided
        if vllm_guidance and vllm_guidance.strip():
            debug_lines.extend([
                "VLLM Guidance (instructions for analysis):",
                "-" * 40,
                vllm_guidance.strip(),
                "-" * 40,
                "",
            ])
        debug_lines.extend([
            "Spatial Context per Tile:",
            "-" * 40,
        ])
        # Add spatial context for each tile
        for y in range(tiles_y):
            for x in range(tiles_x):
                spatial = get_spatial_context(y, x, tiles_y, tiles_x, use_case)
                debug_lines.append(f"  [{y},{x}]: {spatial['spatial_phrase']}")
        debug_lines.extend([
            "-" * 40,
            "",
            f"API Calls: {total_tiles + 2} (1 style + 1 grid + {total_tiles} tiles)",
            f"Total Time: {elapsed:.1f}s",
            f"Time per call: {elapsed / (total_tiles + 2):.1f}s",
            "",
            "Global Style:",
            "-" * 40,
            scene_description,
            "-" * 40,
            "",
            "Materials (consistent across all tiles):",
            f"  Floor: {floor_material}",
            f"  Walls: {wall_material}",
            f"  Ceiling: {ceiling_material}",
            "",
            "Surface Hints per Tile:",
            "-" * 40,
        ])
        for key, label in tile_labels.items():
            debug_lines.append(f"  [{key}]: {label}")
        debug_lines.extend([
            "-" * 40,
            "",
            "SEGS Integration:",
            f"  tile_prompts_labels: {len(tile_prompts_labels)} chars",
            "=" * 50,
        ])
        debug_info = "\n".join(debug_lines)

        print(f"[Smart Tile Prompter Turbo v3.9] Done in {elapsed:.1f}s (Render-to-Real)")

        # Create output bundle (pass through input bundle data + add prompts)
        output_bundle = {}
        if bundle is not None:
            output_bundle.update(bundle)  # Copy all data from input bundle
        # Add/update prompt data
        output_bundle["tile_prompts_labels"] = tile_prompts_labels
        output_bundle["tile_prompts_list"] = tile_prompts_list  # Full list with positions
        output_bundle["scene_description"] = scene_description  # Short scene
        output_bundle["materials"] = materials  # Material consistency (v2.1)
        output_bundle["tile_labels"] = tile_labels  # Grid labels (v2.1)
        output_bundle["tiles_x"] = tiles_x
        output_bundle["tiles_y"] = tiles_y
        output_bundle["total_tiles"] = total_tiles

        # Cache result - global_context is now scene_description (short, 1-2 sentences)
        result = (scene_description, tile_prompts_combined, tile_prompts_json,
                  tile_prompts_labels, tiles_x, tiles_y, debug_info, output_bundle)
        if use_cache and cache_key:
            _cache_set(cache_key, result)
            print(f"[Smart Tile Prompter Turbo] Result cached ({len(TILE_PROMPTER_TURBO_CACHE)}/{TILE_PROMPTER_TURBO_CACHE_MAX_SIZE} entries)")

        # Stop server if requested (frees VRAM for diffusion model)
        if stop_server_after:
            print(f"[Smart Tile Prompter Turbo] Stopping server to free VRAM...")
            stop_success = stop_server(model_size)
            if not stop_success:
                print(f"[Smart Tile Prompter Turbo] WARNING: Server stop failed! VRAM may not be freed.")
                print(f"  Try manually: pkill -f 'llama-server'")

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
