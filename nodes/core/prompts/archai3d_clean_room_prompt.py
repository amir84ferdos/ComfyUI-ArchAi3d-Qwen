# ArchAi3D Clean Room Prompt Builder — Generate optimized prompts for room cleaning and redesign
#
# OVERVIEW:
# Visual prompt builder for empty room creation and interior redesign workflows.
# Combines object removal with material specification using tested prompt patterns.
# Output connects directly to the prompt input of ArchAi3D_Qwen_Encoder nodes.
#
# FEATURES:
# - 3 workflow modes: Remove Only, Remove + Paint All, Remove + Paint Selective
# - Material preset dropdowns for floor, walls, ceiling (loaded from YAML config)
# - User-customizable material library via config/materials.yaml
# - Custom material text override option
# - Quality control toggles (preserve lighting, perspective, clean edges)
# - Dual output: user_prompt + system_prompt
# - Generates structured, optimized prompts based on proven patterns
#
# Author: Amir Ferdos (ArchAi3d)
# Email: Amir84ferdos@gmail.com
# LinkedIn: https://www.linkedin.com/in/archai3d/
# GitHub: https://github.com/amir84ferdos
# Category: ArchAi3d/Qwen
# Node ID: ArchAi3D_Clean_Room_Prompt
# License: MIT

import yaml
import os

# ============================================================
# Load Material Library from YAML Configuration
# ============================================================

def load_materials_config():
    """Load materials from config/materials.yaml file.

    Users can edit this file to customize materials, add new ones,
    or translate to other languages.

    Returns:
        dict: Parsed YAML configuration with floors, walls, ceilings, and style_filters
    """
    # Navigate from nodes/core/prompts/ up to the root custom_nodes directory
    current_dir = os.path.dirname(__file__)  # nodes/core/prompts/
    root_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))  # Go up 3 levels
    config_path = os.path.join(root_dir, "config", "materials.yaml")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            print(f"[ArchAi3d-Qwen] Loaded {len(config.get('floors', []))} floor materials, "
                  f"{len(config.get('walls', []))} wall materials, "
                  f"{len(config.get('ceilings', []))} ceiling materials from YAML config")
            return config
    except FileNotFoundError:
        print(f"[ArchAi3d-Qwen] ERROR: Material config file not found: {config_path}")
        print("[ArchAi3d-Qwen] Using minimal fallback config...")
        # Return minimal default config if file doesn't exist
        return {
            'floors': [
                {'name': 'Keep Original', 'description': '', 'tags': ['all']},
                {'name': 'Custom', 'description': 'CUSTOM', 'tags': ['all']}
            ],
            'walls': [
                {'name': 'Keep Original', 'description': '', 'tags': ['all']},
                {'name': 'Custom', 'description': 'CUSTOM', 'tags': ['all']}
            ],
            'ceilings': [
                {'name': 'Keep Original', 'description': '', 'tags': ['all']},
                {'name': 'Custom', 'description': 'CUSTOM', 'tags': ['all']}
            ],
            'style_filters': [
                {'label': 'All Materials', 'description': 'Show all materials', 'tags': ['all']}
            ]
        }
    except Exception as e:
        print(f"[ArchAi3d-Qwen] ERROR loading materials config: {e}")
        print("[ArchAi3d-Qwen] Using minimal fallback config...")
        return {
            'floors': [{'name': 'Keep Original', 'description': '', 'tags': ['all']}, {'name': 'Custom', 'description': 'CUSTOM', 'tags': ['all']}],
            'walls': [{'name': 'Keep Original', 'description': '', 'tags': ['all']}, {'name': 'Custom', 'description': 'CUSTOM', 'tags': ['all']}],
            'ceilings': [{'name': 'Keep Original', 'description': '', 'tags': ['all']}, {'name': 'Custom', 'description': 'CUSTOM', 'tags': ['all']}],
            'style_filters': [{'label': 'All Materials', 'description': 'Show all materials', 'tags': ['all']}]
        }

# Load materials configuration once at module load
MATERIALS_CONFIG = load_materials_config()

# Build material dictionaries from YAML config
FLOOR_MATERIALS = {m['name']: m['description'] for m in MATERIALS_CONFIG.get('floors', [])}
WALL_MATERIALS = {m['name']: m['description'] for m in MATERIALS_CONFIG.get('walls', [])}
CEILING_MATERIALS = {m['name']: m['description'] for m in MATERIALS_CONFIG.get('ceilings', [])}

# Build photography style and lighting preset dictionaries
PHOTOGRAPHY_STYLES = {p['name']: p['description'] for p in MATERIALS_CONFIG.get('photography_styles', [])}
LIGHTING_PRESETS = {l['name']: l['description'] for l in MATERIALS_CONFIG.get('lighting_presets', [])}

# Common construction objects to remove (optimized keywords for AI image models)
DEFAULT_REMOVE_LIST = "scaffolding/metal frames/construction debris/tools/toolboxes/power tools/buckets/pails/plastic bags/cement bags/plaster bags/insulation/foam sheets/cables/wires/scraps/wrapping materials"

# Workflow mode templates
WORKFLOW_MODES = [
    "Remove Only",
    "Remove + Paint All",
    "Remove + Paint Selective"
]

# System prompt presets for room transformation workflows
SYSTEM_PROMPTS_ROOM_TRANSFORM = {
    "None": "",
    "Room Transform Specialist": "You are a room transformation specialist. Your task: 1) Remove ALL specified objects completely (tools, debris, furniture, materials, watermarks, text overlays, logos) - leave no trace. Intelligently inpaint removed areas to blend seamlessly. 2) Apply specified surface materials (floor/walls/ceiling) with photorealistic detail, proper lighting, and realistic reflections. 3) PRESERVE (CRITICAL): architectural structure, windows (maintain window positions, size, and natural light), doors, lighting conditions, camera perspective, and POV. 4) ENSURE: clean edges, no halos, seamless material transitions, consistent lighting. 5) Generate photorealistic results that look like professional interior photography.",
    "Object Remover & Designer": "You are an expert at cleaning and redesigning interior spaces. Follow these rules strictly: Remove specified objects entirely (no remnants, shadows, or artifacts). Apply materials exactly as described with photorealistic accuracy. Never alter: camera angle, perspective, lighting direction, architectural elements. Maintain: natural shadows, reflections, material textures, depth perception. Deliver: clean professional result with sharp edges and seamless integration.",
    "Photorealistic Room Editor": "You are a photorealistic image editor specializing in interior spaces. Execute transformations following these principles: REMOVE: All specified objects, debris, and materials - complete erasure with proper background reconstruction. APPLY: Surface materials with accurate physical properties (reflections, texture, color accuracy). PRESERVE: Original perspective, lighting setup, architectural features, spatial relationships. QUALITY: Photorealistic rendering, clean edges, no visual artifacts, seamless material boundaries. OUTPUT: Professional interior photography standard with natural lighting and realistic materials.",
    "Interior Designer": "You are an expert interior designer. Analyze spaces and create detailed, photorealistic design transformations while preserving architectural structure, lighting, and perspective.",
    "Renovation Expert": "You are a home renovation expert. Suggest practical, budget-conscious improvements that enhance functionality and aesthetics while respecting existing structure.",
    "Custom": "CUSTOM",
}


def build_watermark_removal_phrase(watermark_type, location):
    """Build watermark removal phrase following Qwen WanX API patterns.

    Based on research from existing Watermark Removal node and official Qwen documentation.
    Uses proven "Remove [TYPE] from [LOCATION]" pattern.

    Args:
        watermark_type: Type of element to remove (watermark, logo, text, etc.)
        location: Where to remove from (anywhere, bottom right, etc.)

    Returns:
        str: Removal phrase (e.g., "the watermark from the bottom right corner")

    Examples:
        >>> build_watermark_removal_phrase("watermark", "bottom right")
        'the watermark from the bottom right corner'
        >>> build_watermark_removal_phrase("logo", "anywhere")
        'the logo'
    """
    # Map watermark types to phrases
    type_map = {
        "watermark": "the watermark",
        "logo": "the logo",
        "text": "the text",
        "English text": "the English text",
        "Chinese text": "the Chinese text"
    }

    # Map locations to phrases
    location_map = {
        "anywhere": "",
        "bottom right": "from the bottom right corner",
        "bottom left": "from the bottom left corner",
        "top right": "from the top right corner",
        "top left": "from the top left corner",
        "center": "from the center"
    }

    phrase = type_map.get(watermark_type, "the watermark")
    location_phrase = location_map.get(location, "")

    if location_phrase:
        return f"{phrase} {location_phrase}"
    return phrase


class ArchAi3D_Clean_Room_Prompt:
    """Visual prompt builder for room cleaning and interior redesign workflows.

    Features:
    - 3 workflow modes (remove only, full redesign, selective redesign)
    - Scene context field for room type description (NEW v2.1.0)
    - Watermark/logo removal option (NEW v2.1.0)
    - Material preset dropdowns for floor/walls/ceiling (loaded from YAML)
    - 103+ material presets (32 floors, 36 walls, 35 ceilings)
    - 8 photography style presets (realism, sharpness, quality)
    - 15 lighting presets (daylight, golden hour, studio, etc.)
    - User-customizable material library via config/materials.yaml
    - Custom material text override
    - System prompt presets with enhanced window preservation
    - Quality control options
    - Generates optimized prompts using research-validated Qwen patterns

    Use this to quickly build consistent prompts for empty room creation
    and interior transformation workflows with comprehensive cleanup options.

    Version: 2.1.1
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (WORKFLOW_MODES, {"default": "Remove + Paint All"}),
                "image_reference": ("STRING", {"default": "image1"}),
                "objects_to_remove": ("STRING", {
                    "multiline": True,
                    "default": DEFAULT_REMOVE_LIST
                }),
            },
            "optional": {
                # Scene context (NEW in v2.1.0)
                "scene_context": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Optional: Describe the room/space context (e.g., 'modern office with large windows'). Helps preserve windows, architectural features, and overall character."
                }),

                # Watermark removal options (NEW in v2.1.0)
                "remove_watermark": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Add watermark/logo/text removal to the cleaning process"
                }),
                "watermark_type": (["watermark", "logo", "text", "English text", "Chinese text"], {
                    "default": "watermark",
                    "tooltip": "Type of element to remove"
                }),
                "watermark_location": (["anywhere", "bottom right", "bottom left", "top right", "top left", "center"], {
                    "default": "anywhere",
                    "tooltip": "Location of watermark/logo"
                }),

                # Floor options
                "floor_material": (list(FLOOR_MATERIALS.keys()), {"default": "Polished Black Marble"}),
                "floor_custom": ("STRING", {"multiline": False, "default": ""}),

                # Wall options
                "wall_material": (list(WALL_MATERIALS.keys()), {"default": "Flat White"}),
                "wall_custom": ("STRING", {"multiline": False, "default": ""}),

                # Ceiling options
                "ceiling_material": (list(CEILING_MATERIALS.keys()), {"default": "Flat White"}),
                "ceiling_custom": ("STRING", {"multiline": False, "default": ""}),

                # Photography & Lighting
                "photography_style": (list(PHOTOGRAPHY_STYLES.keys()), {"default": "None"}),
                "lighting_preset": (list(LIGHTING_PRESETS.keys()), {"default": "Keep Original"}),

                # Quality controls
                "preserve_lighting": ("BOOLEAN", {"default": True}),
                "preserve_perspective": ("BOOLEAN", {"default": True}),
                "preserve_pov": ("BOOLEAN", {"default": True}),
                "clean_edges": ("BOOLEAN", {"default": True}),
                "no_halos": ("BOOLEAN", {"default": True}),

                # System prompt options
                "system_preset": (list(SYSTEM_PROMPTS_ROOM_TRANSFORM.keys()), {"default": "Room Transform Specialist"}),
                "custom_system_prompt": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("user_prompt", "system_prompt")
    FUNCTION = "build_prompt"
    CATEGORY = "ArchAi3d/Qwen"

    def build_prompt(self, mode, image_reference, objects_to_remove,
                     scene_context="",  # NEW in v2.1.0
                     remove_watermark=False,  # NEW in v2.1.0
                     watermark_type="watermark",  # NEW in v2.1.0
                     watermark_location="anywhere",  # NEW in v2.1.0
                     floor_material="Polished Black Marble", floor_custom="",
                     wall_material="Flat White", wall_custom="",
                     ceiling_material="Flat White", ceiling_custom="",
                     photography_style="None", lighting_preset="Keep Original",
                     preserve_lighting=True, preserve_perspective=True,
                     preserve_pov=True, clean_edges=True, no_halos=True,
                     system_preset="Room Transform Specialist", custom_system_prompt=""):
        """Build optimized room transformation prompt.

        Args:
            mode: Workflow mode (Remove Only, Remove + Paint All, Remove + Paint Selective)
            image_reference: Image identifier (e.g., "image1")
            objects_to_remove: Semicolon/slash-separated list of objects to remove
            scene_context: Optional scene description (e.g., "modern office with large windows") [v2.1.0]
            remove_watermark: Enable watermark/logo removal [v2.1.0]
            watermark_type: Type of element to remove (watermark, logo, text, etc.) [v2.1.0]
            watermark_location: Location of watermark (anywhere, bottom right, etc.) [v2.1.0]
            floor_material: Preset or "Custom" or "Keep Original"
            floor_custom: Custom floor material description (if Custom selected)
            wall_material: Preset or "Custom" or "Keep Original"
            wall_custom: Custom wall material description
            ceiling_material: Preset or "Custom" or "Keep Original"
            ceiling_custom: Custom ceiling material description
            photography_style: Photography quality preset (None, Architectural, Real Estate, etc.)
            lighting_preset: Lighting condition preset (Keep Original, Natural Daylight, Golden Hour, etc.)
            preserve_lighting: Keep original lighting
            preserve_perspective: Keep original perspective
            preserve_pov: Keep original point of view/crop
            clean_edges: Ensure clean edges in output
            no_halos: Prevent halos around objects
            system_preset: System prompt preset selection
            custom_system_prompt: Custom system prompt text (if Custom selected)

        Returns:
            Tuple of (user_prompt, system_prompt)
        """

        # NEW v2.1.0: Build scene description with context (Qwen best practice: context first)
        scene_parts = []
        if scene_context.strip():
            scene_parts.append(scene_context.strip())

        # Add transformation goal
        if mode == "Remove Only":
            scene_parts.append("clean empty room")
        else:
            scene_parts.append("clean finished interior")

        base_prompt = f"Transform {image_reference}: {', '.join(scene_parts)}."

        # NEW v2.1.0: Enhanced removal instruction with optional watermark removal
        removal_items = [objects_to_remove]

        if remove_watermark:
            watermark_phrase = build_watermark_removal_phrase(watermark_type, watermark_location)
            removal_items.append(watermark_phrase)

        remove_instruction = f" Remove {'/'.join(removal_items)}."

        # Build surface specifications
        surface_specs = []

        # Helper function to get material description
        def get_material(preset_dict, selected, custom_text):
            if selected == "Keep Original":
                return None
            elif selected == "Custom":
                return custom_text if custom_text.strip() else None
            else:
                return preset_dict.get(selected, "")

        # Process floor
        floor_desc = get_material(FLOOR_MATERIALS, floor_material, floor_custom)
        if mode != "Remove Only":
            # For "Remove + Paint All": include all surfaces (even if Keep Original)
            # For "Remove + Paint Selective": only include if explicitly changed (not Keep Original)
            if mode == "Remove + Paint All":
                if floor_desc:
                    surface_specs.append(f"Floor → {floor_desc}")
            elif mode == "Remove + Paint Selective":
                # Only add if user explicitly selected a material (not "Keep Original")
                if floor_material != "Keep Original" and floor_desc:
                    surface_specs.append(f"Floor → {floor_desc}")

        # Process walls
        wall_desc = get_material(WALL_MATERIALS, wall_material, wall_custom)
        if mode != "Remove Only":
            if mode == "Remove + Paint All":
                if wall_desc:
                    surface_specs.append(f"Walls → {wall_desc}")
            elif mode == "Remove + Paint Selective":
                if wall_material != "Keep Original" and wall_desc:
                    surface_specs.append(f"Walls → {wall_desc}")

        # Process ceiling
        ceiling_desc = get_material(CEILING_MATERIALS, ceiling_material, ceiling_custom)
        if mode != "Remove Only":
            if mode == "Remove + Paint All":
                if ceiling_desc:
                    surface_specs.append(f"Ceiling → {ceiling_desc}")
            elif mode == "Remove + Paint Selective":
                if ceiling_material != "Keep Original" and ceiling_desc:
                    surface_specs.append(f"Ceiling → {ceiling_desc}")

        # Combine surface specs
        surface_text = ""
        if surface_specs:
            surface_text = "\n" + ". ".join(surface_specs) + "."

        # Build preservation/quality constraints
        constraints = []

        # NEW v2.1.0: Add window preservation if mentioned in scene context
        window_mentioned = scene_context.strip() and "window" in scene_context.lower()

        if preserve_lighting and preserve_perspective:
            if window_mentioned:
                constraints.append("Preserve windows and natural light; preserve lighting/perspective")
            else:
                constraints.append("Preserve lighting/perspective")
        elif preserve_lighting:
            if window_mentioned:
                constraints.append("Preserve windows and natural light; preserve lighting")
            else:
                constraints.append("Preserve lighting")
        elif preserve_perspective:
            if window_mentioned:
                constraints.append("Preserve windows; preserve perspective")
            else:
                constraints.append("Preserve perspective")
        elif window_mentioned:
            # Window mentioned but no other preservation - add window-only clause
            constraints.append("Preserve windows and natural light")

        if preserve_pov:
            if len(constraints) == 0:
                constraints.append("Keep POV/crop")
            else:
                constraints[0] += "; keep POV/crop"

        if clean_edges:
            constraints.append("Clean edges")

        if no_halos:
            if clean_edges:
                constraints[-1] += "; no halos"
            else:
                constraints.append("No halos")

        constraints_text = ""
        if constraints:
            constraints_text = "\n" + ". ".join(constraints) + "."

        # Add photography style and lighting
        style_text = ""
        photo_desc = PHOTOGRAPHY_STYLES.get(photography_style, "")
        lighting_desc = LIGHTING_PRESETS.get(lighting_preset, "")

        style_components = []
        if photo_desc:
            style_components.append(photo_desc)
        if lighting_desc:
            style_components.append(lighting_desc)

        if style_components:
            style_text = "\n" + ". ".join(style_components) + "."

        # Combine all parts
        final_prompt = base_prompt + remove_instruction + surface_text + constraints_text + style_text

        # Build system prompt
        system_prompt_text = ""
        if system_preset == "Custom":
            system_prompt_text = custom_system_prompt
        elif system_preset != "None":
            system_prompt_text = SYSTEM_PROMPTS_ROOM_TRANSFORM.get(system_preset, "")

        return (final_prompt, system_prompt_text)


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "ArchAi3D_Clean_Room_Prompt": ArchAi3D_Clean_Room_Prompt
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_Clean_Room_Prompt": "ArchAi3D Clean Room Prompt"
}
