# ArchAi3D_Qwen_Object_Rotation_V2 — Enhanced object rotation control for Qwen Edit 2509
#
# NEW FEATURES IN V2:
# - Cinematography presets (Product Turntable, Architectural Walkaround, etc.)
# - Orbit distance control (close/medium/wide)
# - Subject type presets with optimizations
# - Extended angle presets (30°, 60°, 120°, 135°, 270°)
# - Speed/transition hints (smooth, slow, fast, cinematic)
# - Elevation during orbit (rising, descending, eye-to-bird)
# - Quick presets library for one-click professional results
#
# ORIGINAL FEATURES:
# - "Orbit around" technique (most reliable method)
# - Precise angle control
# - Multi-step mode for video sequences
# - Subject-aware rotation control
#
# Author: Amir Ferdos (ArchAi3d)
# Email: Amir84ferdos@gmail.com
# LinkedIn: https://www.linkedin.com/in/archai3d/
# GitHub: https://github.com/amir84ferdos
# Category: ArchAi3d/Qwen
# Node ID: ArchAi3D_Qwen_Object_Rotation_V2
# License: MIT

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override
from typing import Dict, List, Tuple


# ============================================================================
# SYSTEM PROMPTS FOR CINEMATOGRAPHY PRESETS (v4.0.0)
# ============================================================================

ROTATION_SYSTEM_PROMPTS = {
    "custom": "You are a camera operator. Orbit camera around subject smoothly. Maintain object centered in frame. Preserve scene lighting and details.",

    # E-COMMERCE & PRODUCTS
    "product_turntable": "You are a product photographer. Execute smooth 360-degree turntable rotation. Keep product perfectly centered, maintain consistent lighting. Professional e-commerce quality. Preserve all product details and textures.",

    "product_four_views": "You are a product photographer. Rotate product in 90-degree increments showing all four sides. Maintain professional framing and lighting. Preserve product details, ensure consistent perspective each view.",

    "inspection_view": "You are a product quality inspector. Detailed 360-degree examination rotation. Reveal all angles for thorough inspection. Maintain consistent distance and lighting. Preserve fine details and surface features.",

    "detail_close_inspection": "You are a product photographer. Close-up 180-degree detail rotation. Emphasize textures, materials, and craftsmanship. Maintain focus on details. Preserve surface qualities and fine features.",

    "three_angle_showcase": "You are a product photographer. Show three key angles (120-degree increments). Each angle presents distinct viewpoint. Professional showcase quality. Preserve product features from each perspective.",

    # ARCHITECTURE & REAL ESTATE
    "architectural_walkaround": "You are an architectural photographer. Cinematic 360-degree building exterior rotation. Maintain wide establishing perspective. Show building in context of surroundings. Preserve architectural details, proportions, and site relationship.",

    "interior_walkthrough": "You are a real estate photographer. Smooth 180-degree room exploration. Natural walking pace, reveal space gradually. Show room layout and flow. Preserve interior details, lighting, and spatial relationships.",

    "room_corner_90": "You are an interior photographer. 90-degree corner sweep showing two walls meeting. Reveal room depth and dimensions. Natural perspective from doorway. Preserve room proportions and interior design elements.",

    "drone_ascending_orbit": "You are a drone operator. Orbit building while ascending upward. Create dramatic rising perspective. Smooth spiral motion revealing scale. Preserve building details and site context throughout ascent.",

    # CINEMATIC & PROFESSIONAL
    "hero_shot": "You are a cinematic camera operator. Execute dramatic 180-degree hero reveal. Wide cinematic framing, professional quality. Build anticipation through camera movement. Preserve subject presence and cinematic mood.",

    "reveal_shot": "You are a cinematic camera operator. Slow dramatic 180-degree reveal rotation. Build tension and interest through controlled movement. Cinematic pacing and framing. Preserve dramatic lighting and subject details.",

    "slow_cinema_orbit": "You are a film camera operator. Ultra-smooth 360-degree film-quality rotation. Professional cinema standards, smooth as silk. Perfect for high-end productions. Preserve all details with cinematic precision.",

    "dramatic_reveal": "You are a cinematic camera operator. Slow 90-degree quarter-turn dramatic reveal. Single cinematic frame transition. Build anticipation. Preserve dramatic impact and subject presentation.",

    # QUICK & EFFICIENT
    "quick_peek_45": "You are a camera operator. Subtle 45-degree angle adjustment. Quick peek around subject for different perspective. Smooth transition. Preserve subject details and lighting.",

    "opposite_view_180": "You are a camera operator. Direct 180-degree rotation to opposite side. Efficient transition to back view. Maintain framing consistency. Preserve all subject details from new angle.",

    "quick_spin": "You are a camera operator. Fast dynamic 360-degree spin. Energetic movement, lively presentation. Professional control despite speed. Preserve subject clarity throughout rapid rotation.",

    # SPECIAL EFFECTS
    "spiral_ascent": "You are a camera operator. Orbit subject while simultaneously ascending. Create spiral rising effect. Smooth combined rotation and elevation. Preserve subject details throughout complex movement.",

    "spiral_descent": "You are a camera operator. Orbit subject while simultaneously descending. Create spiral lowering effect. Smooth combined rotation and descent. Preserve subject details throughout complex movement.",

    "social_media_spin": "You are a social media content creator. 360-degree rotation optimized for short-form video. Engaging pace, attention-grabbing movement. Platform-ready quality. Preserve subject appeal and visual interest.",
}

# ============================================================================
# CONFIGURATION CONSTANTS - Easy to modify!
# ============================================================================

# Text mappings for orbit distance
ORBIT_DISTANCE_MAP = {
    "close": "close",
    "medium": "",  # Medium doesn't add a modifier word
    "wide": "wide"
}

# Text mappings for elevation (full prompt version)
ELEVATION_MAP_FULL = {
    "rising": "while rising upward",
    "descending": "while descending",
    "eye_to_bird": "from eye level to bird's eye view",
    "bird_to_eye": "from bird's eye view to eye level"
}

# Text mappings for elevation (multi-step version - shorter)
ELEVATION_MAP_STEP = {
    "rising": "while rising slightly",
    "descending": "while descending slightly",
    "eye_to_bird": "gradually rising",
    "bird_to_eye": "gradually descending"
}

# Text mappings for speed hints (full prompt version)
SPEED_MAP_FULL = {
    "smooth": "smooth camera movement",
    "slow": "slow steady movement",
    "fast": "quick movement",
    "cinematic": "cinematic slow movement"
}

# Text mappings for speed hints (multi-step version - shorter)
SPEED_MAP_STEP = {
    "smooth": "smooth",
    "slow": "slow steady",
    "fast": "quick",
    "cinematic": "cinematic"
}


# ============================================================================
# PRESET DEFINITIONS
# ============================================================================

# Subject type presets
SUBJECT_PRESETS = {
    "custom": {
        "text": "",
        "orbit_hint": ""
    },
    "generic": {
        "text": "SUBJECT",
        "orbit_hint": ""
    },
    "product": {
        "text": "the product",
        "orbit_hint": "showcasing all angles"
    },
    "building": {
        "text": "the building",
        "orbit_hint": "exterior view"
    },
    "furniture": {
        "text": "the furniture",
        "orbit_hint": "showing details"
    },
    "vehicle": {
        "text": "the vehicle",
        "orbit_hint": "displaying all sides"
    },
    "character": {
        "text": "the character",
        "orbit_hint": "centered in frame"
    },
    "room": {
        "text": "the room",
        "orbit_hint": "interior view"
    },
}

# Default settings for all cinematography presets
PRESET_DEFAULTS = {
    "rotation_axis": "horizontal",
    "rotation_style": "orbit_around",
    "maintain_distance": True,
    "keep_level": True,
    "auto_config": True
}

# Cinematography preset configurations
CINEMATOGRAPHY_PRESETS = {
    "custom": {
        "description": "Manual configuration - use sliders below",
        "auto_config": False
    },

    # === E-COMMERCE & PRODUCTS ===
    "product_turntable": {
        "description": "360° smooth rotation for e-commerce (8 frames)",
        "angle": 360,
        "steps": 8,
        "multi_step": True,
        "direction": "right",
        "orbit_distance": "close",
        "speed_hint": "smooth",
        "elevation": "level",
        **PRESET_DEFAULTS
    },
    "product_four_views": {
        "description": "360° rotation in 4 steps (90° increments, 4 frames)",
        "angle": 360,
        "steps": 4,
        "multi_step": True,
        "direction": "right",
        "orbit_distance": "close",
        "speed_hint": "smooth",
        "elevation": "level",
        **PRESET_DEFAULTS
    },
    "inspection_view": {
        "description": "Multiple angles for detailed examination (12 frames)",
        "angle": 360,
        "steps": 12,
        "multi_step": True,
        "direction": "right",
        "orbit_distance": "close",
        "speed_hint": "smooth",
        "elevation": "level",
        **PRESET_DEFAULTS
    },
    "detail_close_inspection": {
        "description": "Close-up 180° rotation for detail inspection (6 frames)",
        "angle": 180,
        "steps": 6,
        "multi_step": True,
        "direction": "right",
        "orbit_distance": "close",
        "speed_hint": "slow",
        "elevation": "level",
        **PRESET_DEFAULTS
    },
    "three_angle_showcase": {
        "description": "360° rotation in 3 steps (120° increments, 3 frames)",
        "angle": 360,
        "steps": 3,
        "multi_step": True,
        "direction": "right",
        "orbit_distance": "medium",
        "speed_hint": "smooth",
        "elevation": "level",
        **PRESET_DEFAULTS
    },

    # === ARCHITECTURE & REAL ESTATE ===
    "architectural_walkaround": {
        "description": "360° exterior building rotation (8 frames)",
        "angle": 360,
        "steps": 8,
        "multi_step": True,
        "direction": "right",
        "orbit_distance": "wide",
        "speed_hint": "cinematic",
        "elevation": "level",
        **PRESET_DEFAULTS
    },
    "interior_walkthrough": {
        "description": "180° room exploration (4 frames)",
        "angle": 180,
        "steps": 4,
        "multi_step": True,
        "direction": "right",
        "orbit_distance": "medium",
        "speed_hint": "smooth",
        "elevation": "level",
        "rotation_axis": "horizontal",
        "rotation_style": "orbit_around",
        "maintain_distance": False,  # Different from default
        "keep_level": True,
        "auto_config": True
    },
    "room_corner_90": {
        "description": "Interior corner sweep 90° (2 frames)",
        "angle": 90,
        "steps": 2,
        "multi_step": True,
        "direction": "right",
        "orbit_distance": "medium",
        "speed_hint": "smooth",
        "elevation": "level",
        "rotation_axis": "horizontal",
        "rotation_style": "orbit_around",
        "maintain_distance": False,  # Different from default
        "keep_level": True,
        "auto_config": True
    },
    "drone_ascending_orbit": {
        "description": "270° orbit while rising (drone-style, 9 frames)",
        "angle": 270,
        "steps": 9,
        "multi_step": True,
        "direction": "right",
        "orbit_distance": "wide",
        "speed_hint": "cinematic",
        "elevation": "rising",
        "rotation_axis": "horizontal",
        "rotation_style": "orbit_around",
        "maintain_distance": True,
        "keep_level": False,  # Different from default
        "auto_config": True
    },

    # === CINEMATIC & PROFESSIONAL ===
    "hero_shot": {
        "description": "180° cinematic wide shot (single frame)",
        "angle": 180,
        "steps": 1,
        "multi_step": False,
        "direction": "right",
        "orbit_distance": "wide",
        "speed_hint": "cinematic",
        "elevation": "level",
        **PRESET_DEFAULTS
    },
    "reveal_shot": {
        "description": "180° dramatic reveal (4 frames)",
        "angle": 180,
        "steps": 4,
        "multi_step": True,
        "direction": "right",
        "orbit_distance": "medium",
        "speed_hint": "slow",
        "elevation": "level",
        **PRESET_DEFAULTS
    },
    "slow_cinema_orbit": {
        "description": "Ultra-smooth 360° film-quality rotation (24 frames)",
        "angle": 360,
        "steps": 24,
        "multi_step": True,
        "direction": "right",
        "orbit_distance": "medium",
        "speed_hint": "cinematic",
        "elevation": "level",
        **PRESET_DEFAULTS
    },
    "dramatic_reveal": {
        "description": "90° slow quarter turn reveal (single cinematic frame)",
        "angle": 90,
        "steps": 1,
        "multi_step": False,
        "direction": "right",
        "orbit_distance": "medium",
        "speed_hint": "cinematic",
        "elevation": "level",
        **PRESET_DEFAULTS
    },

    # === QUICK & EFFICIENT ===
    "quick_peek_45": {
        "description": "Subtle 45° angle change (single frame)",
        "angle": 45,
        "steps": 1,
        "multi_step": False,
        "direction": "right",
        "orbit_distance": "medium",
        "speed_hint": "smooth",
        "elevation": "level",
        **PRESET_DEFAULTS
    },
    "opposite_view_180": {
        "description": "180° rotation to opposite side (single frame)",
        "angle": 180,
        "steps": 1,
        "multi_step": False,
        "direction": "right",
        "orbit_distance": "medium",
        "speed_hint": "none",
        "elevation": "level",
        **PRESET_DEFAULTS
    },
    "quick_spin": {
        "description": "Fast 360° for dynamic effect (16 frames)",
        "angle": 360,
        "steps": 16,
        "multi_step": True,
        "direction": "right",
        "orbit_distance": "medium",
        "speed_hint": "fast",
        "elevation": "level",
        **PRESET_DEFAULTS
    },

    # === SPECIAL EFFECTS ===
    "spiral_ascent": {
        "description": "Orbit while rising upward (8 frames)",
        "angle": 360,
        "steps": 8,
        "multi_step": True,
        "direction": "right",
        "orbit_distance": "medium",
        "speed_hint": "smooth",
        "elevation": "rising",
        "rotation_axis": "horizontal",
        "rotation_style": "orbit_around",
        "maintain_distance": True,
        "keep_level": False,  # Different from default
        "auto_config": True
    },
    "spiral_descent": {
        "description": "Orbit while descending (8 frames)",
        "angle": 360,
        "steps": 8,
        "multi_step": True,
        "direction": "right",
        "orbit_distance": "medium",
        "speed_hint": "smooth",
        "elevation": "descending",
        "rotation_axis": "horizontal",
        "rotation_style": "orbit_around",
        "maintain_distance": True,
        "keep_level": False,  # Different from default
        "auto_config": True
    },
    "social_media_spin": {
        "description": "360° optimized for social media (15 frames, ~7.5sec @2fps)",
        "angle": 360,
        "steps": 15,
        "multi_step": True,
        "direction": "right",
        "orbit_distance": "medium",
        "speed_hint": "fast",
        "elevation": "level",
        **PRESET_DEFAULTS
    },
}


# ============================================================================
# HELPER FUNCTIONS - Pure logic, easy to test and modify
# ============================================================================

def apply_preset_settings(preset_name: str, user_params: Dict) -> Dict:
    """
    Apply cinematography preset settings to user parameters.

    Args:
        preset_name: Name of the preset to apply
        user_params: Dictionary of user-provided parameters

    Returns:
        Dictionary with preset values overriding user values (or original values if custom)
    """
    if preset_name == "custom" or not CINEMATOGRAPHY_PRESETS[preset_name].get("auto_config", False):
        # Custom mode - use all user parameters as-is
        return {
            "angle": int(user_params["angle_preset"]) if user_params["angle_preset"] != "custom" else user_params["custom_angle"],
            "steps_to_use": user_params["steps"],
            "multi_step_mode": user_params["multi_step_mode"],
            "rotation_axis": user_params["rotation_axis"],
            "direction": user_params["direction"],
            "rotation_style": user_params["rotation_style"],
            "orbit_distance": user_params["orbit_distance"],
            "speed_hint": user_params["speed_hint"],
            "elevation": user_params["elevation"],
            "maintain_distance": user_params["maintain_distance"],
            "keep_level": user_params["keep_level"],
        }

    # Preset mode - override with preset values
    preset = CINEMATOGRAPHY_PRESETS[preset_name]
    return {
        "angle": preset["angle"],
        "steps_to_use": preset["steps"],
        "multi_step_mode": preset["multi_step"],
        "rotation_axis": preset.get("rotation_axis", user_params["rotation_axis"]),
        "direction": preset["direction"],
        "rotation_style": preset.get("rotation_style", user_params["rotation_style"]),
        "orbit_distance": preset["orbit_distance"],
        "speed_hint": preset["speed_hint"],
        "elevation": preset["elevation"],
        "maintain_distance": preset["maintain_distance"],
        "keep_level": preset["keep_level"],
    }


def get_subject_info(subject_type: str, custom_subject: str) -> Tuple[str, str]:
    """
    Get subject text and hint based on subject type.

    Args:
        subject_type: Type of subject (product, building, etc.)
        custom_subject: Custom subject text if type is "custom"

    Returns:
        Tuple of (subject_text, subject_hint)
    """
    if subject_type == "custom":
        return (custom_subject, "")

    subject_data = SUBJECT_PRESETS[subject_type]
    subject_text = subject_data["text"] if subject_data["text"] else custom_subject
    subject_hint = subject_data["orbit_hint"]
    return (subject_text, subject_hint)


def build_rotation_base(
    add_prefix: bool,
    rotation_style: str,
    orbit_distance: str,
    direction: str,
    subject_text: str,
    angle: int
) -> str:
    """
    Build the base rotation command (before modifiers).

    Args:
        add_prefix: Whether to add "camera" prefix
        rotation_style: orbit_around, revolve_around, or rotate_camera
        orbit_distance: close, medium, or wide
        direction: left, right, up, or down
        subject_text: The subject being orbited
        angle: Rotation angle in degrees

    Returns:
        Base rotation command string
    """
    parts = []

    # Add prefix
    if add_prefix:
        parts.append("camera")

    # Get orbit distance text
    orbit_distance_text = ORBIT_DISTANCE_MAP.get(orbit_distance, "")

    # Build rotation command with style and distance
    if rotation_style == "orbit_around":
        if orbit_distance_text:
            parts.append(f"{orbit_distance_text} orbit {direction} around")
        else:
            parts.append(f"orbit {direction} around")
    elif rotation_style == "revolve_around":
        if orbit_distance_text:
            parts.append(f"{orbit_distance_text} revolve {direction} around")
        else:
            parts.append(f"revolve {direction} around")
    elif rotation_style == "rotate_camera":
        parts.append(f"rotate {direction} around")

    # Add subject and angle
    parts.append(subject_text)
    parts.append(f"by {angle} degrees")

    return " ".join(parts)


def build_modifiers(
    maintain_distance: bool,
    keep_level: bool,
    rotation_axis: str,
    elevation: str,
    subject_hint: str
) -> List[str]:
    """
    Build list of modifier phrases.

    Args:
        maintain_distance: Whether to maintain distance
        keep_level: Whether to keep camera level
        rotation_axis: horizontal or vertical
        elevation: level, rising, descending, etc.
        subject_hint: Additional subject-specific hint

    Returns:
        List of modifier strings
    """
    modifiers = []

    # Add distance/level modifiers only for horizontal, level orbits
    if maintain_distance and rotation_axis == "horizontal" and elevation == "level":
        modifiers.append("maintaining distance")
    if keep_level and rotation_axis == "horizontal" and elevation == "level":
        modifiers.append("keeping camera level")

    # Add subject hint
    if subject_hint:
        modifiers.append(subject_hint)

    return modifiers


def build_full_rotation_prompt(
    base_rotation: str,
    elevation: str,
    modifiers: List[str],
    speed_hint: str,
    is_multi_step: bool = False
) -> str:
    """
    Build complete rotation prompt with all modifiers.

    Args:
        base_rotation: Base rotation command
        elevation: Elevation setting
        modifiers: List of modifier phrases
        speed_hint: Speed hint setting
        is_multi_step: Whether this is for multi-step mode (uses shorter text)

    Returns:
        Complete rotation prompt
    """
    prompt = base_rotation

    # Add elevation
    if elevation != "level":
        elevation_map = ELEVATION_MAP_STEP if is_multi_step else ELEVATION_MAP_FULL
        elevation_text = elevation_map.get(elevation, "")
        if elevation_text:
            prompt += f" {elevation_text}"

    # Add modifiers
    if modifiers:
        prompt += " " + " ".join(modifiers)

    # Add speed hint
    if speed_hint != "none":
        speed_map = SPEED_MAP_STEP if is_multi_step else SPEED_MAP_FULL
        speed_text = speed_map.get(speed_hint, "")
        if speed_text:
            prompt += f" {speed_text}"

    return prompt


def add_scene_context(prompt: str, scene_context: str) -> str:
    """
    Add scene context to prompt.

    Args:
        prompt: The prompt to add context to
        scene_context: Scene context string

    Returns:
        Prompt with scene context prepended (if context provided)
    """
    if scene_context.strip():
        return f"{scene_context.strip()} {prompt}"
    return prompt


def generate_multi_step_prompts(
    settings: Dict,
    subject_text: str,
    subject_hint: str,
    scene_context: str
) -> Tuple[str, int]:
    """
    Generate multi-step rotation prompts.

    Args:
        settings: Dictionary of rotation settings
        subject_text: The subject being rotated
        subject_hint: Additional subject hint
        scene_context: Scene context string

    Returns:
        Tuple of (multi_step_prompts_string, frame_count)
    """
    steps_to_use = settings["steps_to_use"]
    step_angle = settings["angle"] // steps_to_use
    multi_steps = []

    for i in range(steps_to_use):
        # Build base rotation for this step
        base_rotation = build_rotation_base(
            settings["add_prefix"],
            settings["rotation_style"],
            settings["orbit_distance"],
            settings["direction"],
            subject_text,
            step_angle
        )

        # Build modifiers for this step
        modifiers = build_modifiers(
            settings["maintain_distance"],
            settings["keep_level"],
            settings["rotation_axis"],
            settings["elevation"],
            subject_hint
        )

        # Build complete prompt for this step
        step_prompt = build_full_rotation_prompt(
            base_rotation,
            settings["elevation"],
            modifiers,
            settings["speed_hint"],
            is_multi_step=True
        )

        # Add scene context
        step_prompt_with_context = add_scene_context(step_prompt, scene_context)

        # Add to list with frame number
        multi_steps.append(f"Frame {i+1}/{steps_to_use}: {step_prompt_with_context}")

    return ("\n".join(multi_steps), steps_to_use)


def print_debug_info(
    preset_name: str,
    settings: Dict,
    subject_type: str,
    subject_text: str,
    rotation_prompt: str,
    full_prompt: str,
    scene_context: str,
    multi_step_prompts: str
):
    """
    Print debug information to console.

    Args:
        preset_name: Name of cinematography preset
        settings: Dictionary of rotation settings
        subject_type: Type of subject
        subject_text: Subject text
        rotation_prompt: Base rotation prompt
        full_prompt: Full prompt with context
        scene_context: Scene context
        multi_step_prompts: Multi-step prompts (if any)
    """
    print("=" * 70)
    print("ArchAi3D_Qwen_Object_Rotation_V2 - Generated Prompt")
    print("=" * 70)

    if preset_name != "custom":
        print(f"📽️  Cinematography Preset: {preset_name}")
        print(f"   {CINEMATOGRAPHY_PRESETS[preset_name]['description']}")

    print(f"\nSubject Type: {subject_type} → '{subject_text}'")
    print(f"Orbit Distance: {settings['orbit_distance']}")
    print(f"Rotation Axis: {settings['rotation_axis']}")
    print(f"Direction: {settings['direction']}")
    print(f"Angle: {settings['angle']}°")
    print(f"Elevation: {settings['elevation']}")
    print(f"Speed Hint: {settings['speed_hint']}")
    print(f"Style: {settings['rotation_style']}")

    print(f"\n💡 TIP: 'orbit around' is the most reliable method!")
    if settings['rotation_axis'] == "vertical":
        print("⚠️  WARNING: Vertical orbits are less reliable than horizontal!")
    if subject_type == "character":
        print("⚠️  NOTE: Keep character centered for best results!")

    print(f"\nRotation Prompt:")
    print(f"  {rotation_prompt}")

    if scene_context:
        print(f"\nScene Context:")
        print(f"  {scene_context}")

    print(f"\nFull Prompt:")
    print(f"  {full_prompt}")

    if settings['multi_step_mode']:
        print(f"\nMulti-Step Mode: {settings['steps_to_use']} frames")
        print(f"{multi_step_prompts}")

    print(f"\nFrame Count: {settings['steps_to_use'] if settings['multi_step_mode'] else 1}")
    print("=" * 70)


# ============================================================================
# COMFYUI NODE CLASS
# ============================================================================

class ArchAi3D_Qwen_Object_Rotation_V2(io.ComfyNode):
    """Enhanced professional object rotation control for Qwen Edit 2509 (V2).

    NEW in V2:
    - Cinematography presets for instant professional results
    - Orbit distance control (close/medium/wide)
    - Subject type presets
    - Extended angle options
    - Speed/transition hints
    - Elevation control during orbit
    - One-click quick presets

    Perfect for product visualization, architectural walkarounds, and video sequences.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ArchAi3D_Qwen_Object_Rotation_V2",
            category="ArchAi3d/Qwen/Camera",
            inputs=[
                # === QUICK CINEMATOGRAPHY PRESETS ===
                io.Combo.Input(
                    "cinematography_preset",
                    options=list(CINEMATOGRAPHY_PRESETS.keys()),
                    default="custom",
                    tooltip="One-click professional presets (auto-configures all settings below). Select 'custom' for manual control."
                ),

                # === SUBJECT TYPE PRESET ===
                io.Combo.Input(
                    "subject_type",
                    options=list(SUBJECT_PRESETS.keys()),
                    default="generic",
                    tooltip="Subject type with optimized prompts. 'custom' uses the custom_subject field below."
                ),

                io.String.Input(
                    "custom_subject",
                    default="SUBJECT",
                    tooltip="Custom subject text (used when subject_type='custom')"
                ),

                # === ORBIT DISTANCE ===
                io.Combo.Input(
                    "orbit_distance",
                    options=["close", "medium", "wide"],
                    default="medium",
                    tooltip="How far from subject: close=detail shots, medium=standard, wide=establishing shots"
                ),

                # === ROTATION AXIS ===
                io.Combo.Input(
                    "rotation_axis",
                    options=["horizontal", "vertical"],
                    default="horizontal",
                    tooltip="Axis of rotation (horizontal = left/right orbit, most reliable)"
                ),

                # === DIRECTION ===
                io.Combo.Input(
                    "direction",
                    options=["left", "right", "up", "down"],
                    default="right",
                    tooltip="Direction to orbit (NOTE: left/right = picture left/right, not subject's perspective)"
                ),

                # === ANGLE ===
                io.Combo.Input(
                    "angle_preset",
                    options=["30", "45", "60", "90", "120", "135", "180", "270", "360", "custom"],
                    default="90",
                    tooltip="Rotation angle preset (more options in V2!)"
                ),

                io.Int.Input(
                    "custom_angle",
                    default=60,
                    min=1,
                    max=360,
                    step=1,
                    tooltip="Custom angle in degrees (used when angle_preset='custom')"
                ),

                # === ELEVATION DURING ORBIT ===
                io.Combo.Input(
                    "elevation",
                    options=["level", "rising", "descending", "eye_to_bird", "bird_to_eye"],
                    default="level",
                    tooltip="Elevation change during orbit (level=no height change, rising/descending=spiral effect)"
                ),

                # === SPEED/TRANSITION HINTS ===
                io.Combo.Input(
                    "speed_hint",
                    options=["none", "smooth", "slow", "fast", "cinematic"],
                    default="smooth",
                    tooltip="Movement quality hint for Qwen (affects video smoothness)"
                ),

                # === ROTATION STYLE ===
                io.Combo.Input(
                    "rotation_style",
                    options=["orbit_around", "revolve_around", "rotate_camera"],
                    default="orbit_around",
                    tooltip="Prompting style (orbit_around is most reliable based on community testing)"
                ),

                # === ADVANCED OPTIONS ===
                io.Boolean.Input(
                    "maintain_distance",
                    default=True,
                    tooltip="Try to maintain distance from subject during orbit"
                ),

                io.Boolean.Input(
                    "keep_level",
                    default=True,
                    tooltip="Try to keep camera level during horizontal orbit (no tilt)"
                ),

                io.String.Input(
                    "scene_context",
                    multiline=True,
                    default="",
                    tooltip="Optional: Scene context (e.g., 'in the living room', 'exterior view') - helps with consistency"
                ),

                # === MULTI-STEP ROTATION ===
                io.Boolean.Input(
                    "multi_step_mode",
                    default=False,
                    tooltip="Generate multiple rotation steps (e.g., 360° as 8x45° steps)"
                ),

                io.Int.Input(
                    "steps",
                    default=8,
                    min=2,
                    max=24,
                    step=1,
                    tooltip="Number of steps for multi-step rotation (increased max to 24 for smooth videos)"
                ),

                # === OUTPUT OPTIONS ===
                io.Boolean.Input(
                    "add_prefix",
                    default=True,
                    tooltip="Add 'camera' prefix to prompt (recommended)"
                ),

                io.Boolean.Input(
                    "debug_mode",
                    default=False,
                    tooltip="Print generated prompt to console"
                ),
            ],
            outputs=[
                io.String.Output("rotation_prompt", tooltip="Generated rotation prompt optimized for Qwen"),
                io.String.Output("full_prompt", tooltip="Rotation prompt with scene context"),
                io.String.Output("multi_step_prompts", tooltip="Multiple rotation steps (if multi_step_mode=True)"),
                io.Int.Output("frame_count", tooltip="Number of frames for multi-step mode"),
                io.String.Output("system_prompt", tooltip="⭐ NEW v4.0: Perfect system prompt for this rotation style! Connect to encoder's system_prompt input."),
            ],
        )

    @classmethod
    def execute(cls,
                cinematography_preset,
                subject_type,
                custom_subject,
                orbit_distance,
                rotation_axis,
                direction,
                angle_preset,
                custom_angle,
                elevation,
                speed_hint,
                rotation_style,
                maintain_distance,
                keep_level,
                scene_context,
                multi_step_mode,
                steps,
                add_prefix,
                debug_mode) -> io.NodeOutput:

        # ===================================================================
        # STEP 1: Apply preset settings or use manual settings
        # ===================================================================
        user_params = {
            "angle_preset": angle_preset,
            "custom_angle": custom_angle,
            "steps": steps,
            "multi_step_mode": multi_step_mode,
            "rotation_axis": rotation_axis,
            "direction": direction,
            "rotation_style": rotation_style,
            "orbit_distance": orbit_distance,
            "speed_hint": speed_hint,
            "elevation": elevation,
            "maintain_distance": maintain_distance,
            "keep_level": keep_level,
        }

        settings = apply_preset_settings(cinematography_preset, user_params)
        settings["add_prefix"] = add_prefix  # Not affected by presets

        # ===================================================================
        # STEP 2: Get subject information
        # ===================================================================
        subject_text, subject_hint = get_subject_info(subject_type, custom_subject)

        # ===================================================================
        # STEP 3: Build single rotation prompt
        # ===================================================================
        base_rotation = build_rotation_base(
            settings["add_prefix"],
            settings["rotation_style"],
            settings["orbit_distance"],
            settings["direction"],
            subject_text,
            settings["angle"]
        )

        modifiers = build_modifiers(
            settings["maintain_distance"],
            settings["keep_level"],
            settings["rotation_axis"],
            settings["elevation"],
            subject_hint
        )

        rotation_prompt = build_full_rotation_prompt(
            base_rotation,
            settings["elevation"],
            modifiers,
            settings["speed_hint"],
            is_multi_step=False
        )

        full_prompt = add_scene_context(rotation_prompt, scene_context)

        # ===================================================================
        # STEP 4: Generate multi-step prompts if requested
        # ===================================================================
        multi_step_prompts = ""
        frame_count = 1

        if settings["multi_step_mode"]:
            multi_step_prompts, frame_count = generate_multi_step_prompts(
                settings,
                subject_text,
                subject_hint,
                scene_context
            )

        # ===================================================================
        # STEP 5: Get system prompt for this preset (v4.0.0)
        # ===================================================================
        system_prompt = ROTATION_SYSTEM_PROMPTS.get(cinematography_preset, ROTATION_SYSTEM_PROMPTS["custom"])

        # ===================================================================
        # STEP 6: Debug output
        # ===================================================================
        if debug_mode:
            print_debug_info(
                cinematography_preset,
                settings,
                subject_type,
                subject_text,
                rotation_prompt,
                full_prompt,
                scene_context,
                multi_step_prompts
            )
            print(f"\n⭐ System Prompt (NEW v4.0.0):")
            print(f"  {system_prompt}")
            print("=" * 70)

        return io.NodeOutput(rotation_prompt, full_prompt, multi_step_prompts, frame_count, system_prompt)


# ============================================================================
# EXTENSION API GLUE
# ============================================================================

class ArchAi3DQwenObjectRotationV2Extension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [ArchAi3D_Qwen_Object_Rotation_V2]

async def comfy_entrypoint():
    return ArchAi3DQwenObjectRotationV2Extension()
