# -*- coding: utf-8 -*-
"""
ArchAi3D Qwen Exterior Focus Node (v1)
Author: Amir Ferdos (ArchAi3d)
Email: Amir84ferdos@gmail.com
LinkedIn: https://www.linkedin.com/in/archai3d/

Description:
    Focus camera on specific exterior elements (outdoor furniture, architectural details, features).
    Perfect for highlighting elements in outdoor scenes.

    Perfect for:
    - Outdoor furniture closeups (deck chairs, patio tables)
    - Architectural details (doors, windows, columns)
    - Garden features (fountains, sculptures)
    - Building elements (facades, balconies, signage)

Based on research:
    - Vantage point change pattern (QWEN_PROMPT_GUIDE)
    - "change the view to [HEIGHT] vantage point [DISTANCE]m to the [DIRECTION] facing [TARGET]"
    - Auto-facing mode keeps target centered
    - Distance-based positioning (meters, not degrees)
"""

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

# ============================================================================
# EXTERIOR FOCUS SYSTEM PROMPT - v5.0.0 Research-Validated
# ============================================================================

EXTERIOR_FOCUS_SYSTEM_PROMPT = """You are an architectural photographer. Position camera to focus on specific exterior element, outdoor furniture, or architectural feature. Frame subject prominently in foreground while showing environmental context. Preserve all building details, outdoor furniture, architectural proportions, outdoor lighting, sky, and environmental context exactly. Professional outdoor photography framing with proper depth and composition."""

# ============================================================================
# PROMPT STYLE PATTERNS - Dual System (User-Tested + Doc-Verified)
# ============================================================================

# User-tested pattern: "edit camera view, zoom the camera on [target], [framing]"
# Real-world test: ✅ WORKS! "edit camera view, zoom the camera on sink, closeup shot"

# Doc-verified pattern: "change the view to a vantage point [HEIGHT] [DISTANCE] to the [DIRECTION] facing [TARGET]"
# QWEN Guide Function 2: ⭐⭐⭐⭐⭐ Vantage Point (Most Reliable)

FRAMING_DESCRIPTORS = {
    "macro": "macro extreme closeup",
    "extreme": "extreme closeup shot",
    "closeup": "closeup shot",
    "detail": "detail view",
    "medium": "medium shot",
    "standard": "standard view",
    "wide": "wide shot",
    "establishing": "establishing shot",
    "flatlay": "overhead flat-lay view",
}

# ============================================================================
# FOCUS TARGET PRESETS - 18 (12 original + 6 NEW)
# ============================================================================

EXTERIOR_FOCUS_PRESETS = {
    "custom": {
        "description": "Manual control of focus target and positioning",
        "target": "",
        "direction": "front",
        "distance": 3,
        "height": "face level",
        "auto_facing": True,
    },

    # OUTDOOR FURNITURE (5) - USER'S PRIMARY USE CASE!
    "outdoor_furniture_closeup": {
        "description": "Close to outdoor furniture (deck chairs, patio tables) - 2-3m closeup",
        "target": "the outdoor furniture",
        "direction": "front",
        "distance": 2.5,
        "height": "face level",
        "auto_facing": True,
    },
    "outdoor_furniture_detail": {
        "description": "Very close to outdoor furniture - reveal material and texture details",
        "target": "the outdoor furniture",
        "direction": "front",
        "distance": 1.5,
        "height": "face level",
        "auto_facing": True,
    },
    "patio_seating": {
        "description": "Focus on patio seating area - show arrangement and context",
        "target": "the patio seating area",
        "direction": "front",
        "distance": 4,
        "height": "face level",
        "auto_facing": True,
    },
    "deck_furniture": {
        "description": "Focus on deck furniture - outdoor deck setting",
        "target": "the deck furniture",
        "direction": "front",
        "distance": 3,
        "height": "face level",
        "auto_facing": True,
    },
    "garden_furniture": {
        "description": "Focus on garden furniture - outdoor garden setting",
        "target": "the garden furniture",
        "direction": "front",
        "distance": 3,
        "height": "face level",
        "auto_facing": True,
    },

    # ARCHITECTURAL DETAILS (5)
    "entrance_door": {
        "description": "Focus on entrance door - main entry architectural feature",
        "target": "the entrance door",
        "direction": "front",
        "distance": 3,
        "height": "face level",
        "auto_facing": True,
    },
    "window_exterior": {
        "description": "Focus on exterior window - architectural window detail",
        "target": "the window",
        "direction": "front",
        "distance": 2,
        "height": "face level",
        "auto_facing": True,
    },
    "balcony_detail": {
        "description": "Focus on balcony - balcony architectural feature and railing",
        "target": "the balcony",
        "direction": "front",
        "distance": 4,
        "height": "slightly below",
        "auto_facing": True,
    },
    "facade_detail": {
        "description": "Focus on facade detail - specific building facade element or texture",
        "target": "the facade detail",
        "direction": "front",
        "distance": 2,
        "height": "face level",
        "auto_facing": True,
    },
    "column_detail": {
        "description": "Focus on column or pillar - architectural column detail",
        "target": "the column",
        "direction": "front",
        "distance": 2,
        "height": "face level",
        "auto_facing": True,
    },

    # LANDSCAPE & FEATURES (4)
    "landscape_element": {
        "description": "Focus on landscape element - trees, plants, natural features",
        "target": "the landscape element",
        "direction": "front",
        "distance": 3,
        "height": "face level",
        "auto_facing": True,
    },
    "outdoor_lighting": {
        "description": "Focus on outdoor lighting fixture - lamp posts, wall sconces",
        "target": "the outdoor lighting",
        "direction": "front",
        "distance": 2,
        "height": "face level",
        "auto_facing": True,
    },
    "signage_focus": {
        "description": "Focus on building signage or address numbers",
        "target": "the signage",
        "direction": "front",
        "distance": 2,
        "height": "face level",
        "auto_facing": True,
    },
    "garden_feature": {
        "description": "Focus on garden feature - fountain, sculpture, statue, garden art",
        "target": "the garden feature",
        "direction": "front",
        "distance": 3,
        "height": "face level",
        "auto_facing": True,
    },

    # NEW PRESETS (6) - Enhanced framing options based on research + user testing
    "macro_texture_detail": {
        "description": "⭐ NEW: Macro extreme closeup - finest material textures (<1m)",
        "target": "the architectural detail",
        "direction": "front",
        "distance": 0.5,
        "height": "face level",
        "auto_facing": True,
        "framing": "macro",
    },
    "closeup_shot": {
        "description": "⭐ NEW: Closeup shot - user-tested pattern! Tight framing (1-1.5m)",
        "target": "the exterior element",
        "direction": "front",
        "distance": 1,
        "height": "face level",
        "auto_facing": True,
        "framing": "closeup",
    },
    "detail_view": {
        "description": "⭐ NEW: Detail view - emphasize features and design (1.5-2m)",
        "target": "the building detail",
        "direction": "front",
        "distance": 1.5,
        "height": "face level",
        "auto_facing": True,
        "framing": "detail",
    },
    "wide_context": {
        "description": "⭐ NEW: Wide shot - element with full exterior context (8-10m)",
        "target": "the building feature",
        "direction": "front",
        "distance": 9,
        "height": "face level",
        "auto_facing": True,
        "framing": "wide",
    },
    "establishing_exterior": {
        "description": "⭐ NEW: Establishing shot - introduce building and surroundings (12m+)",
        "target": "the building",
        "direction": "front",
        "distance": 13,
        "height": "slightly above",
        "auto_facing": True,
        "framing": "establishing",
    },
    "overhead_aerial": {
        "description": "⭐ NEW: Overhead aerial - top-down exterior view",
        "target": "the building",
        "direction": "front",
        "distance": 5,
        "height": "high",
        "auto_facing": True,
        "framing": "flatlay",
    },
}

# ============================================================================
# DIRECTION AND HEIGHT MAPPINGS
# ============================================================================

DIRECTION_MAP = {
    "front": "front",
    "left": "left",
    "right": "right",
    "back": "back",
    "front_left": "front-left",
    "front_right": "front-right",
    "back_left": "back-left",
    "back_right": "back-right",
}

HEIGHT_MAP = {
    "ground_level": "ground level",
    "slightly_below": "slightly below face level",
    "face_level": "face level",
    "slightly_above": "slightly above face level",
    "elevated": "elevated",
    "high": "high",
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def number_to_words(num: float) -> str:
    """Convert number to written words for better AI understanding.

    Based on user feedback: "two meters" works better than "2m"

    Args:
        num: Distance in meters (0.5 to 20)

    Returns:
        Written form like "one and a half", "two", "three and a half"
    """
    # Handle half meters
    if num % 1 == 0.5:
        whole = int(num)
        if whole == 0:
            return "half a"
        else:
            return f"{number_to_words(float(whole))} and a half"

    # Whole numbers
    num_int = int(num)

    words_map = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
        11: "eleven",
        12: "twelve",
        13: "thirteen",
        14: "fourteen",
        15: "fifteen",
        16: "sixteen",
        17: "seventeen",
        18: "eighteen",
        19: "nineteen",
        20: "twenty"
    }

    return words_map.get(num_int, str(num_int))


def build_exterior_focus_prompt(
    scene_context: str,
    target: str,
    direction: str,
    distance: float,
    height: str,
    auto_facing: bool,
    prompt_style: str = "vantage_point",
    framing: str = None,
) -> str:
    """Build exterior focus prompt with dual-pattern system.

    Two supported patterns:

    1. VANTAGE POINT pattern (QWEN Guide Function 2 - ⭐⭐⭐⭐⭐):
       [SCENE_CONTEXT], change the view to a vantage point at [HEIGHT] [DISTANCE] to the [DIRECTION] facing [TARGET]

    2. ZOOM pattern (User-tested - ✅ Real-world proven):
       edit camera view, zoom the camera on [TARGET], [FRAMING_DESCRIPTOR]
       Example: "edit camera view, zoom the camera on sink, closeup shot"

    Args:
        scene_context: Description of the outdoor scene
        target: What to focus on
        direction: Direction from target
        distance: Distance from target in meters
        height: Camera height descriptor
        auto_facing: Whether to face the target
        prompt_style: "vantage_point" (doc-verified) or "zoom" (user-tested)
        framing: Framing descriptor key for zoom pattern (e.g., "closeup", "wide")

    Returns:
        Complete prompt string
    """
    parts = []

    # Build camera command based on selected pattern
    if prompt_style == "zoom":
        # User-tested pattern: "edit camera view, zoom the camera on [target], [framing]"
        camera_command = f"edit camera view, zoom the camera on {target}"

        # Add framing descriptor if available
        if framing and framing in FRAMING_DESCRIPTORS:
            camera_command += f", {FRAMING_DESCRIPTORS[framing]}"

        parts.append(camera_command)

    else:  # vantage_point (default)
        # Scene context (for vantage point pattern only)
        if scene_context.strip():
            parts.append(scene_context.strip())

        # Build vantage point change phrase
        height_phrase = HEIGHT_MAP.get(height, height)
        direction_phrase = DIRECTION_MAP.get(direction, direction)

        # Convert distance to words (user feedback: "two meters" works better than "2m")
        distance_words = number_to_words(distance)

        vantage = f"change the view to a vantage point at {height_phrase} {distance_words} meters to the {direction_phrase}"

        # Add auto-facing if enabled
        if auto_facing and target:
            vantage += f" facing {target}"

        parts.append(vantage)

    return ", ".join(parts)


# ============================================================================
# NODE DEFINITION
# ============================================================================

class ArchAi3D_Qwen_Exterior_Focus(io.ComfyNode):
    """Exterior Focus: Position camera to focus on outdoor elements.

    This node positions the camera to focus on specific exterior elements:
    - Outdoor furniture (deck chairs, patio tables, garden benches)
    - Architectural details (doors, windows, columns, facades)
    - Landscape elements (plants, trees, natural features)
    - Garden features (fountains, sculptures, art)

    Key Features:
    - 18 preset focus targets for exterior scenes (12 original + 6 NEW)
    - Research-validated vantage point change formula
    - Auto-facing mode (keeps target centered)
    - Distance control (meters, not degrees)
    - Height control (ground level to elevated)
    - Automatic system prompt output

    Perfect For:
    - Outdoor furniture closeups (USER'S PRIMARY USE CASE!)
    - Real estate detail shots (show deck furniture, patio)
    - Architectural photography (building details)
    - Garden and landscape photography

    Based on research:
    - Vantage point change formula (QWEN_PROMPT_GUIDE.md)
    - "change the view to [HEIGHT] vantage point [DISTANCE]m to the [DIRECTION] facing [TARGET]"
    - Auto-facing mode more reliable than manual rotation
    - Distance measurements (meters) for precise positioning

    USE THIS NODE WHEN:
    - You want to FOCUS on a specific outdoor element
    - Element is far away and you want closeup
    - Need to highlight outdoor furniture or architectural detail

    SCENE TYPE: Exterior/Outdoor/Architectural
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ArchAi3D_Qwen_Exterior_Focus",
            category="ArchAi3d/Camera/Exterior",
            inputs=[
                # Group 1: Preset Selection
                io.Combo.Input(
                    "preset",
                    options=list(EXTERIOR_FOCUS_PRESETS.keys()),
                    default="outdoor_furniture_closeup",
                    tooltip="Select focus target or use 'custom' for manual control. 'outdoor_furniture_closeup' = YOUR USE CASE!"
                ),

                # Group 2: Prompt Style (DUAL PATTERN SYSTEM)
                io.Combo.Input(
                    "prompt_style",
                    options=["vantage_point", "zoom"],
                    default="vantage_point",
                    tooltip="⭐ Pattern choice: 'vantage_point' (QWEN Guide ⭐⭐⭐⭐⭐ verified) OR 'zoom' (user-tested real-world pattern). Both work!"
                ),

                # Group 3: Scene Context
                io.String.Input(
                    "scene_context",
                    multiline=True,
                    default="",
                    tooltip="Optional: Outdoor scene description. Example: 'wooden deck with outdoor furniture on the right side'. Improves consistency."
                ),

                # Group 4: Focus Target (Custom Mode)
                io.String.Input(
                    "custom_target",
                    default="the outdoor furniture",
                    tooltip="What to focus on. Examples: 'the outdoor furniture on the deck', 'the entrance door', 'the garden fountain'"
                ),

                # Group 5: Camera Positioning (Custom Mode)
                io.Combo.Input(
                    "direction",
                    options=["front", "left", "right", "back", "front_left", "front_right", "back_left", "back_right"],
                    default="front",
                    tooltip="Direction to position camera relative to target. 'front' = directly in front."
                ),
                io.Float.Input(
                    "distance",
                    default=2.5,
                    min=0.5,
                    max=20.0,
                    step=0.5,
                    tooltip="Distance from target in meters (0.5-20m). Smaller = closer/more intimate, larger = more context."
                ),
                io.Combo.Input(
                    "height",
                    options=["ground_level", "slightly_below", "face_level", "slightly_above", "elevated", "high"],
                    default="face_level",
                    tooltip="Camera height: face_level = standard, elevated = bird's eye, ground_level = worm's eye"
                ),

                # Group 6: Auto-Facing
                io.Boolean.Input(
                    "auto_facing",
                    default=True,
                    tooltip="Automatically face the target (recommended). Keeps target centered in frame."
                ),

                # Group 7: Options
                io.Boolean.Input(
                    "debug_mode",
                    default=False,
                    tooltip="Print detailed prompt information to console"
                ),
            ],
            outputs=[
                io.String.Output(
                    "prompt",
                    tooltip="Generated prompt for Qwen Edit 2509"
                ),
                io.String.Output(
                    "preset_description",
                    tooltip="Description of the selected focus preset"
                ),
                io.String.Output(
                    "system_prompt",
                    tooltip="⭐ v5.0: Research-validated system prompt for exterior focus! Connect to encoder's system_prompt input."
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        preset,
        prompt_style,
        scene_context,
        custom_target,
        direction,
        distance,
        height,
        auto_facing,
        debug_mode,
    ) -> io.NodeOutput:
        """Execute the Exterior Focus node."""

        # Step 1: Apply preset or use custom parameters
        if preset == "custom":
            # Use custom UI values directly
            final_target = custom_target
            final_direction = direction
            final_distance = distance
            final_height = height
            final_auto_facing = auto_facing
            final_framing = None
        else:
            # Use preset values
            preset_data = EXTERIOR_FOCUS_PRESETS.get(preset, {})
            final_target = preset_data.get("target", custom_target)
            final_direction = preset_data.get("direction", direction)
            final_distance = preset_data.get("distance", distance)
            final_height = preset_data.get("height", height)
            final_auto_facing = preset_data.get("auto_facing", auto_facing)
            final_framing = preset_data.get("framing", None)  # New: framing descriptor

        # Get preset description
        preset_desc = ""
        if preset != "custom":
            preset_desc = EXTERIOR_FOCUS_PRESETS[preset].get("description", "")

        # Step 2: Build focus prompt with selected pattern
        prompt = build_exterior_focus_prompt(
            scene_context=scene_context,
            target=final_target,
            direction=final_direction,
            distance=final_distance,
            height=final_height,
            auto_facing=final_auto_facing,
            prompt_style=prompt_style,
            framing=final_framing,
        )

        # Step 3: Get system prompt (v5.0.0 feature - research-validated)
        system_prompt = EXTERIOR_FOCUS_SYSTEM_PROMPT

        # Step 4: Debug output
        if debug_mode:
            debug_lines = [
                "=" * 70,
                "ArchAi3D_Qwen_Exterior_Focus - Generated Prompt (v5.0.0)",
                "=" * 70,
                f"Preset: {preset}",
                f"Description: {preset_desc}",
                f"Scene Context: {scene_context[:80]}..." if len(scene_context) > 80 else f"Scene Context: {scene_context}",
                f"Target: {final_target}",
                f"Direction: {final_direction}",
                f"Distance: {final_distance}m",
                f"Height: {final_height}",
                f"Auto-Facing: {final_auto_facing}",
                "=" * 70,
                "Generated Prompt:",
                prompt,
                "=" * 70,
                "⭐ System Prompt (v5.0.0 - Research-Validated):",
                system_prompt,
                "=" * 70,
                "",
                "💡 TIP: For best results:",
                "  - Use this node to FOCUS on specific outdoor elements",
                "  - For MOVING through space, use Exterior_Navigation",
                "  - For changing VIEW ANGLE only, use Exterior_View_Control",
                "=" * 70,
            ]
            print("\n".join(debug_lines))

        # Step 5: Return
        return io.NodeOutput(prompt, preset_desc, system_prompt)


# ============================================================================
# EXTENSION REGISTRATION
# ============================================================================

class ExteriorFocusExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [ArchAi3D_Qwen_Exterior_Focus]


async def comfy_entrypoint():
    return ExteriorFocusExtension()
