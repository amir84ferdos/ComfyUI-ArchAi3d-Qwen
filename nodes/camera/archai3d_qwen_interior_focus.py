# -*- coding: utf-8 -*-
"""
ArchAi3D Qwen Interior Focus Node (v1)
Author: Amir Ferdos (ArchAi3d)
Email: Amir84ferdos@gmail.com
LinkedIn: https://www.linkedin.com/in/archai3d/

Description:
    Focus camera on specific interior elements (furniture, appliances, fixtures, decor).
    Perfect for highlighting specific objects within rooms.

    Perfect for:
    - Kitchen appliances closeups (toaster, coffee maker, oven) ⭐ USER'S TOASTER USE CASE!
    - Furniture details (sofa, dining table, bed)
    - Fixtures and features (fireplace, window, chandelier)
    - Decor elements (artwork, plants, accessories)

Based on research:
    - Vantage point change pattern (QWEN_PROMPT_GUIDE)
    - "change the view to [HEIGHT] vantage point [DISTANCE]m to the [DIRECTION] facing [TARGET]"
    - Auto-facing mode keeps target centered
    - Distance-based positioning (meters, not degrees)
"""

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

# ============================================================================
# INTERIOR FOCUS SYSTEM PROMPT - v5.0.0 Research-Validated
# ============================================================================

INTERIOR_FOCUS_SYSTEM_PROMPT = """You are an interior photographer. Position camera to focus on specific furniture, appliance, fixture, or decor element within room. Frame subject prominently in foreground while showing room context. Preserve all furniture, decor, lighting fixtures, wall features, flooring, ceiling details, and spatial relationships exactly. Professional interior photography framing with proper depth and composition."""

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
# FOCUS TARGET PRESETS - 18 Interior Elements (15 original + 3 NEW)
# ============================================================================

INTERIOR_FOCUS_PRESETS = {
    "custom": {
        "description": "Manual control of focus target and positioning",
        "target": "",
        "direction": "front",
        "distance": 2,
        "height": "face level",
        "auto_facing": True,
    },

    # KITCHEN APPLIANCES (5) - ⭐ USER'S TOASTER USE CASE!
    "kitchen_appliance_closeup": {
        "description": "Close to kitchen appliance (toaster, coffee maker, microwave) - 1.5-2m closeup ⭐ YOUR TOASTER SOLUTION!",
        "target": "the kitchen appliance",
        "direction": "front",
        "distance": 1.5,
        "height": "face level",
        "auto_facing": True,
    },
    "countertop_appliance": {
        "description": "Focus on countertop appliance with kitchen context",
        "target": "the countertop appliance",
        "direction": "front",
        "distance": 2,
        "height": "slightly above",
        "auto_facing": True,
    },
    "stove_oven": {
        "description": "Focus on stove or oven - cooking area",
        "target": "the stove",
        "direction": "front",
        "distance": 2.5,
        "height": "face level",
        "auto_facing": True,
    },
    "refrigerator": {
        "description": "Focus on refrigerator - major kitchen appliance",
        "target": "the refrigerator",
        "direction": "front",
        "distance": 3,
        "height": "face level",
        "auto_facing": True,
    },
    "sink_faucet": {
        "description": "Focus on kitchen sink and faucet",
        "target": "the sink",
        "direction": "front",
        "distance": 2,
        "height": "slightly above",
        "auto_facing": True,
    },

    # FURNITURE (5)
    "sofa_closeup": {
        "description": "Close to sofa or couch - living room seating",
        "target": "the sofa",
        "direction": "front",
        "distance": 2.5,
        "height": "face level",
        "auto_facing": True,
    },
    "dining_table": {
        "description": "Focus on dining table - table setting view",
        "target": "the dining table",
        "direction": "front",
        "distance": 3,
        "height": "slightly above",
        "auto_facing": True,
    },
    "bed_closeup": {
        "description": "Close to bed - bedroom focal point",
        "target": "the bed",
        "direction": "front",
        "distance": 3,
        "height": "face level",
        "auto_facing": True,
    },
    "desk_workspace": {
        "description": "Focus on desk or workspace area",
        "target": "the desk",
        "direction": "front",
        "distance": 2,
        "height": "face level",
        "auto_facing": True,
    },
    "chair_detail": {
        "description": "Close to chair - furniture detail shot",
        "target": "the chair",
        "direction": "front",
        "distance": 1.5,
        "height": "face level",
        "auto_facing": True,
    },

    # FIXTURES & FEATURES (5)
    "fireplace": {
        "description": "Focus on fireplace - architectural feature",
        "target": "the fireplace",
        "direction": "front",
        "distance": 2.5,
        "height": "face level",
        "auto_facing": True,
    },
    "window_view": {
        "description": "Focus on window - show view and natural light",
        "target": "the window",
        "direction": "front",
        "distance": 2,
        "height": "face level",
        "auto_facing": True,
    },
    "chandelier_lighting": {
        "description": "Focus on chandelier or ceiling light fixture",
        "target": "the chandelier",
        "direction": "front",
        "distance": 3,
        "height": "slightly below",
        "auto_facing": True,
    },
    "artwork_wall": {
        "description": "Focus on wall artwork or decoration",
        "target": "the artwork on the wall",
        "direction": "front",
        "distance": 2,
        "height": "face level",
        "auto_facing": True,
    },
    "plant_decor": {
        "description": "Focus on plant or decorative element",
        "target": "the plant",
        "direction": "front",
        "distance": 1.5,
        "height": "face level",
        "auto_facing": True,
    },

    # NEW PRESETS (6) - Enhanced framing options based on research + user testing
    "macro_texture_detail": {
        "description": "⭐ NEW: Macro extreme closeup - finest material textures (<1m)",
        "target": "the surface detail",
        "direction": "front",
        "distance": 0.5,
        "height": "face level",
        "auto_facing": True,
        "framing": "macro",
    },
    "closeup_shot": {
        "description": "⭐ NEW: Closeup shot - user-tested pattern! Tight framing (1-1.5m)",
        "target": "the interior element",
        "direction": "front",
        "distance": 1,
        "height": "face level",
        "auto_facing": True,
        "framing": "closeup",
    },
    "detail_view": {
        "description": "⭐ NEW: Detail view - emphasize features and design (1.5-2m)",
        "target": "the furniture detail",
        "direction": "front",
        "distance": 1.5,
        "height": "face level",
        "auto_facing": True,
        "framing": "detail",
    },
    "wide_room_context": {
        "description": "⭐ NEW: Wide shot - element with full room context (4-5m)",
        "target": "the interior feature",
        "direction": "front",
        "distance": 4.5,
        "height": "face level",
        "auto_facing": True,
        "framing": "wide",
    },
    "establishing_room": {
        "description": "⭐ NEW: Establishing shot - introduce room and featured element (5m+)",
        "target": "the room feature",
        "direction": "front",
        "distance": 5.5,
        "height": "slightly above",
        "auto_facing": True,
        "framing": "establishing",
    },
    "overhead_flatlay_interior": {
        "description": "⭐ NEW: Overhead flat-lay - top-down interior element view",
        "target": "the table surface",
        "direction": "front",
        "distance": 2,
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
    """Convert number to written words - prevents numbers appearing in images!"""
    if num % 1 == 0.5:
        whole = int(num)
        if whole == 0:
            return "half a"
        return f"{number_to_words(float(whole))} and a half"

    num_int = int(num)
    words_map = {
        0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
        5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
        11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
        16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty"
    }
    return words_map.get(num_int, str(num_int))


def build_interior_focus_prompt(
    scene_context: str,
    target: str,
    direction: str,
    distance: float,
    height: str,
    auto_facing: bool,
    prompt_style: str = "vantage_point",
    framing: str = None,
) -> str:
    """Build interior focus prompt with dual-pattern system.

    Two supported patterns:

    1. VANTAGE POINT pattern (QWEN Guide Function 2 - ⭐⭐⭐⭐⭐):
       [SCENE_CONTEXT], change the view to a vantage point at [HEIGHT] [DISTANCE] to the [DIRECTION] facing [TARGET]

    2. ZOOM pattern (User-tested - ✅ Real-world proven):
       edit camera view, zoom the camera on [TARGET], [FRAMING_DESCRIPTOR]
       Example: "edit camera view, zoom the camera on sink, closeup shot"

    Args:
        scene_context: Description of the interior room
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

        # Convert distance to words (prevents numbers appearing in images!)
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

class ArchAi3D_Qwen_Interior_Focus(io.ComfyNode):
    """Interior Focus: Position camera to focus on interior elements.

    This node positions the camera to focus on specific interior elements:
    - Kitchen appliances (toaster, coffee maker, oven, fridge) ⭐ TOASTER USE CASE!
    - Furniture (sofa, dining table, bed, desk, chairs)
    - Fixtures & features (fireplace, windows, lighting, artwork)
    - Decor elements (plants, accessories)

    Key Features:
    - 15 preset focus targets for interior rooms
    - Research-validated vantage point change formula
    - Auto-facing mode (keeps target centered)
    - Distance control (meters, not degrees)
    - Height control (ground level to elevated)
    - Automatic system prompt output

    Perfect For:
    - Kitchen appliance closeups (YOUR TOASTER USE CASE! 🍞)
    - Real estate detail shots (show furniture, fixtures)
    - Interior design photography (highlight elements)
    - Product shots within rooms (appliances in context)

    Based on research:
    - Vantage point change formula (QWEN_PROMPT_GUIDE.md)
    - "change the view to [HEIGHT] vantage point [DISTANCE]m to the [DIRECTION] facing [TARGET]"
    - Auto-facing mode more reliable than manual rotation
    - Distance measurements (meters) for precise positioning

    USE THIS NODE WHEN:
    - You want to FOCUS on a specific object in a room
    - Object is far away and you want closeup
    - Need to highlight furniture, appliances, or fixtures
    - Example: "stand in front of toaster and show closeup" ⭐

    SCENE TYPE: Interior/Room/Indoor
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ArchAi3D_Qwen_Interior_Focus",
            category="ArchAi3d/Camera/Interior",
            inputs=[
                # Group 1: Preset Selection
                io.Combo.Input(
                    "preset",
                    options=list(INTERIOR_FOCUS_PRESETS.keys()),
                    default="kitchen_appliance_closeup",
                    tooltip="Select focus target or use 'custom' for manual control. 'kitchen_appliance_closeup' = TOASTER USE CASE! 🍞"
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
                    tooltip="Optional: Room description. Example: 'modern kitchen with white countertops'. Improves consistency."
                ),

                # Group 4: Focus Target (Custom Mode)
                io.String.Input(
                    "custom_target",
                    default="the toaster",
                    tooltip="What to focus on. Examples: 'the toaster', 'the coffee maker', 'the sofa', 'the fireplace'"
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
                    default=1.5,
                    min=0.5,
                    max=10.0,
                    step=0.5,
                    tooltip="Distance from target in meters (0.5-10m). Smaller = closer/more intimate, larger = more context."
                ),
                io.Combo.Input(
                    "height",
                    options=["ground_level", "slightly_below", "face_level", "slightly_above", "elevated", "high"],
                    default="face_level",
                    tooltip="Camera height: face_level = standard, slightly_above = countertop view, elevated = bird's eye"
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
                    tooltip="⭐ v5.0: Research-validated system prompt for interior focus! Connect to encoder's system_prompt input."
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
        """Execute the Interior Focus node."""

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
            preset_data = INTERIOR_FOCUS_PRESETS.get(preset, {})
            final_target = preset_data.get("target", custom_target)
            final_direction = preset_data.get("direction", direction)
            final_distance = preset_data.get("distance", distance)
            final_height = preset_data.get("height", height)
            final_auto_facing = preset_data.get("auto_facing", auto_facing)
            final_framing = preset_data.get("framing", None)  # New: framing descriptor

        # Get preset description
        preset_desc = ""
        if preset != "custom":
            preset_desc = INTERIOR_FOCUS_PRESETS[preset].get("description", "")

        # Step 2: Build focus prompt with selected pattern
        prompt = build_interior_focus_prompt(
            scene_context=scene_context,
            target=final_target,
            direction=final_direction,
            distance=final_distance,
            height=final_height,
            auto_facing=final_auto_facing,
            prompt_style=prompt_style,
            framing=final_framing,
        )

        # Step 3: Get system prompt
        system_prompt = INTERIOR_FOCUS_SYSTEM_PROMPT

        # Step 4: Debug output
        if debug_mode:
            debug_lines = [
                "=" * 70,
                "ArchAi3D_Qwen_Interior_Focus - Generated Prompt (v5.0.0)",
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
                "  - Use this node to FOCUS on specific objects in rooms",
                "  - For MOVING through room, use Interior_Navigation",
                "  - For changing VIEW ANGLE only, use Interior_View_Control",
                "  - kitchen_appliance_closeup preset perfect for toaster scenario! 🍞",
                "=" * 70,
            ]
            print("\n".join(debug_lines))

        # Step 5: Return
        return io.NodeOutput(prompt, preset_desc, system_prompt)


# ============================================================================
# EXTENSION REGISTRATION
# ============================================================================

class InteriorFocusExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [ArchAi3D_Qwen_Interior_Focus]


async def comfy_entrypoint():
    return InteriorFocusExtension()
