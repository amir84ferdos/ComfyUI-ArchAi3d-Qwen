# -*- coding: utf-8 -*-
"""
ArchAi3D Qwen Exterior Navigation Node (v1)
Author: Amir Ferdos (ArchAi3d)
Email: Amir84ferdos@gmail.com
LinkedIn: https://www.linkedin.com/in/archai3d/

Description:
    Navigate camera through exterior spaces (move through outdoor environments).
    Perfect for architectural walkarounds, building flybys, outdoor explorations.

    Perfect for:
    - Approaching/retreating from buildings
    - Walking around structures
    - Drone movements (flyby, orbit, rise/descend)
    - Outdoor path navigation

Based on research:
    - Environment_Navigator patterns (smooth movement + rotation)
    - QWEN_PROMPT_GUIDE combined movement function
    - Distance-based positioning (meters, not degrees)
    - Speed modifiers for cinematic quality
"""

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

# ============================================================================
# EXTERIOR NAVIGATION SYSTEM PROMPT - v5.0.0 Research-Validated
# ============================================================================

EXTERIOR_NAVIGATION_SYSTEM_PROMPT = """You are an architectural cinematographer. Navigate camera through exterior outdoor space. Move smoothly maintaining steady framing. Reveal architecture or environment progressively through camera movement. Preserve all building details, architectural proportions, outdoor furniture, outdoor lighting, sky, and environmental context exactly. Natural movement with cinematic quality and proper perspective throughout."""

# ============================================================================
# MOVEMENT AND ROTATION MAPPINGS
# ============================================================================

MOVEMENT_DIRECTION_MAP = {
    "forward": "forward",
    "backward": "backward",
    "left": "to the left",
    "right": "to the right",
    "up": "upward",
    "down": "downward",
}

ROTATION_DURING_MAP = {
    "none": "",
    "left": "while rotating left",
    "right": "while rotating right",
    "up": "while tilting up",
    "down": "while tilting down",
}

SPEED_MAP = {
    "slow": "slowly",
    "normal": "",
    "fast": "quickly",
    "smooth": "with smooth cinematic camera movement",
}

# ============================================================================
# NAVIGATION PRESETS - 15 Exterior Movement Patterns
# ============================================================================

EXTERIOR_NAVIGATION_PRESETS = {
    "custom": {
        "description": "Manual control of navigation parameters",
        "movement": "forward",
        "distance": 5,
        "rotation": "none",
        "speed": "smooth",
    },

    # APPROACH/RETREAT (2)
    "approach_building": {
        "description": "Move toward building - reveal details as you approach",
        "movement": "forward",
        "distance": 5,
        "rotation": "none",
        "speed": "smooth",
    },
    "retreat_building": {
        "description": "Move away from building - reveal context and surroundings",
        "movement": "backward",
        "distance": 8,
        "rotation": "none",
        "speed": "smooth",
    },

    # WALK AROUND (2)
    "walk_around_left": {
        "description": "Walk around building to left - circular exploration",
        "movement": "left",
        "distance": 8,
        "rotation": "right",  # Rotate right to keep building centered
        "speed": "smooth",
    },
    "walk_around_right": {
        "description": "Walk around building to right - circular exploration",
        "movement": "right",
        "distance": 8,
        "rotation": "left",  # Rotate left to keep building centered
        "speed": "smooth",
    },

    # FLYBY (2)
    "flyby_left": {
        "description": "Elevated horizontal pass from left to right - drone flyby",
        "movement": "right",
        "distance": 12,
        "rotation": "none",
        "speed": "smooth",
    },
    "flyby_right": {
        "description": "Elevated horizontal pass from right to left - drone flyby",
        "movement": "left",
        "distance": 12,
        "rotation": "none",
        "speed": "smooth",
    },

    # VERTICAL MOVEMENTS (2)
    "rise_reveal": {
        "description": "Rise up (drone ascending) - dramatic upward reveal",
        "movement": "up",
        "distance": 6,
        "rotation": "down",  # Tilt down to keep subject in frame
        "speed": "slow",
    },
    "descend_approach": {
        "description": "Descend down (drone landing) - descending approach",
        "movement": "down",
        "distance": 5,
        "rotation": "down",
        "speed": "normal",
    },

    # ORBIT (2)
    "orbit_ascending": {
        "description": "Orbit building while ascending - spiral rising reveal",
        "movement": "right",
        "distance": 6,
        "rotation": "left",
        "speed": "smooth",
    },
    "orbit_descending": {
        "description": "Orbit building while descending - spiral lowering",
        "movement": "right",
        "distance": 6,
        "rotation": "left",
        "speed": "smooth",
    },

    # PATH NAVIGATION (2)
    "path_walk_forward": {
        "description": "Walk along path forward - natural walking pace",
        "movement": "forward",
        "distance": 8,
        "rotation": "none",
        "speed": "smooth",
    },
    "strafe_left": {
        "description": "Parallel movement left - tracking shot",
        "movement": "left",
        "distance": 6,
        "rotation": "none",
        "speed": "smooth",
    },
    "strafe_right": {
        "description": "Parallel movement right - tracking shot",
        "movement": "right",
        "distance": 6,
        "rotation": "none",
        "speed": "smooth",
    },

    # COMPLETE ORBIT (1)
    "circle_building": {
        "description": "Complete 360° around building - comprehensive view",
        "movement": "right",
        "distance": 10,
        "rotation": "left",
        "speed": "smooth",
    },
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def number_to_words(num: float) -> str:
    """Convert number to written words for better AI understanding.

    Based on user feedback: "two meters" works better than "2m"
    Numbers in numeric format cause AI to render them IN the image!
    """
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


def build_exterior_navigation_prompt(
    scene_context: str,
    movement: str,
    distance: int,
    rotation: str,
    speed: str,
) -> str:
    """Build exterior navigation prompt.

    Formula from QWEN Guide Function 4 (Line 232):
    [SCENE_CONTEXT], change the view and move [SPEED] [DISTANCE] meters [MOVEMENT_DIRECTION] [ROTATION]

    Args:
        scene_context: Description of the outdoor environment
        movement: Direction of movement
        distance: How many meters to move
        rotation: Camera rotation during movement
        speed: Movement speed modifier

    Returns:
        Complete prompt string
    """
    parts = []

    # Scene context
    if scene_context.strip():
        parts.append(scene_context.strip())

    # Build movement phrase
    speed_phrase = SPEED_MAP.get(speed, "")
    movement_phrase = MOVEMENT_DIRECTION_MAP.get(movement, "forward")

    # Convert distance to words (user feedback: "five meters" works better than "5m")
    distance_words = number_to_words(distance)

    # QWEN Guide Function 4 (Line 232): "change the view and move the camera [DIRECTION]"
    nav = "change the view and move"
    if speed_phrase:
        nav += f" {speed_phrase}"

    nav += f" {distance_words} meters {movement_phrase}"

    # Add rotation if specified
    rotation_phrase = ROTATION_DURING_MAP.get(rotation, "")
    if rotation_phrase:
        nav += f" {rotation_phrase}"

    parts.append(nav)

    return ", ".join(parts)


# ============================================================================
# NODE DEFINITION
# ============================================================================

class ArchAi3D_Qwen_Exterior_Navigation(io.ComfyNode):
    """Exterior Navigation: Move camera through outdoor spaces.

    This node enables camera movement through exterior environments:
    - Approach/retreat from buildings
    - Walk around structures (left/right)
    - Drone movements (flyby, rise/descend, orbit)
    - Path navigation and tracking shots

    Key Features:
    - 15 preset navigation patterns for exterior scenes
    - Research-validated movement patterns (distance-based)
    - Speed control (slow, normal, fast, smooth)
    - Combined movement + rotation for complex paths
    - Automatic system prompt output

    Perfect For:
    - Architectural walkarounds (360° building tours)
    - Real estate flyovers (drone perspectives)
    - Outdoor space exploration
    - Cinematic establishing shots

    Based on research:
    - Environment_Navigator patterns (ARCHAI3D_QWEN_MEMORY.md)
    - Combined movement function (QWEN_PROMPT_GUIDE.md)
    - Distance measurements (meters) more reliable than degrees
    - "smooth camera movement" modifier for cinematic quality

    USE THIS NODE WHEN:
    - You want to MOVE through exterior space
    - Building or outdoor environment is the scene
    - You need walkthrough/flyby/orbit movements

    SCENE TYPE: Exterior/Outdoor/Architectural
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ArchAi3D_Qwen_Exterior_Navigation",
            category="ArchAi3d/Camera/Exterior",
            inputs=[
                # Group 1: Preset Selection
                io.Combo.Input(
                    "preset",
                    options=list(EXTERIOR_NAVIGATION_PRESETS.keys()),
                    default="approach_building",
                    tooltip="Select navigation pattern or use 'custom' for manual control"
                ),

                # Group 2: Scene Context
                io.String.Input(
                    "scene_context",
                    multiline=True,
                    default="",
                    tooltip="Optional: Outdoor scene description. Example: 'modern glass building', 'outdoor deck with furniture'. Improves consistency."
                ),

                # Group 3: Movement Control (Custom Mode)
                io.Combo.Input(
                    "movement",
                    options=["forward", "backward", "left", "right", "up", "down"],
                    default="forward",
                    tooltip="Movement direction: forward=approach, backward=retreat, left/right=strafe, up/down=vertical (drone)"
                ),
                io.Int.Input(
                    "distance",
                    default=5,
                    min=1,
                    max=20,
                    step=1,
                    tooltip="Movement distance in meters (1-20m). Longer = more dramatic transition"
                ),

                # Group 4: Rotation During Movement
                io.Combo.Input(
                    "rotation",
                    options=["none", "left", "right", "up", "down"],
                    default="none",
                    tooltip="Camera rotation during movement: none=straight, left/right=pan, up/down=tilt. Creates fluid combined movements."
                ),

                # Group 5: Speed Control
                io.Combo.Input(
                    "speed",
                    options=["slow", "normal", "fast", "smooth"],
                    default="smooth",
                    tooltip="Movement speed: slow=deliberate, normal=natural, fast=dynamic, smooth=cinematic"
                ),

                # Group 6: Options
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
                    tooltip="Description of the selected navigation pattern"
                ),
                io.String.Output(
                    "system_prompt",
                    tooltip="⭐ v5.0: Research-validated system prompt for exterior navigation! Connect to encoder's system_prompt input."
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        preset,
        scene_context,
        movement,
        distance,
        rotation,
        speed,
        debug_mode,
    ) -> io.NodeOutput:
        """Execute the Exterior Navigation node.

        Steps:
        1. Apply preset or use custom parameters
        2. Build navigation prompt
        3. Get system prompt
        4. Debug output if requested
        5. Return outputs
        """

        # Step 1: Apply preset or use custom parameters
        if preset == "custom":
            # Use custom UI values directly
            final_movement = movement
            final_distance = distance
            final_rotation = rotation
            final_speed = speed
        else:
            # Use preset values
            preset_data = EXTERIOR_NAVIGATION_PRESETS.get(preset, {})
            final_movement = preset_data.get("movement", movement)
            final_distance = preset_data.get("distance", distance)
            final_rotation = preset_data.get("rotation", rotation)
            final_speed = preset_data.get("speed", speed)

        # Get preset description
        preset_desc = ""
        if preset != "custom":
            preset_desc = EXTERIOR_NAVIGATION_PRESETS[preset].get("description", "")

        # Step 2: Build navigation prompt
        prompt = build_exterior_navigation_prompt(
            scene_context=scene_context,
            movement=final_movement,
            distance=final_distance,
            rotation=final_rotation,
            speed=final_speed,
        )

        # Step 3: Get system prompt (v5.0.0 feature - research-validated)
        system_prompt = EXTERIOR_NAVIGATION_SYSTEM_PROMPT

        # Step 4: Debug output
        if debug_mode:
            debug_lines = [
                "=" * 70,
                "ArchAi3D_Qwen_Exterior_Navigation - Generated Prompt (v5.0.0)",
                "=" * 70,
                f"Preset: {preset}",
                f"Description: {preset_desc}",
                f"Scene Context: {scene_context[:80]}..." if len(scene_context) > 80 else f"Scene Context: {scene_context}",
                f"Movement: {final_movement}",
                f"Distance: {final_distance}m",
                f"Rotation: {final_rotation}",
                f"Speed: {final_speed}",
                "=" * 70,
                "Generated Prompt:",
                prompt,
                "=" * 70,
                "⭐ System Prompt (v5.0.0 - Research-Validated):",
                system_prompt,
                "=" * 70,
                "",
                "💡 TIP: For best results:",
                "  - Use this node to MOVE through exterior space",
                "  - For changing view ANGLE only, use Exterior_View_Control",
                "  - For FOCUSING on outdoor furniture, use Exterior_Focus",
                "=" * 70,
            ]
            print("\n".join(debug_lines))

        # Step 5: Return
        return io.NodeOutput(prompt, preset_desc, system_prompt)


# ============================================================================
# EXTENSION REGISTRATION
# ============================================================================

class ExteriorNavigationExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [ArchAi3D_Qwen_Exterior_Navigation]


async def comfy_entrypoint():
    return ExteriorNavigationExtension()
