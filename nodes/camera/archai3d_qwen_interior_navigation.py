# -*- coding: utf-8 -*-
"""
ArchAi3D Qwen Interior Navigation Node (v1)
Author: Amir Ferdos (ArchAi3d)
Email: Amir84ferdos@gmail.com
LinkedIn: https://www.linkedin.com/in/archai3d/

Description:
    Navigate camera through interior rooms and spaces.
    Perfect for walkthroughs, room tours, and cinematic interior movements.

    Perfect for:
    - Virtual real estate tours (move through rooms)
    - Interior design walkthroughs (show space flow)
    - Room-to-room transitions (doorway passages)
    - Cinematic interior shots (dolly, tracking)

Based on research:
    - Camera movement pattern (QWEN_PROMPT_GUIDE)
    - "move [SPEED] [DISTANCE]m [DIRECTION] [ROTATION_DURING]"
    - Distance-based positioning (meters, not pixels)
    - Natural movement descriptions
"""

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

# ============================================================================
# INTERIOR NAVIGATION SYSTEM PROMPT - v5.0.0 Research-Validated
# ============================================================================

INTERIOR_NAVIGATION_SYSTEM_PROMPT = """You are an interior cinematographer. Navigate camera smoothly through interior rooms and spaces. Move through doorways, along walls, and across rooms maintaining steady framing. Reveal interior architecture and design progressively through camera movement. Preserve all furniture, decor, lighting fixtures, wall features, flooring, ceiling details, and spatial relationships exactly. Natural movement with cinematic quality and proper interior perspective throughout."""

# ============================================================================
# NAVIGATION PRESETS - 15 Interior Movement Patterns
# ============================================================================

INTERIOR_NAVIGATION_PRESETS = {
    "custom": "Manual control of navigation movement",

    # ROOM ENTRY/EXIT (3)
    "enter_room": "Enter room through doorway - welcoming reveal",
    "exit_room": "Exit room through doorway - departing view",
    "doorway_transition": "Pass through doorway - room-to-room transition",

    # FORWARD/BACKWARD (2)
    "walk_forward": "Walk forward into room - progressive reveal",
    "walk_backward": "Walk backward out of room - departing perspective",

    # ALONG WALLS (4)
    "wall_track_left": "Track along left wall - parallel movement",
    "wall_track_right": "Track along right wall - parallel movement",
    "corner_to_corner_left": "Move from corner to corner (left side)",
    "corner_to_corner_right": "Move from corner to corner (right side)",

    # ACROSS ROOM (2)
    "cross_room_left": "Cross room from right to left",
    "cross_room_right": "Cross room from left to right",

    # VERTICAL (2)
    "rise_ceiling": "Rise up revealing ceiling details",
    "lower_floor": "Lower down revealing floor details",

    # CIRCULAR (2)
    "circle_room_left": "Circle around room to the left",
    "circle_room_right": "Circle around room to the right",
}

# ============================================================================
# MOVEMENT MAPPINGS
# ============================================================================

MOVEMENT_DIRECTION_MAP = {
    "forward": "forward",
    "backward": "backward",
    "left": "to the left",
    "right": "to the right",
    "up": "upward",
    "down": "downward",
}

SPEED_MAP = {
    "very_slow": "very slowly",
    "slow": "slowly",
    "normal": "",
    "fast": "quickly",
    "very_fast": "very quickly",
}

ROTATION_DURING_MAP = {
    "none": "",
    "look_left": "while looking left",
    "look_right": "while looking right",
    "look_up": "while looking up",
    "look_down": "while looking down",
    "follow_wall": "following the wall",
}

# Navigation descriptions for presets
NAVIGATION_DESCRIPTIONS = {
    # Room entry/exit
    "enter_room": {"direction": "forward", "distance": 3, "rotation": "none", "speed": "slow"},
    "exit_room": {"direction": "backward", "distance": 3, "rotation": "none", "speed": "slow"},
    "doorway_transition": {"direction": "forward", "distance": 2, "rotation": "none", "speed": "slow"},

    # Forward/backward
    "walk_forward": {"direction": "forward", "distance": 4, "rotation": "none", "speed": "normal"},
    "walk_backward": {"direction": "backward", "distance": 4, "rotation": "none", "speed": "normal"},

    # Along walls
    "wall_track_left": {"direction": "left", "distance": 4, "rotation": "follow_wall", "speed": "slow"},
    "wall_track_right": {"direction": "right", "distance": 4, "rotation": "follow_wall", "speed": "slow"},
    "corner_to_corner_left": {"direction": "left", "distance": 5, "rotation": "follow_wall", "speed": "slow"},
    "corner_to_corner_right": {"direction": "right", "distance": 5, "rotation": "follow_wall", "speed": "slow"},

    # Across room
    "cross_room_left": {"direction": "left", "distance": 5, "rotation": "none", "speed": "normal"},
    "cross_room_right": {"direction": "right", "distance": 5, "rotation": "none", "speed": "normal"},

    # Vertical
    "rise_ceiling": {"direction": "up", "distance": 2, "rotation": "look_up", "speed": "slow"},
    "lower_floor": {"direction": "down", "distance": 2, "rotation": "look_down", "speed": "slow"},

    # Circular
    "circle_room_left": {"direction": "left", "distance": 6, "rotation": "follow_wall", "speed": "slow"},
    "circle_room_right": {"direction": "right", "distance": 6, "rotation": "follow_wall", "speed": "slow"},
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


def build_interior_navigation_prompt(
    scene_context: str,
    movement: str,
    distance: float,
    rotation: str,
    speed: str,
) -> str:
    """Build interior navigation prompt.

    Formula from QWEN Guide Function 4 (Line 232):
    [SCENE_CONTEXT], change the view and move [SPEED] [DISTANCE] meters [DIRECTION] [ROTATION_DURING]

    Args:
        scene_context: Description of the interior room
        movement: Movement direction
        distance: Distance to move in meters
        rotation: Look direction during movement
        speed: Movement speed

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

    # Convert distance to words (prevents numbers appearing in images!)
    distance_words = number_to_words(distance)

    # QWEN Guide Function 4 (Line 232): "change the view and move the camera [DIRECTION]"
    nav = "change the view and move"
    if speed_phrase:
        nav += f" {speed_phrase}"
    nav += f" {distance_words} meters {movement_phrase}"

    # Add rotation during movement
    rotation_phrase = ROTATION_DURING_MAP.get(rotation, "")
    if rotation_phrase:
        nav += f" {rotation_phrase}"

    parts.append(nav)

    return ", ".join(parts)


# ============================================================================
# NODE DEFINITION
# ============================================================================

class ArchAi3D_Qwen_Interior_Navigation(io.ComfyNode):
    """Interior Navigation: Move camera through interior rooms and spaces.

    This node moves the camera through interior environments:
    - Room entry/exit (enter through doorway, exit room)
    - Forward/backward walks (walk into room, walk out)
    - Wall tracking (parallel movement along walls)
    - Cross-room movements (left to right, right to left)
    - Vertical movements (rise to ceiling, lower to floor)
    - Circular movements (circle around room)

    Key Features:
    - 15 preset navigation patterns for interior spaces
    - Research-validated movement descriptions
    - Distance control (meters, not pixels)
    - Speed control (very slow to very fast)
    - Look direction during movement
    - Automatic system prompt output

    Perfect For:
    - Virtual real estate tours (walk through homes)
    - Interior design presentations (show room flow)
    - Architectural interiors (reveal spatial relationships)
    - Cinematic interior shots (professional camera movement)

    Based on research:
    - Camera movement formula (QWEN_PROMPT_GUIDE.md)
    - "move [SPEED] [DISTANCE]m [DIRECTION] [ROTATION_DURING]"
    - Natural language descriptions most reliable
    - Distance-based positioning (meters)

    USE THIS NODE WHEN:
    - You want to MOVE through interior rooms
    - Need walkthrough or tour movements
    - Creating cinematic interior sequences

    SCENE TYPE: Interior/Room/Indoor
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ArchAi3D_Qwen_Interior_Navigation",
            category="ArchAi3d/Camera/Interior",
            inputs=[
                # Group 1: Preset Selection
                io.Combo.Input(
                    "preset",
                    options=list(INTERIOR_NAVIGATION_PRESETS.keys()),
                    default="enter_room",
                    tooltip="Select navigation movement or use 'custom' for manual control. 'enter_room' is popular for real estate tours."
                ),

                # Group 2: Scene Context
                io.String.Input(
                    "scene_context",
                    multiline=True,
                    default="",
                    tooltip="Optional: Room description. Example: 'modern living room with open kitchen'. Improves consistency."
                ),

                # Group 3: Movement Parameters (Custom Mode)
                io.Combo.Input(
                    "movement",
                    options=["forward", "backward", "left", "right", "up", "down"],
                    default="forward",
                    tooltip="Movement direction. forward/backward = depth, left/right = strafe, up/down = vertical"
                ),
                io.Float.Input(
                    "distance",
                    default=3.0,
                    min=0.5,
                    max=10.0,
                    step=0.5,
                    tooltip="Distance to move in meters (0.5-10m). Typical room: 3-5m"
                ),
                io.Combo.Input(
                    "speed",
                    options=["very_slow", "slow", "normal", "fast", "very_fast"],
                    default="normal",
                    tooltip="Movement speed. slow = cinematic, normal = natural, fast = dynamic"
                ),
                io.Combo.Input(
                    "rotation",
                    options=["none", "look_left", "look_right", "look_up", "look_down", "follow_wall"],
                    default="none",
                    tooltip="Look direction during movement. 'follow_wall' keeps wall in frame during tracking shots."
                ),

                # Group 4: Options
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
                    tooltip="Description of the selected navigation preset"
                ),
                io.String.Output(
                    "system_prompt",
                    tooltip="⭐ v5.0: Research-validated system prompt for interior navigation! Connect to encoder's system_prompt input."
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
        speed,
        rotation,
        debug_mode,
    ) -> io.NodeOutput:
        """Execute the Interior Navigation node."""

        # Step 1: Apply preset or use custom parameters
        if preset == "custom":
            # Use custom UI values directly
            final_movement = movement
            final_distance = distance
            final_speed = speed
            final_rotation = rotation
        else:
            # Use preset values
            preset_data = NAVIGATION_DESCRIPTIONS.get(preset, {})
            final_movement = preset_data.get("direction", movement)
            final_distance = preset_data.get("distance", distance)
            final_speed = preset_data.get("speed", speed)
            final_rotation = preset_data.get("rotation", rotation)

        # Get preset description
        preset_desc = INTERIOR_NAVIGATION_PRESETS.get(preset, "")

        # Step 2: Build navigation prompt
        prompt = build_interior_navigation_prompt(
            scene_context=scene_context,
            movement=final_movement,
            distance=final_distance,
            rotation=final_rotation,
            speed=final_speed,
        )

        # Step 3: Get system prompt
        system_prompt = INTERIOR_NAVIGATION_SYSTEM_PROMPT

        # Step 4: Debug output
        if debug_mode:
            debug_lines = [
                "=" * 70,
                "ArchAi3D_Qwen_Interior_Navigation - Generated Prompt (v5.0.0)",
                "=" * 70,
                f"Preset: {preset}",
                f"Description: {preset_desc}",
                f"Scene Context: {scene_context[:80]}..." if len(scene_context) > 80 else f"Scene Context: {scene_context}",
                f"Movement: {final_movement}",
                f"Distance: {final_distance}m",
                f"Speed: {final_speed}",
                f"Rotation: {final_rotation}",
                "=" * 70,
                "Generated Prompt:",
                prompt,
                "=" * 70,
                "⭐ System Prompt (v5.0.0 - Research-Validated):",
                system_prompt,
                "=" * 70,
                "",
                "💡 TIP: For best results:",
                "  - Use this node to MOVE through interior space",
                "  - For changing VIEW ANGLE only, use Interior_View_Control",
                "  - For focusing on specific furniture, use Interior_Focus",
                "=" * 70,
            ]
            print("\n".join(debug_lines))

        # Step 5: Return
        return io.NodeOutput(prompt, preset_desc, system_prompt)


# ============================================================================
# EXTENSION REGISTRATION
# ============================================================================

class InteriorNavigationExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [ArchAi3D_Qwen_Interior_Navigation]


async def comfy_entrypoint():
    return InteriorNavigationExtension()
