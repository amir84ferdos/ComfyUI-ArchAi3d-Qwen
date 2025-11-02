"""
ArchAi3D Qwen Environment Navigator Node
Author: Amir Ferdos (ArchAi3d)
Email: Amir84ferdos@gmail.com
LinkedIn: https://www.linkedin.com/in/archai3d/

Description:
    Move and rotate through environments like landscapes, interiors, and scenes.
    Perfect for:
    - Interior walkthroughs: "move through the living room while rotating right"
    - Landscape exploration: "move forward through the forest"
    - Architectural tours: "move around the building while looking at facade"
    - Scene navigation: "move left while camera pans right to follow subject"

Based on research: Combined movement + rotation works well for environment
exploration. Distance-based movements (meters) more reliable than degrees.
"""

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override
from typing import Dict

# ============================================================================
# CONSTANTS - Movement and rotation mappings
# ============================================================================

MOVEMENT_DIRECTION_MAP = {
    "forward": "forward",
    "backward": "backward",
    "left": "to the left",
    "right": "to the right",
    "up": "upward",
    "down": "downward",
}

ROTATION_DIRECTION_MAP = {
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
    "smooth": "with smooth camera movement",
}

# ============================================================================
# NAVIGATION SYSTEM PROMPTS - v4.0.0 Feature
# ============================================================================

NAVIGATION_SYSTEM_PROMPTS = {
    "custom": "You are a camera operator. Move camera through environment smoothly. Maintain stable framing during movement. Preserve scene details and natural perspective.",

    # Interior Walkthroughs
    "walkthrough_forward": "You are an interior cinematographer. Execute smooth forward walkthrough movement. Natural walking pace, steady camera. Reveal space gradually as you move. Preserve architectural details, lighting, and room proportions throughout movement.",

    "walkthrough_pan_right": "You are an interior cinematographer. Walk forward while smoothly panning camera right. Natural walking pace with fluid rotation. Reveal room details progressively. Preserve spatial relationships and interior design elements throughout combined movement.",

    "walkthrough_pan_left": "You are an interior cinematographer. Walk forward while smoothly panning camera left. Natural walking pace with fluid rotation. Reveal room details progressively. Preserve spatial relationships and interior design elements throughout combined movement.",

    "strafe_right": "You are an interior cinematographer. Execute smooth rightward strafe movement parallel to walls. Maintain consistent distance from surfaces. Professional tracking shot quality. Preserve architectural details and perspective throughout lateral movement.",

    "strafe_left": "You are an interior cinematographer. Execute smooth leftward strafe movement parallel to walls. Maintain consistent distance from surfaces. Professional tracking shot quality. Preserve architectural details and perspective throughout lateral movement.",

    # Landscape/Exterior Navigation
    "landscape_forward": "You are a nature cinematographer. Move forward through landscape with smooth camera movement. Natural walking pace revealing environment gradually. Maintain horizon stability. Preserve landscape features, natural lighting, and atmospheric depth throughout forward movement.",

    "landscape_rise": "You are a drone operator. Execute smooth ascending movement while tilting camera down. Reveal landscape progressively from elevated perspective. Professional drone shot quality. Preserve landscape scale, natural features, and environmental context throughout ascent.",

    "landscape_pan_360": "You are a nature cinematographer. Slow 360-degree pan while moving slightly forward. Smooth rotation revealing entire environment. Natural pacing for full environmental context. Preserve landscape features and atmospheric perspective throughout circular reveal.",

    # Architectural Navigation
    "building_approach": "You are an architectural cinematographer. Move forward approaching building while tilting camera up. Reveal building scale and height gradually. Professional architectural photography standards. Preserve building details, facade features, and architectural proportions throughout approach.",

    "building_circle": "You are an architectural cinematographer. Circle around building moving right while camera rotates left. Smooth orbital movement revealing all facades. Professional architectural tour quality. Preserve building details, proportions, and site context throughout complete rotation.",

    "building_flyby": "You are a drone operator. Execute elevated horizontal flyby moving right past building. Professional aerial cinematography standards. Smooth fast movement maintaining consistent altitude. Preserve building details and urban context throughout flyby movement.",

    # Dramatic/Cinematic Moves
    "dramatic_retreat": "You are a cinematic camera operator. Execute slow dramatic backward movement revealing wider context. Build tension through controlled retreat. Professional cinema quality. Preserve subject details and environmental context throughout backward reveal.",

    "dramatic_rise": "You are a cinematic camera operator. Execute slow dramatic upward movement while tilting down. Hero moment reveal with rising perspective. Professional cinema standards. Preserve subject presence and environmental scale throughout ascending movement.",

    "dramatic_descent": "You are a cinematic camera operator. Descend downward toward subject with controlled downward tilt. Focus attention through descending movement. Professional cinema quality. Preserve subject details and framing throughout descent.",
}

# ============================================================================
# NAVIGATION PRESETS - Common environment navigation patterns
# ============================================================================

NAVIGATION_PRESETS = {
    # Interior Walkthroughs
    "walkthrough_forward": {
        "description": "Walk forward through space (hallway, room entry)",
        "movement": "forward",
        "distance": 5,
        "rotation": "none",
        "speed": "smooth",
    },
    "walkthrough_pan_right": {
        "description": "Walk forward while panning right (reveal room)",
        "movement": "forward",
        "distance": 5,
        "rotation": "right",
        "speed": "smooth",
    },
    "walkthrough_pan_left": {
        "description": "Walk forward while panning left (reveal room)",
        "movement": "forward",
        "distance": 5,
        "rotation": "left",
        "speed": "smooth",
    },
    "strafe_right": {
        "description": "Strafe right (parallel to wall/subject)",
        "movement": "right",
        "distance": 5,
        "rotation": "none",
        "speed": "smooth",
    },
    "strafe_left": {
        "description": "Strafe left (parallel to wall/subject)",
        "movement": "left",
        "distance": 5,
        "rotation": "none",
        "speed": "smooth",
    },

    # Landscape/Exterior Navigation
    "landscape_forward": {
        "description": "Move forward through landscape/nature",
        "movement": "forward",
        "distance": 10,
        "rotation": "none",
        "speed": "smooth",
    },
    "landscape_rise": {
        "description": "Rise up to reveal landscape (drone shot)",
        "movement": "up",
        "distance": 5,
        "rotation": "down",
        "speed": "slow",
    },
    "landscape_pan_360": {
        "description": "Slow 360° pan revealing environment",
        "movement": "forward",
        "distance": 2,
        "rotation": "right",
        "speed": "slow",
    },

    # Architectural Navigation
    "building_approach": {
        "description": "Approach building while looking up",
        "movement": "forward",
        "distance": 5,
        "rotation": "up",
        "speed": "normal",
    },
    "building_circle": {
        "description": "Circle around building (reveal all facades)",
        "movement": "right",
        "distance": 8,
        "rotation": "left",
        "speed": "smooth",
    },
    "building_flyby": {
        "description": "Fly by building (elevated pass)",
        "movement": "right",
        "distance": 10,
        "rotation": "none",
        "speed": "fast",
    },

    # Dramatic/Cinematic Moves
    "dramatic_retreat": {
        "description": "Dramatic backward movement (reveal context)",
        "movement": "backward",
        "distance": 10,
        "rotation": "none",
        "speed": "slow",
    },
    "dramatic_rise": {
        "description": "Dramatic rise up (hero moment)",
        "movement": "up",
        "distance": 8,
        "rotation": "down",
        "speed": "slow",
    },
    "dramatic_descent": {
        "description": "Descend down toward subject (focus in)",
        "movement": "down",
        "distance": 5,
        "rotation": "down",
        "speed": "normal",
    },
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def apply_preset(preset_name: str, user_params: Dict) -> Dict:
    """Apply preset settings, allowing user overrides.

    Args:
        preset_name: Name of the preset to apply
        user_params: User-provided parameters that override preset

    Returns:
        Merged parameters dictionary
    """
    if preset_name == "custom":
        return user_params

    preset = NAVIGATION_PRESETS.get(preset_name, {})

    # Merge: preset defaults + user overrides
    result = preset.copy()
    result.update({k: v for k, v in user_params.items() if v is not None})

    return result


def build_navigation_prompt(
    scene_context: str,
    movement: str,
    distance: int,
    rotation: str,
    speed: str,
    maintain_focus: bool,
    focus_subject: str,
) -> str:
    """Build environment navigation prompt.

    Formula:
    [SCENE_CONTEXT], move [SPEED] [DISTANCE]m [MOVEMENT_DIRECTION] [ROTATION] [FOCUS]

    Args:
        scene_context: Description of the environment
        movement: Direction of movement
        distance: How many meters to move
        rotation: Camera rotation during movement
        speed: Movement speed modifier
        maintain_focus: Whether to keep focus on a subject
        focus_subject: What to keep in frame

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

    nav = "move"
    if speed_phrase:
        nav += f" {speed_phrase}"

    nav += f" {distance}m {movement_phrase}"

    # Add rotation if specified
    rotation_phrase = ROTATION_DIRECTION_MAP.get(rotation, "")
    if rotation_phrase:
        nav += f" {rotation_phrase}"

    # Maintain focus on subject if requested
    if maintain_focus and focus_subject.strip():
        nav += f" keeping {focus_subject} in frame"

    parts.append(nav)

    return ", ".join(parts)


# ============================================================================
# NODE DEFINITION
# ============================================================================

class ArchAi3D_Qwen_Environment_Navigator(io.ComfyNode):
    """Environment Navigator: Move and rotate through scenes.

    This node enables fluid camera movement through environments:
    - Walk through interiors (forward, backward, strafe left/right)
    - Explore landscapes (forward through forest, rise up for aerial view)
    - Circle around architecture (building tours, 360° reveals)
    - Dramatic cinematic moves (retreat, rise, descent)

    Key Features:
    - 14 preset navigation patterns for common movements
    - Combined movement + rotation for complex camera paths
    - Speed control (slow, normal, fast, smooth)
    - Maintain focus mode (keep subject in frame during movement)
    - Distance-based movements (more reliable than abstract directions)

    Perfect For:
    - Interior walkthroughs: Hallways, room reveals, space tours
    - Landscape exploration: Nature walks, environment reveals
    - Architectural tours: Building approaches, 360° circling, fly-bys
    - Cinematic shots: Dramatic retreats, hero rises, descent focuses

    Based on research: Combined "move X while rotating Y" pattern works
    well for environment exploration. Always specify distance in meters.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ArchAi3D_Qwen_Environment_Navigator",
            category="ArchAi3d/Qwen/Camera",
            inputs=[
                # Group 1: Preset Selection
                io.Combo.Input(
                    "preset",
                    options=["custom"] + list(NAVIGATION_PRESETS.keys()),
                    default="custom",
                    tooltip="Select a preset navigation pattern or use 'custom' for manual control",
                ),

                # Group 2: Scene Context
                io.String.Input(
                    "scene_context",
                    multiline=True,
                    default="",
                    tooltip="Scene description. Example: 'modern office hallway with glass walls', 'dense forest with morning mist', 'urban street with tall buildings'",
                ),

                # Group 3: Movement Control (Custom Mode)
                io.Combo.Input(
                    "movement",
                    options=["forward", "backward", "left", "right", "up", "down"],
                    default="forward",
                    tooltip="Movement direction: forward=walk ahead, left/right=strafe, up/down=vertical movement (drone-like)",
                ),
                io.Int.Input(
                    "distance",
                    default=5,
                    min=1,
                    max=20,
                    step=1,
                    tooltip="Movement distance in meters (1-20m). Longer = more dramatic transition",
                ),

                # Group 4: Rotation During Movement
                io.Combo.Input(
                    "rotation",
                    options=["none", "left", "right", "up", "down"],
                    default="none",
                    tooltip="Camera rotation during movement: none=straight, left/right=pan, up/down=tilt. Creates fluid combined movements.",
                ),

                # Group 5: Speed and Focus
                io.Combo.Input(
                    "speed",
                    options=["slow", "normal", "fast", "smooth"],
                    default="smooth",
                    tooltip="Movement speed: slow=deliberate, normal=natural, fast=dynamic, smooth=cinematic",
                ),
                io.Boolean.Input(
                    "maintain_focus",
                    default=False,
                    tooltip="Keep a subject in frame during movement (useful for tracking shots)",
                ),
                io.String.Input(
                    "focus_subject",
                    default="",
                    tooltip="What to keep in frame if maintain_focus is enabled. Example: 'the building', 'the person', 'the product'",
                ),

                # Group 6: Options
                io.Boolean.Input(
                    "debug_mode",
                    default=False,
                    tooltip="Print detailed prompt information to console",
                ),
            ],
            outputs=[
                io.String.Output(
                    "prompt",
                    tooltip="Generated prompt for Qwen Edit 2509",
                ),
                io.String.Output(
                    "preset_description",
                    tooltip="Description of the selected preset (empty for custom)",
                ),
                io.String.Output(
                    "system_prompt",
                    tooltip="⭐ NEW v4.0: Perfect system prompt for this navigation style! Connect to encoder's system_prompt input for optimal results.",
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
        maintain_focus,
        focus_subject,
        debug_mode,
    ) -> io.NodeOutput:
        """Execute the Environment Navigator node.

        Steps:
        1. Apply preset if selected
        2. Build navigation prompt
        3. Output debug info if requested
        4. Return prompt and preset description
        """

        # Step 1: Apply preset or use custom parameters
        user_params = {
            "movement": movement,
            "distance": distance,
            "rotation": rotation,
            "speed": speed,
        }

        final_params = apply_preset(preset, user_params)

        # Extract final values
        final_movement = final_params["movement"]
        final_distance = final_params["distance"]
        final_rotation = final_params["rotation"]
        final_speed = final_params["speed"]

        # Get preset description
        preset_desc = ""
        if preset != "custom":
            preset_desc = NAVIGATION_PRESETS[preset].get("description", "")

        # Get system prompt for this preset (v4.0.0 feature)
        system_prompt = NAVIGATION_SYSTEM_PROMPTS.get(preset, NAVIGATION_SYSTEM_PROMPTS["custom"])

        # Step 2: Build navigation prompt
        prompt = build_navigation_prompt(
            scene_context=scene_context,
            movement=final_movement,
            distance=final_distance,
            rotation=final_rotation,
            speed=final_speed,
            maintain_focus=maintain_focus,
            focus_subject=focus_subject,
        )

        # Step 3: Debug output
        if debug_mode:
            debug_lines = [
                "=" * 70,
                "ArchAi3D_Qwen_Environment_Navigator - Generated Prompt (v4.0.0)",
                "=" * 70,
                f"Preset: {preset}",
                f"Scene Context: {scene_context[:80]}..." if len(scene_context) > 80 else f"Scene Context: {scene_context}",
                f"Movement: {final_movement}",
                f"Distance: {final_distance}m",
                f"Rotation: {final_rotation}",
                f"Speed: {final_speed}",
                f"Maintain Focus: {maintain_focus}",
                f"Focus Subject: {focus_subject}",
                "=" * 70,
                "Generated Prompt:",
                prompt,
                "=" * 70,
                "⭐ System Prompt (NEW v4.0.0):",
                system_prompt,
                "=" * 70,
            ]
            print("\n".join(debug_lines))

        # Step 4: Return (now includes system_prompt)
        return io.NodeOutput(prompt, preset_desc, system_prompt)


# ============================================================================
# EXTENSION REGISTRATION
# ============================================================================

class EnvironmentNavigatorExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [ArchAi3D_Qwen_Environment_Navigator]


async def comfy_entrypoint():
    return EnvironmentNavigatorExtension()
