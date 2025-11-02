"""
ArchAi3D Qwen Scene Photographer Node
Author: Amir Ferdos (ArchAi3d)
Email: Amir84ferdos@gmail.com
LinkedIn: https://www.linkedin.com/in/archai3d/

Description:
    Move the photographer/camera to a specific position in the scene to frame
    a target subject from a desired angle. Perfect for:
    - Interior design: "move to the right side of room to frame the sofa"
    - Product photography: "move 2m in front of the watch to capture details"
    - Architectural: "move to ground level in front of the building looking up"

Based on research from 7 PDF files about Qwen's capabilities.
Uses natural language positioning (NOT pixel coordinates - not supported by Qwen).
"""

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override
from typing import Dict, List, Tuple

# ============================================================================
# CONSTANTS - Height and tilt mappings
# ============================================================================

HEIGHT_MAP = {
    "ground_level": "at ground level",
    "lower": "lower",
    "same": "new",
    "higher": "higher",
    "face_level": "at face level",
}

TILT_MAP = {
    "none": "",
    "up_slightly": "camera tilted up slightly",
    "down_slightly": "camera tilted down slightly",
    "way_up": "camera tilted way up towards the ceiling",
    "way_down": "camera aiming downwards",
}

# ============================================================================
# VIEWPOINT SYSTEM PROMPTS - v4.0.0 Feature
# ============================================================================

VIEWPOINT_SYSTEM_PROMPTS = {
    "custom": "You are a photographer. Position camera at specific vantage point in scene. Frame target subject from new angle. Preserve scene details and natural perspective throughout position change.",

    # Product Photography
    "product_front": "You are a product photographer. Position camera directly in front of product at eye level for professional showcase. Natural perspective, clear view of all product features. Maintain proper framing and lighting. Preserve product details and background consistency.",

    "product_hero_low": "You are a product photographer. Position camera at ground level in front of product creating dramatic hero shot. Low angle perspective emphasizing power and presence. Maintain product details and professional lighting. Preserve all features from heroic viewpoint.",

    "product_overhead": "You are a product photographer. Position camera directly above product for overhead flat lay perspective. Top-down view with straight-down angle. Professional product photography standards. Preserve all product features and flat lay composition.",

    # Interior Design
    "room_corner_wide": "You are an interior photographer. Position camera in corner of room to capture entire room layout. Wide perspective showing spatial relationships. Professional interior photography standards. Preserve room proportions, architectural details, and design elements.",

    "room_opposite_wall": "You are an interior photographer. Position camera on opposite wall for full room perspective. Comprehensive view of entire space from facing wall. Professional interior photography quality. Preserve room layout, furniture arrangement, and architectural features.",

    "room_ceiling_view": "You are an interior photographer. Position camera elevated above room with downward angle showing layout from above. High angle perspective revealing room organization. Professional interior documentation. Preserve spatial relationships and room design elements.",

    # Architectural Exterior
    "building_ground_up": "You are an architectural photographer. Position camera at ground level in front of building with camera tilted way up. Dramatic upward perspective emphasizing building height and scale. Professional architectural photography. Preserve facade details and architectural features throughout upward view.",

    "building_elevated": "You are an architectural photographer. Position camera at elevated viewpoint to right of building with slight downward tilt. Reveal building in context of surroundings. Professional architectural documentation standards. Preserve building details, proportions, and site context.",

    # Food Photography
    "food_45_angle": "You are a food photographer. Position camera at 45-degree angle above food with slight downward tilt. Classic appetizing perspective for food photography. Professional food styling standards. Preserve food details, textures, and presentation throughout angled view.",

    "food_overhead": "You are a food photographer. Position camera directly above food for overhead flat lay perspective. Instagram-style top-down food photography. Professional food styling quality. Preserve all food details, plating, and composition from overhead viewpoint.",

    # Portrait/Fashion
    "fashion_eye_level": "You are a fashion photographer. Position camera directly in front of subject at face level. Natural, approachable eye-level perspective. Professional fashion photography standards. Preserve subject's features, styling, and natural proportions throughout eye-level framing.",

    "fashion_low_power": "You are a fashion photographer. Position camera at lower height in front of subject with slight upward tilt. Create powerful, dominant perspective through low angle. Professional fashion photography quality. Preserve subject's features and styling while emphasizing commanding presence.",

    # Landscape/Environment
    "landscape_wide": "You are a landscape photographer. Position camera at distance from scene for wide expansive perspective. Comprehensive environmental view. Professional landscape photography standards. Preserve natural features, depth, and atmospheric perspective throughout wide composition.",

    "landscape_foreground": "You are a landscape photographer. Position camera at ground level in front of scene with slight upward tilt. Low angle perspective with foreground interest. Professional landscape photography quality. Preserve foreground details, depth layers, and natural features throughout composition.",
}

# ============================================================================
# PRESET VIEWPOINTS - Common photography positions
# ============================================================================

VIEWPOINT_PRESETS = {
    # Product Photography
    "product_front": {
        "description": "Front view at eye level for product showcase",
        "direction": "front",
        "distance": 2,
        "height": "face_level",
        "tilt": "none",
        "auto_facing": True,
    },
    "product_hero_low": {
        "description": "Low angle hero shot (dramatic, powerful look)",
        "direction": "front",
        "distance": 2,
        "height": "ground_level",
        "tilt": "up_slightly",
        "auto_facing": True,
    },
    "product_overhead": {
        "description": "Overhead flat lay view (top-down)",
        "direction": "front",
        "distance": 1,
        "height": "higher",
        "tilt": "way_down",
        "auto_facing": True,
    },

    # Interior Design
    "room_corner_wide": {
        "description": "Corner view capturing entire room layout",
        "direction": "left side of room",
        "distance": 8,
        "height": "face_level",
        "tilt": "none",
        "auto_facing": False,
    },
    "room_opposite_wall": {
        "description": "View from opposite wall (full room perspective)",
        "direction": "back",
        "distance": 10,
        "height": "face_level",
        "tilt": "none",
        "auto_facing": False,
    },
    "room_ceiling_view": {
        "description": "High angle showing room from above",
        "direction": "back",
        "distance": 6,
        "height": "higher",
        "tilt": "way_down",
        "auto_facing": False,
    },

    # Architectural Exterior
    "building_ground_up": {
        "description": "Ground level looking up (emphasize height)",
        "direction": "front",
        "distance": 3,
        "height": "ground_level",
        "tilt": "way_up",
        "auto_facing": True,
    },
    "building_elevated": {
        "description": "Elevated view showing context",
        "direction": "right",
        "distance": 10,
        "height": "higher",
        "tilt": "down_slightly",
        "auto_facing": True,
    },

    # Food Photography
    "food_45_angle": {
        "description": "Classic 45° angle for food (most appetizing)",
        "direction": "front",
        "distance": 1,
        "height": "higher",
        "tilt": "down_slightly",
        "auto_facing": True,
    },
    "food_overhead": {
        "description": "Overhead flat lay (Instagram style)",
        "direction": "front",
        "distance": 1,
        "height": "higher",
        "tilt": "way_down",
        "auto_facing": True,
    },

    # Portrait/Fashion
    "fashion_eye_level": {
        "description": "Eye level for natural, approachable feel",
        "direction": "front",
        "distance": 3,
        "height": "face_level",
        "tilt": "none",
        "auto_facing": True,
    },
    "fashion_low_power": {
        "description": "Low angle for powerful, dominant look",
        "direction": "front",
        "distance": 3,
        "height": "lower",
        "tilt": "up_slightly",
        "auto_facing": True,
    },

    # Landscape/Environment
    "landscape_wide": {
        "description": "Wide view for expansive landscape",
        "direction": "back",
        "distance": 15,
        "height": "face_level",
        "tilt": "none",
        "auto_facing": False,
    },
    "landscape_foreground": {
        "description": "Low angle with foreground interest",
        "direction": "front",
        "distance": 5,
        "height": "ground_level",
        "tilt": "up_slightly",
        "auto_facing": False,
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

    preset = VIEWPOINT_PRESETS.get(preset_name, {})

    # Merge: preset defaults + user overrides
    result = preset.copy()
    result.update({k: v for k, v in user_params.items() if v is not None})

    return result


def build_vantage_point_prompt(
    scene_context: str,
    target_subject: str,
    direction: str,
    distance: int,
    height: str,
    tilt: str,
    auto_facing: bool,
) -> str:
    """Build Scene Photographer vantage point prompt.

    This implements the formula from QWEN_PROMPT_GUIDE.md:
    [SCENE_CONTEXT], change the view to a vantage point [HEIGHT] [DISTANCE]m
    to the [DIRECTION] [FACING] [OPTIONAL_TILT]

    Args:
        scene_context: Description of the environment
        target_subject: What to frame in the shot
        direction: Which direction to move
        distance: How many meters to move
        height: Camera height position
        tilt: Camera tilt angle
        auto_facing: Whether to automatically face the target

    Returns:
        Complete prompt string
    """
    parts = []

    # Scene context
    if scene_context.strip():
        parts.append(scene_context.strip())

    # Build vantage point phrase
    height_desc = HEIGHT_MAP.get(height, "new")
    vantage = f"change the view to a {height_desc} vantage point"

    # Distance and direction
    vantage += f" {distance}m to the {direction}"

    # Auto-facing the target subject
    if auto_facing and target_subject.strip():
        vantage += f" facing {target_subject}"

    # Optional tilt
    tilt_phrase = TILT_MAP.get(tilt, "")
    if tilt_phrase:
        vantage += f" {tilt_phrase}"

    parts.append(vantage)

    return ", ".join(parts)


def build_debug_info(
    preset_name: str,
    scene_context: str,
    target_subject: str,
    direction: str,
    distance: int,
    height: str,
    tilt: str,
    auto_facing: bool,
    prompt: str,
    system_prompt: str,
) -> str:
    """Build formatted debug information string.

    Args:
        All prompt parameters plus the generated prompt and system prompt

    Returns:
        Formatted debug string for console output
    """
    debug_lines = [
        "=" * 70,
        "ArchAi3D_Qwen_Scene_Photographer - Generated Prompt (v4.0.0)",
        "=" * 70,
        f"Preset: {preset_name}",
        f"Scene Context: {scene_context[:80]}..." if len(scene_context) > 80 else f"Scene Context: {scene_context}",
        f"Target Subject: {target_subject}",
        f"Direction: {direction}",
        f"Distance: {distance}m",
        f"Height: {height}",
        f"Tilt: {tilt}",
        f"Auto-Facing: {auto_facing}",
        "=" * 70,
        "Generated Prompt:",
        prompt,
        "=" * 70,
        "⭐ System Prompt (NEW v4.0.0):",
        system_prompt,
        "=" * 70,
    ]

    return "\n".join(debug_lines)


# ============================================================================
# NODE DEFINITION
# ============================================================================

class ArchAi3D_Qwen_Scene_Photographer(io.ComfyNode):
    """Scene Photographer: Position camera to frame specific subjects.

    This node solves your requirement: "move in a scene and take photo from
    different angles" and "go in front of some object in the scene and take
    a photo with that subject in front of camera view".

    Key Features:
    - Natural language positioning (Qwen doesn't support pixel coordinates)
    - 14 preset viewpoints for common photography scenarios
    - Precise control: direction, distance, height, tilt
    - Auto-facing mode to always frame your target subject
    - Works great with scene context for consistency

    Perfect For:
    - Interior design: Frame furniture from specific angles
    - Product photography: Hero shots, overhead, low angle
    - Architecture: Ground-up views, elevated perspectives
    - Food photography: 45° angle, overhead flat lay
    - Fashion: Eye level, low angle power shots
    - Landscape: Wide views, foreground interest

    Based on research finding: Distance-based natural language works best.
    "5m to the right facing the sofa" (✅) NOT pixel coordinates (❌)
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ArchAi3D_Qwen_Scene_Photographer",
            category="ArchAi3d/Qwen/Camera",
            inputs=[
                # Group 1: Preset Selection
                io.Combo.Input(
                    "preset",
                    options=["custom"] + list(VIEWPOINT_PRESETS.keys()),
                    default="custom",
                    tooltip="Select a preset viewpoint or use 'custom' for manual control. Presets provide professional photography positions.",
                ),

                # Group 2: Scene and Subject
                io.String.Input(
                    "scene_context",
                    multiline=True,
                    default="",
                    tooltip="Optional scene description for consistency. Example: 'modern kitchen with marble countertop and stainless appliances'",
                ),
                io.String.Input(
                    "target_subject",
                    default="",
                    tooltip="What to frame in the shot. Example: 'the espresso machine on the counter', 'the leather sofa', 'the building facade'",
                ),

                # Group 3: Position Control (Custom Mode)
                io.Combo.Input(
                    "direction",
                    options=[
                        "front",
                        "back",
                        "left",
                        "right",
                        "left side of room",
                        "right side of room",
                    ],
                    default="front",
                    tooltip="Which direction to move. 'left/right' = picture left/right (NOT subject's left/right)",
                ),
                io.Int.Input(
                    "distance",
                    default=3,
                    min=1,
                    max=20,
                    step=1,
                    tooltip="How many meters to move in that direction (1-20m). Closer = detail, farther = context",
                ),

                # Group 4: Height and Angle
                io.Combo.Input(
                    "height",
                    options=[
                        "ground_level",
                        "lower",
                        "same",
                        "higher",
                        "face_level",
                    ],
                    default="face_level",
                    tooltip="Camera height: ground_level=dramatic low, face_level=natural, higher=overview",
                ),
                io.Combo.Input(
                    "tilt",
                    options=[
                        "none",
                        "up_slightly",
                        "down_slightly",
                        "way_up",
                        "way_down",
                    ],
                    default="none",
                    tooltip="Camera tilt: none=straight ahead, way_up=ceiling/sky, way_down=bird's eye",
                ),

                # Group 5: Options
                io.Boolean.Input(
                    "auto_facing",
                    default=True,
                    tooltip="Automatically face the target subject. Turn off for landscape/wide shots where you want to capture the general view.",
                ),
                io.Boolean.Input(
                    "debug_mode",
                    default=False,
                    tooltip="Print detailed prompt information to console for debugging",
                ),
            ],
            outputs=[
                io.String.Output(
                    "prompt",
                    tooltip="Generated prompt for Qwen Edit 2509. Connect to your Qwen node.",
                ),
                io.String.Output(
                    "preset_description",
                    tooltip="Description of the selected preset (empty for custom)",
                ),
                io.String.Output(
                    "system_prompt",
                    tooltip="⭐ NEW v4.0: Perfect system prompt for this viewpoint! Connect to encoder's system_prompt input for optimal results.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        preset,
        scene_context,
        target_subject,
        direction,
        distance,
        height,
        tilt,
        auto_facing,
        debug_mode,
    ) -> io.NodeOutput:
        """Execute the Scene Photographer node.

        Steps:
        1. Apply preset if selected (preset overrides manual settings)
        2. Build vantage point prompt using formula
        3. Output debug info if requested
        4. Return prompt and preset description
        """

        # Step 1: Apply preset or use custom parameters
        user_params = {
            "direction": direction,
            "distance": distance,
            "height": height,
            "tilt": tilt,
            "auto_facing": auto_facing,
        }

        final_params = apply_preset(preset, user_params)

        # Extract final values
        final_direction = final_params["direction"]
        final_distance = final_params["distance"]
        final_height = final_params["height"]
        final_tilt = final_params["tilt"]
        final_auto_facing = final_params["auto_facing"]

        # Get preset description
        preset_desc = ""
        if preset != "custom":
            preset_desc = VIEWPOINT_PRESETS[preset].get("description", "")

        # Get system prompt for this viewpoint (v4.0.0 feature)
        system_prompt = VIEWPOINT_SYSTEM_PROMPTS.get(preset, VIEWPOINT_SYSTEM_PROMPTS["custom"])

        # Step 2: Build vantage point prompt
        prompt = build_vantage_point_prompt(
            scene_context=scene_context,
            target_subject=target_subject,
            direction=final_direction,
            distance=final_distance,
            height=final_height,
            tilt=final_tilt,
            auto_facing=final_auto_facing,
        )

        # Step 3: Debug output
        if debug_mode:
            debug_info = build_debug_info(
                preset_name=preset,
                scene_context=scene_context,
                target_subject=target_subject,
                direction=final_direction,
                distance=final_distance,
                height=final_height,
                tilt=final_tilt,
                auto_facing=final_auto_facing,
                prompt=prompt,
                system_prompt=system_prompt,
            )
            print(debug_info)

        # Step 4: Return (now includes system_prompt)
        return io.NodeOutput(prompt, preset_desc, system_prompt)


# ============================================================================
# EXTENSION REGISTRATION
# ============================================================================

class ScenePhotographerExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [ArchAi3D_Qwen_Scene_Photographer]


async def comfy_entrypoint():
    return ScenePhotographerExtension()
