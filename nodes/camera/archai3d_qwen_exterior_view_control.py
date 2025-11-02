# -*- coding: utf-8 -*-
"""
ArchAi3D Qwen Exterior View Control Node (v1)
Author: Amir Ferdos (ArchAi3d)
Email: Amir84ferdos@gmail.com
LinkedIn: https://www.linkedin.com/in/archai3d/

Description:
    Control camera viewing angles for exterior scenes (buildings, outdoor spaces, architecture).
    Changes viewing direction without moving camera position.

    Perfect for:
    - Building facades (front, side, corner views)
    - Outdoor architectural photography
    - Elevated/ground perspectives
    - Aerial and street-level views

Based on research:
    - Qwen PROMPT_GUIDE (Camera Tilt + FOV functions)
    - Community findings: Distance-based > degree-based
    - Natural language positioning works best
    - Scene context improves consistency
"""

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

# ============================================================================
# EXTERIOR VIEW SYSTEM PROMPT - v5.0.0 Research-Validated
# ============================================================================

EXTERIOR_VIEW_SYSTEM_PROMPT = """You are an architectural photographer. Change viewing angle of exterior building or outdoor scene. Show architecture from different direction or height. Keep camera position same, only rotate view direction. Preserve all building details, architectural proportions, facade features, outdoor furniture, outdoor lighting, sky, and environmental context exactly. Maintain natural outdoor perspective and spatial relationships."""

# ============================================================================
# VIEW ANGLE PRESETS - 20 Exterior Photography Angles (12 original + 8 NEW)
# ============================================================================

EXTERIOR_VIEW_PRESETS = {
    "custom": {
        "description": "Manual control of viewing angle",
        "angle": "",
        "camera_position_note": "Current position maintained"
    },

    # FACADE VIEWS (4)
    "facade_frontal": {
        "description": "Direct front facade view - symmetric, formal",
        "angle": "front facade view",
        "camera_position_note": "Face building directly from front"
    },
    "facade_side_left": {
        "description": "Left side facade view - reveals side details",
        "angle": "left side facade view",
        "camera_position_note": "View from left side of building"
    },
    "facade_side_right": {
        "description": "Right side facade view - reveals side details",
        "angle": "right side facade view",
        "camera_position_note": "View from right side of building"
    },
    "facade_rear": {
        "description": "Back of building - rear facade view",
        "angle": "rear facade view",
        "camera_position_note": "View from back of building"
    },

    # CORNER VIEWS (2)
    "corner_view_left": {
        "description": "Corner view showing two facades (left corner) - creates depth",
        "angle": "corner view showing front and left facades",
        "camera_position_note": "45° angle from left corner"
    },
    "corner_view_right": {
        "description": "Corner view showing two facades (right corner) - creates depth",
        "angle": "corner view showing front and right facades",
        "camera_position_note": "45° angle from right corner"
    },

    # HEIGHT PERSPECTIVES (3)
    "ground_level_up": {
        "description": "Ground level looking up - emphasizes height and monumentality",
        "angle": "ground level view looking up",
        "camera_position_note": "Camera at ground level, tilted way up"
    },
    "elevated_down": {
        "description": "Elevated view looking down - shows context and surroundings",
        "angle": "elevated view looking down",
        "camera_position_note": "Camera elevated above, tilted slightly down"
    },
    "aerial_top_down": {
        "description": "Aerial drone view from directly above - bird's eye perspective",
        "angle": "aerial top-down view",
        "camera_position_note": "Drone positioned directly overhead"
    },

    # STREET VIEWS (2)
    "street_level": {
        "description": "Human eye height street view - natural pedestrian perspective",
        "angle": "street level view at eye height",
        "camera_position_note": "5.5 feet height, natural human perspective"
    },
    "entrance_frontal": {
        "description": "Direct entrance view - inviting, welcoming perspective",
        "angle": "direct entrance view",
        "camera_position_note": "Facing main entrance straight-on"
    },

    # NEW ANGLES (8) - Enhanced views based on QWEN research + user testing
    "closeup_shot_angle": {
        "description": "⭐ NEW: Closeup shot view - tight framing of building details",
        "angle": "closeup shot view with tight framing of building details",
        "camera_position_note": "Close framing showing architectural details"
    },
    "macro_detail_angle": {
        "description": "⭐ NEW: Macro extreme closeup - finest architectural textures and materials",
        "angle": "macro extreme closeup view revealing finest architectural textures and materials",
        "camera_position_note": "Extreme close view of textures and materials"
    },
    "wide_contextual_view": {
        "description": "⭐ NEW: Wide contextual view - complete building with full surroundings",
        "angle": "wide shot showing complete building with full surroundings and context",
        "camera_position_note": "Wide angle showing full context"
    },
    "overhead_birds_eye": {
        "description": "⭐ NEW: Overhead bird's eye view - looking straight down at building",
        "angle": "overhead bird's eye view looking straight down at building",
        "camera_position_note": "Directly overhead looking down"
    },
    "ground_level_dramatic": {
        "description": "⭐ NEW: Ground level dramatic view - looking up from ground level",
        "angle": "ground level dramatic view looking up from ground level",
        "camera_position_note": "Ground level with dramatic upward angle"
    },
    "profile_silhouette": {
        "description": "⭐ NEW: Side profile view - clean building silhouette and elevation",
        "angle": "side profile view showing clean building silhouette and elevation",
        "camera_position_note": "Pure side profile view"
    },
    "establishing_intro": {
        "description": "⭐ NEW: Establishing intro view - building introduction and context",
        "angle": "establishing shot view introducing building and context",
        "camera_position_note": "Establishing shot composition"
    },
    "aerial_high_angle": {
        "description": "⭐ NEW: Aerial high angle view - elevated perspective of entire building",
        "angle": "aerial high angle view from elevated perspective of entire building",
        "camera_position_note": "High aerial vantage point"
    },
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def build_exterior_view_prompt(
    view_angle: str,
    scene_context: str,
) -> str:
    """Build exterior view angle change prompt.

    Formula from research:
    [SCENE_CONTEXT], change the view to [VIEW_ANGLE]

    Args:
        view_angle: The viewing angle description
        scene_context: Optional scene description

    Returns:
        Complete prompt string
    """
    parts = []

    if scene_context.strip():
        parts.append(scene_context.strip())

    if view_angle:
        parts.append(f"change the view to {view_angle}")

    return ", ".join(parts) if parts else "change the view"


# ============================================================================
# NODE DEFINITION
# ============================================================================

class ArchAi3D_Qwen_Exterior_View_Control(io.ComfyNode):
    """Exterior View Control: Change camera viewing angle for exterior scenes.

    This node changes viewing direction for outdoor/architectural scenes:
    - Building facades (front, side, corner, rear)
    - Height perspectives (ground level up, elevated down, aerial)
    - Street-level and entrance views
    - Detail views (closeup, macro, wide contextual)
    - Dramatic angles (ground-level, aerial, establishing)
    - All while keeping camera position the same

    Key Features:
    - 20 preset viewing angles for exterior photography (12 original + 8 NEW)
    - Research-validated prompts from Qwen documentation
    - Automatic system prompt output for perfect results
    - Scene context support for consistency

    Perfect For:
    - Architectural photography (building facades)
    - Real estate marketing (showcase different angles)
    - Outdoor furniture displays (view from different sides)
    - Site documentation (comprehensive views)

    Based on research:
    - Camera tilt function (QWEN_PROMPT_GUIDE.md)
    - Natural language > pixel coordinates
    - Scene context improves consistency
    - Architectural photographer persona in system prompt

    USE THIS NODE WHEN:
    - You have an outdoor/exterior scene
    - You want to change viewing angle (not position)
    - Building or outdoor furniture is the subject

    SCENE TYPE: Exterior/Outdoor/Architectural
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ArchAi3D_Qwen_Exterior_View_Control",
            category="ArchAi3d/Camera/Exterior",
            inputs=[
                # Group 1: View Angle Selection
                io.Combo.Input(
                    "preset",
                    options=list(EXTERIOR_VIEW_PRESETS.keys()),
                    default="facade_frontal",
                    tooltip="Select viewing angle preset. Changes view direction while keeping camera position same."
                ),

                # Group 2: Scene Context
                io.String.Input(
                    "scene_context",
                    multiline=True,
                    default="",
                    tooltip="Optional: Scene description. Example: 'modern glass office building', 'outdoor deck with furniture'. Improves consistency."
                ),

                # Group 3: Options
                io.Boolean.Input(
                    "debug_mode",
                    default=False,
                    tooltip="Print generated prompt to console for debugging"
                ),
            ],
            outputs=[
                io.String.Output(
                    "prompt",
                    tooltip="Generated prompt for Qwen Edit 2509"
                ),
                io.String.Output(
                    "view_description",
                    tooltip="Description of the selected view angle"
                ),
                io.String.Output(
                    "system_prompt",
                    tooltip="⭐ v5.0: Research-validated system prompt for exterior view changes! Connect to encoder's system_prompt input."
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        preset,
        scene_context,
        debug_mode,
    ) -> io.NodeOutput:
        """Execute the Exterior View Control node.

        Steps:
        1. Get preset view angle data
        2. Build view change prompt
        3. Get system prompt
        4. Debug output if requested
        5. Return outputs
        """

        # Step 1: Get preset data
        preset_data = EXTERIOR_VIEW_PRESETS.get(preset, EXTERIOR_VIEW_PRESETS["custom"])
        view_description = preset_data["description"]
        view_angle = preset_data["angle"]

        # Step 2: Build prompt
        prompt = build_exterior_view_prompt(
            view_angle=view_angle,
            scene_context=scene_context,
        )

        # Step 3: Get system prompt (v5.0.0 feature - research-validated)
        system_prompt = EXTERIOR_VIEW_SYSTEM_PROMPT

        # Step 4: Debug output
        if debug_mode:
            debug_lines = [
                "=" * 70,
                "ArchAi3D_Qwen_Exterior_View_Control - Generated Prompt (v5.0.0)",
                "=" * 70,
                f"Preset: {preset}",
                f"Description: {view_description}",
                f"View Angle: {view_angle}",
                f"Scene Context: {scene_context[:80]}..." if len(scene_context) > 80 else f"Scene Context: {scene_context}",
                "=" * 70,
                "Generated Prompt:",
                prompt,
                "=" * 70,
                "⭐ System Prompt (v5.0.0 - Research-Validated):",
                system_prompt,
                "=" * 70,
                "",
                "💡 TIP: For best results:",
                "  - Use this node when you want to change viewing ANGLE",
                "  - Camera position stays same, only view direction changes",
                "  - For MOVING to new position, use Exterior_Navigation instead",
                "  - For FOCUSING on outdoor furniture, use Exterior_Focus instead",
                "=" * 70,
            ]
            print("\n".join(debug_lines))

        # Step 5: Return
        return io.NodeOutput(prompt, view_description, system_prompt)


# ============================================================================
# EXTENSION REGISTRATION
# ============================================================================

class ExteriorViewControlExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [ArchAi3D_Qwen_Exterior_View_Control]


async def comfy_entrypoint():
    return ExteriorViewControlExtension()
