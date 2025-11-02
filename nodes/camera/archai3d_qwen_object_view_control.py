# -*- coding: utf-8 -*-
"""
ArchAi3D Qwen Object View Control Node (v1)
Author: Amir Ferdos (ArchAi3d)
Email: Amir84ferdos@gmail.com
LinkedIn: https://www.linkedin.com/in/archai3d/

Description:
    Control camera viewing angles for object/product photography.
    Changes viewing direction to show different sides and angles of products.

    Perfect for:
    - Product photography (show from different angles)
    - E-commerce showcases (front, side, top views)
    - Detail shots (close-up angles)
    - Professional product presentations

Based on research:
    - QWEN_PROMPT_GUIDE Camera Tilt function
    - "change the view to [ANGLE_DESCRIPTION]"
    - Natural language angle descriptions
    - Product photography best practices
"""

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

# ============================================================================
# OBJECT VIEW SYSTEM PROMPT - v5.0.0 Research-Validated
# ============================================================================

OBJECT_VIEW_SYSTEM_PROMPT = """You are a product photographer. Change viewing angle to show product from different perspective. Show product from requested direction or height. Keep camera distance same, only rotate view direction. Preserve all product details, materials, textures, colors, and lighting exactly. Maintain professional product photography framing and composition."""

# ============================================================================
# VIEW ANGLE PRESETS - 20 Product Photography Angles (12 original + 8 new)
# ============================================================================

OBJECT_VIEW_PRESETS = {
    "custom": "Manual control of viewing angle",

    # CARDINAL VIEWS (4) - Essential product angles
    "front_view": "Direct front view - main product angle, symmetric composition",
    "side_left_view": "Left side view - reveals product depth and profile",
    "side_right_view": "Right side view - reveals product depth and profile",
    "back_view": "Back view - shows rear details and features",

    # CORNER VIEWS (2) - Dynamic product angles
    "corner_front_left": "Front-left corner view - shows two sides, creates depth",
    "corner_front_right": "Front-right corner view - shows two sides, creates depth",

    # HEIGHT PERSPECTIVES (3) - Elevation angles
    "top_down_view": "Top-down overhead view - bird's eye product perspective",
    "eye_level_view": "Eye level straight-on view - natural human perspective",
    "low_angle_up": "Low angle looking up - emphasizes product presence and height",

    # DETAIL VIEWS (2) - Close-up angles
    "detail_closeup": "Close-up detail view - emphasizes textures and materials",
    "three_quarter_view": "Three-quarter view - classic product photography angle showing front and side",

    # SPECIAL (1)
    "hero_angle": "Hero angle - dramatic showcase angle, optimal product presentation",

    # NEW ANGLES (8) - Enhanced views based on QWEN research + user testing
    "closeup_shot_angle": "⭐ NEW: Closeup shot view - user-tested pattern for tight framing",
    "macro_detail_angle": "⭐ NEW: Macro extreme closeup view - finest texture and material details",
    "wide_contextual_view": "⭐ NEW: Wide contextual view - product with surrounding environment",
    "overhead_flatlay_view": "⭐ NEW: Overhead flat-lay view - perfect for flat lay product photography",
    "ground_level_dramatic": "⭐ NEW: Ground level dramatic view - extreme low angle, heroic presentation",
    "profile_silhouette": "⭐ NEW: Side profile silhouette view - clean profile for design docs",
    "establishing_intro": "⭐ NEW: Establishing intro view - context-setting introduction shot",
    "aerial_birds_eye": "⭐ NEW: Aerial bird's eye view - high overhead perspective",
}

# ============================================================================
# ANGLE DESCRIPTIONS FOR PROMPTS
# ============================================================================

ANGLE_DESCRIPTIONS = {
    # Cardinal views
    "front_view": "direct front view showing the main product face",
    "side_left_view": "left side view showing the product profile",
    "side_right_view": "right side view showing the product profile",
    "back_view": "back view showing the rear of the product",

    # Corner views
    "corner_front_left": "front-left corner view showing two adjacent sides",
    "corner_front_right": "front-right corner view showing two adjacent sides",

    # Height perspectives
    "top_down_view": "top-down overhead view looking directly down at the product",
    "eye_level_view": "eye level view at natural human perspective",
    "low_angle_up": "low angle view looking up at the product from below",

    # Detail views
    "detail_closeup": "close-up detail view emphasizing product textures and materials",
    "three_quarter_view": "three-quarter view showing both front and side at 45-degree angle",

    # Special
    "hero_angle": "hero angle for optimal product showcase and presentation",

    # NEW angles (8) - Based on QWEN research + user testing
    "closeup_shot_angle": "closeup shot view with tight framing",
    "macro_detail_angle": "macro extreme closeup view revealing finest details",
    "wide_contextual_view": "wide shot showing product with surrounding environment",
    "overhead_flatlay_view": "overhead flat-lay view looking directly down",
    "ground_level_dramatic": "ground level dramatic view looking up from below",
    "profile_silhouette": "side profile silhouette view",
    "establishing_intro": "establishing shot view",
    "aerial_birds_eye": "aerial bird's eye view from high above",
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def build_object_view_prompt(view_angle: str, scene_context: str, custom_angle: str) -> str:
    """Build object view angle change prompt.

    Formula from QWEN Guide Function 3:
    [SCENE_CONTEXT], change the view to [ANGLE_DESCRIPTION]

    Args:
        view_angle: Selected preset view angle
        scene_context: Description of the product/object
        custom_angle: Custom angle description (if preset is "custom")

    Returns:
        Complete prompt string
    """
    parts = []

    # Scene context
    if scene_context.strip():
        parts.append(scene_context.strip())

    # View angle
    if view_angle == "custom":
        if custom_angle.strip():
            parts.append(f"change the view to {custom_angle.strip()}")
    else:
        angle_desc = ANGLE_DESCRIPTIONS.get(view_angle, "")
        if angle_desc:
            parts.append(f"change the view to {angle_desc}")

    return ", ".join(parts) if parts else "change the view"


# ============================================================================
# NODE DEFINITION
# ============================================================================

class ArchAi3D_Qwen_Object_View_Control(io.ComfyNode):
    """Object View Control: Change viewing angle for product/object photography.

    This node changes camera viewing direction for product photography:
    - Cardinal views (front, sides, back)
    - Corner views (dynamic two-side angles)
    - Height perspectives (top-down, eye-level, low-angle, ground-level, aerial)
    - Detail views (close-up, macro, three-quarter)
    - Contextual views (wide shot, establishing shot)
    - Hero angles (optimal showcase)
    - Profile views (silhouette)

    Key Features:
    - 20 preset viewing angles for product photography (12 original + 8 NEW)
    - Research-validated angle descriptions + user-tested patterns
    - Camera position stays same (only rotation changes)
    - Professional e-commerce quality
    - Automatic system prompt output

    Perfect For:
    - E-commerce product photography
    - Product catalogs (multiple angle showcases)
    - Amazon/Shopify listings (professional views)
    - 3D product visualization
    - Marketing materials

    Based on research:
    - Camera Tilt function (QWEN_PROMPT_GUIDE.md)
    - "change the view to [ANGLE_DESCRIPTION]"
    - Product photography best practices
    - E-commerce standards

    USE THIS NODE WHEN:
    - You have a product/object in the image
    - You want to change viewing ANGLE (not position)
    - Need professional product photography angles

    SCENE TYPE: Object/Product/Item
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ArchAi3D_Qwen_Object_View_Control",
            category="ArchAi3d/Camera/Object",
            inputs=[
                # Group 1: View Angle Selection
                io.Combo.Input(
                    "preset",
                    options=list(OBJECT_VIEW_PRESETS.keys()),
                    default="three_quarter_view",
                    tooltip="Select product viewing angle. Three-quarter view is classic product photography angle."
                ),

                # Group 2: Scene Context
                io.String.Input(
                    "scene_context",
                    multiline=True,
                    default="",
                    tooltip="Optional: Product description. Example: 'wooden chair', 'coffee mug on white background'. Improves consistency."
                ),

                # Group 3: Custom Angle (Custom Mode)
                io.String.Input(
                    "custom_angle",
                    default="",
                    tooltip="Custom angle description. Example: 'angled view emphasizing the handle', 'view showing the logo'"
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
                    "view_description",
                    tooltip="Description of the selected viewing angle"
                ),
                io.String.Output(
                    "system_prompt",
                    tooltip="⭐ v5.0: Research-validated system prompt for object view control! Connect to encoder's system_prompt input."
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        preset,
        scene_context,
        custom_angle,
        debug_mode,
    ) -> io.NodeOutput:
        """Execute the Object View Control node.

        Steps:
        1. Get preset description
        2. Build view change prompt
        3. Get system prompt
        4. Debug output if requested
        5. Return outputs
        """

        # Step 1: Get preset description
        preset_desc = OBJECT_VIEW_PRESETS.get(preset, "")

        # Step 2: Build view change prompt
        prompt = build_object_view_prompt(
            view_angle=preset,
            scene_context=scene_context,
            custom_angle=custom_angle,
        )

        # Step 3: Get system prompt (v5.0.0 feature - research-validated)
        system_prompt = OBJECT_VIEW_SYSTEM_PROMPT

        # Step 4: Debug output
        if debug_mode:
            debug_lines = [
                "=" * 70,
                "ArchAi3D_Qwen_Object_View_Control - Generated Prompt (v5.0.0)",
                "=" * 70,
                f"Preset: {preset}",
                f"Description: {preset_desc}",
                f"Scene Context: {scene_context[:80]}..." if len(scene_context) > 80 else f"Scene Context: {scene_context}",
                f"Custom Angle: {custom_angle}" if preset == "custom" else "",
                "=" * 70,
                "Generated Prompt:",
                prompt,
                "=" * 70,
                "⭐ System Prompt (v5.0.0 - Research-Validated):",
                system_prompt,
                "=" * 70,
                "",
                "💡 TIP: For best results:",
                "  - Use this node to CHANGE VIEW ANGLE only",
                "  - For MOVING closer to product, use Object_Position_Control",
                "  - For ROTATING product (turntable), use Object_Rotation_Control",
                "  - Three-quarter view is the gold standard for product photography",
                "=" * 70,
            ]
            print("\n".join(debug_lines))

        # Step 5: Return
        return io.NodeOutput(prompt, preset_desc, system_prompt)


# ============================================================================
# EXTENSION REGISTRATION
# ============================================================================

class ObjectViewControlExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [ArchAi3D_Qwen_Object_View_Control]


async def comfy_entrypoint():
    return ObjectViewControlExtension()
