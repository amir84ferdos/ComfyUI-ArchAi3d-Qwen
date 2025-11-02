"""
ArchAi3D Qwen Style Transfer Node
Author: Amir Ferdos (ArchAi3d)
Email: Amir84ferdos@gmail.com
LinkedIn: https://www.linkedin.com/in/archai3d/

Description:
    Apply artistic style transfers to specific objects or areas. Perfect for:
    - Creative product photography (ice sculpture effect, fluffy texture)
    - Artistic interior design visualization
    - Social media content with unique styles
    - Conceptual design exploration

Based on research from Qwen-repo3 (WanX API local stylization):
- 8 styles available: ice, cloud, chinese festive lantern, wooden,
  blue and white porcelain, fluffy, weaving, balloon
- Prompt pattern: "Change [object] to [style] style"
"""

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

# ============================================================================
# STYLE TRANSFER SYSTEM PROMPT - v4.0.0 Feature
# ============================================================================

STYLE_TRANSFER_SYSTEM_PROMPT = "You are a creative visual artist. Apply specified artistic style to target object only using local stylization technique. Transform object's appearance, texture, and material to match artistic style completely. Preserve all other scene elements, background, lighting, and composition exactly. Maintain object's form and position while changing its visual style. Create believable stylistic transformation with appropriate texture, color, and material properties."

# ============================================================================
# STYLE PRESETS - 8 local stylization styles from WanX API
# ============================================================================

STYLE_MAP = {
    "ice": {
        "description": "Ice sculpture effect (frozen, crystalline, transparent)",
        "keyword": "ice",
    },
    "cloud": {
        "description": "Cloud effect (soft, fluffy, ethereal white)",
        "keyword": "cloud",
    },
    "chinese_lantern": {
        "description": "Chinese festive lantern (red, decorative, glowing)",
        "keyword": "chinese festive lantern",
    },
    "wooden": {
        "description": "Wooden texture (natural wood grain, warm tones)",
        "keyword": "wooden",
    },
    "blue_white_porcelain": {
        "description": "Blue and white porcelain (traditional Chinese ceramic)",
        "keyword": "blue and white porcelain",
    },
    "fluffy": {
        "description": "Fluffy texture (soft, cotton-like, plush)",
        "keyword": "fluffy",
    },
    "weaving": {
        "description": "Woven yarn texture (knitted, textile, fabric)",
        "keyword": "weaving",
    },
    "balloon": {
        "description": "Balloon effect (inflated, shiny, colorful)",
        "keyword": "balloon",
    },
}

# Common objects for style transfer
COMMON_OBJECTS_STYLE = [
    "the house",
    "the building",
    "the car",
    "the chair",
    "the table",
    "the vase",
    "the sculpture",
    "the product",
    "the object",
    "the tree",
    "the flower",
    "the background",
]

# ============================================================================
# NODE DEFINITION
# ============================================================================

class ArchAi3D_Qwen_Style_Transfer(io.ComfyNode):
    """Style Transfer: Apply artistic styles to objects (local stylization).

    This node applies 8 unique artistic styles to specific objects or areas:
    - Ice: Frozen crystalline sculpture effect
    - Cloud: Soft ethereal white cloud texture
    - Chinese Lantern: Red decorative festive lantern
    - Wooden: Natural wood grain and texture
    - Blue & White Porcelain: Traditional Chinese ceramic style
    - Fluffy: Soft cotton-like plush texture
    - Weaving: Knitted yarn textile effect
    - Balloon: Inflated shiny colorful balloon

    Key Features:
    - 8 professional artistic styles
    - Targets specific objects (not entire image)
    - 12 common object quick-select options
    - Scene context support for consistency
    - Based on official Qwen WanX API

    Perfect For:
    - Creative product visualization
    - Artistic interior design concepts
    - Social media unique content
    - Conceptual design exploration
    - Playful brand imagery

    Based on research: "Change [object] to [style] style" pattern.
    This is LOCAL stylization (specific object), not global (entire image).
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ArchAi3D_Qwen_Style_Transfer",
            category="ArchAi3d/Qwen/Editing",
            inputs=[
                # Group 1: Object Selection
                io.Combo.Input(
                    "object_preset",
                    options=["custom"] + COMMON_OBJECTS_STYLE,
                    default="the house",
                    tooltip="Select common object or use 'custom'. The style will be applied to this object only.",
                ),
                io.String.Input(
                    "custom_object",
                    default="",
                    tooltip="Custom object description (used if object_preset is 'custom'). Example: 'the coffee mug on the table'",
                ),

                # Group 2: Style Selection
                io.Combo.Input(
                    "style",
                    options=list(STYLE_MAP.keys()),
                    default="ice",
                    tooltip="Select artistic style to apply to the object",
                ),

                # Group 3: Scene Context
                io.String.Input(
                    "scene_context",
                    multiline=True,
                    default="",
                    tooltip="Optional scene description. Example: 'modern architectural exterior with glass facade'",
                ),

                # Group 4: Options
                io.Boolean.Input(
                    "debug_mode",
                    default=False,
                    tooltip="Print generated prompt to console",
                ),
            ],
            outputs=[
                io.String.Output(
                    "prompt",
                    tooltip="Generated prompt for Qwen Edit 2509",
                ),
                io.String.Output(
                    "style_description",
                    tooltip="Description of the selected style",
                ),
                io.String.Output(
                    "system_prompt",
                    tooltip="⭐ NEW v4.0: Perfect system prompt for style transfer! Connect to encoder's system_prompt input for optimal results.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        object_preset,
        custom_object,
        style,
        scene_context,
        debug_mode,
    ) -> io.NodeOutput:
        """Execute the Style Transfer node.

        Steps:
        1. Determine object (preset or custom)
        2. Get style keyword and description
        3. Build style transfer prompt
        4. Output debug info if requested
        5. Return prompt and style description
        """

        # Step 1: Determine object
        if object_preset == "custom":
            object_to_style = custom_object.strip() if custom_object.strip() else "the object"
        else:
            object_to_style = object_preset

        # Step 2: Get style data
        style_data = STYLE_MAP.get(style, STYLE_MAP["ice"])
        style_keyword = style_data["keyword"]
        style_description = style_data["description"]

        # Step 3: Build prompt
        # Pattern from research: "Change [object] to [style] style"
        prompt = f"Change {object_to_style} to {style_keyword} style"

        # Add scene context if provided
        if scene_context.strip():
            prompt = f"{scene_context.strip()}, {prompt}"

        # Get system prompt (v4.0.0 feature)
        system_prompt = STYLE_TRANSFER_SYSTEM_PROMPT

        # Step 4: Debug output
        if debug_mode:
            debug_lines = [
                "=" * 70,
                "ArchAi3D_Qwen_Style_Transfer - Generated Prompt (v4.0.0)",
                "=" * 70,
                f"Object: {object_to_style}",
                f"Style: {style}",
                f"Style Description: {style_description}",
                f"Scene Context: {scene_context[:80]}..." if len(scene_context) > 80 else f"Scene Context: {scene_context}",
                "=" * 70,
                "Generated Prompt:",
                prompt,
                "=" * 70,
                "⭐ System Prompt (NEW v4.0.0):",
                system_prompt,
                "=" * 70,
            ]
            print("\n".join(debug_lines))

        # Step 5: Return (now includes system_prompt)
        return io.NodeOutput(prompt, style_description, system_prompt)


# ============================================================================
# EXTENSION REGISTRATION
# ============================================================================

class StyleTransferExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [ArchAi3D_Qwen_Style_Transfer]


async def comfy_entrypoint():
    return StyleTransferExtension()
