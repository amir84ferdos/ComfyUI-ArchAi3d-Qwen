"""
ArchAi3D Qwen Watermark Removal Node
Author: Amir Ferdos (ArchAi3d)
Email: Amir84ferdos@gmail.com
LinkedIn: https://www.linkedin.com/in/archai3d/

Description:
    Remove watermarks, text, and unwanted elements from images. Perfect for:
    - Cleaning up stock photos with watermarks
    - Removing text overlays from images
    - Cleaning up screenshots with UI elements
    - Removing logos and brand marks

Based on research from Qwen-repo3 (WanX API documentation):
- Prompt: "Remove the text in the image" or "Remove the watermark"
- Can specify location: "from the bottom right corner"
- Can specify language: "Remove the English text"
"""

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

# ============================================================================
# WATERMARK REMOVAL SYSTEM PROMPT - v4.0.0 Feature
# ============================================================================

WATERMARK_REMOVAL_SYSTEM_PROMPT = "You are an image cleanup specialist. Remove specified watermarks, text overlays, logos, or unwanted elements from image cleanly. Intelligently inpaint removed areas to blend seamlessly with surrounding content. Preserve all other image content, composition, colors, and quality exactly. Maintain natural appearance after removal with no visible artifacts."

# ============================================================================
# CONSTANTS
# ============================================================================

TEXT_TYPE_MAP = {
    "all_text": "the text",
    "watermark": "the watermark",
    "english_text": "the English text",
    "chinese_text": "the Chinese text",
    "logo": "the logo",
    "brand_mark": "the brand mark",
    "ui_elements": "the UI elements",
}

LOCATION_MAP = {
    "anywhere": "",
    "bottom_right": "from the bottom right corner",
    "bottom_left": "from the bottom left corner",
    "top_right": "from the top right corner",
    "top_left": "from the top left corner",
    "center": "from the center",
    "bottom": "from the bottom",
    "top": "from the top",
}

# ============================================================================
# NODE DEFINITION
# ============================================================================

class ArchAi3D_Qwen_Watermark_Removal(io.ComfyNode):
    """Watermark Removal: Clean up images by removing unwanted elements.

    This node provides simple but powerful cleanup capabilities:
    - Remove watermarks and logos
    - Remove text overlays (English, Chinese, or all text)
    - Remove UI elements from screenshots
    - Remove brand marks

    Key Features:
    - 7 text/element types (watermark, logo, English/Chinese text, etc.)
    - 8 location options (corner-specific, center, top, bottom, anywhere)
    - Simple one-step cleanup
    - Based on official Qwen API documentation

    Perfect For:
    - Stock photo cleanup
    - Screenshot cleaning
    - Social media image preparation
    - Removing branding for presentations

    Based on research: "Remove the [TYPE] from the [LOCATION] of the image"
    pattern works best. If no location specified, omit it.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ArchAi3D_Qwen_Watermark_Removal",
            category="ArchAi3d/Qwen/Editing",
            inputs=[
                # Group 1: What to Remove
                io.Combo.Input(
                    "text_type",
                    options=list(TEXT_TYPE_MAP.keys()),
                    default="watermark",
                    tooltip="What to remove: watermark, text, logo, UI elements, etc.",
                ),

                # Group 2: Location
                io.Combo.Input(
                    "location",
                    options=list(LOCATION_MAP.keys()),
                    default="anywhere",
                    tooltip="Where to remove from. Use 'anywhere' to let model detect automatically, or specify location for precision.",
                ),

                # Group 3: Options
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
                    "system_prompt",
                    tooltip="⭐ NEW v4.0: Perfect system prompt for watermark removal! Connect to encoder's system_prompt input for optimal results.",
                ),
            ],
        )

    @classmethod
    def execute(cls, text_type, location, debug_mode) -> io.NodeOutput:
        """Execute the Watermark Removal node.

        Steps:
        1. Build removal prompt based on type and location
        2. Output debug info if requested
        3. Return prompt
        """

        # Step 1: Build prompt
        text_phrase = TEXT_TYPE_MAP.get(text_type, "the text")
        location_phrase = LOCATION_MAP.get(location, "")

        if location_phrase:
            prompt = f"Remove {text_phrase} {location_phrase} of the image"
        else:
            prompt = f"Remove {text_phrase} in the image"

        # Get system prompt (v4.0.0 feature)
        system_prompt = WATERMARK_REMOVAL_SYSTEM_PROMPT

        # Step 2: Debug output
        if debug_mode:
            debug_lines = [
                "=" * 70,
                "ArchAi3D_Qwen_Watermark_Removal - Generated Prompt (v4.0.0)",
                "=" * 70,
                f"Text Type: {text_type}",
                f"Location: {location}",
                "=" * 70,
                "Generated Prompt:",
                prompt,
                "=" * 70,
                "⭐ System Prompt (NEW v4.0.0):",
                system_prompt,
                "=" * 70,
            ]
            print("\n".join(debug_lines))

        # Step 3: Return (now includes system_prompt)
        return io.NodeOutput(prompt, system_prompt)


# ============================================================================
# EXTENSION REGISTRATION
# ============================================================================

class WatermarkRemovalExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [ArchAi3D_Qwen_Watermark_Removal]


async def comfy_entrypoint():
    return WatermarkRemovalExtension()
