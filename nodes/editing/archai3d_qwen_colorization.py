"""
ArchAi3D Qwen Colorization Node
Author: Amir Ferdos (ArchAi3d)
Email: Amir84ferdos@gmail.com
LinkedIn: https://www.linkedin.com/in/archai3d/

Description:
    Colorize black & white or grayscale images with realistic colors. Perfect for:
    - Historical photo restoration
    - Old family photo colorization
    - Vintage image enhancement
    - Black & white art colorization

Based on research from Qwen-repo3 (WanX API documentation):
- Can specify colors: "Blue background, yellow leaves"
- Can let model auto-select: "colorize with realistic colors"
- Can add era context: "appropriate for the 1950s era"
"""

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

# ============================================================================
# COLORIZATION SYSTEM PROMPT - v4.0.0 Feature
# ============================================================================

COLORIZATION_SYSTEM_PROMPT = "You are a photo restoration specialist. Colorize black and white or grayscale images with realistic, natural colors. Apply period-appropriate color palettes when era is specified. Maintain natural skin tones, realistic color relationships, and proper lighting throughout. Preserve all image details, composition, contrast, and clarity. Create believable colorization with historically accurate or naturally realistic color choices."

# ============================================================================
# ERA PRESETS - Historical color accuracy
# ============================================================================

ERA_PRESETS = {
    "none": "",
    "1900s": "appropriate for the 1900s era",
    "1920s": "appropriate for the 1920s era",
    "1940s": "appropriate for the 1940s era",
    "1950s": "appropriate for the 1950s era",
    "1960s": "appropriate for the 1960s era",
    "1970s": "appropriate for the 1970s era",
    "1980s": "appropriate for the 1980s era",
    "victorian": "appropriate for the Victorian period",
    "medieval": "appropriate for the medieval period",
}

# ============================================================================
# NODE DEFINITION
# ============================================================================

class ArchAi3D_Qwen_Colorization(io.ComfyNode):
    """Colorization: Convert black & white images to color.

    This node brings old photos and images to life with realistic colors:
    - Automatic color selection (model chooses realistic colors)
    - Custom color hints (specify desired colors)
    - Era-appropriate colorization (1900s-1980s, Victorian, Medieval)
    - Natural skin tone preservation

    Key Features:
    - Auto mode: Let model choose realistic colors
    - Custom mode: Specify color hints ("blue sky, green grass")
    - 9 era presets for historically accurate colorization
    - Maintains natural skin tones
    - Based on official Qwen API documentation

    Perfect For:
    - Family photo restoration
    - Historical archive colorization
    - Art restoration
    - Documentary filmmaking
    - Photo collection enhancement

    Based on research:
    - Auto mode: "colorize this black and white photo with realistic colors"
    - Custom mode: "colorize this image: blue background, yellow leaves"
    - With era: "...appropriate for the 1950s era"
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ArchAi3D_Qwen_Colorization",
            category="ArchAi3d/Qwen/Editing",
            inputs=[
                # Group 1: Colorization Mode
                io.Boolean.Input(
                    "auto_color",
                    default=True,
                    tooltip="Let model automatically choose realistic colors. Turn off to specify custom color hints.",
                ),
                io.String.Input(
                    "color_hints",
                    multiline=True,
                    default="",
                    tooltip="Custom color hints (used if auto_color is off). Example: 'blue sky, green grass, red car, brown brick building'",
                ),

                # Group 2: Era Context
                io.Combo.Input(
                    "era",
                    options=list(ERA_PRESETS.keys()),
                    default="none",
                    tooltip="Optional: Specify historical era for period-accurate colorization",
                ),

                # Group 3: Options
                io.Boolean.Input(
                    "preserve_skin_tones",
                    default=True,
                    tooltip="Maintain natural skin tones (adds hint to prompt)",
                ),
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
                    tooltip="⭐ NEW v4.0: Perfect system prompt for colorization! Connect to encoder's system_prompt input for optimal results.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        auto_color,
        color_hints,
        era,
        preserve_skin_tones,
        debug_mode,
    ) -> io.NodeOutput:
        """Execute the Colorization node.

        Steps:
        1. Build colorization prompt (auto or custom)
        2. Add era context if specified
        3. Add skin tone preservation hint if requested
        4. Output debug info if requested
        5. Return prompt
        """

        # Step 1: Build base colorization prompt
        if auto_color:
            prompt = "colorize this black and white photo with realistic colors"
        else:
            color_hints_text = color_hints.strip() if color_hints.strip() else "natural realistic colors"
            prompt = f"colorize this image: {color_hints_text}"

        # Step 2: Add era context
        era_phrase = ERA_PRESETS.get(era, "")
        if era_phrase:
            prompt += f" {era_phrase}"

        # Step 3: Add skin tone preservation hint
        if preserve_skin_tones:
            prompt += ", maintaining natural skin tones"

        # Get system prompt (v4.0.0 feature)
        system_prompt = COLORIZATION_SYSTEM_PROMPT

        # Step 4: Debug output
        if debug_mode:
            debug_lines = [
                "=" * 70,
                "ArchAi3D_Qwen_Colorization - Generated Prompt (v4.0.0)",
                "=" * 70,
                f"Auto Color: {auto_color}",
                f"Color Hints: {color_hints if not auto_color else '(automatic)'}",
                f"Era: {era}",
                f"Preserve Skin Tones: {preserve_skin_tones}",
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
        return io.NodeOutput(prompt, system_prompt)


# ============================================================================
# EXTENSION REGISTRATION
# ============================================================================

class ColorizationExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [ArchAi3D_Qwen_Colorization]


async def comfy_entrypoint():
    return ColorizationExtension()
