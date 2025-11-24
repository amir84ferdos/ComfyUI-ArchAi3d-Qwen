"""
Position Guide Prompt Builder Node for ComfyUI
==============================================

Auto-formats position guide prompts for Qwen image editing.

Author: ArchAi3d
Version: 1.1.0
Created: 2025-10-17
Updated: 2025-10-18 - Added removal options (start/end positions, separate rectangle/number controls)

Description:
-----------
Takes user-defined object descriptions (separated by /) and automatically
formats them into proper Qwen position guide prompts with numbered rectangles.

Usage:
------
1. Define object descriptions separated by / (e.g., "boy seats on sofa / man seats on sofa")
2. Choose a template preset (standard, no_removal, minimal, custom)
3. Optionally add additional instructions
4. Get formatted prompt ready for Qwen

Template Presets:
----------------
- standard: Full prompt with combined removal command (default)
- standard_explicit: Split removal commands (use if rectangles/numbers appear in output) ⭐
- strong_removal: Ultra-explicit removal with IMPORTANT prefix (use if standard_explicit fails) ⭐⭐
- no_removal: Keep rectangles visible in output (for debugging)
- minimal: Just object mappings, no extra commands
- custom: User-defined template with {MAPPINGS} and {ADDITIONAL} placeholders

Output Format:
-------------
Mappings are formatted as: rectangle 1= description, rectangle 2= description, ...

Example:
--------
Input: "boy seats on sofa / man seats on sofa / woman seats on sofa"
Additional: "man is close to boy"
Template: "standard"

Output: "using the second image as a position reference guide, the red rectangles
are numbered, add objects to the first image according to this mapping: rectangle 1=
boy seats on sofa, rectangle 2= man seats on sofa, rectangle 3= woman seats on sofa,
place each object inside its numbered rectangle area, then remove all red rectangles
and numbers from the image, keep everything else in the first image identical, man is
close to boy"
"""

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override


# System prompt templates
SYSTEM_PROMPT_TEMPLATES = {
    "none": "",

    "user_validated": "You are an expert image compositor. You receive two inputs: Image 1 (scene to edit) and Image 2 (numbered position guide with red rectangles). Each rectangle in Image 2 has a number. The prompt provides a mapping of which object goes in which numbered rectangle. Read the number in each rectangle, find its mapping in the prompt, then add that specific object to Image 1 at the rectangle's position. Red rectangles and numbers are invisible reference guides only - do not draw them. Maintain all original elements in Image 1 unchanged.",

    "standard": "You are an expert image editor. You receive two inputs: Image 1 (scene to edit) and Image 2 (numbered position guide with red rectangles). Each rectangle in Image 2 has a number. The prompt provides a mapping of which object goes in which numbered rectangle. Read the number in each rectangle, find its mapping in the prompt, then add that specific object to Image 1 at the rectangle's position. Red rectangles and numbers are invisible reference guides only - do not draw them. Keep all other parts of Image 1 unchanged.",

    "detailed": "You are a professional image editor specialized in precise object placement. You receive two inputs: Image 1 (scene to edit) and Image 2 (numbered position guide with red rectangles). Each rectangle in Image 2 has a number. The prompt provides a mapping of which object goes in which numbered rectangle. Read the number in each rectangle, find its mapping in the prompt, then add that specific object to Image 1 at the rectangle's position. Red rectangles and numbers are invisible reference guides only - do not draw them in the output. Maintain photorealistic quality, lighting consistency, and do not modify any existing elements in Image 1 outside the numbered areas.",

    "preserve_everything": "You are an expert image editor. You receive two inputs: Image 1 (scene to edit) and Image 2 (numbered position guide with red rectangles). Each rectangle in Image 2 has a number. The prompt provides a mapping of which object goes in which numbered rectangle. Read the number in each rectangle, find its mapping in the prompt, then add that specific object to Image 1 at the rectangle's position. Red rectangles and numbers are invisible reference guides only - do not draw them. CRITICAL: Keep ALL existing elements in Image 1 EXACTLY as they are: furniture, walls, floors, ceilings, lighting, decorations, colors, textures, and spatial relationships. Only add the specified objects in the marked positions. Maintain identical lighting, shadows, and perspective.",

    "minimal_change": "You receive Image 1 (scene to edit) and Image 2 (position guide with numbered red rectangles). Each rectangle has a number. Read the number, find its mapping in the prompt, add that object to Image 1 at the rectangle's position. Red rectangles and numbers are invisible guides - do not draw them. Keep everything else in Image 1 identical.",

    "three_stage": "You are an expert image compositor with cleanup capability. You receive two inputs: Image 1 (main scene to edit) and Image 2 (numbered position guide with red rectangles). Your task has three stages: Stage 1 - Read: Each red rectangle in Image 2 has a number. The prompt maps numbers to objects (example: rectangle 1 = flower). Stage 2 - Add: Add each mapped object to Image 1 at its numbered rectangle's position. Stage 3 - Clean: Remove all red rectangles and all numbers from the final image. These are temporary reference guides and must not appear in output. Use the remove command to clean them. Maintain all original elements in Image 1 unchanged. Only the mapped objects should be added, and all guide markers should be removed.",

    "custom": "",  # User provides custom system prompt
}


# Helper function to generate removal instructions
def generate_removal_text(removal_option: str) -> str:
    """
    Generate removal instruction text based on option.

    Args:
        removal_option: One of "none", "combined", "rectangles_only", "numbers_only", "both_separate"

    Returns:
        Formatted removal instruction text
    """
    if removal_option == "none":
        return ""
    elif removal_option == "combined":
        return "remove all red rectangles and numbers from the image"
    elif removal_option == "rectangles_only":
        return "remove all red rectangles from the image"
    elif removal_option == "numbers_only":
        return "remove all numbers from the image"
    elif removal_option == "both_separate":
        return "remove all red rectangles from the image, remove all numbers from the image"
    else:
        return ""


# Main prompt templates
TEMPLATE_PRESETS = {
    "standard": (
        "using the second image as a position reference guide, the red rectangles are numbered, "
        "add objects to the first image according to this mapping: {MAPPINGS}, "
        "place each object inside its numbered rectangle area, "
        "then remove all red rectangles and numbers from the image, "
        "keep everything else in the first image identical{ADDITIONAL}"
    ),
    "standard_explicit": (
        "using the second image as a position reference guide, the red rectangles are numbered, "
        "add objects to the first image according to this mapping: {MAPPINGS}, "
        "place each object inside its numbered rectangle area, "
        "remove all red rectangles from the image, "
        "remove all numbers from the image, "
        "keep everything else in the first image identical{ADDITIONAL}"
    ),
    "strong_removal": (
        "using the second image as a position reference guide, the red rectangles are numbered, "
        "add objects to the first image according to this mapping: {MAPPINGS}, "
        "place each object inside its numbered rectangle area, "
        "IMPORTANT: remove all red rectangles from the image, "
        "IMPORTANT: remove all numbers from the image, "
        "the rectangles and numbers are temporary guides only and must not be drawn or visible, "
        "keep everything else in the first image identical{ADDITIONAL}"
    ),
    "no_removal": (
        "using the second image as a position reference guide, the red rectangles are numbered, "
        "add objects to the first image according to this mapping: {MAPPINGS}, "
        "place each object inside its numbered rectangle area, "
        "keep everything else in the first image identical{ADDITIONAL}"
    ),
    "minimal": (
        "add objects to the first image: {MAPPINGS}{ADDITIONAL}"
    ),
}


class ArchAi3D_Position_Guide_Prompt_Builder(io.ComfyNode):
    """
    Builds formatted prompts for position guide workflows with Qwen.

    Automatically converts user descriptions into properly formatted
    position guide prompts with numbered rectangle mappings.
    """

    @classmethod
    def define_schema(cls):
        """Define node schema for ComfyUI."""
        return io.Schema(
            node_id="ArchAi3D_Position_Guide_Prompt_Builder",
            category="ArchAi3d/Utils",
            inputs=[
                io.String.Input(
                    "object_descriptions",
                    multiline=True,
                    tooltip=(
                        "Describe objects separated by / (slash). "
                        "Example: 'boy seats on sofa / man seats on sofa / woman seats on sofa'. "
                        "Each description will be mapped to rectangle 1, 2, 3... in order."
                    ),
                ),
                io.Combo.Input(
                    "prompt_template",
                    options=["standard", "standard_explicit", "strong_removal", "no_removal", "minimal", "custom"],
                    default="standard",
                    tooltip=(
                        "Template preset:\n"
                        "- standard: Combined removal command (default)\n"
                        "- standard_explicit: Split removal (use if rectangles/numbers appear in output) ⭐\n"
                        "- strong_removal: Ultra-explicit with IMPORTANT prefix (use if standard_explicit fails) ⭐⭐\n"
                        "- no_removal: Keep rectangles visible (for debugging)\n"
                        "- minimal: Just mappings, no extra commands\n"
                        "- custom: Use custom_template field below"
                    ),
                ),
                io.String.Input(
                    "custom_template",
                    multiline=True,
                    default="",
                    tooltip=(
                        "Custom template (only used when template is 'custom'). "
                        "Use {MAPPINGS} for object mappings, {ADDITIONAL} for extra instructions. "
                        "Example: 'Add to image 1 using image 2 guide: {MAPPINGS}. {ADDITIONAL}'"
                    ),
                ),
                io.Combo.Input(
                    "system_prompt_template",
                    options=["user_validated", "three_stage", "none", "standard", "detailed", "preserve_everything", "minimal_change", "custom"],
                    default="user_validated",
                    tooltip=(
                        "System prompt template:\n"
                        "- user_validated: YOUR WORKING PROMPT with 3-stage cleanup! (recommended) ⭐\n"
                        "- three_stage: Explicit 3-stage process (Read→Add→Clean) ⭐⭐\n"
                        "- none: No system prompt\n"
                        "- standard: Basic guidance + removal reminder\n"
                        "- detailed: Precise placement + explicit removal\n"
                        "- preserve_everything: STRONG preservation + removal rules\n"
                        "- minimal_change: Short and direct with removal\n"
                        "- custom: Use custom_system_prompt field below"
                    ),
                ),
                io.String.Input(
                    "custom_system_prompt",
                    multiline=True,
                    default="",
                    tooltip=(
                        "Custom system prompt (only used when system_prompt_template is 'custom'). "
                        "Example: 'You are an expert in architectural visualization...'"
                    ),
                ),
                io.String.Input(
                    "additional_instructions",
                    multiline=False,
                    default="",
                    tooltip=(
                        "Extra details to add at the end (optional). "
                        "Example: 'man is close to boy' or 'warm lighting, photorealistic style'"
                    ),
                ),

                # Removal instruction controls
                io.Combo.Input(
                    "removal_at_start",
                    options=["none", "combined", "rectangles_only", "numbers_only", "both_separate"],
                    default="none",
                    tooltip=(
                        "Add removal instruction at START of prompt:\n"
                        "- none: No removal at start\n"
                        "- combined: 'remove all red rectangles and numbers from the image'\n"
                        "- rectangles_only: 'remove all red rectangles from the image'\n"
                        "- numbers_only: 'remove all numbers from the image'\n"
                        "- both_separate: Both commands separately (rectangles, then numbers)"
                    ),
                ),
                io.Combo.Input(
                    "removal_at_end",
                    options=["none", "combined", "rectangles_only", "numbers_only", "both_separate"],
                    default="combined",
                    tooltip=(
                        "Add removal instruction at END of prompt:\n"
                        "- none: No removal at end\n"
                        "- combined: 'remove all red rectangles and numbers from the image' (default)\n"
                        "- rectangles_only: 'remove all red rectangles from the image'\n"
                        "- numbers_only: 'remove all numbers from the image'\n"
                        "- both_separate: Both commands separately (rectangles, then numbers)"
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    "formatted_prompt",
                    tooltip="Complete formatted prompt ready for Qwen image editing",
                ),
                io.String.Output(
                    "system_prompt",
                    tooltip="System prompt (empty if 'none' selected)",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        object_descriptions: str,
        prompt_template: str,
        custom_template: str,
        system_prompt_template: str,
        custom_system_prompt: str,
        additional_instructions: str,
        removal_at_start: str = "none",
        removal_at_end: str = "combined",
    ) -> io.NodeOutput:
        """
        Build formatted position guide prompt and system prompt.

        Args:
            object_descriptions: User descriptions separated by /
            prompt_template: Template preset (standard, no_removal, minimal, custom)
            custom_template: Custom template (only used if prompt_template is "custom")
            system_prompt_template: System prompt preset (none, standard, detailed, custom)
            custom_system_prompt: Custom system prompt (only used if system_prompt_template is "custom")
            additional_instructions: Optional extra instructions
            removal_at_start: Removal instruction at start (none, combined, rectangles_only, numbers_only, both_separate)
            removal_at_end: Removal instruction at end (none, combined, rectangles_only, numbers_only, both_separate)

        Returns:
            NodeOutput containing (formatted_prompt, system_prompt)
        """
        # Parse object descriptions
        descriptions = [d.strip() for d in object_descriptions.split("/") if d.strip()]

        if not descriptions:
            # No descriptions provided
            formatted_prompt = ""
            system_prompt = ""
            return io.NodeOutput(formatted_prompt, system_prompt)

        # Build mappings with format: rectangle 1 = description
        mappings = []
        for i, desc in enumerate(descriptions, start=1):
            mappings.append(f"rectangle {i} = {desc}")

        mappings_text = ", ".join(mappings)

        # Handle additional instructions
        if additional_instructions and additional_instructions.strip():
            additional_text = f", {additional_instructions.strip()}"
        else:
            additional_text = ""

        # Get main prompt template
        if prompt_template == "custom":
            # Use custom template
            template = custom_template if custom_template.strip() else TEMPLATE_PRESETS["standard"]
        else:
            # Use preset
            template = TEMPLATE_PRESETS.get(prompt_template, TEMPLATE_PRESETS["standard"])

        # Replace placeholders in main prompt
        formatted_prompt = template.replace("{MAPPINGS}", mappings_text).replace("{ADDITIONAL}", additional_text)

        # Generate removal instructions
        removal_start_text = generate_removal_text(removal_at_start)
        removal_end_text = generate_removal_text(removal_at_end)

        # Build final prompt with removal instructions
        prompt_parts = []

        # Add removal at start if specified
        if removal_start_text:
            prompt_parts.append(removal_start_text + ".")

        # Add main prompt content
        prompt_parts.append(formatted_prompt)

        # Add removal at end if specified (replace template's built-in removal if present)
        if removal_end_text:
            # Remove template's built-in removal instructions if they exist
            main_prompt = prompt_parts[-1]

            # Remove common built-in removal phrases (be thorough with variations)
            removal_phrases = [
                "then remove all red rectangles and numbers from the image, ",
                "then remove all red rectangles and numbers from the image",
                ", remove all red rectangles and numbers from the image",
                "remove all red rectangles and numbers from the image, ",
                "remove all red rectangles and numbers from the image",
                "remove all red rectangles from the image, ",
                "remove all red rectangles from the image",
                ", remove all red rectangles from the image",
                "remove all numbers from the image, ",
                "remove all numbers from the image",
                ", remove all numbers from the image",
                "IMPORTANT: remove all red rectangles from the image, ",
                "IMPORTANT: remove all red rectangles from the image",
                "IMPORTANT: remove all numbers from the image, ",
                "IMPORTANT: remove all numbers from the image",
                "the rectangles and numbers are temporary guides only and must not be drawn or visible, ",
                "the rectangles and numbers are temporary guides only and must not be drawn or visible",
            ]

            for phrase in removal_phrases:
                main_prompt = main_prompt.replace(phrase, "")

            # Clean up any double commas or spaces
            main_prompt = main_prompt.replace(",,", ",").replace("  ", " ").strip()

            # Remove trailing comma if present
            if main_prompt.endswith(","):
                main_prompt = main_prompt[:-1].strip()

            prompt_parts[-1] = main_prompt

            # Add custom removal at end
            # Insert removal BEFORE "keep everything else" if present
            if ", keep everything else in the first image identical" in main_prompt:
                # Insert removal before the "keep everything" part
                main_prompt = main_prompt.replace(
                    ", keep everything else in the first image identical",
                    ", then " + removal_end_text + ", keep everything else in the first image identical"
                )
                prompt_parts[-1] = main_prompt
            elif "keep everything else in the first image identical" in main_prompt:
                # No comma before "keep", add removal before it
                main_prompt = main_prompt.replace(
                    "keep everything else in the first image identical",
                    "then " + removal_end_text + ", keep everything else in the first image identical"
                )
                prompt_parts[-1] = main_prompt
            elif main_prompt.endswith("."):
                # Already ends with period, add removal as new sentence
                prompt_parts.append(removal_end_text)
            else:
                # Default: add with "then" as separate part
                prompt_parts.append("then " + removal_end_text)

        # Join all parts
        formatted_prompt = " ".join(prompt_parts)

        # Clean up spacing
        formatted_prompt = formatted_prompt.replace("  ", " ").strip()

        # Get system prompt
        if system_prompt_template == "custom":
            # Use custom system prompt
            system_prompt = custom_system_prompt if custom_system_prompt.strip() else SYSTEM_PROMPT_TEMPLATES["none"]
        else:
            # Use preset system prompt
            system_prompt = SYSTEM_PROMPT_TEMPLATES.get(system_prompt_template, SYSTEM_PROMPT_TEMPLATES["none"])

        # Debug output (optional, can be removed in production)
        print("\n" + "="*80)
        print("📝 Position Guide Prompt Builder - v1.1.0")
        print("="*80)
        print(f"Prompt template: {prompt_template}")
        print(f"System prompt template: {system_prompt_template}")
        print(f"Removal at start: {removal_at_start}")
        print(f"Removal at end: {removal_at_end}")
        print(f"Object count: {len(descriptions)}")
        print(f"\nObject descriptions:")
        for i, desc in enumerate(descriptions, start=1):
            print(f"  Rectangle {i}: {desc}")
        if additional_text:
            print(f"\nAdditional instructions: {additional_instructions.strip()}")
        if system_prompt:
            print(f"\n🔧 System prompt:")
            print(f"{system_prompt}")
        print(f"\n📋 Main prompt:")
        print(f"{formatted_prompt}")
        print("="*80 + "\n")

        return io.NodeOutput(formatted_prompt, system_prompt)


# ============================================================================
# EXTENSION REGISTRATION
# ============================================================================

class PositionGuidePromptBuilderExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [ArchAi3D_Position_Guide_Prompt_Builder]


async def comfy_entrypoint():
    return PositionGuidePromptBuilderExtension()
