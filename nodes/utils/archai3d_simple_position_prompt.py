"""
Simple Position Prompt Builder for ComfyUI
===========================================

Builds prompts in parentheses format for Qwen position guide workflow.

Author: ArchAi3d
Version: 1.0.0
Created: 2025-10-18

Description:
-----------
Simple node for building position guide prompts using parentheses format.
Each rectangle instruction is wrapped in parentheses.

Format:
-------
(rectangle 1 = add a bicycle near the wall.)
(rectangle 2 = add a new man seats on chair.)
(rectangle 3 = add many dogs are playing in near and far.)

(Maintain all original elements in Image 1 unchanged.)

Usage:
------
1. Enter object descriptions separated by /
2. Optionally add start description (appears at beginning)
3. Optionally add end description (defaults to preservation message)
4. Get formatted prompt ready for Qwen

Example:
--------
Input: "add a bicycle near the wall / add a new man seats on chair / add many dogs are playing in near and far"
Output: (rectangle 1 = add a bicycle near the wall.)
        (rectangle 2 = add a new man seats on chair.)
        (rectangle 3 = add many dogs are playing in near and far.)

        (Maintain all original elements in Image 1 unchanged.)
"""

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override


# Default system prompt (your working prompt)
DEFAULT_SYSTEM_PROMPT = "You are an expert image compositor. You receive two inputs: Image 1 (scene to edit) and Image 2 (numbered position guide with red rectangles). Each rectangle in Image 2 has a number. The prompt provides a mapping of which object goes in which numbered rectangle. Read the number in each rectangle, find its mapping in the prompt, then add that specific object to Image 1 at the rectangle's position. Red rectangles and numbers are invisible reference guides only - do not draw them. Maintain all original elements in Image 1 unchanged."


class ArchAi3D_Simple_Position_Prompt(io.ComfyNode):
    """
    Simple Position Prompt Builder - Parentheses Format.

    Builds prompts in the format:
    (rectangle 1 = description.)
    (rectangle 2 = description.)
    ...
    (preservation message.)
    """

    @classmethod
    def define_schema(cls):
        """Define node schema for ComfyUI."""
        return io.Schema(
            node_id="ArchAi3D_Simple_Position_Prompt",
            category="ArchAi3d/Utils",
            inputs=[
                io.String.Input(
                    "object_descriptions",
                    multiline=True,
                    tooltip=(
                        "Describe objects separated by / (slash). "
                        "Example: 'add a bicycle near the wall / add a new man seats on chair / add many dogs'. "
                        "Each description will be mapped to rectangle 1, 2, 3... in order."
                    ),
                ),
                io.String.Input(
                    "start_description",
                    multiline=True,
                    default="",
                    tooltip=(
                        "Optional description to add at the START (before rectangle mappings). "
                        "Will be wrapped in parentheses. Leave empty if not needed."
                    ),
                ),
                io.String.Input(
                    "end_description",
                    multiline=False,
                    default="Maintain all other elements and colors in Image 1 unchanged.",
                    tooltip=(
                        "Description to add at the END (after rectangle mappings). "
                        "Will be wrapped in parentheses. Default: enhanced preservation with color protection."
                    ),
                ),
                io.Combo.Input(
                    "use_system_prompt",
                    options=["yes", "no"],
                    default="yes",
                    tooltip=(
                        "Include system prompt in output:\n"
                        "- yes: Include validated system prompt (recommended)\n"
                        "- no: No system prompt (empty string)"
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    "formatted_prompt",
                    tooltip="Formatted prompt with parentheses format",
                ),
                io.String.Output(
                    "system_prompt",
                    tooltip="System prompt (empty if 'no' selected)",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        object_descriptions: str,
        start_description: str,
        end_description: str,
        use_system_prompt: str,
    ) -> io.NodeOutput:
        """
        Build formatted position guide prompt in parentheses format.

        Args:
            object_descriptions: User descriptions separated by /
            start_description: Optional description at start
            end_description: Description at end (preservation message)
            use_system_prompt: Whether to include system prompt ("yes" or "no")

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

        # Build prompt lines
        prompt_lines = []

        # Add start description if provided
        if start_description and start_description.strip():
            start_text = start_description.strip()
            # Ensure period at end
            if not start_text.endswith("."):
                start_text += "."
            prompt_lines.append(f"({start_text})")

        # Add rectangle mappings
        for i, desc in enumerate(descriptions, start=1):
            desc_text = desc.strip()
            # Ensure period at end
            if not desc_text.endswith("."):
                desc_text += "."
            prompt_lines.append(f"(rectangle {i} = {desc_text})")

        # Add blank line before end description
        if prompt_lines:
            prompt_lines.append("")

        # Add end description if provided
        if end_description and end_description.strip():
            end_text = end_description.strip()
            # Ensure period at end
            if not end_text.endswith("."):
                end_text += "."
            prompt_lines.append(f"({end_text})")

        # Join all lines
        formatted_prompt = "\n".join(prompt_lines)

        # Get system prompt
        if use_system_prompt == "yes":
            system_prompt = DEFAULT_SYSTEM_PROMPT
        else:
            system_prompt = ""

        # Debug output
        print("\n" + "="*80)
        print("📝 Simple Position Prompt Builder - v1.0.0")
        print("="*80)
        print(f"Object count: {len(descriptions)}")
        print(f"System prompt: {use_system_prompt}")
        if start_description and start_description.strip():
            print(f"Start description: {start_description.strip()}")
        print(f"\nObject descriptions:")
        for i, desc in enumerate(descriptions, start=1):
            print(f"  Rectangle {i}: {desc}")
        print(f"\nEnd description: {end_description.strip()}")
        print(f"\n📋 Formatted Prompt:")
        print(formatted_prompt)
        if system_prompt:
            print(f"\n🔧 System Prompt:")
            print(system_prompt)
        print("="*80 + "\n")

        return io.NodeOutput(formatted_prompt, system_prompt)


# ============================================================================
# EXTENSION REGISTRATION
# ============================================================================

class SimplePositionPromptExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [ArchAi3D_Simple_Position_Prompt]


async def comfy_entrypoint():
    return SimplePositionPromptExtension()
