"""
Simple Position Prompt Builder for ComfyUI
===========================================

Builds prompts in parentheses format for Qwen position guide workflow.

Author: ArchAi3d
Version: 1.1.0
Created: 2025-10-18
Updated: 2025-12-15

Description:
-----------
Simple node for building position guide prompts using parentheses format.
Each rectangle instruction is wrapped in parentheses.
Supports both ADD and REMOVE object operations.

Format (Add Mode):
------------------
(rectangle 1 = add a bicycle near the wall.)
(rectangle 2 = add a new man seats on chair.)
(rectangle 3 = add many dogs are playing in near and far.)

(Maintain all original elements in Image 1 unchanged.)

Format (Remove Mode):
---------------------
(rectangle 1 = remove the car.)
(rectangle 2 = remove the person.)

(Fill removed areas naturally with appropriate background.)

Usage:
------
1. Select operation mode (add_objects or remove_objects)
2. Enter object descriptions separated by /
3. Optionally add start description (appears at beginning)
4. Optionally add end description (defaults based on mode)
5. Get formatted prompt ready for Qwen

Example (Add):
--------------
Input: "add a bicycle near the wall / add a new man seats on chair"
Output: (rectangle 1 = add a bicycle near the wall.)
        (rectangle 2 = add a new man seats on chair.)

        (Maintain all original elements in Image 1 unchanged.)

Example (Remove):
-----------------
Input: "remove the car / remove the person"
Output: (rectangle 1 = remove the car.)
        (rectangle 2 = remove the person.)

        (Fill removed areas naturally with appropriate background.)
"""

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override


# System prompt for ADD objects mode
ADD_OBJECTS_SYSTEM_PROMPT = "You are an expert image compositor. You receive two inputs: Image 1 (scene to edit) and Image 2 (numbered position guide with red rectangles). Each rectangle in Image 2 has a number. The prompt provides a mapping of which object goes in which numbered rectangle. Read the number in each rectangle, find its mapping in the prompt, then add that specific object to Image 1 at the rectangle's position. Red rectangles and numbers are invisible reference guides only - do not draw them. Maintain all original elements in Image 1 unchanged."

# System prompt for REMOVE objects mode
REMOVE_OBJECTS_SYSTEM_PROMPT = "You are an expert image editor specializing in object removal and seamless inpainting. You receive two inputs: Image 1 (scene to edit) and Image 2 (numbered position guide with red rectangles marking areas to remove). Each rectangle in Image 2 has a number indicating an object or area to be removed. The prompt provides a mapping of what to remove in each numbered rectangle. Read the number in each rectangle, find its mapping in the prompt, then completely remove that object from Image 1 at the rectangle's position. Fill the removed area naturally with appropriate background content that seamlessly blends with the surrounding context (e.g., extend floor, wall, sky, grass, or whatever background is appropriate). Red rectangles and numbers are invisible reference guides only - do not draw them. The result should look as if the removed object was never there. Maintain all other original elements in Image 1 unchanged."

# Default end descriptions for each mode
DEFAULT_END_ADD = "Maintain all other elements and colors in Image 1 unchanged."
DEFAULT_END_REMOVE = "Fill all removed areas naturally with seamless background. Preserve lighting and perspective."


class ArchAi3D_Simple_Position_Prompt(io.ComfyNode):
    """
    Simple Position Prompt Builder - Parentheses Format.

    Supports two operation modes:
    - add_objects: Add new objects to the image at specified positions
    - remove_objects: Remove existing objects from the image

    Builds prompts in the format:
    (rectangle 1 = description.)
    (rectangle 2 = description.)
    ...
    (preservation/fill message.)
    """

    @classmethod
    def define_schema(cls):
        """Define node schema for ComfyUI."""
        return io.Schema(
            node_id="ArchAi3D_Simple_Position_Prompt",
            category="ArchAi3d/Utils",
            inputs=[
                io.Combo.Input(
                    "operation_mode",
                    options=["add_objects", "remove_objects"],
                    default="add_objects",
                    tooltip=(
                        "Select the operation mode:\n"
                        "- add_objects: Add new objects at rectangle positions\n"
                        "- remove_objects: Remove objects at rectangle positions and fill with background"
                    ),
                ),
                io.String.Input(
                    "object_descriptions",
                    multiline=True,
                    tooltip=(
                        "Describe objects separated by / (slash).\n"
                        "ADD mode: 'add a bicycle / add a man sitting on chair'\n"
                        "REMOVE mode: 'remove the car / remove the person / remove the sign'\n"
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
                    default="",
                    tooltip=(
                        "Description to add at the END (after rectangle mappings). "
                        "Leave empty to use smart default based on operation mode:\n"
                        "- ADD: 'Maintain all other elements and colors unchanged'\n"
                        "- REMOVE: 'Fill removed areas naturally with seamless background'"
                    ),
                ),
                io.Combo.Input(
                    "use_system_prompt",
                    options=["yes", "no"],
                    default="yes",
                    tooltip=(
                        "Include system prompt in output:\n"
                        "- yes: Include optimized system prompt for selected mode\n"
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
                    tooltip="System prompt optimized for selected mode (empty if 'no' selected)",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        operation_mode: str,
        object_descriptions: str,
        start_description: str,
        end_description: str,
        use_system_prompt: str,
    ) -> io.NodeOutput:
        """
        Build formatted position guide prompt in parentheses format.

        Args:
            operation_mode: "add_objects" or "remove_objects"
            object_descriptions: User descriptions separated by /
            start_description: Optional description at start
            end_description: Description at end (uses smart default if empty)
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

        # Determine mode-specific defaults
        is_remove_mode = operation_mode == "remove_objects"
        mode_label = "REMOVE" if is_remove_mode else "ADD"

        # Select appropriate system prompt based on mode
        if use_system_prompt == "yes":
            system_prompt = REMOVE_OBJECTS_SYSTEM_PROMPT if is_remove_mode else ADD_OBJECTS_SYSTEM_PROMPT
        else:
            system_prompt = ""

        # Select appropriate end description (use smart default if empty)
        if end_description and end_description.strip():
            final_end_description = end_description.strip()
        else:
            final_end_description = DEFAULT_END_REMOVE if is_remove_mode else DEFAULT_END_ADD

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

        # Add end description
        end_text = final_end_description
        # Ensure period at end
        if not end_text.endswith("."):
            end_text += "."
        prompt_lines.append(f"({end_text})")

        # Join all lines
        formatted_prompt = "\n".join(prompt_lines)

        # Debug output
        print("\n" + "="*80)
        print(f"📝 Simple Position Prompt Builder - v1.1.0 [{mode_label} MODE]")
        print("="*80)
        print(f"Operation mode: {operation_mode}")
        print(f"Object count: {len(descriptions)}")
        print(f"System prompt: {use_system_prompt}")
        if start_description and start_description.strip():
            print(f"Start description: {start_description.strip()}")
        print(f"\nObject descriptions:")
        for i, desc in enumerate(descriptions, start=1):
            print(f"  Rectangle {i}: {desc}")
        print(f"\nEnd description: {final_end_description}")
        print(f"\n📋 Formatted Prompt:")
        print(formatted_prompt)
        if system_prompt:
            print(f"\n🔧 System Prompt ({mode_label} mode):")
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
