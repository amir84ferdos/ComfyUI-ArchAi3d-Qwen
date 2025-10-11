# ArchAi3D Qwen System Prompt — Preset loader for Qwen-VL system prompts
#
# OVERVIEW:
# Provides quick access to common system prompts for Qwen-VL 2.5 with preset dropdown.
# Choose from curated presets or write custom system prompts. Output connects directly
# to the system_prompt input of ArchAi3D_Qwen_Encoder or similar nodes.
#
# FEATURES:
# - Preset dropdown with common system prompts
# - Custom text input for writing your own
# - Preset overrides custom when selected
# - String output for connecting to encoder nodes
#
# Author: Amir Ferdos (ArchAi3d)
# Email: Amir84ferdos@gmail.com
# LinkedIn: https://www.linkedin.com/in/archai3d/
# GitHub: https://github.com/amir84ferdos
# Category: ArchAi3d/Qwen
# Node ID: ArchAi3D_Qwen_System_Prompt
# License: MIT

# Preset system prompts
SYSTEM_PROMPT_PRESETS = {
    "None": "",
    "Default Helper": "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.",
    "Interior Designer": "You are an expert interior designer. Analyze spaces and create detailed, photorealistic design transformations while preserving architectural structure, lighting, and perspective.",
    "Architect": "You are a professional architect. Focus on structural integrity, spatial planning, materials, and design principles when analyzing or transforming spaces.",
    "Creative Director": "You are a creative director with expertise in visual storytelling. Transform spaces with bold, artistic vision while maintaining coherence and aesthetic appeal.",
    "Renovation Expert": "You are a home renovation expert. Suggest practical, budget-conscious improvements that enhance functionality and aesthetics while respecting existing structure.",
    "Minimalist Designer": "You are a minimalist design specialist. Create clean, uncluttered spaces with focus on essential elements, natural light, and simple elegance.",
    "Luxury Designer": "You are a luxury interior designer. Transform spaces into opulent, high-end environments with premium materials, sophisticated details, and refined elegance.",
    "Photo Analyst": "You are a detailed image analyst. Describe what you see with precision, including objects, colors, materials, lighting, composition, and spatial relationships.",
    "Style Matcher": "You are a design consistency expert. Analyze the style of reference images and apply the same aesthetic principles, color palettes, and design language to new spaces.",
}

class ArchAi3D_Qwen_System_Prompt:
    """System prompt preset loader for Qwen-VL encoder nodes.

    Features:
    - Dropdown with curated system prompt presets
    - Custom text input for writing your own prompts
    - Preset selection overrides custom input
    - String output connects to encoder system_prompt input

    Use this to quickly switch between different AI personas/roles for Qwen-VL.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (list(SYSTEM_PROMPT_PRESETS.keys()), {"default": "Default Helper"}),
            },
            "optional": {
                "custom_prompt": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("system_prompt",)
    FUNCTION = "load_prompt"
    CATEGORY = "ArchAi3d/Qwen"

    def load_prompt(self, preset="Default Helper", custom_prompt=""):
        """Execute the system prompt loader.

        Args:
            preset: Selected preset name
            custom_prompt: Custom user-written prompt

        Returns:
            System prompt string based on preset or custom input
        """
        # Get preset text
        preset_text = SYSTEM_PROMPT_PRESETS.get(preset, "")

        # If preset is "None" or empty, use custom prompt
        if preset == "None" or not preset_text:
            output = custom_prompt
        else:
            output = preset_text

        return (output,)


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "ArchAi3D_Qwen_System_Prompt": ArchAi3D_Qwen_System_Prompt
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_Qwen_System_Prompt": "ArchAi3D Qwen System Prompt"
}
