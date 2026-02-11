"""
ArchAi3D Gemini Model Selector Node
A model selector node for Gemini API with name field for web interface integration.

Updated February 2026 with latest Gemini models.
"""


# Available Gemini Models (Updated February 2026)
GEMINI_MODELS = [
    # Gemini 3 (Preview)
    "gemini-3-pro-preview",
    "gemini-3-pro-image-preview",
    "gemini-3-flash-preview",
    # Gemini 2.5 (Stable GA)
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash-image",
    "gemini-2.5-pro",
    # Gemini 2.0 (Retiring March 31, 2026)
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]


class ArchAi3D_Gemini_Model:
    """
    Gemini Model Selector node for controlling multiple Gemini nodes.

    The 'name' field identifies this input in web interfaces, allowing
    dynamic HTML form generation based on workflow inputs.

    Connect the output to the 'model_override' input of ArchAi3D_Gemini nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {
                    "default": "gemini_model",
                    "multiline": False,
                    "tooltip": "Identifier name for this input (used by web interface)"
                }),
                "model": (GEMINI_MODELS, {
                    "default": "gemini-2.5-flash",
                    "tooltip": "Select Gemini model to use"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model",)
    FUNCTION = "execute"
    CATEGORY = "ArchAi3d/Inputs"

    def execute(self, name, model):
        return (model,)
