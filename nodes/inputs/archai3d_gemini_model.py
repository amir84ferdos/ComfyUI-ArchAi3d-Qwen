"""
ArchAi3D Gemini Model Selector Node
A model selector node for Gemini API with name field for web interface integration.

Updated November 2025 with latest Gemini models.
"""


# Available Gemini Models (Updated November 2025)
GEMINI_MODELS = [
    # Gemini 3 (Latest - Released Nov 18, 2025)
    "gemini-3.0-pro",
    "gemini-3.0-flash",
    # Gemini 2.5 Models
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-pro-preview-05-06",
    # Gemini 2.0 Models
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    # Gemini 1.5 Models (Legacy)
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
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
                    "default": "gemini-3.0-flash",
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
