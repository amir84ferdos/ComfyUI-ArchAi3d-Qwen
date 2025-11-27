"""
ArchAi3D Conditioning Balance Node
A global settings node for conditioning balance that can be connected to multiple encoder nodes.
Includes name field for web interface integration.
"""


# Conditioning Balance Presets (matching V3 encoder)
CONDITIONING_PRESETS = [
    "Image-Dominant",
    "Image-Priority",
    "Balanced",
    "Text-Priority",
    "Text-Dominant",
    "Custom"
]


class ArchAi3D_Conditioning_Balance:
    """
    Global Conditioning Balance node for controlling multiple ArchAi3D_Qwen_Encoder_V3 nodes.

    The 'name' field identifies this input in web interfaces, allowing
    dynamic HTML form generation based on workflow inputs.

    Connect the output to the 'conditioning_balance_override' input of multiple encoder nodes
    to control them all from one place.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {
                    "default": "conditioning_balance",
                    "multiline": False,
                    "tooltip": "Identifier name for this input (used by web interface)"
                }),
                "conditioning_balance": (CONDITIONING_PRESETS, {
                    "default": "Balanced",
                    "tooltip": "Choose conditioning balance preset (Image-Dominant → Text-Dominant)"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("conditioning_balance",)
    FUNCTION = "execute"
    CATEGORY = "ArchAi3d/Inputs"

    def execute(self, name, conditioning_balance):
        return (conditioning_balance,)
