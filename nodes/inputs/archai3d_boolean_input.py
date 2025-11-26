"""
ArchAi3D Boolean Input Node
A boolean input node with a name field for web interface integration.
The name field is used by web interfaces to generate HTML form elements.
"""


class ArchAi3D_Boolean_Input:
    """
    Boolean input node with a configurable name for web interface integration.

    The 'name' field identifies this input in web interfaces, allowing
    dynamic HTML form generation based on workflow inputs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {
                    "default": "bool_input",
                    "multiline": False,
                    "tooltip": "Identifier name for this input (used by web interface)"
                }),
                "value": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "The boolean value to output"
                }),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("BOOLEAN",)
    FUNCTION = "execute"
    CATEGORY = "ArchAi3d/Inputs"

    def execute(self, name, value):
        return (value,)
