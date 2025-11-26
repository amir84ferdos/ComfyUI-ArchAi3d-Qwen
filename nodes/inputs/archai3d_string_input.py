"""
ArchAi3D String Input Node
A string input node with a name field for web interface integration.
The name field is used by web interfaces to generate HTML form elements.
"""


class ArchAi3D_String_Input:
    """
    String input node with a configurable name for web interface integration.

    The 'name' field identifies this input in web interfaces, allowing
    dynamic HTML form generation based on workflow inputs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {
                    "default": "string_input",
                    "multiline": False,
                    "tooltip": "Identifier name for this input (used by web interface)"
                }),
                "value": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "The string value to output"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "ArchAi3d/Inputs"

    def execute(self, name, value):
        return (value,)
