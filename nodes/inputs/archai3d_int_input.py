"""
ArchAi3D Integer Input Node
An integer input node with a name field for web interface integration.
The name field is used by web interfaces to generate HTML form elements.
"""


class ArchAi3D_Int_Input:
    """
    Integer input node with a configurable name for web interface integration.

    The 'name' field identifies this input in web interfaces, allowing
    dynamic HTML form generation based on workflow inputs.

    Supports configurable min, max, and step values.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {
                    "default": "int_input",
                    "multiline": False,
                    "tooltip": "Identifier name for this input (used by web interface)"
                }),
                "value": ("INT", {
                    "default": 0,
                    "min": -2147483648,
                    "max": 2147483647,
                    "step": 1,
                    "tooltip": "The integer value to output"
                }),
                "min": ("INT", {
                    "default": -2147483648,
                    "min": -2147483648,
                    "max": 2147483647,
                    "step": 1,
                    "tooltip": "Minimum allowed value (for web interface validation)"
                }),
                "max": ("INT", {
                    "default": 2147483647,
                    "min": -2147483648,
                    "max": 2147483647,
                    "step": 1,
                    "tooltip": "Maximum allowed value (for web interface validation)"
                }),
                "step": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 1000000,
                    "step": 1,
                    "tooltip": "Step increment (for web interface slider/input)"
                }),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("INT",)
    FUNCTION = "execute"
    CATEGORY = "ArchAi3d/Inputs"

    def execute(self, name, value, min, max, step):
        # Clamp value to min/max range
        clamped_value = builtins_max(min, builtins_min(max, value))
        return (clamped_value,)


# Avoid shadowing built-in min/max
builtins_min = min
builtins_max = max
