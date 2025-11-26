"""
ArchAi3D Float Input Node
A float input node with a name field for web interface integration.
The name field is used by web interfaces to generate HTML form elements.
"""


# Store built-in functions before they get shadowed
_min = min
_max = max


class ArchAi3D_Float_Input:
    """
    Float input node with a configurable name for web interface integration.

    The 'name' field identifies this input in web interfaces, allowing
    dynamic HTML form generation based on workflow inputs.

    Supports configurable min, max, and step values.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {
                    "default": "float_input",
                    "multiline": False,
                    "tooltip": "Identifier name for this input (used by web interface)"
                }),
                "value": ("FLOAT", {
                    "default": 0.0,
                    "min": -1e10,
                    "max": 1e10,
                    "step": 0.01,
                    "tooltip": "The float value to output"
                }),
                "min": ("FLOAT", {
                    "default": -1e10,
                    "min": -1e10,
                    "max": 1e10,
                    "step": 0.01,
                    "tooltip": "Minimum allowed value (for web interface validation)"
                }),
                "max": ("FLOAT", {
                    "default": 1e10,
                    "min": -1e10,
                    "max": 1e10,
                    "step": 0.01,
                    "tooltip": "Maximum allowed value (for web interface validation)"
                }),
                "step": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.0001,
                    "max": 1000.0,
                    "step": 0.0001,
                    "tooltip": "Step increment (for web interface slider/input)"
                }),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("FLOAT",)
    FUNCTION = "execute"
    CATEGORY = "ArchAi3d/Inputs"

    def execute(self, name, value, min, max, step):
        # Clamp value to min/max range
        clamped_value = _max(min, _min(max, value))
        return (clamped_value,)
