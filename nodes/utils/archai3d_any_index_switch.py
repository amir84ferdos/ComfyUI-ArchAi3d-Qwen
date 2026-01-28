# ArchAi3D Any Index Switch Node
#
# Selects one value from multiple inputs based on an index.
# Similar to ComfyUI-Easy-Use's anythingIndexSwitch.
#
# Author: Amir Ferdos (ArchAi3d)
# Version: 1.0.0
# License: Dual License (Free for personal use, Commercial license required for business use)


# Maximum number of input slots
MAX_INPUTS = 10

# Any type definition for ComfyUI
any_type = ("*", {})

# Lazy evaluation options - only load the input that's actually needed
lazy_options = {"lazy": True}


class ArchAi3D_Any_Index_Switch:
    """
    Selects one value from multiple inputs based on an index.

    Connect different values to value0, value1, etc., and use the index
    to choose which one to output. Supports any data type.

    Features:
    - Accepts any type of input (images, latents, models, strings, etc.)
    - Lazy evaluation: only loads the selected input, saving memory
    - Supports up to 10 input slots (0-9)
    """

    CATEGORY = "ArchAi3d/Utils"
    RETURN_TYPES = (any_type[0],)
    RETURN_NAMES = ("value",)
    FUNCTION = "index_switch"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": MAX_INPUTS - 1,
                    "step": 1,
                    "tooltip": "Index of the input to select (0-9)"
                }),
            },
            "optional": {}
        }

        # Add optional input slots
        for i in range(MAX_INPUTS):
            inputs["optional"][f"value{i}"] = (any_type[0], {
                "lazy": True,
                "tooltip": f"Input value {i}"
            })

        return inputs

    def check_lazy_status(self, index, **kwargs):
        """
        ComfyUI lazy evaluation callback.
        Returns list of inputs that need to be evaluated.
        Only the selected index needs to be loaded.
        """
        key = f"value{index}"
        if kwargs.get(key, None) is None:
            return [key]
        return []

    def index_switch(self, index, **kwargs):
        """
        Return the value at the specified index.
        """
        key = f"value{index}"
        value = kwargs.get(key, None)

        if value is None:
            # Return None tuple if no value connected
            # This allows the node to work even with missing connections
            return (None,)

        return (value,)


# ============================================================================
# NODE REGISTRATION
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "ArchAi3D_Any_Index_Switch": ArchAi3D_Any_Index_Switch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_Any_Index_Switch": "🔀 Any Index Switch",
}
