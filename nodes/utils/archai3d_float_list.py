"""
Float List Node
================
Generate N float values for sequential multi-run processing.
Each value is output as a list item, so ComfyUI runs downstream nodes
once per value — same VRAM as a single run.

Use with GRAG lambda/delta, CFG scale, denoise, or any FLOAT input.
"""

import random


class ArchAi3D_FloatList:
    """Generate N float values as a list for sequential batch processing without extra VRAM."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {
                    "default": 0.5,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "Base value (used in fixed mode)"
                }),
                "value_min": ("FLOAT", {
                    "default": 0.0,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "Minimum value (used in linspace and random modes)"
                }),
                "value_max": ("FLOAT", {
                    "default": 1.0,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "Maximum value (used in linspace and random modes)"
                }),
                "count": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "How many values to generate (= how many times downstream runs)"
                }),
                "mode": (["linspace", "random", "fixed"], {
                    "default": "linspace",
                    "tooltip": "linspace: evenly spaced min→max. random: random between min/max (seeded). fixed: same value N times."
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Seed for reproducible random mode"
                }),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("value",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "generate"
    CATEGORY = "ArchAi3d/Utils"
    DESCRIPTION = "Generate N float values as a list. Downstream nodes run once per value sequentially, using the same VRAM as a single run."

    def generate(self, value, value_min, value_max, count, mode, seed):
        if mode == "fixed":
            values = [value] * count
        elif mode == "random":
            rng = random.Random(seed)
            values = [round(rng.uniform(value_min, value_max), 4) for _ in range(count)]
        else:  # linspace
            if count == 1:
                values = [value_min]
            else:
                step = (value_max - value_min) / (count - 1)
                values = [round(value_min + i * step, 4) for i in range(count)]

        print(f"[FloatList] mode={mode}, count={count}, values={values}")
        return (values,)
