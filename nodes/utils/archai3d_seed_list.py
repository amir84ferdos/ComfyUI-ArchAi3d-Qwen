"""
Seed List Node
==============
Generate N seeds for sequential multi-run processing.
Each seed is output as a list item, so ComfyUI runs downstream nodes
once per seed — same VRAM as a single run.
"""

import random


class ArchAi3D_SeedList:
    """Generate N seeds as a list for sequential batch processing without extra VRAM."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Base seed. In random mode, used to derive N deterministic seeds."
                }),
                "count": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "How many seeds to generate (= how many times downstream runs)"
                }),
                "mode": (["random", "increment", "fixed"], {
                    "default": "random",
                    "tooltip": "random: derive N seeds from base seed. increment: seed, seed+1, seed+2... fixed: same seed N times."
                }),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "generate"
    CATEGORY = "ArchAi3d/Utils"
    DESCRIPTION = "Generate N seeds as a list. Downstream nodes run once per seed sequentially, using the same VRAM as a single run."

    def generate(self, seed, count, mode):
        if mode == "fixed":
            seeds = [seed] * count
        elif mode == "increment":
            seeds = [seed + i for i in range(count)]
        else:  # random
            rng = random.Random(seed)
            seeds = [rng.randint(0, 0xffffffffffffffff) for _ in range(count)]

        print(f"[SeedList] mode={mode}, count={count}, seeds={seeds}")
        return (seeds,)
