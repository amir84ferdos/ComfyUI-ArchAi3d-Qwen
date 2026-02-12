"""
PromptLine Node
===============
Splits multi-line text into individual prompts.
Each line becomes a separate output that ComfyUI processes one at a time.
Similar to easy promptLine from comfyui-easy-use.
"""


class ArchAi3D_PromptLine:
    """Split multi-line text into individual prompts for per-image processing."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "front wall prompt\nright wall prompt\nback wall prompt\nleft wall prompt\nceiling prompt\nfloor prompt",
                    "tooltip": "One prompt per line. Each line will be output separately."
                }),
                "start_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 99,
                    "step": 1,
                    "tooltip": "Which line to start from (0 = first line)"
                }),
                "max_rows": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 99,
                    "step": 1,
                    "tooltip": "Max lines to output (0 = all lines)"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("prompt", "index", "total")
    OUTPUT_IS_LIST = (True, True, True)
    FUNCTION = "split"
    CATEGORY = "ArchAi3d/Panorama"
    DESCRIPTION = "Splits multi-line text into individual prompts. Each line is output separately for per-image processing."

    def split(self, text, start_index, max_rows):
        # Split into lines, keep only non-empty
        lines = [line.strip() for line in text.strip().split('\n') if line.strip()]

        if not lines:
            return ([""], [0], [0])

        total = len(lines)

        # Apply start_index
        start = min(start_index, total - 1)
        lines = lines[start:]

        # Apply max_rows
        if max_rows > 0:
            lines = lines[:max_rows]

        # Build index list
        indices = list(range(start, start + len(lines)))

        return (lines, indices, [total] * len(lines))
