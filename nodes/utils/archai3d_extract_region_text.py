"""
ArchAi3D Extract Region Text Node
Extracts quoted text from region descriptions.

Input format:
  (region 1: #FF0000, "add the oven on the left side", left)
  (region 2: #0000FF, "add table and chairs", right)

Output:
  add the oven on the left side.
  add table and chairs.
"""

import re


class ArchAi3D_Extract_Region_Text:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Region descriptions with quoted text to extract"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "execute"
    CATEGORY = "ArchAi3d/Utils"

    def execute(self, text):
        # Extract all quoted strings
        quotes = re.findall(r'"([^"]*)"', text)

        # Add period at end if not already there, join with newline
        lines = []
        for q in quotes:
            q = q.strip()
            if q and not q.endswith('.'):
                q += '.'
            lines.append(q)

        result = '\n'.join(lines)
        return (result,)
