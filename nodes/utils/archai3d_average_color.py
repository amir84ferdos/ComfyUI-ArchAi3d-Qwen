# -*- coding: utf-8 -*-
"""
ArchAi3D Average Color Node
Author: Amir Ferdos (ArchAi3d)
Email: Amir84ferdos@gmail.com
LinkedIn: https://www.linkedin.com/in/archai3d/

Description:
    Calculates the average color of an image and outputs it as a hexadecimal color code.
    Perfect for extracting dominant colors to use with the Mask to Position Guide node's
    custom_hex color options.

Usage:
    1. Input: Image (from ComfyUI workflow)
    2. Output: Hexadecimal color code (e.g., "A3B5C7")
    3. Use output with Mask to Position Guide node's rect_hex_color or num_hex_color inputs

Version: 1.0.0
"""

import numpy as np
import torch
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_average_color(image_tensor):
    """Calculate average RGB color from ComfyUI image tensor.

    Args:
        image_tensor: Tensor of shape (B, H, W, C) with values 0-1

    Returns:
        Tuple (hex_color, rgb_string):
        - hex_color: Hexadecimal color code (e.g., "A3B5C7")
        - rgb_string: Human-readable RGB values (e.g., "R:163, G:181, B:199")
    """
    # Convert to numpy (handle batch dimension)
    if len(image_tensor.shape) == 4:
        # Take first image if batched
        img_np = image_tensor[0].cpu().numpy()
    else:
        img_np = image_tensor.cpu().numpy()

    # Calculate average across height and width (axis 0 and 1)
    avg_color = img_np.mean(axis=(0, 1))

    # Convert from 0-1 range to 0-255 range
    r = int(avg_color[0] * 255)
    g = int(avg_color[1] * 255)
    b = int(avg_color[2] * 255)

    # Clamp values to 0-255 range
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))

    # Convert to hexadecimal (uppercase, no '#')
    hex_color = f"{r:02X}{g:02X}{b:02X}"

    # Create human-readable RGB string
    rgb_string = f"R:{r}, G:{g}, B:{b}"

    return hex_color, rgb_string


# ============================================================================
# NODE DEFINITION
# ============================================================================

class ArchAi3D_Average_Color(io.ComfyNode):
    """Average Color: Extract average color from image as hexadecimal code.

    This node calculates the average RGB color across an entire image and
    outputs it as a hexadecimal color code that can be used directly with
    the Mask to Position Guide node's custom_hex color inputs.

    Workflow:
    1. Input an image from your ComfyUI workflow
    2. Node calculates average R, G, B values
    3. Converts to hexadecimal format (e.g., "A3B5C7")
    4. Outputs hex code for use in other nodes

    Use Case:
    - Extract dominant color from reference images
    - Match position guide colors to scene colors
    - Create consistent color schemes across workflow
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ArchAi3D_Average_Color",
            category="ArchAi3d/Utils",
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="Input image to calculate average color from"
                ),
            ],
            outputs=[
                io.String.Output(
                    "hex_color",
                    tooltip="Hexadecimal color code (e.g., 'A3B5C7') - use with custom_hex color options"
                ),
                io.String.Output(
                    "rgb_values",
                    tooltip="Human-readable RGB values (e.g., 'R:163, G:181, B:199')"
                ),
            ],
        )

    @classmethod
    def execute(cls, image) -> io.NodeOutput:
        """Execute the Average Color node.

        Steps:
        1. Convert image tensor to numpy array
        2. Calculate average R, G, B values
        3. Convert to hexadecimal format
        4. Return hex code and readable RGB string
        """

        # Calculate average color
        hex_color, rgb_string = calculate_average_color(image)

        # Debug output
        print(f"\n{'='*70}")
        print(f"ArchAi3D_Average_Color - Color Analysis")
        print(f"{'='*70}")
        print(f"Average Color: #{hex_color}")
        print(f"RGB Values: {rgb_string}")
        print(f"{'='*70}\n")

        return io.NodeOutput(hex_color, rgb_string)


# ============================================================================
# EXTENSION REGISTRATION
# ============================================================================

class AverageColorExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [ArchAi3D_Average_Color]


async def comfy_entrypoint():
    return AverageColorExtension()
