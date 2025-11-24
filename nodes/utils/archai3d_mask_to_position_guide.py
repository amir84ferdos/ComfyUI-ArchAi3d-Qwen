# -*- coding: utf-8 -*-
"""
ArchAi3D Mask to Position Guide Node
Author: Amir Ferdos (ArchAi3d)
Email: Amir84ferdos@gmail.com
LinkedIn: https://www.linkedin.com/in/archai3d/

Description:
    Converts a mask to a numbered position guide image.
    Detects regions in mask, draws numbered rectangles with user-selected order.
    Perfect for creating position reference images for Qwen multi-image prompting.

Usage:
    1. Input: Mask with selected regions (white areas)
    2. Choose: Numbering order (left-to-right, right-to-left, top-to-bottom, or bottom-to-top)
    3. Output: RGB guide image with numbered rectangles
    4. Use output with Multi Image Text Encoder V3 for precise object placement

Numbering Orders:
    - left_to_right: Numbers 1,2,3... from left to right →
    - right_to_left: Numbers 1,2,3... from right to left ←
    - top_to_bottom: Numbers 1,2,3... from top to bottom ↓
    - bottom_to_top: Numbers 1,2,3... from bottom to top ↑
"""

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
import json
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

# ============================================================================
# COLOR DEFINITIONS
# ============================================================================

COLORS = {
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "green": (0, 255, 0),
    "yellow": (255, 255, 0),
    "magenta": (255, 0, 255),
    "cyan": (0, 255, 255),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def hex_to_rgb(hex_color):
    """Convert hexadecimal color code to RGB tuple.

    Args:
        hex_color: Hex color string (e.g., "#FF0000" or "FF0000")

    Returns:
        Tuple (R, G, B) with values 0-255
    """
    # Remove '#' if present
    hex_color = hex_color.lstrip('#')

    # Convert to RGB
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (r, g, b)
    except (ValueError, IndexError):
        # Invalid hex code, return red as fallback
        print(f"Warning: Invalid hex color '{hex_color}', using red (FF0000) as fallback")
        return (255, 0, 0)

def mask_to_pil(mask_tensor):
    """Convert ComfyUI mask tensor to PIL Image.

    Args:
        mask_tensor: Tensor of shape (H, W) or (1, H, W) with values 0-1

    Returns:
        PIL Image in 'L' mode (grayscale)
    """
    # Handle different tensor shapes
    if len(mask_tensor.shape) == 3:
        mask_tensor = mask_tensor[0]  # Take first if batched

    # Convert to numpy and scale to 0-255
    mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)

    return Image.fromarray(mask_np, mode='L')


def detect_regions(mask_pil, padding=5, numbering_order="left_to_right"):
    """Detect contiguous regions in mask and return bounding boxes.

    Args:
        mask_pil: PIL Image in 'L' mode (grayscale mask)
        padding: Pixels to pad around each detected region
        numbering_order: "left_to_right", "right_to_left", "top_to_bottom", or "bottom_to_top"

    Returns:
        List of dicts with bbox info, sorted by selected order:
        [
            {"bbox": [x1, y1, x2, y2], "center_x": int, "center_y": int, "number": int},
            ...
        ]
    """
    # Convert to numpy for OpenCV
    mask_np = np.array(mask_pil)

    # Threshold to binary (white = 255, black = 0)
    _, binary = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)

    # Find contours (connected components)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []

    for contour in contours:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Add padding
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(mask_pil.width, x + w + padding)
        y2 = min(mask_pil.height, y + h + padding)

        # Calculate center X and Y for sorting
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        regions.append({
            "bbox": [x1, y1, x2, y2],
            "center_x": center_x,
            "center_y": center_y,
            "number": 0  # Will assign after sorting
        })

    # Sort based on numbering order
    if numbering_order == "left_to_right":
        # Sort by center_x, ascending (left to right →)
        regions.sort(key=lambda r: r["center_x"])
    elif numbering_order == "right_to_left":
        # Sort by center_x, descending (right to left ←)
        regions.sort(key=lambda r: r["center_x"], reverse=True)
    elif numbering_order == "top_to_bottom":
        # Sort by center_y, ascending (top to bottom ↓)
        regions.sort(key=lambda r: r["center_y"])
    elif numbering_order == "bottom_to_top":
        # Sort by center_y, descending (bottom to top ↑)
        regions.sort(key=lambda r: r["center_y"], reverse=True)

    # Assign numbers (1, 2, 3...)
    for i, region in enumerate(regions, start=1):
        region["number"] = i

    return regions


def draw_numbered_rectangles(
    mask_pil,
    regions,
    rectangle_color="red",
    line_thickness=3,
    number_size=48,
    number_color="white",
    background_color="black",
    bg_custom_hex=None,
    rect_custom_hex=None,
    num_custom_hex=None
):
    """Draw numbered rectangles on image.

    Args:
        mask_pil: PIL Image (will create RGB version)
        regions: List of region dicts from detect_regions()
        rectangle_color: Color name for rectangles or "custom_hex"
        line_thickness: Thickness of rectangle borders
        number_size: Font size for numbers
        number_color: Color name for numbers or "custom_hex"
        background_color: Color name for background or "custom_hex"
        bg_custom_hex: Hex color string when background_color="custom_hex"
        rect_custom_hex: Hex color string when rectangle_color="custom_hex"
        num_custom_hex: Hex color string when number_color="custom_hex"

    Returns:
        PIL Image (RGB) with numbered rectangles
    """
    # Get background color
    if background_color == "custom_hex" and bg_custom_hex is not None:
        bg_color = hex_to_rgb(bg_custom_hex)
    else:
        bg_color = COLORS.get(background_color, COLORS["black"])

    # Create RGB image with selected background color
    img = Image.new('RGB', mask_pil.size, color=bg_color)
    draw = ImageDraw.Draw(img)

    # Get rectangle color
    if rectangle_color == "custom_hex" and rect_custom_hex is not None:
        rect_color = hex_to_rgb(rect_custom_hex)
    else:
        rect_color = COLORS.get(rectangle_color, COLORS["red"])

    # Get number color
    if number_color == "custom_hex" and num_custom_hex is not None:
        num_color = hex_to_rgb(num_custom_hex)
    else:
        num_color = COLORS.get(number_color, COLORS["white"])

    # Try to load a font, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", number_size)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", number_size)
        except:
            font = ImageFont.load_default()

    # Draw each region
    for region in regions:
        x1, y1, x2, y2 = region["bbox"]
        number = region["number"]

        # Draw rectangle
        draw.rectangle(
            [x1, y1, x2, y2],
            outline=rect_color,
            width=line_thickness
        )

        # Draw number (centered in rectangle)
        number_text = str(number)

        # Get text bounding box for centering
        bbox = draw.textbbox((0, 0), number_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Calculate center position
        text_x = x1 + (x2 - x1 - text_width) // 2
        text_y = y1 + (y2 - y1 - text_height) // 2

        # Draw number
        draw.text(
            (text_x, text_y),
            number_text,
            fill=num_color,
            font=font
        )

    return img


def pil_to_tensor(pil_img):
    """Convert PIL Image to ComfyUI tensor.

    Args:
        pil_img: PIL Image (RGB)

    Returns:
        Tensor of shape (1, H, W, 3) with values 0-1
    """
    # Convert to numpy array
    img_np = np.array(pil_img).astype(np.float32) / 255.0

    # Add batch dimension and convert to tensor
    img_tensor = torch.from_numpy(img_np)[None,]

    return img_tensor


# ============================================================================
# NODE DEFINITION
# ============================================================================

class ArchAi3D_Mask_To_Position_Guide(io.ComfyNode):
    """Mask to Position Guide: Convert mask to numbered rectangle guide image.

    This node takes a mask with selected regions and creates a position guide image
    with numbered rectangles. Perfect for Qwen multi-image position mapping workflow.

    Workflow:
    1. Input mask with white regions (selected areas)
    2. Choose numbering order (left-to-right or top-to-bottom)
    3. Node detects each region and assigns numbers
    4. Draws numbered rectangles
    5. Outputs guide image for Multi Image Text Encoder V3

    Use Case:
    - Create position reference for "rectangle 1 = flower, rectangle 2 = man" prompts
    - Automate guide image creation from ComfyUI masks
    - No manual rectangle drawing needed!

    Based on research:
    - Position guide workflow (number-based mapping)
    - Proven to work with Qwen Edit 2509
    - User validated: "it is working"
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ArchAi3D_Mask_To_Position_Guide",
            category="ArchAi3d/Utils",
            inputs=[
                # Input mask
                io.Mask.Input(
                    "mask",
                    tooltip="Input mask with selected regions (white = selected, black = background)"
                ),

                # Numbering order
                io.Combo.Input(
                    "numbering_order",
                    options=["left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top"],
                    default="left_to_right",
                    tooltip=(
                        "Order for numbering regions:\n"
                        "- left_to_right: Numbers increase from left to right (→)\n"
                        "- right_to_left: Numbers increase from right to left (←)\n"
                        "- top_to_bottom: Numbers increase from top to bottom (↓)\n"
                        "- bottom_to_top: Numbers increase from bottom to top (↑)"
                    )
                ),

                # Rectangle settings
                io.Combo.Input(
                    "rectangle_color",
                    options=["red", "blue", "green", "yellow", "magenta", "cyan", "custom_hex"],
                    default="red",
                    tooltip="Color for rectangle borders (use 'custom_hex' to define hexadecimal color code below)"
                ),

                io.String.Input(
                    "rect_hex_color",
                    default="FF0000",
                    tooltip="Hexadecimal color code for rectangles (e.g., 'FF0000' or '#FF0000') - only used when rectangle_color='custom_hex'"
                ),

                io.Int.Input(
                    "line_thickness",
                    default=2,
                    min=1,
                    max=20,
                    tooltip="Thickness of rectangle borders in pixels"
                ),

                # Number settings
                io.Int.Input(
                    "number_size",
                    default=18,
                    min=12,
                    max=200,
                    tooltip="Font size for numbers inside rectangles"
                ),

                io.Combo.Input(
                    "number_color",
                    options=["red", "white", "black", "blue", "green", "yellow", "custom_hex"],
                    default="red",
                    tooltip="Color for numbers (use 'custom_hex' to define hexadecimal color code below)"
                ),

                io.String.Input(
                    "num_hex_color",
                    default="FF0000",
                    tooltip="Hexadecimal color code for numbers (e.g., 'FF0000' or '#FF0000') - only used when number_color='custom_hex'"
                ),

                # Detection settings
                io.Int.Input(
                    "padding",
                    default=5,
                    min=0,
                    max=50,
                    tooltip="Padding around detected regions in pixels"
                ),

                # Background settings
                io.Combo.Input(
                    "background_color",
                    options=["custom_hex", "black", "white", "red", "blue", "green", "yellow", "magenta", "cyan"],
                    default="custom_hex",
                    tooltip="Background color for the guide image (use 'custom_hex' to define hexadecimal color code below)"
                ),

                # Custom hex input (default: 0F0F0F = RGB(15, 15, 15) - dark gray)
                io.String.Input(
                    "bg_hex_color",
                    default="0F0F0F",
                    tooltip="Hexadecimal color code for background (e.g., '0F0F0F' or '#0F0F0F') - only used when background_color='custom_hex'"
                ),
            ],
            outputs=[
                io.Image.Output(
                    "guide_image",
                    tooltip="Position guide image with numbered rectangles (use with Multi Image Text Encoder V3)"
                ),
                io.Int.Output(
                    "region_count",
                    tooltip="Number of regions detected"
                ),
                io.String.Output(
                    "bbox_list",
                    tooltip="JSON list of bounding boxes for each region"
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        mask,
        numbering_order,
        rectangle_color,
        rect_hex_color,
        line_thickness,
        number_size,
        number_color,
        num_hex_color,
        padding,
        background_color,
        bg_hex_color,
    ) -> io.NodeOutput:
        """Execute the Mask to Position Guide node.

        Steps:
        1. Convert mask tensor to PIL Image
        2. Detect regions and get bounding boxes
        3. Sort regions by selected order and assign numbers
        4. Draw numbered rectangles
        5. Convert back to tensor and output
        """

        # Step 1: Convert mask to PIL
        mask_pil = mask_to_pil(mask)

        # Step 2: Detect regions with selected numbering order
        regions = detect_regions(
            mask_pil,
            padding=padding,
            numbering_order=numbering_order
        )

        # Step 3: Draw numbered rectangles
        guide_pil = draw_numbered_rectangles(
            mask_pil=mask_pil,
            regions=regions,
            rectangle_color=rectangle_color,
            line_thickness=line_thickness,
            number_size=number_size,
            number_color=number_color,
            background_color=background_color,
            bg_custom_hex=bg_hex_color if background_color == "custom_hex" else None,
            rect_custom_hex=rect_hex_color if rectangle_color == "custom_hex" else None,
            num_custom_hex=num_hex_color if number_color == "custom_hex" else None
        )

        # Step 4: Convert to tensor
        guide_tensor = pil_to_tensor(guide_pil)

        # Step 5: Prepare outputs
        region_count = len(regions)

        # Create JSON bbox list
        bbox_list = json.dumps([
            {
                "number": r["number"],
                "bbox": r["bbox"],
                "center_x": r["center_x"],
                "center_y": r["center_y"]
            }
            for r in regions
        ], indent=2)

        # Debug output
        print(f"\n{'='*70}")
        print(f"ArchAi3D_Mask_To_Position_Guide - Detected {region_count} regions")
        print(f"Numbering Order: {numbering_order}")

        if background_color == "custom_hex":
            bg_rgb = hex_to_rgb(bg_hex_color)
            print(f"Background Color: custom_hex (#{bg_hex_color.lstrip('#')}) = RGB{bg_rgb}")
        else:
            print(f"Background Color: {background_color}")

        if rectangle_color == "custom_hex":
            rect_rgb = hex_to_rgb(rect_hex_color)
            print(f"Rectangle Color: custom_hex (#{rect_hex_color.lstrip('#')}) = RGB{rect_rgb}")
        else:
            print(f"Rectangle Color: {rectangle_color}")

        if number_color == "custom_hex":
            num_rgb = hex_to_rgb(num_hex_color)
            print(f"Number Color: custom_hex (#{num_hex_color.lstrip('#')}) = RGB{num_rgb}")
        else:
            print(f"Number Color: {number_color}")

        print(f"{'='*70}")
        for region in regions:
            print(f"Region {region['number']}: bbox={region['bbox']}, center=({region['center_x']}, {region['center_y']})")
        print(f"{'='*70}\n")

        return io.NodeOutput(guide_tensor, region_count, bbox_list)


# ============================================================================
# EXTENSION REGISTRATION
# ============================================================================

class MaskToPositionGuideExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [ArchAi3D_Mask_To_Position_Guide]


async def comfy_entrypoint():
    return MaskToPositionGuideExtension()
