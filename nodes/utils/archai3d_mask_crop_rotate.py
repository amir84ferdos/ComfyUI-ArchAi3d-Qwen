# -*- coding: utf-8 -*-
"""
ArchAi3D Mask Crop & Rotate Node
Author: Amir Ferdos (ArchAi3d)
Email: Amir84ferdos@gmail.com
LinkedIn: https://www.linkedin.com/in/archai3d/

Description:
    Crops an image based on mask bounding box and optionally rotates it.
    Detects the region in the mask, crops the image to that region,
    and applies user-defined rotation.

Usage:
    1. Input: Image + Mask (white = region to crop)
    2. Choose: Rotation angle (0-360 degrees)
    3. Output: Cropped and rotated image

Version: 1.0.0
Created: 2025-10-18
"""

import numpy as np
import torch
from PIL import Image
import cv2
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

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


def image_to_pil(image_tensor):
    """Convert ComfyUI image tensor to PIL Image.

    Args:
        image_tensor: Tensor of shape (B, H, W, C) with values 0-1

    Returns:
        PIL Image in 'RGB' mode
    """
    # Take first image if batched
    if len(image_tensor.shape) == 4:
        image_tensor = image_tensor[0]

    # Convert to numpy and scale to 0-255
    img_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)

    return Image.fromarray(img_np, mode='RGB')


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


def get_mask_bbox(mask_pil, padding=0):
    """Get bounding box of mask region.

    Args:
        mask_pil: PIL Image in 'L' mode (grayscale mask)
        padding: Pixels to pad around detected region

    Returns:
        Tuple (x1, y1, x2, y2) or None if no region found
    """
    # Convert to numpy for OpenCV
    mask_np = np.array(mask_pil)

    # Threshold to binary (white = 255, black = 0)
    _, binary = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)

    # Find contours (connected components)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Get bounding box that encompasses all contours
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = 0, 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    # Add padding
    x1 = max(0, x_min - padding)
    y1 = max(0, y_min - padding)
    x2 = min(mask_pil.width, x_max + padding)
    y2 = min(mask_pil.height, y_max + padding)

    return (x1, y1, x2, y2)


def rotate_image(pil_img, angle, expand=True, fill_color=(0, 0, 0)):
    """Rotate PIL image by given angle.

    Args:
        pil_img: PIL Image to rotate
        angle: Rotation angle in degrees (counter-clockwise)
        expand: If True, expand canvas to fit rotated image
        fill_color: RGB color for background (if expand=True)

    Returns:
        Rotated PIL Image
    """
    # Rotate image
    rotated = pil_img.rotate(angle, expand=expand, fillcolor=fill_color, resample=Image.BICUBIC)

    return rotated


# ============================================================================
# NODE DEFINITION
# ============================================================================

class ArchAi3D_Mask_Crop_Rotate(io.ComfyNode):
    """Mask Crop & Rotate: Crop image based on mask and apply rotation.

    This node detects the region in a mask, crops the input image to that
    region's bounding box, and optionally rotates the result.

    Workflow:
    1. Detect mask region bounding box
    2. Crop image to bounding box (with optional padding)
    3. Rotate cropped image by specified angle
    4. Output cropped and rotated image

    Use Cases:
    - Extract specific regions from images
    - Prepare cropped regions for further processing
    - Auto-crop to mask area with rotation correction
    - Extract and orient objects
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ArchAi3D_Mask_Crop_Rotate",
            category="ArchAi3d/Utils",
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="Input image to crop"
                ),
                io.Mask.Input(
                    "mask",
                    tooltip="Mask defining region to crop (white = region to keep)"
                ),
                io.Float.Input(
                    "rotation_angle",
                    default=0.0,
                    min=-360.0,
                    max=360.0,
                    step=1.0,
                    tooltip="Rotation angle in degrees (counter-clockwise). 0 = no rotation"
                ),
                io.Int.Input(
                    "padding",
                    default=0,
                    min=0,
                    max=200,
                    tooltip="Padding around mask region in pixels"
                ),
                io.Combo.Input(
                    "expand_canvas",
                    options=["yes", "no"],
                    default="yes",
                    tooltip=(
                        "Expand canvas to fit rotated image:\n"
                        "- yes: Canvas expands to fit entire rotated image (no cropping)\n"
                        "- no: Keep original crop size (may crop corners after rotation)"
                    )
                ),
                io.Combo.Input(
                    "background_color",
                    options=["black", "white", "custom_hex"],
                    default="black",
                    tooltip="Background color for expanded canvas (when expand_canvas=yes)"
                ),
                io.String.Input(
                    "bg_hex_color",
                    default="000000",
                    tooltip="Hexadecimal color code for background (e.g., '000000' for black) - only used when background_color='custom_hex'"
                ),
            ],
            outputs=[
                io.Image.Output(
                    "cropped_image",
                    tooltip="Cropped and rotated image"
                ),
                io.Int.Output(
                    "crop_width",
                    tooltip="Width of cropped region (before rotation)"
                ),
                io.Int.Output(
                    "crop_height",
                    tooltip="Height of cropped region (before rotation)"
                ),
                io.String.Output(
                    "crop_bbox",
                    tooltip="Bounding box coordinates: [x1, y1, x2, y2]"
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        mask,
        rotation_angle,
        padding,
        expand_canvas,
        background_color,
        bg_hex_color,
    ) -> io.NodeOutput:
        """Execute the mask crop and rotate operation.

        Steps:
        1. Convert mask to PIL and find bounding box
        2. Convert image to PIL and crop to bounding box
        3. Rotate cropped image by specified angle
        4. Convert back to tensor and output
        """

        # Step 1: Convert mask to PIL and get bounding box
        mask_pil = mask_to_pil(mask)
        bbox = get_mask_bbox(mask_pil, padding=padding)

        if bbox is None:
            # No region found in mask - return empty/error
            print("Warning: No region found in mask. Returning original image.")
            return io.NodeOutput(
                image,
                image.shape[2],  # width
                image.shape[1],  # height
                "No region found"
            )

        x1, y1, x2, y2 = bbox

        # Step 2: Convert image to PIL and crop
        img_pil = image_to_pil(image)
        cropped_pil = img_pil.crop((x1, y1, x2, y2))

        crop_width = x2 - x1
        crop_height = y2 - y1

        # Step 3: Rotate if angle is not zero
        if rotation_angle != 0:
            # Get background color
            if background_color == "custom_hex":
                # Convert hex to RGB
                hex_color = bg_hex_color.lstrip('#')
                try:
                    r = int(hex_color[0:2], 16)
                    g = int(hex_color[2:4], 16)
                    b = int(hex_color[4:6], 16)
                    bg_color = (r, g, b)
                except (ValueError, IndexError):
                    print(f"Warning: Invalid hex color '{bg_hex_color}', using black")
                    bg_color = (0, 0, 0)
            elif background_color == "white":
                bg_color = (255, 255, 255)
            else:  # black
                bg_color = (0, 0, 0)

            # Rotate
            expand = (expand_canvas == "yes")
            cropped_pil = rotate_image(
                cropped_pil,
                angle=rotation_angle,
                expand=expand,
                fill_color=bg_color
            )

        # Step 4: Convert back to tensor
        output_tensor = pil_to_tensor(cropped_pil)

        # Prepare bbox string
        bbox_str = f"[{x1}, {y1}, {x2}, {y2}]"

        # Debug output
        print("\n" + "="*70)
        print("✂️ ArchAi3D Mask Crop & Rotate - v1.0.0")
        print("="*70)
        print(f"Original image size: {img_pil.width}x{img_pil.height}")
        print(f"Crop bounding box: {bbox_str}")
        print(f"Crop size (before rotation): {crop_width}x{crop_height}")
        print(f"Padding: {padding}px")
        print(f"Rotation angle: {rotation_angle}°")
        print(f"Expand canvas: {expand_canvas}")
        if rotation_angle != 0:
            print(f"Background color: {background_color}")
            if background_color == "custom_hex":
                print(f"  Hex: #{bg_hex_color} = RGB{bg_color}")
        print(f"Output size: {cropped_pil.width}x{cropped_pil.height}")
        print("="*70 + "\n")

        return io.NodeOutput(output_tensor, crop_width, crop_height, bbox_str)


# ============================================================================
# EXTENSION REGISTRATION
# ============================================================================

class MaskCropRotateExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [ArchAi3D_Mask_Crop_Rotate]


async def comfy_entrypoint():
    return MaskCropRotateExtension()
