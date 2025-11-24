# -*- coding: utf-8 -*-
"""
ArchAi3D Solid Color Image Generator Node
Author: Amir Ferdos (ArchAi3d)
Email: Amir84ferdos@gmail.com
LinkedIn: https://www.linkedin.com/in/archai3d/

Description:
    Creates a solid color (flat) image with dimensions from reference image OR panoramic presets.
    Color can be defined using RGB values (0-255) or hexadecimal color codes.

Usage:
    1. Optional: Connect reference image (for dimension extraction) OR select preset
    2. Choose color mode (RGB or Hex)
    3. Define color using RGB sliders or hex code
    4. Get solid color image

Presets:
    13 panoramic presets (2:1 aspect ratio):
    - Range: 1024×512 (0.52 MP) to 3072×1536 (4.72 MP)
    - All divisible by 32 for optimal AI processing
    - Perfect for MoGe-2, Qwen, and panorama workflows

Use Cases:
    - Create panoramic base images for AI generation
    - Generate backgrounds for compositing
    - Base images for position guides
    - Test different resolutions for VRAM/quality optimization

Version: 2.0.0
"""

import torch
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override


# ============================================================================
# PRESET DIMENSIONS (2:1 Panorama - All divisible by 32)
# ============================================================================

PRESET_DIMENSIONS = {
    "2:1 - 512 (1024×512)": (1024, 512),      # 0.52 MP
    "2:1 - 576 (1152×576)": (1152, 576),      # 0.66 MP
    "2:1 - 640 (1280×640)": (1280, 640),      # 0.82 MP
    "2:1 - 704 (1408×704)": (1408, 704),      # 0.99 MP
    "2:1 - 768 (1536×768)": (1536, 768),      # 1.18 MP - Default
    "2:1 - 832 (1664×832)": (1664, 832),      # 1.38 MP
    "2:1 - 896 (1792×896)": (1792, 896),      # 1.60 MP
    "2:1 - 960 (1920×960)": (1920, 960),      # 1.84 MP
    "2:1 - 1024 (2048×1024)": (2048, 1024),   # 2.10 MP
    "2:1 - 1088 (2176×1088)": (2176, 1088),   # 2.37 MP
    "2:1 - 1152 (2304×1152)": (2304, 1152),   # 2.65 MP
    "2:1 - 1280 (2560×1280)": (2560, 1280),   # 3.28 MP
    "2:1 - 1536 (3072×1536)": (3072, 1536),   # 4.72 MP
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
        # Invalid hex code, return gray as fallback
        print(f"Warning: Invalid hex color '{hex_color}', using gray (808080) as fallback")
        return (128, 128, 128)


def rgb_to_hex(r, g, b):
    """Convert RGB values to hexadecimal color code.

    Args:
        r, g, b: RGB values (0-255)

    Returns:
        Hex color string (e.g., "FF0000")
    """
    return f"{r:02X}{g:02X}{b:02X}"


# ============================================================================
# NODE DEFINITION
# ============================================================================

class ArchAi3D_Solid_Color_Image(io.ComfyNode):
    """Solid Color Image: Create flat color image with reference or preset dimensions.

    This node creates a solid color image using either reference image dimensions
    or panoramic presets (2:1 aspect ratio). Color can be defined using RGB values or hex codes.

    Workflow:
    1. Optional: Provide reference image (dimensions extracted) OR select preset
    2. Choose color mode (RGB or Hex)
    3. Define color using sliders or hex code
    4. Get solid color output

    Presets:
    - 13 panoramic sizes (2:1 aspect ratio)
    - Range: 1024×512 (0.52 MP) to 3072×1536 (4.72 MP)
    - All divisible by 32 for optimal AI processing

    Use Cases:
    - Panorama base images: Perfect for MoGe-2, Qwen panorama workflows
    - Create backgrounds: Match scene size with custom color
    - Position guide base: RGB(15,15,15) base for guide rectangles
    - Testing: Cycle through different resolutions for VRAM/quality tests
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ArchAi3D_Solid_Color_Image",
            category="ArchAi3d/Utils",
            inputs=[
                io.Image.Input(
                    "reference_image",
                    optional=True,
                    tooltip="Optional: Reference image to extract dimensions from. If not provided, uses preset dimensions below."
                ),

                io.Combo.Input(
                    "preset_dimensions",
                    options=list(PRESET_DIMENSIONS.keys()),
                    default="2:1 - 768 (1536×768)",
                    tooltip="Preset panoramic dimensions (2:1 aspect ratio). Used when no reference image is provided. All sizes divisible by 32 for optimal AI processing. Range: 0.52 MP to 4.72 MP."
                ),

                io.Combo.Input(
                    "color_mode",
                    options=["rgb", "hex"],
                    default="rgb",
                    tooltip="Choose color definition method: RGB values (0-255) or hexadecimal code"
                ),

                # RGB inputs
                io.Int.Input(
                    "rgb_r",
                    default=128,
                    min=0,
                    max=255,
                    tooltip="Red channel value (0-255) - only used when color_mode='rgb'"
                ),
                io.Int.Input(
                    "rgb_g",
                    default=128,
                    min=0,
                    max=255,
                    tooltip="Green channel value (0-255) - only used when color_mode='rgb'"
                ),
                io.Int.Input(
                    "rgb_b",
                    default=128,
                    min=0,
                    max=255,
                    tooltip="Blue channel value (0-255) - only used when color_mode='rgb'"
                ),

                # Hex input
                io.String.Input(
                    "hex_color",
                    default="808080",
                    tooltip="Hexadecimal color code (e.g., 'FF0000' or '#FF0000') - only used when color_mode='hex'"
                ),

                # Debug
                io.Boolean.Input(
                    "debug_mode",
                    default=False,
                    tooltip="Enable console logging (shows dimensions, color info)"
                ),
            ],
            outputs=[
                io.Image.Output(
                    "solid_image",
                    tooltip="Solid color image matching reference dimensions"
                ),
            ],
        )

    @classmethod
    def execute(cls, reference_image=None, preset_dimensions="2:1 - 768 (1536×768)", color_mode="rgb", rgb_r=128, rgb_g=128, rgb_b=128, hex_color="808080", debug_mode=False) -> io.NodeOutput:
        """Execute the Solid Color Image generator.

        Steps:
        1. Determine dimensions (from reference image or preset)
        2. Get color based on mode (RGB or Hex)
        3. Create solid color tensor
        4. Return image
        """

        # ============================================================
        # SECTION 1: Determine Dimensions (Reference or Preset)
        # ============================================================
        if reference_image is not None:
            # Use reference image dimensions
            batch, height, width, channels = reference_image.shape
            dimension_source = "Reference Image"
        else:
            # Use preset dimensions
            width, height = PRESET_DIMENSIONS[preset_dimensions]
            batch = 1  # Default batch size for preset mode
            dimension_source = f"Preset: {preset_dimensions}"

        # ============================================================
        # SECTION 2: Get Color Based on Mode
        # ============================================================
        if color_mode == "hex":
            # Convert hex to RGB
            r, g, b = hex_to_rgb(hex_color)
        else:  # rgb
            # Use RGB values directly
            r, g, b = rgb_r, rgb_g, rgb_b

        # Clamp RGB values to 0-255 range (safety check)
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))

        # ============================================================
        # SECTION 3: Create Solid Color Tensor
        # ============================================================
        # Convert RGB (0-255) to float (0-1) for ComfyUI
        color_normalized = torch.tensor([r/255.0, g/255.0, b/255.0], dtype=torch.float32)

        # Create solid color image tensor
        # Shape: (batch, height, width, 3)
        solid_tensor = torch.ones(batch, height, width, 3, dtype=torch.float32)
        solid_tensor = solid_tensor * color_normalized.view(1, 1, 1, 3)

        # ============================================================
        # SECTION 4: Debug Output
        # ============================================================
        if debug_mode:
            hex_repr = rgb_to_hex(r, g, b)
            print(f"\n{'='*70}")
            print(f"ArchAi3D_Solid_Color_Image - Generator")
            print(f"{'='*70}")
            print(f"Dimension Source: {dimension_source}")
            print(f"Output Dimensions: {width}×{height} (batch={batch})")
            if reference_image is not None:
                print(f"  ✓ Extracted from reference image")
            else:
                megapixels = (width * height) / 1_000_000
                div_32_w = "✓" if width % 32 == 0 else "✗"
                div_32_h = "✓" if height % 32 == 0 else "✗"
                print(f"  ✓ Using preset: {preset_dimensions}")
                print(f"  ✓ Megapixels: {megapixels:.2f} MP")
                print(f"  ✓ Divisible by 32: Width {div_32_w} ({width}÷32={width//32}), Height {div_32_h} ({height}÷32={height//32})")
            print(f"Color Mode: {color_mode}")
            if color_mode == "hex":
                print(f"  Input Hex: #{hex_color.lstrip('#')}")
                print(f"  Converted RGB: ({r}, {g}, {b})")
            else:
                print(f"  Input RGB: ({r}, {g}, {b})")
                print(f"  Hex Equivalent: #{hex_repr}")
            print(f"Output Shape: {solid_tensor.shape}")
            print(f"Output Color (normalized): ({r/255.0:.3f}, {g/255.0:.3f}, {b/255.0:.3f})")
            print(f"{'='*70}\n")

        # ============================================================
        # SECTION 5: Return Output
        # ============================================================
        return io.NodeOutput(solid_tensor)


# ============================================================================
# EXTENSION REGISTRATION
# ============================================================================

class SolidColorImageExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [ArchAi3D_Solid_Color_Image]


async def comfy_entrypoint():
    return SolidColorImageExtension()
