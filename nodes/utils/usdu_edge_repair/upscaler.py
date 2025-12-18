"""
USDU Edge Repair - Upscaler Module
===================================

TREE MAP:
---------
upscaler.py
├── Upscaler (class)              - Handles image upscaling operations
│   └── upscale(img, scale)       - Upscale image using model or LANCZOS
│
└── UpscalerData (class)          - Data wrapper for upscaler (A1111 compatibility)
    ├── name                      - Upscaler name (display)
    ├── data_path                 - Path to upscaler model
    └── scaler                    - Upscaler instance

DATA FLOW:
----------
1. UpscalerData wraps an Upscaler instance
2. USDUpscaler (ultimate_upscale.py) calls upscaler.scaler.upscale()
3. Upscaler.upscale() either:
   a) Uses LANCZOS if no model (shared.actual_upscaler is None)
   b) Uses ImageUpscaleWithModel if model is set
4. Result is stored in shared.batch

USAGE:
------
- For USDU Edge Repair: actual_upscaler is None (image is pre-upscaled)
- For regular USDU: actual_upscaler contains the upscaler model

NOTES:
------
- selected_model parameter in upscale() is unused (kept for compatibility)
- Supports both old and new ComfyUI API (execute vs upscale method)
"""

from PIL import Image
from .utils import tensor_to_pil, pil_to_tensor
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
from . import shared

# Pillow compatibility: older versions don't have Resampling enum
if (not hasattr(Image, 'Resampling')):
    Image.Resampling = Image


class Upscaler:
    """
    Handles image upscaling using either:
    1. A neural network upscaler model (ESRGAN, RealESRGAN, etc.)
    2. Simple LANCZOS resampling (fallback when no model)

    The actual upscaler model is stored in shared.actual_upscaler.
    When None, LANCZOS is used instead.
    """

    def upscale(self, img: Image, scale, selected_model: str = None):
        """
        Upscale an image by the given scale factor.

        Args:
            img: PIL Image to upscale
            scale: Scale factor (e.g., 2.0 for 2x upscale)
            selected_model: UNUSED - kept for A1111 compatibility

        Returns:
            PIL Image: Upscaled image

        Behavior:
            - If scale == 1.0: Return original (no upscaling needed)
            - If no model (actual_upscaler is None): Use LANCZOS resize
            - If model exists: Use neural network upscaler

        Note:
            When using a model, the ENTIRE batch is upscaled at once
            (shared.batch_as_tensor), but only the first image is returned.
            The full batch is stored in shared.batch.
        """
        # No upscaling needed
        if scale == 1.0:
            return img

        # Fallback: No model available, use simple resize
        if (shared.actual_upscaler is None):
            return img.resize((img.width * scale, img.height * scale), Image.Resampling.LANCZOS)

        # Use neural network upscaler model
        # Check for ComfyUI V3 API (execute method) vs old API (upscale method)
        if "execute" in dir(ImageUpscaleWithModel):
            # V3 schema: https://github.com/comfyanonymous/ComfyUI/pull/10149
            (upscaled,) = ImageUpscaleWithModel.execute(shared.actual_upscaler, shared.batch_as_tensor)
        else:
            # Old API: instance method
            (upscaled,) = ImageUpscaleWithModel().upscale(shared.actual_upscaler, shared.batch_as_tensor)

        # Convert tensor back to PIL images and store in shared batch
        shared.batch = [tensor_to_pil(upscaled, i) for i in range(len(upscaled))]

        # Return first image (USDU processes one at a time)
        return shared.batch[0]


class UpscalerData:
    """
    A1111-compatible wrapper for upscaler data.

    This class mimics the structure expected by the original Ultimate SD Upscale
    script from Automatic1111's WebUI. It holds:
    - Metadata about the upscaler (name, path)
    - An Upscaler instance that does the actual work

    Attributes:
        name (str): Display name of the upscaler (e.g., "RealESRGAN_x4plus")
        data_path (str): Path to the upscaler model file
        scaler (Upscaler): The actual upscaler instance

    Usage:
        upscaler_data = UpscalerData()
        result = upscaler_data.scaler.upscale(image, scale=2.0)
    """
    name = ""        # Upscaler display name
    data_path = ""   # Path to model file

    def __init__(self):
        """Create UpscalerData with a new Upscaler instance."""
        self.scaler = Upscaler()
