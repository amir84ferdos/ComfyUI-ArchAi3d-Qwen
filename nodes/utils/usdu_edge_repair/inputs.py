"""
USDU Edge Repair - Input Definitions
=====================================

Input definitions for the USDU Edge Repair ComfyUI node.
"""

import comfy
from .constants import MAX_RESOLUTION, MODES, SEAM_FIX_MODES


def USDU_edge_repair_inputs():
    """
    Define inputs for USDU Edge Repair node.

    Returns:
        Tuple of (required_inputs, optional_inputs)
    """
    required = [
        ("upscaled_image", ("IMAGE", {"tooltip": "Pre-upscaled image from Matrix Search"})),
        # Matrix Search values
        ("output_width", ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 1})),
        ("output_height", ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 1})),
        ("tiles_x", ("INT", {"default": 2, "min": 1, "max": 64, "step": 1})),
        ("tiles_y", ("INT", {"default": 2, "min": 1, "max": 64, "step": 1})),
        # Toggles
        ("safe_guard", ("BOOLEAN", {"default": True})),
        ("enable_diffdiff", ("BOOLEAN", {"default": True})),
        ("enable_controlnet", ("BOOLEAN", {"default": True})),
        ("preview_mode", ("BOOLEAN", {"default": False,
                                       "tooltip": "Preview mode: Output tile images + masks for verification"})),
        # Sampling
        ("model", ("MODEL",)),
        ("conditionings", ("CONDITIONING_LIST",)),
        ("negative", ("CONDITIONING",)),
        ("vae", ("VAE",)),
        ("seed", ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})),
        ("steps", ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1})),
        ("cfg", ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0})),
        ("sampler_name", (comfy.samplers.KSampler.SAMPLERS,)),
        ("scheduler", (comfy.samplers.KSampler.SCHEDULERS,)),
        ("denoise", ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01})),
        # Tile params
        ("mode_type", (list(MODES.keys()),)),
        ("tile_width", ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8})),
        ("tile_height", ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8})),
        ("mask_blur", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1})),
        ("tile_padding", ("INT", {"default": 32, "min": 0, "max": MAX_RESOLUTION, "step": 8})),
        # Edge mask params
        ("edge_mask_width", ("INT", {"default": 20, "min": 0, "max": 128, "step": 1})),
        ("edge_mask_feather", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1})),
        ("use_edge_mask_diffdiff", ("BOOLEAN", {"default": False,
                                                 "tooltip": "Apply edge mask to DiffDiff for smoother tile blending"})),
        # Seam fix
        ("seam_fix_mode", (list(SEAM_FIX_MODES.keys()),)),
        ("seam_fix_denoise", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})),
        ("seam_fix_width", ("INT", {"default": 64, "min": 0, "max": MAX_RESOLUTION, "step": 8})),
        ("seam_fix_mask_blur", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1})),
        ("seam_fix_padding", ("INT", {"default": 16, "min": 0, "max": MAX_RESOLUTION, "step": 8})),
        # Misc
        ("tiled_decode", ("BOOLEAN", {"default": False})),
    ]

    optional = [
        ("denoise_mask", ("MASK", {"tooltip": "Optional mask for per-pixel denoise"})),
        ("multiplier", ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001})),
        ("model_patch", ("MODEL_PATCH", {"tooltip": "ControlNet patch"})),
        ("control_image", ("IMAGE", {"tooltip": "Control image for ControlNet"})),
        ("control_strength", ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})),
        ("control_mask", ("MASK", {"tooltip": "Optional mask for ControlNet"})),
    ]

    return required, optional


def prepare_inputs(required, optional=None):
    """
    Convert input lists to ComfyUI INPUT_TYPES format.

    Args:
        required: List of (name, type_spec) tuples
        optional: List of (name, type_spec) tuples

    Returns:
        Dict in ComfyUI INPUT_TYPES format
    """
    inputs = {}
    if required:
        inputs["required"] = {name: type_spec for name, type_spec in required}
    if optional:
        inputs["optional"] = {name: type_spec for name, type_spec in optional}
    return inputs
