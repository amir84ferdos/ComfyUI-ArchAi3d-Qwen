# ArchAi3D GRAG Attention Utilities
#
# OVERVIEW:
# Implements GRAG (Group-Relative Attention Guidance) attention key reweighting
# for fine-grained image editing control in Diffusion-in-Transformer (DiT) models.
#
# GRAG ALGORITHM:
# Based on arXiv paper 2510.24657 (October 2024)
#
# Mathematical formulation:
#   1. Decompose keys: k_i = k_bias + Δk_i
#   2. Group bias: k_bias = mean(k_1, k_2, ..., k_N)
#   3. Token deviation: Δk_i = k_i - k_bias
#   4. Reweight: k̂_i = λ * k_bias + δ * Δk_i
#
# Where:
#   - λ (lambda/cond_b): Controls bias strength (>1 enhances, <1 reduces)
#   - δ (delta/cond_delta): Controls deviation intensity
#
# INTEGRATION POINT:
# Applied AFTER rotary position embeddings (RoPE), BEFORE attention computation
#
# Author: Amir Ferdos (ArchAi3d)
# Email: Amir84ferdos@gmail.com
# LinkedIn: https://www.linkedin.com/in/archai3d/
# GitHub: https://github.com/amir84ferdos
# Based on: GRAG-Image-Editing by little-misfit
# License: MIT

import torch


def apply_grag_to_keys(joint_key, seq_txt, lambda_val, delta_val, heads):
    """Apply GRAG reweighting to joint attention keys.

    This implements the core GRAG algorithm that decomposes attention keys into
    group bias and token-specific deviations, then reweights them independently
    for fine-grained editing control.

    The algorithm operates on two separate streams:
    - Text stream: First seq_txt tokens (prompt/instructions)
    - Image stream: Remaining tokens (visual content)

    Args:
        joint_key (torch.Tensor): Joint attention keys [B, S, C] after RoPE
            B = batch size
            S = sequence length (text + image tokens)
            C = channels (heads * head_dim)
        seq_txt (int): Length of text sequence (separates text/image streams)
        lambda_val (float): Bias strength parameter (cond_b)
            - >1.0: Enhances group editing direction
            - <1.0: Reduces group influence
            - 1.0: Neutral (no change to bias)
        delta_val (float): Deviation strength parameter (cond_delta)
            - >1.0: Concentrates token-specific details
            - <1.0: Diffuses individual variations
            - 1.0: Neutral (no change to deviation)
        heads (int): Number of attention heads

    Returns:
        torch.Tensor: Modified joint keys with GRAG reweighting [B, S, C]

    Mathematical Operations:
        For each stream (text and image):
        1. k_mean = mean(k_tokens, dim=1)  # Group bias
        2. Δk = k - k_mean                 # Token deviations
        3. k_reweighted = λ * k_mean + δ * Δk
    """
    # Get tensor dimensions
    batch, seq, channels = joint_key.shape
    head_dim = channels // heads

    # Reshape from ComfyUI format [B, S, C] to GRAG format [B, S, H, D]
    # This separates the heads dimension for per-head operations
    joint_key = joint_key.unflatten(-1, (heads, head_dim))

    # ===== TEXT STREAM GRAG =====
    # Extract text tokens (first seq_txt positions)
    txt_key = joint_key[:, :seq_txt, :, :]  # [B, seq_txt, H, D]

    # Compute group mean (bias vector) across token dimension
    txt_key_mean = txt_key.mean(dim=1, keepdim=True)  # [B, 1, H, D]

    # Apply GRAG reweighting: k̂ = λ * k_bias + δ * (k - k_bias)
    # Equivalent to: k̂ = λ * k_mean + δ * Δk
    txt_key = lambda_val * txt_key_mean + delta_val * (txt_key - txt_key_mean)

    # ===== IMAGE STREAM GRAG =====
    # Extract image tokens (remaining positions after text)
    img_key = joint_key[:, seq_txt:, :, :]  # [B, seq_img, H, D]

    # Compute group mean (bias vector) across token dimension
    img_key_mean = img_key.mean(dim=1, keepdim=True)  # [B, 1, H, D]

    # Apply GRAG reweighting
    img_key = lambda_val * img_key_mean + delta_val * (img_key - img_key_mean)

    # ===== RECOMBINE STREAMS =====
    # Concatenate text and image streams back together
    joint_key = torch.cat([txt_key, img_key], dim=1)  # [B, S, H, D]

    # Reshape back to ComfyUI format [B, S, C]
    joint_key = joint_key.flatten(start_dim=2)  # [B, S, H*D]

    return joint_key


def create_grag_patch(grag_config):
    """Factory function creating GRAG attention patch for ComfyUI.

    Creates a patch function that can be injected into ComfyUI's attention
    pipeline via transformer_options["patches"]. The patch intercepts
    attention keys after RoPE and applies GRAG reweighting if enabled.

    Args:
        grag_config (dict): GRAG configuration with keys:
            - "enabled" (bool): Whether GRAG is active
            - "lambda" (float): Bias strength (cond_b)
            - "delta" (float): Deviation strength (cond_delta)
            - "heads" (int): Number of attention heads

    Returns:
        callable: Patch function with signature patch(args) -> args
            The patch function receives and returns args dict containing
            attention computation parameters.

    Usage:
        grag_config = {
            "enabled": True,
            "lambda": 1.0,
            "delta": 1.0,
            "heads": 16
        }
        patch_fn = create_grag_patch(grag_config)
        transformer_options["patches"]["attention_pre"] = [patch_fn]
    """
    def grag_patch(args):
        """Attention patch function that applies GRAG reweighting.

        Args:
            args (dict): Attention computation arguments, should contain:
                - "joint_key": Attention keys after RoPE [B, S, C]
                - "seq_txt": Text sequence length
                - (other attention parameters)

        Returns:
            dict: Modified args with GRAG-reweighted keys
        """
        # Check if GRAG is enabled
        if not grag_config.get("enabled", False):
            return args

        # Extract required parameters from args
        joint_key = args.get("joint_key")
        seq_txt = args.get("seq_txt")

        # Validate that we have the necessary data
        if joint_key is None or seq_txt is None:
            # Missing required data, pass through unchanged
            return args

        # Apply GRAG reweighting to keys
        try:
            joint_key = apply_grag_to_keys(
                joint_key,
                seq_txt,
                grag_config["lambda"],
                grag_config["delta"],
                grag_config["heads"]
            )

            # Update args with modified keys
            args["joint_key"] = joint_key

        except Exception as e:
            # If GRAG fails, pass through original keys (graceful degradation)
            print(f"[GRAG] Warning: Reweighting failed, using original keys: {e}")
            pass

        return args

    return grag_patch


def extract_grag_config_from_conditioning(conditioning):
    """Extract GRAG configuration from ComfyUI conditioning metadata.

    Reads GRAG parameters embedded in conditioning by the GRAG Modifier
    or GRAG Encoder nodes. Returns None if GRAG is not enabled.

    Args:
        conditioning (list): ComfyUI conditioning format
            [(embeddings_tensor, metadata_dict), ...]

    Returns:
        dict or None: GRAG config dict if enabled, None otherwise
            Dict format: {
                "enabled": bool,
                "lambda": float,
                "delta": float,
                "heads": int
            }

    Example conditioning metadata:
        {
            "grag_enabled": True,
            "grag_cond_b": 1.0,
            "grag_cond_delta": 1.0,
            "grag_strength": 1.0,
            ...
        }
    """
    # Validate conditioning format
    if not conditioning or len(conditioning) == 0:
        return None

    if len(conditioning[0]) < 2:
        return None

    # Extract metadata from first conditioning entry
    metadata = conditioning[0][1]

    if not isinstance(metadata, dict):
        return None

    # Check if GRAG is enabled
    if not metadata.get("grag_enabled", False):
        return None

    # Extract GRAG parameters
    grag_config = {
        "enabled": True,
        "lambda": metadata.get("grag_cond_b", 1.0),
        "delta": metadata.get("grag_cond_delta", 1.0),
        "strength": metadata.get("grag_strength", 1.0),
        "heads": 16,  # Qwen default: 16 heads
    }

    return grag_config


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_grag_parameters(lambda_val, delta_val):
    """Validate GRAG parameter ranges and warn if outside stable range.

    Testing range: [0.1, 2.0] for full experimentation
    Paper (arXiv 2510.24657) recommends: lambda and delta in [0.95, 1.15]
    for stable, training-free image editing.

    Args:
        lambda_val (float): Bias strength (lambda)
        delta_val (float): Deviation strength (delta)

    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(lambda_val, (int, float)):
        return False, "lambda must be numeric"

    if not isinstance(delta_val, (int, float)):
        return False, "delta must be numeric"

    # Hard limits (testing range)
    if lambda_val < 0.1 or lambda_val > 2.0:
        return False, "lambda should be in range [0.1, 2.0]"

    if delta_val < 0.1 or delta_val > 2.0:
        return False, "delta should be in range [0.1, 2.0]"

    # Soft warnings (paper's stable range)
    STABLE_MIN = 0.95
    STABLE_MAX = 1.15

    if lambda_val < STABLE_MIN or lambda_val > STABLE_MAX:
        print(f"[GRAG] Info: lambda={lambda_val:.3f} outside paper's stable range [{STABLE_MIN}, {STABLE_MAX}]")
        print(f"[GRAG] Experimenting with wider range - expect stronger effects")

    if delta_val < STABLE_MIN or delta_val > STABLE_MAX:
        print(f"[GRAG] Info: delta={delta_val:.3f} outside paper's stable range [{STABLE_MIN}, {STABLE_MAX}]")
        print(f"[GRAG] Experimenting with wider range - expect stronger effects")

    return True, ""


def get_recommended_grag_preset(preset_name):
    """Get recommended GRAG parameter presets.

    Updated v2.2.1 with wider ranges for VISIBLE effects (0.1-2.0 testing range).
    Paper's stable range [0.95, 1.15] was too conservative for visible changes.

    Args:
        preset_name (str): Preset identifier
            - "subtle": Gentle edits, preserve structure (visible but conservative)
            - "balanced": Recommended default (visible effects, good balance)
            - "strong": Maximum transformation (dramatic changes)
            - "extreme": Testing extremes (for experimentation)

    Returns:
        dict: Parameter dictionary with lambda, delta, strength
    """
    presets = {
        "subtle": {
            "lambda": 0.80,
            "delta": 1.20,
            "strength": 1.0,
            "description": "Subtle edits - reduced bias, amplified deviations (20% change)"
        },
        "balanced": {
            "lambda": 1.0,
            "delta": 1.50,
            "strength": 1.0,
            "description": "Balanced control - neutral bias, strong deviations (50% amplification)"
        },
        "strong": {
            "lambda": 1.50,
            "delta": 2.00,
            "strength": 1.0,
            "description": "Strong transformation - enhanced bias and maximum deviations (100% amplification)"
        },
        "extreme_low": {
            "lambda": 0.10,
            "delta": 0.10,
            "strength": 1.0,
            "description": "Extreme suppression - testing minimum values (experimental)"
        },
        "extreme_high": {
            "lambda": 2.00,
            "delta": 2.00,
            "strength": 1.0,
            "description": "Extreme amplification - testing maximum values (experimental)"
        }
    }

    return presets.get(preset_name, presets["balanced"])


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "apply_grag_to_keys",
    "create_grag_patch",
    "extract_grag_config_from_conditioning",
    "validate_grag_parameters",
    "get_recommended_grag_preset"
]
