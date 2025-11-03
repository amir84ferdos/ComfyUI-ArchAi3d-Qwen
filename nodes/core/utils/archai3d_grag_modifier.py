# ArchAi3D GRAG Modifier — Universal GRAG Conditioning Modifier
#
# OVERVIEW:
# Universal conditioning modifier that adds GRAG (Group-Relative Attention Guidance) metadata
# to any encoder's output. Works with ALL encoders (V1, V2, V3, Simple, etc.).
#
# WHAT IS GRAG:
# - Training-free fine-grained image editing technique
# - Re-weights attention deltas between tokens and shared biases
# - Provides continuous control (0.8-1.7) instead of binary on/off
# - Better structure/window preservation
# - Reduced artifacts and halos
#
# HOW IT WORKS:
# 1. Takes conditioning from ANY encoder
# 2. Optionally adds GRAG metadata (when enabled)
# 3. Passes through unchanged if disabled
# 4. Clean, modular, universal compatibility
#
# USAGE:
# [Any Encoder] → [GRAG Modifier] → [Sampler] → [Output]
#
# Or skip it entirely for standard workflow:
# [Any Encoder] → [Sampler] → [Output]
#
# PARAMETERS:
# - enable_grag: Toggle GRAG on/off (passthrough when false)
# - grag_strength: Main intensity control (0.8-1.7, default 1.0)
# - grag_cond_b: Lambda (bias strength) - Paper range: 0.95-1.15, default 1.0
# - grag_cond_delta: Delta (deviation intensity) - Paper range: 0.95-1.15, default 1.05
#
# BENEFITS:
# ✅ Works with ALL existing encoders
# ✅ No code duplication
# ✅ Easy A/B testing (add/remove node)
# ✅ Optional and clean
# ✅ Future-proof (update once, works everywhere)
#
# Author: Amir Ferdos (ArchAi3d)
# Email: Amir84ferdos@gmail.com
# LinkedIn: https://www.linkedin.com/in/archai3d/
# GitHub: https://github.com/amir84ferdos
# Category: ArchAi3d/Qwen
# Node ID: ArchAi3D_GRAG_Modifier
# License: MIT
# Based on: GRAG-Image-Editing by little-misfit (https://github.com/little-misfit/GRAG-Image-Editing)

import torch
import copy


class ArchAi3D_GRAG_Modifier:
    """Universal GRAG conditioning modifier - works with ANY encoder output.

    Adds GRAG (Group-Relative Attention Guidance) metadata to conditioning for
    fine-grained editing control. Passthrough mode when disabled.

    Perfect for:
    - Testing GRAG with different encoders
    - Optional fine-grained control
    - A/B testing (add/remove node)
    - Clean workflow organization

    Version: 2.1.1
    """

    # Define 40 fine-tuned GRAG presets (0.56-0.64 range with varied parameters)
    # Each preset uses DIFFERENT values for strength, lambda, and delta for experimentation
    GRAG_PRESETS = {
        "Custom": {"strength": 1.0, "lambda": 1.0, "delta": 1.0, "desc": "Manual control - adjust all parameters yourself"},

        # 40 varied presets in the 0.56-0.64 range (Level 03-04 equivalent)
        # Format: strength varies, lambda varies, delta varies independently
        "Preset 01": {"strength": 0.56, "lambda": 0.56, "delta": 0.64, "desc": "Low str, low λ, mid δ"},
        "Preset 02": {"strength": 0.56, "lambda": 0.58, "delta": 0.62, "desc": "Low str, low-mid λ, mid-low δ"},
        "Preset 03": {"strength": 0.56, "lambda": 0.60, "delta": 0.60, "desc": "Low str, mid λ, mid δ"},
        "Preset 04": {"strength": 0.56, "lambda": 0.62, "delta": 0.58, "desc": "Low str, mid-high λ, low-mid δ"},
        "Preset 05": {"strength": 0.56, "lambda": 0.64, "delta": 0.56, "desc": "Low str, high λ, low δ"},

        "Preset 06": {"strength": 0.57, "lambda": 0.57, "delta": 0.63, "desc": "Low+ str, low+ λ, mid+ δ"},
        "Preset 07": {"strength": 0.57, "lambda": 0.59, "delta": 0.61, "desc": "Low+ str, mid- λ, mid δ"},
        "Preset 08": {"strength": 0.57, "lambda": 0.61, "delta": 0.59, "desc": "Low+ str, mid+ λ, mid- δ"},
        "Preset 09": {"strength": 0.57, "lambda": 0.63, "delta": 0.57, "desc": "Low+ str, mid++ λ, low+ δ"},
        "Preset 10": {"strength": 0.57, "lambda": 0.56, "delta": 0.64, "desc": "Low+ str, low λ, high δ"},

        "Preset 11": {"strength": 0.58, "lambda": 0.56, "delta": 0.62, "desc": "Mid- str, low λ, mid-low δ"},
        "Preset 12": {"strength": 0.58, "lambda": 0.58, "delta": 0.60, "desc": "Mid- str, mid- λ, mid δ"},
        "Preset 13": {"strength": 0.58, "lambda": 0.60, "delta": 0.58, "desc": "Mid- str, mid λ, mid- δ"},
        "Preset 14": {"strength": 0.58, "lambda": 0.62, "delta": 0.56, "desc": "Mid- str, mid-high λ, low δ"},
        "Preset 15": {"strength": 0.58, "lambda": 0.64, "delta": 0.64, "desc": "Mid- str, high λ, high δ"},

        "Preset 16": {"strength": 0.59, "lambda": 0.57, "delta": 0.61, "desc": "Mid str, low+ λ, mid δ"},
        "Preset 17": {"strength": 0.59, "lambda": 0.59, "delta": 0.59, "desc": "Mid str, mid- λ, mid- δ"},
        "Preset 18": {"strength": 0.59, "lambda": 0.61, "delta": 0.57, "desc": "Mid str, mid+ λ, low+ δ"},
        "Preset 19": {"strength": 0.59, "lambda": 0.63, "delta": 0.63, "desc": "Mid str, mid++ λ, mid++ δ"},
        "Preset 20": {"strength": 0.59, "lambda": 0.56, "delta": 0.60, "desc": "Mid str, low λ, mid δ"},

        "Preset 21": {"strength": 0.60, "lambda": 0.56, "delta": 0.58, "desc": "Mid str, low λ, mid- δ"},
        "Preset 22": {"strength": 0.60, "lambda": 0.58, "delta": 0.56, "desc": "Mid str, mid- λ, low δ"},
        "Preset 23": {"strength": 0.60, "lambda": 0.60, "delta": 0.64, "desc": "Mid str, mid λ, high δ"},
        "Preset 24": {"strength": 0.60, "lambda": 0.62, "delta": 0.62, "desc": "Mid str, mid-high λ, mid-low δ"},
        "Preset 25": {"strength": 0.60, "lambda": 0.64, "delta": 0.60, "desc": "Mid str, high λ, mid δ"},

        "Preset 26": {"strength": 0.61, "lambda": 0.57, "delta": 0.59, "desc": "Mid+ str, low+ λ, mid- δ"},
        "Preset 27": {"strength": 0.61, "lambda": 0.59, "delta": 0.57, "desc": "Mid+ str, mid- λ, low+ δ"},
        "Preset 28": {"strength": 0.61, "lambda": 0.61, "delta": 0.63, "desc": "Mid+ str, mid+ λ, mid++ δ"},
        "Preset 29": {"strength": 0.61, "lambda": 0.63, "delta": 0.61, "desc": "Mid+ str, mid++ λ, mid+ δ"},
        "Preset 30": {"strength": 0.61, "lambda": 0.56, "delta": 0.64, "desc": "Mid+ str, low λ, high δ"},

        "Preset 31": {"strength": 0.62, "lambda": 0.56, "delta": 0.60, "desc": "Mid-high str, low λ, mid δ"},
        "Preset 32": {"strength": 0.62, "lambda": 0.58, "delta": 0.58, "desc": "Mid-high str, mid- λ, mid- δ"},
        "Preset 33": {"strength": 0.62, "lambda": 0.60, "delta": 0.56, "desc": "Mid-high str, mid λ, low δ"},
        "Preset 34": {"strength": 0.62, "lambda": 0.62, "delta": 0.64, "desc": "Mid-high str, mid-high λ, high δ"},
        "Preset 35": {"strength": 0.62, "lambda": 0.64, "delta": 0.62, "desc": "Mid-high str, high λ, mid-low δ"},

        "Preset 36": {"strength": 0.63, "lambda": 0.57, "delta": 0.61, "desc": "High- str, low+ λ, mid δ"},
        "Preset 37": {"strength": 0.63, "lambda": 0.59, "delta": 0.63, "desc": "High- str, mid- λ, mid++ δ"},
        "Preset 38": {"strength": 0.63, "lambda": 0.61, "delta": 0.59, "desc": "High- str, mid+ λ, mid- δ"},
        "Preset 39": {"strength": 0.63, "lambda": 0.63, "delta": 0.57, "desc": "High- str, mid++ λ, low+ δ"},
        "Preset 40": {"strength": 0.63, "lambda": 0.64, "delta": 0.64, "desc": "High- str, high λ, high δ"},

        "Preset 41": {"strength": 0.64, "lambda": 0.56, "delta": 0.56, "desc": "High str, low λ, low δ"},
    }

    @classmethod
    def INPUT_TYPES(cls):
        preset_names = list(cls.GRAG_PRESETS.keys())

        return {
            "required": {
                # Conditioning from any encoder
                "conditioning": ("CONDITIONING",),

                # GRAG toggle and preset selector
                "enable_grag": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable GRAG attention guidance (passthrough if disabled)"
                }),
                "preset": (preset_names, {
                    "default": "Preset 01",
                    "tooltip": "Choose a preset or 'Custom' for manual control"
                }),

                # Manual parameters (active when preset="Custom")
                "grag_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Main GRAG intensity - 0.1-2.0 range (0.1=minimum, 1.0=neutral, 2.0=maximum)"
                }),
                "grag_cond_b": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Lambda (bias strength) - 0.1-2.0 range (paper's stable: 0.95-1.15, neutral: 1.0)"
                }),
                "grag_cond_delta": ("FLOAT", {
                    "default": 1.05,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Delta (deviation intensity) - 0.1-2.0 range (paper's stable: 0.95-1.15, neutral: 1.0)"
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "modify"
    CATEGORY = "ArchAi3d/Qwen"

    def build_grag_scale(self, cond_b, cond_delta, num_steps=60):
        """Build GRAG scale configuration for attention guidance.

        Creates multi-tier resolution scaling pattern:
        - Tier 1: 512×512 with 1.0 scale (base reference)
        - Tier 2: 4096×4096 with custom scaling (cond_b, cond_delta)

        Args:
            cond_b: Base conditioning strength
            cond_delta: Delta conditioning strength
            num_steps: Number of inference steps (default 60 for Qwen)

        Returns:
            List of tuples: [((res1, scale1_a, scale1_b), (res2, scale2_a, scale2_b))] * num_steps
        """
        # Two-tier resolution: base (512) and high (4096)
        # Base tier uses 1.0 scale, high tier uses custom cond_b and cond_delta
        tier_config = ((512, 1.0, 1.0), (4096, cond_b, cond_delta))

        # Repeat for all inference steps
        grag_scale = [tier_config] * num_steps

        return grag_scale

    def apply_grag_strength(self, grag_scale, grag_strength):
        """Apply grag_strength multiplier to the scale configuration.

        NOTE: As of v2.2.0, grag_strength is stored but NOT multiplied with cond_b/cond_delta
        to prevent parameter overflow. Paper recommends keeping lambda/delta in 0.95-1.15 range.

        Args:
            grag_scale: Base GRAG scale configuration
            grag_strength: Overall strength multiplier (stored for future use, not applied)

        Returns:
            GRAG configuration (unmodified - cond_b/cond_delta used directly)
        """
        # FIXED in v2.2.0: Don't multiply cond_b/cond_delta by grag_strength
        # This was causing parameter overflow (values reaching 3.4 instead of 0.95-1.15)
        # Paper shows stable range is 0.95-1.15, so we use cond_b/cond_delta directly

        # Simply return the original scale config without modification
        # grag_strength is still stored in metadata for potential future use
        return grag_scale

    def modify(self, conditioning, enable_grag, preset, grag_strength, grag_cond_b, grag_cond_delta):
        """Modify conditioning with GRAG metadata or passthrough.

        Args:
            conditioning: Input conditioning from any encoder
            enable_grag: Enable GRAG modification (passthrough if False)
            preset: Preset name or "Custom" for manual control
            grag_strength: Stored for future use (NOT multiplied as of v2.2.0)
            grag_cond_b: Lambda - bias strength (0.1-2.0 range, default 1.0)
            grag_cond_delta: Delta - deviation intensity (0.1-2.0 range, default 1.05)

        Returns:
            Tuple of (modified_conditioning,) or (original_conditioning,)

        Note:
            v2.2.1 added 20 presets for different use cases. Choose preset or use "Custom"
            for manual control. Parameter ranges expanded to 0.1-2.0 for visible effects.
        """
        # Passthrough mode: GRAG disabled
        if not enable_grag:
            return (conditioning,)

        # Apply preset if not "Custom"
        if preset != "Custom" and preset in self.GRAG_PRESETS:
            preset_values = self.GRAG_PRESETS[preset]
            grag_strength = preset_values["strength"]
            grag_cond_b = preset_values["lambda"]
            grag_cond_delta = preset_values["delta"]
            print(f"[GRAG Modifier] Using preset: {preset} - {preset_values['desc']}")
            print(f"[GRAG Modifier] Parameters: strength={grag_strength:.2f}, λ={grag_cond_b:.2f}, δ={grag_cond_delta:.2f}")
        else:
            print(f"[GRAG Modifier] Custom parameters: strength={grag_strength:.2f}, λ={grag_cond_b:.2f}, δ={grag_cond_delta:.2f}")

        # GRAG enabled: Build scale configuration
        grag_scale = self.build_grag_scale(grag_cond_b, grag_cond_delta, num_steps=60)

        # Apply grag_strength multiplier
        scaled_grag_config = self.apply_grag_strength(grag_scale, grag_strength)

        # Deep copy conditioning to avoid modifying original
        grag_cond = copy.deepcopy(conditioning)

        # Add GRAG metadata to conditioning
        for i in range(len(grag_cond)):
            if len(grag_cond[i]) >= 2:
                # conditioning format: [(embeddings, metadata_dict)]
                metadata = grag_cond[i][1].copy() if isinstance(grag_cond[i][1], dict) else {}

                # Add GRAG configuration
                metadata['grag_scale'] = scaled_grag_config
                metadata['grag_enabled'] = True
                metadata['grag_strength'] = grag_strength
                metadata['grag_cond_b'] = grag_cond_b
                metadata['grag_cond_delta'] = grag_cond_delta

                # Update conditioning with GRAG metadata
                grag_cond[i] = (grag_cond[i][0], metadata)

        return (grag_cond,)


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "ArchAi3D_GRAG_Modifier": ArchAi3D_GRAG_Modifier
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_GRAG_Modifier": "🎚️ GRAG Modifier (Fine-Grained Control)"
}
