"""
USDU Edge Repair - Differential Diffusion
==========================================

Per-pixel denoise control via mask.
Copied from KJNodes DifferentialDiffusionAdvanced.
"""

import torch


class DifferentialDiffusionAdvanced:
    """
    Differential Diffusion - Per-pixel denoise control via mask.

    This class patches the model's denoise_mask_function to enable
    variable denoise strength across the image based on a grayscale mask.

    How it works:
        1. A grayscale mask is provided (white=more denoise, black=less)
        2. At each timestep, a threshold is calculated based on progress
        3. Pixels where mask >= threshold are denoised, others are skipped
        4. multiplier < 1.0 = stronger effect, > 1.0 = weaker effect

    Source: KJNodes (comfyui-kjnodes)
    """

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "samples": ("LATENT",),
            "mask": ("MASK",),
            "multiplier": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001}),
        }}

    RETURN_TYPES = ("MODEL", "LATENT")
    FUNCTION = "apply"
    CATEGORY = "_for_testing"
    INIT = False

    def apply(self, model, samples, mask, multiplier):
        """
        Apply Differential Diffusion to model.

        Args:
            model: ComfyUI MODEL to patch
            samples: LATENT dict
            mask: Grayscale MASK for denoise control
            multiplier: Strength multiplier

        Returns:
            (patched_model, latent_with_noise_mask)
        """
        self.multiplier = multiplier
        model = model.clone()
        model.set_model_denoise_mask_function(self.forward)
        s = samples.copy()
        s["noise_mask"] = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
        return (model, s)

    def forward(self, sigma: torch.Tensor, denoise_mask: torch.Tensor, extra_options: dict):
        """
        Denoise mask function called at each timestep.

        Args:
            sigma: Current noise level tensor
            denoise_mask: Grayscale mask (0-1 values)
            extra_options: Dict with 'model', 'sigmas', etc.

        Returns:
            Binary mask tensor (1=denoise, 0=skip)
        """
        model = extra_options["model"]
        step_sigmas = extra_options["sigmas"]
        sigma_to = model.inner_model.model_sampling.sigma_min
        if step_sigmas[-1] > sigma_to:
            sigma_to = step_sigmas[-1]
        sigma_from = step_sigmas[0]

        ts_from = model.inner_model.model_sampling.timestep(sigma_from)
        ts_to = model.inner_model.model_sampling.timestep(sigma_to)
        current_ts = model.inner_model.model_sampling.timestep(sigma[0])

        # Calculate threshold: progresses from 1.0 -> 0.0 over sampling
        threshold = (current_ts - ts_to) / (ts_from - ts_to) / self.multiplier

        return (denoise_mask >= threshold).to(denoise_mask.dtype)
