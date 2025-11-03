# ArchAi3D GRAG-Aware Sampler Node
#
# OVERVIEW:
# Custom sampler that injects GRAG (Group-Relative Attention Guidance) attention
# patches into the sampling process for fine-grained image editing control.
#
# HOW IT WORKS:
# 1. Extracts GRAG configuration from positive conditioning metadata
# 2. Creates GRAG attention patch using the reweighting utilities
# 3. Injects the patch via model transformer_options
# 4. Calls standard ComfyUI sampler with GRAG-enhanced model
# 5. CRITICAL: Restores original forward methods in finally block (v2.2.1 fix)
#
# USAGE:
# [Any Encoder] → [GRAG Modifier] → [GRAG Sampler] → [Output]
#
# Or with GRAG Encoder:
# [GRAG Encoder] → [GRAG Sampler] → [Output]
#
# BENEFITS:
# - No ComfyUI core modifications
# - Works with all existing encoders
# - Update-safe implementation
# - Clean on/off toggle
# - Proper cleanup prevents global contamination (fixed in v2.2.1)
#
# CRITICAL FIX (v2.2.1):
# Fixed global contamination bug where GRAG patches persisted across samplers.
# Root cause: model.clone() creates shallow clone sharing diffusion_model references.
# Solution: Store original forward methods and restore in finally block after sampling.
# This ensures GRAG only affects intended generations and doesn't contaminate other samplers.
#
# Author: Amir Ferdos (ArchAi3d)
# Email: Amir84ferdos@gmail.com
# LinkedIn: https://www.linkedin.com/in/archai3d/
# GitHub: https://github.com/amir84ferdos
# Category: ArchAi3d/Qwen
# Node ID: ArchAi3D_GRAG_Sampler
# License: MIT
# Based on: GRAG-Image-Editing by little-misfit

import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import comfy.samplers
import comfy.sample
import comfy.model_management
import comfy.utils
import latent_preview

from core.utils.grag_attention import (
    extract_grag_config_from_conditioning,
    create_grag_patch
)


class ArchAi3D_GRAG_Sampler:
    """GRAG-aware sampler that injects attention guidance during sampling.

    This sampler wraps ComfyUI's standard KSampler and injects GRAG attention
    patches to enable fine-grained editing control. It reads GRAG metadata from
    conditioning (set by GRAG Modifier or GRAG Encoder) and applies attention
    reweighting during the diffusion process.

    Key Features:
    - Extracts GRAG config from conditioning metadata
    - Injects attention patches via transformer_options
    - Falls back to standard sampling if GRAG disabled
    - Compatible with all ComfyUI schedulers and samplers

    Version: 2.1.1
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Standard KSampler parameters
                "model": ("MODEL", {
                    "tooltip": "The diffusion model used for denoising"
                }),
                "positive": ("CONDITIONING", {
                    "tooltip": "Positive conditioning (should contain GRAG metadata if using GRAG Modifier/Encoder)"
                }),
                "negative": ("CONDITIONING", {
                    "tooltip": "Negative conditioning"
                }),
                "latent_image": ("LATENT", {
                    "tooltip": "Input latent to denoise"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed for noise generation"
                }),
                "steps": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 10000,
                    "tooltip": "Number of denoising steps"
                }),
                "cfg": ("FLOAT", {
                    "default": 8.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Classifier-Free Guidance scale"
                }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {
                    "tooltip": "Sampling algorithm to use"
                }),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {
                    "tooltip": "Noise schedule for denoising"
                }),
                "denoise": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Denoising strength (1.0 = full denoise)"
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "sample"
    CATEGORY = "ArchAi3d/Qwen"

    def _patch_qwen_attention(self, model, grag_config):
        """Monkey-patch Qwen attention layers to apply GRAG reweighting.

        This function finds all Attention modules in the model and wraps their
        forward method to apply GRAG key reweighting after RoPE but before attention.

        Args:
            model: ComfyUI model object with diffusion_model attribute
            grag_config: Dict with GRAG parameters (lambda, delta, heads)

        Returns:
            dict: Dictionary mapping modules to their original forward methods.
                  Used for restoration after sampling completes.
                  Returns empty dict if patching fails.
        """
        from core.utils.grag_attention import apply_grag_to_keys

        # Dictionary to store original forward methods for restoration
        original_forwards = {}

        # Access the actual diffusion model
        if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
            diffusion_model = model.model.diffusion_model
        else:
            print("[GRAG Sampler] Warning: Could not access diffusion_model")
            return original_forwards

        # Find and patch all Attention modules
        patched_count = 0
        for name, module in diffusion_model.named_modules():
            # Look for Qwen Attention modules specifically
            # Check class name AND verify it has the right attributes
            if (module.__class__.__name__ == 'Attention' and
                hasattr(module, 'to_q') and
                hasattr(module, 'add_q_proj') and
                hasattr(module, 'norm_q')):
                # Store original forward method for restoration
                original_forward = module.forward
                original_forwards[module] = original_forward

                # Create wrapped forward function with GRAG
                def create_grag_forward(orig_forward, grag_cfg, attn_module):
                    def grag_forward(hidden_states, encoder_hidden_states=None, encoder_hidden_states_mask=None,
                                   attention_mask=None, image_rotary_emb=None, transformer_options={}):
                        # Call original forward up to the point where we need to inject GRAG
                        # We'll need to replicate the forward pass with GRAG insertion

                        seq_txt = encoder_hidden_states.shape[1]

                        # Image stream QKV
                        img_query = attn_module.to_q(hidden_states).unflatten(-1, (attn_module.heads, -1))
                        img_key = attn_module.to_k(hidden_states).unflatten(-1, (attn_module.heads, -1))
                        img_value = attn_module.to_v(hidden_states).unflatten(-1, (attn_module.heads, -1))

                        # Text stream QKV
                        txt_query = attn_module.add_q_proj(encoder_hidden_states).unflatten(-1, (attn_module.heads, -1))
                        txt_key = attn_module.add_k_proj(encoder_hidden_states).unflatten(-1, (attn_module.heads, -1))
                        txt_value = attn_module.add_v_proj(encoder_hidden_states).unflatten(-1, (attn_module.heads, -1))

                        # Normalization
                        img_query = attn_module.norm_q(img_query)
                        img_key = attn_module.norm_k(img_key)
                        txt_query = attn_module.norm_added_q(txt_query)
                        txt_key = attn_module.norm_added_k(txt_key)

                        # Combine streams
                        joint_query = torch.cat([txt_query, img_query], dim=1)
                        joint_key = torch.cat([txt_key, img_key], dim=1)
                        joint_value = torch.cat([txt_value, img_value], dim=1)

                        # Apply RoPE
                        from comfy.ldm.qwen_image.model import apply_rotary_emb
                        joint_query = apply_rotary_emb(joint_query, image_rotary_emb)
                        joint_key = apply_rotary_emb(joint_key, image_rotary_emb)

                        # ===== GRAG INJECTION POINT =====
                        # Apply GRAG reweighting to keys BEFORE final flattening
                        # Note: joint_key is currently [B, S, H, D], but apply_grag_to_keys expects [B, S, C]
                        try:
                            # Flatten keys temporarily for GRAG
                            joint_key_flat = joint_key.flatten(start_dim=2)  # [B, S, H*D]

                            # Apply GRAG reweighting
                            joint_key_flat = apply_grag_to_keys(
                                joint_key_flat,
                                seq_txt,
                                grag_cfg['lambda'],
                                grag_cfg['delta'],
                                attn_module.heads
                            )

                            # Unflatten back to [B, S, H, D] for consistency
                            joint_key = joint_key_flat.unflatten(-1, (attn_module.heads, -1))
                        except Exception as e:
                            print(f"[GRAG] Warning: Reweighting failed: {e}")
                            import traceback
                            traceback.print_exc()
                            pass  # Continue with original keys if GRAG fails
                        # ===== END GRAG =====

                        # Flatten for attention
                        joint_query = joint_query.flatten(start_dim=2)
                        joint_key = joint_key.flatten(start_dim=2)
                        joint_value = joint_value.flatten(start_dim=2)

                        # Standard attention
                        from comfy.ldm.modules.attention import optimized_attention_masked
                        joint_hidden_states = optimized_attention_masked(
                            joint_query, joint_key, joint_value, attn_module.heads,
                            attention_mask, transformer_options=transformer_options
                        )

                        # Split streams
                        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
                        img_attn_output = joint_hidden_states[:, seq_txt:, :]

                        # Output projections
                        img_attn_output = attn_module.to_out[0](img_attn_output)
                        img_attn_output = attn_module.to_out[1](img_attn_output)
                        txt_attn_output = attn_module.to_add_out(txt_attn_output)

                        return img_attn_output, txt_attn_output

                    return grag_forward

                # Replace forward method
                module.forward = create_grag_forward(original_forward, grag_config, module)
                patched_count += 1

        print(f"[GRAG Sampler] Patched {patched_count} Attention layers")
        return original_forwards

    def sample(self, model, positive, negative, latent_image, seed, steps, cfg, sampler_name, scheduler, denoise):
        """Perform sampling with GRAG attention guidance.

        This is the main entry point for the sampler. It:
        1. Extracts GRAG configuration from positive conditioning
        2. Creates a model clone with GRAG monkey-patch injected
        3. Calls ComfyUI's standard sampling with the enhanced model
        4. Returns the denoised latent samples

        Args:
            model: ComfyUI MODEL object
            positive: Positive conditioning (may contain GRAG metadata)
            negative: Negative conditioning
            latent_image: Input latent {"samples": tensor}
            seed: Random seed for reproducibility
            steps: Number of denoising steps
            cfg: Classifier-Free Guidance scale
            sampler_name: Sampler algorithm (euler, dpmpp_2m, etc.)
            scheduler: Noise schedule (normal, karras, etc.)
            denoise: Denoising strength (0.0-1.0)

        Returns:
            tuple: (latent_dict,) with denoised samples
        """
        # Extract GRAG configuration from conditioning metadata
        grag_config = extract_grag_config_from_conditioning(positive)

        # Clone model to avoid modifying original
        model_clone = model.clone()

        # Store original forward methods for restoration
        original_forwards = {}

        # If GRAG is enabled, monkey-patch the attention forward function
        if grag_config and grag_config.get("enabled", False):
            print(f"[GRAG Sampler] GRAG enabled - λ={grag_config['lambda']:.2f}, δ={grag_config['delta']:.2f}, strength={grag_config.get('strength', 1.0):.2f}")

            # Try to patch Qwen attention layers
            try:
                original_forwards = self._patch_qwen_attention(model_clone, grag_config)
                print(f"[GRAG Sampler] GRAG patches injected successfully")
            except Exception as e:
                print(f"[GRAG Sampler] Failed to inject GRAG patches: {e}")
                print(f"[GRAG Sampler] Falling back to standard sampling")
        else:
            print(f"[GRAG Sampler] GRAG disabled - using standard sampling")

        # Call ComfyUI's standard sampling function with try/finally for cleanup
        # This handles all the complex diffusion logic
        try:
            samples = self._common_ksampler(
                model_clone,
                seed,
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive,
                negative,
                latent_image,
                denoise=denoise
            )

            return samples

        except Exception as e:
            print(f"[GRAG Sampler] Error during sampling: {e}")
            print(f"[GRAG Sampler] Falling back to standard sampler")

            # Fallback: Try without GRAG patches
            model_clean = model.clone()
            samples = self._common_ksampler(
                model_clean,
                seed,
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive,
                negative,
                latent_image,
                denoise=denoise
            )

            return samples

        finally:
            # CRITICAL: Always restore original forward methods to prevent contamination
            # This fixes the global contamination bug where GRAG affects other samplers
            if original_forwards:
                for module, original_forward in original_forwards.items():
                    module.forward = original_forward
                print(f"[GRAG Sampler] Restored {len(original_forwards)} attention modules")

    def _common_ksampler(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0):
        """Wrapper around ComfyUI's common_ksampler function.

        This replicates the logic from nodes.py:common_ksampler to ensure
        compatibility with ComfyUI's sampling infrastructure.

        Args:
            model: MODEL object (possibly with GRAG patches)
            seed: Random seed
            steps: Denoising steps
            cfg: CFG scale
            sampler_name: Sampler algorithm
            scheduler: Noise scheduler
            positive: Positive conditioning
            negative: Negative conditioning
            latent: Latent dict {"samples": tensor}
            denoise: Denoising strength

        Returns:
            tuple: (latent_dict,) with denoised samples
        """
        # Extract latent samples
        latent_image = latent["samples"]

        # Fix empty latent channels if needed
        latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

        # Prepare noise
        batch_inds = latent.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

        # Handle noise mask if present
        noise_mask = latent.get("noise_mask", None)

        # Setup progress callback
        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        # Perform sampling
        samples = comfy.sample.sample(
            model,
            noise,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            denoise=denoise,
            disable_noise=False,
            start_step=None,
            last_step=None,
            force_full_denoise=False,
            noise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed
        )

        # Return in ComfyUI latent format
        out = latent.copy()
        out["samples"] = samples

        return (out,)


# ============================================================================
# COMFYUI NODE REGISTRATION
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "ArchAi3D_GRAG_Sampler": ArchAi3D_GRAG_Sampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_GRAG_Sampler": "🎚️ GRAG Sampler (Fine-Grained Control)"
}
