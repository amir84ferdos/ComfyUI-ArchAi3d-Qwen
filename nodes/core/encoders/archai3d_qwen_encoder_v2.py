# ArchAi3D Qwen Encoder V2 — Qwen-VL 2.5 encoder with interpolation-based strength
#
# OVERVIEW:
# V2 uses baseline interpolation for conditioning strength instead of raw multiplication.
# This provides stable, predictable strength behavior even when system prompts are added.
#
# DIFFERENCES FROM V1:
# - V1: Uses raw multiply (conditioning *= strength) - can feel "10× stronger" with system prompts
# - V2: Uses interpolation (baseline + alpha * delta) - smooth, predictable scaling
# - V1: Keeps llama_template for backward compatibility
# - V2: ChatML-only, no llama_template parameter
# - V2: Scales pooled_output along with token embeddings for balanced guidance
#
# INPUTS:
# - 3 images for Qwen-VL vision encoder (RGB only, expects correct size)
# - 3 images for VAE reference latents (RGB only, expects correct size)
# - Text prompt (wrapped automatically in ChatML format)
# - Optional system prompt (for ChatML system block)
#
# STRENGTH CONTROLS (TWO-STAGE):
# 1. context_strength (0.0-1.5):
#    - Stage A: vision → vision+context (system + labels)
#    - 0.0 = pure vision (images only, no text)
#    - 1.0 = full context (normal)
#    - >1.0 = extrapolated context
#    - **USE THIS to control system prompt influence!**
#    - Recommended: 0.8-1.0 (turn down if system feels too heavy)
#
# 2. user_strength (0.0-1.5):
#    - Stage B: vision+context → full (adds user text)
#    - 0.0 = no user text (context only)
#    - 1.0 = full user text (normal)
#    - >1.0 = extrapolated user text
#    - Recommended: 0.35-0.8 for balanced results
#
# 3. image1/2/3_latent_strength (0.0-2.0):
#    - Multiplies each reference latent independently (per-image control)
#    - <1.0 = weaker reference, 1.0 = normal, >1.0 = stronger reference
#    - Controls signal strength in latent space (NOT noise)
#
# OUTPUTS:
# - conditioning: Text+vision embeddings with reference latents metadata
# - latent: Image1 latent in standard format (for VAEDecode)
# - formatted_prompt: Final ChatML prompt with vision tokens (for debugging)
#
# Author: Amir Ferdos (ArchAi3d)
# Email: Amir84ferdos@gmail.com
# LinkedIn: https://www.linkedin.com/in/archai3d/
# GitHub: https://github.com/amir84ferdos
# Category: ArchAi3d/Qwen
# Node ID: ArchAi3D_Qwen_Encoder_V2
# License: MIT

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override
import node_helpers
import torch
import copy


# ============================================================
# Helper Functions for Two-Stage Interpolation
# ============================================================

def _pad_tokens_right(x: torch.Tensor, T: int):
    """Right-pad token dimension (-2 axis) to target length."""
    if x is None or x.dim() < 2:
        return x
    cur = x.shape[-2]
    if cur >= T:
        return x
    pad_shape = list(x.shape)
    pad_shape[-2] = T - cur
    pad = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=-2)


def _pad_mask_right(mask: torch.Tensor, T: int):
    """Right-pad attention mask to target length."""
    if mask is None or not isinstance(mask, torch.Tensor):
        return mask
    cur = mask.shape[-1]
    if cur >= T:
        return mask
    pad = torch.zeros(*mask.shape[:-1], T - cur, dtype=mask.dtype, device=mask.device)
    return torch.cat([mask, pad], dim=-1)


def _interp_conditioning(pos_cond, null_cond, alpha: float, cap: float = None):
    """Interpolate conditioning: new = null + alpha * (pos - null).

    Uses baseline (null) masks/extras to prevent "10× spike" when system prompt is present.

    Args:
        pos_cond: Target conditioning (e.g., full prompt)
        null_cond: Baseline conditioning (e.g., no user text)
        alpha: Interpolation weight (0.0 = null, 1.0 = pos, >1.0 = extrapolate)
        cap: Optional RMS cap on delta for very long prompts

    Returns:
        Interpolated conditioning list
    """
    out = []
    L = min(len(pos_cond), len(null_cond))

    for i in range(L):
        pt, pextra = pos_cond[i]
        nt, nextra = null_cond[i]
        # Start from NULL extras to prevent positive masks amplifying at low alpha
        new_extra = copy.deepcopy(nextra) if isinstance(nextra, dict) else {}

        # Pad token embeddings to same length
        T = max(pt.shape[-2] if pt.dim() >= 2 else 1,
                nt.shape[-2] if nt.dim() >= 2 else 1)
        pt_p = _pad_tokens_right(pt, T)
        nt_p = _pad_tokens_right(nt, T)
        new_t = nt_p + float(alpha) * (pt_p - nt_p)

        # Interpolate pooled_output
        p_pool = (pextra or {}).get("pooled_output", None)
        n_pool = (nextra or {}).get("pooled_output", None)
        if isinstance(p_pool, torch.Tensor) and isinstance(n_pool, torch.Tensor):
            new_extra["pooled_output"] = n_pool + float(alpha) * (p_pool - n_pool)
        elif isinstance(p_pool, torch.Tensor):
            new_extra["pooled_output"] = float(alpha) * p_pool

        # Handle attention masks (support both key names)
        p_mask = (pextra or {}).get("attention_mask") or (pextra or {}).get("attn_mask")
        n_mask = (nextra or {}).get("attention_mask") or (nextra or {}).get("attn_mask")
        if isinstance(p_mask, torch.Tensor):
            p_mask = _pad_mask_right(p_mask, T)
        if isinstance(n_mask, torch.Tensor):
            n_mask = _pad_mask_right(n_mask, T)

        if isinstance(p_mask, torch.Tensor) and isinstance(n_mask, torch.Tensor):
            # Use null mask at alpha≈0, pos mask at alpha≈1, otherwise OR them
            eps = 1e-6
            if alpha <= eps:
                merged = n_mask
            elif alpha >= 1.0 - eps:
                merged = p_mask
            else:
                merged = torch.maximum(n_mask.to(torch.int64), p_mask.to(torch.int64)).to(n_mask.dtype)
            new_extra["attention_mask"] = merged
            new_extra["attn_mask"] = merged
        elif isinstance(n_mask, torch.Tensor):
            new_extra["attention_mask"] = n_mask
            new_extra["attn_mask"] = n_mask
        elif isinstance(p_mask, torch.Tensor):
            new_extra["attention_mask"] = p_mask
            new_extra["attn_mask"] = p_mask

        # Optional soft cap on delta RMS (helps with very long prompts)
        if cap and cap > 0:
            delta = new_t - nt_p
            rms = torch.sqrt(torch.mean(delta * delta, dim=(-2, -1), keepdim=True) + 1e-12)
            scale = torch.minimum(torch.ones_like(rms), cap / rms)
            new_t = nt_p + delta * scale

        out.append((new_t, new_extra))

    return out


def _lerp(FROM, TO, t):
    """Linear interpolation: (1-t)*FROM + t*TO

    Thin wrapper to avoid argument order mistakes.
    """
    return _interp_conditioning(TO, FROM, float(t))


# ============================================================
# Main Node Class
# ============================================================

class ArchAi3D_Qwen_Encoder_V2(io.ComfyNode):
    """Qwen-VL encoder V2 with interpolation-based conditioning strength.

    Features:
    - 6 image inputs: 3 for Qwen-VL vision encoder + 3 for VAE reference latents
    - ChatML formatting for proper Qwen-VL 2.5 conditioning (always enabled)
    - Per-image latent strength controls (adjust each reference image independently)
    - Interpolation-based conditioning strength (stable behavior with system prompts)
    - Auto-labeling for multi-image inputs (when 2+ images)
    - RGB-only extraction (alpha channel removed automatically)
    - No size validation (expects pre-scaled images from Dual-Scale node)

    V2 Improvements:
    - Uses baseline interpolation instead of raw multiply
    - No "10× spike" when adding system prompts
    - Scales pooled_output along with token embeddings
    - ChatML-only (no llama_template parameter)

    Outputs:
    - conditioning: Text+vision embeddings with reference latents attached
    - latent: Image1 latent in standard format (for VAEDecode)
    - formatted_prompt: Final ChatML-formatted prompt with vision tokens
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ArchAi3D_Qwen_Encoder_V2",
            category="ArchAi3d/Qwen",
            inputs=[
                # === Core Inputs ===
                io.Clip.Input("clip",
                            tooltip="Qwen-VL CLIP model for encoding text and vision tokens"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True,
                              tooltip="Text prompt (vision tokens inserted automatically in ChatML format)"),
                io.Vae.Input("vae", optional=True,
                           tooltip="VAE for encoding reference latents (required if using latent images)"),

                # === Qwen-VL Images (Vision Encoder Path) ===
                io.Image.Input("image1_vl", optional=True,
                             tooltip="Image 1 for vision encoder (RGB only, expects correct size)"),
                io.Image.Input("image2_vl", optional=True,
                             tooltip="Image 2 for vision encoder (RGB only, expects correct size)"),
                io.Image.Input("image3_vl", optional=True,
                             tooltip="Image 3 for vision encoder (RGB only, expects correct size)"),

                # === Reference Latent Images (VAE Encoder Path) ===
                io.Image.Input("image1_latent", optional=True,
                             tooltip="Image 1 for reference latent (RGB only, expects correct size)"),
                io.Image.Input("image2_latent", optional=True,
                             tooltip="Image 2 for reference latent (RGB only, expects correct size)"),
                io.Image.Input("image3_latent", optional=True,
                             tooltip="Image 3 for reference latent (RGB only, expects correct size)"),

                # === Prompt Formatting ===
                io.String.Input("system_prompt", multiline=True, default="",
                              tooltip="Optional system prompt (wrapped in ChatML <|im_start|>system block)"),

                # === Image Label Customization ===
                io.String.Input("image1_label", default="Image 1",
                              tooltip="Custom label for Image 1 (e.g., 'Image 1 (target)', 'Image 1 (room)')"),
                io.String.Input("image2_label", default="Image 2",
                              tooltip="Custom label for Image 2 (e.g., 'Image 2 (style ref)', 'Image 2 (material)')"),
                io.String.Input("image3_label", default="Image 3",
                              tooltip="Custom label for Image 3 (e.g., 'Image 3 (color ref)', 'Image 3 (lighting)')"),

                # === Two-Stage Strength Control (V2) ===
                io.Float.Input("context_strength", default=1.0, min=0.0, max=1.5, step=0.01,
                             tooltip="V2 Stage A: Controls system prompt + labels influence (0.0=vision only, 1.0=full context). Turn down if system feels too heavy!"),
                io.Float.Input("user_strength", default=1.0, min=0.0, max=1.5, step=0.01,
                             tooltip="V2 Stage B: Controls user text influence (0.0=no user text, 1.0=full user text). Typical: 0.35-0.8"),

                # === Image Latent Strength Controls ===
                io.Float.Input("image1_latent_strength", default=1.0, min=0.0, max=2.0, step=0.01,
                             tooltip="Image1 latent strength (1.0=normal, <1.0=weaker, >1.0=stronger)"),
                io.Float.Input("image2_latent_strength", default=1.0, min=0.0, max=2.0, step=0.01,
                             tooltip="Image2 latent strength (1.0=normal, <1.0=weaker, >1.0=stronger)"),
                io.Float.Input("image3_latent_strength", default=1.0, min=0.0, max=2.0, step=0.01,
                             tooltip="Image3 latent strength (1.0=normal, <1.0=weaker, >1.0=stronger)"),

                # === Debug Options ===
                io.Boolean.Input("debug_mode", default=False,
                               tooltip="Enable console logging (shows strengths, shapes, and formatted prompt)"),
            ],
            outputs=[
                io.Conditioning.Output("conditioning",
                                      tooltip="Text+vision embeddings with reference latents metadata attached"),
                io.Latent.Output("latent",
                                tooltip="Image1 latent in standard format (for VAEDecode or other latent nodes)"),
                io.String.Output("formatted_prompt",
                                tooltip="Final ChatML-formatted prompt with vision tokens (for debugging)"),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, vae=None,
                image1_vl=None, image2_vl=None, image3_vl=None,
                image1_latent=None, image2_latent=None, image3_latent=None,
                system_prompt="",
                image1_label="Image 1",
                image2_label="Image 2",
                image3_label="Image 3",
                context_strength=1.0,
                user_strength=1.0,
                image1_latent_strength=1.0,
                image2_latent_strength=1.0,
                image3_latent_strength=1.0,
                debug_mode=False,
                auto_label=True) -> io.NodeOutput:  # Hidden params with defaults

        # ============================================================
        # SECTION 1: Collect Images for Qwen-VL Vision Encoder
        # ============================================================
        images_vl = []
        vl_inputs = [image1_vl, image2_vl, image3_vl]

        # Extract RGB channels only (remove alpha if present)
        for img in vl_inputs:
            if img is not None:
                images_vl.append(img[:, :, :, :3])

        # ============================================================
        # SECTION 2: Build Vision Token String with Custom Labels
        # ============================================================
        # Map custom labels to images
        custom_labels = [image1_label, image2_label, image3_label]

        if len(images_vl) == 0:
            image_prompt = ""
        elif len(images_vl) == 1:
            # Single image: use label if auto_label is enabled
            if auto_label:
                image_prompt = f"{custom_labels[0]}: <|vision_start|><|image_pad|><|vision_end|>\n"
            else:
                image_prompt = "<|vision_start|><|image_pad|><|vision_end|>"
        else:
            # Multiple images: auto-label if enabled
            if auto_label:
                image_prompt = "".join([
                    f"{custom_labels[i]}: <|vision_start|><|image_pad|><|vision_end|>\n"
                    for i in range(len(images_vl))
                ])
            else:
                image_prompt = "".join([
                    "<|vision_start|><|image_pad|><|vision_end|>\n"
                    for _ in range(len(images_vl))
                ])

        # ============================================================
        # SECTION 3: Encode Reference Latents with Strength Control
        # ============================================================
        ref_latents = []
        if vae is not None:
            latent_inputs = [
                (image1_latent, image1_latent_strength),
                (image2_latent, image2_latent_strength),
                (image3_latent, image3_latent_strength)
            ]
            for img, strength in latent_inputs:
                if img is None:
                    continue

                # Encode image to latent (RGB only)
                ref_latent = vae.encode(img[:, :, :, :3])

                # Apply strength multiplier to control reference influence
                if abs(strength - 1.0) > 0.001:
                    ref_latent = ref_latent * strength

                ref_latents.append(ref_latent)

        # Prepare latent output (image1 only, for VAEDecode)
        combined_latent = None
        if vae is not None and image1_latent is not None:
            ref_latent = vae.encode(image1_latent[:, :, :, :3])

            # Apply strength to output latent
            if abs(image1_latent_strength - 1.0) > 0.001:
                ref_latent = ref_latent * image1_latent_strength

            combined_latent = {"samples": ref_latent}

        # ============================================================
        # SECTION 4: Build ChatML-Formatted Prompts (Full + Baseline)
        # ============================================================
        # Combine vision tokens + text with proper separation
        user_content = image_prompt
        if prompt:
            user_content += "\n" + prompt if image_prompt else prompt

        # Full prompt (system + images + user text)
        if system_prompt:
            formatted_prompt = (
                f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n"
                f"<|im_start|>user\n{user_content}\n<|im_end|>\n"
                f"<|im_start|>assistant"
            )
        else:
            formatted_prompt = (
                f"<|im_start|>user\n{user_content}\n<|im_end|>\n"
                f"<|im_start|>assistant"
            )

        # ============================================================
        # SECTION 5: Two-Stage Tokenize and Encode (V2 FIX)
        # ============================================================
        # Encode THREE variants (no llama_template to avoid double-templating)

        # 1) VISION-ONLY: Images only, no text at all
        if len(images_vl) > 0:
            vision_only_prompt = "".join([
                "<|vision_start|><|image_pad|><|vision_end|>\n"
                for _ in range(len(images_vl))
            ])
        else:
            vision_only_prompt = ""

        tokens_vis = clip.tokenize(vision_only_prompt, images=images_vl)  # NO llama_template!
        if hasattr(clip, "encode_from_tokens_scheduled"):
            cond_vision = clip.encode_from_tokens_scheduled(tokens_vis)
        else:
            cond_vision = clip.encode_from_tokens(tokens_vis)

        # 2) NO-USER: System + labels + images (empty user text)
        if system_prompt:
            no_user_prompt = (
                f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n"
                f"<|im_start|>user\n{image_prompt}\n<|im_end|>\n"
                f"<|im_start|>assistant"
            )
        else:
            no_user_prompt = (
                f"<|im_start|>user\n{image_prompt}\n<|im_end|>\n"
                f"<|im_start|>assistant"
            )

        tokens_no_user = clip.tokenize(no_user_prompt, images=images_vl)  # NO llama_template!
        if hasattr(clip, "encode_from_tokens_scheduled"):
            cond_no_user = clip.encode_from_tokens_scheduled(tokens_no_user)
        else:
            cond_no_user = clip.encode_from_tokens(tokens_no_user)

        # 3) FULL: System + labels + images + user text
        tokens_full = clip.tokenize(formatted_prompt, images=images_vl)  # NO llama_template!
        if hasattr(clip, "encode_from_tokens_scheduled"):
            conditioning_full = clip.encode_from_tokens_scheduled(tokens_full)
        else:
            conditioning_full = clip.encode_from_tokens(tokens_full)

        # Apply TWO-STAGE interpolation to fix "10× spike" bug
        # Stage A: vision → vision+context (controls system prompt influence)
        cond_context = _lerp(cond_vision, cond_no_user, context_strength)

        # Stage B: vision+context → full (controls user text influence)
        conditioning = _lerp(cond_context, conditioning_full, user_strength)

        # ============================================================
        # SECTION 6: Debug Output (if enabled)
        # ============================================================
        if debug_mode:
            print("=" * 70)
            print("ArchAi3D Qwen Encoder V2 - Debug Output (Two-Stage)")
            print("=" * 70)
            print(f"Two-Stage Interpolation:")
            print(f"  • Stage A (context): vision → vision+context | strength={context_strength:.3f}")
            print(f"  • Stage B (user): context → full | strength={user_strength:.3f}")
            print(f"  • Method: (1-t)*FROM + t*TO (baseline-first masks/extras)")

            # Calculate deltas to show if interpolation is working
            def _delta_norm(a, b):
                """Calculate delta with padding for different sequence lengths."""
                ta, tb = a[0][0], b[0][0]
                # Pad to same length if needed
                if ta.shape[-2] != tb.shape[-2]:
                    max_len = max(ta.shape[-2], tb.shape[-2])
                    ta = _pad_tokens_right(ta, max_len)
                    tb = _pad_tokens_right(tb, max_len)
                return float(torch.linalg.norm(ta - tb))

            delta_context = _delta_norm(cond_no_user, cond_vision)
            delta_user = _delta_norm(conditioning_full, cond_no_user)
            print(f"\n  • Δ_context = {delta_context:.3f} (vision vs no-user)")
            print(f"  • Δ_user = {delta_user:.3f} (no-user vs full)")

            print(f"\nLatent Strengths:")
            for idx, (img, strength, name) in enumerate([
                (image1_latent, image1_latent_strength, "Image1"),
                (image2_latent, image2_latent_strength, "Image2"),
                (image3_latent, image3_latent_strength, "Image3")
            ]):
                if img is not None:
                    status = "applied" if abs(strength - 1.0) > 0.001 else "default"
                    print(f"  • {name}: {strength:.3f} ({status})")

            print(f"\nActive Inputs:")
            if prompt:
                print(f"  • User Text: {len(prompt)} chars")
            if system_prompt:
                print(f"  • System Prompt: {len(system_prompt)} chars")
            for idx, img in enumerate([image1_vl, image2_vl, image3_vl]):
                if img is not None:
                    print(f"  • VL Image{idx+1}: {img.shape[2]}×{img.shape[1]} px")

            print(f"\nOutput Shapes:")
            print(f"  • Conditioning: {conditioning[0][0].shape}")
            if combined_latent:
                print(f"  • Latent: {combined_latent['samples'].shape}")

            # Check what's in the conditioning metadata
            print(f"\nConditioning Metadata (extras dict):")
            if len(conditioning) > 0 and len(conditioning[0]) > 1:
                extras = conditioning[0][1]
                if isinstance(extras, dict):
                    print(f"  • Keys in extras: {list(extras.keys())}")
                    if "pooled_output" in extras:
                        print(f"  • pooled_output shape: {extras['pooled_output'].shape}")
                    else:
                        print(f"  • ⚠️ NO pooled_output found!")
                    if "reference_latents" in extras:
                        print(f"  • reference_latents: {len(extras['reference_latents'])} latents")
                    else:
                        print(f"  • ℹ️ reference_latents not attached yet (attached later)")
                else:
                    print(f"  • extras is not a dict: {type(extras)}")
            else:
                print(f"  • ⚠️ Conditioning structure unexpected")

            print(f"\nPrompt Variants:")
            print(f"  [1] Vision-only: {len(vision_only_prompt)} chars")
            print(f"      {vision_only_prompt[:100]}...")
            print(f"  [2] No-user: {len(no_user_prompt)} chars")
            print(f"      {no_user_prompt[:100]}...")
            print(f"  [3] Full: {len(formatted_prompt)} chars")
            print(f"      {formatted_prompt[:100]}...")
            print("=" * 70)

        # ============================================================
        # SECTION 7: Attach Reference Latents to Conditioning
        # ============================================================
        if ref_latents:
            conditioning = node_helpers.conditioning_set_values(
                conditioning, {"reference_latents": ref_latents}, append=True
            )

        # ============================================================
        # SECTION 8: Return Outputs
        # ============================================================
        return io.NodeOutput(conditioning, combined_latent, formatted_prompt)


# ---- Extension API glue (so ComfyUI auto-discovers this node) ----
class ArchAi3DQwenEncoderV2Extension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [ArchAi3D_Qwen_Encoder_V2]

async def comfy_entrypoint():
    return ArchAi3DQwenEncoderV2Extension()
