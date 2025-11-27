# ArchAi3D Qwen Encoder V3 — Research-Based Conditioning Control
#
# OVERVIEW:
# V3 uses research-validated presets + CFG scale output for optimal conditioning control.
# Based on extensive research of official Qwen-Image-Edit and ComfyUI best practices.
#
# WHY V3 EXISTS:
# - V2's manual sliders (context_strength, user_strength) have effects that are too subtle
# - Users reported "not very visible" changes when adjusting sliders
# - Research shows: Official Qwen-Image-Edit uses CFG SCALE (not embedding interpolation)
# - Solution: Provide BOTH preset-based interpolation + CFG scale recommendation
#
# KEY DIFFERENCES FROM V2:
# - V2: Manual sliders (context_strength, user_strength) with subtle effects
# - V3: 5 calibrated presets (Image-Dominant → Text-Dominant) with clear differences
# - V2: Two-stage interpolation (vision → context → full)
# - V3: Same interpolation, but with extreme preset values for visibility
# - V3: Optional manual override with extended range (0.0-3.0) for advanced users
#
# RESEARCH-BASED DUAL CONTROL SYSTEM:
#
# CONTROL METHOD 1: Preset Interpolation (Encoding-time)
# - Adjusts embedding interpolation between vision-only → full conditioning
# - Creates distinct conditioning variants for ConditioningAverage workflow
#
# CONTROL METHOD 2: CFG Scale (Sampling-time) **RECOMMENDED**
# - Based on official Qwen-Image-Edit implementation
# - Range: 2.0 (strong image fidelity) → 5.5 (strong text guidance)
# - More effective than embedding interpolation alone
# - Connect "recommended_cfg" output to KSampler's cfg parameter
#
# CONDITIONING BALANCE PRESETS (with CFG recommendations):
#
# 1. "Image-Dominant" (interp: 0.2/0.1, CFG: 2.5)
#    - Follow reference images EXACTLY, ignore most text
#    - Use case: Precise material matching, exact reproduction
#
# 2. "Image-Priority" (interp: 0.5/0.3, CFG: 3.0)
#    - Reference images guide result, text provides hints
#    - Use case: Style transfer with small tweaks
#
# 3. "Balanced" (interp: 1.0/1.0, CFG: 4.0) [DEFAULT]
#    - Equal weight to images and text (Qwen default)
#    - Use case: General editing with good balance
#
# 4. "Text-Priority" (interp: 1.3/1.3, CFG: 4.75)
#    - Text instructions guide result, images provide context
#    - Use case: Creative reinterpretation, dramatic changes
#
# 5. "Text-Dominant" (interp: 1.5/1.5, CFG: 5.5)
#    - Follow text closely, images as loose reference
#    - Use case: Text-to-image with image hints
#
# WORKFLOW WITH CONDITIONINGAVERAGE:
#
# Method 1: Blend two extreme presets
#   V3 Encoder (Image-Dominant) ──┐
#                                  ├──> ConditioningAverage (0.5) ──> Balanced result
#   V3 Encoder (Text-Dominant) ───┘
#
# Method 2: Blend preset with baseline
#   V3 Encoder (Text-Priority) ────┐
#                                   ├──> ConditioningAverage (0.7) ──> Fine-tuned
#   V3 Encoder (Balanced) ─────────┘
#
# ADVANCED: Manual Override
# - Set preset to "Custom" to reveal manual sliders
# - Range: 0.0-3.0 (extended for extreme conditioning control)
# - Allows fine-tuning beyond presets (for advanced users)
#
# INPUTS:
# - Same as V2: 3 VL images, 3 latent images, text prompt, system prompt
# - NEW: conditioning_balance preset (6 options including Custom)
# - NEW: manual_context_strength (0.0-3.0, hidden unless preset="Custom")
# - NEW: manual_user_strength (0.0-3.0, hidden unless preset="Custom")
#
# OUTPUTS:
# - conditioning: Text+vision embeddings (same as V2)
# - latent: Image1 latent (same as V2)
# - formatted_prompt: ChatML prompt (same as V2)
# - **recommended_cfg: CFG scale value** (NEW! Connect to KSampler)
#
# Author: Amir Ferdos (ArchAi3d)
# Email: Amir84ferdos@gmail.com
# LinkedIn: https://www.linkedin.com/in/archai3d/
# GitHub: https://github.com/amir84ferdos
# Category: ArchAi3d/Qwen/Encoders
# Node ID: ArchAi3D_Qwen_Encoder_V3
# Version: 1.0.0
# License: MIT

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override
import node_helpers
import torch
import copy


# ============================================================
# Conditioning Balance Presets (Calibrated for Visibility)
# ============================================================

CONDITIONING_PRESETS = {
    "Image-Dominant": {
        "context_strength": 0.2,
        "user_strength": 0.1,
        "cfg_scale": 2.5,
        "description": "Follow images EXACTLY, ignore most text"
    },
    "Image-Priority": {
        "context_strength": 0.5,
        "user_strength": 0.3,
        "cfg_scale": 3.0,
        "description": "Images guide result, text provides hints"
    },
    "Balanced": {
        "context_strength": 1.0,
        "user_strength": 1.0,
        "cfg_scale": 4.0,
        "description": "Equal weight (Qwen default behavior)"
    },
    "Text-Priority": {
        "context_strength": 1.3,
        "user_strength": 1.3,
        "cfg_scale": 4.75,
        "description": "Text guides result, images provide context"
    },
    "Text-Dominant": {
        "context_strength": 1.5,
        "user_strength": 1.5,
        "cfg_scale": 5.5,
        "description": "Follow text closely, images as loose reference"
    },
    "Custom": {
        "context_strength": 1.0,
        "user_strength": 1.0,
        "cfg_scale": 4.0,
        "description": "Manual control (advanced users)"
    }
}


# ============================================================
# Helper Functions (Copied from V2)
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

        # Handle attention masks
        p_mask = (pextra or {}).get("attention_mask") or (pextra or {}).get("attn_mask")
        n_mask = (nextra or {}).get("attention_mask") or (nextra or {}).get("attn_mask")
        if isinstance(p_mask, torch.Tensor):
            p_mask = _pad_mask_right(p_mask, T)
        if isinstance(n_mask, torch.Tensor):
            n_mask = _pad_mask_right(n_mask, T)

        if isinstance(p_mask, torch.Tensor) and isinstance(n_mask, torch.Tensor):
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

        # Optional soft cap
        if cap and cap > 0:
            delta = new_t - nt_p
            rms = torch.sqrt(torch.mean(delta * delta, dim=(-2, -1), keepdim=True) + 1e-12)
            scale = torch.minimum(torch.ones_like(rms), cap / rms)
            new_t = nt_p + delta * scale

        out.append((new_t, new_extra))

    return out


def _lerp(FROM, TO, t):
    """Linear interpolation: (1-t)*FROM + t*TO"""
    return _interp_conditioning(TO, FROM, float(t))


# ============================================================
# Main Node Class
# ============================================================

class ArchAi3D_Qwen_Encoder_V3(io.ComfyNode):
    """Qwen-VL encoder V3 - Research-validated conditioning control.

    Features:
    - 6 image inputs: 3 for Qwen-VL vision encoder + 3 for VAE reference latents
    - 5 calibrated presets (Image-Dominant → Text-Dominant)
    - **NEW: CFG scale output** - Research-validated control method (connect to KSampler)
    - Preset interpolation for ConditioningAverage workflows
    - Optional manual override for advanced users

    DUAL CONTROL METHODS:
    1. **CFG Scale (RECOMMENDED)**: Connect recommended_cfg to KSampler's cfg parameter
       - Range: 2.5 (strong image fidelity) → 5.5 (strong text guidance)
       - Based on official Qwen-Image-Edit implementation
       - Most effective control method

    2. **Preset Interpolation**: Use with ConditioningAverage for advanced blending
       - Create two V3 nodes with different presets
       - Blend with ConditioningAverage at desired ratio
       - Example: "Image-Dominant" + "Text-Priority" @ 0.5 = custom balance

    Outputs:
    - conditioning: Text+vision embeddings with reference latents attached
    - latent: Image1 latent in standard format (for VAEDecode)
    - formatted_prompt: Final ChatML-formatted prompt with vision tokens
    - **recommended_cfg: CFG scale value** (NEW! Connect to KSampler)
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ArchAi3D_Qwen_Encoder_V3",
            category="ArchAi3d/Qwen/Encoders",
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

                # === NEW: Conditioning Balance Preset ===
                io.Combo.Input("conditioning_balance",
                            options=list(CONDITIONING_PRESETS.keys()),
                            tooltip="V3 PRESET: Choose conditioning balance (Image-Dominant → Text-Dominant). Works great with ConditioningAverage!"),

                # === Optional: External Conditioning Balance Override ===
                io.String.Input("conditioning_balance_override", default="",
                              tooltip="OPTIONAL: Connect ⚖️ Conditioning Balance node here to override the preset above. Leave empty to use the dropdown."),

                # === Manual Override (Hidden by Default) ===
                io.Float.Input("manual_context_strength", default=1.0, min=0.0, max=3.0, step=0.01,
                             tooltip="CUSTOM ONLY: Manual context strength (only used when preset = Custom). Extended range: 0.0-3.0 for extreme conditioning control"),
                io.Float.Input("manual_user_strength", default=1.0, min=0.0, max=3.0, step=0.01,
                             tooltip="CUSTOM ONLY: Manual user strength (only used when preset = Custom). Extended range: 0.0-3.0 for extreme conditioning control"),

                # === Image Label Customization ===
                io.String.Input("image1_label", default="Image 1",
                              tooltip="Custom label for Image 1"),
                io.String.Input("image2_label", default="Image 2",
                              tooltip="Custom label for Image 2"),
                io.String.Input("image3_label", default="Image 3",
                              tooltip="Custom label for Image 3"),

                # === Image Latent Strength Controls ===
                io.Float.Input("image1_latent_strength", default=1.0, min=0.0, max=2.0, step=0.01,
                             tooltip="Image1 latent strength (1.0=normal, <1.0=weaker, >1.0=stronger)"),
                io.Float.Input("image2_latent_strength", default=1.0, min=0.0, max=2.0, step=0.01,
                             tooltip="Image2 latent strength (1.0=normal, <1.0=weaker, >1.0=stronger)"),
                io.Float.Input("image3_latent_strength", default=1.0, min=0.0, max=2.0, step=0.01,
                             tooltip="Image3 latent strength (1.0=normal, <1.0=weaker, >1.0=stronger)"),

                # === Debug Options ===
                io.Boolean.Input("debug_mode", default=False,
                               tooltip="Enable console logging (shows preset values, strengths, shapes)"),
            ],
            outputs=[
                io.Conditioning.Output("conditioning",
                                      tooltip="Text+vision embeddings with reference latents metadata attached"),
                io.Latent.Output("latent",
                                tooltip="Image1 latent in standard format (for VAEDecode or other latent nodes)"),
                io.String.Output("formatted_prompt",
                                tooltip="Final ChatML-formatted prompt with vision tokens (for debugging)"),
                io.Float.Output("recommended_cfg",
                               tooltip="⭐ NEW: Recommended CFG scale based on preset (2.5-5.5). Connect to KSampler's cfg parameter for optimal image/text balance!"),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, vae=None,
                image1_vl=None, image2_vl=None, image3_vl=None,
                image1_latent=None, image2_latent=None, image3_latent=None,
                system_prompt="",
                conditioning_balance="Balanced",
                conditioning_balance_override="",
                manual_context_strength=1.0,
                manual_user_strength=1.0,
                image1_label="Image 1",
                image2_label="Image 2",
                image3_label="Image 3",
                image1_latent_strength=1.0,
                image2_latent_strength=1.0,
                image3_latent_strength=1.0,
                debug_mode=False,
                auto_label=True) -> io.NodeOutput:

        # ============================================================
        # SECTION 1: Determine Strength Values and CFG from Preset
        # ============================================================

        # Check if override is provided and valid
        if conditioning_balance_override and conditioning_balance_override.strip():
            override_value = conditioning_balance_override.strip()
            if override_value in CONDITIONING_PRESETS:
                conditioning_balance = override_value
                if debug_mode:
                    print(f"[V3] Using override conditioning_balance: {conditioning_balance}")

        if conditioning_balance == "Custom":
            # Use manual values
            context_strength = manual_context_strength
            user_strength = manual_user_strength
            # Calculate CFG from manual strengths (interpolate between 2.5-5.5)
            avg_strength = (context_strength + user_strength) / 2.0
            if avg_strength < 0.3:
                recommended_cfg = 2.5
            elif avg_strength < 0.7:
                recommended_cfg = 2.5 + (avg_strength - 0.3) * (3.0 - 2.5) / 0.4
            elif avg_strength < 1.1:
                recommended_cfg = 3.0 + (avg_strength - 0.7) * (4.0 - 3.0) / 0.4
            elif avg_strength < 1.4:
                recommended_cfg = 4.0 + (avg_strength - 1.1) * (4.75 - 4.0) / 0.3
            else:
                recommended_cfg = 4.75 + min((avg_strength - 1.4) * (5.5 - 4.75) / 0.2, 0.75)
        else:
            # Use preset values
            preset = CONDITIONING_PRESETS.get(conditioning_balance, CONDITIONING_PRESETS["Balanced"])
            context_strength = preset["context_strength"]
            user_strength = preset["user_strength"]
            recommended_cfg = preset["cfg_scale"]

        # ============================================================
        # SECTION 2: Collect Images for Qwen-VL Vision Encoder
        # ============================================================
        images_vl = []
        vl_inputs = [image1_vl, image2_vl, image3_vl]

        # Extract RGB channels only
        for img in vl_inputs:
            if img is not None:
                images_vl.append(img[:, :, :, :3])

        # ============================================================
        # SECTION 3: Build Vision Token String with Custom Labels
        # ============================================================
        custom_labels = [image1_label, image2_label, image3_label]

        if len(images_vl) == 0:
            image_prompt = ""
        elif len(images_vl) == 1:
            if auto_label:
                image_prompt = f"{custom_labels[0]}: <|vision_start|><|image_pad|><|vision_end|>\n"
            else:
                image_prompt = "<|vision_start|><|image_pad|><|vision_end|>"
        else:
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
        # SECTION 4: Encode Reference Latents with Strength Control
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

                ref_latent = vae.encode(img[:, :, :, :3])

                if abs(strength - 1.0) > 0.001:
                    ref_latent = ref_latent * strength

                ref_latents.append(ref_latent)

        # Prepare latent output (image1 only)
        combined_latent = None
        if vae is not None and image1_latent is not None:
            ref_latent = vae.encode(image1_latent[:, :, :, :3])

            if abs(image1_latent_strength - 1.0) > 0.001:
                ref_latent = ref_latent * image1_latent_strength

            combined_latent = {"samples": ref_latent}

        # ============================================================
        # SECTION 5: Build ChatML-Formatted Prompts
        # ============================================================
        user_content = image_prompt
        if prompt:
            user_content += "\n" + prompt if image_prompt else prompt

        # Full prompt
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
        # SECTION 6: Two-Stage Tokenize and Encode
        # ============================================================
        # 1) VISION-ONLY: Images only, no text
        if len(images_vl) > 0:
            vision_only_prompt = "".join([
                "<|vision_start|><|image_pad|><|vision_end|>\n"
                for _ in range(len(images_vl))
            ])
        else:
            vision_only_prompt = ""

        tokens_vis = clip.tokenize(vision_only_prompt, images=images_vl)
        if hasattr(clip, "encode_from_tokens_scheduled"):
            cond_vision = clip.encode_from_tokens_scheduled(tokens_vis)
        else:
            cond_vision = clip.encode_from_tokens(tokens_vis)

        # 2) NO-USER: System + labels + images (no user text)
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

        tokens_no_user = clip.tokenize(no_user_prompt, images=images_vl)
        if hasattr(clip, "encode_from_tokens_scheduled"):
            cond_no_user = clip.encode_from_tokens_scheduled(tokens_no_user)
        else:
            cond_no_user = clip.encode_from_tokens(tokens_no_user)

        # 3) FULL: System + labels + images + user text
        tokens_full = clip.tokenize(formatted_prompt, images=images_vl)
        if hasattr(clip, "encode_from_tokens_scheduled"):
            conditioning_full = clip.encode_from_tokens_scheduled(tokens_full)
        else:
            conditioning_full = clip.encode_from_tokens(tokens_full)

        # Apply TWO-STAGE interpolation (same as V2, but with preset values)
        # Stage A: vision → vision+context (controls system prompt influence)
        cond_context = _lerp(cond_vision, cond_no_user, context_strength)

        # Stage B: vision+context → full (controls user text influence)
        conditioning = _lerp(cond_context, conditioning_full, user_strength)

        # ============================================================
        # SECTION 7: Debug Output
        # ============================================================
        if debug_mode:
            print("=" * 70)
            print("ArchAi3D Qwen Encoder V3 - Debug Output")
            print("=" * 70)
            print(f"Conditioning Balance Preset: {conditioning_balance}")
            if conditioning_balance == "Custom":
                print(f"  • Manual context_strength: {context_strength:.3f}")
                print(f"  • Manual user_strength: {user_strength:.3f}")
                print(f"  • Calculated recommended_cfg: {recommended_cfg:.2f}")
            else:
                preset = CONDITIONING_PRESETS[conditioning_balance]
                print(f"  • Preset context_strength: {context_strength:.3f}")
                print(f"  • Preset user_strength: {user_strength:.3f}")
                print(f"  • Preset recommended_cfg: {recommended_cfg:.2f}")
                print(f"  • Description: {preset['description']}")

            print(f"\n⭐ CFG SCALE RECOMMENDATION:")
            print(f"  • Connect 'recommended_cfg' output to KSampler's cfg parameter")
            print(f"  • Current value: {recommended_cfg:.2f}")
            print(f"  • Range: 2.5 (strong image) → 5.5 (strong text)")
            print(f"  • This is the PRIMARY control method (research-validated)")

            print(f"\nTwo-Stage Interpolation:")
            print(f"  • Stage A (context): vision → vision+context | strength={context_strength:.3f}")
            print(f"  • Stage B (user): context → full | strength={user_strength:.3f}")

            # Calculate deltas
            def _delta_norm(a, b):
                ta, tb = a[0][0], b[0][0]
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

            print("=" * 70)

        # ============================================================
        # SECTION 8: Attach Reference Latents to Conditioning
        # ============================================================
        if ref_latents:
            conditioning = node_helpers.conditioning_set_values(
                conditioning, {"reference_latents": ref_latents}, append=True
            )

        # ============================================================
        # SECTION 9: Return Outputs (Including CFG Recommendation)
        # ============================================================
        return io.NodeOutput(conditioning, combined_latent, formatted_prompt, recommended_cfg)


# ---- Extension API glue ----
class ArchAi3DQwenEncoderV3Extension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [ArchAi3D_Qwen_Encoder_V3]

async def comfy_entrypoint():
    return ArchAi3DQwenEncoderV3Extension()
