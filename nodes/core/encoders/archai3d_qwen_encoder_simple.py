# ArchAi3D_Qwen_Encoder — Advanced Qwen-VL encoder with encoder controls
#
# FEATURES:
# - 3 images go to Qwen-VL tokenize (as-is, RGB only)
# - 3 images go to VAE.encode as reference latents (RGB only, standard 4D format [B,C,H,W])
# - Conditioning strength control (multiplies final embeddings)
# - Latent strength control (multiplies reference latent values per-image)
# - Auto-labeling for multi-image prompts (hidden by default)
# - Debug mode for troubleshooting
#
# STRENGTH CONTROLS:
# 1. conditioning_strength (0.0-2.0):
#    - Multiplies the final text+vision embeddings (after CLIP encoding)
#    - Controls how strongly the sampler follows the entire prompt
#    - Acts like CFG weight for the conditioning
#
# 2. image1/2/3_latent_strength (0.0-2.0) [EXPERIMENTAL]:
#    - Multiplies individual reference latent tensor values (before model processing)
#    - Controls the "signal strength" of each reference image in latent space
#    - Similar to Interior Encoder's "multiply_latent" mode
#    - Lower values = weaker reference influence, Higher values = stronger reference influence
#    - NOT adding noise - adjusting attention weight by scaling latent values
#
# Author: Amir Ferdos (ArchAi3d)
# Email: Amir84ferdos@gmail.com
# LinkedIn: https://www.linkedin.com/in/archai3d/
# GitHub: https://github.com/amir84ferdos
# Category: ArchAi3d/Qwen
# Node ID: ArchAi3D_Qwen_Encoder
# License: MIT

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override
import node_helpers
import torch


def ensure_even_dimensions(latent: torch.Tensor) -> torch.Tensor:
    """
    Ensure latent dimensions are even for patch processing.
    Required for model compatibility.

    Args:
        latent: Input latent tensor (4D format [B, C, H, W])

    Returns:
        Latent with even dimensions (may be padded)
    """
    if len(latent.shape) == 4:  # [B, C, H, W]
        _, _, h, w = latent.shape
        if h % 2 != 0 or w % 2 != 0:
            new_h = h + (h % 2)
            new_w = w + (w % 2)
            latent = torch.nn.functional.pad(
                latent,
                (0, new_w - w, 0, new_h - h),
                mode='constant',
                value=0
            )
    return latent

class ArchAi3D_Qwen_Encoder_Simple(io.ComfyNode):
    """Advanced Qwen-VL + Latent reference encoder with encoder controls.

    Features:
    - 6 explicit image inputs (3 VL + 3 latent paths)
    - No automatic resizing (use with Dual-Scale node for preprocessing)
    - RGB channel extraction for safety
    - ChatML system prompt formatting
    - Auto-labeling for multi-image inputs
    - 3 outputs: conditioning, latent, formatted_prompt

    This is the advanced version with more controls. For simple usage, use Qwen_Encoder_RAW.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ArchAi3D_Qwen_Encoder_Simple",
            category="ArchAi3d/Qwen",
            inputs=[
                # === Core Inputs ===
                io.Clip.Input("clip", tooltip="Qwen-VL CLIP model for tokenization and encoding"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True,
                              tooltip="User prompt text (vision tokens added automatically)"),
                io.Vae.Input("vae", optional=True,
                           tooltip="VAE for encoding reference latents (optional, needed for latent outputs)"),

                # === Qwen-VL Images (Vision Encoder Path) ===
                io.Image.Input("image1_vl", optional=True,
                             tooltip="Image 1 for Qwen-VL vision encoder (RGB channels only, no resize)"),
                io.Image.Input("image2_vl", optional=True,
                             tooltip="Image 2 for Qwen-VL vision encoder (RGB channels only, no resize)"),
                io.Image.Input("image3_vl", optional=True,
                             tooltip="Image 3 for Qwen-VL vision encoder (RGB channels only, no resize)"),

                # === Reference Latent Images (VAE Encoder Path) ===
                io.Image.Input("image1_latent", optional=True,
                             tooltip="Image 1 for VAE encoding as reference latent (RGB only, standard 4D format)"),
                io.Image.Input("image2_latent", optional=True,
                             tooltip="Image 2 for VAE encoding as reference latent (RGB only, standard 4D format)"),
                io.Image.Input("image3_latent", optional=True,
                             tooltip="Image 3 for VAE encoding as reference latent (RGB only, standard 4D format)"),

                # === Prompt Formatting ===
                io.String.Input("system_prompt", multiline=True, default="",
                              tooltip="Optional system prompt (uses ChatML format: <|im_start|>system...when provided)"),

                # === Conditioning Strength Control ===
                io.Float.Input("conditioning_strength", default=1.0, min=0.0, max=2.0, step=0.01,
                             tooltip="Global conditioning strength multiplier (1.0=normal, >1.0=stronger, <1.0=weaker). Acts like CFG weight."),

                # === Latent Strength Control (Experimental) ===
                io.Float.Input("image1_latent_strength", default=1.0, min=0.0, max=2.0, step=0.01,
                             tooltip="[EXPERIMENTAL] Multiplies image1 reference latent values (1.0=normal, >1.0=stronger, <1.0=weaker)"),
                io.Float.Input("image2_latent_strength", default=1.0, min=0.0, max=2.0, step=0.01,
                             tooltip="[EXPERIMENTAL] Multiplies image2 reference latent values (1.0=normal, >1.0=stronger, <1.0=weaker)"),
                io.Float.Input("image3_latent_strength", default=1.0, min=0.0, max=2.0, step=0.01,
                             tooltip="[EXPERIMENTAL] Multiplies image3 reference latent values (1.0=normal, >1.0=stronger, <1.0=weaker)"),

                # === Debug Options ===
                io.Boolean.Input("debug_mode", default=False,
                               tooltip="Print detailed info to console (conditioning shapes, strengths applied, etc.)"),
            ],
            outputs=[
                io.Conditioning.Output("conditioning", tooltip="Encoded conditioning with vision tokens and reference latents"),
                io.Latent.Output("latent", tooltip="image1_latent encoded output in standard format (compatible with VAEDecode)"),
                io.String.Output("formatted_prompt", tooltip="Final formatted prompt sent to the model (includes vision tokens and ChatML)"),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, vae=None,
                image1_vl=None, image2_vl=None, image3_vl=None,
                image1_latent=None, image2_latent=None, image3_latent=None,
                system_prompt="",
                conditioning_strength=1.0,
                image1_latent_strength=1.0,
                image2_latent_strength=1.0,
                image3_latent_strength=1.0,
                debug_mode=False,
                llama_template="",  # llama_template kept for backward compatibility
                auto_label=True, label_format="Image") -> io.NodeOutput:  # Hidden params with defaults

        # ============================================================
        # SECTION 1: Process Qwen-VL Images (Vision Encoder Path)
        # ============================================================
        images_vl = []
        vl_inputs = [image1_vl, image2_vl, image3_vl]

        # Collect valid images (RGB only, no resizing)
        for img in vl_inputs:
            if img is not None:
                images_vl.append(img[:, :, :, :3])  # Trim to RGB channels

        # ============================================================
        # SECTION 2: Build Vision Token Prompt with Smart Labeling
        # ============================================================
        if len(images_vl) == 0:
            image_prompt = ""
        elif len(images_vl) == 1:
            # Single image - no label needed
            image_prompt = "<|vision_start|><|image_pad|><|vision_end|>"
        else:
            # Multiple images - use labels if auto_label is enabled
            if auto_label:
                # Format: "Image 1: <vision_tokens>Image 2: <vision_tokens>..."
                image_prompt = "".join([
                    f"{label_format} {i+1}: <|vision_start|><|image_pad|><|vision_end|>"
                    for i in range(len(images_vl))
                ])
            else:
                # No labels, just concatenate vision tokens
                image_prompt = "".join([
                    "<|vision_start|><|image_pad|><|vision_end|>"
                    for _ in range(len(images_vl))
                ])

        # ============================================================
        # SECTION 3: Process Reference Latents (VAE Encoder Path)
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

                # Encode to latent (RGB only, no resizing)
                ref_latent = vae.encode(img[:, :, :, :3])

                # Apply latent strength multiplication (experimental feature)
                # How it works:
                # - Multiplies all values in the latent tensor by the strength factor
                # - strength < 1.0: Makes latent values smaller → weaker reference influence
                # - strength > 1.0: Makes latent values larger → stronger reference influence
                # - This controls the "signal strength" in latent space, NOT adding noise
                # - Similar to Interior Encoder's "multiply_latent" mode
                if abs(strength - 1.0) > 0.001:
                    ref_latent = ref_latent * strength

                # Ensure dimensions are even for patch processing
                ref_latent = ensure_even_dimensions(ref_latent)

                ref_latents.append(ref_latent)

        # Convert to standard ComfyUI latent format
        # Only output image1_latent (first latent) for VAEDecode compatibility
        # All latents still attached to conditioning metadata for model use
        combined_latent = None
        if vae is not None and image1_latent is not None:
            # Encode only image1_latent for the latent output
            ref_latent = vae.encode(image1_latent[:, :, :, :3])

            # Apply latent strength to output latent as well (for consistency)
            if abs(image1_latent_strength - 1.0) > 0.001:
                ref_latent = ref_latent * image1_latent_strength

            # Ensure dimensions are even for patch processing
            ref_latent = ensure_even_dimensions(ref_latent)

            # Wrap in standard ComfyUI format (required by VAEDecode)
            combined_latent = {"samples": ref_latent}

        # ============================================================
        # SECTION 4: Format Final Prompt (ChatML + Vision Tokens)
        # ============================================================
        if system_prompt:
            # Use ChatML format with system prompt
            formatted_prompt = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{image_prompt}{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        else:
            # Simple format: vision tokens + user prompt
            formatted_prompt = image_prompt + prompt

        # ============================================================
        # SECTION 5: Tokenize and Encode (Combined)
        # ============================================================
        # Encode everything together (text + all images)
        # Qwen-VL is trained with interleaved text and vision tokens

        template = llama_template.strip() or None
        tokens = clip.tokenize(formatted_prompt, images=images_vl, llama_template=template)

        # Use scheduled encoding if available
        if hasattr(clip, "encode_from_tokens_scheduled"):
            conditioning = clip.encode_from_tokens_scheduled(tokens)
        else:
            conditioning = clip.encode_from_tokens(tokens)

        # Apply global conditioning strength multiplier
        # Use abs comparison to avoid floating point precision issues
        if abs(conditioning_strength - 1.0) > 0.001:
            for i in range(len(conditioning)):
                conditioning[i][0] = conditioning[i][0] * conditioning_strength

        # Debug output
        if debug_mode:
            print("=" * 70)
            print("ArchAi3D_Qwen_Encoder - Debug Info")
            print("=" * 70)
            print(f"Encoding Mode: Combined (text + images together)")
            print(f"Conditioning Strength: {conditioning_strength}")
            print(f"Strength Applied: {'NO (= 1.0)' if abs(conditioning_strength - 1.0) <= 0.001 else 'YES'}")
            print(f"\nLatent Strengths (Experimental):")
            latent_strengths = [
                (image1_latent, image1_latent_strength, "Image1"),
                (image2_latent, image2_latent_strength, "Image2"),
                (image3_latent, image3_latent_strength, "Image3")
            ]
            for img, strength, name in latent_strengths:
                if img is not None:
                    applied = "YES" if abs(strength - 1.0) > 0.001 else "NO (= 1.0)"
                    print(f"  • {name} Latent: {strength:.3f} - Applied: {applied}")
            print(f"\nActive Elements:")
            if prompt:
                print(f"  • Text prompt: {len(prompt)} chars")
            for idx, img in enumerate([image1_vl, image2_vl, image3_vl]):
                if img is not None:
                    print(f"  • Image{idx+1}: {img.shape[2]}x{img.shape[1]} (WxH)")
            print(f"\nFinal Conditioning Shape: {conditioning[0][0].shape}")
            print(f"Formatted Prompt Preview:")
            preview = formatted_prompt[:300] + "..." if len(formatted_prompt) > 300 else formatted_prompt
            print(f"  {preview}")
            print("=" * 70)

        # ============================================================
        # SECTION 6: Attach Reference Latents as Metadata
        # ============================================================
        if ref_latents:
            conditioning = node_helpers.conditioning_set_values(
                conditioning, {"reference_latents": ref_latents}, append=True
            )

        # ============================================================
        # SECTION 7: Return All Outputs
        # ============================================================
        # Return 3 outputs: conditioning, latent (standard format), formatted_prompt
        return io.NodeOutput(conditioning, combined_latent, formatted_prompt)


# ---- Extension API glue (so ComfyUI auto-discovers this node) ----
class ArchAi3DQwenEncoderSimpleExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [ArchAi3D_Qwen_Encoder_Simple]

async def comfy_entrypoint():
    return ArchAi3DQwenEncoderSimpleExtension()
