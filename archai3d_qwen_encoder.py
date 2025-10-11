# ArchAi3D Qwen Encoder — Qwen-VL 2.5 encoder with strength controls
#
# OVERVIEW:
# Encodes text prompts and up to 3 images for Qwen-VL 2.5, with separate controls for
# conditioning strength and per-image latent strength. Always uses proper ChatML formatting
# for stable, predictable results. No image size validation (expects pre-scaled inputs).
#
# INPUTS:
# - 3 images for Qwen-VL vision encoder (RGB only, expects correct size)
# - 3 images for VAE reference latents (RGB only, expects correct size)
# - Text prompt (wrapped automatically in ChatML format)
# - Optional system prompt (for ChatML system block)
#
# STRENGTH CONTROLS:
# 1. conditioning_strength (0.0-2.0):
#    - Multiplies final text+vision embeddings (global control)
#    - <1.0 = weaker, 1.0 = normal, >1.0 = stronger
#
# 2. image1/2/3_latent_strength (0.0-2.0):
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
# Node ID: ArchAi3D_Qwen_Encoder
# License: MIT

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override
import node_helpers

class ArchAi3D_Qwen_Encoder(io.ComfyNode):
    """Qwen-VL encoder with vision tokens, reference latents, and strength controls.

    Features:
    - 6 image inputs: 3 for Qwen-VL vision encoder + 3 for VAE reference latents
    - ChatML formatting for proper Qwen-VL 2.5 conditioning (always enabled)
    - Per-image latent strength controls (adjust each reference image independently)
    - Global conditioning strength control (affects text+vision embeddings)
    - Auto-labeling for multi-image inputs (when 2+ images)
    - RGB-only extraction (alpha channel removed automatically)
    - No size validation (expects pre-scaled images from Dual-Scale node)

    Outputs:
    - conditioning: Text+vision embeddings with reference latents attached
    - latent: Image1 latent in standard format (for VAEDecode)
    - formatted_prompt: Final ChatML-formatted prompt with vision tokens
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ArchAi3D_Qwen_Encoder",
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

                # === Conditioning Strength Control ===
                io.Float.Input("conditioning_strength", default=1.0, min=0.0, max=2.0, step=0.01,
                             tooltip="Global strength for text+vision embeddings (1.0=normal, <1.0=weaker, >1.0=stronger)"),

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
                conditioning_strength=1.0,
                image1_latent_strength=1.0,
                image2_latent_strength=1.0,
                image3_latent_strength=1.0,
                debug_mode=False,
                llama_template="",  # llama_template kept for backward compatibility
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
        # SECTION 4: Build ChatML-Formatted Prompt
        # ============================================================
        # Combine vision tokens + text with proper separation
        user_content = image_prompt
        if prompt:
            user_content += "\n" + prompt if image_prompt else prompt

        # Wrap in ChatML format (required for proper Qwen-VL 2.5 conditioning)
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
        # SECTION 5: Tokenize and Encode
        # ============================================================
        # Tokenize prompt + images together
        template = llama_template.strip() or None
        tokens = clip.tokenize(formatted_prompt, images=images_vl, llama_template=template)

        # Encode tokens to embeddings
        if hasattr(clip, "encode_from_tokens_scheduled"):
            conditioning = clip.encode_from_tokens_scheduled(tokens)
        else:
            conditioning = clip.encode_from_tokens(tokens)

        # Apply global conditioning strength
        if abs(conditioning_strength - 1.0) > 0.001:
            for i in range(len(conditioning)):
                conditioning[i][0] = conditioning[i][0] * conditioning_strength

        # ============================================================
        # SECTION 6: Debug Output (if enabled)
        # ============================================================
        if debug_mode:
            print("=" * 70)
            print("ArchAi3D Qwen Encoder - Debug Output")
            print("=" * 70)
            print(f"Conditioning Strength: {conditioning_strength:.3f} ({'applied' if abs(conditioning_strength - 1.0) > 0.001 else 'default'})")

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
                print(f"  • Text: {len(prompt)} chars")
            for idx, img in enumerate([image1_vl, image2_vl, image3_vl]):
                if img is not None:
                    print(f"  • VL Image{idx+1}: {img.shape[2]}×{img.shape[1]} px")

            print(f"\nOutput Shapes:")
            print(f"  • Conditioning: {conditioning[0][0].shape}")
            if combined_latent:
                print(f"  • Latent: {combined_latent['samples'].shape}")

            print(f"\nFormatted Prompt:")
            preview = formatted_prompt[:250] + "..." if len(formatted_prompt) > 250 else formatted_prompt
            print(f"  {preview}")
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
class ArchAi3DQwenEncoderExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [ArchAi3D_Qwen_Encoder]

async def comfy_entrypoint():
    return ArchAi3DQwenEncoderExtension()
