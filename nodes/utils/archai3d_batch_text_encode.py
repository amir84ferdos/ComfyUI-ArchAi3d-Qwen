# ArchAi3D Batch Text Encode
#
# Encodes multiple prompts (one per line) into a single batched CONDITIONING tensor.
# Enables parallel processing of multiple prompts in a single GPU forward pass.
#
# Author: Amir Ferdos (ArchAi3d)
# Version: 1.0.0
# License: Dual License (Free for personal use, Commercial license required for business use)

import torch


class ArchAi3D_Batch_Text_Encode:
    """
    Encode multiple prompts into a single batched conditioning tensor.

    Each prompt is encoded separately, then concatenated along the batch dimension.
    This allows different prompts for each image in a batch, processed in a single
    GPU forward pass.

    Works with: SD1.5, SD2.x, Flux, and other non-SDXL models.
    """

    CATEGORY = "ArchAi3d/Conditioning"
    RETURN_TYPES = ("CONDITIONING", "INT")
    RETURN_NAMES = ("conditioning", "batch_size")
    FUNCTION = "encode_batch"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {
                    "tooltip": "CLIP model for text encoding"
                }),
                "prompts": ("STRING", {
                    "multiline": True,
                    "default": "a beautiful landscape\na city at night\na forest in autumn\na beach at sunset",
                    "tooltip": "Multiple prompts - one per line, or comma-separated"
                }),
            },
            "optional": {
                "separator": (["newline", "comma"], {
                    "default": "newline",
                    "tooltip": "How prompts are separated: newline (one per line) or comma"
                }),
            }
        }

    def encode_batch(self, clip, prompts, separator="newline"):
        """Encode multiple prompts into batched conditioning."""

        # Parse prompts based on separator
        if separator == "comma":
            prompt_list = [p.strip() for p in prompts.split(',') if p.strip()]
        else:
            prompt_list = [p.strip() for p in prompts.strip().split('\n') if p.strip()]

        if not prompt_list:
            raise ValueError("No prompts provided. Enter at least one prompt.")

        batch_size = len(prompt_list)
        print(f"[Batch Text Encode] Encoding {batch_size} prompts...")

        cond_list = []
        pooled_list = []

        for i, text in enumerate(prompt_list):
            tokens = clip.tokenize(text)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            cond_list.append(cond)
            if pooled is not None:
                pooled_list.append(pooled)

        # Handle variable-length sequences (Z-Image/Qwen uses variable tokenization)
        # Find max sequence length and pad all tensors to match
        max_seq_len = max(c.shape[1] for c in cond_list)
        padded_cond_list = []
        for cond in cond_list:
            if cond.shape[1] < max_seq_len:
                # Pad with zeros to match max length
                pad_size = max_seq_len - cond.shape[1]
                padding = torch.zeros(cond.shape[0], pad_size, cond.shape[2],
                                     dtype=cond.dtype, device=cond.device)
                cond = torch.cat([cond, padding], dim=1)
            padded_cond_list.append(cond)

        # Concatenate along batch dimension (dim=0)
        # Shape: [batch_size, seq_len, hidden_dim] e.g. [4, 77, 768]
        batched_cond = torch.cat(padded_cond_list, dim=0)

        result_dict = {}
        if pooled_list:
            batched_pooled = torch.cat(pooled_list, dim=0)
            result_dict["pooled_output"] = batched_pooled

        conditioning = [[batched_cond, result_dict]]

        print(f"[Batch Text Encode] Output shape: {batched_cond.shape} (padded to {max_seq_len} tokens)")

        return (conditioning, batch_size)


class ArchAi3D_Batch_Text_Encode_SDXL:
    """
    SDXL version - handles both CLIP G and CLIP L encoders with resolution parameters.

    Includes width, height, crop, and target size parameters required by SDXL.
    """

    CATEGORY = "ArchAi3d/Conditioning"
    RETURN_TYPES = ("CONDITIONING", "INT")
    RETURN_NAMES = ("conditioning", "batch_size")
    FUNCTION = "encode_batch"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {
                    "tooltip": "SDXL CLIP model"
                }),
                "prompts": ("STRING", {
                    "multiline": True,
                    "default": "a beautiful landscape\na city at night\na forest in autumn\na beach at sunset",
                    "tooltip": "Multiple prompts - one per line, or comma-separated"
                }),
                "width": ("INT", {
                    "default": 1024,
                    "min": 0,
                    "max": 8192,
                    "tooltip": "Image width for SDXL conditioning"
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 0,
                    "max": 8192,
                    "tooltip": "Image height for SDXL conditioning"
                }),
            },
            "optional": {
                "separator": (["newline", "comma"], {
                    "default": "newline",
                    "tooltip": "How prompts are separated"
                }),
                "crop_w": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 8192,
                    "tooltip": "Crop width offset"
                }),
                "crop_h": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 8192,
                    "tooltip": "Crop height offset"
                }),
                "target_width": ("INT", {
                    "default": 1024,
                    "min": 0,
                    "max": 8192,
                    "tooltip": "Target width"
                }),
                "target_height": ("INT", {
                    "default": 1024,
                    "min": 0,
                    "max": 8192,
                    "tooltip": "Target height"
                }),
            }
        }

    def encode_batch(self, clip, prompts, width, height, separator="newline",
                     crop_w=0, crop_h=0, target_width=1024, target_height=1024):
        """Encode multiple prompts into batched SDXL conditioning."""

        # Parse prompts based on separator
        if separator == "comma":
            prompt_list = [p.strip() for p in prompts.split(',') if p.strip()]
        else:
            prompt_list = [p.strip() for p in prompts.strip().split('\n') if p.strip()]

        if not prompt_list:
            raise ValueError("No prompts provided. Enter at least one prompt.")

        batch_size = len(prompt_list)
        print(f"[Batch Text Encode SDXL] Encoding {batch_size} prompts...")

        cond_list = []
        pooled_list = []

        for text in prompt_list:
            tokens = clip.tokenize(text)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            cond_list.append(cond)
            if pooled is not None:
                pooled_list.append(pooled)

        # Handle variable-length sequences (Z-Image/Qwen uses variable tokenization)
        # Find max sequence length and pad all tensors to match
        max_seq_len = max(c.shape[1] for c in cond_list)
        padded_cond_list = []
        for cond in cond_list:
            if cond.shape[1] < max_seq_len:
                # Pad with zeros to match max length
                pad_size = max_seq_len - cond.shape[1]
                padding = torch.zeros(cond.shape[0], pad_size, cond.shape[2],
                                     dtype=cond.dtype, device=cond.device)
                cond = torch.cat([cond, padding], dim=1)
            padded_cond_list.append(cond)

        # Concatenate along batch dimension
        batched_cond = torch.cat(padded_cond_list, dim=0)

        result_dict = {
            "width": width,
            "height": height,
            "crop_w": crop_w,
            "crop_h": crop_h,
            "target_width": target_width,
            "target_height": target_height,
        }

        if pooled_list:
            batched_pooled = torch.cat(pooled_list, dim=0)
            result_dict["pooled_output"] = batched_pooled

        conditioning = [[batched_cond, result_dict]]

        print(f"[Batch Text Encode SDXL] Output shape: {batched_cond.shape} (padded to {max_seq_len} tokens)")

        return (conditioning, batch_size)


class ArchAi3D_Empty_Latent_Batch:
    """
    Create empty latent with specified batch size.

    Use this to create latents matching the batch size from Batch Text Encode.
    """

    CATEGORY = "ArchAi3d/Latent"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_size": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 64,
                    "tooltip": "Number of latents to create (connect from Batch Text Encode)"
                }),
                "width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 8192,
                    "step": 8,
                    "tooltip": "Image width"
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 8192,
                    "step": 8,
                    "tooltip": "Image height"
                }),
            }
        }

    def generate(self, batch_size, width, height):
        """Generate empty latent batch."""
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        print(f"[Empty Latent Batch] Created latent shape: {latent.shape}")
        return ({"samples": latent},)


# ============================================================================
# NODE REGISTRATION
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "ArchAi3D_Batch_Text_Encode": ArchAi3D_Batch_Text_Encode,
    "ArchAi3D_Batch_Text_Encode_SDXL": ArchAi3D_Batch_Text_Encode_SDXL,
    "ArchAi3D_Empty_Latent_Batch": ArchAi3D_Empty_Latent_Batch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_Batch_Text_Encode": "📦 Batch Text Encode",
    "ArchAi3D_Batch_Text_Encode_SDXL": "📦 Batch Text Encode (SDXL)",
    "ArchAi3D_Empty_Latent_Batch": "📦 Empty Latent (Batch)",
}
