"""
ArchAi3D DiffSynth ControlNet - Fixed version of QwenImageDiffsynthControlnet

This is a fixed copy of ComfyUI's QwenImageDiffsynthControlnet node.
The original has a bug where mask dimensions are handled incorrectly:
- Two consecutive `if` statements cause double unsqueeze
- This results in 5D tensor when 4D is expected

Bug location in original: comfy_extras/nodes_model_patch.py lines 382-385
Fix: Changed second `if mask.ndim == 4` to `elif mask.ndim == 4`
"""

import torch
import comfy.utils
import comfy.model_management
import comfy.latent_formats
import comfy.ldm.lumina.controlnet


class DiffSynthCnetPatch:
    """ControlNet patch for DiffSynth/Qwen models (QwenImageBlockWiseControlNet)."""

    def __init__(self, model_patch, vae, image, strength, mask=None):
        self.model_patch = model_patch
        self.vae = vae
        self.image = image
        self.strength = strength
        self.mask = mask
        self.encoded_image = model_patch.model.process_input_latent_image(self.encode_latent_cond(image))
        self.encoded_image_size = (image.shape[1], image.shape[2])

    def encode_latent_cond(self, image):
        latent_image = self.vae.encode(image)
        if self.model_patch.model.additional_in_dim > 0:
            if self.mask is None:
                mask_ = torch.ones_like(latent_image)[:, :self.model_patch.model.additional_in_dim // 4]
            else:
                mask_ = comfy.utils.common_upscale(self.mask.mean(dim=1, keepdim=True), latent_image.shape[-1], latent_image.shape[-2], "bilinear", "none")

            return torch.cat([latent_image, mask_], dim=1)
        else:
            return latent_image

    def __call__(self, kwargs):
        x = kwargs.get("x")
        img = kwargs.get("img")
        block_index = kwargs.get("block_index")
        spacial_compression = self.vae.spacial_compression_encode()
        if self.encoded_image is None or self.encoded_image_size != (x.shape[-2] * spacial_compression, x.shape[-1] * spacial_compression):
            image_scaled = comfy.utils.common_upscale(self.image.movedim(-1, 1), x.shape[-1] * spacial_compression, x.shape[-2] * spacial_compression, "area", "center")
            loaded_models = comfy.model_management.loaded_models(only_currently_used=True)
            self.encoded_image = self.model_patch.model.process_input_latent_image(self.encode_latent_cond(image_scaled.movedim(1, -1)))
            self.encoded_image_size = (image_scaled.shape[-2], image_scaled.shape[-1])
            comfy.model_management.load_models_gpu(loaded_models)

        img[:, :self.encoded_image.shape[1]] += (self.model_patch.model.control_block(img[:, :self.encoded_image.shape[1]], self.encoded_image.to(img.dtype), block_index) * self.strength)
        kwargs['img'] = img
        return kwargs

    def to(self, device_or_dtype):
        if isinstance(device_or_dtype, torch.device):
            self.encoded_image = self.encoded_image.to(device_or_dtype)
        return self

    def models(self):
        return [self.model_patch]


class ZImageControlPatch:
    """ControlNet patch for Z-Image ControlNet (ZImage_Control)."""

    def __init__(self, model_patch, vae, image, strength):
        self.model_patch = model_patch
        self.vae = vae
        self.image = image
        self.strength = strength
        self.encoded_image = self.encode_latent_cond(image)
        self.encoded_image_size = (image.shape[1], image.shape[2])
        self.temp_data = None

    def encode_latent_cond(self, image):
        latent_image = comfy.latent_formats.Flux().process_in(self.vae.encode(image))
        return latent_image

    def __call__(self, kwargs):
        x = kwargs.get("x")
        img = kwargs.get("img")
        txt = kwargs.get("txt")
        pe = kwargs.get("pe")
        vec = kwargs.get("vec")
        block_index = kwargs.get("block_index")
        spacial_compression = self.vae.spacial_compression_encode()
        if self.encoded_image is None or self.encoded_image_size != (x.shape[-2] * spacial_compression, x.shape[-1] * spacial_compression):
            image_scaled = comfy.utils.common_upscale(self.image.movedim(-1, 1), x.shape[-1] * spacial_compression, x.shape[-2] * spacial_compression, "area", "center")
            loaded_models = comfy.model_management.loaded_models(only_currently_used=True)
            self.encoded_image = self.encode_latent_cond(image_scaled.movedim(1, -1))
            self.encoded_image_size = (image_scaled.shape[-2], image_scaled.shape[-1])
            comfy.model_management.load_models_gpu(loaded_models)

        cnet_index = (block_index // 5)
        cnet_index_float = (block_index / 5)

        kwargs.pop("img", None)  # we do ops in place
        kwargs.pop("txt", None)

        cnet_blocks = self.model_patch.model.n_control_layers
        if cnet_index_float > (cnet_blocks - 1):
            self.temp_data = None
            return kwargs

        if self.temp_data is None or self.temp_data[0] > cnet_index:
            self.temp_data = (-1, (None, self.model_patch.model(txt, self.encoded_image.to(img.dtype), pe, vec)))

        while self.temp_data[0] < cnet_index and (self.temp_data[0] + 1) < cnet_blocks:
            next_layer = self.temp_data[0] + 1
            self.temp_data = (next_layer, self.model_patch.model.forward_control_block(next_layer, self.temp_data[1][1], img[:, :self.temp_data[1][1].shape[1]], None, pe, vec))

        if cnet_index_float == self.temp_data[0]:
            img[:, :self.temp_data[1][0].shape[1]] += (self.temp_data[1][0] * self.strength)
            if cnet_blocks == self.temp_data[0] + 1:
                self.temp_data = None

        return kwargs

    def to(self, device_or_dtype):
        if isinstance(device_or_dtype, torch.device):
            self.encoded_image = self.encoded_image.to(device_or_dtype)
            self.temp_data = None
        return self

    def models(self):
        return [self.model_patch]


class ArchAi3D_DiffSynth_ControlNet:
    """
    Fixed version of QwenImageDiffsynthControlnet with corrected mask handling.

    BUG FIX: Original had consecutive if statements that caused double unsqueeze:
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)  # [B,H,W] → [B,1,H,W]
        if mask.ndim == 4:            # BUG: This is now TRUE after first if!
            mask = mask.unsqueeze(2)  # [B,1,H,W] → [B,1,1,H,W] (5D!)

    Fixed by changing second `if` to `elif`.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "model_patch": ("MODEL_PATCH",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_controlnet"
    CATEGORY = "ArchAi3d/ControlNet"

    def apply_controlnet(self, model, model_patch, vae, image, strength, mask=None):
        model_patched = model.clone()
        image = image[:, :, :, :3]

        if mask is not None:
            # === BUG FIX: Changed second `if` to `elif` ===
            # Original bug: Two consecutive `if` statements caused double unsqueeze
            # Input [B,H,W] (3D) → [B,1,H,W] (4D) → [B,1,1,H,W] (5D) - WRONG!
            # Fixed: Only one unsqueeze happens, resulting in correct 4D tensor
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)  # [B, H, W] → [B, 1, H, W]
            elif mask.ndim == 4:
                mask = mask.unsqueeze(2)  # [B, C, H, W] → [B, C, 1, H, W] (rare case)
            mask = 1.0 - mask

        if isinstance(model_patch.model, comfy.ldm.lumina.controlnet.ZImage_Control):
            model_patched.set_model_double_block_patch(ZImageControlPatch(model_patch, vae, image, strength))
        else:
            model_patched.set_model_double_block_patch(DiffSynthCnetPatch(model_patch, vae, image, strength, mask))

        return (model_patched,)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "ArchAi3D_DiffSynth_ControlNet": ArchAi3D_DiffSynth_ControlNet,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_DiffSynth_ControlNet": "🔧 DiffSynth ControlNet (Fixed)",
}
