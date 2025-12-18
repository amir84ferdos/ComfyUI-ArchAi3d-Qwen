"""
USDU Edge Repair - ControlNet Per-Tile Support
===============================================

ControlNet patches for per-tile application in USDU.
These patches wrap the control model and apply it to cropped tile regions.
"""

import torch
import comfy.utils
import comfy.model_management
import comfy.latent_formats


class ControlNetPatchBase:
    """
    Base class for ControlNet patches.

    Provides common functionality for model management and device movement.
    Subclasses implement encode_latent_cond() and __call__() for specific models.
    """

    def __init__(self, model_patch, vae, image, strength, mask=None):
        self.model_patch = model_patch
        self.vae = vae
        self.image = image
        self.strength = strength
        self.mask = mask
        self.encoded_image = None
        self.encoded_image_size = None

    def to(self, device_or_dtype):
        """Move encoded image to device."""
        if isinstance(device_or_dtype, torch.device):
            self.encoded_image = self.encoded_image.to(device_or_dtype)
        return self

    def models(self):
        """Return list of models used by this patch."""
        return [self.model_patch]

    def needs_reencode(self, x):
        """Check if image needs re-encoding due to size mismatch."""
        sc = self.vae.spacial_compression_encode()
        target = (x.shape[-2] * sc, x.shape[-1] * sc)
        return self.encoded_image is None or self.encoded_image_size != target

    def scale_image(self, x):
        """Scale control image to match latent size."""
        sc = self.vae.spacial_compression_encode()
        return comfy.utils.common_upscale(
            self.image.movedim(-1, 1),
            x.shape[-1] * sc, x.shape[-2] * sc,
            "area", "center"
        )


class DiffSynthCnetPatch(ControlNetPatchBase):
    """
    ControlNet patch for DiffSynth/Qwen models (QwenImageBlockWiseControlNet).

    Encodes the control image to latent space and applies control signal
    at each diffusion block.
    """

    def __init__(self, model_patch, vae, image, strength, mask=None):
        super().__init__(model_patch, vae, image, strength, mask)
        self.encoded_image = model_patch.model.process_input_latent_image(
            self.encode_latent_cond(image)
        )
        self.encoded_image_size = (image.shape[1], image.shape[2])

    def encode_latent_cond(self, image):
        """Encode control image to latent space."""
        latent_image = self.vae.encode(image)
        if self.model_patch.model.additional_in_dim > 0:
            if self.mask is None:
                mask_ = torch.ones_like(latent_image)[:, :self.model_patch.model.additional_in_dim // 4]
            else:
                mask_ = comfy.utils.common_upscale(
                    self.mask.mean(dim=1, keepdim=True),
                    latent_image.shape[-1], latent_image.shape[-2],
                    "bilinear", "none"
                )
            return torch.cat([latent_image, mask_], dim=1)
        return latent_image

    def __call__(self, kwargs):
        """Apply control at a diffusion block."""
        x = kwargs.get("x")
        img = kwargs.get("img")
        block_index = kwargs.get("block_index")

        # Re-encode if size mismatch
        if self.needs_reencode(x):
            image_scaled = self.scale_image(x)
            loaded_models = comfy.model_management.loaded_models(only_currently_used=True)
            self.encoded_image = self.model_patch.model.process_input_latent_image(
                self.encode_latent_cond(image_scaled.movedim(1, -1))
            )
            self.encoded_image_size = (image_scaled.shape[-2], image_scaled.shape[-1])
            comfy.model_management.load_models_gpu(loaded_models)

        # Apply control
        img[:, :self.encoded_image.shape[1]] += (
            self.model_patch.model.control_block(
                img[:, :self.encoded_image.shape[1]],
                self.encoded_image.to(img.dtype),
                block_index
            ) * self.strength
        )
        kwargs['img'] = img
        return kwargs


class ZImageControlPatch(ControlNetPatchBase):
    """
    ControlNet patch for Z-Image ControlNet (ZImage_Control).

    Uses Flux latent format and processes control through multiple layers.
    """

    def __init__(self, model_patch, vae, image, strength):
        super().__init__(model_patch, vae, image, strength)
        self.encoded_image = self.encode_latent_cond(image)
        self.encoded_image_size = (image.shape[1], image.shape[2])
        self.temp_data = None

    def encode_latent_cond(self, image):
        """Encode control image to Flux latent space."""
        return comfy.latent_formats.Flux().process_in(self.vae.encode(image))

    def to(self, device_or_dtype):
        """Move encoded image to device and reset temp data."""
        super().to(device_or_dtype)
        self.temp_data = None
        return self

    def __call__(self, kwargs):
        """Apply Z-Image control at a diffusion block."""
        x = kwargs.get("x")
        img = kwargs.get("img")
        txt = kwargs.get("txt")
        pe = kwargs.get("pe")
        vec = kwargs.get("vec")
        block_index = kwargs.get("block_index")

        # Re-encode if size mismatch
        if self.needs_reencode(x):
            image_scaled = self.scale_image(x)
            loaded_models = comfy.model_management.loaded_models(only_currently_used=True)
            self.encoded_image = self.encode_latent_cond(image_scaled.movedim(1, -1))
            self.encoded_image_size = (image_scaled.shape[-2], image_scaled.shape[-1])
            comfy.model_management.load_models_gpu(loaded_models)

        cnet_index = (block_index // 5)
        cnet_index_float = (block_index / 5)

        kwargs.pop("img", None)
        kwargs.pop("txt", None)

        cnet_blocks = self.model_patch.model.n_control_layers
        if cnet_index_float > (cnet_blocks - 1):
            self.temp_data = None
            return kwargs

        if self.temp_data is None or self.temp_data[0] > cnet_index:
            self.temp_data = (-1, (None, self.model_patch.model(txt, self.encoded_image.to(img.dtype), pe, vec)))

        while self.temp_data[0] < cnet_index and (self.temp_data[0] + 1) < cnet_blocks:
            next_layer = self.temp_data[0] + 1
            self.temp_data = (next_layer, self.model_patch.model.forward_control_block(
                next_layer, self.temp_data[1][1],
                img[:, :self.temp_data[1][1].shape[1]], None, pe, vec
            ))

        if cnet_index_float == self.temp_data[0]:
            img[:, :self.temp_data[1][0].shape[1]] += (self.temp_data[1][0] * self.strength)
            if cnet_blocks == self.temp_data[0] + 1:
                self.temp_data = None

        return kwargs


def is_zimage_control(model_patch):
    """
    Check if model_patch is a Z-Image ControlNet.

    Args:
        model_patch: MODEL_PATCH to check

    Returns:
        bool: True if Z-Image ControlNet, False otherwise
    """
    try:
        import comfy.ldm.lumina.controlnet
        return isinstance(model_patch.model, comfy.ldm.lumina.controlnet.ZImage_Control)
    except (ImportError, AttributeError):
        return hasattr(model_patch.model, 'n_control_layers')
