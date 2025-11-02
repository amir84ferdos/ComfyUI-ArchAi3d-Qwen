import node_helpers


class ArchAi3dQwenEncoderSimpleV2:
    """
    Simplified Qwen encoder node that accepts pre-processed latents and VL images directly.
    Supports up to 3 VL images and 3 latents.
    No image resizing or manipulation - gives full control over preprocessing to the user.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."
                }),
                "vae": ("VAE",),          # VAE for encoding reference images to latents
                "vl_image1": ("IMAGE",),  # Pre-processed image 1 for Qwen VL tokenizer
                "vl_image2": ("IMAGE",),  # Pre-processed image 2 for Qwen VL tokenizer
                "vl_image3": ("IMAGE",),  # Pre-processed image 3 for Qwen VL tokenizer
                "ref_image1": ("IMAGE",), # Reference image 1 for VAE encoding to latent
                "ref_image2": ("IMAGE",), # Reference image 2 for VAE encoding to latent
                "ref_image3": ("IMAGE",), # Reference image 3 for VAE encoding to latent
            }
        }

    RETURN_TYPES = ("CONDITIONING", "STRING",)
    RETURN_NAMES = ("conditioning", "formatted_prompt",)
    FUNCTION = "encode"
    CATEGORY = "ArchAi3d/Qwen"

    def encode(self, clip, prompt,
               system_prompt="Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.",
               vae=None, vl_image1=None, vl_image2=None, vl_image3=None,
               ref_image1=None, ref_image2=None, ref_image3=None):
        """
        Encode text prompt with optional VL images and reference images.

        Args:
            clip: CLIP model for tokenization
            prompt: Text prompt string
            system_prompt: System prompt for ChatML template (editable from UI)
            vae: VAE model for encoding reference images to latents
            vl_image1-3: Optional pre-processed image tensors for vision language model
            ref_image1-3: Optional reference images to encode as latents

        Returns:
            Tuple of (conditioning, formatted_prompt)
        """
        # Build llama template dynamically with user-provided system prompt
        llama_template = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{{}}<|im_end|>\n<|im_start|>assistant\n"

        # Collect VL images and build image prompt
        vl_images_list = [vl_image1, vl_image2, vl_image3]
        images_vl = []
        image_prompt = ""

        for i, vl_image in enumerate(vl_images_list):
            if vl_image is not None:
                images_vl.append(vl_image)
                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)

        # Tokenize with image prompt + user prompt
        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        # Build final formatted prompt for debugging output
        final_prompt = llama_template.format(image_prompt + prompt)

        # Collect reference latents by encoding images with VAE
        ref_images_list = [ref_image1, ref_image2, ref_image3]
        ref_latents = []

        if vae is not None:
            for ref_image in ref_images_list:
                if ref_image is not None:
                    # VAE encode the reference image (RGB only, no resizing)
                    latent_samples = vae.encode(ref_image[:, :, :, :3])
                    ref_latents.append(latent_samples)

        # Add reference latents to conditioning if any provided
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(
                conditioning,
                {"reference_latents": ref_latents},
                append=True
            )

        return (conditioning, final_prompt)


NODE_CLASS_MAPPINGS = {
    "ArchAi3dQwenEncoderSimpleV2": ArchAi3dQwenEncoderSimpleV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3dQwenEncoderSimpleV2": "ArchAi3d Qwen Encoder Simple V2",
}
