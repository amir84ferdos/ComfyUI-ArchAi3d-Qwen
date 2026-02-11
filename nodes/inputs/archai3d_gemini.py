"""
ArchAi3D Gemini API Node
A Gemini API node with name fields for web interface integration.
Supports text and image inputs with model override capability.

Updated to use the new google-genai SDK (GA 2025).
The old google-generativeai SDK is deprecated and will stop receiving updates.

Features smart caching to avoid redundant API calls when inputs haven't changed.
"""

import PIL.Image
import numpy as np
import time
import os
import torch
import json
import hashlib
import gc

# Try to import the new SDK first, fall back to old if not available
GEMINI_AVAILABLE = True
try:
    from google import genai
    from google.genai import types
    from google.genai import errors
    USE_NEW_SDK = True
except ImportError:
    try:
        import google.generativeai as genai_old
        USE_NEW_SDK = False
    except ImportError:
        GEMINI_AVAILABLE = False
        USE_NEW_SDK = None
        print("[ArchAi3D Gemini] WARNING: google-genai not installed. "
              "Install with: pip install google-genai")


# Available Gemini Models (Updated February 2026)
GEMINI_MODELS = [
    # Gemini 3 (Preview)
    "gemini-3-pro-preview",
    "gemini-3-pro-image-preview",
    "gemini-3-flash-preview",
    # Gemini 2.5 (Stable GA)
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash-image",
    "gemini-2.5-pro",
    # Gemini 2.0 (Retiring March 31, 2026)
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]


class ArchAi3D_Gemini:
    """
    Gemini API node with name fields for web interface integration.

    Features:
    - name field for prompt (web interface can identify inputs)
    - model_override STRING input for connecting Gemini Model Selector
    - Supports text and image inputs (up to 4 images)
    - API key stored in config.json
    - Uses new google-genai SDK (with fallback to legacy)
    - Proper error handling and retry logic
    """

    # Class-level cache (shared across all instances)
    _cache = {}
    _cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gemini_cache.json')
    _max_cache_entries = 100  # Limit cache size

    def __init__(self):
        self.api_key = None
        self.client = None
        # Our config path
        self.config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gemini_config.json')
        # Original Gemini Pro node config path (for compatibility)
        self.gemini_pro_config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', '..', '..', 'ComfyUI_Gemini_Pro', 'config.json'
        )
        self.load_config()
        self._load_cache()

    def load_config(self):
        """Load API key from config file. Checks multiple locations for compatibility."""
        # Priority: 1. Our own config, 2. Original Gemini Pro config
        config_locations = [
            (self.config_path, "ArchAi3D config"),
            (self.gemini_pro_config_path, "Gemini Pro config"),
        ]

        for config_path, source in config_locations:
            try:
                if os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                        api_key = config.get('api_key', '')
                        if api_key:
                            self.api_key = api_key
                            self._configure_client(self.api_key)
                            print(f"[ArchAi3D Gemini] Loaded API key from {source}")
                            return
            except Exception as e:
                print(f"[ArchAi3D Gemini] Failed to load {source}: {str(e)}")

    def _configure_client(self, api_key):
        """Configure the API client with the given key."""
        if USE_NEW_SDK:
            self.client = genai.Client(api_key=api_key)
        else:
            genai_old.configure(api_key=api_key)
            self.client = None

    def save_config(self, api_key):
        """Save API key to config file."""
        try:
            config = {'api_key': api_key}
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)
            print("[ArchAi3D Gemini] Saved API key to config")
        except Exception as e:
            print(f"[ArchAi3D Gemini] Failed to save config: {str(e)}")

    # =========================================================================
    # CACHE SYSTEM
    # =========================================================================

    def _load_cache(self):
        """Load cache from file on startup."""
        try:
            if os.path.exists(self._cache_file):
                with open(self._cache_file, 'r', encoding='utf-8') as f:
                    ArchAi3D_Gemini._cache = json.load(f)
                print(f"[ArchAi3D Gemini] Loaded {len(self._cache)} cached responses")
        except Exception as e:
            print(f"[ArchAi3D Gemini] Failed to load cache: {str(e)}")
            ArchAi3D_Gemini._cache = {}

    def _save_cache(self):
        """Save cache to file."""
        try:
            # Limit cache size - remove oldest entries if too large
            if len(self._cache) > self._max_cache_entries:
                # Sort by timestamp and keep newest entries
                sorted_keys = sorted(
                    self._cache.keys(),
                    key=lambda k: self._cache[k].get('timestamp', 0),
                    reverse=True
                )
                ArchAi3D_Gemini._cache = {
                    k: self._cache[k] for k in sorted_keys[:self._max_cache_entries]
                }

            with open(self._cache_file, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            print(f"[ArchAi3D Gemini] Failed to save cache: {str(e)}")

    def _compute_image_hash(self, img):
        """Compute a hash for an image tensor."""
        if img is None:
            return "none"

        with torch.no_grad():
            if isinstance(img, torch.Tensor):
                # Sample pixels for faster hashing (don't hash entire image)
                img_cpu = img.detach().cpu()
                if len(img_cpu.shape) == 4:
                    img_cpu = img_cpu[0]

                # Get image dimensions and sample data
                h, w = img_cpu.shape[0], img_cpu.shape[1]
                # Sample corners and center for hash
                samples = [
                    img_cpu[0, 0].mean().item(),
                    img_cpu[0, -1].mean().item(),
                    img_cpu[-1, 0].mean().item(),
                    img_cpu[-1, -1].mean().item(),
                    img_cpu[h//2, w//2].mean().item(),
                ]
                # Include shape in hash
                hash_data = f"{img_cpu.shape}_{samples}"
                del img_cpu
            else:
                # Numpy array
                hash_data = f"{img.shape}_{img.mean()}_{img.std()}"

        return hashlib.md5(hash_data.encode()).hexdigest()[:16]

    def _compute_cache_key(self, prompt, model, system_prompt, image1, image2, image3, image4,
                           temperature, top_p, top_k, max_tokens):
        """Compute a unique cache key based on all inputs."""
        # Hash all text inputs
        text_hash = hashlib.md5(
            f"{prompt}|{model}|{system_prompt}|{temperature}|{top_p}|{top_k}|{max_tokens}".encode()
        ).hexdigest()[:16]

        # Hash images
        img_hashes = [
            self._compute_image_hash(image1),
            self._compute_image_hash(image2),
            self._compute_image_hash(image3),
            self._compute_image_hash(image4),
        ]

        return f"{text_hash}_{'-'.join(img_hashes)}"

    def _get_cached_response(self, cache_key):
        """Get cached response if available."""
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            print(f"[ArchAi3D Gemini] ✅ CACHE HIT - Using cached response")
            return entry.get('response')
        return None

    def _set_cached_response(self, cache_key, response):
        """Cache a response."""
        self._cache[cache_key] = {
            'response': response,
            'timestamp': time.time()
        }
        self._save_cache()
        print(f"[ArchAi3D Gemini] 💾 Response cached")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {
                    "default": "gemini_prompt",
                    "multiline": False,
                    "tooltip": "Identifier name for this input (used by web interface)"
                }),
                "prompt": ("STRING", {
                    "default": "Describe this image in detail.",
                    "multiline": True,
                    "tooltip": "The prompt to send to Gemini"
                }),
                "model": (GEMINI_MODELS, {
                    "default": "gemini-2.5-flash",
                    "tooltip": "Select Gemini model (can be overridden by model_override)"
                }),
            },
            "optional": {
                "model_override": ("STRING", {
                    "default": "",
                    "tooltip": "Connect ArchAi3D Gemini Model node here to override model selection"
                }),
                "system_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "System instructions to guide model behavior"
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "API key (saved to config after first use)"
                }),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "temperature": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Temperature (0=deterministic, 1=balanced, 2=creative)"
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Top-p nucleus sampling (0.95 recommended)"
                }),
                "top_k": ("INT", {
                    "default": 40,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Top-k token selection"
                }),
                "max_tokens": ("INT", {
                    "default": 8192,
                    "min": 1,
                    "max": 65536,
                    "tooltip": "Maximum output tokens"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffff,
                    "tooltip": "Random seed for reproducibility (0 = random)"
                }),
                "thinking_budget": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 24576,
                    "step": 1024,
                    "tooltip": "Thinking token budget for Gemini 2.5/3 models (0=off, 1024-24576=on). Model uses these tokens to reason before responding."
                }),
                "use_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use cached response if inputs haven't changed (saves API calls)"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "execute"
    CATEGORY = "ArchAi3d/Inputs"

    def _convert_image(self, img):
        """Convert ComfyUI image tensor to PIL Image. Memory optimized."""
        if isinstance(img, torch.Tensor):
            # Move to CPU and detach from computation graph
            with torch.no_grad():
                img_cpu = img.detach().cpu()

                # Handle batch dimension
                if len(img_cpu.shape) == 4:
                    img_cpu = img_cpu[0]  # Take first image from batch

                # Handle channel-first format (C, H, W)
                if len(img_cpu.shape) == 3 and img_cpu.shape[0] in [1, 3, 4]:
                    if img_cpu.shape[0] != img_cpu.shape[-1]:  # Not already HWC
                        img_cpu = img_cpu.permute(1, 2, 0)

                # Convert to uint8 numpy array
                img_np = (img_cpu * 255).round().clamp(0, 255).to(torch.uint8).numpy()

                # Explicitly delete tensor to free memory
                del img_cpu
        elif isinstance(img, np.ndarray):
            img_np = img.copy() if img.dtype != np.uint8 else img
            if img_np.dtype != np.uint8:
                img_np = (img_np * 255).astype(np.uint8)
        else:
            raise ValueError("Image must be tensor or numpy array")

        # Handle grayscale
        if len(img_np.shape) == 2:
            img_np = np.stack([img_np, img_np, img_np], axis=-1)
        # Handle RGBA - keep only RGB
        elif len(img_np.shape) == 3 and img_np.shape[-1] == 4:
            img_np = img_np[..., :3].copy()  # Copy to release original memory

        pil_img = PIL.Image.fromarray(img_np)

        # Free numpy array
        del img_np

        return pil_img

    def _generate_with_new_sdk(self, model, contents, config):
        """Generate content using the new google-genai SDK."""
        response = self.client.models.generate_content(
            model=model,
            contents=contents,
            config=config
        )
        return response.text

    def _generate_with_old_sdk(self, model, contents, system_prompt, temperature,
                                top_p, top_k, max_tokens):
        """Generate content using the legacy google-generativeai SDK."""
        model_instance = genai_old.GenerativeModel(
            model,
            system_instruction=system_prompt if system_prompt else None
        )
        response = model_instance.generate_content(
            contents,
            generation_config=genai_old.types.GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_output_tokens=max_tokens,
                candidate_count=1
            )
        )
        # Extract text from response
        if hasattr(response, 'text'):
            return response.text
        # Fallback extraction
        if hasattr(response, 'candidates'):
            for cand in response.candidates:
                if hasattr(cand, 'content') and hasattr(cand.content, 'parts'):
                    for part in cand.content.parts:
                        if hasattr(part, 'text'):
                            return part.text
        return None

    def execute(self, name, prompt, model, model_override="", system_prompt="",
                api_key="", image1=None, image2=None, image3=None, image4=None,
                temperature=1.0, top_p=0.95, top_k=40, max_tokens=8192, seed=0,
                thinking_budget=0, use_cache=True):
        try:
            # Handle API Key
            if api_key and api_key.strip():
                if api_key != self.api_key:
                    self.api_key = api_key
                    self._configure_client(api_key)
                    self.save_config(api_key)
            elif not self.api_key:
                return ("Error: Please provide an API key. Get one at https://aistudio.google.com/apikey",)

            # Handle model override
            if model_override and model_override.strip():
                model = model_override.strip()
                print(f"[ArchAi3D Gemini] Using override model: {model}")

            # Check cache first (before processing images to save time)
            if use_cache:
                cache_key = self._compute_cache_key(
                    prompt, model, system_prompt,
                    image1, image2, image3, image4,
                    temperature, top_p, top_k, max_tokens
                )
                cached_response = self._get_cached_response(cache_key)
                if cached_response is not None:
                    return (cached_response,)

            # Process images - only convert what we need
            images = [img for img in [image1, image2, image3, image4] if img is not None]
            converted_images = []

            for idx, img in enumerate(images, start=1):
                try:
                    pil_img = self._convert_image(img)
                    converted_images.append(pil_img)
                    print(f"[ArchAi3D Gemini] Image {idx}: {pil_img.size[0]}x{pil_img.size[1]}")
                except Exception as e:
                    # Clean up any already converted images
                    for pil in converted_images:
                        pil.close()
                    return (f"Error: Failed to process image {idx} - {str(e)}",)

            # Build content list - images first, then text (better for vision models)
            contents = []
            contents.extend(converted_images)
            contents.append(prompt)

            print(f"[ArchAi3D Gemini] Model: {model}")
            print(f"[ArchAi3D Gemini] Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
            if system_prompt:
                print(f"[ArchAi3D Gemini] System: {system_prompt[:50]}{'...' if len(system_prompt) > 50 else ''}")

            # Retry logic with exponential backoff
            max_retries = 3
            last_error = None

            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        delay = 2 * (2 ** attempt)  # 4s, 8s
                        print(f"[ArchAi3D Gemini] Retry {attempt}/{max_retries-1} in {delay}s...")
                        time.sleep(delay)

                    if USE_NEW_SDK:
                        # Build generation config for new SDK
                        config_params = {
                            'temperature': temperature,
                            'top_p': top_p,
                            'top_k': top_k,
                            'max_output_tokens': max_tokens,
                            'candidate_count': 1,
                        }

                        # Add seed if specified (for reproducibility)
                        if seed > 0:
                            config_params['seed'] = seed

                        # Add thinking config for Gemini 2.5/3 models
                        if thinking_budget > 0:
                            config_params['thinking_config'] = types.ThinkingConfig(
                                thinking_budget=thinking_budget
                            )
                            print(f"[ArchAi3D Gemini] Thinking enabled: {thinking_budget} token budget")

                        # Add system instruction if provided
                        if system_prompt and system_prompt.strip():
                            config_params['system_instruction'] = system_prompt.strip()

                        config = types.GenerateContentConfig(**config_params)
                        result = self._generate_with_new_sdk(model, contents, config)
                    else:
                        result = self._generate_with_old_sdk(
                            model, contents, system_prompt,
                            temperature, top_p, top_k, max_tokens
                        )

                    if result:
                        print(f"[ArchAi3D Gemini] Response: {len(result)} chars")
                        # Clean up PIL images to free memory
                        for pil_img in converted_images:
                            pil_img.close()
                        del converted_images

                        # Cache the result if caching is enabled
                        if use_cache:
                            self._set_cached_response(cache_key, result)

                        # Aggressive memory cleanup for low VRAM systems
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()

                        return (result,)
                    else:
                        # Clean up PIL images
                        for pil_img in converted_images:
                            pil_img.close()
                        # Memory cleanup
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        return ("Error: Empty response from API",)

                except Exception as e:
                    error_str = str(e)
                    last_error = e

                    # Check for retryable errors
                    retryable = any(code in error_str for code in ['429', '503', '500', 'RESOURCE_EXHAUSTED'])

                    if retryable and attempt < max_retries - 1:
                        print(f"[ArchAi3D Gemini] Retryable error: {error_str[:100]}")
                        continue

                    # Handle specific error types
                    if USE_NEW_SDK:
                        try:
                            if isinstance(e, errors.APIError):
                                return (f"API Error ({e.code}): {e.message}",)
                        except:
                            pass

                    raise

            # Clean up PIL images before returning error
            for pil_img in converted_images:
                pil_img.close()
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return (f"Error after {max_retries} retries: {str(last_error)}",)

        except Exception as e:
            error_msg = str(e)
            print(f"[ArchAi3D Gemini] Error: {error_msg}")
            # Clean up any converted images on error
            if 'converted_images' in locals():
                for pil_img in converted_images:
                    try:
                        pil_img.close()
                    except:
                        pass

            # Aggressive memory cleanup for low VRAM systems
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            print("[ArchAi3D Gemini] Memory cleaned up")

            # Provide helpful error messages
            if "API_KEY" in error_msg.upper() or "authentication" in error_msg.lower():
                return ("Error: Invalid API key. Get a valid key at https://aistudio.google.com/apikey",)
            elif "quota" in error_msg.lower() or "429" in error_msg:
                return ("Error: Rate limit exceeded. Please wait and try again.",)
            elif "not found" in error_msg.lower() or "404" in error_msg:
                return (f"Error: Model '{model}' not found. Try 'gemini-2.5-flash' or 'gemini-2.0-flash'.",)

            return (f"Error: {error_msg}",)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Always re-execute to get fresh API responses."""
        return float("nan")
