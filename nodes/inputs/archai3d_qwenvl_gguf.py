# ArchAi3D QwenVL GGUF Node
#
# Fast VLM inference using llama.cpp GGUF models via HTTP API
# Connects to llama-server running Qwen3-VL GGUF models
#
# Author: Amir Ferdos (ArchAi3d)
# Email: Amir84ferdos@gmail.com
# LinkedIn: https://www.linkedin.com/in/archai3d/
# GitHub: https://github.com/amir84ferdos
# Version: 1.5.0 - Use bash to run script (no execute permission needed)
# License: Dual License (Free for personal use, Commercial license required for business use)

import base64
import hashlib
import io
import json
import os
import signal
import subprocess
from pathlib import Path

import numpy as np
import requests
from PIL import Image

# Load preset prompts from config.json
CONFIG_PATH = Path(__file__).parent.parent.parent / "config.json"

def load_config():
    """Load configuration from config.json"""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[QwenVL-GGUF] Warning: Could not load config.json: {e}")
        return {}

CONFIG = load_config()
PRESET_PROMPTS = CONFIG.get("_preset_prompts", ["🖼️ Tags", "🖼️ Simple Description", "🖼️ Detailed Description"])
SYSTEM_PROMPTS = CONFIG.get("_system_prompts", {})

# Result cache - stores {input_hash: output_text}
GGUF_RESULT_CACHE = {}

# ============================================================================
# QUALITY PRESETS - Simplified settings for different use cases
# ============================================================================
QUALITY_PRESETS = {
    "⚡ Fast (Quick Tags)": {
        "max_tokens": 256,
        "max_image_size": "768",
        "description": "Fast tagging and simple descriptions"
    },
    "⚖️ Balanced (Default)": {
        "max_tokens": 512,
        "max_image_size": "1024",
        "description": "Good balance of speed and quality"
    },
    "🎯 Detailed (Analysis)": {
        "max_tokens": 1024,
        "max_image_size": "1536",
        "description": "Detailed descriptions and analysis"
    },
    "🏠 Interior Design (Best)": {
        "max_tokens": 2048,
        "max_image_size": "1536",
        "description": "Maximum detail for interior design prompts"
    },
    "🎨 Creative (High Tokens)": {
        "max_tokens": 3072,
        "max_image_size": "2048",
        "description": "Maximum creativity and token output"
    },
    "🔬 Ultra Focused (Accurate)": {
        "max_tokens": 16384,
        "max_image_size": "Original",
        "creativity": 0.3,
        "description": "Ultra quality, focused but not too restrictive"
    },
    "🔬 Ultra Balanced": {
        "max_tokens": 16384,
        "max_image_size": "Original",
        "creativity": 0.5,
        "description": "Ultra quality, medium creativity - balanced output"
    },
    "🔬 Ultra Creative": {
        "max_tokens": 16384,
        "max_image_size": "Original",
        "creativity": 0.9,
        "description": "Ultra quality, high creativity - creative/varied output"
    },
    "📋 Qwen Official": {
        "max_tokens": 16384,
        "max_image_size": "Original",
        "params": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "repeat_penalty": 1.0
        },
        "description": "Official Qwen3-VL 8B recommended settings"
    },
    "📋 Qwen Focused": {
        "max_tokens": 16384,
        "max_image_size": "Original",
        "params": {
            "temperature": 0.5,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "repeat_penalty": 1.0
        },
        "description": "Qwen settings with lower temp for accuracy"
    },
    "⚙️ Custom": {
        "max_tokens": None,  # Use manual settings
        "max_image_size": None,
        "description": "Use manual max_tokens and max_image_size settings"
    }
}
QUALITY_PRESET_NAMES = list(QUALITY_PRESETS.keys())


def creativity_to_params(creativity: float) -> dict:
    """Convert creativity slider (0.0-1.0) to sampling parameters.

    creativity 0.0 = Focused/Deterministic (best for accurate descriptions)
    creativity 0.5 = Balanced (default)
    creativity 1.0 = Creative/Random (best for stories, creative writing)

    Returns dict with: temperature, top_p, top_k, min_p, repeat_penalty
    """
    # Clamp to valid range
    c = max(0.0, min(1.0, creativity))

    # Temperature: 0.1 (focused) to 1.0 (creative)
    temperature = 0.1 + (c * 0.9)

    # Top-P: 0.7 (focused) to 0.98 (creative)
    top_p = 0.7 + (c * 0.28)

    # Top-K: 20 (focused) to 100 (creative), 0 at max creativity (disabled)
    if c >= 0.95:
        top_k = 0  # Disable top_k at max creativity
    else:
        top_k = int(20 + (c * 80))

    # Min-P: 0.1 (focused) to 0.01 (creative)
    min_p = 0.1 - (c * 0.09)

    # Repeat penalty: 1.2 (focused, penalize more) to 1.0 (creative, no penalty)
    repeat_penalty = 1.2 - (c * 0.2)

    return {
        "temperature": round(temperature, 2),
        "top_p": round(top_p, 2),
        "top_k": top_k,
        "min_p": round(min_p, 3),
        "repeat_penalty": round(repeat_penalty, 2)
    }


# Available model sizes with their ports
MODEL_CONFIGS = {
    "2B (Fast, ~4GB VRAM)": {"port": 8032, "name": "Qwen3-VL-2B"},
    "4B (Balanced, ~7GB VRAM)": {"port": 8033, "name": "Qwen3-VL-4B"},
    "8B (Best Quality, ~12GB VRAM)": {"port": 8034, "name": "Qwen3-VL-8B"},
}
MODEL_SIZES = list(MODEL_CONFIGS.keys())

# Script path for auto-starting server
SCRIPT_PATH = Path(__file__).parent.parent.parent / "start_qwenvl_server.sh"


def kill_llama_server(port):
    """Kill llama-server process running on specific port."""
    try:
        # Find llama-server process specifically using the port
        # Use pgrep to find llama-server processes, then check which one uses this port
        result = subprocess.run(
            ["pgrep", "-f", "llama-server"],
            capture_output=True, text=True
        )
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                try:
                    pid_int = int(pid)
                    # Verify this process is using our port by checking its command line
                    cmdline_path = f"/proc/{pid}/cmdline"
                    if os.path.exists(cmdline_path):
                        with open(cmdline_path, 'r') as f:
                            cmdline = f.read()
                        if f"--port\x00{port}" in cmdline or f"-p\x00{port}" in cmdline or str(port) in cmdline:
                            os.kill(pid_int, signal.SIGTERM)
                            print(f"[QwenVL-GGUF] Killed llama-server process {pid} on port {port}")
                            return True
                except (ProcessLookupError, ValueError, FileNotFoundError, PermissionError):
                    pass
        return False
    except Exception as e:
        print(f"[QwenVL-GGUF] Error killing server: {e}")
        return False


def start_llama_server(model_size, port):
    """Start llama-server for the specified model size."""
    model_short = model_size.split()[0]  # "4B" from "4B (Balanced, ~7GB VRAM)"

    if not SCRIPT_PATH.exists():
        print(f"[QwenVL-GGUF] Warning: Server script not found at {SCRIPT_PATH}")
        return None

    # Start server in background
    env = os.environ.copy()
    env["CTX"] = "8192"
    env["GPU_LAYERS"] = "99"

    process = subprocess.Popen(
        ["bash", str(SCRIPT_PATH), model_short],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        env=env
    )
    print(f"[QwenVL-GGUF] Started server for {model_short} on port {port} (PID: {process.pid})")
    return process


class ArchAi3D_QwenVL_GGUF:
    """Fast QwenVL inference using llama.cpp GGUF models via HTTP API.

    This node connects to a running llama-server instance to perform
    vision-language inference using GGUF quantized models.

    Benefits:
    - 20x faster inference than transformers backend (~126 tok/s)
    - Lower VRAM usage with Q4_K_M quantization (~7GB)
    - Can hot-swap models by changing server
    - Supports multiple images in a single request

    Requirements:
    - llama-server running with Qwen3-VL GGUF model
    - Start server with: ./start_qwenvl_server.sh 4B
    """

    CATEGORY = "ArchAi3d/Qwen/VLM"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # ===== MAIN SETTINGS (Simple) =====
                "model_size": (MODEL_SIZES, {
                    "default": "4B (Balanced, ~7GB VRAM)",
                    "tooltip": "Select model size. 2B=Fast, 4B=Balanced, 8B=Best quality"
                }),
                "preset_prompt": (PRESET_PROMPTS, {
                    "default": PRESET_PROMPTS[0] if PRESET_PROMPTS else "🖼️ Tags",
                    "tooltip": "Select a preset prompt for image analysis"
                }),
                "quality_preset": (QUALITY_PRESET_NAMES, {
                    "default": "🏠 Interior Design (Best)",
                    "tooltip": "Quality preset: Fast=256 tokens, Balanced=512, Interior Design=2048, Custom=use manual settings"
                }),
                "creativity": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "0.0=Focused/Accurate (best for descriptions), 0.5=Balanced, 1.0=Creative/Random (best for stories)"
                }),
                "seed": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 2**32 - 1,
                    "tooltip": "Random seed for reproducibility"
                }),
            },
            "optional": {
                # ===== IMAGES =====
                "image": ("IMAGE", {
                    "tooltip": "Primary image for analysis"
                }),
                "image2": ("IMAGE", {
                    "tooltip": "Optional second image for comparison/analysis"
                }),
                "image3": ("IMAGE", {
                    "tooltip": "Optional third image"
                }),
                "image4": ("IMAGE", {
                    "tooltip": "Optional fourth image"
                }),
                # ===== CUSTOM PROMPT =====
                "custom_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Custom prompt (overrides preset if provided)"
                }),
                "system_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "forceInput": True,
                    "tooltip": "Optional system prompt to prepend"
                }),
                # ===== ADVANCED SETTINGS (only used when quality_preset=Custom) =====
                "max_tokens": ("INT", {
                    "default": 2048,
                    "min": 64,
                    "max": 16384,
                    "tooltip": "Maximum tokens (only used with Custom quality preset)"
                }),
                "max_image_size": (["Original", "512", "768", "1024", "1536", "2048"], {
                    "default": "1536",
                    "tooltip": "Max image dimension (only used with Custom quality preset)"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Override creativity temperature (only used with Custom quality preset)"
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Override nucleus sampling (only used with Custom quality preset)"
                }),
                "top_k": ("INT", {
                    "default": 40,
                    "min": 0,
                    "max": 100,
                    "tooltip": "Override top-K sampling (only used with Custom quality preset)"
                }),
                "min_p": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Override min-P threshold (only used with Custom quality preset)"
                }),
                "repeat_penalty": ("FLOAT", {
                    "default": 1.1,
                    "min": 1.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Override repetition penalty (only used with Custom quality preset)"
                }),
                # ===== SERVER CONTROL =====
                "use_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Cache results - same image + settings = instant result"
                }),
                "keep_server_running": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep server running after inference (faster)"
                }),
                "auto_start_server": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically start server if not running"
                }),
            }
        }

    def image_to_base64(self, image_tensor, index=0, max_size=None):
        """Convert ComfyUI image tensor to base64 string with optional resizing.

        Args:
            image_tensor: Tensor of shape [B, H, W, C] in range [0, 1]
            index: Which image in batch to convert (default 0)
            max_size: Maximum dimension (width or height). None = no resize.

        Returns:
            Base64 encoded PNG string
        """
        idx = min(index, image_tensor.shape[0] - 1)
        img_np = (image_tensor[idx].cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)

        # Resize if max_size specified and image exceeds it
        if max_size is not None:
            w, h = pil_img.size
            if w > max_size or h > max_size:
                # Calculate new size preserving aspect ratio
                if w > h:
                    new_w = max_size
                    new_h = int(h * max_size / w)
                else:
                    new_h = max_size
                    new_w = int(w * max_size / h)
                pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                print(f"[QwenVL-GGUF] Resized image from {w}x{h} to {new_w}x{new_h}")

        # Encode as PNG
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def compute_cache_key(self, images, prompt, max_tokens, temperature, top_p, top_k, min_p, repeat_penalty, seed, server_url):
        """Create unique hash from all inputs for result caching."""
        hasher = hashlib.md5()

        # Hash all images
        for img in images:
            if img is not None:
                hasher.update(img.cpu().numpy().tobytes())

        hasher.update(prompt.encode())
        hasher.update(str(seed).encode())
        hasher.update(str(max_tokens).encode())
        hasher.update(f"{temperature:.4f}".encode())
        hasher.update(f"{top_p:.4f}".encode())
        hasher.update(str(top_k).encode())
        hasher.update(f"{min_p:.4f}".encode())
        hasher.update(f"{repeat_penalty:.4f}".encode())
        hasher.update(server_url.encode())
        return hasher.hexdigest()

    def generate(self, model_size, preset_prompt, quality_preset, creativity, seed,
                 image=None, image2=None, image3=None, image4=None,
                 custom_prompt="", system_prompt="",
                 max_tokens=2048, max_image_size="1536",
                 temperature=0.3, top_p=0.9, top_k=40, min_p=0.05, repeat_penalty=1.1,
                 use_cache=True, keep_server_running=True, auto_start_server=True):
        """Generate text from image(s) using llama-server API. Images are optional for text-only prompts."""

        # ===== APPLY QUALITY PRESET =====
        preset_config = QUALITY_PRESETS.get(quality_preset, QUALITY_PRESETS["⚖️ Balanced (Default)"])

        # Use preset values unless "Custom" is selected
        if quality_preset != "⚙️ Custom":
            max_tokens = preset_config["max_tokens"]
            max_image_size = preset_config["max_image_size"]

        # ===== APPLY SAMPLING PARAMETERS =====
        # Priority: 1) Direct params in preset, 2) Creativity slider/preset, 3) Custom manual
        if "params" in preset_config:
            # Use direct parameters from preset (Qwen Official presets)
            temperature = preset_config["params"]["temperature"]
            top_p = preset_config["params"]["top_p"]
            top_k = preset_config["params"]["top_k"]
            min_p = preset_config["params"]["min_p"]
            repeat_penalty = preset_config["params"]["repeat_penalty"]
        elif quality_preset != "⚙️ Custom":
            # Use creativity slider (or preset's creativity if set)
            preset_creativity = preset_config.get("creativity", creativity)
            creativity_params = creativity_to_params(preset_creativity)
            temperature = creativity_params["temperature"]
            top_p = creativity_params["top_p"]
            top_k = creativity_params["top_k"]
            min_p = creativity_params["min_p"]
            repeat_penalty = creativity_params["repeat_penalty"]

        # Log applied settings
        if "params" in preset_config:
            print(f"[QwenVL-GGUF] Quality: {quality_preset} (direct params)")
        else:
            actual_creativity = preset_config.get("creativity", creativity) if quality_preset != "⚙️ Custom" else creativity
            print(f"[QwenVL-GGUF] Quality: {quality_preset}, Creativity: {actual_creativity:.2f}")
        print(f"[QwenVL-GGUF] Settings: tokens={max_tokens}, temp={temperature:.2f}, top_p={top_p:.2f}, top_k={top_k}, min_p={min_p:.3f}, repeat_pen={repeat_penalty:.2f}")

        # Parse max_image_size
        resize_max = None if max_image_size == "Original" else int(max_image_size)

        # Get server URL from model size selection
        config = MODEL_CONFIGS.get(model_size, MODEL_CONFIGS["4B (Balanced, ~7GB VRAM)"])
        port = config['port']
        server_url = f"http://localhost:{port}"

        # Check if server is running and model is loaded
        import time
        server_was_started = False

        def check_server_ready():
            """Check if server is running AND model is loaded (not returning 503)."""
            try:
                resp = requests.get(f"{server_url}/health", timeout=2)
                if resp.status_code == 200:
                    # Also verify model is loaded by checking if we get 503
                    return True
                return False
            except requests.exceptions.ConnectionError:
                return False

        if not check_server_ready():
            if auto_start_server:
                print(f"[QwenVL-GGUF] Server not running, starting automatically...")
                start_llama_server(model_size, port)
                server_was_started = True

                # Wait for server to be fully ready (health check + model loaded)
                for i in range(60):  # Wait up to 60 seconds for model loading
                    time.sleep(1)
                    try:
                        resp = requests.get(f"{server_url}/health", timeout=2)
                        if resp.status_code == 200:
                            # Test with a simple request to verify model is loaded
                            test_resp = requests.post(
                                f"{server_url}/v1/chat/completions",
                                json={"model": "test", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
                                timeout=5
                            )
                            if test_resp.status_code != 503:
                                print(f"[QwenVL-GGUF] Server and model ready after {i+1} seconds")
                                break
                    except requests.exceptions.ConnectionError:
                        pass
                    except requests.exceptions.Timeout:
                        # Timeout on test request means model is processing = ready
                        print(f"[QwenVL-GGUF] Server and model ready after {i+1} seconds")
                        break

                    if i % 5 == 0:
                        print(f"[QwenVL-GGUF] Loading model... ({i+1}s)")
                else:
                    return (f"ERROR: Server/model failed to load after 60 seconds. Try manually:\n./start_qwenvl_server.sh {model_size.split()[0]}",)

        # Collect all images (filter out None values)
        images = []
        if image is not None:
            images.append(image)
        if image2 is not None:
            images.append(image2)
        if image3 is not None:
            images.append(image3)
        if image4 is not None:
            images.append(image4)

        # Resolve prompt - custom overrides preset
        if custom_prompt and custom_prompt.strip():
            prompt = custom_prompt.strip()
        else:
            prompt = SYSTEM_PROMPTS.get(preset_prompt, preset_prompt)

        # Add system prompt if provided
        if system_prompt and system_prompt.strip():
            prompt = f"{system_prompt.strip()}\n\n{prompt}"

        # Check cache first
        cache_key = None
        if use_cache:
            cache_key = self.compute_cache_key(
                images, prompt, max_tokens, temperature, top_p, top_k, min_p, repeat_penalty, seed, server_url
            )
            if cache_key in GGUF_RESULT_CACHE:
                print("[QwenVL-GGUF] Cache hit - returning stored result")
                return (GGUF_RESULT_CACHE[cache_key],)

        # Build content array - images first, then text
        content = []

        # Add all images first (if any)
        if images:
            for i, img in enumerate(images):
                # Handle batched images - process all in batch
                batch_size = img.shape[0]
                for j in range(batch_size):
                    img_b64 = self.image_to_base64(img, index=j, max_size=resize_max)
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}"
                        }
                    })

        # Add text prompt
        content.append({
            "type": "text",
            "text": prompt
        })

        # Build OpenAI-compatible request payload
        payload = {
            "model": "qwen3-vl",
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k if top_k > 0 else -1,  # llama.cpp uses -1 to disable
            "min_p": min_p,
            "repeat_penalty": repeat_penalty,
            "seed": seed,
            "stream": False
        }

        # Send request to llama-server
        try:
            num_images = sum(img.shape[0] for img in images) if images else 0
            mode = f"with {num_images} image(s)" if num_images > 0 else "text-only"
            print(f"[QwenVL-GGUF] Sending request {mode} to {server_url}...")
            response = requests.post(
                f"{server_url}/v1/chat/completions",
                json=payload,
                timeout=300  # 5 minute timeout for multiple images
            )
            response.raise_for_status()

            result = response.json()
            text = result["choices"][0]["message"]["content"]

            # Cache result
            if use_cache and cache_key:
                GGUF_RESULT_CACHE[cache_key] = text
                print(f"[QwenVL-GGUF] Result cached (total: {len(GGUF_RESULT_CACHE)} entries)")

            # Kill server after inference if not keeping it running
            if not keep_server_running:
                print(f"[QwenVL-GGUF] Stopping server to free GPU VRAM...")
                kill_llama_server(port)

            return (text,)

        except requests.exceptions.ConnectionError:
            model_short = model_size.split()[0]  # "4B" from "4B (Balanced, ~7GB VRAM)"
            error_msg = (
                f"ERROR: Cannot connect to {config['name']} server on port {config['port']}.\n\n"
                f"Start the server with:\n"
                f"./start_qwenvl_server.sh {model_short}\n\n"
                f"Or with custom settings:\n"
                f"CTX=8192 GPU_LAYERS=99 ./start_qwenvl_server.sh {model_short}"
            )
            print(f"[QwenVL-GGUF] {error_msg}")
            return (error_msg,)

        except requests.exceptions.Timeout:
            error_msg = "ERROR: Request timed out after 300 seconds. Try fewer/smaller images."
            print(f"[QwenVL-GGUF] {error_msg}")
            return (error_msg,)

        except requests.exceptions.HTTPError as e:
            error_msg = f"ERROR: HTTP error from server: {e.response.status_code} - {e.response.text}"
            print(f"[QwenVL-GGUF] {error_msg}")
            return (error_msg,)

        except Exception as e:
            error_msg = f"ERROR: {type(e).__name__}: {str(e)}"
            print(f"[QwenVL-GGUF] {error_msg}")
            return (error_msg,)


class ArchAi3D_QwenVL_Server_Control:
    """Control the QwenVL GGUF llama-server: start, stop, or check status.

    Use this node to:
    - Stop the server to free GPU VRAM for other tasks
    - Start the server with specific model and settings
    - Check if the server is currently running

    The server runs on different ports based on model size:
    - 2B: port 8032
    - 4B: port 8033
    - 8B: port 8034
    """

    CATEGORY = "ArchAi3d/Qwen/VLM"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "control"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (["Stop Server", "Start Server", "Check Status", "Restart Server"], {
                    "default": "Check Status",
                    "tooltip": "Action to perform on the llama-server"
                }),
                "model_size": (MODEL_SIZES, {
                    "default": "4B (Balanced, ~7GB VRAM)",
                    "tooltip": "Model size determines which port to use"
                }),
                "gpu_layers": ("INT", {
                    "default": 99,
                    "min": 0,
                    "max": 99,
                    "tooltip": "GPU layers (99=all on GPU, lower values use more CPU to save VRAM)"
                }),
                "context_size": ("INT", {
                    "default": 8192,
                    "min": 1024,
                    "max": 32768,
                    "step": 1024,
                    "tooltip": "Context window size in tokens"
                }),
            },
            "optional": {
                "trigger": ("*", {
                    "tooltip": "Optional input to trigger this node in a workflow"
                }),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always execute to get current status
        import time
        return time.time()

    def get_server_status(self, port):
        """Check if server is running and get details."""
        try:
            resp = requests.get(f"http://localhost:{port}/health", timeout=2)
            if resp.status_code == 200:
                return True, "running"
            return False, "not responding"
        except requests.exceptions.ConnectionError:
            return False, "not running"
        except Exception as e:
            return False, str(e)

    def control(self, action, model_size, gpu_layers, context_size, trigger=None):
        """Control the llama-server."""
        import time

        config = MODEL_CONFIGS.get(model_size, MODEL_CONFIGS["4B (Balanced, ~7GB VRAM)"])
        port = config['port']
        model_name = config['name']
        model_short = model_size.split()[0]

        if action == "Check Status":
            running, status = self.get_server_status(port)
            if running:
                return (f"✅ {model_name} server is RUNNING on port {port}",)
            else:
                return (f"❌ {model_name} server is NOT running (port {port})\n\nStart with:\n./start_qwenvl_server.sh {model_short}",)

        elif action == "Stop Server":
            running, _ = self.get_server_status(port)
            if not running:
                return (f"ℹ️ {model_name} server was not running (port {port})",)

            killed = kill_llama_server(port)
            if killed:
                # Wait a moment for cleanup
                time.sleep(1)
                return (f"✅ {model_name} server STOPPED (port {port})\nGPU VRAM freed!",)
            else:
                return (f"⚠️ Could not find {model_name} server process to stop",)

        elif action == "Start Server":
            running, _ = self.get_server_status(port)
            if running:
                return (f"ℹ️ {model_name} server is already running on port {port}",)

            # Start with custom settings
            env = os.environ.copy()
            env["CTX"] = str(context_size)
            env["GPU_LAYERS"] = str(gpu_layers)

            if not SCRIPT_PATH.exists():
                return (f"❌ Server script not found at {SCRIPT_PATH}",)

            process = subprocess.Popen(
                ["bash", str(SCRIPT_PATH), model_short],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                env=env
            )

            # Wait for server to be ready
            for i in range(60):
                time.sleep(1)
                running, _ = self.get_server_status(port)
                if running:
                    return (f"✅ {model_name} server STARTED on port {port}\n\nSettings:\n- GPU Layers: {gpu_layers}\n- Context: {context_size} tokens",)
                if i % 5 == 0 and i > 0:
                    print(f"[QwenVL-Server] Loading model... ({i}s)")

            return (f"⚠️ Server started but not responding after 60s. Check logs.",)

        elif action == "Restart Server":
            # Stop first
            kill_llama_server(port)
            time.sleep(2)

            # Then start
            env = os.environ.copy()
            env["CTX"] = str(context_size)
            env["GPU_LAYERS"] = str(gpu_layers)

            if not SCRIPT_PATH.exists():
                return (f"❌ Server script not found at {SCRIPT_PATH}",)

            process = subprocess.Popen(
                ["bash", str(SCRIPT_PATH), model_short],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                env=env
            )

            # Wait for server to be ready
            for i in range(60):
                time.sleep(1)
                running, _ = self.get_server_status(port)
                if running:
                    return (f"✅ {model_name} server RESTARTED on port {port}\n\nSettings:\n- GPU Layers: {gpu_layers}\n- Context: {context_size} tokens",)
                if i % 5 == 0 and i > 0:
                    print(f"[QwenVL-Server] Loading model... ({i}s)")

            return (f"⚠️ Server restart initiated but not responding after 60s.",)

        return ("Unknown action",)
