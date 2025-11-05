"""
Object Focus Camera Node for dx8152 LoRAs

Simple, focused node for object close-up photography.
Works with both Next Scene and Multiple Angles LoRAs.

Features:
- 5 camera positions (front, angled, side, top-down, low angle)
- 3 lens types (normal, close-up, macro)
- 4 distance presets
- Chinese prompt generation (best performance with dx8152)
- Always adds "Next Scene: " prefix for LoRA compatibility

Author: ArchAi3d
Version: 1.0.0 - Simple and direct object focus
"""

class ArchAi3D_Object_Focus_Camera:
    """Simple object focus camera node for dx8152 LoRAs.

    Purpose: Get close-up shots of specific objects with proper positioning.
    Optimized for: Product photography, macro shots, detail captures.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_object": ("STRING", {
                    "default": "the object",
                    "multiline": False,
                    "tooltip": "What to focus on: 'the watch', 'the ring', 'the door handle'"
                }),
                "camera_position": ([
                    "Front View",
                    "Angled View (30°)",
                    "Side View (90°)",
                    "Top-Down View",
                    "Low Angle View"
                ], {
                    "default": "Front View",
                    "tooltip": "Camera position relative to the object"
                }),
                "camera_distance": ([
                    "Very Close (Macro)",
                    "Close",
                    "Medium",
                    "Far"
                ], {
                    "default": "Close",
                    "tooltip": "Distance from the object"
                }),
                "lens_type": ([
                    "Normal Lens",
                    "Close-Up Lens",
                    "Macro Lens"
                ], {
                    "default": "Close-Up Lens",
                    "tooltip": "Lens type - Close-Up and Macro are optimized for dx8152 LoRA"
                }),
                "lora_mode": ([
                    "Multiple Angles LoRA",
                    "Next Scene LoRA"
                ], {
                    "default": "Multiple Angles LoRA",
                    "tooltip": "Which dx8152 LoRA you're using. Multiple Angles = camera movements, Next Scene = scene transitions"
                }),
            },
            "optional": {
                "show_details": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Optional: What details to show. Example: 'showing fine texture and engravings'"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "system_prompt", "description")
    FUNCTION = "generate_object_focus_prompt"
    CATEGORY = "ArchAi3d/Qwen/Camera"

    def generate_object_focus_prompt(self, target_object, camera_position,
                                     camera_distance, lens_type, lora_mode,
                                     show_details=""):
        """
        Generate simple, direct prompt for object focus camera work.

        Format: "Next Scene: " + Chinese camera instructions
        """

        # Get Chinese translations
        lens_cn = self._get_lens_chinese(lens_type)
        position_cn = self._get_position_chinese(camera_position)
        distance_cn = self._get_distance_chinese(camera_distance)

        # Build Chinese prompt parts
        parts = []

        # 1. Lens change (always first)
        parts.append(f"将镜头{lens_cn}")

        # 2. Position + object
        parts.append(f"{position_cn}{target_object}")

        # 3. Distance
        parts.append(f"距离{distance_cn}")

        # 4. Optional details
        if show_details and show_details.strip():
            parts.append(show_details.strip())

        # Combine with proper Chinese grammar
        prompt_chinese = "，".join(parts)

        # Add "Next Scene: " prefix (required for dx8152 LoRAs)
        prompt = f"Next Scene: {prompt_chinese}"

        # Generate English description for user
        lens_en = lens_type.replace(" Lens", "")
        position_en = camera_position
        distance_en = camera_distance
        description = f"{lens_en} | {position_en} | {distance_en} | Object: {target_object}"
        if show_details:
            description += f" | {show_details}"

        # System prompt (optimized for object preservation)
        system_prompt = self._get_system_prompt(lora_mode)

        return (prompt, system_prompt, description)

    def _get_lens_chinese(self, lens_type):
        """Convert lens type to Chinese."""
        lens_map = {
            "Normal Lens": "转为标准镜头",
            "Close-Up Lens": "转为特写镜头",
            "Macro Lens": "转为微距镜头"
        }
        return lens_map.get(lens_type, "转为特写镜头")

    def _get_position_chinese(self, position):
        """Convert camera position to Chinese."""
        position_map = {
            "Front View": "正面查看",
            "Angled View (30°)": "从30度角查看",
            "Side View (90°)": "从侧面查看",
            "Top-Down View": "从俯视角度查看",
            "Low Angle View": "从仰视角度查看"
        }
        return position_map.get(position, "正面查看")

    def _get_distance_chinese(self, distance):
        """Convert distance to Chinese."""
        distance_map = {
            "Very Close (Macro)": "很近",
            "Close": "近距离",
            "Medium": "中等距离",
            "Far": "远距离"
        }
        return distance_map.get(distance, "近距离")

    def _get_system_prompt(self, lora_mode):
        """Get system prompt based on LoRA mode."""

        if lora_mode == "Multiple Angles LoRA":
            # For camera movements (preserve scene perfectly)
            return (
                "You are a precision camera operator for object photography. "
                "Execute camera positioning exactly as instructed while keeping "
                "the object and scene completely unchanged. Preserve all details, "
                "textures, colors, materials, and lighting exactly as they are. "
                "Your only job is to change the camera viewpoint - do not modify, "
                "redesign, or reimagine anything in the scene."
            )
        else:
            # For scene transitions (Next Scene LoRA)
            return (
                "You are creating a new scene view while maintaining visual consistency. "
                "Focus on the specified object with the requested camera angle. "
                "Preserve the object's appearance, style, and quality. Generate a "
                "natural, intentional composition that highlights the object as instructed."
            )


# Node registration
NODE_CLASS_MAPPINGS = {
    "ArchAi3D_Object_Focus_Camera": ArchAi3D_Object_Focus_Camera
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_Object_Focus_Camera": "📦 Object Focus Camera"
}
