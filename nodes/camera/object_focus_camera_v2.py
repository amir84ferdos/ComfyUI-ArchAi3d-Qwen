"""
Object Focus Camera v2 - Reddit-Validated Prompts

Simple node for object close-ups using community-tested camera control prompts.
Based on Reddit research documented in:
  E:\\Comfy\\help\\03-RESEARCH\\QWEN_PROMPT_WRITING\\REDDIT_RESEARCH\\reddit-camera-prompts-no-lora.md

Key Findings from Community Testing:
- ⭐⭐⭐⭐⭐ "camera orbit around" is #1 most reliable method
- "dolly in/out" is most consistent for zoom control
- Works with native Qwen (no LoRA required)
- Works universally with both dx8152 LoRAs loaded

Author: ArchAi3d
Version: 2.0.0 - Reddit-validated prompts
"""

class ArchAi3D_Object_Focus_Camera_V2:
    """Object Focus Camera v2 - Community-validated camera control.

    Uses Reddit-tested prompts that work universally:
    - Native Qwen Image Edit 2509 (no LoRA)
    - dx8152 Multiple Angles LoRA
    - dx8152 Next Scene LoRA

    All prompts can work with both LoRAs loaded simultaneously.
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
                "camera_action": ([
                    "Orbit Left 30°",
                    "Orbit Left 45°",
                    "Orbit Left 90°",
                    "Orbit Right 30°",
                    "Orbit Right 45°",
                    "Orbit Right 90°",
                    "Orbit Up 30°",
                    "Orbit Up 45°",
                    "Orbit Down 30°",
                    "Orbit Down 45°",
                    "Dolly In (Zoom Closer)",
                    "Dolly Out (Zoom Away)",
                    "View from Above (Bird's Eye)",
                    "View from Ground Level (Worm's Eye)",
                    "Tilt Up Slightly",
                    "Tilt Down Slightly"
                ], {
                    "default": "Orbit Right 45°",
                    "tooltip": "Reddit-validated camera movements (orbit around = ⭐⭐⭐⭐⭐)"
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

    def generate_object_focus_prompt(self, target_object, camera_action, show_details=""):
        """
        Generate object focus camera prompt using Reddit-validated patterns.

        Based on community testing:
        - "orbit around" works great even at 90 degrees
        - "dolly" is most consistent for zoom
        - Simple patterns work better than complex ones
        """

        # Generate prompt based on camera action
        prompt = self._build_camera_prompt(target_object, camera_action, show_details)

        # Generate English description for user
        description = f"{camera_action} | Object: {target_object}"
        if show_details:
            description += f" | {show_details}"

        # System prompt optimized for object preservation
        system_prompt = (
            "You are a precision camera operator for object photography. "
            "Execute camera positioning exactly as instructed while keeping "
            "the object and scene completely unchanged. Preserve all details, "
            "textures, colors, materials, and lighting exactly as they are. "
            "Your only job is to change the camera viewpoint - do not modify, "
            "redesign, or reimagine anything in the scene."
        )

        return (prompt, system_prompt, description)

    def _build_camera_prompt(self, target_object, camera_action, show_details):
        """Build camera prompt using Reddit-validated patterns."""

        # Orbit movements (⭐⭐⭐⭐⭐ most reliable per Reddit)
        if "Orbit Left" in camera_action:
            degrees = self._extract_degrees(camera_action)
            prompt = f"camera orbit left around {target_object} by {degrees} degrees"

        elif "Orbit Right" in camera_action:
            degrees = self._extract_degrees(camera_action)
            prompt = f"camera orbit right around {target_object} by {degrees} degrees"

        elif "Orbit Up" in camera_action:
            degrees = self._extract_degrees(camera_action)
            prompt = f"camera orbit up around {target_object} by {degrees} degrees"

        elif "Orbit Down" in camera_action:
            degrees = self._extract_degrees(camera_action)
            prompt = f"camera orbit down around {target_object} by {degrees} degrees"

        # Dolly movements (⭐⭐⭐⭐⭐ most consistent for zoom per Reddit)
        elif "Dolly In" in camera_action:
            prompt = f"dolly in"

        elif "Dolly Out" in camera_action:
            prompt = f"dolly out"

        # View from above/below (⭐⭐⭐⭐ effective per Reddit)
        elif "View from Above" in camera_action:
            prompt = f"view from above, bird's eye view"

        elif "View from Ground Level" in camera_action:
            prompt = f"view from ground level, worm's eye view"

        # Tilt movements (⭐⭐⭐⭐ reliable per Reddit)
        elif "Tilt Up" in camera_action:
            prompt = f"change the view and tilt the camera up slightly"

        elif "Tilt Down" in camera_action:
            prompt = f"change the view and tilt the camera down slightly"

        else:
            # Fallback to orbit right 45°
            prompt = f"camera orbit right around {target_object} by 45 degrees"

        # Add optional details
        if show_details and show_details.strip():
            prompt += f", {show_details.strip()}"

        return prompt

    def _extract_degrees(self, camera_action):
        """Extract degree number from camera action string."""
        if "30°" in camera_action or "30" in camera_action:
            return "30"
        elif "45°" in camera_action or "45" in camera_action:
            return "45"
        elif "90°" in camera_action or "90" in camera_action:
            return "90"
        else:
            return "45"  # Default


# Node registration
NODE_CLASS_MAPPINGS = {
    "ArchAi3D_Object_Focus_Camera_V2": ArchAi3D_Object_Focus_Camera_V2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_Object_Focus_Camera_V2": "📦 Object Focus Camera v2 (Reddit)"
}
