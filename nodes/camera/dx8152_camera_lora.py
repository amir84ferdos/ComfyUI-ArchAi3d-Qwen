"""
dx8152 Camera LoRA Node for Qwen Image Edit

Super simple node for dx8152 Multiple Angles LoRA with automatic "Next Scene: " prefix.

Features:
- English interface, Chinese output (better performance)
- 6 movement directions (forward, backward, left, right, up, down)
- 4 rotation types (left, right, top-down, low angle upward)
- Flexible rotation angle (0-180 degrees, step 15)
- 6 lens types (wide-angle, close-up, telephoto, fisheye, macro)
- Auto-generates proper Chinese grammar with 并 (and) connector
- Always adds "Next Scene: " prefix in English (required for dx8152 LoRA)
- Camera movements in Chinese (better performance per user testing)
- Optional scene description for what camera sees
- Mix movements + rotations + lens changes in one prompt!

Based on user testing: Chinese prompts work better than English!
Source: https://huggingface.co/dx8152/Qwen-Edit-2509-Multiple-angles

Author: ArchAi3d
Version: 2.1.2 - Optimized system prompt (removed LoRA references Qwen doesn't understand)
"""

class ArchAi3D_Qwen_DX8152_Camera_LoRA:
    """dx8152 Camera LoRA - Simple Chinese Prompt Generator

    Dedicated node for dx8152 Multiple Angles LoRA with proper Chinese formatting.
    Shows English options in UI, outputs Chinese prompts for best performance.
    Always adds "Next Scene: " prefix to all generated prompts.

    Supports:
    - 6 camera movements (forward, backward, left, right, up, down)
    - 4 rotation types (left, right, top-down, low angle upward)
    - 6 lens types (wide-angle, close-up + experimental: telephoto, fisheye, macro)
    - Flexible angle control (0-180 degrees)
    - Mixing movements + rotations + lens changes in one prompt
    - Optional scene descriptions

    Based on HuggingFace dx8152/Qwen-Edit-2509-Multiple-angles LoRA.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "movement_type": ([
                    "None",
                    "Move Forward",
                    "Move Backward",
                    "Move Left",
                    "Move Right",
                    "Move Up",
                    "Move Down"
                ], {
                    "default": "None",
                    "tooltip": "Camera movement direction. Tested: forward, backward, left, right, up, down. Can combine with rotation!"
                }),
                "rotation_type": ([
                    "None",
                    "Rotate Left",
                    "Rotate Right",
                    "Top-Down View",
                    "Low Angle (Upward)"
                ], {
                    "default": "None",
                    "tooltip": "Camera rotation. Can combine with movement! Note: Low Angle has limited training data."
                }),
                "rotation_angle": ("INT", {
                    "default": 45,
                    "min": 0,
                    "max": 180,
                    "step": 15,
                    "tooltip": "Rotation angle in degrees. Only used if Rotate Left/Right selected. Tested angles: 45, 90, 180"
                }),
                "lens_type": ([
                    "None",
                    "Wide-Angle",
                    "Close-Up",
                    "Telephoto",
                    "Fisheye",
                    "Macro"
                ], {
                    "default": "None",
                    "tooltip": "Lens type change. Primary support: Wide-Angle, Close-Up. Experimental: Telephoto, Fisheye, Macro"
                }),
                "output_language": ([
                    "Chinese Only (Best Performance)",
                    "English Only",
                    "Both (Chinese + English)"
                ], {
                    "default": "Chinese Only (Best Performance)",
                    "tooltip": "User tested: Chinese works better! Use 'Both' to see translation."
                }),
            },
            "optional": {
                "scene_description": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Describe what the camera sees from the new viewpoint. Example: 'show the fireplace with chairs on both sides'"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "system_prompt", "description")
    FUNCTION = "generate_lora_prompt"
    CATEGORY = "ArchAi3d/Qwen/Camera"

    def generate_lora_prompt(self, movement_type, rotation_type,
                            rotation_angle, lens_type, output_language,
                            scene_description=""):
        """
        Generate dx8152 LoRA prompt with proper Chinese/English formatting.

        Format: "Next Scene: " (English) + camera_prompt (Chinese/English/Both)
        Combines movement + rotation + lens with Chinese grammar using 并 connector.
        """

        # Generate camera movement prompt
        camera_prompt, description = self._generate_multiple_angles_prompt(
            movement_type, rotation_type, rotation_angle,
            lens_type, output_language, scene_description
        )

        # Always add "Next Scene: " prefix (English only, as per dx8152 LoRA requirements)
        prompt = f"Next Scene: {camera_prompt}"

        system_prompt = self._get_system_prompt("multiple_angles")

        return (prompt, system_prompt, description)

    def _generate_multiple_angles_prompt(self, movement, rotation, angle, lens, output_language, scene_description=""):
        """
        Generate Multiple Angles LoRA prompt with proper Chinese grammar.

        Combines movement + rotation + lens with proper 并 (and) connectors.
        Optionally adds scene description of what camera sees.
        """

        # Build Chinese prompt parts
        chinese_parts = []
        english_parts = []

        # 1. Movement
        if movement != "None":
            movement_cn, movement_en = self._get_movement_phrase(movement)
            chinese_parts.append(movement_cn)
            english_parts.append(movement_en)

        # 2. Rotation
        if rotation != "None":
            rotation_cn, rotation_en = self._get_rotation_phrase(rotation, angle)
            chinese_parts.append(rotation_cn)
            english_parts.append(rotation_en)

        # 3. Lens
        if lens != "None":
            lens_cn, lens_en = self._get_lens_phrase(lens)
            chinese_parts.append(lens_cn)
            english_parts.append(lens_en)

        # Check if anything selected
        if not chinese_parts:
            camera_prompt_cn = "将镜头保持不变"
            camera_prompt_en = "Keep camera unchanged"
        else:
            camera_prompt_cn = "将镜头" + "并".join(chinese_parts)
            camera_prompt_en = " and ".join(english_parts)

        # Add scene description if provided
        if scene_description and scene_description.strip():
            scene_desc = scene_description.strip()

            # Build full prompt with scene description
            if output_language == "Chinese Only (Best Performance)":
                prompt = f"{camera_prompt_cn}，{scene_desc}"
                description = f"{camera_prompt_en} → {scene_desc}"
            elif output_language == "English Only":
                prompt = f"{camera_prompt_en}, {scene_desc}."
                description = f"{camera_prompt_en} → {scene_desc}"
            else:  # Both
                prompt = f"{camera_prompt_cn}，{scene_desc} ({camera_prompt_en}, {scene_desc}.)"
                description = f"{camera_prompt_en} → {scene_desc}"
        else:
            # No scene description - original behavior
            if output_language == "Chinese Only (Best Performance)":
                prompt = camera_prompt_cn
                description = camera_prompt_en
            elif output_language == "English Only":
                prompt = camera_prompt_en + "."
                description = camera_prompt_en
            else:  # Both
                prompt = f"{camera_prompt_cn} ({camera_prompt_en}.)"
                description = camera_prompt_en

        return prompt, description

    def _get_movement_phrase(self, movement):
        """
        Get movement phrase in Chinese and English.

        Based on dx8152 Multiple Angles LoRA documentation.
        Returns: (chinese, english)
        """
        movement_map = {
            "Move Forward": ("向前移动", "Move the camera forward"),
            "Move Backward": ("向后移动", "Move the camera backward"),
            "Move Left": ("向左移动", "Move the camera left"),
            "Move Right": ("向右移动", "Move the camera right"),
            "Move Up": ("向上移动", "Move the camera up"),
            "Move Down": ("向下移动", "Move the camera down"),
        }
        return movement_map.get(movement, ("", ""))

    def _get_rotation_phrase(self, rotation, angle):
        """
        Get rotation phrase in Chinese and English.

        For Rotate Left/Right, includes the angle.
        Tested angles: 45, 90, 180 degrees.
        Returns: (chinese, english)
        """
        if rotation == "Rotate Left":
            chinese = f"向左旋转{angle}度"
            english = f"Rotate the camera {angle} degrees to the left"
        elif rotation == "Rotate Right":
            chinese = f"向右旋转{angle}度"
            english = f"Rotate the camera {angle} degrees to the right"
        elif rotation == "Top-Down View":
            chinese = "转为俯视"
            english = "Turn the camera to a top-down view"
        elif rotation == "Low Angle (Upward)":
            # Note: Limited training data for upward angles per community feedback
            chinese = "转为仰视"
            english = "Turn the camera to a low angle view (looking upward)"
        else:
            chinese = ""
            english = ""

        return chinese, english

    def _get_lens_phrase(self, lens):
        """
        Get lens change phrase in Chinese and English.

        Primary support: Wide-Angle, Close-Up (from dx8152 LoRA)
        Experimental: Telephoto, Fisheye, Macro (may require standard Qwen)
        Returns: (chinese, english)
        """
        lens_map = {
            # Primary dx8152 LoRA support
            "Wide-Angle": ("转为广角镜头", "Turn the camera to a wide-angle lens"),
            "Close-Up": ("转为特写镜头", "Turn the camera to a close-up"),

            # Experimental (may work better with standard Qwen)
            "Telephoto": ("转为长焦镜头", "Turn the camera to a telephoto lens"),
            "Fisheye": ("转为鱼眼镜头", "Turn the camera to a fisheye lens"),
            "Macro": ("转为微距镜头", "Turn the camera to a macro lens"),
        }
        return lens_map.get(lens, ("", ""))

    def _get_system_prompt(self, lora_mode):
        """
        Get optimized system prompt for camera movement with dx8152 LoRA.

        Based on user testing and research:
        - Virtual Camera Operator style (92% consistency)
        - Focus on Qwen's behavior, not LoRA technical details
        - Qwen doesn't know what LoRAs are - keep instructions direct
        """

        system_prompts = {
            "multiple_angles":
                "You are a virtual camera operator. Execute camera movements precisely as instructed "
                "while keeping the scene completely unchanged. Preserve all architectural elements, "
                "furniture, objects, textures, colors, and lighting exactly as they are. Your only job "
                "is to change the camera viewpoint - do not redesign, modify, or reimagine the space. "
                "Maintain perfect consistency of all scene elements across different camera angles.",

            "next_scene":
                "Your task is scene transition. When given scene change instructions with the "
                "(Next Scene: ) prefix, generate the new scene while maintaining consistent style, "
                "lighting quality, and atmosphere. Focus on smooth transitions that feel natural "
                "and intentional. Preserve the visual style and quality of the original image."
        }

        return system_prompts.get(lora_mode, system_prompts["multiple_angles"])


# Node registration
NODE_CLASS_MAPPINGS = {
    "ArchAi3D_Qwen_DX8152_Camera_LoRA": ArchAi3D_Qwen_DX8152_Camera_LoRA
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_Qwen_DX8152_Camera_LoRA": "dx8152 Camera LoRA"
}
