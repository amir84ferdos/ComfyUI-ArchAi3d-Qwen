"""
Simple Camera Control Node for Qwen Image Edit

A user-friendly camera control node that allows you to specify:
- Camera position in natural language
- Look-at target (specific object or area)
- Camera angle (eye-level, high, low, birds-eye, etc.)
- Lens type (normal, wide-angle, close-up, macro, etc.)
- Camera movement type (orbit, dolly, pan, tilt, etc.)
- LoRA support (dx8152 Multiple Angles, Next Scene)

Generates research-validated prompts using proven formulas from community testing.

Author: ArchAi3d
Version: 2.0.0 - Added LoRA support and camera movement types
"""

class ArchAi3D_Qwen_Simple_Camera_Control:
    """Simple Camera Control for Qwen Image Edit

    Control camera position, look-at target, angle, lens type, and movement
    with simple text inputs. Generates research-validated prompts
    with 85-95% consistency ratings.

    Features:
    - Natural language position inputs
    - Automatic number-to-word conversion (prevents text rendering)
    - Research-validated system prompts
    - Flexible look-at targeting
    - 10 camera angles + 7 lens types
    - 10+ camera movement types (orbit, dolly, pan, tilt, etc.)
    - dx8152 LoRA support (Multiple Angles + Next Scene)
    - Automatic Chinese/English prompt generation for LoRAs
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene_context": ("STRING", {
                    "multiline": True,
                    "default": "modern living room with grey sofa",
                    "tooltip": "Description of the scene. Ex: 'modern living room', 'brick building exterior'"
                }),
                "camera_movement_type": ([
                    "none (position only)",
                    "orbit_around",
                    "dolly_in_out",
                    "vantage_point",
                    "pan_left_right",
                    "tilt_up_down",
                    "rotate_90_degrees",
                    "camera_tilt",
                    "truck_lateral",
                    "pedestal_vertical",
                    "combined_movement"
                ], {
                    "default": "none (position only)",
                    "tooltip": "Camera movement type. Orbit (95% success), Dolly (90%), Vantage (85%). Use 'none' for static positioning."
                }),
                "movement_direction": ([
                    "left",
                    "right",
                    "forward",
                    "backward",
                    "up",
                    "down",
                    "clockwise",
                    "counterclockwise"
                ], {
                    "default": "right",
                    "tooltip": "Direction for movement (if movement type is selected)"
                }),
                "movement_distance": ("STRING", {
                    "default": "five meters",
                    "tooltip": "Distance/amount for movement. Ex: 'five meters', 'ninety degrees', 'three meters'. Use WORDS not numbers!"
                }),
                "look_at_target": ("STRING", {
                    "default": "the sofa",
                    "tooltip": "What camera focuses on. Ex: 'the sofa', 'the main entrance', 'the wooden chair'"
                }),
                "camera_angle": ([
                    "eye_level",
                    "high_angle",
                    "low_angle",
                    "birds_eye",
                    "worms_eye",
                    "front_view",
                    "side_view",
                    "corner_view",
                    "top_down",
                    "aerial"
                ], {
                    "default": "eye_level",
                    "tooltip": "Viewing angle. Eye-level is most natural, birds-eye is from above, worms-eye is from ground up"
                }),
                "lens_type": ([
                    "normal",
                    "wide_angle",
                    "ultra_wide",
                    "fisheye",
                    "close_up",
                    "macro",
                    "telephoto"
                ], {
                    "default": "normal",
                    "tooltip": "Lens type. Normal is standard, wide-angle shows more context, close-up for details"
                }),
                "lora_mode": ([
                    "None (Standard Qwen)",
                    "dx8152 Multiple Angles",
                    "dx8152 Next Scene"
                ], {
                    "default": "None (Standard Qwen)",
                    "tooltip": "LoRA support. dx8152 Multiple Angles = camera movements with consistency. Next Scene = scene changes."
                }),
                "system_prompt_preset": ([
                    "Scene Preservation Camera (95%)",
                    "Virtual Camera Operator (92%)",
                    "Cinematographer (85%)",
                    "Auto-select"
                ], {
                    "default": "Scene Preservation Camera (95%)",
                    "tooltip": "Research-validated system prompts. 95% = highest consistency rating"
                }),
            },
            "optional": {
                "preservation_clause": ("STRING", {
                    "default": "keep all objects, materials, and lighting identical",
                    "tooltip": "What to preserve in the scene. Critical for consistency!"
                }),
                "debug_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print generated prompts to console for debugging"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "system_prompt", "description")
    FUNCTION = "generate_camera_prompt"
    CATEGORY = "ArchAi3d/Qwen/Camera"

    def generate_camera_prompt(self, scene_context, camera_movement_type,
                              movement_direction, movement_distance,
                              look_at_target, camera_angle, lens_type,
                              lora_mode, system_prompt_preset,
                              preservation_clause="", debug_mode=False):
        """
        Generate camera control prompt using research-validated formulas.

        Supports:
        - Standard Qwen prompts (research-validated formulas)
        - dx8152 Multiple Angles LoRA (Chinese + English)
        - dx8152 Next Scene LoRA (Chinese trigger)
        - 10+ camera movement types with 85-95% success rates

        Based on research from:
        - qwen-prompts-consolidated.md (7 core functions, reliability rated)
        - qwen-dx8152-lora-collection.md (LoRA prompts)
        - qwen-camera-rotation-90-degrees.md (Movement patterns)
        """

        # Convert numbers to words (prevents text rendering in images)
        movement_distance = self._convert_numbers_to_words(movement_distance)

        # Check if using LoRA mode
        if lora_mode == "dx8152 Multiple Angles":
            prompt = self._generate_dx8152_multiple_angles_prompt(
                camera_movement_type, movement_direction, movement_distance, lens_type
            )
            system_prompt = self._get_system_prompt("Virtual Camera Operator (92%)", scene_context)
            description = f"dx8152 Multiple Angles: {camera_movement_type} {movement_direction} {movement_distance}"

        elif lora_mode == "dx8152 Next Scene":
            prompt = self._generate_dx8152_next_scene_prompt(scene_context)
            system_prompt = self._get_system_prompt("Scene Preservation Camera (95%)", scene_context)
            description = f"dx8152 Next Scene: {scene_context}"

        else:
            # Standard Qwen mode (original formula)
            prompt_parts = []

            # 1. Scene context (essential for consistency)
            prompt_parts.append(scene_context.strip())

            # 2. Camera movement or position
            if camera_movement_type == "none (position only)":
                # Static positioning (original behavior)
                movement_phrase = f"camera positioned looking at {look_at_target}"
            else:
                # Dynamic movement with research-validated formulas
                movement_phrase = self._get_movement_phrase(
                    camera_movement_type, movement_direction,
                    movement_distance, look_at_target
                )
            prompt_parts.append(movement_phrase)

            # 3. Camera angle (convert to descriptive phrase)
            angle_phrase = self._get_angle_phrase(camera_angle)
            prompt_parts.append(angle_phrase)

            # 4. Lens type (convert to descriptive phrase)
            lens_phrase = self._get_lens_phrase(lens_type)
            prompt_parts.append(lens_phrase)

            # 5. Preservation clause (CRITICAL for scene consistency)
            if preservation_clause:
                prompt_parts.append(preservation_clause)

            # Join with commas (research-validated format)
            prompt = ", ".join(prompt_parts)

            # Select system prompt based on preset
            system_prompt = self._get_system_prompt(system_prompt_preset, scene_context)

            # Create human-readable description
            description = f"{camera_movement_type}: {movement_direction} {movement_distance}, {camera_angle}, {lens_type} lens"

        if debug_mode:
            print("\n" + "="*60)
            print("SIMPLE CAMERA CONTROL V2 - DEBUG OUTPUT")
            print("="*60)
            print(f"LoRA Mode: {lora_mode}")
            print(f"Scene: {scene_context}")
            print(f"Movement Type: {camera_movement_type}")
            print(f"Direction: {movement_direction}")
            print(f"Distance: {movement_distance}")
            print(f"Target: {look_at_target}")
            print(f"Angle: {camera_angle}")
            print(f"Lens: {lens_type}")
            print("-"*60)
            print(f"Generated Prompt:")
            print(f"  {prompt}")
            print("-"*60)
            print(f"System Prompt: {system_prompt_preset}")
            print(f"  {system_prompt[:150]}...")
            print("-"*60)
            print(f"Description: {description}")
            print("="*60 + "\n")

        return (prompt, system_prompt, description)

    def _get_movement_phrase(self, movement_type, direction, distance, target):
        """
        Generate camera movement phrase using research-validated formulas.

        Based on research:
        - Orbit: 95-100% success rate (qwen-prompts-consolidated.md)
        - Dolly: 90-95% success rate
        - Vantage Point: 85-90% success rate
        - Pan/Tilt: 80-85% with LoRA, 70% without

        Returns properly formatted movement instruction.
        """
        if movement_type == "orbit_around":
            # Formula: "camera orbit [DIRECTION] around [TARGET] by [DISTANCE]"
            return f"camera orbit {direction} around {target} by {distance}"

        elif movement_type == "dolly_in_out":
            # Formula: "change the view dolly [IN/OUT] towards/from the [TARGET]"
            dolly_direction = "in towards" if direction in ["forward", "in"] else "out from"
            return f"change the view dolly {dolly_direction} {target}"

        elif movement_type == "vantage_point":
            # Formula: "change the view to a new vantage point [DISTANCE] to the [DIRECTION]"
            return f"change the view to a new vantage point {distance} to the {direction}"

        elif movement_type == "pan_left_right":
            # Formula: "move the camera [LEFT/RIGHT]"
            return f"move the camera {direction}"

        elif movement_type == "tilt_up_down":
            # Formula: "move the camera [UP/DOWN]"
            return f"move the camera {direction}"

        elif movement_type == "rotate_90_degrees":
            # Formula: "Turn the camera 90 degrees to the [DIRECTION]"
            return f"Turn the camera ninety degrees to the {direction}"

        elif movement_type == "camera_tilt":
            # Formula: "camera tilted [DIRECTION]"
            return f"camera tilted {direction}"

        elif movement_type == "truck_lateral":
            # Formula: "camera truck [LEFT/RIGHT] by [DISTANCE]"
            return f"camera truck {direction} by {distance}"

        elif movement_type == "pedestal_vertical":
            # Formula: "camera pedestal [UP/DOWN] by [DISTANCE]"
            return f"camera pedestal {direction} by {distance}"

        elif movement_type == "combined_movement":
            # Formula: "change the view and move [DISTANCE] [DIRECTION]"
            return f"change the view and move {distance} {direction}"

        else:
            # Fallback: basic position
            return f"camera positioned {distance} to the {direction} looking at {target}"

    def _generate_dx8152_multiple_angles_prompt(self, movement_type, direction, distance, lens_type):
        """
        Generate dx8152 Multiple Angles LoRA prompt (Chinese + English).

        Based on research: qwen-dx8152-lora-collection.md
        Complete prompt list with Chinese + English translations.

        Success rate: 85-95% with LoRA (improved consistency)
        """
        # Map movement types to dx8152 LoRA prompts (Chinese + English)
        lora_prompts = {
            # Camera movements (pan)
            ("pan_left_right", "left"): "将镜头向左移动 (Move the camera left.)",
            ("pan_left_right", "right"): "将镜头向右移动 (Move the camera right.)",
            ("tilt_up_down", "down"): "将镜头向下移动 (Move the camera down.)",

            # Rotations
            ("rotate_90_degrees", "left"): "将镜头向左旋转90度 (Rotate the camera 90 degrees to the left.)",
            ("rotate_90_degrees", "right"): "将镜头向右旋转90度 (Rotate the camera 90 degrees to the right.)",

            # Views
            ("camera_tilt", "down"): "将镜头转为俯视 (Turn the camera to a top-down view.)",

            # Lens changes
            ("lens", "wide_angle"): "将镜头转为广角镜头 (Turn the camera to a wide-angle lens.)",
            ("lens", "close_up"): "将镜头转为特写镜头 (Turn the camera to a close-up.)",
        }

        # Try to match movement + direction
        key = (movement_type, direction)
        if key in lora_prompts:
            return lora_prompts[key]

        # Try lens type match
        lens_key = ("lens", lens_type)
        if lens_key in lora_prompts:
            return lora_prompts[lens_key]

        # Fallback: generic movement with Chinese format
        direction_cn = {
            "left": "左", "right": "右", "up": "上", "down": "下",
            "forward": "前", "backward": "后"
        }.get(direction, direction)

        return f"将镜头向{direction_cn}移动 (Move the camera {direction}.)"

    def _generate_dx8152_next_scene_prompt(self, scene_context):
        """
        Generate dx8152 Next Scene LoRA prompt.

        This LoRA is for scene changes, not camera movements.
        Currently returns scene context - user can customize for scene changes.
        """
        # For next scene LoRA, just use the scene context
        # User can modify this for specific scene change prompts
        return scene_context

    def _convert_numbers_to_words(self, text):
        """
        Convert numbers to words to prevent Qwen from rendering numbers as text in images.

        Research finding: Using "five meters" instead of "5m" prevents text artifacts.
        Source: qwen-prompts-consolidated.md - Written numbers pattern
        """
        number_map = {
            "1": "one", "2": "two", "3": "three", "4": "four",
            "5": "five", "6": "six", "7": "seven", "8": "eight",
            "9": "nine", "10": "ten", "90": "ninety", "45": "forty-five",
            "180": "one hundred eighty"
        }

        # Convert "5 meters" or "5m" to "five meters"
        for num, word in number_map.items():
            text = text.replace(f"{num} meter", f"{word} meter")
            text = text.replace(f"{num}m", f"{word} meters")
            text = text.replace(f"{num} m", f"{word} meters")
            text = text.replace(f"{num} degree", f"{word} degree")

        return text

    def _get_angle_phrase(self, angle):
        """
        Convert angle preset to descriptive phrase for prompt.

        Based on research from Camera View Selector presets
        and community-tested patterns.
        """
        angle_phrases = {
            "eye_level": "eye level view",
            "high_angle": "high angle view looking down",
            "low_angle": "low angle view looking up",
            "birds_eye": "bird's eye view from directly above",
            "worms_eye": "worm's eye view from ground level looking up",
            "front_view": "front view facing directly",
            "side_view": "side view showing profile",
            "corner_view": "corner view showing two sides",
            "top_down": "top-down overhead view",
            "aerial": "aerial view from elevated perspective"
        }
        return angle_phrases.get(angle, f"{angle} view")

    def _get_lens_phrase(self, lens):
        """
        Convert lens type to descriptive phrase for prompt.

        Based on FOV control research and dx8152 LoRA prompts.
        """
        lens_phrases = {
            "normal": "normal lens",
            "wide_angle": "wide angle lens",
            "ultra_wide": "ultra wide angle lens",
            "fisheye": "fisheye lens with 180-degree view",
            "close_up": "close-up lens with tight framing",
            "macro": "macro lens for extreme detail",
            "telephoto": "telephoto lens with compressed perspective"
        }
        return lens_phrases.get(lens, f"{lens} lens")

    def _get_system_prompt(self, preset, scene_context):
        """
        Get research-validated system prompt.

        All prompts tested and rated by community:
        - Scene Preservation Camera: 95% consistency (TOP RATED)
        - Virtual Camera Operator: 92% consistency
        - Cinematographer: 85% consistency

        Source: qwen-system-prompts-library.md
        """

        # System prompt library (research-validated)
        prompts = {
            "Scene Preservation Camera (95%)":
                "Your task is camera repositioning ONLY. When given camera movement instructions "
                "(orbit, vantage point change, tilt, etc.), generate the new view while keeping "
                "100% of the scene content identical: same objects, same materials, same colors, "
                "same lighting, same atmosphere. Think of it as moving a camera in a frozen, "
                "unchanging 3D environment.",

            "Virtual Camera Operator (92%)":
                "You are a virtual camera operator. Execute camera movements precisely as instructed "
                "while keeping the scene completely unchanged. Preserve all architectural elements, "
                "furniture, objects, textures, colors, and lighting exactly as they are. Your job is "
                "to move the camera, not to redesign the space.",

            "Cinematographer (85%)":
                "You are a professional cinematographer. When given camera movement instructions, "
                "generate a new viewpoint of the same scene while preserving all objects, materials, "
                "lighting, and atmosphere. Change ONLY the camera position and angle as instructed. "
                "Do not add, remove, or alter any scene content. Maintain spatial consistency and "
                "realistic perspective."
        }

        # Auto-select best system prompt based on scene context
        if preset == "Auto-select":
            scene_lower = scene_context.lower()

            # Interior scenes: Scene Preservation Camera (highest consistency)
            if any(word in scene_lower for word in ["interior", "room", "living", "bedroom", "kitchen", "bathroom"]):
                selected_preset = "Scene Preservation Camera (95%)"

            # Exterior/architectural: Virtual Camera Operator (best for buildings)
            elif any(word in scene_lower for word in ["building", "exterior", "facade", "architecture", "house"]):
                selected_preset = "Virtual Camera Operator (92%)"

            # Objects/products: Virtual Camera Operator (best for precise control)
            elif any(word in scene_lower for word in ["chair", "table", "product", "object", "furniture"]):
                selected_preset = "Virtual Camera Operator (92%)"

            # Default: Cinematographer (general purpose)
            else:
                selected_preset = "Cinematographer (85%)"

            return prompts[selected_preset]

        # Return selected preset or default
        return prompts.get(preset, prompts["Scene Preservation Camera (95%)"])


# Node registration
NODE_CLASS_MAPPINGS = {
    "ArchAi3D_Qwen_Simple_Camera_Control": ArchAi3D_Qwen_Simple_Camera_Control
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_Qwen_Simple_Camera_Control": "Simple Camera Control (Qwen)"
}
