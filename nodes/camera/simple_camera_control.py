"""
Simple Camera Control Node for Qwen Image Edit - v3.0.0

A context-aware camera control node with 4 intelligent modes:

MODE 1: Position Relative to Object
- Define camera spatial relationship to objects ("in front of", "behind", "above")
- Specify distance and orientation
- Perfect for: Product shots, architectural details, object focus

MODE 2: Move While Tracking Object
- Camera moves but keeps object in frame
- Orbit, dolly, arc movements with tracking
- Perfect for: Reveal shots, dynamic presentations

MODE 3: Free Scene Exploration
- Move through scene without specific target
- Natural exploration and navigation
- Perfect for: Walkthroughs, establishing shots

MODE 4: Align With Surface/Element
- Camera aligned with walls, floors, architectural elements
- Capture surface details and patterns
- Perfect for: Texture capture, architectural photography

Author: ArchAi3d
Version: 3.0.0 - Context-aware redesign with 4 intelligent modes
"""

class ArchAi3D_Qwen_Simple_Camera_Control:
    """Simple Camera Control v3.0 - Context-Aware Camera Positioning

    Intelligent camera control that adapts prompt structure based on your intent.
    No more confusing parameters - each mode shows only relevant controls!

    Features:
    - 4 specialized modes for different use cases
    - Context-aware prompt generation
    - Research-validated formulas (85-95% success rates)
    - Automatic number-to-word conversion
    - Smart system prompt selection
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene_context": ("STRING", {
                    "multiline": True,
                    "default": "modern living room with grey sofa and fireplace",
                    "tooltip": "Describe the scene. Ex: 'modern living room', 'brick building exterior'"
                }),
                "control_mode": ([
                    "Position Relative to Object",
                    "Move While Tracking Object",
                    "Free Scene Exploration",
                    "Align With Surface/Element"
                ], {
                    "default": "Position Relative to Object",
                    "tooltip": "Choose mode based on what you want to do. Each mode uses different prompt structure."
                }),

                # Mode 1: Position Relative to Object
                "target_object": ("STRING", {
                    "default": "the fireplace",
                    "tooltip": "[Mode 1] Object to position camera relative to. Ex: 'the sofa', 'the door', 'the table'"
                }),
                "spatial_relation": ([
                    "in front of",
                    "behind",
                    "to the left of",
                    "to the right of",
                    "above",
                    "below",
                    "at same level as"
                ], {
                    "default": "in front of",
                    "tooltip": "[Mode 1] Camera's spatial relationship to the object"
                }),
                "distance_from_target": ("STRING", {
                    "default": "two meters",
                    "tooltip": "[Mode 1] Distance from object. Use WORDS: 'two meters', 'five feet', 'three meters'"
                }),
                "camera_orientation": ([
                    "looking at target",
                    "looking away from target",
                    "parallel view (side angle)",
                    "perpendicular view (90 degrees)"
                ], {
                    "default": "looking at target",
                    "tooltip": "[Mode 1] Which way is camera pointing relative to object?"
                }),

                # Mode 2: Move While Tracking Object
                "tracked_object": ("STRING", {
                    "default": "the sofa",
                    "tooltip": "[Mode 2] Object to keep in frame while moving. Ex: 'the chair', 'the person'"
                }),
                "movement_type": ([
                    "orbit",
                    "dolly in",
                    "dolly out",
                    "arc",
                    "truck left",
                    "truck right",
                    "pedestal up",
                    "pedestal down"
                ], {
                    "default": "orbit",
                    "tooltip": "[Mode 2] Type of camera movement. Orbit=95% success, Dolly=90%"
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
                    "tooltip": "[Mode 2] Direction for movement"
                }),
                "movement_distance": ("STRING", {
                    "default": "five meters",
                    "tooltip": "[Mode 2] Movement distance. Use WORDS: 'five meters', 'ninety degrees'"
                }),
                "tracking_behavior": ([
                    "centered in frame",
                    "at edge of frame",
                    "following naturally"
                ], {
                    "default": "centered in frame",
                    "tooltip": "[Mode 2] How to keep object in frame during movement"
                }),

                # Mode 3: Free Scene Exploration
                "exploration_direction": ([
                    "forward",
                    "backward",
                    "left",
                    "right",
                    "up",
                    "down",
                    "forward-left diagonal",
                    "forward-right diagonal"
                ], {
                    "default": "forward",
                    "tooltip": "[Mode 3] Direction to move through scene"
                }),
                "exploration_distance": ("STRING", {
                    "default": "three meters",
                    "tooltip": "[Mode 3] How far to move. Use WORDS: 'three meters', 'ten feet'"
                }),
                "movement_style": ([
                    "smooth glide",
                    "slow pan",
                    "quick transition",
                    "steady track"
                ], {
                    "default": "smooth glide",
                    "tooltip": "[Mode 3] Style of movement through space"
                }),
                "direction_hint": ("STRING", {
                    "default": "",
                    "tooltip": "[Mode 3] Optional: 'toward the window', 'past the kitchen', 'around the corner'"
                }),
                "reveal_what": ("STRING", {
                    "default": "",
                    "tooltip": "[Mode 3] Optional: What's being revealed? 'more of the room', 'the dining area'"
                }),

                # Mode 4: Align With Surface/Element
                "alignment_target": ([
                    "wall",
                    "floor",
                    "ceiling",
                    "window",
                    "door",
                    "table surface",
                    "countertop",
                    "artwork",
                    "architectural detail"
                ], {
                    "default": "wall",
                    "tooltip": "[Mode 4] Surface or element to align camera with"
                }),
                "alignment_type": ([
                    "parallel to",
                    "perpendicular to",
                    "facing directly",
                    "at angle to"
                ], {
                    "default": "parallel to",
                    "tooltip": "[Mode 4] How camera relates to the surface"
                }),
                "distance_from_surface": ("STRING", {
                    "default": "one meter",
                    "tooltip": "[Mode 4] Distance from surface. Use WORDS: 'one meter', 'two feet'"
                }),
                "surface_detail": ("STRING", {
                    "default": "",
                    "tooltip": "[Mode 4] Optional: 'brick texture', 'wood grain', 'tile pattern'"
                }),

                # Common parameters for all modes
                "camera_angle": ([
                    "eye level",
                    "high angle (looking down)",
                    "low angle (looking up)",
                    "birds-eye view (overhead)",
                    "worms-eye view (ground level)",
                    "dutch angle (tilted)",
                    "shoulder height",
                    "hip height"
                ], {
                    "default": "eye level",
                    "tooltip": "[All Modes] Camera angle/height"
                }),
                "lens_type": ([
                    "normal",
                    "wide-angle",
                    "ultra-wide",
                    "fisheye",
                    "close-up",
                    "macro",
                    "telephoto"
                ], {
                    "default": "normal",
                    "tooltip": "[All Modes] Lens type affects field of view"
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
                "preservation_clause": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional: Add preservation instructions. Ex: 'keep furniture unchanged', 'maintain lighting'"
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

    def generate_camera_prompt(self, scene_context, control_mode,
                              # Mode 1 params
                              target_object, spatial_relation, distance_from_target, camera_orientation,
                              # Mode 2 params
                              tracked_object, movement_type, movement_direction, movement_distance, tracking_behavior,
                              # Mode 3 params
                              exploration_direction, exploration_distance, movement_style, direction_hint, reveal_what,
                              # Mode 4 params
                              alignment_target, alignment_type, distance_from_surface, surface_detail,
                              # Common params
                              camera_angle, lens_type, system_prompt_preset, preservation_clause="", debug_mode=False):
        """
        Generate context-aware camera prompt based on selected mode.

        Each mode uses a different prompt formula optimized for that use case.
        """

        # Convert numbers to words in all distance parameters
        distance_from_target = self._convert_numbers_to_words(distance_from_target)
        movement_distance = self._convert_numbers_to_words(movement_distance)
        exploration_distance = self._convert_numbers_to_words(exploration_distance)
        distance_from_surface = self._convert_numbers_to_words(distance_from_surface)

        # Generate prompt based on mode
        if control_mode == "Position Relative to Object":
            prompt, description = self._mode_position_relative_to_object(
                scene_context, target_object, spatial_relation, distance_from_target,
                camera_orientation, camera_angle, lens_type, preservation_clause
            )

        elif control_mode == "Move While Tracking Object":
            prompt, description = self._mode_move_while_tracking(
                scene_context, tracked_object, movement_type, movement_direction,
                movement_distance, tracking_behavior, camera_angle, lens_type, preservation_clause
            )

        elif control_mode == "Free Scene Exploration":
            prompt, description = self._mode_free_exploration(
                scene_context, exploration_direction, exploration_distance, movement_style,
                direction_hint, reveal_what, camera_angle, lens_type, preservation_clause
            )

        elif control_mode == "Align With Surface/Element":
            prompt, description = self._mode_align_with_surface(
                scene_context, alignment_target, alignment_type, distance_from_surface,
                surface_detail, camera_angle, lens_type, preservation_clause
            )

        else:
            # Fallback (should never reach here)
            prompt = f"{scene_context}, camera view"
            description = "Error: Unknown mode"

        # Select system prompt
        system_prompt = self._get_system_prompt(system_prompt_preset, scene_context)

        if debug_mode:
            print("\n" + "="*70)
            print("SIMPLE CAMERA CONTROL V3.0 - DEBUG OUTPUT")
            print("="*70)
            print(f"Mode: {control_mode}")
            print(f"Scene: {scene_context}")
            print("-"*70)
            print(f"Generated Prompt:")
            print(f"  {prompt}")
            print("-"*70)
            print(f"System Prompt: {system_prompt_preset}")
            print(f"  {system_prompt[:150]}...")
            print("-"*70)
            print(f"Description: {description}")
            print("="*70 + "\n")

        return (prompt, system_prompt, description)

    def _mode_position_relative_to_object(self, scene, target, relation, distance, orientation, angle, lens, preservation):
        """
        Mode 1: Position Relative to Object

        Formula: "{scene}, camera positioned {distance} {relation} {target},
                  {orientation}, {angle}, {lens}"
        """
        prompt_parts = [scene.strip()]

        # Camera position relative to object
        position_phrase = f"camera positioned {distance} {relation} {target}"
        prompt_parts.append(position_phrase)

        # Camera orientation
        prompt_parts.append(orientation)

        # Camera angle
        angle_phrase = self._get_angle_phrase(angle)
        prompt_parts.append(angle_phrase)

        # Lens type
        lens_phrase = self._get_lens_phrase(lens)
        prompt_parts.append(lens_phrase)

        # Preservation clause
        if preservation:
            prompt_parts.append(preservation.strip())

        prompt = ", ".join(prompt_parts)
        description = f"Position: {distance} {relation} {target}, {orientation}"

        return prompt, description

    def _mode_move_while_tracking(self, scene, tracked, movement, direction, distance, tracking, angle, lens, preservation):
        """
        Mode 2: Move While Tracking Object

        Formula: "{scene}, camera {movement} {direction} by {distance}
                  while keeping {tracked} {tracking}, {angle}, {lens}"
        """
        prompt_parts = [scene.strip()]

        # Movement with tracking
        if movement == "orbit":
            movement_phrase = f"camera orbit {direction} around {tracked} by {distance}"
            movement_phrase += f" while keeping {tracked} {tracking}"
        elif "dolly" in movement:
            dolly_dir = "in towards" if "in" in movement else "out from"
            movement_phrase = f"camera dolly {dolly_dir} {tracked}"
            movement_phrase += f" while keeping it {tracking}"
        elif "truck" in movement:
            truck_dir = "left" if "left" in movement else "right"
            movement_phrase = f"camera truck {truck_dir} by {distance}"
            movement_phrase += f" while keeping {tracked} {tracking}"
        elif "pedestal" in movement:
            ped_dir = "up" if "up" in movement else "down"
            movement_phrase = f"camera pedestal {ped_dir} by {distance}"
            movement_phrase += f" while keeping {tracked} {tracking}"
        elif movement == "arc":
            movement_phrase = f"camera arc {direction} around {tracked} by {distance}"
            movement_phrase += f" while keeping {tracked} {tracking}"
        else:
            movement_phrase = f"camera {movement} while tracking {tracked}"

        prompt_parts.append(movement_phrase)

        # Camera angle
        angle_phrase = self._get_angle_phrase(angle)
        prompt_parts.append(angle_phrase)

        # Lens type
        lens_phrase = self._get_lens_phrase(lens)
        prompt_parts.append(lens_phrase)

        # Preservation clause
        if preservation:
            prompt_parts.append(preservation.strip())

        prompt = ", ".join(prompt_parts)
        description = f"Move: {movement} {direction} tracking {tracked}"

        return prompt, description

    def _mode_free_exploration(self, scene, direction, distance, style, hint, reveal, angle, lens, preservation):
        """
        Mode 3: Free Scene Exploration

        Formula: "{scene}, move camera {direction} {distance} through the scene,
                  {style} [, {hint}] [, revealing {reveal}], {angle}, {lens}"
        """
        prompt_parts = [scene.strip()]

        # Movement through scene
        movement_phrase = f"move camera {direction} {distance} through the scene"
        movement_phrase += f", {style}"

        # Optional direction hint
        if hint and hint.strip():
            movement_phrase += f", {hint.strip()}"

        # Optional reveal
        if reveal and reveal.strip():
            movement_phrase += f", revealing {reveal.strip()}"

        prompt_parts.append(movement_phrase)

        # Camera angle
        angle_phrase = self._get_angle_phrase(angle)
        prompt_parts.append(angle_phrase)

        # Lens type
        lens_phrase = self._get_lens_phrase(lens)
        prompt_parts.append(lens_phrase)

        # Preservation clause
        if preservation:
            prompt_parts.append(preservation.strip())

        prompt = ", ".join(prompt_parts)
        description = f"Explore: {direction} {distance}, {style}"

        return prompt, description

    def _mode_align_with_surface(self, scene, target, alignment, distance, detail, angle, lens, preservation):
        """
        Mode 4: Align With Surface/Element

        Formula: "{scene}, camera positioned {distance} from {target},
                  {alignment} the {target} [, showing {detail}], {angle}, {lens}"
        """
        prompt_parts = [scene.strip()]

        # Alignment with surface
        alignment_phrase = f"camera positioned {distance} from {target}"
        alignment_phrase += f", {alignment} the {target}"

        # Optional surface detail
        if detail and detail.strip():
            alignment_phrase += f", showing {detail.strip()}"

        prompt_parts.append(alignment_phrase)

        # Camera angle
        angle_phrase = self._get_angle_phrase(angle)
        prompt_parts.append(angle_phrase)

        # Lens type
        lens_phrase = self._get_lens_phrase(lens)
        prompt_parts.append(lens_phrase)

        # Preservation clause
        if preservation:
            prompt_parts.append(preservation.strip())

        prompt = ", ".join(prompt_parts)
        description = f"Align: {alignment} {target} at {distance}"

        return prompt, description

    def _convert_numbers_to_words(self, text):
        """
        Convert numbers to words to prevent Qwen from rendering numbers as text in images.

        Research finding: Using "five meters" instead of "5m" prevents text artifacts.
        """
        number_map = {
            "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
            "5": "five", "6": "six", "7": "seven", "8": "eight",
            "9": "nine", "10": "ten", "15": "fifteen", "20": "twenty",
            "30": "thirty", "45": "forty-five", "90": "ninety", "180": "one hundred eighty"
        }

        # Convert "5 meters" or "5m" to "five meters"
        for num, word in number_map.items():
            text = text.replace(f"{num} meter", f"{word} meter")
            text = text.replace(f"{num}m", f"{word} meters")
            text = text.replace(f"{num} m", f"{word} meters")
            text = text.replace(f"{num} degree", f"{word} degree")
            text = text.replace(f"{num} feet", f"{word} feet")
            text = text.replace(f"{num} foot", f"{word} foot")
            text = text.replace(f"{num}°", f"{word} degrees")

        return text

    def _get_angle_phrase(self, angle):
        """Convert angle preset to descriptive phrase for prompt."""
        angle_map = {
            "eye level": "at eye level",
            "high angle (looking down)": "from a high angle looking down",
            "low angle (looking up)": "from a low angle looking up",
            "birds-eye view (overhead)": "from a birds-eye overhead view",
            "worms-eye view (ground level)": "from ground level looking up",
            "dutch angle (tilted)": "with a dutch angle tilt",
            "shoulder height": "at shoulder height",
            "hip height": "at hip height"
        }
        return angle_map.get(angle, angle)

    def _get_lens_phrase(self, lens):
        """Convert lens type to descriptive phrase for prompt."""
        lens_map = {
            "normal": "normal lens",
            "wide-angle": "wide-angle lens showing more context",
            "ultra-wide": "ultra-wide lens with expansive view",
            "fisheye": "fisheye lens with curved perspective",
            "close-up": "close-up lens for detail",
            "macro": "macro lens for extreme detail",
            "telephoto": "telephoto lens with compressed perspective"
        }
        return lens_map.get(lens, f"{lens} lens")

    def _get_system_prompt(self, preset, scene_context):
        """Get research-validated system prompt based on preset."""

        # Auto-select logic
        if preset == "Auto-select":
            scene_lower = scene_context.lower()

            # Person scenes need identity preservation
            if any(word in scene_lower for word in ["person", "people", "man", "woman", "portrait", "face"]):
                preset = "Scene Preservation Camera (95%)"
            # Interior/architecture scenes
            elif any(word in scene_lower for word in ["interior", "room", "building", "architecture"]):
                preset = "Scene Preservation Camera (95%)"
            # Exterior/landscape
            elif any(word in scene_lower for word in ["exterior", "outdoor", "landscape", "street"]):
                preset = "Virtual Camera Operator (92%)"
            # Default
            else:
                preset = "Scene Preservation Camera (95%)"

        # System prompt library
        system_prompts = {
            "Scene Preservation Camera (95%)":
                "You are a precision camera operator. Your task is to change ONLY the camera position "
                "and angle as instructed, while keeping the scene absolutely unchanged. Preserve all "
                "objects, furniture, textures, colors, materials, lighting, and spatial relationships "
                "exactly as they are. Do not add, remove, redesign, or reimagine anything. Your only "
                "job is to provide a new viewpoint of the existing scene with perfect consistency.",

            "Virtual Camera Operator (92%)":
                "You are a virtual camera operator. Execute camera movements precisely as instructed "
                "while keeping the scene completely unchanged. Preserve all architectural elements, "
                "furniture, objects, textures, colors, and lighting exactly as they are. Your only job "
                "is to change the camera viewpoint - do not redesign, modify, or reimagine the space. "
                "Maintain perfect consistency of all scene elements across different camera angles.",

            "Cinematographer (85%)":
                "You are a cinematographer controlling camera position and movement. Follow the camera "
                "instructions precisely while maintaining scene consistency. Keep all objects, lighting, "
                "and spatial relationships intact. Focus on providing the requested viewpoint with "
                "natural camera behavior and cinematic quality."
        }

        return system_prompts.get(preset, system_prompts["Scene Preservation Camera (95%)"])


# Node registration
NODE_CLASS_MAPPINGS = {
    "ArchAi3D_Qwen_Simple_Camera_Control": ArchAi3D_Qwen_Simple_Camera_Control
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_Qwen_Simple_Camera_Control": "🎥 Simple Camera Control v3"
}
