"""
Object Focus Camera v4 - Enhanced Edition

Combines v3 features with two major enhancements:
1. Distance-Aware Positioning: Adjusts prompt strength based on camera distance
   - CLOSE: Strong positioning (centering desired) ✓
   - MEDIUM: Gentle positioning (preserve spatial relationships)
   - FAR: Weakest positioning (no repositioning, preserve composition)

2. Environmental Focus Mode: Intentional repositioning for wide-to-tight transitions
   - Standard Mode: Distance-aware positioning (gentle at far distances)
   - Focus Transition Mode: Strong repositioning regardless of distance
   - Perfect for: corner kitchen view → face refrigerator surface

Based on user feedback and Reddit research:
- E:\\Comfy\\help\\03-RESEARCH\\QWEN_PROMPT_WRITING\\REDDIT_RESEARCH\\reddit-camera-prompts-no-lora.md
- E:\\Comfy\\help\\03-RESEARCH\\QWEN_PROMPT_WRITING\\REDDIT_RESEARCH\\reddit-next-scene-perspectives.md

Author: ArchAi3d
Version: 4.0.0 - Enhanced with distance-aware + environmental focus
"""

class ArchAi3D_Object_Focus_Camera_V4:
    """Enhanced Object Focus Camera with distance-aware positioning and environmental focus mode.

    Purpose: Professional object photography with intelligent prompt adaptation.
    Optimized for: Product photography, macro shots, environmental transitions, 360° views.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_object": ("STRING", {
                    "default": "the object",
                    "multiline": False,
                    "tooltip": "What to focus on: 'the watch', 'the ring', 'the refrigerator'"
                }),
                "camera_position": ([
                    "Front View",
                    "Angled View (15°)",
                    "Angled View (30°)",
                    "Angled View (45°)",
                    "Angled View (60°)",
                    "Side View (90°)",
                    "Back View (180°)",
                    "Top-Down View (Bird's Eye)",
                    "Low Angle View (Worm's Eye)",
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
                ], {
                    "default": "Front View",
                    "tooltip": "Camera position relative to the object (includes orbit movements from Reddit research)"
                }),
                "camera_movement": ([
                    "None (Static)",
                    "Dolly In (Zoom Closer)",
                    "Dolly Out (Zoom Away)",
                    "Tilt Up Slightly",
                    "Tilt Down Slightly",
                    "Pan Left",
                    "Pan Right"
                ], {
                    "default": "None (Static)",
                    "tooltip": "Additional camera movement (Reddit-validated: dolly is most consistent)"
                }),
                "camera_distance": ([
                    "Very Close (Macro)",
                    "Close",
                    "Medium",
                    "Far",
                    "Very Far"
                ], {
                    "default": "Close",
                    "tooltip": "Distance from the object - affects prompt strength in Standard mode"
                }),
                "lens_type": ([
                    "Normal Lens (50mm)",
                    "Close-Up Lens",
                    "Macro Lens",
                    "Wide Angle (24mm)",
                    "Ultra Wide (16mm)",
                    "Fisheye",
                    "Telephoto (85mm)",
                    "Telephoto with Bokeh (135mm)",
                    "Tilt-Shift",
                    "Panoramic"
                ], {
                    "default": "Close-Up Lens",
                    "tooltip": "Lens type - backend adds detailed technical descriptions for better AI understanding"
                }),
                "prompt_language": ([
                    "Chinese (Best for dx8152)",
                    "English (Reddit-validated)",
                    "Hybrid (Chinese + English)"
                ], {
                    "default": "Chinese (Best for dx8152)",
                    "tooltip": "Prompt language - Chinese works best with dx8152 LoRAs, English uses Reddit patterns"
                }),
                "focus_transition_mode": ([
                    "Standard (Maintain Position)",
                    "Focus Transition (Reposition to Object)"
                ], {
                    "default": "Standard (Maintain Position)",
                    "tooltip": "Standard: Distance-aware positioning (gentle at far). Focus Transition: Intentional repositioning from wide environmental view to stand directly in front of target object (e.g., kitchen corner → face refrigerator)"
                }),
                "add_detailed_explanation": ([
                    "None (Simple)",
                    "Basic (Short description)",
                    "Detailed (Full perspective explanation)"
                ], {
                    "default": "Basic (Short description)",
                    "tooltip": "Add detailed explanation after base prompt for better AI understanding of camera intent"
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

    def generate_object_focus_prompt(self, target_object, camera_position, camera_movement,
                                     camera_distance, lens_type, prompt_language,
                                     focus_transition_mode, add_detailed_explanation, show_details=""):
        """
        Generate enhanced object focus camera prompt with distance-aware positioning.

        Supports three prompt languages:
        - Chinese: Best for dx8152 LoRAs with "Next Scene: " prefix
        - English: Reddit-validated patterns
        - Hybrid: Chinese structure with English technical terms

        Supports two focus modes:
        - Standard: Distance-aware positioning (gentle at far distances to preserve composition)
        - Focus Transition: Intentional STRONG repositioning for environmental → object workflows

        Supports three explanation levels:
        - None: Simple structured prompt only
        - Basic: Short description of camera effect
        - Detailed: Full perspective and composition explanation (with distance-aware strength)
        """

        # Get lens technical details
        lens_details = self._get_lens_details(lens_type)

        # Build prompt based on selected language
        if "Chinese" in prompt_language:
            prompt = self._build_chinese_prompt(
                target_object, camera_position, camera_movement,
                camera_distance, lens_type, lens_details, show_details,
                add_detailed_explanation, focus_transition_mode
            )
        elif "English" in prompt_language:
            prompt = self._build_english_prompt(
                target_object, camera_position, camera_movement,
                camera_distance, lens_type, lens_details, show_details,
                add_detailed_explanation, focus_transition_mode
            )
        else:  # Hybrid
            prompt = self._build_hybrid_prompt(
                target_object, camera_position, camera_movement,
                camera_distance, lens_type, lens_details, show_details,
                add_detailed_explanation, focus_transition_mode
            )

        # Generate English description for user
        movement_str = "" if camera_movement == "None (Static)" else f" + {camera_movement}"
        mode_indicator = "🎯" if "Focus Transition" in focus_transition_mode else "📍"
        description = f"{mode_indicator} {lens_type} | {camera_position}{movement_str} | {camera_distance} | Object: {target_object}"
        if show_details:
            description += f" | {show_details}"

        # Universal system prompt (works with both LoRAs loaded)
        system_prompt = self._get_enhanced_system_prompt(focus_transition_mode)

        return (prompt, system_prompt, description)

    def _get_lens_details(self, lens_type):
        """Get detailed technical description for each lens type."""
        lens_details = {
            "Normal Lens (50mm)": {
                "chinese": "标准镜头",
                "technical": "standard 50mm lens with natural perspective and balanced field of view",
                "characteristics": "natural perspective, no distortion"
            },
            "Close-Up Lens": {
                "chinese": "特写镜头",
                "technical": "close-up lens with shallow depth of field and enhanced detail capture",
                "characteristics": "shallow depth of field, detail focus"
            },
            "Macro Lens": {
                "chinese": "微距镜头",
                "technical": "macro lens with 1:1 magnification ratio and extreme close-up detail capability",
                "characteristics": "1:1 magnification, extreme detail, very shallow depth of field"
            },
            "Wide Angle (24mm)": {
                "chinese": "广角镜头",
                "technical": "wide-angle 24mm lens with expanded field of view and slight perspective distortion",
                "characteristics": "expanded view, slight distortion at edges"
            },
            "Ultra Wide (16mm)": {
                "chinese": "超广角镜头",
                "technical": "ultra-wide 16mm lens with dramatic perspective and significant barrel distortion",
                "characteristics": "very wide view, dramatic perspective, barrel distortion"
            },
            "Fisheye": {
                "chinese": "鱼眼镜头",
                "technical": "fisheye lens with extreme barrel distortion and 180-degree field of view",
                "characteristics": "180° view, extreme barrel distortion, spherical effect"
            },
            "Telephoto (85mm)": {
                "chinese": "长焦镜头",
                "technical": "telephoto 85mm lens with compressed perspective and subject isolation",
                "characteristics": "compressed perspective, background compression"
            },
            "Telephoto with Bokeh (135mm)": {
                "chinese": "长焦虚化镜头",
                "technical": "telephoto 135mm lens with shallow depth of field and creamy bokeh background blur",
                "characteristics": "strong subject isolation, creamy bokeh, compressed perspective"
            },
            "Tilt-Shift": {
                "chinese": "移轴镜头",
                "technical": "tilt-shift lens with selective focus plane and perspective control",
                "characteristics": "selective focus plane, miniature effect, perspective correction"
            },
            "Panoramic": {
                "chinese": "全景镜头",
                "technical": "panoramic lens with ultra-wide horizontal field of view and minimal distortion",
                "characteristics": "ultra-wide horizontal view, cinematic aspect"
            }
        }
        return lens_details.get(lens_type, lens_details["Close-Up Lens"])

    def _get_distance_category(self, camera_distance):
        """Map 5 distance presets to 3 categories for prompt strength."""
        if camera_distance in ["Very Close (Macro)", "Close"]:
            return "CLOSE"
        elif camera_distance == "Medium":
            return "MEDIUM"
        else:  # "Far", "Very Far"
            return "FAR"

    def _build_chinese_prompt(self, target_object, camera_position, camera_movement,
                              camera_distance, lens_type, lens_details, show_details,
                              add_detailed_explanation, focus_transition_mode):
        """Build Chinese prompt (v1 style) with enhanced details and optional explanations."""

        parts = []

        # 1. Lens change with technical details
        lens_cn = lens_details["chinese"]
        lens_tech = lens_details["characteristics"]
        parts.append(f"将镜头转为{lens_cn}（{lens_tech}）")

        # 2. Camera position
        position_cn = self._get_position_chinese(camera_position)
        parts.append(f"{position_cn}{target_object}")

        # 3. Distance
        distance_cn = self._get_distance_chinese(camera_distance)
        parts.append(f"距离{distance_cn}")

        # 4. Camera movement (if not static)
        if camera_movement != "None (Static)":
            movement_cn = self._get_movement_chinese(camera_movement)
            parts.append(movement_cn)

        # 5. Optional details
        if show_details and show_details.strip():
            parts.append(show_details.strip())

        # Combine with proper Chinese grammar
        prompt_chinese = "，".join(parts)

        # Add "Next Scene: " prefix (works with both LoRAs)
        base_prompt = f"Next Scene: {prompt_chinese}"

        # 6. Add detailed explanation if requested (with distance-aware strength)
        if "None" not in add_detailed_explanation:
            position_explain = self._get_position_explanation(
                camera_position, add_detailed_explanation, camera_distance, focus_transition_mode
            )
            movement_explain = self._get_movement_explanation(camera_movement, add_detailed_explanation)

            # Combine explanations
            explanations = []
            if position_explain:
                explanations.append(position_explain)
            if movement_explain:
                explanations.append(movement_explain)

            if explanations:
                base_prompt += "，" + " ".join(explanations)

        return base_prompt

    def _build_english_prompt(self, target_object, camera_position, camera_movement,
                              camera_distance, lens_type, lens_details, show_details,
                              add_detailed_explanation, focus_transition_mode):
        """Build English prompt (v2 style Reddit-validated) with enhanced details and optional explanations."""

        parts = []

        # 1. Lens technical description
        parts.append(f"Change to {lens_details['technical']}")

        # 2. Camera position (use Reddit patterns for orbit)
        if "Orbit" in camera_position:
            position_prompt = self._get_orbit_english(camera_position, target_object)
            parts.append(position_prompt)
        elif "Bird's Eye" in camera_position:
            parts.append("view from above, bird's eye view")
        elif "Worm's Eye" in camera_position:
            parts.append("view from ground level, worm's eye view")
        else:
            position_en = self._get_position_english(camera_position)
            parts.append(f"camera positioned at {position_en} of {target_object}")

        # 3. Distance
        distance_en = self._get_distance_english(camera_distance)
        parts.append(f"distance {distance_en}")

        # 4. Camera movement (Reddit-validated patterns)
        if camera_movement != "None (Static)":
            if "Dolly In" in camera_movement:
                parts.append("dolly in")
            elif "Dolly Out" in camera_movement:
                parts.append("dolly out")
            elif "Tilt Up" in camera_movement:
                parts.append("tilt the camera up slightly")
            elif "Tilt Down" in camera_movement:
                parts.append("tilt the camera down slightly")
            elif "Pan Left" in camera_movement:
                parts.append("pan the camera left")
            elif "Pan Right" in camera_movement:
                parts.append("pan the camera right")

        # 5. Optional details
        if show_details and show_details.strip():
            parts.append(show_details.strip())

        # Combine with commas
        base_prompt = ", ".join(parts)

        # 6. Add detailed explanation if requested (with distance-aware strength)
        if "None" not in add_detailed_explanation:
            position_explain = self._get_position_explanation(
                camera_position, add_detailed_explanation, camera_distance, focus_transition_mode
            )
            movement_explain = self._get_movement_explanation(camera_movement, add_detailed_explanation)

            # Combine explanations
            explanations = []
            if position_explain:
                explanations.append(position_explain)
            if movement_explain:
                explanations.append(movement_explain)

            if explanations:
                base_prompt += ", " + " ".join(explanations)

        return base_prompt

    def _build_hybrid_prompt(self, target_object, camera_position, camera_movement,
                             camera_distance, lens_type, lens_details, show_details,
                             add_detailed_explanation, focus_transition_mode):
        """Build hybrid prompt (Chinese structure + English technical terms) with optional explanations."""

        parts = []

        # 1. Lens with English technical term
        lens_cn = lens_details["chinese"]
        lens_tech_en = lens_type
        parts.append(f"将镜头转为{lens_cn} ({lens_tech_en})")

        # 2. Position with mixed terms
        if "Orbit" in camera_position:
            # Use English for orbit (Reddit-validated)
            orbit_prompt = self._get_orbit_english(camera_position, target_object)
            parts.append(orbit_prompt)
        else:
            position_cn = self._get_position_chinese(camera_position)
            parts.append(f"{position_cn}{target_object}")

        # 3. Distance in Chinese
        distance_cn = self._get_distance_chinese(camera_distance)
        parts.append(f"距离{distance_cn}")

        # 4. Movement in English (Reddit patterns)
        if camera_movement != "None (Static)":
            if "Dolly In" in camera_movement:
                parts.append("dolly in")
            elif "Dolly Out" in camera_movement:
                parts.append("dolly out")
            elif "Tilt" in camera_movement:
                movement_cn = self._get_movement_chinese(camera_movement)
                parts.append(movement_cn)

        # 5. Optional details
        if show_details and show_details.strip():
            parts.append(show_details.strip())

        # Mix Chinese commas and English commas
        prompt = "，".join(parts[:3])  # Chinese parts
        if len(parts) > 3:
            prompt += "，" + ", ".join(parts[3:])  # English parts

        base_prompt = f"Next Scene: {prompt}"

        # 6. Add detailed explanation if requested (always in English for hybrid mode, with distance-aware strength)
        if "None" not in add_detailed_explanation:
            position_explain = self._get_position_explanation(
                camera_position, add_detailed_explanation, camera_distance, focus_transition_mode
            )
            movement_explain = self._get_movement_explanation(camera_movement, add_detailed_explanation)

            # Combine explanations
            explanations = []
            if position_explain:
                explanations.append(position_explain)
            if movement_explain:
                explanations.append(movement_explain)

            if explanations:
                base_prompt += "，" + " ".join(explanations)

        return base_prompt

    def _get_position_chinese(self, position):
        """Convert camera position to Chinese."""
        position_map = {
            "Front View": "正面查看",
            "Angled View (15°)": "从15度角查看",
            "Angled View (30°)": "从30度角查看",
            "Angled View (45°)": "从45度角查看",
            "Angled View (60°)": "从60度角查看",
            "Side View (90°)": "从侧面查看",
            "Back View (180°)": "从背面查看",
            "Top-Down View (Bird's Eye)": "从俯视角度查看",
            "Low Angle View (Worm's Eye)": "从仰视角度查看",
            # Orbit movements stay in English for Chinese mode too (Reddit patterns work better)
            "Orbit Left 30°": "镜头围绕左侧旋转30度",
            "Orbit Left 45°": "镜头围绕左侧旋转45度",
            "Orbit Left 90°": "镜头围绕左侧旋转90度",
            "Orbit Right 30°": "镜头围绕右侧旋转30度",
            "Orbit Right 45°": "镜头围绕右侧旋转45度",
            "Orbit Right 90°": "镜头围绕右侧旋转90度",
            "Orbit Up 30°": "镜头围绕上方旋转30度",
            "Orbit Up 45°": "镜头围绕上方旋转45度",
            "Orbit Down 30°": "镜头围绕下方旋转30度",
            "Orbit Down 45°": "镜头围绕下方旋转45度",
        }
        return position_map.get(position, "正面查看")

    def _get_position_english(self, position):
        """Convert camera position to English description."""
        position_map = {
            "Front View": "front view",
            "Angled View (15°)": "15-degree angled view",
            "Angled View (30°)": "30-degree angled view",
            "Angled View (45°)": "45-degree angled view",
            "Angled View (60°)": "60-degree angled view",
            "Side View (90°)": "90-degree side view",
            "Back View (180°)": "back view 180 degrees",
            "Top-Down View (Bird's Eye)": "top-down bird's eye view",
            "Low Angle View (Worm's Eye)": "low angle worm's eye view",
        }
        return position_map.get(position, "front view")

    def _get_orbit_english(self, position, target_object):
        """Get Reddit-validated orbit prompt (⭐⭐⭐⭐⭐ most reliable)."""
        if "Orbit Left" in position:
            degrees = position.split("°")[0].split()[-1]
            return f"camera orbit left around {target_object} by {degrees} degrees"
        elif "Orbit Right" in position:
            degrees = position.split("°")[0].split()[-1]
            return f"camera orbit right around {target_object} by {degrees} degrees"
        elif "Orbit Up" in position:
            degrees = position.split("°")[0].split()[-1]
            return f"camera orbit up around {target_object} by {degrees} degrees"
        elif "Orbit Down" in position:
            degrees = position.split("°")[0].split()[-1]
            return f"camera orbit down around {target_object} by {degrees} degrees"
        return f"camera orbit around {target_object}"

    def _get_distance_chinese(self, distance):
        """Convert distance to Chinese."""
        distance_map = {
            "Very Close (Macro)": "很近（几厘米）",
            "Close": "近距离",
            "Medium": "中等距离",
            "Far": "远距离",
            "Very Far": "很远"
        }
        return distance_map.get(distance, "近距离")

    def _get_distance_english(self, distance):
        """Convert distance to English description."""
        distance_map = {
            "Very Close (Macro)": "very close (a few centimeters away)",
            "Close": "close distance",
            "Medium": "medium distance",
            "Far": "far distance",
            "Very Far": "very far distance"
        }
        return distance_map.get(distance, "close distance")

    def _get_movement_chinese(self, movement):
        """Convert camera movement to Chinese."""
        movement_map = {
            "Dolly In (Zoom Closer)": "推近镜头",
            "Dolly Out (Zoom Away)": "拉远镜头",
            "Tilt Up Slightly": "微微向上倾斜",
            "Tilt Down Slightly": "微微向下倾斜",
            "Pan Left": "向左平移",
            "Pan Right": "向右平移"
        }
        return movement_map.get(movement, "")

    def _get_enhanced_system_prompt(self, focus_transition_mode):
        """Get enhanced system prompt based on focus transition mode."""

        if "Focus Transition" in focus_transition_mode:
            # For environmental → object focus transitions (intentional repositioning)
            return (
                "You are a precision camera operator specializing in dynamic scene-to-object transitions. "
                "Execute the requested camera repositioning to move from a wide environmental view to a "
                "focused, centered view of the target subject. Reposition the camera to stand directly in "
                "front, aligned with the surface. Apply the specified lens characteristics "
                "including depth of field, distortion, and perspective. Maintain appearance, "
                "materials, and details while executing the transition from environmental context to "
                "focused composition."
            )
        else:
            # Standard mode (preserve composition, distance-aware)
            return (
                "You are a precision camera operator and lens specialist for professional object photography. "
                "Execute camera positioning and lens characteristics exactly as instructed while keeping "
                "scene composition appropriately preserved based on viewing distance. "
                "For close-up views, precise subject centering is expected. For medium and far views, "
                "preserve spatial relationships and surrounding context. Maintain all details, textures, "
                "colors, materials, and lighting. Pay special attention to lens-specific characteristics "
                "such as depth of field, distortion, and perspective compression. Your job is to change "
                "the camera viewpoint and apply appropriate lens rendering while respecting the compositional "
                "intent for the selected viewing distance."
            )

    def _get_position_explanation(self, camera_position, detail_level, camera_distance, focus_transition_mode):
        """Get detailed explanation for camera position with distance-aware strength.

        Logic:
        - If Focus Transition mode: Use STRONG positioning regardless of distance (centering desired)
        - If Standard mode: Use distance-aware positioning:
            - CLOSE: Strong positioning (centering desired)
            - MEDIUM: Gentle positioning (preserve spatial relationships)
            - FAR: Weakest positioning (no repositioning, preserve composition)
        """

        # Return empty string if no explanation requested
        if "None" in detail_level:
            return ""

        # Determine if we should use strong positioning
        use_strong_positioning = "Focus Transition" in focus_transition_mode

        # If Standard mode, check distance category
        if not use_strong_positioning:
            distance_category = self._get_distance_category(camera_distance)
        else:
            distance_category = "CLOSE"  # Focus Transition always uses CLOSE (strong) positioning

        # Get appropriate explanation based on position, detail level, and distance
        if "Basic" in detail_level:
            return self._get_position_explanation_basic(camera_position)
        else:  # Detailed
            return self._get_position_explanation_detailed(camera_position, distance_category)

    def _get_position_explanation_basic(self, camera_position):
        """Get basic explanation for camera position (distance-independent)."""

        explanations = {
            "Front View": "creating a straightforward front-facing perspective",
            "Angled View (15°)": "creating a subtle angled perspective that reveals slight depth",
            "Angled View (30°)": "creating a moderate angled perspective that shows both front and side",
            "Angled View (45°)": "creating a dynamic angled perspective that emphasizes dimensionality",
            "Angled View (60°)": "creating a steep angled perspective favoring the side view",
            "Side View (90°)": "creating a complete side profile perspective",
            "Back View (180°)": "creating a rear perspective showing the back side",
            "Top-Down View (Bird's Eye)": "creating a bird's eye view perspective from above",
            "Low Angle View (Worm's Eye)": "creating a worm's eye view perspective from ground level that emphasizes vertical height",
            "Orbit Left 30°": "circling 30 degrees left around the subject to reveal a different angle",
            "Orbit Left 45°": "circling 45 degrees left around the subject for side-angled view",
            "Orbit Left 90°": "circling 90 degrees left to complete side profile",
            "Orbit Right 30°": "circling 30 degrees right around the subject to reveal a different angle",
            "Orbit Right 45°": "circling 45 degrees right around the subject for side-angled view",
            "Orbit Right 90°": "circling 90 degrees right to complete side profile",
            "Orbit Up 30°": "circling 30 degrees upward around the subject for elevated perspective",
            "Orbit Up 45°": "circling 45 degrees upward for top-angled perspective",
            "Orbit Down 30°": "circling 30 degrees downward around the subject for lower perspective",
            "Orbit Down 45°": "circling 45 degrees downward for low-angle perspective",
        }

        return explanations.get(camera_position, "")

    def _get_position_explanation_detailed(self, camera_position, distance_category):
        """Get detailed explanation for camera position with distance-aware strength.

        Distance categories:
        - CLOSE: Strong positioning phrases (centering desired)
        - MEDIUM: Gentle positioning phrases (preserve spatial relationships)
        - FAR: Weakest positioning phrases (no repositioning, preserve composition)
        """

        # Create 3-tier explanation database (19 positions × 3 distances)
        explanations = {
            "Front View": {
                "CLOSE": "camera positioned directly in front at eye level, creating a neutral, balanced view that shows the primary face clearly with natural proportions and no distortion",
                "MEDIUM": "camera oriented towards the front, maintaining position in surrounding context while showing the primary face clearly with natural proportions",
                "FAR": "camera viewing the scene from the front direction, keeping all spatial relationships exactly as they are without reframing, showing the environment in its natural context"
            },
            "Angled View (15°)": {
                "CLOSE": "camera positioned at a 15-degree angle from the front, creating a gentle three-dimensional view that reveals a hint's side profile while maintaining focus on the front face",
                "MEDIUM": "camera angled slightly to show both front and side at 15 degrees, maintaining the object within its spatial context while revealing subtle dimensional depth",
                "FAR": "camera viewing from a subtle 15-degree angle without repositioning elements, preserving the environmental composition while showing a hint of dimensional perspective"
            },
            "Angled View (30°)": {
                "CLOSE": "camera positioned at a 30-degree angle from the front, creating a balanced three-dimensional view that equally reveals both the front face and side profile with natural depth perception",
                "MEDIUM": "camera angled at 30 degrees to show front and side profiles, preserving placement within the surrounding space while revealing dimensional form",
                "FAR": "camera viewing from a 30-degree angle maintaining all spatial relationships, showing the scene composition with dimensional perspective without reframing any elements"
            },
            "Angled View (45°)": {
                "CLOSE": "camera positioned at a 45-degree angle from the front, creating a strong three-dimensional view that prominently shows both the front and side faces with dynamic depth and form revelation",
                "MEDIUM": "camera angled at 45 degrees revealing both primary faces, keeping the object within its spatial context while emphasizing dimensional characteristics",
                "FAR": "camera viewing from a 45-degree perspective without repositioning scene elements, maintaining environmental composition while showing angular dimensional depth"
            },
            "Angled View (60°)": {
                "CLOSE": "camera positioned at a 60-degree angle from the front, creating a dramatic three-dimensional view that emphasizes the side profile while still maintaining visibility of the front face",
                "MEDIUM": "camera angled steeply at 60 degrees emphasizing the side profile, preserving spatial context while showing strong dimensional characteristics",
                "FAR": "camera viewing from a steep 60-degree angle maintaining scene composition, showing the environment with pronounced angular perspective without reframing"
            },
            "Side View (90°)": {
                "CLOSE": "camera positioned at a 90-degree side angle perpendicular to the object, creating a pure profile view that shows the complete side silhouette with no front or back elements visible, revealing thickness and side contours",
                "MEDIUM": "camera perpendicular to the object at 90 degrees showing complete side profile, maintaining position within surrounding context while revealing lateral dimensions",
                "FAR": "camera viewing from a perpendicular 90-degree side angle without reframing, showing the scene's lateral relationships and environmental context with pure profile perspective"
            },
            "Back View (180°)": {
                "CLOSE": "camera positioned directly behind the object at 180 degrees, creating a back view that reveals details, textures, and features visible only from the rear angle",
                "MEDIUM": "camera behind the object at 180 degrees showing rear features, preserving spatial context and surrounding elements while revealing back-facing details",
                "FAR": "camera viewing from behind at 180 degrees maintaining all spatial relationships, showing the environment from the rear perspective without repositioning any elements"
            },
            "Top-Down View (Bird's Eye)": {
                "CLOSE": "camera positioned far above looking directly down at the object, creating a bird's eye view perspective that diminishes vertical height and emphasizes the top surface with clear detail of upper features",
                "MEDIUM": "camera elevated above looking down, showing top surface within its surrounding spatial context while maintaining environmental relationships",
                "FAR": "camera viewing from high above without reframing, showing the entire scene layout and spatial organization from bird's eye perspective with all elements preserved"
            },
            "Low Angle View (Worm's Eye)": {
                "CLOSE": "change the view to a vantage point at ground level camera tilted way up towards the object, creating a worm's eye view perspective that exaggerates vertical elements and creates a sense of monumentality and grandeur",
                "MEDIUM": "camera positioned low looking upward at the object, maintaining surrounding spatial context while creating upward perspective that emphasizes vertical presence",
                "FAR": "camera viewing from ground level looking upward without repositioning scene elements, showing the environment with dramatic upward perspective and vertical emphasis"
            },
            "Orbit Left 30°": {
                "CLOSE": "camera orbits in a smooth circular path 30 degrees to the left around the subject, maintaining consistent distance and height while revealing the left side profile, creating a dynamic perspective shift",
                "MEDIUM": "camera circles 30 degrees left maintaining distance, showing the object from a new angle while preserving its relationship to surrounding space",
                "FAR": "camera arcs 30 degrees to the left maintaining all scene relationships, revealing a new perspective without repositioning environmental elements"
            },
            "Orbit Left 45°": {
                "CLOSE": "camera orbits in a smooth circular path 45 degrees to the left around the subject, maintaining consistent distance while transitioning from front to side-front view with dimensional depth",
                "MEDIUM": "camera circles 45 degrees left revealing side-front view, maintaining the object within its spatial context while showing dimensional characteristics",
                "FAR": "camera arcs 45 degrees to the left without reframing scene composition, showing angular perspective while preserving environmental relationships"
            },
            "Orbit Left 90°": {
                "CLOSE": "camera orbits in a smooth circular path 90 degrees to the left around the subject, completing a quarter circle to reveal the full left side profile perpendicular to the starting position",
                "MEDIUM": "camera circles 90 degrees left to perpendicular side view, maintaining surrounding spatial context while revealing complete lateral profile",
                "FAR": "camera arcs 90 degrees to the left maintaining scene composition, showing side perspective without repositioning environmental elements"
            },
            "Orbit Right 30°": {
                "CLOSE": "camera orbits in a smooth circular path 30 degrees to the right around the subject, maintaining consistent distance and height while revealing the right side profile with dynamic perspective shift",
                "MEDIUM": "camera circles 30 degrees right maintaining distance, showing the object from a new angle while preserving spatial relationships",
                "FAR": "camera arcs 30 degrees to the right without reframing, revealing a new perspective while maintaining all environmental relationships"
            },
            "Orbit Right 45°": {
                "CLOSE": "camera orbits in a smooth circular path 45 degrees to the right around the subject, transitioning from front to side-front view with dimensional depth revelation",
                "MEDIUM": "camera circles 45 degrees right showing side-front angle, maintaining the object within its spatial context while revealing dimensional form",
                "FAR": "camera arcs 45 degrees to the right maintaining scene composition, showing angular perspective without repositioning scene elements"
            },
            "Orbit Right 90°": {
                "CLOSE": "camera orbits in a smooth circular path 90 degrees to the right around the subject, completing a quarter circle to reveal the full right side profile perpendicular to the starting position",
                "MEDIUM": "camera circles 90 degrees right to perpendicular view, maintaining surrounding context while showing complete right lateral profile",
                "FAR": "camera arcs 90 degrees to the right without reframing environmental composition, showing side perspective with preserved spatial relationships"
            },
            "Orbit Up 30°": {
                "CLOSE": "camera orbits in a smooth arc 30 degrees upward around the subject, elevating to a higher vantage point creating a gentle downward-looking angle that reveals more of the top surface",
                "MEDIUM": "camera arcs 30 degrees upward maintaining distance, showing elevated perspective while preserving position within surrounding space",
                "FAR": "camera elevates 30 degrees upward without reframing scene composition, showing gentle downward angle while maintaining all environmental relationships"
            },
            "Orbit Up 45°": {
                "CLOSE": "camera orbits in a smooth arc 45 degrees upward around the subject, elevating significantly to create a strong downward-looking angle that emphasizes the top surface and aerial perspective",
                "MEDIUM": "camera arcs 45 degrees upward showing strong elevated perspective, maintaining spatial context while revealing top-down dimensional characteristics",
                "FAR": "camera elevates 45 degrees upward maintaining scene relationships, showing pronounced downward perspective without repositioning environmental elements"
            },
            "Orbit Down 30°": {
                "CLOSE": "camera orbits in a smooth arc 30 degrees downward around the subject, descending to a lower vantage point creating a gentle upward-looking angle that reveals more of the bottom or base",
                "MEDIUM": "camera arcs 30 degrees downward maintaining distance, showing lowered perspective while preserving position within surrounding context",
                "FAR": "camera descends 30 degrees downward without reframing scene composition, showing gentle upward angle while maintaining all spatial relationships"
            },
            "Orbit Down 45°": {
                "CLOSE": "camera orbits in a smooth arc 45 degrees downward around the subject, descending significantly to create a strong upward-looking angle that emphasizes vertical height and monumentality",
                "MEDIUM": "camera arcs 45 degrees downward showing strong low-angle perspective, maintaining spatial context while emphasizing upward vertical characteristics",
                "FAR": "camera descends 45 degrees downward maintaining scene relationships, showing pronounced upward perspective without repositioning environmental elements"
            },
        }

        # Get explanation for this position and distance category
        position_explanations = explanations.get(camera_position, {})
        return position_explanations.get(distance_category, "")

    def _get_movement_explanation(self, camera_movement, detail_level):
        """Get detailed explanation for camera movement based on detail level."""

        # All movement explanations database for 7 movements
        explanations = {
            "Dolly In (Zoom Closer)": {
                "Basic": "gradually moving closer to emphasize details",
                "Detailed": "camera moves smoothly forward on a straight path, gradually filling more of the frame to emphasize intricate details, textures, and fine craftsmanship as the subject grows larger in the frame"
            },
            "Dolly Out (Zoom Away)": {
                "Basic": "gradually moving away to show more context",
                "Detailed": "camera moves smoothly backward on a straight path, gradually revealing more surrounding context and environmental setting as the subject becomes smaller in the frame, providing spatial awareness"
            },
            "Tilt Up Slightly": {
                "Basic": "tilting upward to reveal upper portions",
                "Detailed": "camera tilts slightly upward on its axis while position remains fixed, shifting the view from the middle or lower portions towards the upper sections, creating a gentle upward scanning motion"
            },
            "Tilt Down Slightly": {
                "Basic": "tilting downward to reveal lower portions",
                "Detailed": "camera tilts slightly downward on its axis while position remains fixed, shifting the view from the middle or upper portions towards the lower sections, creating a gentle downward scanning motion"
            },
            "Pan Left": {
                "Basic": "panning left to reveal adjacent areas",
                "Detailed": "camera pans horizontally to the left while position remains fixed, rotating on its vertical axis to sweep the view leftward across the scene, revealing adjacent areas and context to the left side"
            },
            "Pan Right": {
                "Basic": "panning right to reveal adjacent areas",
                "Detailed": "camera pans horizontally to the right while position remains fixed, rotating on its vertical axis to sweep the view rightward across the scene, revealing adjacent areas and context to the right side"
            },
        }

        # Return appropriate explanation level, or empty string if None or movement is static
        if "None" in detail_level or camera_movement == "None (Static)":
            return ""
        elif "Basic" in detail_level:
            return explanations.get(camera_movement, {}).get("Basic", "")
        else:  # Detailed
            return explanations.get(camera_movement, {}).get("Detailed", "")


# Node registration
NODE_CLASS_MAPPINGS = {
    "ArchAi3D_Object_Focus_Camera_V4": ArchAi3D_Object_Focus_Camera_V4
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_Object_Focus_Camera_V4": "📦 Object Focus Camera v4 (Enhanced)"
}
