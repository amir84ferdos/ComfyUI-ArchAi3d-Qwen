"""
Object Focus Camera v3 - Ultimate Merged Edition

Combines the best features from v1 (Chinese prompts) and v2 (Reddit-validated patterns)
with significant enhancements:
- Expanded camera positions (20+ options including orbit movements)
- Advanced lens types (10 options with detailed technical descriptions)
- Camera movements (dolly, tilt, pan)
- Multi-language support (Chinese/English/Hybrid)
- Enhanced prompt generation with detailed explanations
- Universal compatibility (works with both Next Scene + Multiple Angles LoRAs)

Based on research from:
- E:\\Comfy\\help\\03-RESEARCH\\QWEN_PROMPT_WRITING\\REDDIT_RESEARCH\\reddit-camera-prompts-no-lora.md
- E:\\Comfy\\help\\03-RESEARCH\\QWEN_PROMPT_WRITING\\REDDIT_RESEARCH\\reddit-next-scene-perspectives.md

Author: ArchAi3d
Version: 3.0.0 - Ultimate merged edition
"""

class ArchAi3D_Object_Focus_Camera_V3:
    """Ultimate Object Focus Camera - Merged best features from v1 and v2.

    Purpose: Professional object photography with maximum control and flexibility.
    Optimized for: Product photography, macro shots, detail captures, 360° views.
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
                    "tooltip": "Distance from the object"
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
                                     add_detailed_explanation, show_details=""):
        """
        Generate enhanced object focus camera prompt with detailed technical descriptions.

        Supports three prompt languages:
        - Chinese: Best for dx8152 LoRAs with "Next Scene: " prefix
        - English: Reddit-validated patterns
        - Hybrid: Chinese structure with English technical terms

        Supports three explanation levels:
        - None: Simple structured prompt only
        - Basic: Short description of camera effect
        - Detailed: Full perspective and composition explanation
        """

        # Get lens technical details
        lens_details = self._get_lens_details(lens_type)

        # Build prompt based on selected language
        if "Chinese" in prompt_language:
            prompt = self._build_chinese_prompt(
                target_object, camera_position, camera_movement,
                camera_distance, lens_type, lens_details, show_details,
                add_detailed_explanation
            )
        elif "English" in prompt_language:
            prompt = self._build_english_prompt(
                target_object, camera_position, camera_movement,
                camera_distance, lens_type, lens_details, show_details,
                add_detailed_explanation
            )
        else:  # Hybrid
            prompt = self._build_hybrid_prompt(
                target_object, camera_position, camera_movement,
                camera_distance, lens_type, lens_details, show_details,
                add_detailed_explanation
            )

        # Generate English description for user
        movement_str = "" if camera_movement == "None (Static)" else f" + {camera_movement}"
        description = f"{lens_type} | {camera_position}{movement_str} | {camera_distance} | Object: {target_object}"
        if show_details:
            description += f" | {show_details}"

        # Universal system prompt (works with both LoRAs loaded)
        system_prompt = self._get_enhanced_system_prompt()

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

    def _build_chinese_prompt(self, target_object, camera_position, camera_movement,
                              camera_distance, lens_type, lens_details, show_details,
                              add_detailed_explanation):
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

        # 6. Add detailed explanation if requested
        if "None" not in add_detailed_explanation:
            position_explain = self._get_position_explanation(camera_position, add_detailed_explanation)
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
                              add_detailed_explanation):
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

        # 6. Add detailed explanation if requested
        if "None" not in add_detailed_explanation:
            position_explain = self._get_position_explanation(camera_position, add_detailed_explanation)
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
                             add_detailed_explanation):
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

        # 6. Add detailed explanation if requested (always in English for hybrid mode)
        if "None" not in add_detailed_explanation:
            position_explain = self._get_position_explanation(camera_position, add_detailed_explanation)
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

    def _get_enhanced_system_prompt(self):
        """Get enhanced system prompt that works with both LoRAs loaded simultaneously."""
        return (
            "You are a precision camera operator and lens specialist for professional object photography. "
            "Execute camera positioning and lens characteristics exactly as instructed while keeping "
            "the object and scene completely unchanged. Preserve all details, textures, colors, "
            "materials, and lighting exactly as they are. Pay special attention to the lens-specific "
            "characteristics such as depth of field, distortion, and perspective compression. "
            "Your only job is to change the camera viewpoint and apply the appropriate lens rendering - "
            "do not modify, redesign, or reimagine anything in the scene. Maintain perfect object "
            "preservation while executing the requested camera and lens changes."
        )

    def _get_position_explanation(self, camera_position, detail_level):
        """Get detailed explanation for camera position based on detail level."""

        # All explanations database for 19 positions
        explanations = {
            "Front View": {
                "Basic": "creating a straightforward front-facing perspective",
                "Detailed": "camera positioned directly in front of the object at eye level, creating a neutral, balanced view that shows the primary face of the object clearly with natural proportions and no distortion"
            },
            "Angled View (15°)": {
                "Basic": "creating a subtle angled perspective that reveals slight depth",
                "Detailed": "camera positioned at a 15-degree angle from the front, creating a gentle three-dimensional view that reveals a hint of the object's side profile while maintaining focus on the front face"
            },
            "Angled View (30°)": {
                "Basic": "creating a moderate angled perspective that shows both front and side",
                "Detailed": "camera positioned at a 30-degree angle from the front, creating a balanced three-dimensional view that equally reveals both the front face and side profile of the object with natural depth perception"
            },
            "Angled View (45°)": {
                "Basic": "creating a dynamic angled perspective that emphasizes dimensionality",
                "Detailed": "camera positioned at a 45-degree angle from the front, creating a strong three-dimensional view that prominently shows both the front and side faces of the object with dynamic depth and form revelation"
            },
            "Angled View (60°)": {
                "Basic": "creating a steep angled perspective favoring the side view",
                "Detailed": "camera positioned at a 60-degree angle from the front, creating a dramatic three-dimensional view that emphasizes the side profile while still maintaining visibility of the front face"
            },
            "Side View (90°)": {
                "Basic": "creating a complete side profile perspective",
                "Detailed": "camera positioned at a 90-degree side angle perpendicular to the object, creating a pure profile view that shows the complete side silhouette with no front or back elements visible, revealing the object's thickness and side contours"
            },
            "Back View (180°)": {
                "Basic": "creating a rear perspective showing the back side",
                "Detailed": "camera positioned directly behind the object at 180 degrees, creating a back view that reveals details, textures, and features visible only from the rear angle"
            },
            "Top-Down View (Bird's Eye)": {
                "Basic": "creating a bird's eye view perspective from above",
                "Detailed": "camera positioned far above looking directly down at the object, creating a bird's eye view perspective that diminishes vertical height and emphasizes the top surface, surrounding context, and spatial relationships, creating a sense of overview and layout clarity"
            },
            "Low Angle View (Worm's Eye)": {
                "Basic": "creating a worm's eye view perspective from ground level that emphasizes vertical height",
                "Detailed": "change the view to a vantage point at ground level camera tilted way up towards the object, creating a worm's eye view perspective that exaggerates vertical elements and creates a sense of monumentality and grandeur, prominently showcasing ground-level details while upper elements dramatically rise upward with foreshortening effect"
            },
            "Orbit Left 30°": {
                "Basic": "circling 30 degrees left around the object to reveal a different angle",
                "Detailed": "camera orbits in a smooth circular path 30 degrees to the left around the object, maintaining consistent distance and height while revealing the left side profile, creating a dynamic perspective shift that shows the object from a new vantage point"
            },
            "Orbit Left 45°": {
                "Basic": "circling 45 degrees left around the object for side-angled view",
                "Detailed": "camera orbits in a smooth circular path 45 degrees to the left around the object, maintaining consistent distance and height while transitioning from front to side-front view, creating a dynamic perspective that reveals dimensional depth"
            },
            "Orbit Left 90°": {
                "Basic": "circling 90 degrees left to complete side profile",
                "Detailed": "camera orbits in a smooth circular path 90 degrees to the left around the object, maintaining consistent distance and height while completing a quarter circle to reveal the full left side profile perpendicular to the starting position"
            },
            "Orbit Right 30°": {
                "Basic": "circling 30 degrees right around the object to reveal a different angle",
                "Detailed": "camera orbits in a smooth circular path 30 degrees to the right around the object, maintaining consistent distance and height while revealing the right side profile, creating a dynamic perspective shift that shows the object from a new vantage point"
            },
            "Orbit Right 45°": {
                "Basic": "circling 45 degrees right around the object for side-angled view",
                "Detailed": "camera orbits in a smooth circular path 45 degrees to the right around the object, maintaining consistent distance and height while transitioning from front to side-front view, creating a dynamic perspective that reveals dimensional depth"
            },
            "Orbit Right 90°": {
                "Basic": "circling 90 degrees right to complete side profile",
                "Detailed": "camera orbits in a smooth circular path 90 degrees to the right around the object, maintaining consistent distance and height while completing a quarter circle to reveal the full right side profile perpendicular to the starting position"
            },
            "Orbit Up 30°": {
                "Basic": "circling 30 degrees upward around the object for elevated perspective",
                "Detailed": "camera orbits in a smooth arc 30 degrees upward around the object, maintaining consistent distance while elevating to a higher vantage point, creating a gentle downward-looking angle that reveals more of the top surface"
            },
            "Orbit Up 45°": {
                "Basic": "circling 45 degrees upward for top-angled perspective",
                "Detailed": "camera orbits in a smooth arc 45 degrees upward around the object, maintaining consistent distance while elevating significantly, creating a strong downward-looking angle that emphasizes the top surface and creates a sense of looking down at the object"
            },
            "Orbit Down 30°": {
                "Basic": "circling 30 degrees downward around the object for lower perspective",
                "Detailed": "camera orbits in a smooth arc 30 degrees downward around the object, maintaining consistent distance while descending to a lower vantage point, creating a gentle upward-looking angle that reveals more of the bottom or base"
            },
            "Orbit Down 45°": {
                "Basic": "circling 45 degrees downward for low-angle perspective",
                "Detailed": "camera orbits in a smooth arc 45 degrees downward around the object, maintaining consistent distance while descending significantly, creating a strong upward-looking angle that emphasizes vertical height and creates a sense of looking up at the object"
            },
        }

        # Return appropriate explanation level, or empty string if None
        if "None" in detail_level:
            return ""
        elif "Basic" in detail_level:
            return explanations.get(camera_position, {}).get("Basic", "")
        else:  # Detailed
            return explanations.get(camera_position, {}).get("Detailed", "")

    def _get_movement_explanation(self, camera_movement, detail_level):
        """Get detailed explanation for camera movement based on detail level."""

        # All movement explanations database for 7 movements
        explanations = {
            "Dolly In (Zoom Closer)": {
                "Basic": "gradually moving closer to emphasize details",
                "Detailed": "camera moves smoothly forward towards the object on a straight path, gradually filling more of the frame to emphasize intricate details, textures, and fine craftsmanship as the object grows larger in the frame"
            },
            "Dolly Out (Zoom Away)": {
                "Basic": "gradually moving away to show more context",
                "Detailed": "camera moves smoothly backward away from the object on a straight path, gradually revealing more surrounding context and environmental setting as the object becomes smaller in the frame, providing spatial awareness"
            },
            "Tilt Up Slightly": {
                "Basic": "tilting upward to reveal upper portions",
                "Detailed": "camera tilts slightly upward on its axis while position remains fixed, shifting the view from the middle or lower portions towards the upper sections of the object, creating a gentle upward scanning motion"
            },
            "Tilt Down Slightly": {
                "Basic": "tilting downward to reveal lower portions",
                "Detailed": "camera tilts slightly downward on its axis while position remains fixed, shifting the view from the middle or upper portions towards the lower sections of the object, creating a gentle downward scanning motion"
            },
            "Pan Left": {
                "Basic": "panning left to reveal adjacent areas",
                "Detailed": "camera pans horizontally to the left while position remains fixed, rotating on its vertical axis to sweep the view leftward across the scene, revealing adjacent areas and context to the left side of the object"
            },
            "Pan Right": {
                "Basic": "panning right to reveal adjacent areas",
                "Detailed": "camera pans horizontally to the right while position remains fixed, rotating on its vertical axis to sweep the view rightward across the scene, revealing adjacent areas and context to the right side of the object"
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
    "ArchAi3D_Object_Focus_Camera_V3": ArchAi3D_Object_Focus_Camera_V3
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_Object_Focus_Camera_V3": "📦 Object Focus Camera v3 (Ultimate)"
}
