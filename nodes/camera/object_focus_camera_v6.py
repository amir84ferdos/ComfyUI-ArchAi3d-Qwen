"""
Object Focus Camera v6 - Ultimate Edition

Combines v5 features with Interior Focus vantage point system.

New in v6:
1. Positioning Style Toggle: Choose between Orbit Pattern (v5) or Vantage Point (Interior Focus)
2. Vantage Point Features: height, direction, auto_facing, numeric distance with word conversion
3. Distance Mode Toggle: Preset dropdown OR custom numeric meters
4. number_to_words() converter: Prevents numbers appearing in images

From v5:
- 37 Material Detail Presets (6 categories)
- 15 Photography Quality Presets (3 categories)
- Distance-Aware Positioning (CLOSE/MEDIUM/FAR)
- Environmental Focus Mode (Standard/Focus Transition)
- 19 positions, 7 movements, 10 lenses, 3 languages

Perfect for: Product photography, architectural details, interior photography, macro capture

Author: ArchAi3d
Version: 6.0.0 - Ultimate merged system
"""

class ArchAi3D_Object_Focus_Camera_V6:
    """Ultimate Object Focus Camera with dual positioning systems and professional presets.

    Purpose: Professional object photography with choice of orbit or vantage point positioning.
    Optimized for: Product photography, macro shots, architectural details, interior photography.
    """

    # Height mapping for vantage point mode
    HEIGHT_MAP = {
        "ground_level": "ground level",
        "slightly_below": "slightly below face level",
        "face_level": "face level",
        "slightly_above": "slightly above face level",
        "elevated": "elevated",
        "high": "high"
    }

    # Direction mapping for vantage point mode
    DIRECTION_MAP = {
        "front": "front",
        "left": "left",
        "right": "right",
        "back": "back",
        "front_left": "front-left",
        "front_right": "front-right",
        "back_left": "back-left",
        "back_right": "back-right"
    }

    # Convert preset distances to approximate meters for vantage point mode
    PRESET_TO_METERS = {
        "Very Close (Macro)": 0.5,
        "Close": 1.5,
        "Medium": 3.0,
        "Far": 5.0,
        "Very Far": 8.0
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_object": ("STRING", {
                    "default": "the object",
                    "multiline": False,
                    "tooltip": "What to focus on: 'chandelier crystal', 'brass handle', 'silk fabric'"
                }),
                "positioning_style": ([
                    "Orbit Pattern (v5 style)",
                    "Vantage Point (Interior Focus style)"
                ], {
                    "default": "Orbit Pattern (v5 style)",
                    "tooltip": "Choose camera positioning style: Orbit uses camera_position parameter, Vantage uses height+direction"
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
                    "tooltip": "Camera position relative to object (used in Orbit Pattern mode)"
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
                    "tooltip": "Additional camera movement (dolly is most consistent)"
                }),
                "height": ([
                    "ground_level",
                    "slightly_below",
                    "face_level",
                    "slightly_above",
                    "elevated",
                    "high"
                ], {
                    "default": "face_level",
                    "tooltip": "Camera height (used in Vantage Point mode)"
                }),
                "direction": ([
                    "front",
                    "left",
                    "right",
                    "back",
                    "front_left",
                    "front_right",
                    "back_left",
                    "back_right"
                ], {
                    "default": "front",
                    "tooltip": "Camera direction relative to target (used in Vantage Point mode)"
                }),
                "distance_mode": ([
                    "Preset (Very Close/Close/Medium/Far/Very Far)",
                    "Custom (Numeric meters)"
                ], {
                    "default": "Preset (Very Close/Close/Medium/Far/Very Far)",
                    "tooltip": "Choose distance input method"
                }),
                "camera_distance": ([
                    "Very Close (Macro)",
                    "Close",
                    "Medium",
                    "Far",
                    "Very Far"
                ], {
                    "default": "Close",
                    "tooltip": "Distance from object (used in Preset mode) - affects prompt strength in Orbit mode"
                }),
                "distance_meters": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.5,
                    "max": 10.0,
                    "step": 0.5,
                    "tooltip": "Distance in meters (used in Custom mode) - auto-converted to words"
                }),
                "auto_facing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically face camera toward target (adds 'facing [TARGET]' in Vantage Point mode)"
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
                    "tooltip": "Lens type with technical details"
                }),
                "prompt_language": ([
                    "Chinese (Best for dx8152)",
                    "English (Reddit-validated)",
                    "Hybrid (Chinese + English)"
                ], {
                    "default": "Chinese (Best for dx8152)",
                    "tooltip": "Prompt language - Chinese works best with dx8152 LoRAs"
                }),
                "focus_transition_mode": ([
                    "Standard (Maintain Position)",
                    "Focus Transition (Reposition to Object)"
                ], {
                    "default": "Standard (Maintain Position)",
                    "tooltip": "Standard: Distance-aware. Focus Transition: Intentional repositioning (e.g., corner → refrigerator)"
                }),
                "add_detailed_explanation": ([
                    "None (Simple)",
                    "Basic (Short description)",
                    "Detailed (Full perspective explanation)"
                ], {
                    "default": "Basic (Short description)",
                    "tooltip": "Add camera perspective explanation after base prompt"
                }),
                "material_detail_preset": ([
                    "None (Manual entry)",
                    "--- Crystalline/Glass ---",
                    "Crystal Facets",
                    "Glass Transparency",
                    "Diamond Brilliance",
                    "Frosted Glass",
                    "Stained Glass",
                    "Ice Crystals",
                    "--- Metallic Surfaces ---",
                    "Polished Metal",
                    "Brushed Metal",
                    "Oxidized Patina",
                    "Hammered Metal",
                    "Engraved Details",
                    "Gold Leaf",
                    "Chrome Reflection",
                    "Rust Texture",
                    "--- Fabric/Textile ---",
                    "Silk Weave",
                    "Linen Texture",
                    "Velvet Pile",
                    "Lace Pattern",
                    "Embroidery Details",
                    "Leather Grain",
                    "--- Organic/Natural ---",
                    "Wood Grain",
                    "Stone Texture",
                    "Crystal Formation",
                    "Bark Texture",
                    "Leaf Veins",
                    "Shell Spiral",
                    "Mineral Striations",
                    "--- Technological/Modern ---",
                    "Circuit Board",
                    "Carbon Fiber",
                    "Plastic Molding",
                    "3D Printed Layers",
                    "Screen Pixels",
                    "--- Precious/Luxury ---",
                    "Gemstone Clarity",
                    "Pearl Luster",
                    "Ivory Grain",
                    "Porcelain Glaze",
                    "Enamel Finish"
                ], {
                    "default": "None (Manual entry)",
                    "tooltip": "Pre-written material-specific detail descriptions (37 options)"
                }),
                "photography_quality_preset": ([
                    "None (No quality enhancement)",
                    "--- Technical Excellence ---",
                    "Razor Sharp Focus",
                    "Professional Lighting",
                    "High Dynamic Range",
                    "Color Accuracy",
                    "Bokeh Background",
                    "--- Artistic Style ---",
                    "Editorial Quality",
                    "Commercial Product",
                    "Fine Art Photography",
                    "Documentary Realism",
                    "Cinematic Quality",
                    "--- Detail Enhancement ---",
                    "Extreme Macro Detail",
                    "Texture Emphasis",
                    "Material Authenticity",
                    "Architectural Precision",
                    "Atmospheric Depth"
                ], {
                    "default": "None (No quality enhancement)",
                    "tooltip": "Professional photography quality and style presets (15 options)"
                }),
            },
            "optional": {
                "show_details": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Optional manual details (combines with presets). Example: 'showing intricate patterns'"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "system_prompt", "description")
    FUNCTION = "generate_object_focus_prompt"
    CATEGORY = "ArchAi3d/Qwen/Camera"

    def generate_object_focus_prompt(self, target_object, positioning_style,
                                     camera_position, camera_movement,
                                     height, direction,
                                     distance_mode, camera_distance, distance_meters, auto_facing,
                                     lens_type, prompt_language,
                                     focus_transition_mode, add_detailed_explanation,
                                     material_detail_preset, photography_quality_preset,
                                     show_details=""):
        """
        Generate professional object focus prompt with dual positioning systems.

        Prompt Assembly Order:
        1. Base structure (orbit OR vantage point pattern)
        2. Material Detail Preset (if selected)
        3. Photography Quality Preset (if selected)
        4. Manual show_details (if provided)
        5. Distance-aware explanation (if enabled)
        """

        # Get lens technical details
        lens_details = self._get_lens_details(lens_type)

        # Build prompt based on selected language and positioning style
        if "Chinese" in prompt_language:
            prompt = self._build_chinese_prompt(
                target_object, positioning_style, camera_position, camera_movement,
                height, direction, distance_mode, camera_distance, distance_meters, auto_facing,
                lens_type, lens_details, show_details,
                add_detailed_explanation, focus_transition_mode,
                material_detail_preset, photography_quality_preset
            )
        elif "English" in prompt_language:
            prompt = self._build_english_prompt(
                target_object, positioning_style, camera_position, camera_movement,
                height, direction, distance_mode, camera_distance, distance_meters, auto_facing,
                lens_type, lens_details, show_details,
                add_detailed_explanation, focus_transition_mode,
                material_detail_preset, photography_quality_preset
            )
        else:  # Hybrid
            prompt = self._build_hybrid_prompt(
                target_object, positioning_style, camera_position, camera_movement,
                height, direction, distance_mode, camera_distance, distance_meters, auto_facing,
                lens_type, lens_details, show_details,
                add_detailed_explanation, focus_transition_mode,
                material_detail_preset, photography_quality_preset
            )

        # Generate English description for user
        style_indicator = "🔄" if "Vantage" in positioning_style else "🎯" if "Focus Transition" in focus_transition_mode else "📍"
        movement_str = "" if camera_movement == "None (Static)" else f" + {camera_movement}"

        # Distance display
        if "Custom" in distance_mode:
            distance_str = f"{distance_meters}m"
        else:
            distance_str = camera_distance

        preset_indicator = ""
        if material_detail_preset != "None (Manual entry)" and "---" not in material_detail_preset:
            preset_indicator += f" | Mat: {material_detail_preset}"
        if photography_quality_preset != "None (No quality enhancement)" and "---" not in photography_quality_preset:
            preset_indicator += f" | Qual: {photography_quality_preset}"

        if "Vantage" in positioning_style:
            facing_str = f" facing {target_object}" if auto_facing else ""
            description = f"{style_indicator} {lens_type} | Vantage: {height} @ {direction} {distance_str}{facing_str}{preset_indicator}"
        else:
            description = f"{style_indicator} {lens_type} | {camera_position}{movement_str} | {distance_str}{preset_indicator} | {target_object}"

        # System prompt
        system_prompt = self._get_enhanced_system_prompt(focus_transition_mode, positioning_style)

        return (prompt, system_prompt, description)

    def number_to_words(self, num):
        """Convert number to written words - prevents numbers appearing in images!"""
        if num % 1 == 0.5:  # Handle half numbers (e.g., 1.5)
            whole = int(num)
            if whole == 0:
                return "half a"
            return f"{self.number_to_words(float(whole))} and a half"

        num_int = int(num)
        words_map = {
            0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
            5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
            11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
            16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty"
        }
        return words_map.get(num_int, str(num_int))

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

    def _get_material_detail(self, preset):
        """Get material detail description from preset."""
        material_details = {
            # Crystalline/Glass (6)
            "Crystal Facets": "showing intricate cut patterns, prismatic light reflections, and crystal clarity with sharp edges and geometric precision",
            "Glass Transparency": "revealing internal depth, light refraction patterns, and surface smoothness with subtle imperfections and bubble formations",
            "Diamond Brilliance": "capturing extreme light dispersion, rainbow fire patterns, and microscopic facet precision with brilliant sparkle",
            "Frosted Glass": "showing delicate surface texture, diffused light patterns, and translucent depth with soft edges",
            "Stained Glass": "revealing color transitions, lead came details, and light transmission patterns with artistic craftsmanship",
            "Ice Crystals": "showing hexagonal formations, internal fracture patterns, and crystalline structure with frozen clarity",

            # Metallic Surfaces (8)
            "Polished Metal": "revealing mirror-like reflections, surface scratches, and metallic luster with high contrast highlights",
            "Brushed Metal": "showing parallel grain lines, directional texture, and matte metallic finish with subtle light play",
            "Oxidized Patina": "capturing color variations, corrosion patterns, and aged surface character with historical depth",
            "Hammered Metal": "revealing hand-forged texture, impact marks, and artisan craftsmanship with dimensional depth",
            "Engraved Details": "showing carved lines, depth variations, and precision tooling marks with sharp definition",
            "Gold Leaf": "capturing gilded layers, delicate thickness, and luxurious shimmer with fragile edges",
            "Chrome Reflection": "revealing extreme mirror finish, distortion patterns, and high contrast reflections",
            "Rust Texture": "showing oxidation layers, flaking patterns, and color gradients with weathered character",

            # Fabric/Textile (6)
            "Silk Weave": "revealing thread intersections, subtle sheen, and fabric drape with delicate fiber structure",
            "Linen Texture": "showing natural fiber irregularities, woven pattern, and organic texture with rustic character",
            "Velvet Pile": "capturing directional nap, light absorption, and soft fiber density with luxurious depth",
            "Lace Pattern": "revealing intricate threadwork, negative space design, and delicate craftsmanship with dimensional holes",
            "Embroidery Details": "showing raised stitching, thread texture, and layered patterns with colorful precision",
            "Leather Grain": "capturing pore patterns, natural creases, and surface texture with organic variation",

            # Organic/Natural (7)
            "Wood Grain": "revealing growth rings, fiber direction, and natural color variations with organic patterns",
            "Stone Texture": "showing mineral composition, surface roughness, and geological patterns with natural depth",
            "Crystal Formation": "capturing natural growth patterns, geometric structures, and mineral inclusions with geological beauty",
            "Bark Texture": "revealing layered patterns, natural cracks, and organic texture with weathered character",
            "Leaf Veins": "showing vascular network, cellular structure, and natural patterns with botanical precision",
            "Shell Spiral": "capturing growth lines, nacreous layers, and mathematical patterns with natural elegance",
            "Mineral Striations": "revealing color banding, crystalline structure, and geological layers with natural beauty",

            # Technological/Modern (5)
            "Circuit Board": "showing copper traces, solder joints, and electronic component details with technical precision",
            "Carbon Fiber": "revealing woven pattern, resin surface, and directional fiber alignment with modern aesthetics",
            "Plastic Molding": "capturing injection lines, surface finish, and manufacturing marks with industrial precision",
            "3D Printed Layers": "showing layer lines, extrusion patterns, and additive structure with modern technology",
            "Screen Pixels": "revealing subpixel array, RGB pattern, and display structure with microscopic detail",

            # Precious/Luxury (5)
            "Gemstone Clarity": "revealing internal inclusions, color saturation, and light transmission with valuable perfection",
            "Pearl Luster": "showing iridescent layers, surface smoothness, and orient effect with organic luxury",
            "Ivory Grain": "capturing microscopic texture, color depth, and organic patterns with rare beauty",
            "Porcelain Glaze": "revealing ceramic smoothness, glaze crackle, and translucent depth with delicate perfection",
            "Enamel Finish": "showing glass-like surface, color depth, and reflective quality with artistic precision"
        }
        return material_details.get(preset, "")

    def _get_photography_quality(self, preset):
        """Get photography quality description from preset."""
        quality_presets = {
            # Technical Excellence (5)
            "Razor Sharp Focus": "with extreme sharpness, perfect focus clarity, and microscopic detail resolution",
            "Professional Lighting": "with studio-quality lighting, balanced exposure, and perfect highlight-shadow detail",
            "High Dynamic Range": "with extended dynamic range capturing both bright highlights and deep shadows with rich tonal gradation",
            "Color Accuracy": "with precise color reproduction, accurate white balance, and true-to-life color saturation",
            "Bokeh Background": "with beautiful bokeh background blur, creamy out-of-focus areas, and subject isolation",

            # Artistic Style (5)
            "Editorial Quality": "editorial photography quality with intentional composition, professional styling, and magazine-worthy presentation",
            "Commercial Product": "commercial product photography with clean presentation, optimal angles, and marketing-ready quality",
            "Fine Art Photography": "fine art photography aesthetic with artistic interpretation, mood emphasis, and gallery-worthy composition",
            "Documentary Realism": "documentary photography style with authentic capture, natural moments, and journalistic integrity",
            "Cinematic Quality": "cinematic photography with dramatic lighting, film-like color grading, and movie-quality production values",

            # Detail Enhancement (5)
            "Extreme Macro Detail": "extreme macro photography revealing microscopic surface details, texture intricacies, and hidden patterns invisible to naked eye",
            "Texture Emphasis": "with pronounced texture visibility, tactile quality appearance, and dimensional surface characteristics",
            "Material Authenticity": "capturing authentic material properties, genuine surface characteristics, and real-world wear patterns",
            "Architectural Precision": "with architectural photography precision, geometric accuracy, and structural detail clarity",
            "Atmospheric Depth": "with atmospheric depth, spatial relationships, and three-dimensional presence"
        }
        return quality_presets.get(preset, "")

    def _build_chinese_prompt(self, target_object, positioning_style, camera_position, camera_movement,
                              height, direction, distance_mode, camera_distance, distance_meters, auto_facing,
                              lens_type, lens_details, show_details,
                              add_detailed_explanation, focus_transition_mode,
                              material_detail_preset, photography_quality_preset):
        """Build Chinese prompt with dual positioning systems."""

        parts = []

        # Choose positioning style
        if "Vantage" in positioning_style:
            # VANTAGE POINT MODE - Interior Focus style
            # Determine distance in meters
            if "Custom" in distance_mode:
                distance_m = distance_meters
            else:
                distance_m = self.PRESET_TO_METERS[camera_distance]

            distance_words = self.number_to_words(distance_m)
            height_phrase = self.HEIGHT_MAP.get(height, height)
            direction_phrase = self.DIRECTION_MAP.get(direction, direction)

            vantage = f"change the view to a vantage point at {height_phrase} {distance_words} meters to the {direction_phrase}"

            if auto_facing and target_object:
                vantage += f" facing {target_object}"

            base_prompt = vantage

        else:
            # ORBIT PATTERN MODE - v5 style
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

            # Combine base prompt
            prompt_chinese = "，".join(parts)
            base_prompt = f"Next Scene: {prompt_chinese}"

        # 5. Add Material Detail Preset
        material_detail = self._get_material_detail(material_detail_preset)
        if material_detail:
            base_prompt += f"，{material_detail}"

        # 6. Add Photography Quality Preset
        quality_detail = self._get_photography_quality(photography_quality_preset)
        if quality_detail:
            base_prompt += f"，{quality_detail}"

        # 7. Add manual show_details
        if show_details and show_details.strip():
            base_prompt += f"，{show_details.strip()}"

        # 8. Add detailed explanation if requested (only for Orbit mode)
        if "None" not in add_detailed_explanation and "Orbit" in positioning_style:
            position_explain = self._get_position_explanation(
                camera_position, add_detailed_explanation, camera_distance, focus_transition_mode
            )
            movement_explain = self._get_movement_explanation(camera_movement, add_detailed_explanation)

            explanations = []
            if position_explain:
                explanations.append(position_explain)
            if movement_explain:
                explanations.append(movement_explain)

            if explanations:
                base_prompt += "，" + " ".join(explanations)

        return base_prompt

    def _build_english_prompt(self, target_object, positioning_style, camera_position, camera_movement,
                              height, direction, distance_mode, camera_distance, distance_meters, auto_facing,
                              lens_type, lens_details, show_details,
                              add_detailed_explanation, focus_transition_mode,
                              material_detail_preset, photography_quality_preset):
        """Build English prompt with dual positioning systems."""

        parts = []

        # Choose positioning style
        if "Vantage" in positioning_style:
            # VANTAGE POINT MODE - Interior Focus style
            # Determine distance in meters
            if "Custom" in distance_mode:
                distance_m = distance_meters
            else:
                distance_m = self.PRESET_TO_METERS[camera_distance]

            distance_words = self.number_to_words(distance_m)
            height_phrase = self.HEIGHT_MAP.get(height, height)
            direction_phrase = self.DIRECTION_MAP.get(direction, direction)

            vantage = f"change the view to a vantage point at {height_phrase} {distance_words} meters to the {direction_phrase}"

            if auto_facing and target_object:
                vantage += f" facing {target_object}"

            base_prompt = vantage

        else:
            # ORBIT PATTERN MODE - v5 style
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

            # 4. Camera movement
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

            # Combine base prompt
            base_prompt = ", ".join(parts)

        # 5. Add Material Detail Preset
        material_detail = self._get_material_detail(material_detail_preset)
        if material_detail:
            base_prompt += f", {material_detail}"

        # 6. Add Photography Quality Preset
        quality_detail = self._get_photography_quality(photography_quality_preset)
        if quality_detail:
            base_prompt += f", {quality_detail}"

        # 7. Add manual show_details
        if show_details and show_details.strip():
            base_prompt += f", {show_details.strip()}"

        # 8. Add detailed explanation if requested (only for Orbit mode)
        if "None" not in add_detailed_explanation and "Orbit" in positioning_style:
            position_explain = self._get_position_explanation(
                camera_position, add_detailed_explanation, camera_distance, focus_transition_mode
            )
            movement_explain = self._get_movement_explanation(camera_movement, add_detailed_explanation)

            explanations = []
            if position_explain:
                explanations.append(position_explain)
            if movement_explain:
                explanations.append(movement_explain)

            if explanations:
                base_prompt += ", " + " ".join(explanations)

        return base_prompt

    def _build_hybrid_prompt(self, target_object, positioning_style, camera_position, camera_movement,
                             height, direction, distance_mode, camera_distance, distance_meters, auto_facing,
                             lens_type, lens_details, show_details,
                             add_detailed_explanation, focus_transition_mode,
                             material_detail_preset, photography_quality_preset):
        """Build hybrid prompt with dual positioning systems."""

        parts = []

        # Choose positioning style
        if "Vantage" in positioning_style:
            # VANTAGE POINT MODE - Interior Focus style (English)
            # Determine distance in meters
            if "Custom" in distance_mode:
                distance_m = distance_meters
            else:
                distance_m = self.PRESET_TO_METERS[camera_distance]

            distance_words = self.number_to_words(distance_m)
            height_phrase = self.HEIGHT_MAP.get(height, height)
            direction_phrase = self.DIRECTION_MAP.get(direction, direction)

            vantage = f"change the view to a vantage point at {height_phrase} {distance_words} meters to the {direction_phrase}"

            if auto_facing and target_object:
                vantage += f" facing {target_object}"

            base_prompt = f"Next Scene: {vantage}"

        else:
            # ORBIT PATTERN MODE - v5 hybrid style
            # 1. Lens with English technical term
            lens_cn = lens_details["chinese"]
            lens_tech_en = lens_type
            parts.append(f"将镜头转为{lens_cn} ({lens_tech_en})")

            # 2. Position with mixed terms
            if "Orbit" in camera_position:
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

            # Mix Chinese commas and English commas
            prompt = "，".join(parts[:3])  # Chinese parts
            if len(parts) > 3:
                prompt += "，" + ", ".join(parts[3:])  # English parts

            base_prompt = f"Next Scene: {prompt}"

        # 5. Add Material Detail Preset (always English for hybrid)
        material_detail = self._get_material_detail(material_detail_preset)
        if material_detail:
            base_prompt += f"，{material_detail}"

        # 6. Add Photography Quality Preset (always English for hybrid)
        quality_detail = self._get_photography_quality(photography_quality_preset)
        if quality_detail:
            base_prompt += f"，{quality_detail}"

        # 7. Add manual show_details
        if show_details and show_details.strip():
            base_prompt += f"，{show_details.strip()}"

        # 8. Add detailed explanation if requested (only for Orbit mode)
        if "None" not in add_detailed_explanation and "Orbit" in positioning_style:
            position_explain = self._get_position_explanation(
                camera_position, add_detailed_explanation, camera_distance, focus_transition_mode
            )
            movement_explain = self._get_movement_explanation(camera_movement, add_detailed_explanation)

            explanations = []
            if position_explain:
                explanations.append(position_explain)
            if movement_explain:
                explanations.append(movement_explain)

            if explanations:
                base_prompt += "，" + " ".join(explanations)

        return base_prompt

    # Helper methods from v5 (keeping all for compatibility)

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
        """Get Reddit-validated orbit prompt."""
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

    def _get_enhanced_system_prompt(self, focus_transition_mode, positioning_style):
        """Get enhanced system prompt based on focus transition mode and positioning style."""
        if "Vantage" in positioning_style:
            return (
                "You are an interior photographer. Position camera to focus on specific furniture, "
                "appliance, fixture, or decor element within room. Frame subject prominently in foreground "
                "while showing room context. Preserve all furniture, decor, lighting fixtures, wall features, "
                "flooring, ceiling details, and spatial relationships exactly. Professional interior "
                "photography framing with proper depth and composition."
            )
        elif "Focus Transition" in focus_transition_mode:
            return (
                "You are a precision camera operator specializing in dynamic scene-to-object transitions. "
                "Execute the requested camera repositioning to move from a wide environmental view to a "
                "focused, centered view of the target object. Reposition the camera to stand directly in "
                "front of the object, aligned with its surface. Apply the specified lens characteristics "
                "including depth of field, distortion, and perspective. Maintain the object's appearance, "
                "materials, and details while executing the transition from environmental context to "
                "focused object composition."
            )
        else:
            return (
                "You are a precision camera operator and lens specialist for professional object photography. "
                "Execute camera positioning and lens characteristics exactly as instructed while keeping "
                "the object and scene composition appropriately preserved based on viewing distance. "
                "For close-up views, precise object centering is expected. For medium and far views, "
                "preserve spatial relationships and surrounding context. Maintain all details, textures, "
                "colors, materials, and lighting. Pay special attention to lens-specific characteristics "
                "such as depth of field, distortion, and perspective compression. Your job is to change "
                "the camera viewpoint and apply appropriate lens rendering while respecting the compositional "
                "intent for the selected viewing distance."
            )

    # Distance-aware explanation methods from v4/v5
    def _get_position_explanation(self, camera_position, detail_level, camera_distance, focus_transition_mode):
        """Get detailed explanation for camera position with distance-aware strength."""
        if "None" in detail_level:
            return ""

        use_strong_positioning = "Focus Transition" in focus_transition_mode
        if not use_strong_positioning:
            distance_category = self._get_distance_category(camera_distance)
        else:
            distance_category = "CLOSE"

        if "Basic" in detail_level:
            return self._get_position_explanation_basic(camera_position)
        else:
            return self._get_position_explanation_detailed(camera_position, distance_category)

    def _get_position_explanation_basic(self, camera_position):
        """Get basic explanation for camera position."""
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
            "Orbit Left 30°": "circling 30 degrees left around the object to reveal a different angle",
            "Orbit Left 45°": "circling 45 degrees left around the object for side-angled view",
            "Orbit Left 90°": "circling 90 degrees left to complete side profile",
            "Orbit Right 30°": "circling 30 degrees right around the object to reveal a different angle",
            "Orbit Right 45°": "circling 45 degrees right around the object for side-angled view",
            "Orbit Right 90°": "circling 90 degrees right to complete side profile",
            "Orbit Up 30°": "circling 30 degrees upward around the object for elevated perspective",
            "Orbit Up 45°": "circling 45 degrees upward for top-angled perspective",
            "Orbit Down 30°": "circling 30 degrees downward around the object for lower perspective",
            "Orbit Down 45°": "circling 45 degrees downward for low-angle perspective",
        }
        return explanations.get(camera_position, "")

    def _get_position_explanation_detailed(self, camera_position, distance_category):
        """Get detailed explanation with distance-aware strength (CLOSE/MEDIUM/FAR)."""
        explanations = {
            "Front View": {
                "CLOSE": "camera positioned directly in front of the object at eye level, creating a neutral, balanced view that shows the primary face of the object clearly with natural proportions and no distortion",
                "MEDIUM": "camera oriented towards the front of the object, maintaining the object's position in its surrounding context while showing the primary face clearly with natural proportions",
                "FAR": "camera viewing the scene from the front direction, keeping all objects and spatial relationships exactly as they are without reframing, showing the environment with the object visible in its natural context"
            },
        }
        position_explanations = explanations.get(camera_position, {})
        return position_explanations.get(distance_category, "")

    def _get_movement_explanation(self, camera_movement, detail_level):
        """Get detailed explanation for camera movement."""
        explanations = {
            "Dolly In (Zoom Closer)": {
                "Basic": "gradually moving closer to emphasize details",
                "Detailed": "camera moves smoothly forward towards the object on a straight path, gradually filling more of the frame to emphasize intricate details, textures, and fine craftsmanship as the object grows larger in the frame"
            },
            "Dolly Out (Zoom Away)": {
                "Basic": "gradually moving away to show more context",
                "Detailed": "camera moves smoothly backward away from the object on a straight path, gradually revealing more surrounding context and environmental setting as the object becomes smaller in the frame, providing spatial awareness"
            },
        }

        if "None" in detail_level or camera_movement == "None (Static)":
            return ""
        elif "Basic" in detail_level:
            return explanations.get(camera_movement, {}).get("Basic", "")
        else:
            return explanations.get(camera_movement, {}).get("Detailed", "")


# Node registration
NODE_CLASS_MAPPINGS = {
    "ArchAi3D_Object_Focus_Camera_V6": ArchAi3D_Object_Focus_Camera_V6
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_Object_Focus_Camera_V6": "📦 Object Focus Camera v6 (Ultimate)"
}
