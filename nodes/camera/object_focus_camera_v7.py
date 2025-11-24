"""
Object Focus Camera v7 - Professional Cinematography Edition

NEW IN V7: Professional Film/Photography Terminology
====================================================

1. SHOT SIZES (replaces distance system):
   - Extreme Close-Up (ECU) - Eyes, details, tiny objects
   - Close-Up (CU) - Face/head, small objects
   - Medium Close-Up (MCU) - Chest/shoulders up
   - Medium Shot (MS) - Waist up, standard framing
   - Medium Long Shot (MLS) - Knees up
   - Full Shot (FS) - Head to toes, complete subject
   - Wide Shot (WS) - Subject + environment
   - Extreme Wide Shot (EWS) - Vast environment, tiny subject

2. CAMERA ANGLES (expanded industry standards):
   - Eye Level - Neutral perspective
   - Shoulder Level - Slightly lower, intimate feel
   - High Angle - Looking down, diminishes subject
   - Low Angle - Looking up, empowers subject
   - Bird's Eye View - Directly overhead
   - Worm's Eye View - Ground level looking up
   - Dutch Angle - Tilted horizon, disorienting

3. CAMERA MOVEMENTS (complete cinematography set):
   - Pan (left/right pivot from fixed base)
   - Tilt (up/down pivot from fixed base)
   - Dolly (forward/back camera movement on track)
   - Truck (left/right camera movement on track)
   - Pedestal (vertical camera raise/lower)
   - Arc/Orbit (circular movement around subject)
   - Zoom (focal length change, stationary camera)

4. LENS TYPES (expanded focal lengths):
   - Ultra Wide (14-24mm) - Dramatic perspective
   - Wide Angle (24-35mm) - Expanded view
   - Normal (50mm) - Natural perspective
   - Portrait (85mm) - Subject isolation
   - Telephoto (100-200mm) - Compressed perspective
   - Macro - Extreme close-up detail

FROM V6: Maintains ALL features
- Vantage Point Mode (Interior Focus)
- Material Detail Presets (37 options)
- Photography Quality Presets (15 options)
- Focus Transition Mode
- Plural-safe grammar
- Chinese/English/Hybrid prompts

Perfect for: Professional cinematography, product photography, architectural details, interior photography

Author: ArchAi3d
Version: 7.0.0 - Professional Cinematography Edition
"""

class ArchAi3D_Object_Focus_Camera_V7:
    """Professional Cinematography Camera Node with Industry-Standard Terminology.

    Purpose: Film-quality object photography with professional shot sizes, angles, and movements.
    Optimized for: Commercial cinematography, professional product photography, architectural visualization.
    """

    # Height mapping for vantage point mode
    HEIGHT_MAP = {
        "ground_level": "ground level",
        "slightly_below": "slightly below eye level",
        "eye_level": "eye level",
        "shoulder_level": "shoulder level",
        "slightly_above": "slightly above eye level",
        "elevated": "elevated",
        "high": "high",
        "bird_eye": "bird's eye"
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

    # Convert shot sizes to approximate meters for vantage point mode
    SHOT_SIZE_TO_METERS = {
        "Extreme Close-Up (ECU)": 0.3,
        "Close-Up (CU)": 0.8,
        "Medium Close-Up (MCU)": 1.2,
        "Medium Shot (MS)": 2.0,
        "Medium Long Shot (MLS)": 3.0,
        "Full Shot (FS)": 4.0,
        "Wide Shot (WS)": 6.0,
        "Extreme Wide Shot (EWS)": 10.0
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_object": ("STRING", {
                    "default": "dishwasher",
                    "multiline": False,
                    "tooltip": "Subject to focus on: 'chandelier crystal', 'brass handle', 'character face'"
                }),
                "shot_size": ([
                    "Extreme Close-Up (ECU)",
                    "Close-Up (CU)",
                    "Medium Close-Up (MCU)",
                    "Medium Shot (MS)",
                    "Medium Long Shot (MLS)",
                    "Full Shot (FS)",
                    "Wide Shot (WS)",
                    "Extreme Wide Shot (EWS)"
                ], {
                    "default": "Medium Shot (MS)",
                    "tooltip": "Cinematography shot size classification. Each size maps to approximate distance: ECU=0.3m, CU=0.8m, MCU=1.2m, MS=2.0m, MLS=3.0m, FS=4.0m, WS=6.0m, EWS=10.0m"
                }),
                "camera_angle": ([
                    "Eye Level",
                    "Shoulder Level",
                    "High Angle",
                    "Low Angle",
                    "Bird's Eye View",
                    "Worm's Eye View",
                    "Dutch Angle",
                    "--- Advanced Angles ---",
                    "Angled View (15°)",
                    "Angled View (30°)",
                    "Angled View (45°)",
                    "Angled View (60°)",
                    "Side View (90°)",
                    "Back View (180°)",
                    "--- Orbit Patterns ---",
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
                    "--- Vantage Point ---",
                    "Vantage: Eye Level @ Front",
                    "Vantage: Eye Level @ Front-Left",
                    "Vantage: Eye Level @ Front-Right",
                    "Vantage: Eye Level @ Left",
                    "Vantage: Eye Level @ Right",
                    "Vantage: Slightly Above @ Front",
                    "Vantage: Elevated @ Front",
                    "Vantage: Ground Level @ Front",
                    "Vantage: Custom"
                ], {
                    "default": "Eye Level",
                    "tooltip": "Professional camera angle terminology"
                }),
                "camera_movement": ([
                    "Static (No Movement)",
                    "--- Pivoting Movements ---",
                    "Pan Left",
                    "Pan Right",
                    "Tilt Up",
                    "Tilt Down",
                    "--- Tracking Movements ---",
                    "Dolly In (Forward)",
                    "Dolly Out (Backward)",
                    "Truck Left (Lateral)",
                    "Truck Right (Lateral)",
                    "--- Vertical Movement ---",
                    "Pedestal Up (Raise)",
                    "Pedestal Down (Lower)",
                    "--- Circular Movement ---",
                    "Arc Left",
                    "Arc Right",
                    "--- Lens Only ---",
                    "Zoom In (Focal Length)",
                    "Zoom Out (Focal Length)"
                ], {
                    "default": "Static (No Movement)",
                    "tooltip": "Complete professional camera movement terminology"
                }),
                "height": ([
                    "ground_level",
                    "slightly_below",
                    "eye_level",
                    "shoulder_level",
                    "slightly_above",
                    "elevated",
                    "high",
                    "bird_eye"
                ], {
                    "default": "slightly_above",
                    "tooltip": "Camera height (for 'Vantage: Custom' position)"
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
                    "tooltip": "Camera direction (for 'Vantage: Custom' position)"
                }),
                "framing_mode": ([
                    "Shot Size Presets",
                    "Custom Meters"
                ], {
                    "default": "Shot Size Presets",
                    "tooltip": "Use cinematic shot size presets OR specify exact distance in meters"
                }),
                "distance_meters": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.3,
                    "max": 15.0,
                    "step": 0.5,
                    "tooltip": "Distance in meters (used in Custom mode) - auto-converted to words"
                }),
                "auto_facing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically face camera toward target (for Vantage positions)"
                }),
                "lens_type": ([
                    "--- Professional Lenses ---",
                    "Ultra Wide (14-24mm)",
                    "Wide Angle (24-35mm)",
                    "Normal (50mm)",
                    "Portrait (85mm)",
                    "Telephoto (100-200mm)",
                    "Macro (Close-Up)",
                    "--- Special Lenses ---",
                    "Fisheye (Ultra Wide)",
                    "Tilt-Shift (Perspective Control)",
                    "Anamorphic (Cinematic)",
                    "Soft Focus (Dreamy)"
                ], {
                    "default": "Normal (50mm)",
                    "tooltip": "Professional lens classification with focal lengths"
                }),
                "prompt_language": ([
                    "Chinese (Best for dx8152)",
                    "English (Universal)",
                    "Hybrid (Chinese + English)"
                ], {
                    "default": "Chinese (Best for dx8152)",
                    "tooltip": "Prompt language - Chinese works best with dx8152 LoRAs"
                }),
                "focus_transition_mode": ([
                    "Standard (Maintain Position)",
                    "Focus Transition (Reposition to Object)"
                ], {
                    "default": "Focus Transition (Reposition to Object)",
                    "tooltip": "Standard: Distance-aware. Focus Transition: Intentional repositioning"
                }),
                "add_detailed_explanation": ([
                    "None (Simple)",
                    "Basic (Short description)",
                    "Detailed (Full cinematography explanation)"
                ], {
                    "default": "Detailed (Full cinematography explanation)",
                    "tooltip": "Add professional cinematography explanation"
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
                    "tooltip": "Optional manual details (combines with presets)"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "system_prompt", "description")
    FUNCTION = "generate_cinematography_prompt"
    CATEGORY = "ArchAi3d/Qwen/Camera"

    def generate_cinematography_prompt(self, target_object,
                                       shot_size, camera_angle, camera_movement,
                                       height, direction,
                                       framing_mode, distance_meters, auto_facing,
                                       lens_type, prompt_language,
                                       focus_transition_mode, add_detailed_explanation,
                                       material_detail_preset, photography_quality_preset,
                                       show_details=""):
        """
        Generate professional cinematography prompt with industry-standard terminology.

        Prompt Assembly Order:
        1. Shot size + lens + angle OR Vantage point
        2. Camera movement (if not Static)
        3. Material Detail Preset (if selected)
        4. Photography Quality Preset (if selected)
        5. Manual show_details (if provided)
        6. Professional cinematography explanation (if enabled)
        """

        # Detect if vantage position selected
        is_vantage = "Vantage:" in camera_angle

        # Parse vantage height and direction
        if is_vantage and "Custom" in camera_angle:
            vantage_height = height
            vantage_direction = direction
        elif is_vantage:
            # Parse preset: "Vantage: Eye Level @ Front" → "eye_level", "front"
            parts = camera_angle.replace("Vantage: ", "").split(" @ ")
            vantage_height = parts[0].lower().replace(" ", "_")
            vantage_direction = parts[1].lower().replace("-", "_")
        else:
            vantage_height = None
            vantage_direction = None

        # Get lens technical details
        lens_details = self._get_lens_details(lens_type)

        # Build prompt based on selected language
        if "Chinese" in prompt_language:
            prompt = self._build_chinese_prompt(
                target_object, is_vantage, vantage_height, vantage_direction,
                shot_size, camera_angle, camera_movement,
                framing_mode, distance_meters, auto_facing,
                lens_type, lens_details, show_details,
                add_detailed_explanation, focus_transition_mode,
                material_detail_preset, photography_quality_preset
            )
        elif "English" in prompt_language:
            prompt = self._build_english_prompt(
                target_object, is_vantage, vantage_height, vantage_direction,
                shot_size, camera_angle, camera_movement,
                framing_mode, distance_meters, auto_facing,
                lens_type, lens_details, show_details,
                add_detailed_explanation, focus_transition_mode,
                material_detail_preset, photography_quality_preset
            )
        else:  # Hybrid
            prompt = self._build_hybrid_prompt(
                target_object, is_vantage, vantage_height, vantage_direction,
                shot_size, camera_angle, camera_movement,
                framing_mode, distance_meters, auto_facing,
                lens_type, lens_details, show_details,
                add_detailed_explanation, focus_transition_mode,
                material_detail_preset, photography_quality_preset
            )

        # Generate English description for user
        style_indicator = "🎬" if not is_vantage else "🔄"
        movement_str = "" if camera_movement == "Static (No Movement)" else f" + {camera_movement}"

        # Shot size display
        if "Custom" in framing_mode:
            size_str = f"{distance_meters}m"
        else:
            size_str = shot_size.split(" (")[1].replace(")", "") if "(" in shot_size else shot_size

        preset_indicator = ""
        if material_detail_preset != "None (Manual entry)" and "---" not in material_detail_preset:
            preset_indicator += f" | Mat: {material_detail_preset}"
        if photography_quality_preset != "None (No quality enhancement)" and "---" not in photography_quality_preset:
            preset_indicator += f" | Qual: {photography_quality_preset}"

        if is_vantage:
            facing_str = f" facing {target_object}" if auto_facing else ""
            description = f"{style_indicator} {lens_type} | Vantage: {vantage_height} @ {vantage_direction} {size_str}{facing_str}{preset_indicator}"
        else:
            description = f"{style_indicator} {lens_type} | {size_str} | {camera_angle}{movement_str}{preset_indicator} | {target_object}"

        # System prompt
        system_prompt = self._get_enhanced_system_prompt(focus_transition_mode, is_vantage)

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
            # Professional focal lengths
            "Ultra Wide (14-24mm)": {
                "chinese": "超广角镜头(14-24mm)",
                "technical": "ultra-wide angle lens 14-24mm with dramatic perspective and barrel distortion",
                "characteristics": "dramatic perspective, very wide field of view, barrel distortion"
            },
            "Wide Angle (24-35mm)": {
                "chinese": "广角镜头(24-35mm)",
                "technical": "wide-angle lens 24-35mm with expanded field of view and slight perspective distortion",
                "characteristics": "expanded view, slight edge distortion, environmental context"
            },
            "Normal (50mm)": {
                "chinese": "标准镜头(50mm)",
                "technical": "standard 50mm lens with natural perspective and balanced field of view",
                "characteristics": "natural perspective, no distortion, human eye equivalent"
            },
            "Portrait (85mm)": {
                "chinese": "人像镜头(85mm)",
                "technical": "portrait 85mm lens with flattering perspective and subject isolation",
                "characteristics": "flattering perspective, shallow depth of field, subject isolation"
            },
            "Telephoto (100-200mm)": {
                "chinese": "长焦镜头(100-200mm)",
                "technical": "telephoto lens 100-200mm with compressed perspective and strong background blur",
                "characteristics": "compressed perspective, strong subject isolation, bokeh"
            },
            "Macro (Close-Up)": {
                "chinese": "微距镜头",
                "technical": "macro lens with 1:1 magnification ratio and extreme close-up detail capability",
                "characteristics": "1:1 magnification, extreme detail, very shallow depth of field"
            },
            # Special lenses
            "Fisheye (Ultra Wide)": {
                "chinese": "鱼眼镜头",
                "technical": "fisheye lens with extreme barrel distortion and 180-degree field of view",
                "characteristics": "180° view, extreme barrel distortion, spherical effect"
            },
            "Tilt-Shift (Perspective Control)": {
                "chinese": "移轴镜头",
                "technical": "tilt-shift lens with selective focus plane and perspective control",
                "characteristics": "selective focus plane, miniature effect, perspective correction"
            },
            "Anamorphic (Cinematic)": {
                "chinese": "变形宽银幕镜头",
                "technical": "anamorphic lens with cinematic aspect ratio and horizontal lens flares",
                "characteristics": "2.39:1 aspect, oval bokeh, horizontal flares, cinematic look"
            },
            "Soft Focus (Dreamy)": {
                "chinese": "柔焦镜头",
                "technical": "soft focus lens with dreamy glow and ethereal quality",
                "characteristics": "soft glow, reduced contrast, dreamy ethereal quality"
            }
        }
        return lens_details.get(lens_type, lens_details["Normal (50mm)"])

    def _get_shot_size_category(self, shot_size):
        """Map 8 shot sizes to 3 categories for prompt strength."""
        if shot_size in ["Extreme Close-Up (ECU)", "Close-Up (CU)", "Medium Close-Up (MCU)"]:
            return "CLOSE"
        elif shot_size in ["Medium Shot (MS)", "Medium Long Shot (MLS)"]:
            return "MEDIUM"
        else:  # "Full Shot (FS)", "Wide Shot (WS)", "Extreme Wide Shot (EWS)"
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

    def _build_chinese_prompt(self, target_object, is_vantage, vantage_height, vantage_direction,
                              shot_size, camera_angle, camera_movement,
                              framing_mode, distance_meters, auto_facing,
                              lens_type, lens_details, show_details,
                              add_detailed_explanation, focus_transition_mode,
                              material_detail_preset, photography_quality_preset):
        """Build Chinese prompt with cinematography terminology."""

        parts = []

        # Choose positioning mode
        if is_vantage:
            # VANTAGE POINT MODE
            lens_cn = lens_details["chinese"]
            lens_tech = lens_details["characteristics"]
            parts.append(f"将镜头转为{lens_cn}（{lens_tech}）")

            # Shot size converted to Chinese
            shot_cn = self._get_shot_size_chinese(shot_size)
            parts.append(f"{shot_cn}构图查看{target_object}")

            # Build vantage point addition
            if "Custom" in framing_mode:
                distance_m = distance_meters
            else:
                distance_m = self.SHOT_SIZE_TO_METERS[shot_size]

            distance_words = self.number_to_words(distance_m)
            height_phrase = self.HEIGHT_MAP.get(vantage_height, vantage_height)
            direction_phrase = self.DIRECTION_MAP.get(vantage_direction, vantage_direction)

            vantage = f"change the view to a vantage point at {height_phrase} {distance_words} meters to the {direction_phrase}"

            if auto_facing and target_object:
                vantage += f" facing {target_object}"

            # Combine: Chinese base + vantage addition
            prompt_chinese = "，".join(parts)
            base_prompt = f"Next Scene: {prompt_chinese}，{vantage}"

        else:
            # CINEMATOGRAPHY MODE
            # 1. Lens with technical details
            lens_cn = lens_details["chinese"]
            lens_tech = lens_details["characteristics"]
            parts.append(f"将镜头转为{lens_cn}（{lens_tech}）")

            # 2. Shot size
            shot_cn = self._get_shot_size_chinese(shot_size)
            parts.append(f"{shot_cn}构图")

            # 3. Camera angle
            angle_cn = self._get_angle_chinese(camera_angle)
            parts.append(f"{angle_cn}{target_object}")

            # 4. Camera movement (if not static)
            if camera_movement != "Static (No Movement)":
                movement_cn = self._get_movement_chinese(camera_movement)
                parts.append(movement_cn)

            # Combine base prompt
            prompt_chinese = "，".join(parts)
            base_prompt = f"Next Scene: {prompt_chinese}"

        # Add Material Detail Preset
        material_detail = self._get_material_detail(material_detail_preset)
        if material_detail:
            base_prompt += f"，{material_detail}"

        # Add Photography Quality Preset
        quality_detail = self._get_photography_quality(photography_quality_preset)
        if quality_detail:
            base_prompt += f"，{quality_detail}"

        # Add manual show_details
        if show_details and show_details.strip():
            base_prompt += f"，{show_details.strip()}"

        # Add detailed explanation if requested (only for non-vantage mode)
        if "None" not in add_detailed_explanation and not is_vantage:
            angle_explain = self._get_angle_explanation(
                camera_angle, add_detailed_explanation, shot_size, focus_transition_mode
            )
            movement_explain = self._get_movement_explanation(camera_movement, add_detailed_explanation)

            explanations = []
            if angle_explain:
                explanations.append(angle_explain)
            if movement_explain:
                explanations.append(movement_explain)

            if explanations:
                base_prompt += "，" + " ".join(explanations)

        return base_prompt

    def _build_english_prompt(self, target_object, is_vantage, vantage_height, vantage_direction,
                              shot_size, camera_angle, camera_movement,
                              framing_mode, distance_meters, auto_facing,
                              lens_type, lens_details, show_details,
                              add_detailed_explanation, focus_transition_mode,
                              material_detail_preset, photography_quality_preset):
        """Build English prompt with cinematography terminology."""

        parts = []

        # Choose positioning mode
        if is_vantage:
            # VANTAGE POINT MODE
            parts.append(f"Change to {lens_details['technical']}")

            # Shot size in English
            shot_desc = self._get_shot_size_english(shot_size)
            parts.append(f"{shot_desc} framing of {target_object}")

            # Build vantage point addition
            if "Custom" in framing_mode:
                distance_m = distance_meters
            else:
                distance_m = self.SHOT_SIZE_TO_METERS[shot_size]

            distance_words = self.number_to_words(distance_m)
            height_phrase = self.HEIGHT_MAP.get(vantage_height, vantage_height)
            direction_phrase = self.DIRECTION_MAP.get(vantage_direction, vantage_direction)

            vantage = f"change the view to a vantage point at {height_phrase} {distance_words} meters to the {direction_phrase}"

            if auto_facing and target_object:
                vantage += f" facing {target_object}"

            # Combine
            base_prompt = ", ".join(parts) + f", {vantage}"

        else:
            # CINEMATOGRAPHY MODE
            # 1. Lens technical description
            parts.append(f"Change to {lens_details['technical']}")

            # 2. Shot size
            shot_desc = self._get_shot_size_english(shot_size)
            parts.append(f"{shot_desc} framing")

            # 3. Camera angle
            angle_desc = self._get_angle_english(camera_angle, target_object)
            parts.append(angle_desc)

            # 4. Camera movement
            if camera_movement != "Static (No Movement)":
                movement_desc = self._get_movement_english(camera_movement)
                parts.append(movement_desc)

            # Combine base prompt
            base_prompt = ", ".join(parts)

        # Add Material Detail Preset
        material_detail = self._get_material_detail(material_detail_preset)
        if material_detail:
            base_prompt += f", {material_detail}"

        # Add Photography Quality Preset
        quality_detail = self._get_photography_quality(photography_quality_preset)
        if quality_detail:
            base_prompt += f", {quality_detail}"

        # Add manual show_details
        if show_details and show_details.strip():
            base_prompt += f", {show_details.strip()}"

        # Add detailed explanation if requested
        if "None" not in add_detailed_explanation and not is_vantage:
            angle_explain = self._get_angle_explanation(
                camera_angle, add_detailed_explanation, shot_size, focus_transition_mode
            )
            movement_explain = self._get_movement_explanation(camera_movement, add_detailed_explanation)

            explanations = []
            if angle_explain:
                explanations.append(angle_explain)
            if movement_explain:
                explanations.append(movement_explain)

            if explanations:
                base_prompt += ", " + " ".join(explanations)

        return base_prompt

    def _build_hybrid_prompt(self, target_object, is_vantage, vantage_height, vantage_direction,
                             shot_size, camera_angle, camera_movement,
                             framing_mode, distance_meters, auto_facing,
                             lens_type, lens_details, show_details,
                             add_detailed_explanation, focus_transition_mode,
                             material_detail_preset, photography_quality_preset):
        """Build hybrid prompt (Chinese + English) with cinematography terminology."""

        parts = []

        if is_vantage:
            # VANTAGE POINT MODE - Hybrid
            lens_cn = lens_details["chinese"]
            lens_tech_en = lens_type
            parts.append(f"将镜头转为{lens_cn} ({lens_tech_en})")

            shot_cn = self._get_shot_size_chinese(shot_size)
            parts.append(f"{shot_cn}构图查看{target_object}")

            # Build vantage point addition (English)
            if "Custom" in framing_mode:
                distance_m = distance_meters
            else:
                distance_m = self.SHOT_SIZE_TO_METERS[shot_size]

            distance_words = self.number_to_words(distance_m)
            height_phrase = self.HEIGHT_MAP.get(vantage_height, vantage_height)
            direction_phrase = self.DIRECTION_MAP.get(vantage_direction, vantage_direction)

            vantage = f"change the view to a vantage point at {height_phrase} {distance_words} meters to the {direction_phrase}"

            if auto_facing and target_object:
                vantage += f" facing {target_object}"

            # Combine
            prompt_chinese = "，".join(parts)
            base_prompt = f"Next Scene: {prompt_chinese}，{vantage}"

        else:
            # CINEMATOGRAPHY MODE - Hybrid
            # 1. Lens with English technical term
            lens_cn = lens_details["chinese"]
            lens_tech_en = lens_type
            parts.append(f"将镜头转为{lens_cn} ({lens_tech_en})")

            # 2. Shot size in Chinese
            shot_cn = self._get_shot_size_chinese(shot_size)
            parts.append(f"{shot_cn}构图")

            # 3. Angle in Chinese
            angle_cn = self._get_angle_chinese(camera_angle)
            parts.append(f"{angle_cn}{target_object}")

            # 4. Movement in English
            if camera_movement != "Static (No Movement)":
                movement_en = self._get_movement_english(camera_movement)
                parts.append(movement_en)

            # Mix Chinese and English
            prompt = "，".join(parts[:3])  # Chinese parts
            if len(parts) > 3:
                prompt += "，" + parts[3]  # English movement

            base_prompt = f"Next Scene: {prompt}"

        # Add Material Detail Preset (always English for hybrid)
        material_detail = self._get_material_detail(material_detail_preset)
        if material_detail:
            base_prompt += f"，{material_detail}"

        # Add Photography Quality Preset (always English for hybrid)
        quality_detail = self._get_photography_quality(photography_quality_preset)
        if quality_detail:
            base_prompt += f"，{quality_detail}"

        # Add manual show_details
        if show_details and show_details.strip():
            base_prompt += f"，{show_details.strip()}"

        # Add detailed explanation if requested
        if "None" not in add_detailed_explanation and not is_vantage:
            angle_explain = self._get_angle_explanation(
                camera_angle, add_detailed_explanation, shot_size, focus_transition_mode
            )
            movement_explain = self._get_movement_explanation(camera_movement, add_detailed_explanation)

            explanations = []
            if angle_explain:
                explanations.append(angle_explain)
            if movement_explain:
                explanations.append(movement_explain)

            if explanations:
                base_prompt += "，" + " ".join(explanations)

        return base_prompt

    # Helper methods - Shot Sizes

    def _get_shot_size_chinese(self, shot_size):
        """Convert shot size to Chinese."""
        shot_map = {
            "Extreme Close-Up (ECU)": "特写",
            "Close-Up (CU)": "近景",
            "Medium Close-Up (MCU)": "中近景",
            "Medium Shot (MS)": "中景",
            "Medium Long Shot (MLS)": "中全景",
            "Full Shot (FS)": "全景",
            "Wide Shot (WS)": "远景",
            "Extreme Wide Shot (EWS)": "超远景"
        }
        return shot_map.get(shot_size, "中景")

    def _get_shot_size_english(self, shot_size):
        """Convert shot size to English description."""
        shot_map = {
            "Extreme Close-Up (ECU)": "extreme close-up",
            "Close-Up (CU)": "close-up",
            "Medium Close-Up (MCU)": "medium close-up",
            "Medium Shot (MS)": "medium shot",
            "Medium Long Shot (MLS)": "medium long shot",
            "Full Shot (FS)": "full shot",
            "Wide Shot (WS)": "wide shot",
            "Extreme Wide Shot (EWS)": "extreme wide shot"
        }
        return shot_map.get(shot_size, "medium shot")

    # Helper methods - Camera Angles

    def _get_angle_chinese(self, angle):
        """Convert camera angle to Chinese."""
        angle_map = {
            # Professional angles
            "Eye Level": "平视",
            "Shoulder Level": "肩平视角",
            "High Angle": "俯视",
            "Low Angle": "仰视",
            "Bird's Eye View": "鸟瞰",
            "Worm's Eye View": "虫瞰",
            "Dutch Angle": "倾斜角度",
            # Advanced angles (legacy v6 compatibility)
            "Angled View (15°)": "从15度角查看",
            "Angled View (30°)": "从30度角查看",
            "Angled View (45°)": "从45度角查看",
            "Angled View (60°)": "从60度角查看",
            "Side View (90°)": "从侧面查看",
            "Back View (180°)": "从背面查看",
            # Orbit patterns
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
        return angle_map.get(angle, "平视")

    def _get_angle_english(self, angle, target_object):
        """Convert camera angle to English description."""
        # Professional angles
        if angle == "Eye Level":
            return f"eye level view of {target_object}"
        elif angle == "Shoulder Level":
            return f"shoulder level view of {target_object}"
        elif angle == "High Angle":
            return f"high angle looking down at {target_object}"
        elif angle == "Low Angle":
            return f"low angle looking up at {target_object}"
        elif angle == "Bird's Eye View":
            return f"bird's eye view directly above {target_object}"
        elif angle == "Worm's Eye View":
            return f"worm's eye view from ground level up at {target_object}"
        elif angle == "Dutch Angle":
            return f"dutch angle with tilted horizon viewing {target_object}"
        # Orbit patterns
        elif "Orbit" in angle:
            if "Left" in angle:
                degrees = angle.split("°")[0].split()[-1]
                return f"camera orbit left around {target_object} by {degrees} degrees"
            elif "Right" in angle:
                degrees = angle.split("°")[0].split()[-1]
                return f"camera orbit right around {target_object} by {degrees} degrees"
            elif "Up" in angle:
                degrees = angle.split("°")[0].split()[-1]
                return f"camera orbit up around {target_object} by {degrees} degrees"
            elif "Down" in angle:
                degrees = angle.split("°")[0].split()[-1]
                return f"camera orbit down around {target_object} by {degrees} degrees"
        # Advanced angles (legacy)
        elif "Angled View" in angle:
            degrees = angle.split("(")[1].split("°")[0]
            return f"{degrees}-degree angled view of {target_object}"
        elif "Side View" in angle:
            return f"90-degree side view of {target_object}"
        elif "Back View" in angle:
            return f"back view 180 degrees of {target_object}"

        return f"view of {target_object}"

    # Helper methods - Camera Movements

    def _get_movement_chinese(self, movement):
        """Convert camera movement to Chinese."""
        movement_map = {
            # Pivoting
            "Pan Left": "向左平移",
            "Pan Right": "向右平移",
            "Tilt Up": "向上倾斜",
            "Tilt Down": "向下倾斜",
            # Tracking
            "Dolly In (Forward)": "推近镜头",
            "Dolly Out (Backward)": "拉远镜头",
            "Truck Left (Lateral)": "横移向左",
            "Truck Right (Lateral)": "横移向右",
            # Vertical
            "Pedestal Up (Raise)": "升高镜头",
            "Pedestal Down (Lower)": "降低镜头",
            # Circular
            "Arc Left": "弧形左移",
            "Arc Right": "弧形右移",
            # Lens only
            "Zoom In (Focal Length)": "变焦推进",
            "Zoom Out (Focal Length)": "变焦拉远"
        }
        return movement_map.get(movement, "")

    def _get_movement_english(self, movement):
        """Convert camera movement to English description."""
        movement_map = {
            # Pivoting
            "Pan Left": "pan camera left on horizontal axis",
            "Pan Right": "pan camera right on horizontal axis",
            "Tilt Up": "tilt camera up on vertical axis",
            "Tilt Down": "tilt camera down on vertical axis",
            # Tracking
            "Dolly In (Forward)": "dolly camera forward on track",
            "Dolly Out (Backward)": "dolly camera backward on track",
            "Truck Left (Lateral)": "truck camera left on lateral track",
            "Truck Right (Lateral)": "truck camera right on lateral track",
            # Vertical
            "Pedestal Up (Raise)": "pedestal camera up raising vertical position",
            "Pedestal Down (Lower)": "pedestal camera down lowering vertical position",
            # Circular
            "Arc Left": "arc camera left in circular motion",
            "Arc Right": "arc camera right in circular motion",
            # Lens only
            "Zoom In (Focal Length)": "zoom in by increasing focal length",
            "Zoom Out (Focal Length)": "zoom out by decreasing focal length"
        }
        return movement_map.get(movement, "")

    def _get_enhanced_system_prompt(self, focus_transition_mode, is_vantage):
        """Get enhanced system prompt for cinematography."""
        if is_vantage:
            return (
                "You are a professional cinematographer specializing in interior photography. "
                "Position camera to focus on specific furniture, appliance, fixture, or decor element "
                "within room using professional framing techniques. Frame subject prominently while "
                "showing room context. Preserve all furniture, decor, lighting fixtures, wall features, "
                "flooring, ceiling details, and spatial relationships exactly. Apply cinematography "
                "principles for proper composition, depth, and visual storytelling."
            )
        elif "Focus Transition" in focus_transition_mode:
            return (
                "You are a professional cinematographer executing dynamic scene-to-subject transitions. "
                "Perform the requested camera repositioning to transition from establishing shot to "
                "focused composition on target subject. Reposition camera following cinematography "
                "principles with proper shot size, angle, and lens characteristics. Apply specified "
                "lens properties including depth of field, distortion, and perspective. Maintain "
                "subject appearance, materials, and details while executing professional transition "
                "from environmental context to focused cinematography composition."
            )
        else:
            return (
                "You are a professional cinematographer and lens specialist. Execute camera positioning "
                "using industry-standard shot sizes, angles, and movements exactly as specified. "
                "For close-up shots, precise subject framing is expected. For medium and wide shots, "
                "preserve spatial relationships and environmental context. Maintain all details, "
                "textures, colors, materials, and lighting with cinematography quality. Apply "
                "lens-specific characteristics including depth of field, distortion, perspective "
                "compression, and bokeh. Your role is to execute professional camera positioning "
                "and lens rendering while maintaining compositional integrity appropriate to "
                "the specified shot size."
            )

    def _get_angle_explanation(self, camera_angle, detail_level, shot_size, focus_transition_mode):
        """Get detailed explanation for camera angle with shot size awareness."""
        if "None" in detail_level:
            return ""

        use_strong_positioning = "Focus Transition" in focus_transition_mode
        if not use_strong_positioning:
            size_category = self._get_shot_size_category(shot_size)
        else:
            size_category = "CLOSE"

        if "Basic" in detail_level:
            return self._get_angle_explanation_basic(camera_angle)
        else:
            return self._get_angle_explanation_detailed(camera_angle, size_category)

    def _get_angle_explanation_basic(self, camera_angle):
        """Get basic explanation for camera angle."""
        explanations = {
            # Professional angles
            "Eye Level": "creating neutral perspective at natural eye height",
            "Shoulder Level": "creating intimate perspective from shoulder height",
            "High Angle": "looking down from elevated position to diminish subject",
            "Low Angle": "looking up from below to empower and emphasize subject",
            "Bird's Eye View": "creating overhead perspective showing spatial layout",
            "Worm's Eye View": "creating ground-level upward perspective emphasizing verticality",
            "Dutch Angle": "creating disorienting perspective with tilted horizon",
            # Orbit patterns
            "Orbit Left 30°": "circling 30 degrees left around the subject to reveal different angle",
            "Orbit Left 45°": "circling 45 degrees left for side-angled perspective",
            "Orbit Left 90°": "circling 90 degrees left to complete side profile",
            "Orbit Right 30°": "circling 30 degrees right around the subject to reveal different angle",
            "Orbit Right 45°": "circling 45 degrees right for side-angled perspective",
            "Orbit Right 90°": "circling 90 degrees right to complete side profile",
            "Orbit Up 30°": "circling 30 degrees upward for elevated perspective",
            "Orbit Up 45°": "circling 45 degrees upward for top-angled perspective",
            "Orbit Down 30°": "circling 30 degrees downward for lower perspective",
            "Orbit Down 45°": "circling 45 degrees downward for low-angle perspective",
        }
        return explanations.get(camera_angle, "")

    def _get_angle_explanation_detailed(self, camera_angle, size_category):
        """Get detailed cinematography explanation with shot size awareness."""
        explanations = {
            "Eye Level": {
                "CLOSE": "camera positioned at eye level creating neutral psychological perspective, showing primary features clearly with natural proportions and authentic representation without distortion",
                "MEDIUM": "camera at eye level maintaining natural perspective within environmental context, showing subject clearly while preserving surrounding spatial relationships",
                "FAR": "camera at eye level viewing scene naturally, maintaining all spatial relationships and environmental context without reframing"
            },
            "High Angle": {
                "CLOSE": "camera elevated above subject looking down, creating psychological sense of vulnerability or diminishment while emphasizing top surfaces and reducing perceived power",
                "MEDIUM": "camera at high angle showing subject from above within environment, revealing spatial relationships from elevated perspective",
                "FAR": "camera at high angle viewing scene from above, showing comprehensive spatial layout and environmental relationships"
            },
            "Low Angle": {
                "CLOSE": "camera positioned below subject looking up, creating psychological sense of power, dominance, or importance while emphasizing vertical height and imposing presence",
                "MEDIUM": "camera at low angle viewing subject from below within environment, creating sense of significance while showing surroundings",
                "FAR": "camera at low angle viewing scene from ground level, emphasizing vertical scale and architectural presence"
            },
        }
        angle_explanations = explanations.get(camera_angle, {})
        return angle_explanations.get(size_category, "")

    def _get_movement_explanation(self, camera_movement, detail_level):
        """Get detailed explanation for professional camera movements."""
        explanations = {
            # Pivoting movements
            "Pan Left": {
                "Basic": "pivoting camera left to reveal new space",
                "Detailed": "camera pivots left on horizontal axis from fixed position, smoothly revealing new spatial information to the left while maintaining vertical framing and creating natural scanning motion"
            },
            "Pan Right": {
                "Basic": "pivoting camera right to reveal new space",
                "Detailed": "camera pivots right on horizontal axis from fixed position, smoothly revealing new spatial information to the right while maintaining vertical framing and creating natural scanning motion"
            },
            "Tilt Up": {
                "Basic": "pivoting camera upward to show higher elements",
                "Detailed": "camera pivots upward on vertical axis from fixed position, smoothly revealing vertical space above while maintaining horizontal framing and creating upward scanning motion"
            },
            "Tilt Down": {
                "Basic": "pivoting camera downward to show lower elements",
                "Detailed": "camera pivots downward on vertical axis from fixed position, smoothly revealing vertical space below while maintaining horizontal framing and creating downward scanning motion"
            },
            # Tracking movements
            "Dolly In (Forward)": {
                "Basic": "moving camera forward closer to subject",
                "Detailed": "camera moves smoothly forward on straight track path, progressively filling more frame with subject to emphasize details, textures, and importance while maintaining perspective relationships"
            },
            "Dolly Out (Backward)": {
                "Basic": "moving camera backward away from subject",
                "Detailed": "camera moves smoothly backward on straight track path, progressively revealing more environmental context and spatial relationships as subject becomes smaller, providing situational awareness"
            },
            "Truck Left (Lateral)": {
                "Basic": "moving camera left parallel to subject",
                "Detailed": "camera moves laterally left on horizontal track parallel to subject plane, maintaining consistent distance while revealing new angles and spatial relationships through lateral motion"
            },
            "Truck Right (Lateral)": {
                "Basic": "moving camera right parallel to subject",
                "Detailed": "camera moves laterally right on horizontal track parallel to subject plane, maintaining consistent distance while revealing new angles and spatial relationships through lateral motion"
            },
            # Vertical movements
            "Pedestal Up (Raise)": {
                "Basic": "raising entire camera vertically upward",
                "Detailed": "entire camera raises vertically upward maintaining framing and angle, creating smooth vertical reveal different from tilt by moving entire camera position rather than pivoting"
            },
            "Pedestal Down (Lower)": {
                "Basic": "lowering entire camera vertically downward",
                "Detailed": "entire camera lowers vertically downward maintaining framing and angle, creating smooth vertical reveal different from tilt by moving entire camera position rather than pivoting"
            },
            # Circular movements
            "Arc Left": {
                "Basic": "moving camera in circular arc to the left",
                "Detailed": "camera moves in circular arc pattern to the left around subject, maintaining relatively consistent distance while revealing subject from progressively changing angles in smooth orbital motion"
            },
            "Arc Right": {
                "Basic": "moving camera in circular arc to the right",
                "Detailed": "camera moves in circular arc pattern to the right around subject, maintaining relatively consistent distance while revealing subject from progressively changing angles in smooth orbital motion"
            },
            # Lens movements
            "Zoom In (Focal Length)": {
                "Basic": "optically zooming in without moving camera",
                "Detailed": "focal length increases while camera remains stationary, magnifying subject and compressing perspective, creating different visual effect than dolly with characteristic perspective flattening"
            },
            "Zoom Out (Focal Length)": {
                "Basic": "optically zooming out without moving camera",
                "Detailed": "focal length decreases while camera remains stationary, revealing wider field of view and expanding perspective, creating different visual effect than dolly with characteristic perspective widening"
            },
        }

        if "None" in detail_level or camera_movement == "Static (No Movement)":
            return ""
        elif "Basic" in detail_level:
            return explanations.get(camera_movement, {}).get("Basic", "")
        else:
            return explanations.get(camera_movement, {}).get("Detailed", "")


# Node registration
NODE_CLASS_MAPPINGS = {
    "ArchAi3D_Object_Focus_Camera_V7": ArchAi3D_Object_Focus_Camera_V7
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_Object_Focus_Camera_V7": "🎬 Object Focus Camera v7 (Pro Cinema)"
}
