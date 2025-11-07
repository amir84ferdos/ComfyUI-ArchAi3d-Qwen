"""
Cinematography Prompt Builder
Based on Nanobanan's 5-Ingredient Formula + Professional Enhancements

Author: Amir Ferdos (ArchAi3d)
Based on Nanobanan's proven simple formula with professional extensions
"""

import os
import yaml

class ArchAi3D_Cinematography_Prompt_Builder:
    """
    Simple yet powerful cinematography prompt builder.

    Core Philosophy:
    - Layer 1 (Required): Nanobanan's 5 Ingredients - Simple & Effective
    - Layer 2 (Optional): Professional cinematography enhancements
    - Layer 3 (Optional): Material details (37 presets)
    - Layer 4 (Optional): Quality presets (15 presets)

    The 5 Ingredients:
    1. Subject - What to photograph
    2. Shot Type - How to frame it
    3. Angle - Where the camera is
    4. Focus/DOF - What's sharp and what's blurred
    5. Style/Mood - The overall vibe
    """

    # Shot size to distance/lens/DOF mapping
    SHOT_DEFAULTS = {
        "Extreme Close-Up (ECU)": {
            "distance": 0.3,
            "lens": "Macro (Close-Up)",
            "dof": "Very Shallow",
            "description": "Extreme detail, tiny subject area"
        },
        "Close-Up (CU)": {
            "distance": 0.8,
            "lens": "Portrait (85mm)",
            "dof": "Shallow",
            "description": "Head and shoulders, intimate"
        },
        "Medium Close-Up (MCU)": {
            "distance": 1.2,
            "lens": "Portrait (85mm)",
            "dof": "Shallow to Medium",
            "description": "Chest up, personal"
        },
        "Medium Shot (MS)": {
            "distance": 2.5,
            "lens": "Normal (50mm)",
            "dof": "Medium",
            "description": "Waist up, conversational"
        },
        "Medium Long Shot (MLS)": {
            "distance": 3.5,
            "lens": "Normal (50mm)",
            "dof": "Medium to Deep",
            "description": "Knees up, shows environment"
        },
        "Full Shot (FS)": {
            "distance": 4.5,
            "lens": "Normal (50mm)",
            "dof": "Deep",
            "description": "Head to toes, complete subject"
        },
        "Wide Shot (WS)": {
            "distance": 6.5,
            "lens": "Wide Angle (24-35mm)",
            "dof": "Deep",
            "description": "Subject in environment"
        },
        "Extreme Wide Shot (EWS)": {
            "distance": 10.0,
            "lens": "Ultra Wide (14-24mm)",
            "dof": "Very Deep",
            "description": "Landscape, establishing shot"
        }
    }

    @classmethod
    def INPUT_TYPES(cls):
        # Load material presets
        config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config")
        materials_path = os.path.join(config_dir, "materials.yaml")

        material_options = ["None (Manual entry)"]
        try:
            with open(materials_path, 'r', encoding='utf-8') as f:
                materials_data = yaml.safe_load(f)
                if materials_data and 'materials' in materials_data:
                    material_options.extend(materials_data['materials'].keys())
        except:
            pass

        return {
            "required": {
                # ===== NANOBANAN'S 5 INGREDIENTS (CORE) =====

                # 1. SUBJECT - What to photograph
                "target_subject": ("STRING", {
                    "default": "the watch",
                    "multiline": False,
                    "tooltip": "What to photograph (e.g., 'the watch', 'the black stove', 'character face')"
                }),

                # 2. SHOT TYPE - How to frame it
                "shot_type": ([
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
                    "tooltip": "How close are we? ECU=super close, WS=showing environment"
                }),

                # 3. ANGLE - Where the camera is
                "camera_angle": ([
                    "Eye Level",
                    "Shoulder Level",
                    "High Angle (looking down)",
                    "Low Angle (looking up)",
                    "Bird's Eye View (overhead)",
                    "Worm's Eye View (ground up)",
                    "Dutch Angle (tilted)",
                    "30-Degree Angled"
                ], {
                    "default": "Eye Level",
                    "tooltip": "Where is the camera? Eye level=normal, Low angle=powerful, High angle=vulnerable"
                }),

                # 3B. HORIZONTAL ANGLE - Camera position around object
                "horizontal_angle": ([
                    "Front View (0°)",
                    "Angled Left 15°",
                    "Angled Left 30°",
                    "Angled Left 45°",
                    "Side Left (90°)",
                    "Back View (180°)",
                    "Side Right (90°)",
                    "Angled Right 45°",
                    "Angled Right 30°",
                    "Angled Right 15°"
                ], {
                    "default": "Front View (0°)",
                    "tooltip": "Horizontal camera position around the object:\n"
                               "• Front (0°) = Straight-on view\n"
                               "• Angled (15-45°) = Corner/three-quarter view\n"
                               "• Side (90°) = Profile view\n"
                               "• Back (180°) = Rear view"
                }),

                # 3C. AUTO-FACING - Automatically point camera at subject
                "auto_facing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Add 'Facing [subject] directly' at the beginning of prompt for maximum attention.\n"
                               "• True = Explicitly states camera is facing the subject\n"
                               "• False = Standard prompt without facing directive\n"
                               "• Recommended: True for better subject focus and composition"
                }),

                # 4. FOCUS/DOF - What's sharp and what's blurred
                "depth_of_field": ([
                    "Auto (based on shot size)",
                    "--- Shallow (blurry background) ---",
                    "Extreme Shallow",
                    "Very Shallow",
                    "Shallow",
                    "--- Deep (everything sharp) ---",
                    "Medium",
                    "Deep",
                    "Very Deep"
                ], {
                    "default": "Auto (based on shot size)",
                    "tooltip": "Auto=smart choice based on shot size, Shallow=blurry background, Deep=everything sharp"
                }),

                # 5. STYLE/MOOD - The overall vibe
                "style_mood": ([
                    "Natural/Neutral",
                    "Cinematic",
                    "Dramatic",
                    "Clean/Minimalist",
                    "Architectural",
                    "Editorial",
                    "Commercial Product",
                    "Documentary",
                    "Artistic/Abstract",
                    "Professional Studio"
                ], {
                    "default": "Natural/Neutral",
                    "tooltip": "What's the vibe? Cinematic=movie-like, Clean=minimalist, Dramatic=high contrast"
                }),

                # ===== PROFESSIONAL ENHANCEMENTS (OPTIONAL) =====

                "prompt_language": ([
                    "English (Simple & Clear)",
                    "Chinese (Best for dx8152 LoRAs)",
                    "Hybrid (Chinese + English)"
                ], {
                    "default": "Hybrid (Chinese + English)",
                    "tooltip": "English=easy to read, Chinese=best performance with dx8152 LoRAs, Hybrid=both"
                }),
            },
            "optional": {
                # ===== LAYER 2: PROFESSIONAL CINEMATOGRAPHY =====

                "lens_type_override": ([
                    "Auto (from shot size)",
                    "--- Close-Up Lenses ---",
                    "Macro (Close-Up)",
                    "Portrait (85mm)",
                    "--- Standard Lenses ---",
                    "Normal (50mm)",
                    "--- Wide Lenses ---",
                    "Wide Angle (24-35mm)",
                    "Ultra Wide (14-24mm)",
                    "--- Telephoto ---",
                    "Telephoto (100-200mm)"
                ], {
                    "default": "Auto (from shot size)",
                    "tooltip": "Auto=smart choice, Manual=override for special effects"
                }),

                "perspective_correction": ([
                    "Natural (Standard Lens)",
                    "Architectural (Keep Verticals Straight)",
                    "Tilt-Shift (Full Perspective Control)"
                ], {
                    "default": "Natural (Standard Lens)",
                    "tooltip": "Control vertical line convergence for architectural photography:\n"
                               "• Natural = Standard perspective with natural converging lines\n"
                               "• Architectural = Keep vertical lines parallel (requires eye-level framing)\n"
                               "• Tilt-Shift = Professional perspective correction with selective focus plane"
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
                    "Truck Left",
                    "Truck Right",
                    "--- Other ---",
                    "Arc Left",
                    "Arc Right",
                    "Zoom In",
                    "Zoom Out"
                ], {
                    "default": "Static (No Movement)",
                    "tooltip": "Static=still photo, Dolly=camera moves on track, Pan=camera rotates"
                }),

                "lighting_style": ([
                    "Auto/Natural",
                    "Bright & Even",
                    "Soft & Diffused",
                    "Soft Directional & Dramatic",
                    "Hard Dramatic",
                    "Natural Window Light",
                    "Golden Hour Sunset",
                    "Overcast Daylight",
                    "Clean Architectural",
                    "Studio Lighting"
                ], {
                    "default": "Auto/Natural",
                    "tooltip": "How is it lit? Golden Hour=warm sunset, Soft=flattering, Dramatic=high contrast"
                }),

                # ===== LAYER 3: MATERIAL DETAILS (37 PRESETS) =====

                "material_detail_preset": (material_options, {
                    "default": "None (Manual entry)",
                    "tooltip": "Add material details (Crystal Facets, Polished Metal, etc.) - from v6"
                }),

                # ===== LAYER 4: QUALITY PRESETS (15 OPTIONS) =====

                "photography_quality_preset": ([
                    "None (Manual entry)",
                    "--- Professional ---",
                    "Cinematic Quality",
                    "Commercial Product",
                    "Editorial Quality",
                    "Architectural Precision",
                    "Fine Art Photography",
                    "--- Specific Styles ---",
                    "Razor Sharp Focus",
                    "High Dynamic Range",
                    "Bokeh Background",
                    "Professional Lighting",
                    "Natural Photorealistic",
                    "Studio Quality",
                    "Magazine Quality",
                    "Documentary Authentic",
                    "Technical Documentation"
                ], {
                    "default": "None (Manual entry)",
                    "tooltip": "Add professional quality enhancement - from v6"
                }),

                # ===== CUSTOM DETAILS =====

                "custom_details": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Add compositional specifics beyond the 5 ingredients. Examples:\n"
                               "• 'The entire stove is centered in the frame, clearly showing it, the marble backsplash, and the range hood above'\n"
                               "• 'focusing on the intricate details of a single burner and the cast-iron grate'\n"
                               "• 'showing dial and hands clearly'\n"
                               "• 'The vantage point is inches away, creating an extremely shallow depth of field'\n"
                               "• 'dissolves into a soft, blurred bokeh'\n"
                               "• 'The lighting is bright and even, keeping the entire area in sharp focus'"
                }),

                "show_advanced_info": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Show detailed technical information in description output"
                })
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("simple_prompt", "professional_prompt", "system_prompt", "description")
    FUNCTION = "generate_cinematography_prompt"
    CATEGORY = "ArchAi3d/Qwen/Camera"

    def __init__(self):
        # Load material details from v6
        self.materials_data = self._load_materials()
        self.quality_presets = self._load_quality_presets()

    def _load_materials(self):
        """Load material presets from YAML"""
        try:
            config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config")
            materials_path = os.path.join(config_dir, "materials.yaml")

            with open(materials_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                return data.get('materials', {}) if data else {}
        except:
            return {}

    def _load_quality_presets(self):
        """Photography quality presets from v6"""
        return {
            "Cinematic Quality": "cinematic photography with dramatic lighting, film-like color grading, and movie-quality production values",
            "Commercial Product": "commercial product photography with clean presentation, optimal angles, and marketing-ready quality",
            "Editorial Quality": "editorial photography quality with intentional composition, magazine-level presentation, and professional styling",
            "Architectural Precision": "with architectural photography precision, geometric accuracy, and structural detail clarity",
            "Fine Art Photography": "fine art photography aesthetic with artistic interpretation, gallery-worthy composition, and creative vision",
            "Razor Sharp Focus": "with razor-sharp focus, tack-sharp details, and microscopic clarity",
            "High Dynamic Range": "with high dynamic range, preserved highlights and shadows, and balanced exposure",
            "Bokeh Background": "with beautiful bokeh background blur, creamy out-of-focus areas, and subject isolation",
            "Professional Lighting": "with studio-quality lighting, balanced exposure, and perfect highlight-shadow detail",
            "Natural Photorealistic": "with natural photorealistic quality, authentic colors, and true-to-life representation",
            "Studio Quality": "with professional studio quality, controlled environment, and flawless execution",
            "Magazine Quality": "magazine-quality photography with editorial standards, professional retouching, and publication-ready presentation",
            "Documentary Authentic": "documentary photography style with authentic capture, candid moments, and unposed naturalness",
            "Technical Documentation": "technical documentation photography with accurate representation, clear details, and informational value"
        }

    def number_to_words(self, num):
        """Convert number to written words - prevents numbers appearing in images!"""
        # Handle decimals
        if '.' in str(num):
            parts = str(num).split('.')
            whole = int(parts[0])
            decimal = int(parts[1]) if len(parts) > 1 else 0

            if decimal == 0:
                return self._int_to_words(whole)
            elif decimal == 5:
                return f"{self._int_to_words(whole)} and a half"
            else:
                return f"{self._int_to_words(whole)} point {self._int_to_words(decimal)}"
        else:
            return self._int_to_words(int(num))

    def _int_to_words(self, n):
        """Convert integer to words"""
        words_map = {
            0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
            5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
            10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
            14: "fourteen", 15: "fifteen"
        }
        return words_map.get(n, str(n))

    def get_shot_abbreviation(self, shot_type):
        """Extract abbreviation from shot type"""
        abbreviations = {
            "Extreme Close-Up (ECU)": "ECU",
            "Close-Up (CU)": "CU",
            "Medium Close-Up (MCU)": "MCU",
            "Medium Shot (MS)": "MS",
            "Medium Long Shot (MLS)": "MLS",
            "Full Shot (FS)": "FS",
            "Wide Shot (WS)": "WS",
            "Extreme Wide Shot (EWS)": "EWS"
        }
        return abbreviations.get(shot_type, shot_type)

    def get_shot_full_name(self, shot_type):
        """Extract full natural language name from shot type (not abbreviation)"""
        full_names = {
            "Extreme Close-Up (ECU)": "extreme close-up",
            "Close-Up (CU)": "close-up",
            "Medium Close-Up (MCU)": "medium close-up",
            "Medium Shot (MS)": "medium shot",
            "Medium Long Shot (MLS)": "medium long shot",
            "Full Shot (FS)": "full shot",
            "Wide Shot (WS)": "wide shot",
            "Extreme Wide Shot (EWS)": "extreme wide shot"
        }
        return full_names.get(shot_type, shot_type.lower().split("(")[0].strip())

    def _get_horizontal_angle_description(self, horizontal_angle, prompt_language="English (Simple & Clear)"):
        """Convert horizontal angle to natural language description"""

        # English descriptions
        english_descriptions = {
            "Front View (0°)": "directly facing the subject from the front",
            "Angled Left 15°": "from fifteen degrees to the left",
            "Angled Left 30°": "from thirty degrees to the left for a corner perspective",
            "Angled Left 45°": "from forty-five degrees to the left for a three-quarter view",
            "Side Left (90°)": "from the left side for a profile view",
            "Back View (180°)": "from behind the subject",
            "Side Right (90°)": "from the right side for a profile view",
            "Angled Right 45°": "from forty-five degrees to the right for a three-quarter view",
            "Angled Right 30°": "from thirty degrees to the right for a corner perspective",
            "Angled Right 15°": "from fifteen degrees to the right"
        }

        # Chinese descriptions
        chinese_descriptions = {
            "Front View (0°)": "正面拍摄,直接面对主体",
            "Angled Left 15°": "从左侧15度拍摄",
            "Angled Left 30°": "从左侧30度拍摄,呈现转角视角",
            "Angled Left 45°": "从左侧45度拍摄,呈现四分之三视角",
            "Side Left (90°)": "从左侧拍摄,呈现侧面视角",
            "Back View (180°)": "从背面拍摄主体",
            "Side Right (90°)": "从右侧拍摄,呈现侧面视角",
            "Angled Right 45°": "从右侧45度拍摄,呈现四分之三视角",
            "Angled Right 30°": "从右侧30度拍摄,呈现转角视角",
            "Angled Right 15°": "从右侧15度拍摄"
        }

        if horizontal_angle == "Front View (0°)":
            # Front view doesn't need explicit mention in most cases
            return ("", "")  # (english, chinese)

        english = english_descriptions.get(horizontal_angle, horizontal_angle.lower())
        chinese = chinese_descriptions.get(horizontal_angle, horizontal_angle)

        return (english, chinese)

    def _get_perspective_correction_prompting(self, perspective_correction, prompt_language="English (Simple & Clear)"):
        """Generate perspective correction guidance text"""

        if perspective_correction == "Natural (Standard Lens)":
            # No special prompting needed for natural perspective
            return ("", "")

        elif perspective_correction == "Architectural (Keep Verticals Straight)":
            english = ("with careful framing to keep all vertical lines parallel and prevent perspective distortion, "
                      "maintaining straight architectural lines throughout the frame")
            chinese = "保持所有垂直线平行,防止透视畸变,确保建筑线条笔直"
            return (english, chinese)

        elif perspective_correction == "Tilt-Shift (Full Perspective Control)":
            english = ("using a tilt-shift lens for perspective correction to keep all vertical lines perfectly parallel, "
                      "with precise control over the focus plane and no keystoning distortion")
            chinese = "使用移轴镜头进行透视校正,保持所有垂直线完美平行,精确控制焦平面,无梯形失真"
            return (english, chinese)

        return ("", "")

    def validate_parameters(self, shot_type, dof, lens_type, camera_angle="Eye Level",
                           perspective_correction="Natural (Standard Lens)"):
        """Validate parameter combinations and return warnings"""
        warnings = []

        # Wide shot + Shallow DOF conflict
        if shot_type in ["Wide Shot (WS)", "Extreme Wide Shot (EWS)"]:
            if dof in ["Extreme Shallow", "Very Shallow", "Shallow"]:
                warnings.append("Wide shots typically need deep DOF to show environment")

        # Macro lens + Wide shot conflict
        if lens_type == "Macro (Close-Up)":
            if shot_type not in ["Extreme Close-Up (ECU)", "Close-Up (CU)"]:
                warnings.append("Macro lens designed for extreme close-ups")

        # Telephoto + Wide shot
        if lens_type == "Telephoto (100-200mm)":
            if shot_type in ["Wide Shot (WS)", "Extreme Wide Shot (EWS)"]:
                warnings.append("Telephoto lens has narrow FOV, not ideal for wide shots")

        # Perspective correction + non-level camera angle conflict
        if perspective_correction in ["Architectural (Keep Verticals Straight)",
                                       "Tilt-Shift (Full Perspective Control)"]:
            if camera_angle in ["High Angle (looking down)", "Low Angle (looking up)",
                               "Bird's Eye View (overhead)", "Worm's Eye View (ground up)"]:
                warnings.append(
                    "⚠️ Perspective correction requires eye-level camera angle. "
                    "Vertical lines will converge with tilted camera positions. "
                    "Use 'Eye Level' or 'Shoulder Level' for straight verticals."
                )

        return warnings

    def _get_cinematography_system_prompt(self, prompt_language, show_advanced_info,
                                          material_preset, quality_preset, perspective_correction="Natural (Standard Lens)"):
        """
        Generate dynamic system prompt based on configuration.

        Returns different prompts based on:
        - Language mode (English/Chinese/Hybrid)
        - Whether advanced features are used
        - Material/Quality presets enabled
        - Perspective correction mode (architectural straight verticals)

        Aligned with research findings from vision-language camera control PDF.
        """

        # Architectural perspective guidance (appended to all modes if enabled)
        architectural_guidance = ""
        if perspective_correction in ["Architectural (Keep Verticals Straight)", "Tilt-Shift (Full Perspective Control)"]:
            architectural_guidance = (
                " IMPORTANT: Maintain parallel vertical lines in architectural photography. "
                "Keep the camera level (no upward or downward tilt) to prevent converging verticals and keystoning. "
                "All vertical architectural elements (walls, doors, columns, windows) must remain straight and parallel in the frame. "
                "This requires eye-level camera positioning without vertical angle deviation."
            )

        # PROFESSIONAL MODE: Chinese + dx8152 LoRA optimization + presets
        if (prompt_language in ["Chinese (Best for dx8152 LoRAs)", "Hybrid (Chinese + English)"]
            and (material_preset != "None (Manual entry)" or quality_preset != "None (Manual entry)")):
            return (
                "You are a professional cinematographer specializing in Qwen-VL camera control. "
                "Execute precise camera positioning using industry-standard shot sizes (ECU to EWS), "
                "camera angles (eye level to bird's eye), and lens characteristics (14mm to 200mm+). "
                "Maintain subject identity across viewpoint changes while allowing visual appearance "
                "to transform appropriately. Use distance-based positioning (e.g., '2.5 meters') "
                "rather than degree-based angular specifications for consistent results. "
                "Process Chinese cinematography terms (构图, 查看) with high accuracy for dx8152 LoRA compatibility."
                + architectural_guidance
            )

        # RESEARCH-VALIDATED MODE: Advanced technical mode with PDF findings
        elif show_advanced_info:
            return (
                "You are an expert cinematographer trained in vision-language spatial reasoning. "
                "Follow the five-ingredient prompting framework: subject description, shot type and framing, "
                "angle and vantage point, focus and depth of field, style or mood. "
                "Process camera instructions through natural language spatial relationships—no pixel coordinates. "
                "Maintain geometric consistency by preserving subject identity (semantic pathway) while "
                "adapting visual appearance (reconstructive pathway) across viewpoint changes. "
                "Use M-RoPE position embeddings for 3D spatial understanding. "
                "Optimal guidance scale: 6-8 for camera control workflows. "
                "Distance-based positioning ('2.5 meters away') produces more reliable results than "
                "degree-based angular specifications ('45 degrees counterclockwise')."
                + architectural_guidance
            )

        # SIMPLE/BEGINNER MODE: Nanobanan's 5-ingredient framework (default)
        else:
            return (
                "You are a professional photographer following the five-ingredient framework: "
                "subject, shot type, angle, focus/depth of field, and style. "
                "Execute camera positioning using natural language descriptions of relative positions, "
                "distances (in meters), and viewpoints. Interpret cinematographic terminology accurately "
                "(extreme close-up, close-up, medium shot, wide shot, etc.) and maintain visual consistency "
                "across viewpoint changes. Preserve subject identity while allowing lighting, perspective, "
                "and visual details to change naturally with camera position."
                + architectural_guidance
            )

    def generate_cinematography_prompt(self, target_subject, shot_type, camera_angle,
                                      horizontal_angle, auto_facing,
                                      depth_of_field, style_mood, prompt_language,
                                      lens_type_override="Auto (from shot size)",
                                      perspective_correction="Natural (Standard Lens)",
                                      camera_movement="Static (No Movement)",
                                      lighting_style="Auto/Natural",
                                      material_detail_preset="None (Manual entry)",
                                      photography_quality_preset="None (Manual entry)",
                                      custom_details="",
                                      show_advanced_info=False):

        # Get shot defaults
        shot_defaults = self.SHOT_DEFAULTS[shot_type]
        distance = shot_defaults["distance"]

        # Determine lens (with tilt-shift auto-selection for perspective correction)
        if perspective_correction == "Tilt-Shift (Full Perspective Control)":
            lens_type = "Tilt-Shift (Perspective Control)"
        elif lens_type_override == "Auto (from shot size)":
            lens_type = shot_defaults["lens"]
        else:
            lens_type = lens_type_override

        # Determine DOF
        if depth_of_field == "Auto (based on shot size)":
            dof = shot_defaults["dof"]
        else:
            dof = depth_of_field.replace("---", "").strip().split("(")[0].strip()

        # Validate parameters (including perspective correction compatibility)
        warnings = self.validate_parameters(shot_type, dof, lens_type, camera_angle, perspective_correction)

        # Generate SIMPLE prompt (Nanobanan style with horizontal angle + perspective correction + auto_facing)
        simple_prompt = self._generate_simple_prompt(
            target_subject, shot_type, camera_angle, dof, style_mood,
            distance, lighting_style, custom_details, horizontal_angle, auto_facing, perspective_correction, prompt_language
        )

        # Generate PROFESSIONAL prompt (v7 style with Chinese + horizontal angle + perspective + auto_facing)
        professional_prompt = self._generate_professional_prompt(
            target_subject, shot_type, camera_angle, lens_type, camera_movement,
            distance, dof, lighting_style, style_mood, material_detail_preset,
            photography_quality_preset, custom_details, prompt_language,
            horizontal_angle, auto_facing, perspective_correction
        )

        # Generate SYSTEM PROMPT (dynamic based on configuration + perspective correction)
        system_prompt = self._get_cinematography_system_prompt(
            prompt_language, show_advanced_info,
            material_detail_preset, photography_quality_preset, perspective_correction
        )

        # Generate description
        description = self._generate_description(
            shot_type, camera_angle, lens_type, dof, style_mood,
            camera_movement, warnings, show_advanced_info
        )

        return (simple_prompt, professional_prompt, system_prompt, description)

    def _generate_simple_prompt(self, subject, shot_type, angle, dof, style,
                                distance, lighting, custom_details,
                                horizontal_angle="Front View (0°)",
                                auto_facing=True,
                                perspective_correction="Natural (Standard Lens)",
                                prompt_language="English (Simple & Clear)"):
        """
        Generate Simple English prompt (Nanobanan style)

        Pattern: "[facing subject], A [angle] [shot_type] of [subject], taken from [distance], [horizontal_angle],
                 [perspective_correction], with [dof] and [style], [lighting], [custom_details]"

        Note: auto_facing is placed FIRST for maximum attention weight (user's experience-based observation)
        """
        # Clean up angle description
        angle_clean = angle.replace(" (looking down)", "").replace(" (looking up)", "").replace(" (overhead)", "").replace(" (ground up)", "").replace(" (tilted)", "")

        # Get FULL shot name (not abbreviation) for natural language
        shot_full = self.get_shot_full_name(shot_type)

        # Convert distance to words
        distance_words = self.number_to_words(distance)

        # Get horizontal angle description
        horizontal_desc_en, horizontal_desc_zh = self._get_horizontal_angle_description(horizontal_angle, prompt_language)

        # Get perspective correction prompting
        perspective_desc_en, perspective_desc_zh = self._get_perspective_correction_prompting(perspective_correction, prompt_language)

        # Build prompt parts
        parts = []

        # AUTO-FACING: Add at the VERY BEGINNING for maximum attention weight
        if auto_facing:
            parts.append(f"Facing {subject} directly")

        # Opening: "An [angle] [shot] of [subject]"
        if angle_clean.lower() == "eye level":
            parts.append(f"An eye-level {shot_full} of {subject}")
        else:
            parts.append(f"A {angle_clean.lower()} {shot_full} of {subject}")

        # Distance: "taken from a vantage point [distance] meters away" (ALWAYS specific)
        # For distances under 1 meter, use "centimeters" for better readability
        if distance < 1.0:
            cm_distance = int(distance * 100)
            cm_words = self._int_to_words(cm_distance)
            parts.append(f"taken from a vantage point {cm_words} centimeters away")
        else:
            parts.append(f"taken from a vantage point {distance_words} meters away")

        # Horizontal angle (if not front view)
        if horizontal_desc_en:
            parts.append(f"positioned {horizontal_desc_en}")

        # Perspective correction (if enabled)
        if perspective_desc_en:
            parts.append(perspective_desc_en)

        # DOF description
        if "shallow" in dof.lower():
            parts.append(f"with {dof.lower()} depth of field creating blurred background")
        elif "deep" in dof.lower():
            parts.append(f"with {dof.lower()} depth of field keeping everything in focus")
        else:
            parts.append(f"with {dof.lower()} depth of field")

        # Style/Mood
        if style != "Natural/Neutral":
            style_desc = style.lower().replace("/", " and ")
            parts.append(f"in {style_desc} style")

        # Lighting
        if lighting != "Auto/Natural" and lighting != "Auto":
            lighting_desc = lighting.lower()
            parts.append(f"with {lighting_desc}")

        # Custom details
        if custom_details.strip():
            parts.append(custom_details.strip())

        # Join with commas and periods appropriately
        prompt = parts[0]
        if len(parts) > 1:
            prompt += ", " + ", ".join(parts[1:])

        return prompt

    def _generate_professional_prompt(self, subject, shot_type, angle, lens, movement,
                                     distance, dof, lighting, style, material_preset,
                                     quality_preset, custom_details, language,
                                     horizontal_angle="Front View (0°)",
                                     auto_facing=True,
                                     perspective_correction="Natural (Standard Lens)"):
        """
        Generate Professional prompt (v7 style with Chinese cinematography terms + horizontal angle + perspective + auto_facing)

        Pattern: "[facing], Next Scene: 将镜头转为[LENS], [SHOT]构图, [ANGLE]查看[SUBJECT], [HORIZONTAL], [PERSPECTIVE], [DETAILS]"

        Note: auto_facing placed at START for maximum attention weight
        """
        prompt_parts = []

        # AUTO-FACING: Add at BEGINNING for maximum attention (before "Next Scene:")
        if auto_facing:
            if language in ["Chinese (Best for dx8152 LoRAs)", "Hybrid (Chinese + English)"]:
                prompt_parts.append(f"面对{subject}")  # "Facing {subject}"
            else:
                prompt_parts.append(f"Facing {subject} directly")

        # Always add "Next Scene:" for dx8152 LoRAs
        prompt_parts.append("Next Scene:")

        # Get horizontal angle and perspective correction descriptions
        horizontal_desc_en, horizontal_desc_zh = self._get_horizontal_angle_description(horizontal_angle, language)
        perspective_desc_en, perspective_desc_zh = self._get_perspective_correction_prompting(perspective_correction, language)

        # CHINESE CINEMATOGRAPHY SECTION (if Chinese or Hybrid)
        if language in ["Chinese (Best for dx8152 LoRAs)", "Hybrid (Chinese + English)"]:
            chinese_parts = []

            # Lens change: 将镜头转为X
            lens_chinese = self._get_lens_chinese(lens)
            chinese_parts.append(f"将镜头转为{lens_chinese}")

            # Shot type: X构图
            shot_chinese = self._get_shot_chinese(shot_type)
            chinese_parts.append(f"{shot_chinese}构图")

            # Angle + Subject: [ANGLE]查看[SUBJECT]
            angle_chinese = self._get_angle_chinese(angle)
            if angle_chinese:
                chinese_parts.append(f"{angle_chinese}查看{subject}")
            else:
                chinese_parts.append(f"查看{subject}")

            # Horizontal angle (if not front view)
            if horizontal_desc_zh:
                chinese_parts.append(horizontal_desc_zh)

            # Distance
            distance_chinese = self._get_distance_chinese(distance)
            chinese_parts.append(f"距离{distance_chinese}")

            # Perspective correction (if enabled)
            if perspective_desc_zh:
                chinese_parts.append(perspective_desc_zh)

            # Movement (if not static)
            if movement != "Static (No Movement)":
                movement_chinese = self._get_movement_chinese(movement)
                if movement_chinese:
                    chinese_parts.append(movement_chinese)

            prompt_parts.append("，".join(chinese_parts))

        # ENGLISH DETAILS SECTION
        english_parts = []

        # Material details
        if material_preset != "None (Manual entry)" and material_preset in self.materials_data:
            material_desc = self.materials_data[material_preset].get('description', '')
            if material_desc:
                english_parts.append(material_desc)

        # Quality preset
        if quality_preset != "None (Manual entry)" and quality_preset in self.quality_presets:
            quality_desc = self.quality_presets[quality_preset]
            english_parts.append(quality_desc)

        # Custom details
        if custom_details.strip():
            english_parts.append(custom_details.strip())

        # Add English parts with proper separator
        if english_parts:
            if language == "Chinese (Best for dx8152 LoRAs)":
                prompt_parts.append("，" + "，".join(english_parts))
            else:
                prompt_parts.append(", " + ", ".join(english_parts))

        # Join all parts
        if language == "English (Simple & Clear)":
            # Pure English mode - simplified with horizontal angle + perspective
            # Build base without auto_facing (already in prompt_parts[0] if enabled)
            base_parts = []

            # Check if auto_facing was added
            if len(prompt_parts) > 1 and "Facing" in prompt_parts[0]:
                base_parts.append(prompt_parts[0])  # Add facing directive
                base_parts.append(prompt_parts[1])  # Add "Next Scene:"
            else:
                base_parts.append(prompt_parts[0])  # Just "Next Scene:"

            # Add main prompt
            base_parts.append(f"Change to {lens}, {self.get_shot_abbreviation(shot_type)} framing, {angle} viewing {subject}")

            if horizontal_desc_en:
                base_parts.append(f"positioned {horizontal_desc_en}")
            if perspective_desc_en:
                base_parts.append(perspective_desc_en)
            if english_parts:
                base_parts.extend(english_parts)

            return ", ".join(base_parts)
        else:
            return " ".join(prompt_parts)

    def _get_lens_chinese(self, lens):
        """Get Chinese translation for lens type"""
        lens_map = {
            "Macro (Close-Up)": "微距镜头",
            "Portrait (85mm)": "人像镜头(85mm)",
            "Normal (50mm)": "标准镜头(50mm)",
            "Wide Angle (24-35mm)": "广角镜头(24-35mm)",
            "Ultra Wide (14-24mm)": "超广角镜头(14-24mm)",
            "Telephoto (100-200mm)": "长焦镜头(100-200mm)"
        }
        return lens_map.get(lens, lens)

    def _get_shot_chinese(self, shot_type):
        """Get Chinese translation for shot size"""
        shot_map = {
            "Extreme Close-Up (ECU)": "特写",
            "Close-Up (CU)": "近景",
            "Medium Close-Up (MCU)": "中近景",
            "Medium Shot (MS)": "中景",
            "Medium Long Shot (MLS)": "中远景",
            "Full Shot (FS)": "全景",
            "Wide Shot (WS)": "远景",
            "Extreme Wide Shot (EWS)": "大远景"
        }
        return shot_map.get(shot_type, shot_type)

    def _get_angle_chinese(self, angle):
        """Get Chinese translation for camera angle"""
        angle_map = {
            "Eye Level": "平视",
            "Shoulder Level": "肩部视角",
            "High Angle (looking down)": "俯视",
            "Low Angle (looking up)": "仰视",
            "Bird's Eye View (overhead)": "鸟瞰",
            "Worm's Eye View (ground up)": "虫瞰",
            "Dutch Angle (tilted)": "倾斜角度",
            "30-Degree Angled": "30度角"
        }
        return angle_map.get(angle, "")

    def _get_distance_chinese(self, distance):
        """
        Convert distance to Chinese words with specific meter values

        Returns exact distance in Chinese characters (e.g., "四米" for 4.0)
        instead of generic descriptions like "远距离" (far distance)
        """
        # Chinese number words
        chinese_numbers = {
            0: "零", 1: "一", 2: "两", 3: "三", 4: "四",
            5: "五", 6: "六", 7: "七", 8: "八", 9: "九",
            10: "十", 15: "十五", 20: "二十"
        }

        # Handle decimals (e.g., 2.5 = "两米半", 0.8 = "零点八米")
        if distance == int(distance):
            # Whole number
            dist_int = int(distance)
            if dist_int in chinese_numbers:
                return f"{chinese_numbers[dist_int]}米"
            else:
                return f"{dist_int}米"  # Fallback to Arabic numerals
        elif distance % 1 == 0.5:
            # Half meter (e.g., 2.5 = "两米半")
            whole = int(distance)
            if whole == 0:
                return "半米"
            elif whole in chinese_numbers:
                return f"{chinese_numbers[whole]}米半"
            else:
                return f"{whole}米半"
        else:
            # Other decimals (e.g., 0.8 = "零点八米")
            whole = int(distance)
            decimal = int((distance - whole) * 10)
            if whole == 0:
                return f"零点{chinese_numbers.get(decimal, str(decimal))}米"
            else:
                whole_cn = chinese_numbers.get(whole, str(whole))
                decimal_cn = chinese_numbers.get(decimal, str(decimal))
                return f"{whole_cn}点{decimal_cn}米"

    def _get_movement_chinese(self, movement):
        """Get Chinese translation for camera movement"""
        movement_map = {
            "Pan Left": "向左平移",
            "Pan Right": "向右平移",
            "Tilt Up": "向上倾斜",
            "Tilt Down": "向下倾斜",
            "Dolly In (Forward)": "推近镜头",
            "Dolly Out (Backward)": "拉远镜头",
            "Truck Left": "左移镜头",
            "Truck Right": "右移镜头",
            "Arc Left": "弧形左移",
            "Arc Right": "弧形右移",
            "Zoom In": "推进镜头",
            "Zoom Out": "拉出镜头"
        }
        return movement_map.get(movement, "")

    def _generate_description(self, shot_type, angle, lens, dof, style,
                             movement, warnings, show_advanced):
        """Generate human-readable description with emojis"""
        shot_abbr = self.get_shot_abbreviation(shot_type)

        desc_parts = [
            f"📸 {shot_abbr}",
            f"📐 {angle}",
            f"🔍 {lens}",
            f"🎯 {dof} DOF"
        ]

        if movement != "Static (No Movement)":
            desc_parts.append(f"🎬 {movement}")

        if style != "Natural/Neutral":
            desc_parts.append(f"🎨 {style}")

        description = " | ".join(desc_parts)

        # Add warnings if any
        if warnings:
            description += "\n\n⚠️ Warnings:\n" + "\n".join(f"  • {w}" for w in warnings)

        # Add advanced info if requested
        if show_advanced:
            shot_defaults = self.SHOT_DEFAULTS[shot_type]
            description += f"\n\n📊 Technical Details:\n"
            description += f"  • Distance: {shot_defaults['distance']}m\n"
            description += f"  • Shot Description: {shot_defaults['description']}"

        return description

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "ArchAi3D_Cinematography_Prompt_Builder": ArchAi3D_Cinematography_Prompt_Builder
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_Cinematography_Prompt_Builder": "📸 Cinematography Prompt Builder"
}
