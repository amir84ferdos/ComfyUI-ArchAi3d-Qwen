# -*- coding: utf-8 -*-
"""
ArchAi3D Qwen Person Cinematographer Node (v1.0)
Author: Amir Ferdos (ArchAi3d)
Email: Amir84ferdos@gmail.com
LinkedIn: https://www.linkedin.com/in/archai3d/

Description:
    ULTIMATE all-in-one character cinematography control - like being a professional
    cameraman filming an actor. Complete control over camera movement, angles, character
    emotion, eye direction, body language, and lighting in ONE node.

    Perfect for:
    - AI movie production (ASCENDRA showreel!)
    - Character-driven scenes
    - Complex camera choreography
    - Professional cinematography
    - Frame-by-frame character consistency

Based on research:
    - QWEN_PROMPT_GUIDE comprehensive camera functions
    - Professional cinematography techniques
    - Character direction best practices
    - Identity preservation is CRITICAL
"""

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

# ============================================================================
# SYSTEM PROMPT
# ============================================================================

CINEMATOGRAPHER_SYSTEM_PROMPT = """You are a professional cinematographer filming a character performance. Execute precise camera movement around actor while maintaining focus and framing. Preserve character's identity, facial features, clothing, hairstyle, and core pose exactly throughout camera movement. Capture specified emotion, eye direction, and body language authentically. Professional cinematic lighting and composition. Character should remain recognizable and consistent across all angles."""

# ============================================================================
# CINEMATIC PRESETS - Ready-to-use scenarios
# ============================================================================

CINEMATIC_PRESETS = {
    "custom": {
        "description": "Full manual control - customize everything",
    },

    # ASCENDRA PROJECT PRESETS ⭐
    "ascendra_woman_hero": {
        "description": "ASCENDRA protagonist - determined hero shot",
        "movement": "dolly_in",
        "distance": "medium",
        "angle": "eye_level",
        "emotion": "determined",
        "intensity": "strong",
        "eyes": "at_camera",
        "pose": "standing_confident",
        "lighting": "front_rim",
        "mood": "bright_heroic",
    },
    "ascendra_man_threatening": {
        "description": "ASCENDRA villain - menacing low angle reveal",
        "movement": "orbit_right",
        "orbit_angle": 45,
        "distance": "close",
        "angle": "low_angle",
        "emotion": "menacing",
        "intensity": "strong",
        "eyes": "at_camera_intense",
        "pose": "standing_threatening",
        "lighting": "side_shadow",
        "mood": "ominous_dark",
    },
    "ascendra_transformation_start": {
        "description": "ASCENDRA transformation beginning - surprised + fearful",
        "movement": "dolly_in_slow",
        "distance": "medium",
        "angle": "eye_level",
        "emotion": "surprised_fearful",
        "intensity": "moderate",
        "eyes": "looking_down",
        "pose": "standing_neutral",
        "lighting": "front_changing",
        "mood": "transitioning_bright",
    },
    "ascendra_wings_triumph": {
        "description": "ASCENDRA final triumph - heroic victory with wings",
        "movement": "crane_down_reveal",
        "distance": "wide",
        "angle": "low_angle",
        "emotion": "triumphant",
        "intensity": "extreme",
        "eyes": "at_camera",
        "pose": "standing_victorious",
        "lighting": "dramatic_backlight",
        "mood": "bright_epic",
    },

    # GENERAL CINEMATIC PRESETS
    "hero_reveal": {
        "description": "Classic hero reveal - powerful low angle push in",
        "movement": "dolly_in_and_tilt_up",
        "distance": "medium",
        "angle": "low_angle",
        "emotion": "confident",
        "intensity": "strong",
        "eyes": "looking_up_right",
        "pose": "standing_confident",
        "lighting": "backlit_rim",
        "mood": "dramatic_heroic",
    },
    "villain_menace": {
        "description": "Villain menace - threatening orbital reveal",
        "movement": "orbit_right",
        "orbit_angle": 90,
        "distance": "close",
        "angle": "low_angle",
        "emotion": "menacing",
        "intensity": "strong",
        "eyes": "at_camera_intense",
        "pose": "standing_threatening",
        "lighting": "split_lighting",
        "mood": "dark_threatening",
    },
    "romantic_closeup": {
        "description": "Romantic intimate closeup - soft and loving",
        "movement": "dolly_in",
        "distance": "close",
        "angle": "eye_level",
        "emotion": "loving",
        "intensity": "subtle",
        "eyes": "at_camera_soft",
        "pose": "standing_relaxed",
        "lighting": "butterfly_soft",
        "mood": "soft_intimate",
    },
    "action_dynamic": {
        "description": "Dynamic action shot - energy and movement",
        "movement": "truck_right_follow",
        "distance": "medium",
        "angle": "dutch_30",
        "emotion": "determined",
        "intensity": "strong",
        "eyes": "forward_focused",
        "pose": "action_running",
        "lighting": "harsh_dramatic",
        "mood": "high_energy",
    },
    "mystery_character": {
        "description": "Mysterious character reveal - slow orbital",
        "movement": "orbit_360",
        "orbit_steps": 8,
        "distance": "medium",
        "angle": "eye_level",
        "emotion": "mysterious",
        "intensity": "moderate",
        "eyes": "averted_gaze",
        "pose": "looking_over_shoulder",
        "lighting": "mysterious_shadow",
        "mood": "enigmatic_dark",
    },
    "dramatic_confrontation": {
        "description": "Tense confrontation - slow push in eye level",
        "movement": "dolly_in_slow",
        "distance": "medium_to_close",
        "angle": "eye_level",
        "emotion": "angry",
        "intensity": "strong",
        "eyes": "at_camera_intense",
        "pose": "standing_defensive",
        "lighting": "front_dramatic",
        "mood": "tense_dramatic",
    },
    "vulnerable_moment": {
        "description": "Vulnerable emotional moment - high angle intimate",
        "movement": "crane_down_slow",
        "distance": "close",
        "angle": "high_angle",
        "emotion": "sad_vulnerable",
        "intensity": "strong",
        "eyes": "looking_down",
        "pose": "slouched_defeated",
        "lighting": "soft_top",
        "mood": "melancholy_intimate",
    },
    "triumphant_hero": {
        "description": "Triumphant victory - crane up wide reveal",
        "movement": "crane_up_wide",
        "distance": "wide",
        "angle": "low_angle",
        "emotion": "triumphant",
        "intensity": "extreme",
        "eyes": "looking_up",
        "pose": "victorious_arms_raised",
        "lighting": "backlit_dramatic",
        "mood": "bright_victorious",
    },

    # CHARACTER STUDY PRESETS
    "character_turnaround_neutral": {
        "description": "360° character study - neutral expression 8 frames",
        "movement": "orbit_360",
        "orbit_steps": 8,
        "distance": "medium",
        "angle": "eye_level",
        "emotion": "neutral",
        "intensity": "none",
        "eyes": "forward",
        "pose": "standing_neutral",
        "lighting": "even_all_sides",
        "mood": "neutral_study",
        "multi_frame": True,
    },
    "emotion_showcase": {
        "description": "Emotion reference - cycle through key emotions",
        "movement": "static",
        "distance": "close",
        "angle": "eye_level",
        "emotion": "cycle_emotions",  # Special: will generate multiple
        "eyes": "at_camera",
        "pose": "standing_neutral",
        "lighting": "even_front",
        "mood": "neutral",
        "multi_frame": True,
    },
}

# ============================================================================
# CAMERA MOVEMENTS
# ============================================================================

CAMERA_MOVEMENTS = {
    "static": "maintain current camera position",

    # ORBIT (Rotate around character) ⭐⭐⭐⭐⭐
    "orbit_right": "orbit right around {target}",
    "orbit_left": "orbit left around {target}",
    "orbit_360": "orbit around {target}",  # Multi-step

    # DOLLY (Move closer/farther) ⭐⭐⭐⭐⭐
    "dolly_in": "dolly in towards {target}",
    "dolly_out": "dolly out from {target}",
    "dolly_in_slow": "dolly in towards {target} smooth camera movement",
    "dolly_out_slow": "dolly out from {target} smooth camera movement",

    # TRUCK (Move left/right) ⭐⭐⭐⭐
    "truck_left": "move to vantage point {distance} to the left of {target}",
    "truck_right": "move to vantage point {distance} to the right of {target}",
    "truck_left_reveal": "move camera left revealing more of the scene",
    "truck_right_follow": "move camera right following {target}",

    # CRANE (Move up/down) ⭐⭐⭐⭐
    "crane_up": "move camera up raising the viewpoint",
    "crane_down": "move camera down lowering the viewpoint",
    "crane_up_wide": "move camera up pulling back revealing environment",
    "crane_down_reveal": "move camera down from above towards {target}",
    "crane_down_slow": "move camera down slowly towards {target}",

    # TILT (Angle up/down) ⭐⭐⭐⭐
    "tilt_up": "tilt the camera up",
    "tilt_down": "tilt the camera down",
    "tilt_up_reveal": "tilt the camera up revealing from feet to face",
    "tilt_down_emphasize": "tilt the camera down from face to feet",

    # COMPLEX COMBINATIONS ⭐⭐⭐
    "dolly_in_and_tilt_up": "dolly in towards {target} while tilting camera up",
    "dolly_out_and_tilt_down": "dolly out from {target} while tilting camera down",
    "orbit_and_dolly_in": "orbit around {target} while moving closer",
    "orbit_and_crane_up": "orbit around {target} while raising camera up",
    "truck_and_tilt": "move camera sideways while tilting",
}

# ============================================================================
# CAMERA ANGLES
# ============================================================================

CAMERA_ANGLES = {
    "eye_level": "at eye level",
    "high_angle": "from a high angle looking down",
    "low_angle": "from a low angle looking up",
    "birds_eye": "from extreme high angle bird's eye view looking straight down",
    "worms_eye": "from extreme low angle worm's eye view looking straight up",
    "dutch_15": "with camera tilted 15 degrees (subtle dutch angle)",
    "dutch_30": "with camera tilted 30 degrees (dutch angle)",
    "dutch_45": "with camera tilted 45 degrees (extreme dutch angle)",
    "overhead": "from directly overhead",
}

# ============================================================================
# EMOTIONS (20 core emotions)
# ============================================================================

EMOTIONS = {
    # Basic emotions
    "neutral": "neutral expression",
    "happy": "happy with genuine smile",
    "sad": "sad melancholy expression",
    "angry": "angry with furrowed brow",
    "fearful": "fearful with wide eyes",
    "surprised": "surprised with raised eyebrows",
    "disgusted": "disgusted with wrinkled nose",

    # Complex character emotions
    "confident": "confident self-assured expression",
    "vulnerable": "vulnerable open expression",
    "menacing": "menacing threatening expression",
    "determined": "determined focused expression",
    "triumphant": "triumphant victorious expression",
    "contemplative": "contemplative thoughtful expression",
    "suspicious": "suspicious distrustful expression",
    "seductive": "seductive alluring expression",
    "mysterious": "mysterious enigmatic expression",
    "exhausted": "exhausted tired expression",
    "hopeful": "hopeful optimistic expression",
    "defiant": "defiant rebellious expression",
    "loving": "loving affectionate expression",

    # Special combinations
    "surprised_fearful": "surprised transitioning to fearful",
    "sad_vulnerable": "sad and vulnerable",
    "angry_determined": "angry but determined",
    "confident_menacing": "confident with menacing undertone",
}

EMOTION_INTENSITY = {
    "none": "",
    "subtle": "subtle",
    "moderate": "",  # Default, no modifier
    "strong": "strong",
    "extreme": "extreme",
}

# ============================================================================
# EYE DIRECTION & EXPRESSION
# ============================================================================

EYE_DIRECTION = {
    "at_camera": "looking directly at camera",
    "at_camera_soft": "looking at camera with soft gentle eyes",
    "at_camera_intense": "looking at camera with intense penetrating stare",
    "away_from_camera": "looking away from camera",
    "up": "looking upward",
    "down": "looking downward",
    "left": "looking to the left",
    "right": "looking to the right",
    "up_left": "looking up and to the left",
    "up_right": "looking up and to the right",
    "down_left": "looking down and to the left",
    "down_right": "looking down and to the right",
    "looking_down": "eyes cast downward",
    "looking_up": "eyes raised upward",
    "eyes_closed": "with eyes closed",
    "averted_gaze": "with averted gaze avoiding eye contact",
    "sidelong_glance": "with sidelong suspicious glance",
    "looking_over_shoulder": "looking back over shoulder",
    "forward": "looking straight forward",
    "forward_focused": "looking forward with focused determined gaze",
    "thousand_yard_stare": "with distant unfocused thousand-yard stare",
}

EYE_EXPRESSION = {
    "normal": "",
    "wide_eyes": "with wide alert eyes",
    "narrowed_eyes": "with narrowed suspicious eyes",
    "soft_eyes": "with soft kind eyes",
    "intense_stare": "with intense unwavering stare",
    "half_closed": "with half-closed relaxed eyes",
}

# ============================================================================
# BODY LANGUAGE & POSES
# ============================================================================

POSES = {
    # Standing poses
    "standing_neutral": "standing in neutral natural pose",
    "standing_confident": "standing confidently with chest out shoulders back",
    "standing_defensive": "standing in defensive guarded stance",
    "standing_relaxed": "standing relaxed with weight on one leg",
    "standing_threatening": "standing in threatening aggressive stance",
    "standing_victorious": "standing victoriously",
    "victorious_arms_raised": "standing with arms raised in victory",

    # Action poses
    "action_running": "in running action pose",
    "walking_forward": "walking forward toward camera",
    "walking_away": "walking away from camera",
    "reaching_forward": "reaching forward with extended arm",
    "pointing": "pointing forward directing attention",
    "gesture_welcome": "with welcoming open arms gesture",

    # Sitting poses
    "sitting_relaxed": "sitting in relaxed casual position",
    "sitting_formal": "sitting with formal professional posture",
    "sitting_slouched": "sitting slouched informally",
    "leaning_forward": "leaning forward engaged",
    "leaning_back": "leaning back casually",

    # Special gestures
    "arms_crossed": "with arms crossed defensively",
    "hands_in_pockets": "with hands in pockets casually",
    "hands_on_hips": "with hands on hips assertively",
    "hand_on_chin": "with hand on chin thoughtfully",
    "covering_face": "with hands covering face",
    "looking_over_shoulder": "in pose looking back over shoulder",

    # Posture modifiers
    "slouched_defeated": "with slouched defeated posture",
}

# ============================================================================
# LIGHTING & MOOD
# ============================================================================

LIGHTING_STYLES = {
    "front_lighting": "with front lighting",
    "front_rim": "with front lighting and rim light",
    "front_dramatic": "with dramatic front lighting",
    "front_changing": "with transitioning front lighting",
    "side_lighting": "with side lighting creating contrast",
    "side_shadow": "with side lighting creating dramatic shadows",
    "back_lighting": "with back lighting creating silhouette",
    "backlit_rim": "with dramatic backlighting and rim light",
    "backlit_dramatic": "with dramatic backlight",
    "dramatic_backlight": "with strong dramatic backlight",
    "top_lighting": "with top lighting from above",
    "soft_top": "with soft top lighting",
    "bottom_lighting": "with bottom lighting",
    "rembrandt_lighting": "with Rembrandt lighting from 45 degrees",
    "split_lighting": "with split lighting half face lit",
    "butterfly_lighting": "with butterfly beauty lighting",
    "butterfly_soft": "with soft butterfly lighting",
    "harsh_dramatic": "with harsh dramatic high-contrast lighting",
    "mysterious_shadow": "with mysterious partial shadows",
    "even_all_sides": "with even lighting from all sides",
    "even_front": "with even front lighting",
}

MOOD_ATMOSPHERE = {
    "bright_cheerful": "in bright cheerful high-key atmosphere",
    "bright_heroic": "in bright heroic atmosphere",
    "bright_epic": "in bright epic atmosphere",
    "bright_victorious": "in bright triumphant victorious atmosphere",
    "dark_moody": "in dark moody low-key atmosphere",
    "dark_threatening": "in dark threatening atmosphere",
    "ominous_dark": "in ominous dark dangerous atmosphere",
    "ominous_threatening": "in ominous threatening atmosphere",
    "soft_intimate": "in soft intimate atmosphere",
    "harsh_dramatic": "in harsh dramatic intense atmosphere",
    "neutral_balanced": "in neutral evenly-balanced atmosphere",
    "neutral": "in neutral atmosphere",
    "neutral_study": "in neutral study lighting for reference",
    "warm_inviting": "in warm inviting golden atmosphere",
    "cool_distant": "in cool distant clinical atmosphere",
    "mysterious_shadowy": "in mysterious shadowy atmosphere",
    "enigmatic_dark": "in enigmatic dark mysterious atmosphere",
    "transitioning_bright": "in transitioning brightening atmosphere",
    "dramatic_heroic": "in dramatic heroic atmosphere",
    "high_energy": "in high-energy dynamic atmosphere",
    "tense_dramatic": "in tense dramatic atmosphere",
    "melancholy_intimate": "in melancholy intimate atmosphere",
}

# ============================================================================
# DISTANCE SETTINGS
# ============================================================================

DISTANCE_SETTINGS = {
    "extreme_close": "extreme closeup on face",
    "close": "close shot",
    "medium": "medium shot",
    "medium_to_close": "from medium moving to close",
    "wide": "wide shot",
    "very_wide": "very wide establishing shot",
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def build_cinematographer_prompt(
    preset_name: str,
    preset_data: dict,
    scene_context: str,
    # Custom parameters (used if preset="custom")
    custom_movement: str = "",
    custom_angle: str = "",
    custom_emotion: str = "",
    custom_eyes: str = "",
    custom_pose: str = "",
    custom_lighting: str = "",
    custom_mood: str = "",
    orbit_angle: int = 45,
    orbit_steps: int = 8,
    distance: str = "medium",
) -> str:
    """Build complete cinematographer prompt from preset or custom parameters.

    Formula:
    [SCENE_CONTEXT], [CAMERA_MOVEMENT] [ANGLE] [DISTANCE] showing [EMOTION] [INTENSITY],
    [EYE_DIRECTION], [POSE], [LIGHTING], [MOOD],
    keep the subject's id, clothes, facial features, pose, and hairstyle identical
    """

    # Determine which parameters to use
    if preset_name == "custom":
        movement = custom_movement
        angle = custom_angle
        emotion = custom_emotion
        eyes = custom_eyes
        pose = custom_pose
        lighting = custom_lighting
        mood = custom_mood
    else:
        movement = preset_data.get("movement", "static")
        angle = preset_data.get("angle", "eye_level")
        emotion = preset_data.get("emotion", "neutral")
        intensity_key = preset_data.get("intensity", "moderate")
        eyes = preset_data.get("eyes", "forward")
        pose = preset_data.get("pose", "standing_neutral")
        lighting = preset_data.get("lighting", "front_lighting")
        mood = preset_data.get("mood", "neutral_balanced")
        distance = preset_data.get("distance", "medium")
        orbit_angle = preset_data.get("orbit_angle", 45)
        orbit_steps = preset_data.get("orbit_steps", 8)

    parts = []

    # 1. Scene context
    if scene_context.strip():
        parts.append(scene_context.strip())

    # 2. Distance setting
    distance_text = DISTANCE_SETTINGS.get(distance, "medium shot")

    # 3. Camera angle
    angle_text = CAMERA_ANGLES.get(angle, "at eye level")

    # 4. Build camera movement
    target = "the person"
    movement_text = CAMERA_MOVEMENTS.get(movement, "maintain current position")

    # Handle orbit with angle
    if "orbit" in movement and orbit_angle:
        if movement == "orbit_360":
            movement_text = f"orbit around {target} by three hundred sixty degrees"
        else:
            # Convert number to words for angles
            angle_words = {
                45: "forty-five",
                90: "ninety",
                120: "one hundred twenty",
                180: "one hundred eighty",
            }.get(orbit_angle, str(orbit_angle))
            direction = "right" if "right" in movement else "left"
            movement_text = f"orbit {direction} around {target} by {angle_words} degrees"

    # Handle other movements that need formatting
    movement_text = movement_text.format(target=target, distance="three meters")

    # Combine: distance + angle + movement
    camera_part = f"{distance_text} {angle_text} {movement_text}"
    parts.append(camera_part)

    # 5. Emotion with intensity
    if emotion and emotion != "neutral":
        emotion_text = EMOTIONS.get(emotion, emotion)
        if preset_name != "custom":
            intensity = EMOTION_INTENSITY.get(intensity_key, "")
            if intensity:
                emotion_text = f"{intensity} {emotion_text}"
        parts.append(f"showing {emotion_text}")

    # 6. Eye direction
    if eyes:
        eye_text = EYE_DIRECTION.get(eyes, eyes)
        parts.append(eye_text)

    # 7. Pose and body language
    if pose:
        pose_text = POSES.get(pose, pose)
        parts.append(pose_text)

    # 8. Lighting
    if lighting:
        lighting_text = LIGHTING_STYLES.get(lighting, lighting)
        parts.append(lighting_text)

    # 9. Mood atmosphere
    if mood:
        mood_text = MOOD_ATMOSPHERE.get(mood, mood)
        parts.append(mood_text)

    # 10. Identity preservation (CRITICAL!)
    parts.append("keep the subject's id, clothes, facial features, pose, and hairstyle identical")

    return ", ".join(parts)


def generate_multi_frame_sequence(
    preset_data: dict,
    scene_context: str,
    steps: int,
) -> str:
    """Generate multi-frame prompts for sequences like 360° turnarounds."""

    if preset_data.get("emotion") == "cycle_emotions":
        # Emotion showcase - cycle through key emotions
        emotions = ["neutral", "happy", "sad", "angry", "fearful", "surprised", "confident", "menacing"]
        frames = []
        for i, emotion in enumerate(emotions):
            temp_preset = preset_data.copy()
            temp_preset["emotion"] = emotion
            prompt = build_cinematographer_prompt("preset", temp_preset, scene_context)
            frames.append(f"Frame {i+1}/{len(emotions)} [{emotion.upper()}]: {prompt}")
        return "\n".join(frames)

    elif preset_data.get("movement") == "orbit_360":
        # 360° turnaround
        total_angle = 360
        step_angle = total_angle // steps
        frames = []
        for i in range(steps):
            temp_preset = preset_data.copy()
            temp_preset["orbit_angle"] = step_angle
            prompt = build_cinematographer_prompt("preset", temp_preset, scene_context)
            frames.append(f"Frame {i+1}/{steps}: {prompt}")
        return "\n".join(frames)

    return ""


# ============================================================================
# NODE DEFINITION
# ============================================================================

class ArchAi3D_Qwen_Person_Cinematographer(io.ComfyNode):
    """Person Cinematographer: Ultimate all-in-one character cinematography control.

    Think like a CAMERAMAN filming an ACTOR - complete control in one node!

    Controls:
    - Camera Movement: Orbit, dolly, truck, crane, tilt, complex combinations
    - Camera Angles: Eye-level, high, low, dutch, birds-eye, worms-eye
    - Character Emotion: 20 emotions × 4 intensity levels
    - Eye Direction: 17 gaze options + eye expressions
    - Body Language: 25+ poses and postures
    - Lighting: 8 lighting styles
    - Mood: 9 atmosphere options

    Perfect For:
    - AI movie production (ASCENDRA showreel!)
    - Complex camera choreography
    - Character-driven scenes
    - Professional cinematography
    - Frame-by-frame consistency

    Key Features:
    - 13 ready-to-use cinematic presets
    - ASCENDRA-specific presets included!
    - Multi-frame sequence support (360° turnarounds, emotion cycles)
    - Identity preservation built-in
    - Professional cinematographer system prompt

    USE THIS NODE WHEN:
    - You need complete control over character + camera
    - Working on AI movies or character scenes
    - Want professional cinematography
    - Need frame-by-frame character consistency

    SCENE TYPE: Person/Character/Actor
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ArchAi3D_Qwen_Person_Cinematographer",
            category="ArchAi3d/Camera/Person",
            inputs=[
                # Group 1: Quick Presets
                io.Combo.Input(
                    "preset",
                    options=list(CINEMATIC_PRESETS.keys()),
                    default="ascendra_man_threatening",
                    tooltip="Select cinematic preset or 'custom' for manual control. ASCENDRA presets included!"
                ),

                # Group 2: Scene Context
                io.String.Input(
                    "scene_context",
                    multiline=True,
                    default="",
                    tooltip="Describe the character and scene. Example: 'hooded man in dark jacket on train platform at night'"
                ),

                # Group 3: Camera Movement (Custom Mode)
                io.Combo.Input(
                    "movement",
                    options=list(CAMERA_MOVEMENTS.keys()),
                    default="orbit_right",
                    tooltip="Camera movement type. Orbit = rotate around character (most important!)"
                ),
                io.Int.Input(
                    "orbit_angle",
                    default=45,
                    min=1,
                    max=360,
                    tooltip="Angle for orbit movement (45°, 90°, 180°, 360°)"
                ),
                io.Combo.Input(
                    "distance",
                    options=list(DISTANCE_SETTINGS.keys()),
                    default="medium",
                    tooltip="Camera distance from subject"
                ),

                # Group 4: Camera Angle (Custom Mode)
                io.Combo.Input(
                    "angle",
                    options=list(CAMERA_ANGLES.keys()),
                    default="eye_level",
                    tooltip="Camera angle. Low=powerful, high=vulnerable"
                ),

                # Group 5: Character Emotion (Custom Mode)
                io.Combo.Input(
                    "emotion",
                    options=list(EMOTIONS.keys()),
                    default="neutral",
                    tooltip="Character emotion/expression (20 options)"
                ),
                io.Combo.Input(
                    "emotion_intensity",
                    options=list(EMOTION_INTENSITY.keys()),
                    default="moderate",
                    tooltip="Emotion intensity level"
                ),

                # Group 6: Eye Direction (Custom Mode)
                io.Combo.Input(
                    "eyes",
                    options=list(EYE_DIRECTION.keys()),
                    default="at_camera",
                    tooltip="Where character is looking (17 options)"
                ),
                io.Combo.Input(
                    "eye_expression",
                    options=list(EYE_EXPRESSION.keys()),
                    default="normal",
                    tooltip="Eye expression modifier"
                ),

                # Group 7: Body Language (Custom Mode)
                io.Combo.Input(
                    "pose",
                    options=list(POSES.keys()),
                    default="standing_neutral",
                    tooltip="Character pose and body language (25+ options)"
                ),

                # Group 8: Lighting & Mood (Custom Mode)
                io.Combo.Input(
                    "lighting",
                    options=list(LIGHTING_STYLES.keys()),
                    default="front_lighting",
                    tooltip="Lighting style (8 professional setups)"
                ),
                io.Combo.Input(
                    "mood",
                    options=list(MOOD_ATMOSPHERE.keys()),
                    default="neutral_balanced",
                    tooltip="Scene mood and atmosphere (9 options)"
                ),

                # Group 9: Multi-Frame Sequences
                io.Boolean.Input(
                    "multi_frame_mode",
                    default=False,
                    tooltip="Generate multi-frame sequence (360° turnaround, emotion cycle, etc.)"
                ),
                io.Int.Input(
                    "steps",
                    default=8,
                    min=2,
                    max=24,
                    tooltip="Number of frames for sequence (8=standard, 12=detailed, 24=ultra-smooth)"
                ),

                # Group 10: Options
                io.Boolean.Input(
                    "debug_mode",
                    default=False,
                    tooltip="Print detailed prompt information to console"
                ),
            ],
            outputs=[
                io.String.Output(
                    "prompt",
                    tooltip="Generated cinematographer prompt"
                ),
                io.String.Output(
                    "system_prompt",
                    tooltip="⭐ Professional cinematographer system prompt"
                ),
                io.String.Output(
                    "multi_frame_prompts",
                    tooltip="Multi-frame sequence prompts (if multi_frame_mode=True)"
                ),
                io.Int.Output(
                    "frame_count",
                    tooltip="Number of frames (1 for single, or steps for sequence)"
                ),
                io.String.Output(
                    "preset_description",
                    tooltip="Description of selected preset"
                ),
                io.String.Output(
                    "technical_notes",
                    tooltip="Technical notes and tips for this shot"
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        preset,
        scene_context,
        movement,
        orbit_angle,
        distance,
        angle,
        emotion,
        emotion_intensity,
        eyes,
        eye_expression,
        pose,
        lighting,
        mood,
        multi_frame_mode,
        steps,
        debug_mode,
    ) -> io.NodeOutput:
        """Execute the Person Cinematographer node."""

        # Get preset data
        preset_data = CINEMATIC_PRESETS.get(preset, {})
        preset_desc = preset_data.get("description", "Custom cinematography")

        # Build prompt
        if preset == "custom":
            prompt = build_cinematographer_prompt(
                preset_name="custom",
                preset_data={},
                scene_context=scene_context,
                custom_movement=movement,
                custom_angle=angle,
                custom_emotion=emotion,
                custom_eyes=eyes,
                custom_pose=pose,
                custom_lighting=lighting,
                custom_mood=mood,
                orbit_angle=orbit_angle,
                distance=distance,
            )
        else:
            prompt = build_cinematographer_prompt(
                preset_name=preset,
                preset_data=preset_data,
                scene_context=scene_context,
                orbit_angle=orbit_angle,
            )

        # Multi-frame sequence
        multi_frame_prompts = ""
        final_frame_count = 1

        if multi_frame_mode or preset_data.get("multi_frame"):
            multi_frame_prompts = generate_multi_frame_sequence(
                preset_data=preset_data,
                scene_context=scene_context,
                steps=steps,
            )
            final_frame_count = steps if multi_frame_prompts else 1

        # System prompt
        system_prompt = CINEMATOGRAPHER_SYSTEM_PROMPT

        # Technical notes
        technical_notes = f"""📹 PRESET: {preset}
🎬 DESCRIPTION: {preset_desc}
💡 TIP: Camera movement around character preserves identity while showing different angles
⚡ PERFECT FOR: Character-driven scenes, AI movies, professional cinematography"""

        # Debug output
        if debug_mode:
            debug_lines = [
                "=" * 80,
                "ArchAi3D_Qwen_Person_Cinematographer - Generated Prompt (v1.0)",
                "=" * 80,
                f"Preset: {preset}",
                f"Description: {preset_desc}",
                f"Scene Context: {scene_context[:100]}..." if len(scene_context) > 100 else f"Scene Context: {scene_context}",
                "=" * 80,
                "Generated Prompt:",
                prompt,
                "=" * 80,
            ]

            if multi_frame_prompts:
                debug_lines.extend([
                    f"Multi-Frame Sequence ({final_frame_count} frames):",
                    multi_frame_prompts,
                    "=" * 80,
                ])

            debug_lines.extend([
                "⭐ System Prompt:",
                system_prompt,
                "=" * 80,
                "",
                "💡 CINEMATOGRAPHER TIPS:",
                "  - Camera MOVES around actor (actor stays in place)",
                "  - Orbit = rotate around character (45°, 90°, 180°, 360°)",
                "  - Dolly = move closer/farther",
                "  - Truck = slide left/right",
                "  - Crane = move up/down",
                "  - Low angle = powerful, High angle = vulnerable",
                "  - Identity preservation is automatic!",
                "",
                "🎬 PERFECT FOR AI MOVIES:",
                "  - Use presets for consistent style",
                "  - Multi-frame mode for 360° turnarounds",
                "  - Character stays consistent across all angles",
                "=" * 80,
            ])
            print("\n".join(debug_lines))

        return io.NodeOutput(
            prompt,
            system_prompt,
            multi_frame_prompts,
            final_frame_count,
            preset_desc,
            technical_notes,
        )


# ============================================================================
# EXTENSION REGISTRATION
# ============================================================================

class PersonCinematographerExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [ArchAi3D_Qwen_Person_Cinematographer]


async def comfy_entrypoint():
    return PersonCinematographerExtension()
