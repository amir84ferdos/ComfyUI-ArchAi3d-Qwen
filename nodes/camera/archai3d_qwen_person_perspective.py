# ArchAi3D_Qwen_Person_Perspective — Person/character perspective control for Qwen Edit 2509
#
# FEATURES:
# - 6 perspective presets optimized for people/characters
# - Identity preservation controls (strict/moderate/loose)
# - Psychological effect hints (vulnerability, power, intimacy, etc.)
# - Focal point selection (which body part to emphasize)
# - Body proportion guidance (exaggerate height, foreshorten, etc.)
# - Background adaptation and lighting reinforcement
# - Based on community-tested prompts from Reddit r/StableDiffusion
#
# USE CASES:
# - Portrait photography with different angles
# - Fashion photography with dramatic perspectives
# - Character art with consistent identity
# - Heroic/power poses with low angles
# - Vulnerable/intimate poses with high angles
#
# IMPORTANT NOTES:
# - Works best with people CENTERED in frame
# - Identity preservation is PRIMARY focus
# - Different from object rotation (this changes PERSPECTIVE not rotation)
# - Use with "Portrait Photographer" or "Cinematographer" system prompt
#
# Based on tutorial by Vortexneonlight (Reddit r/StableDiffusion)
#
# Author: Amir Ferdos (ArchAi3d)
# Email: Amir84ferdos@gmail.com
# LinkedIn: https://www.linkedin.com/in/archai3d/
# GitHub: https://github.com/amir84ferdos
# Category: ArchAi3d/Qwen
# Node ID: ArchAi3D_Qwen_Person_Perspective
# License: MIT

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override


# ============================================================================
# PERSPECTIVE SYSTEM PROMPTS - v4.0.0 Feature
# ============================================================================

PERSPECTIVE_SYSTEM_PROMPTS = {
    "custom": "You are a portrait photographer. Adjust camera perspective to change viewing angle of person. Preserve subject's identity, clothing, and pose exactly. Maintain natural lighting and composition.",

    "high_angle_birds_eye": "You are a portrait photographer. Execute high angle bird's eye view perspective looking down at subject. Create sense of vulnerability or isolation through elevated viewpoint. Preserve subject's identity, facial features, clothing, and pose exactly. Emphasize head and chest area while foreshortening body. Maintain consistent lighting and background context.",

    "low_angle_worms_eye": "You are a portrait photographer. Execute ultra-low angle worm's eye view perspective looking up at subject. Create sense of monumentality, power, and heroic presence through ground-level viewpoint. Preserve subject's identity, facial features, clothing, and pose exactly. Emphasize legs and thighs while upper body rises dramatically. Maintain consistent lighting throughout.",

    "eye_level_front": "You are a portrait photographer. Maintain standard eye-level front view perspective positioned directly in front of subject. Create balanced and neutral perspective. Preserve subject's identity, facial features, clothing, and pose exactly. Natural body proportions, professional framing. Maintain consistent lighting and composition.",

    "side_profile": "You are a portrait photographer. Execute direct side profile perspective positioned perpendicular to subject at eye level. Showcase entire side profile from head to toe clearly. Preserve subject's identity, clothing, hairstyle, and pose exactly. Natural proportions with clean side view. Maintain consistent lighting and professional framing.",

    "three_quarter_view": "You are a portrait photographer. Execute three-quarter angle perspective positioned approximately 45 degrees from front at eye level. Create depth while showing facial features clearly. Preserve subject's identity, facial features, clothing, and pose exactly. Natural proportions with dimensional depth. Maintain consistent lighting and professional composition.",

    "dutch_angle": "You are a portrait photographer. Execute dutch angle perspective with camera tilted 30-45 degrees at eye level. Create tension, drama, and visual interest through tilted horizon. Preserve subject's identity, facial features, clothing, and pose exactly. Diagonal composition with dynamic framing. Maintain consistent lighting throughout tilted perspective.",
}

# Perspective preset configurations
PERSPECTIVE_PRESETS = {
    "custom": {
        "description": "Manual configuration",
        "angle": "",
        "camera_position": "",
        "effect": "",
        "focal_area": "",
        "proportion_hint": ""
    },
    "high_angle_birds_eye": {
        "description": "Bird's eye view - looking down (vulnerability effect)",
        "angle": "high angle shot (bird's eye view)",
        "camera_position": "positioned above and looking directly down",
        "effect": "diminish the subject's height and create a sense of vulnerability or isolation",
        "focal_area": "head, chest, and surrounding environment",
        "proportion_hint": "while the rest of the body is foreshortened but visible"
    },
    "low_angle_worms_eye": {
        "description": "Worm's eye view - looking up (power/heroic effect)",
        "angle": "ultra-low angle shot",
        "camera_position": "positioned very close to the legs looking up",
        "effect": "exaggerate the subject's height and create a sense of monumentality and power",
        "focal_area": "legs and thighs",
        "proportion_hint": "while the upper body dramatically rises towards up"
    },
    "eye_level_front": {
        "description": "Eye level front view - standard portrait",
        "angle": "eye level shot",
        "camera_position": "positioned directly in front at eye level",
        "effect": "create a balanced and neutral perspective",
        "focal_area": "face and upper body",
        "proportion_hint": "maintaining natural body proportions"
    },
    "side_profile": {
        "description": "Side profile - full side view at eye level",
        "angle": "direct side angle shot",
        "camera_position": "at eye level perpendicular to the subject",
        "effect": "clearly showcase the entire side profile",
        "focal_area": "side profile from head to toe",
        "proportion_hint": "maintaining natural proportions"
    },
    "three_quarter_view": {
        "description": "Three-quarter view - slight angle for depth",
        "angle": "three-quarter angle shot",
        "camera_position": "at eye level at approximately 45 degrees from front",
        "effect": "create depth while showing facial features",
        "focal_area": "face, shoulder, and partial side",
        "proportion_hint": "natural proportions with dimensional depth"
    },
    "dutch_angle": {
        "description": "Dutch angle - tilted for dramatic/unsettling effect",
        "angle": "tilted angle shot (dutch angle)",
        "camera_position": "at eye level but camera tilted approximately 30-45 degrees",
        "effect": "create tension, drama, or visual interest",
        "focal_area": "face and upper body in tilted frame",
        "proportion_hint": "diagonal composition"
    }
}

# Identity preservation levels
IDENTITY_PRESERVATION = {
    "none": "",
    "loose": "Keep the subject recognizable",
    "moderate": "Maintain the subject's appearance and clothing",
    "strict": "Keep the subject's id, clothes, facial features, pose, and hairstyle identical",
}

# Psychological effects
PSYCHOLOGICAL_EFFECTS = {
    "none": "",
    "vulnerability": "vulnerability or isolation",
    "power": "monumentality and power",
    "intimacy": "intimacy and connection",
    "grandeur": "grandeur and imposing presence",
    "confidence": "confidence and self-assurance",
    "mystery": "mystery and intrigue",
}

# Focal points
FOCAL_POINTS = {
    "auto": "",  # Let preset decide
    "head_chest": "head and chest",
    "legs_thighs": "legs and thighs",
    "full_body": "entire body proportionally",
    "face_upper": "face and upper body",
    "torso": "torso and midsection",
    "lower_body": "lower body and legs",
}

# Body proportion guidance
BODY_PROPORTIONS = {
    "natural": "maintaining natural body proportions",
    "exaggerate_height": "exaggerating the subject's height",
    "diminish_height": "diminishing the subject's height",
    "foreshorten_upper": "with upper body foreshortened",
    "foreshorten_lower": "with lower body foreshortened",
    "elongate": "elongating the figure",
}


class ArchAi3D_Qwen_Person_Perspective(io.ComfyNode):
    """Professional person/character perspective control for Qwen Edit 2509.

    Specialized node for changing camera perspectives when photographing people/characters.
    Focus on identity preservation while changing angles for dramatic effect.

    Key Features:
    - 6 perspective presets (high/low/side/eye-level/three-quarter/dutch)
    - Identity preservation (keep face, clothes, pose identical)
    - Psychological effects (vulnerability, power, intimacy, etc.)
    - Body proportion guidance
    - Based on community-tested prompts

    Perfect for: Portrait photography, fashion shoots, character art, dramatic angles
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ArchAi3D_Qwen_Person_Perspective",
            category="ArchAi3d/Qwen/Camera",
            inputs=[
                # === PERSPECTIVE PRESET ===
                io.Combo.Input(
                    "perspective_preset",
                    options=list(PERSPECTIVE_PRESETS.keys()),
                    default="eye_level_front",
                    tooltip="Camera perspective preset optimized for people (high angle = vulnerability, low angle = power)"
                ),

                # === IDENTITY PRESERVATION (CRITICAL!) ===
                io.Combo.Input(
                    "identity_preservation",
                    options=list(IDENTITY_PRESERVATION.keys()),
                    default="strict",
                    tooltip="How strictly to preserve person's identity (strict = keep face/clothes/pose identical)"
                ),

                # === PSYCHOLOGICAL EFFECT ===
                io.Combo.Input(
                    "psychological_effect",
                    options=list(PSYCHOLOGICAL_EFFECTS.keys()),
                    default="none",
                    tooltip="Psychological effect to convey (vulnerability, power, intimacy, etc.)"
                ),

                # === FOCAL POINT ===
                io.Combo.Input(
                    "focal_point",
                    options=list(FOCAL_POINTS.keys()),
                    default="auto",
                    tooltip="Which body part to emphasize (auto = preset decides)"
                ),

                # === BODY PROPORTION ===
                io.Combo.Input(
                    "body_proportion",
                    options=list(BODY_PROPORTIONS.keys()),
                    default="natural",
                    tooltip="Body proportion guidance (exaggerate height for low angles, diminish for high angles)"
                ),

                # === SCENE CONTEXT ===
                io.String.Input(
                    "scene_context",
                    multiline=True,
                    default="",
                    tooltip="Optional: Scene description (e.g., 'in modern office', 'outdoor garden')"
                ),

                # === BACKGROUND ADAPTATION ===
                io.Boolean.Input(
                    "background_adaptation",
                    default=True,
                    tooltip="Ensure background elements also change to complement the new perspective"
                ),

                # === LIGHTING REINFORCEMENT ===
                io.Boolean.Input(
                    "lighting_reinforcement",
                    default=True,
                    tooltip="Ensure lighting reinforces the perspective effect"
                ),

                # === COMPOSITION ===
                io.Boolean.Input(
                    "composition_centering",
                    default=True,
                    tooltip="Keep subject centered in composition (recommended for best results)"
                ),

                # === DETAIL SHOWCASE ===
                io.Combo.Input(
                    "detail_showcase",
                    options=["high", "medium", "low"],
                    default="high",
                    tooltip="Level of detail to showcase in focal areas"
                ),

                # === PROMPT STYLE ===
                io.Combo.Input(
                    "prompt_style",
                    options=[
                        "detailed",          # Full detailed prompt (from PDF)
                        "concise",           # Shorter version
                        "balanced",          # Medium detail
                    ],
                    default="balanced",
                    tooltip="Prompt verbosity (detailed = most control, concise = faster)"
                ),

                # === CUSTOM ADDITIONS ===
                io.String.Input(
                    "custom_additions",
                    multiline=True,
                    default="",
                    tooltip="Optional: Additional custom instructions to append"
                ),

                # === DEBUG ===
                io.Boolean.Input(
                    "debug_mode",
                    default=False,
                    tooltip="Print generated prompt to console"
                ),
            ],
            outputs=[
                io.String.Output("perspective_prompt", tooltip="Generated perspective control prompt"),
                io.String.Output("full_prompt", tooltip="Prompt with scene context"),
                io.String.Output("preservation_hints", tooltip="Identity preservation instructions"),
                io.String.Output("system_prompt", tooltip="⭐ NEW v4.0: Perfect system prompt for this perspective! Connect to encoder's system_prompt input for optimal results."),
            ],
        )

    @classmethod
    def execute(cls,
                perspective_preset,
                identity_preservation,
                psychological_effect,
                focal_point,
                body_proportion,
                scene_context,
                background_adaptation,
                lighting_reinforcement,
                composition_centering,
                detail_showcase,
                prompt_style,
                custom_additions,
                debug_mode) -> io.NodeOutput:

        # Get preset configuration
        preset = PERSPECTIVE_PRESETS[perspective_preset]

        # Get system prompt for this perspective (v4.0.0 feature)
        system_prompt = PERSPECTIVE_SYSTEM_PROMPTS.get(perspective_preset, PERSPECTIVE_SYSTEM_PROMPTS["custom"])

        # Build perspective prompt
        prompt_parts = []

        # === STEP 1: Main perspective instruction ===
        if perspective_preset != "custom":
            angle = preset["angle"]
            camera_pos = preset["camera_position"]

            prompt_parts.append(f"Rotate the angle of the photo to a {angle} of the subject")

            if prompt_style == "detailed":
                prompt_parts.append(f"with the camera {camera_pos}")

        # === STEP 2: Psychological effect ===
        if psychological_effect != "none":
            effect_text = PSYCHOLOGICAL_EFFECTS[psychological_effect]
            if preset.get("effect") and prompt_style in ["detailed", "balanced"]:
                # Use preset effect
                prompt_parts.append(f"The perspective should {preset['effect']}")
            elif effect_text:
                prompt_parts.append(f"creating a sense of {effect_text}")

        # === STEP 3: Focal point ===
        focal_text = ""
        if focal_point == "auto" and preset.get("focal_area"):
            focal_text = preset["focal_area"]
        elif focal_point != "auto":
            focal_text = FOCAL_POINTS[focal_point]

        if focal_text and prompt_style in ["detailed", "balanced"]:
            showcase_level = "prominently" if detail_showcase == "high" else "clearly" if detail_showcase == "medium" else ""
            if showcase_level:
                prompt_parts.append(f"{showcase_level} showcasing the details of the {focal_text}")
            else:
                prompt_parts.append(f"showcasing the {focal_text}")

        # === STEP 4: Body proportion guidance ===
        proportion_text = ""
        if body_proportion != "natural":
            proportion_text = BODY_PROPORTIONS[body_proportion]
        elif preset.get("proportion_hint") and prompt_style == "detailed":
            proportion_text = preset["proportion_hint"]

        if proportion_text:
            prompt_parts.append(proportion_text)

        # === STEP 5: Identity preservation (CRITICAL!) ===
        preservation_text = IDENTITY_PRESERVATION[identity_preservation]

        # === STEP 6: Background adaptation ===
        if background_adaptation and prompt_style in ["detailed", "balanced"]:
            prompt_parts.append("Ensure that other elements in the background also change to complement the subject's new perspective")

        # === STEP 7: Lighting reinforcement ===
        if lighting_reinforcement and prompt_style in ["detailed", "balanced"]:
            prompt_parts.append("The lighting and overall composition should reinforce this perspective effect")

        # === STEP 8: Composition ===
        if composition_centering and prompt_style == "detailed":
            prompt_parts.append("Maintain the subject centered in the frame")

        # === STEP 9: Custom additions ===
        if custom_additions.strip():
            prompt_parts.append(custom_additions.strip())

        # Build main prompt
        perspective_prompt = ". ".join(prompt_parts)
        if perspective_prompt and not perspective_prompt.endswith("."):
            perspective_prompt += "."

        # Add identity preservation as separate critical instruction
        if preservation_text:
            perspective_prompt += f" Important: {preservation_text}."

        # Build full prompt with scene context
        full_prompt_parts = []
        if scene_context.strip():
            full_prompt_parts.append(scene_context.strip())
        full_prompt_parts.append(perspective_prompt)
        full_prompt = " ".join(full_prompt_parts)

        # Preservation hints for output
        preservation_hints = preservation_text if preservation_text else "No identity preservation applied"

        # Debug output
        if debug_mode:
            print("=" * 70)
            print("ArchAi3D_Qwen_Person_Perspective - Generated Prompt (v4.0.0)")
            print("=" * 70)
            print(f"Perspective: {perspective_preset}")
            print(f"Description: {preset['description']}")
            print(f"Identity Preservation: {identity_preservation}")
            print(f"Psychological Effect: {psychological_effect}")
            print(f"Focal Point: {focal_point}")
            print(f"Body Proportion: {body_proportion}")
            print(f"Prompt Style: {prompt_style}")
            print(f"\n💡 TIP: Works best with person CENTERED in frame!")
            if perspective_preset == "low_angle_worms_eye":
                print(f"⚡ EFFECT: Power, heroic, monumentality")
            elif perspective_preset == "high_angle_birds_eye":
                print(f"⚡ EFFECT: Vulnerability, isolation, intimacy")
            print(f"\nPerspective Prompt:")
            print(f"  {perspective_prompt}")
            if scene_context:
                print(f"\nScene Context:")
                print(f"  {scene_context}")
            print(f"\nPreservation Hints:")
            print(f"  {preservation_hints}")
            print(f"\n⭐ System Prompt (NEW v4.0.0):")
            print(f"  {system_prompt}")
            print(f"\nFull Prompt:")
            print(f"  {full_prompt}")
            print("=" * 70)

        return io.NodeOutput(perspective_prompt, full_prompt, preservation_hints, system_prompt)


# ---- Extension API glue ----
class ArchAi3DQwenPersonPerspectiveExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [ArchAi3D_Qwen_Person_Perspective]

async def comfy_entrypoint():
    return ArchAi3DQwenPersonPerspectiveExtension()
