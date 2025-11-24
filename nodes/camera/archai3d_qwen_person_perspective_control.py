# -*- coding: utf-8 -*-
"""
ArchAi3D Qwen Person Perspective Control Node (v1)
Author: Amir Ferdos (ArchAi3d)
Email: Amir84ferdos@gmail.com
LinkedIn: https://www.linkedin.com/in/archai3d/

Description:
    Advanced perspective control for portrait photography with psychological effects.
    Simplified refactor of Person_Perspective with v5.0.0 organization.

    Perfect for:
    - Dramatic portrait photography (psychological effects)
    - Fashion photography (creative perspectives)
    - Character art (heroic, vulnerable poses)
    - Social media content (engaging angles)

Based on research:
    - QWEN_PROMPT_GUIDE Function 7: Person Perspective Change
    - Community-tested prompts (Reddit r/StableDiffusion by Vortexneonlight)
    - Psychological effects (vulnerability, power, intimacy)
    - Identity preservation is CRITICAL
"""

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

# ============================================================================
# PERSON PERSPECTIVE SYSTEM PROMPT - v5.0.0 Research-Validated
# ============================================================================

PERSON_PERSPECTIVE_SYSTEM_PROMPT = """You are a portrait photographer. Execute perspective change to create psychological effect. Adjust camera angle and viewpoint to create specific emotional impact. Preserve subject's identity, facial features, clothing, hairstyle, and pose exactly. Maintain professional portrait composition with dramatic perspective. Person should remain recognizable and unchanged."""

# ============================================================================
# PERSPECTIVE PRESETS - 10 Dramatic Portrait Perspectives
# ============================================================================

PERSON_PERSPECTIVE_PRESETS = {
    "custom": {
        "description": "Manual control of perspective",
        "angle_description": "",
        "effect": "Custom effect",
    },

    # PSYCHOLOGICAL EFFECTS (6) - Based on angle psychology
    "vulnerability_high": {
        "description": "High angle for vulnerability - looking down creates weakness, innocence",
        "angle_description": "a high-angle shot of the subject, with the camera positioned above the subject and angled downward",
        "effect": "vulnerability and innocence",
    },
    "power_low": {
        "description": "Low angle for power - looking up creates strength, dominance",
        "angle_description": "a low-angle shot of the subject, with the camera positioned below the subject and angled upward",
        "effect": "power and dominance",
    },
    "heroic_worms_eye": {
        "description": "Worm's eye for heroic - extreme low angle, monumental presence",
        "angle_description": "an ultra-low angle shot (worm's eye view) of the subject, with the camera's point of view positioned very low to the ground, directly looking up at the subject",
        "effect": "heroic monumentality",
    },
    "intimacy_birds_eye": {
        "description": "Bird's eye for intimacy - extreme high angle, intimate vulnerability",
        "angle_description": "an ultra-high angle shot (bird's eye view) of the subject, with the camera's point of view positioned very high up, directly looking down at the subject from above",
        "effect": "intimate vulnerability",
    },
    "confidence_eye_level": {
        "description": "Eye level for confidence - neutral perspective, equal relationship",
        "angle_description": "an eye-level shot of the subject, with the camera positioned at the same height as the subject's eyes",
        "effect": "confidence and equality",
    },
    "mystery_three_quarter": {
        "description": "Three-quarter for mystery - 45° angle creates intrigue, depth",
        "angle_description": "a three-quarter angle shot of the subject, with the camera positioned at approximately 45 degrees from front at eye level",
        "effect": "mystery and depth",
    },

    # CREATIVE PERSPECTIVES (4)
    "dramatic_dutch": {
        "description": "Dutch angle for drama - tilted horizon creates tension, unease",
        "angle_description": "a dutch angle shot of the subject, with the camera tilted approximately 30-45 degrees at eye level",
        "effect": "dramatic tension",
    },
    "fashion_profile": {
        "description": "Side profile for fashion - clean silhouette, elegant",
        "angle_description": "a direct side profile shot of the subject, with the camera positioned perpendicular to the subject at eye level",
        "effect": "elegant fashion aesthetic",
    },
    "editorial_back": {
        "description": "Back view for editorial - mysterious, contemplative",
        "angle_description": "a back view shot of the subject, with the camera positioned behind the subject",
        "effect": "contemplative mystery",
    },
    "overhead_artistic": {
        "description": "Overhead artistic - unique flat lay perspective",
        "angle_description": "an overhead flat lay shot of the subject, with the camera positioned directly above looking straight down",
        "effect": "artistic uniqueness",
    },
}

# ============================================================================
# IDENTITY PRESERVATION LEVELS
# ============================================================================

IDENTITY_PRESERVATION_MAP = {
    "strict": "keep the subject's id, clothes, facial features, pose, and hairstyle identical",
    "moderate": "maintain the subject's appearance and clothing",
    "loose": "keep the subject recognizable",
    "none": "",
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def build_person_perspective_prompt(
    angle_description: str,
    scene_context: str,
    identity_level: str,
) -> str:
    """Build person perspective control prompt.

    Formula from QWEN Guide Function 7 (Lines 339-341):
    Rotate the angle of the photo to [ANGLE_DESCRIPTION], [IDENTITY_PRESERVATION]

    Args:
        angle_description: The angle description from preset
        scene_context: Optional scene/person description
        identity_level: Identity preservation level (strict/moderate/loose/none)

    Returns:
        Complete prompt string
    """
    parts = []

    # Scene context (optional but recommended)
    if scene_context.strip():
        parts.append(scene_context.strip())

    # Main angle change instruction - QWEN Guide Function 7
    if angle_description:
        angle_instruction = f"Rotate the angle of the photo to {angle_description}"
        parts.append(angle_instruction)

    # Identity preservation clause (based on level)
    preservation_clause = IDENTITY_PRESERVATION_MAP.get(identity_level, "")
    if preservation_clause:
        parts.append(preservation_clause)

    return ", ".join(parts)


# ============================================================================
# NODE DEFINITION
# ============================================================================

class ArchAi3D_Qwen_Person_Perspective_Control(io.ComfyNode):
    """Person Perspective Control: Advanced perspective with psychological effects.

    This node creates dramatic perspectives for portraits:
    - Psychological effects (vulnerability, power, intimacy, confidence)
    - Creative angles (dutch, profile, back, overhead)
    - **Identity preservation is CRITICAL** (face, clothes, pose stay same)
    - Based on angle psychology and community-tested prompts

    Key Features:
    - 10 preset dramatic perspectives
    - Psychological effect descriptions
    - Identity preservation levels (strict/moderate/loose)
    - Research-validated from QWEN Guide Function 7
    - Automatic system prompt output

    Perfect For:
    - Dramatic portrait photography (psychological impact)
    - Fashion photography (creative perspectives)
    - Character art (heroic, vulnerable, mysterious)
    - Social media content (engaging, emotional angles)
    - Editorial photography (artistic perspectives)

    Based on research:
    - Person Perspective Change (QWEN_PROMPT_GUIDE.md Function 7)
    - "Rotate the angle to [ANGLE], keep identity identical"
    - Community prompts (Reddit r/StableDiffusion by Vortexneonlight)
    - Angle psychology (high=vulnerable, low=powerful)

    **CRITICAL:** Always preserve identity for person edits!

    USE THIS NODE WHEN:
    - You have a person/character in the image
    - You want DRAMATIC PERSPECTIVE with psychological effect
    - Identity must stay the same (face, clothes, pose)

    SCENE TYPE: Person/Portrait/Character
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ArchAi3D_Qwen_Person_Perspective_Control",
            category="ArchAi3d/Camera/Person",
            inputs=[
                # Group 1: Perspective Selection
                io.Combo.Input(
                    "preset",
                    options=list(PERSON_PERSPECTIVE_PRESETS.keys()),
                    default="confidence_eye_level",
                    tooltip="Select dramatic perspective. Each creates specific psychological effect."
                ),

                # Group 2: Scene Context
                io.String.Input(
                    "scene_context",
                    multiline=True,
                    default="",
                    tooltip="Optional: Person description. Example: 'man in heroic pose', 'woman in elegant dress'. Improves consistency."
                ),

                # Group 3: Custom Angle (Custom Mode)
                io.String.Input(
                    "custom_angle",
                    multiline=True,
                    default="",
                    tooltip="Custom angle description. Example: 'a slightly elevated angle shot creating sense of isolation'"
                ),

                # Group 4: Identity Preservation
                io.Combo.Input(
                    "identity_level",
                    options=["strict", "moderate", "loose", "none"],
                    default="strict",
                    tooltip="Identity preservation: strict=exact match (recommended), moderate=similar, loose=recognizable, none=no preservation"
                ),

                # Group 5: Options
                io.Boolean.Input(
                    "debug_mode",
                    default=False,
                    tooltip="Print detailed prompt information to console"
                ),
            ],
            outputs=[
                io.String.Output(
                    "prompt",
                    tooltip="Generated prompt for Qwen Edit 2509"
                ),
                io.String.Output(
                    "perspective_description",
                    tooltip="Description of the selected perspective preset"
                ),
                io.String.Output(
                    "psychological_effect",
                    tooltip="Psychological effect of this perspective"
                ),
                io.String.Output(
                    "system_prompt",
                    tooltip="⭐ v5.0: Research-validated system prompt for person perspective control! Connect to encoder's system_prompt input."
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        preset,
        scene_context,
        custom_angle,
        identity_level,
        debug_mode,
    ) -> io.NodeOutput:
        """Execute the Person Perspective Control node.

        Steps:
        1. Get preset data or use custom angle
        2. Build person perspective prompt
        3. Get system prompt
        4. Debug output if requested
        5. Return outputs
        """

        # Step 1: Get preset data or use custom angle
        if preset == "custom":
            angle_description = custom_angle
            preset_desc = "Custom perspective"
            effect = "Custom effect"
        else:
            preset_data = PERSON_PERSPECTIVE_PRESETS.get(preset, PERSON_PERSPECTIVE_PRESETS["confidence_eye_level"])
            angle_description = preset_data["angle_description"]
            preset_desc = preset_data["description"]
            effect = preset_data["effect"]

        # Step 2: Build person perspective prompt
        prompt = build_person_perspective_prompt(
            angle_description=angle_description,
            scene_context=scene_context,
            identity_level=identity_level,
        )

        # Step 3: Get system prompt (v5.0.0 feature - research-validated)
        system_prompt = PERSON_PERSPECTIVE_SYSTEM_PROMPT

        # Step 4: Debug output
        if debug_mode:
            debug_lines = [
                "=" * 70,
                "ArchAi3D_Qwen_Person_Perspective_Control - Generated Prompt (v5.0.0)",
                "=" * 70,
                f"Preset: {preset}",
                f"Description: {preset_desc}",
                f"Psychological Effect: {effect}",
                f"Identity Preservation: {identity_level.upper()} ({IDENTITY_PRESERVATION_MAP.get(identity_level, 'none')})",
                f"Scene Context: {scene_context[:80]}..." if len(scene_context) > 80 else f"Scene Context: {scene_context}",
                "=" * 70,
                "Generated Prompt:",
                prompt,
                "=" * 70,
                "⭐ System Prompt (v5.0.0 - Research-Validated):",
                system_prompt,
                "=" * 70,
                "",
                "💡 TIP: Per QWEN Guide Function 7 - Person Perspective ⭐⭐⭐⭐ Very Reliable!",
                "  - Use this node for DRAMATIC PERSPECTIVE with psychological effect",
                "  - For simple ANGLE CHANGES, use Person_View_Control",
                "  - For MOVING CLOSER/FARTHER, use Person_Position_Control",
                "  - ⚠️ ALWAYS use 'strict' identity preservation for people!",
                "",
                "🎭 PSYCHOLOGICAL EFFECTS GUIDE:",
                "  - HIGH ANGLE = Vulnerability, weakness, innocence, isolation",
                "  - LOW ANGLE = Power, dominance, heroic, confidence",
                "  - EYE LEVEL = Neutral, equal, confident, professional",
                "  - DUTCH ANGLE = Tension, drama, unease, dynamic",
                "  - BIRD'S EYE = Intimacy, vulnerability, isolation",
                "  - WORM'S EYE = Monumentality, heroic, imposing",
                "=" * 70,
            ]
            print("\n".join(debug_lines))

        # Step 5: Return
        return io.NodeOutput(prompt, preset_desc, effect, system_prompt)


# ============================================================================
# EXTENSION REGISTRATION
# ============================================================================

class PersonPerspectiveControlExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [ArchAi3D_Qwen_Person_Perspective_Control]


async def comfy_entrypoint():
    return PersonPerspectiveControlExtension()
