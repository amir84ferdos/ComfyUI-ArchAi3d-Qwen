# -*- coding: utf-8 -*-
"""
ArchAi3D Qwen Person View Control Node (v1)
Author: Amir Ferdos (ArchAi3d)
Email: Amir84ferdos@gmail.com
LinkedIn: https://www.linkedin.com/in/archai3d/

Description:
    Change camera angle for portrait photography while preserving identity.
    Perfect for getting different angles of same person/character.

    Perfect for:
    - Portrait photography (different angles)
    - Fashion photography (multiple perspectives)
    - Social media content (varied angles)
    - Character documentation (comprehensive views)

Based on research:
    - QWEN_PROMPT_GUIDE Function 7: Person Perspective Change (⭐⭐⭐⭐)
    - "Rotate the angle of the photo to [ANGLE], keep the subject's id, clothes, facial features, pose, and hairstyle identical"
    - Identity preservation is CRITICAL
    - Formula from community-tested prompts (Reddit r/StableDiffusion)
"""

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

# ============================================================================
# PERSON VIEW SYSTEM PROMPT - v5.0.0 Research-Validated
# ============================================================================

PERSON_VIEW_SYSTEM_PROMPT = """You are a portrait photographer. Change camera angle to show subject from different perspective. Rotate viewing angle only. Preserve subject's identity, facial features, clothing, hairstyle, and pose exactly. Maintain natural lighting and professional portrait composition. Person should remain recognizable and unchanged."""

# ============================================================================
# VIEW ANGLE PRESETS - 10 Portrait Photography Angles
# ============================================================================

PERSON_VIEW_PRESETS = {
    "custom": {
        "description": "Manual control of viewing angle",
        "angle_description": "",
    },

    # HEIGHT PERSPECTIVES (5) - Psychological effects
    "ultra_high_birds_eye": {
        "description": "Bird's eye view - looking down (vulnerability, intimacy) ⭐ EFFECT: Vulnerable",
        "angle_description": "an ultra-high angle shot (bird's eye view) of the subject, with the camera's point of view positioned very high up, directly looking down at the subject from above",
    },
    "high_angle": {
        "description": "High angle - looking down (weakness, innocence) ⭐ EFFECT: Weak/Innocent",
        "angle_description": "a high-angle shot of the subject, with the camera positioned above the subject and angled downward",
    },
    "eye_level": {
        "description": "Eye level - neutral perspective (equal, natural) ⭐ EFFECT: Neutral",
        "angle_description": "an eye-level shot of the subject, with the camera positioned at the same height as the subject's eyes",
    },
    "low_angle": {
        "description": "Low angle - looking up (power, confidence) ⭐ EFFECT: Powerful",
        "angle_description": "a low-angle shot of the subject, with the camera positioned below the subject and angled upward",
    },
    "ultra_low_worms_eye": {
        "description": "Worm's eye view - extreme low (heroic, monumental) ⭐ EFFECT: Heroic",
        "angle_description": "an ultra-low angle shot (worm's eye view) of the subject, with the camera's point of view positioned very low to the ground, directly looking up at the subject",
    },

    # SIDE VIEWS (3) - Classic portrait angles
    "side_profile": {
        "description": "Direct side profile - 90° perpendicular view",
        "angle_description": "a direct side profile shot of the subject, with the camera positioned perpendicular to the subject at eye level",
    },
    "three_quarter_view": {
        "description": "Three-quarter view - 45° angle (classic portrait) ⭐ Most Popular",
        "angle_description": "a three-quarter angle shot of the subject, with the camera positioned at approximately 45 degrees from front at eye level",
    },
    "back_view": {
        "description": "Back view - rear perspective showing back of subject",
        "angle_description": "a back view shot of the subject, with the camera positioned behind the subject",
    },

    # CREATIVE (2)
    "dutch_angle": {
        "description": "Dutch angle - tilted dramatic effect (tension, drama)",
        "angle_description": "a dutch angle shot of the subject, with the camera tilted approximately 30-45 degrees at eye level",
    },
    "overhead_flat": {
        "description": "Overhead flat lay - directly above (unique perspective)",
        "angle_description": "an overhead flat lay shot of the subject, with the camera positioned directly above looking straight down",
    },
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def build_person_view_prompt(
    angle_description: str,
    scene_context: str,
    preserve_identity: bool,
) -> str:
    """Build person view angle change prompt.

    Formula from QWEN Guide Function 7 (Lines 339-341):
    Rotate the angle of the photo to [ANGLE_DESCRIPTION], keep the subject's id, clothes, facial features, pose, and hairstyle identical

    Args:
        angle_description: The angle description from preset
        scene_context: Optional scene/person description
        preserve_identity: Whether to include identity preservation clause

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

    # Identity preservation clause (CRITICAL for person edits)
    if preserve_identity:
        preservation_clause = "keep the subject's id, clothes, facial features, pose, and hairstyle identical"
        parts.append(preservation_clause)

    return ", ".join(parts)


# ============================================================================
# NODE DEFINITION
# ============================================================================

class ArchAi3D_Qwen_Person_View_Control(io.ComfyNode):
    """Person View Control: Change camera angle for portrait photography.

    This node changes viewing angle for portraits (QWEN Guide Function 7):
    - Height perspectives (bird's eye, high, eye-level, low, worm's eye)
    - Side views (profile, three-quarter, back)
    - Creative angles (dutch, overhead)
    - **Identity preservation is CRITICAL** (face, clothes, pose stay same)

    Key Features:
    - 10 preset portrait angles
    - Research-validated formula from QWEN Guide Function 7
    - Identity preservation clause (REQUIRED for person edits)
    - Psychological effects (vulnerability, power, confidence)
    - Automatic system prompt output

    Perfect For:
    - Portrait photography (multiple angles of same person)
    - Fashion shoots (show outfit from different angles)
    - Social media content (varied perspectives)
    - Character documentation (consistent identity, different angles)

    Based on research:
    - Person Perspective Change (QWEN_PROMPT_GUIDE.md Function 7)
    - "Rotate the angle to [ANGLE], keep identity identical"
    - ⭐⭐⭐⭐ Very Reliable with identity preservation
    - Community-tested prompts (Reddit r/StableDiffusion)

    **CRITICAL:** Always preserve identity for person edits!

    USE THIS NODE WHEN:
    - You have a person/character in the image
    - You want to change VIEWING ANGLE
    - Identity must stay the same (face, clothes, pose)

    SCENE TYPE: Person/Portrait/Character
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ArchAi3D_Qwen_Person_View_Control",
            category="ArchAi3d/Camera/Person",
            inputs=[
                # Group 1: Angle Selection
                io.Combo.Input(
                    "preset",
                    options=list(PERSON_VIEW_PRESETS.keys()),
                    default="eye_level",
                    tooltip="Select portrait angle. Eye-level is neutral, low-angle=powerful, high-angle=vulnerable."
                ),

                # Group 2: Scene Context
                io.String.Input(
                    "scene_context",
                    multiline=True,
                    default="",
                    tooltip="Optional: Person description. Example: 'woman in business suit', 'man wearing casual clothes'. Improves consistency."
                ),

                # Group 3: Custom Angle (Custom Mode)
                io.String.Input(
                    "custom_angle",
                    multiline=True,
                    default="",
                    tooltip="Custom angle description. Example: 'a slightly elevated angle shot of the subject'"
                ),

                # Group 4: Identity Preservation (CRITICAL!)
                io.Boolean.Input(
                    "preserve_identity",
                    default=True,
                    tooltip="⭐ CRITICAL: Preserve subject's face, clothes, pose, hairstyle. ALWAYS keep TRUE for people!"
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
                    "angle_description",
                    tooltip="Description of the selected angle preset"
                ),
                io.String.Output(
                    "system_prompt",
                    tooltip="⭐ v5.0: Research-validated system prompt for person view control! Connect to encoder's system_prompt input."
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        preset,
        scene_context,
        custom_angle,
        preserve_identity,
        debug_mode,
    ) -> io.NodeOutput:
        """Execute the Person View Control node.

        Steps:
        1. Get preset data or use custom angle
        2. Build person view prompt
        3. Get system prompt
        4. Debug output if requested
        5. Return outputs
        """

        # Step 1: Get preset data or use custom angle
        if preset == "custom":
            angle_description = custom_angle
            preset_desc = "Custom angle"
        else:
            preset_data = PERSON_VIEW_PRESETS.get(preset, PERSON_VIEW_PRESETS["eye_level"])
            angle_description = preset_data["angle_description"]
            preset_desc = preset_data["description"]

        # Step 2: Build person view prompt
        prompt = build_person_view_prompt(
            angle_description=angle_description,
            scene_context=scene_context,
            preserve_identity=preserve_identity,
        )

        # Step 3: Get system prompt (v5.0.0 feature - research-validated)
        system_prompt = PERSON_VIEW_SYSTEM_PROMPT

        # Step 4: Debug output
        if debug_mode:
            debug_lines = [
                "=" * 70,
                "ArchAi3D_Qwen_Person_View_Control - Generated Prompt (v5.0.0)",
                "=" * 70,
                f"Preset: {preset}",
                f"Description: {preset_desc}",
                f"Identity Preservation: {'✅ ENABLED (Required!)' if preserve_identity else '❌ DISABLED (Not Recommended!)'}",
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
                "  - Use this node to CHANGE VIEWING ANGLE of person",
                "  - For MOVING CLOSER/FARTHER, use Person_Position_Control",
                "  - For DRAMATIC EFFECTS, use Person_Perspective_Control",
                "  - ⚠️ ALWAYS keep 'preserve_identity' TRUE for people!",
                "",
                "🎭 PSYCHOLOGICAL EFFECTS:",
                "  - Bird's eye / High angle = Vulnerability, weakness, intimacy",
                "  - Eye level = Neutral, equal relationship",
                "  - Low angle / Worm's eye = Power, confidence, heroic",
                "=" * 70,
            ]
            print("\n".join(debug_lines))

        # Step 5: Return
        return io.NodeOutput(prompt, preset_desc, system_prompt)


# ============================================================================
# EXTENSION REGISTRATION
# ============================================================================

class PersonViewControlExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [ArchAi3D_Qwen_Person_View_Control]


async def comfy_entrypoint():
    return PersonViewControlExtension()
