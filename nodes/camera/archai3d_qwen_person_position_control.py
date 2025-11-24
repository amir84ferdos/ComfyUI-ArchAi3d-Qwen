# -*- coding: utf-8 -*-
"""
ArchAi3D Qwen Person Position Control Node (v1)
Author: Amir Ferdos (ArchAi3d)
Email: Amir84ferdos@gmail.com
LinkedIn: https://www.linkedin.com/in/archai3d/

Description:
    Position camera closer/farther for portrait photography using dolly movements.
    Move closer for intimate portraits or farther for environmental context.

    Perfect for:
    - Portrait framing (headshot, bust, full-body)
    - Environmental portraits (person in context)
    - Detail shots (face closeups)
    - Professional portrait compositions

Based on research:
    - QWEN_PROMPT_GUIDE Function 6: Dolly (⭐⭐⭐⭐⭐ Most Consistent for Zoom)
    - "change the view dolly [DIRECTION] [PREPOSITION] the [TARGET]"
    - Identity preservation is CRITICAL for person edits
    - Portrait photography composition standards
"""

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

# ============================================================================
# PERSON POSITION SYSTEM PROMPT - v5.0.0 Research-Validated
# ============================================================================

PERSON_POSITION_SYSTEM_PROMPT = """You are a portrait photographer. Move camera closer to or farther from subject along viewing axis. Frame subject at requested distance while maintaining same viewing angle. Preserve subject's identity, facial features, clothing, hairstyle, and pose exactly. Professional portrait photography composition with proper framing and depth. Person should remain recognizable and unchanged."""

# ============================================================================
# POSITION PRESETS - 10 Portrait Photography Distances
# ============================================================================

PERSON_POSITION_PRESETS = {
    "custom": {
        "description": "Manual control of camera position",
        "movement": "in",
        "target": "the subject",
    },

    # ZOOM IN / DOLLY IN (5) - Getting closer
    "extreme_closeup_face": {
        "description": "Extreme closeup - eyes and face details, intimate portrait",
        "movement": "in",
        "target": "the subject's face",
        "speed": "smooth",
    },
    "closeup_headshot": {
        "description": "Closeup headshot - head and shoulders, professional headshot",
        "movement": "in",
        "target": "the subject",
        "speed": "smooth",
    },
    "tight_bust": {
        "description": "Tight bust shot - head to mid-chest, tight framing",
        "movement": "in",
        "target": "the subject",
        "speed": "smooth",
    },
    "medium_closeup": {
        "description": "Medium closeup - head to waist, conversational distance",
        "movement": "in",
        "target": "the subject",
        "speed": "smooth",
    },
    "portrait_standard": {
        "description": "Standard portrait - head to mid-thigh, classic portrait framing",
        "movement": "in",
        "target": "the subject",
        "speed": "smooth",
    },

    # ZOOM OUT / DOLLY OUT (5) - Getting farther
    "full_body": {
        "description": "Full body shot - entire person head to toe, fashion/editorial style",
        "movement": "out",
        "target": "the subject",
        "speed": "smooth",
    },
    "environmental_portrait": {
        "description": "Environmental portrait - person in context, show surroundings",
        "movement": "out",
        "target": "the subject",
        "speed": "smooth",
    },
    "wide_context": {
        "description": "Wide context - person in full environment, storytelling shot",
        "movement": "out",
        "target": "the subject",
        "speed": "smooth",
    },
    "lifestyle_wide": {
        "description": "Lifestyle wide - person as part of scene, lifestyle photography",
        "movement": "out",
        "target": "the subject",
        "speed": "smooth",
    },

    # SPECIAL (1)
    "cinematic_hero": {
        "description": "Cinematic hero framing - optimal showcase distance, dramatic",
        "movement": "in",
        "target": "the subject",
        "speed": "cinematic",
    },
}

# ============================================================================
# MOVEMENT MAPPINGS
# ============================================================================

MOVEMENT_MAP = {
    "in": {"direction": "in", "preposition": "towards"},
    "out": {"direction": "out", "preposition": "from"},
}

SPEED_MAP = {
    "none": "",
    "smooth": "smooth camera movement",
    "slow": "slow steady movement",
    "fast": "quick movement",
    "cinematic": "cinematic smooth movement",
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def build_person_position_prompt(
    scene_context: str,
    movement: str,
    target: str,
    speed: str,
    preserve_identity: bool,
) -> str:
    """Build person position control prompt.

    Formula from QWEN Guide Function 6 (Line 299-301):
    [SCENE_CONTEXT], change the view dolly [DIRECTION] [PREPOSITION] the [TARGET] [SPEED], keep identity identical

    Args:
        scene_context: Description of the person/scene
        movement: "in" (closer) or "out" (farther)
        target: The subject (e.g., "the subject", "the subject's face")
        speed: Movement speed modifier
        preserve_identity: Whether to include identity preservation clause

    Returns:
        Complete prompt string
    """
    parts = []

    # Scene context
    if scene_context.strip():
        parts.append(scene_context.strip())

    # Build dolly command - QWEN Guide Function 6 is most consistent for zoom
    movement_data = MOVEMENT_MAP.get(movement, MOVEMENT_MAP["in"])
    dolly_command = f"change the view dolly {movement_data['direction']} {movement_data['preposition']} {target}"

    # Add speed modifier
    speed_text = SPEED_MAP.get(speed, "")
    if speed_text:
        dolly_command += f" {speed_text}"

    parts.append(dolly_command)

    # Identity preservation clause (CRITICAL for person edits)
    if preserve_identity:
        preservation_clause = "keep the subject's id, clothes, facial features, pose, and hairstyle identical"
        parts.append(preservation_clause)

    return ", ".join(parts)


# ============================================================================
# NODE DEFINITION
# ============================================================================

class ArchAi3D_Qwen_Person_Position_Control(io.ComfyNode):
    """Person Position Control: Move camera closer/farther for portrait photography.

    This node uses dolly movement (QWEN Guide Function 6 - most consistent for zoom):
    - Dolly in (zoom in): Closeups, headshots, intimate portraits
    - Dolly out (zoom out): Full-body, environmental portraits, context
    - **Identity preservation is CRITICAL** (face, clothes, pose stay same)

    Key Features:
    - 10 preset portrait distances
    - Research-validated dolly pattern (most reliable for zoom)
    - Identity preservation clause (REQUIRED for person edits)
    - Professional portrait framing standards
    - Automatic system prompt output

    Perfect For:
    - Professional headshots (closeup headshot preset)
    - Fashion photography (full body preset)
    - Editorial portraits (environmental portrait preset)
    - Lifestyle photography (wide context preset)
    - Intimate portraits (extreme closeup preset)

    Based on research:
    - Dolly function (QWEN_PROMPT_GUIDE.md Function 6)
    - "change the view dolly [in/out] [towards/from] the [TARGET]"
    - ⭐⭐⭐⭐⭐ Most Consistent for Zoom (per guide)
    - Portrait photography composition rules

    **CRITICAL:** Always preserve identity for person edits!

    USE THIS NODE WHEN:
    - You have a person/character in the image
    - You want to move CLOSER or FARTHER
    - Identity must stay the same (face, clothes, pose)

    SCENE TYPE: Person/Portrait/Character
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ArchAi3D_Qwen_Person_Position_Control",
            category="ArchAi3d/Camera/Person",
            inputs=[
                # Group 1: Preset Selection
                io.Combo.Input(
                    "preset",
                    options=list(PERSON_POSITION_PRESETS.keys()),
                    default="portrait_standard",
                    tooltip="Select camera distance preset. Portrait standard is classic head-to-mid-thigh framing."
                ),

                # Group 2: Scene Context
                io.String.Input(
                    "scene_context",
                    multiline=True,
                    default="",
                    tooltip="Optional: Person/scene description. Example: 'woman in business attire', 'man standing in office'. Improves consistency."
                ),

                # Group 3: Custom Settings (Custom Mode)
                io.Combo.Input(
                    "movement",
                    options=["in", "out"],
                    default="in",
                    tooltip="Dolly direction: 'in'=move closer (zoom in), 'out'=move farther (zoom out)"
                ),
                io.String.Input(
                    "custom_target",
                    default="the subject",
                    tooltip="Custom target description. Example: 'the subject', 'the subject's face', 'the person'"
                ),
                io.Combo.Input(
                    "speed",
                    options=["none", "smooth", "slow", "fast", "cinematic"],
                    default="smooth",
                    tooltip="Movement quality: smooth=standard, cinematic=high-end, none=no speed modifier"
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
                    "position_description",
                    tooltip="Description of the selected position preset"
                ),
                io.String.Output(
                    "system_prompt",
                    tooltip="⭐ v5.0: Research-validated system prompt for person position control! Connect to encoder's system_prompt input."
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        preset,
        scene_context,
        movement,
        custom_target,
        speed,
        preserve_identity,
        debug_mode,
    ) -> io.NodeOutput:
        """Execute the Person Position Control node.

        Steps:
        1. Apply preset or use custom parameters
        2. Build position control prompt
        3. Get system prompt
        4. Debug output if requested
        5. Return outputs
        """

        # Step 1: Apply preset or use custom parameters
        if preset == "custom":
            # Use custom UI values
            final_movement = movement
            final_target = custom_target
            final_speed = speed
            preset_desc = "Custom position control"
        else:
            # Use preset values
            preset_data = PERSON_POSITION_PRESETS.get(preset, {})
            final_movement = preset_data.get("movement", movement)
            final_target = preset_data.get("target", custom_target)
            final_speed = preset_data.get("speed", speed)
            preset_desc = preset_data.get("description", "")

        # Step 2: Build position control prompt
        prompt = build_person_position_prompt(
            scene_context=scene_context,
            movement=final_movement,
            target=final_target,
            speed=final_speed,
            preserve_identity=preserve_identity,
        )

        # Step 3: Get system prompt (v5.0.0 feature - research-validated)
        system_prompt = PERSON_POSITION_SYSTEM_PROMPT

        # Step 4: Debug output
        if debug_mode:
            debug_lines = [
                "=" * 70,
                "ArchAi3D_Qwen_Person_Position_Control - Generated Prompt (v5.0.0)",
                "=" * 70,
                f"Preset: {preset}",
                f"Description: {preset_desc}",
                f"Movement: dolly {final_movement}",
                f"Target: {final_target}",
                f"Speed: {final_speed}",
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
                "💡 TIP: Per QWEN Guide - Dolly is ⭐⭐⭐⭐⭐ Most Consistent for Zoom!",
                "  - Use this node to MOVE CLOSER/FARTHER along viewing axis",
                "  - For CHANGING ANGLE, use Person_View_Control",
                "  - For DRAMATIC EFFECTS, use Person_Perspective_Control",
                "  - Dolly in = zoom in (closer)",
                "  - Dolly out = zoom out (farther)",
                "  - ⚠️ ALWAYS keep 'preserve_identity' TRUE for people!",
                "",
                "📸 PORTRAIT FRAMING GUIDE:",
                "  - Extreme closeup: Eyes and face details",
                "  - Headshot: Head and shoulders (LinkedIn style)",
                "  - Bust shot: Head to chest (conversational)",
                "  - Portrait standard: Head to mid-thigh (classic)",
                "  - Full body: Entire person (fashion/editorial)",
                "  - Environmental: Person in context (storytelling)",
                "=" * 70,
            ]
            print("\n".join(debug_lines))

        # Step 5: Return
        return io.NodeOutput(prompt, preset_desc, system_prompt)


# ============================================================================
# EXTENSION REGISTRATION
# ============================================================================

class PersonPositionControlExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [ArchAi3D_Qwen_Person_Position_Control]


async def comfy_entrypoint():
    return PersonPositionControlExtension()
