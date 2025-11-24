# -*- coding: utf-8 -*-
"""
ArchAi3D Qwen Person Rotation Control Node (v1)
Author: Amir Ferdos (ArchAi3d)
Email: Amir84ferdos@gmail.com
LinkedIn: https://www.linkedin.com/in/archai3d/

Description:
    Rotate camera around person/character (orbital turntable effect).
    Perfect for 360° character showcases with identity preservation.

    Perfect for:
    - Character turnarounds (show all sides)
    - Fashion photography (360° outfit view)
    - Character art (multiple angles with consistent identity)
    - AI movies (cinematic orbital shots around characters)

Based on research:
    - QWEN_PROMPT_GUIDE Function 1: Object Rotation (Most Reliable ⭐⭐⭐⭐⭐)
    - "orbit around [TARGET_PERSON] [DISTANCE] showing [WHAT_TO_REVEAL]"
    - Identity preservation is CRITICAL for person edits
    - Adapted from Object_Rotation_Control with person-specific optimizations
"""

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

# ============================================================================
# PERSON ROTATION SYSTEM PROMPT - v5.0.0 Research-Validated
# ============================================================================

PERSON_ROTATION_SYSTEM_PROMPT = """You are a portrait photographer. Execute smooth camera orbit around person while maintaining same elevation. Keep person perfectly centered in frame throughout rotation. Preserve person's identity, facial features, clothing, hairstyle, and pose exactly. Professional portrait photography with consistent lighting and natural framing. Person should remain recognizable and unchanged."""

# ============================================================================
# ROTATION PRESETS - 12 Person Rotation Patterns
# ============================================================================

PERSON_ROTATION_PRESETS = {
    "custom": {
        "description": "Manual control of rotation parameters",
        "angle": 90,
        "direction": "right",
        "distance": "medium",
        "reveal": "all sides",
    },

    # CHARACTER SHOWCASES (4) - Full 360° turnarounds
    "character_turnaround_360": {
        "description": "360° character turnaround - show all sides, 8 frames",
        "angle": 360,
        "steps": 8,
        "direction": "right",
        "distance": "medium",
        "reveal": "all sides of the person",
    },
    "fashion_360_four_sides": {
        "description": "360° fashion showcase - front, side, back, side, 4 frames",
        "angle": 360,
        "steps": 4,
        "direction": "right",
        "distance": "medium",
        "reveal": "outfit from four main angles",
    },
    "character_inspection_12": {
        "description": "360° detailed character inspection - 12 views, 12 frames",
        "angle": 360,
        "steps": 12,
        "direction": "right",
        "distance": "medium",
        "reveal": "detailed character views",
    },
    "editorial_six_angles": {
        "description": "360° editorial portrait - 6 angles, comprehensive view",
        "angle": 360,
        "steps": 6,
        "direction": "right",
        "distance": "medium",
        "reveal": "editorial portrait angles",
    },

    # QUICK ROTATIONS (3) - Partial rotations
    "show_profile_90": {
        "description": "90° to show profile - reveal side view, 1 frame",
        "angle": 90,
        "steps": 1,
        "direction": "right",
        "distance": "medium",
        "reveal": "the profile",
    },
    "show_back_180": {
        "description": "180° to show back - reveal opposite side, 1 frame",
        "angle": 180,
        "steps": 1,
        "direction": "right",
        "distance": "medium",
        "reveal": "the back view",
    },
    "three_quarter_reveal_45": {
        "description": "45° three-quarter reveal - show slight angle, 1 frame",
        "angle": 45,
        "steps": 1,
        "direction": "right",
        "distance": "medium",
        "reveal": "three-quarter view",
    },

    # CINEMATIC (2) - Professional video
    "cinematic_orbit_360": {
        "description": "360° ultra-smooth cinematic orbit - 24 frames",
        "angle": 360,
        "steps": 24,
        "direction": "right",
        "distance": "medium",
        "reveal": "all sides smoothly",
    },
    "dramatic_reveal_180": {
        "description": "180° dramatic character reveal - 4 frames",
        "angle": 180,
        "steps": 4,
        "direction": "right",
        "distance": "wide",
        "reveal": "dramatic reveal",
    },

    # SPECIAL (2)
    "close_detail_inspection": {
        "description": "180° close detail inspection - facial features, 6 frames",
        "angle": 180,
        "steps": 6,
        "direction": "right",
        "distance": "close",
        "reveal": "facial features and details",
    },
    "social_media_spin": {
        "description": "360° social media character spin - 15 frames",
        "angle": 360,
        "steps": 15,
        "direction": "right",
        "distance": "medium",
        "reveal": "character from all angles",
    },

    # ASCENDRA PROJECT (1) - For hooded man scene
    "ascendra_menacing_45": {
        "description": "45° menacing reveal - show threatening character angle",
        "angle": 45,
        "steps": 1,
        "direction": "right",
        "distance": "close",
        "reveal": "menacing profile and expression",
    },
}

# ============================================================================
# MAPPINGS
# ============================================================================

DISTANCE_MAP = {
    "close": "close",
    "medium": "",  # Medium is default, no modifier
    "wide": "wide",
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def build_person_rotation_prompt(
    scene_context: str,
    direction: str,
    angle: int,
    distance: str,
    reveal: str,
    target: str,
    preserve_identity: bool,
) -> str:
    """Build person rotation prompt.

    Formula from QWEN Guide Function 1 (Lines 71-74):
    [SCENE_CONTEXT], orbit around [TARGET_PERSON] [DISTANCE] showing [WHAT_TO_REVEAL], keep identity identical

    Args:
        scene_context: Description of the person/character
        direction: Direction to orbit (left/right)
        angle: Rotation angle in degrees
        distance: Orbit distance (close/medium/wide)
        reveal: What to reveal (all sides, profile, details, etc.)
        target: The target person
        preserve_identity: Whether to include identity preservation clause

    Returns:
        Complete prompt string
    """
    parts = []

    # Scene context
    if scene_context.strip():
        parts.append(scene_context.strip())

    # Build orbit command - QWEN Guide Function 1 is MOST RELIABLE ⭐⭐⭐⭐⭐
    distance_text = DISTANCE_MAP.get(distance, "")

    if distance_text:
        orbit_command = f"{distance_text} orbit {direction} around {target} by {angle} degrees showing {reveal}"
    else:
        orbit_command = f"orbit {direction} around {target} by {angle} degrees showing {reveal}"

    parts.append(orbit_command)

    # Identity preservation clause (CRITICAL for person edits)
    if preserve_identity:
        preservation_clause = "keep the subject's id, clothes, facial features, pose, and hairstyle identical"
        parts.append(preservation_clause)

    return ", ".join(parts)


def generate_multi_step_prompts(
    scene_context: str,
    direction: str,
    total_angle: int,
    steps: int,
    distance: str,
    reveal: str,
    target: str,
    preserve_identity: bool,
) -> str:
    """Generate multi-step rotation prompts for video sequences.

    Args:
        scene_context: Description of the person/character
        direction: Direction to orbit
        total_angle: Total rotation angle
        steps: Number of frames
        distance: Orbit distance
        reveal: What to reveal
        target: The target person
        preserve_identity: Whether to include identity preservation

    Returns:
        Multi-line string with one prompt per frame
    """
    step_angle = total_angle // steps
    multi_steps = []

    for i in range(steps):
        # Build single step prompt
        step_prompt = build_person_rotation_prompt(
            scene_context=scene_context,
            direction=direction,
            angle=step_angle,
            distance=distance,
            reveal=reveal,
            target=target,
            preserve_identity=preserve_identity,
        )

        # Add frame number
        multi_steps.append(f"Frame {i+1}/{steps}: {step_prompt}")

    return "\n".join(multi_steps)


# ============================================================================
# NODE DEFINITION
# ============================================================================

class ArchAi3D_Qwen_Person_Rotation_Control(io.ComfyNode):
    """Person Rotation Control: Rotate camera around person (turntable effect).

    This node uses orbit pattern (QWEN Guide Function 1 - Most Reliable ⭐⭐⭐⭐⭐):
    - 360° character turnarounds
    - Fashion photography showcases
    - Multi-frame video sequences
    - Character art with consistent identity

    Key Features:
    - 12 preset rotation patterns (including ASCENDRA preset!)
    - Research-validated orbit formula (most reliable)
    - Identity preservation (CRITICAL for people!)
    - Multi-step mode for smooth videos
    - Automatic system prompt output

    Perfect For:
    - Character turnarounds (show all sides)
    - Fashion photography (360° outfit showcase)
    - AI movie production (cinematic orbital shots)
    - Character art (multiple angles, same identity)
    - Social media content (engaging character spins)

    Based on research:
    - Object Rotation function (QWEN_PROMPT_GUIDE.md Function 1)
    - "orbit around [TARGET_PERSON] [DISTANCE] showing [WHAT_TO_REVEAL]"
    - ⭐⭐⭐⭐⭐ Most Reliable (per guide)
    - Adapted for person/character with identity preservation

    **CRITICAL:** Always preserve identity for person edits!

    USE THIS NODE WHEN:
    - You have a person/character in the image
    - You want to ROTATE AROUND them (turntable/orbital)
    - Need to show different sides (profile, back, three-quarter)
    - Identity must stay the same (face, clothes, pose)

    SCENE TYPE: Person/Portrait/Character
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ArchAi3D_Qwen_Person_Rotation_Control",
            category="ArchAi3d/Camera/Person",
            inputs=[
                # Group 1: Preset Selection
                io.Combo.Input(
                    "preset",
                    options=list(PERSON_ROTATION_PRESETS.keys()),
                    default="character_turnaround_360",
                    tooltip="Select rotation preset. Character turnaround is standard 360° rotation in 8 frames."
                ),

                # Group 2: Scene Context
                io.String.Input(
                    "scene_context",
                    multiline=True,
                    default="",
                    tooltip="Optional: Person description. Example: 'woman in business suit', 'hooded man in dark jacket'. Improves consistency."
                ),

                # Group 3: Target
                io.String.Input(
                    "target",
                    default="the person",
                    tooltip="Target person to orbit around. Example: 'the person', 'the woman', 'the character', 'the man'"
                ),

                # Group 4: Custom Settings (Custom Mode)
                io.Combo.Input(
                    "direction",
                    options=["left", "right"],
                    default="right",
                    tooltip="Orbit direction: right=clockwise (standard), left=counter-clockwise"
                ),
                io.Int.Input(
                    "angle",
                    default=45,
                    min=1,
                    max=360,
                    step=1,
                    tooltip="Rotation angle in degrees (1-360°). For 360° turnarounds, use multi-step mode."
                ),
                io.Combo.Input(
                    "distance",
                    options=["close", "medium", "wide"],
                    default="medium",
                    tooltip="Orbit distance: close=portrait detail, medium=standard, wide=full context/environment"
                ),
                io.String.Input(
                    "reveal",
                    default="all sides",
                    tooltip="What to reveal: 'all sides', 'profile', 'back view', 'three-quarter view', 'outfit details', etc."
                ),

                # Group 5: Multi-Step Mode
                io.Boolean.Input(
                    "multi_step_mode",
                    default=False,
                    tooltip="Enable multi-step rotation for smooth video sequences (e.g., 360° as 8x45° steps)"
                ),
                io.Int.Input(
                    "steps",
                    default=8,
                    min=2,
                    max=24,
                    step=1,
                    tooltip="Number of frames for multi-step rotation. 8=standard, 12=detailed, 24=ultra-smooth"
                ),

                # Group 6: Identity Preservation (CRITICAL!)
                io.Boolean.Input(
                    "preserve_identity",
                    default=True,
                    tooltip="⭐ CRITICAL: Preserve person's face, clothes, pose, hairstyle. ALWAYS keep TRUE for people!"
                ),

                # Group 7: Options
                io.Boolean.Input(
                    "debug_mode",
                    default=False,
                    tooltip="Print detailed prompt information to console"
                ),
            ],
            outputs=[
                io.String.Output(
                    "prompt",
                    tooltip="Generated rotation prompt for Qwen Edit 2509"
                ),
                io.String.Output(
                    "multi_step_prompts",
                    tooltip="Multi-frame prompts for video sequence (if multi_step_mode=True)"
                ),
                io.Int.Output(
                    "frame_count",
                    tooltip="Number of frames (1 for single, or steps for multi-step)"
                ),
                io.String.Output(
                    "rotation_description",
                    tooltip="Description of the selected rotation preset"
                ),
                io.String.Output(
                    "system_prompt",
                    tooltip="⭐ v5.0: Research-validated system prompt for person rotation! Connect to encoder's system_prompt input."
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        preset,
        scene_context,
        target,
        direction,
        angle,
        distance,
        reveal,
        multi_step_mode,
        steps,
        preserve_identity,
        debug_mode,
    ) -> io.NodeOutput:
        """Execute the Person Rotation Control node.

        Steps:
        1. Apply preset or use custom parameters
        2. Build rotation prompt
        3. Generate multi-step prompts if requested
        4. Get system prompt
        5. Debug output if requested
        6. Return outputs
        """

        # Step 1: Apply preset or use custom parameters
        if preset == "custom":
            # Use custom UI values
            final_angle = angle
            final_steps = steps
            final_direction = direction
            final_distance = distance
            final_reveal = reveal
            final_multi_step = multi_step_mode
            preset_desc = "Custom rotation control"
        else:
            # Use preset values
            preset_data = PERSON_ROTATION_PRESETS.get(preset, {})
            final_angle = preset_data.get("angle", angle)
            final_steps = preset_data.get("steps", 1)
            final_direction = preset_data.get("direction", direction)
            final_distance = preset_data.get("distance", distance)
            final_reveal = preset_data.get("reveal", reveal)
            final_multi_step = final_steps > 1  # Presets with steps>1 are multi-step
            preset_desc = preset_data.get("description", "")

        # Step 2: Build single rotation prompt
        prompt = build_person_rotation_prompt(
            scene_context=scene_context,
            direction=final_direction,
            angle=final_angle if not final_multi_step else (final_angle // final_steps),
            distance=final_distance,
            reveal=final_reveal,
            target=target,
            preserve_identity=preserve_identity,
        )

        # Step 3: Generate multi-step prompts if requested
        multi_step_prompts = ""
        frame_count = 1

        if final_multi_step:
            multi_step_prompts = generate_multi_step_prompts(
                scene_context=scene_context,
                direction=final_direction,
                total_angle=final_angle,
                steps=final_steps,
                distance=final_distance,
                reveal=final_reveal,
                target=target,
                preserve_identity=preserve_identity,
            )
            frame_count = final_steps

        # Step 4: Get system prompt (v5.0.0 feature - research-validated)
        system_prompt = PERSON_ROTATION_SYSTEM_PROMPT

        # Step 5: Debug output
        if debug_mode:
            debug_lines = [
                "=" * 70,
                "ArchAi3D_Qwen_Person_Rotation_Control - Generated Prompt (v5.0.0)",
                "=" * 70,
                f"Preset: {preset}",
                f"Description: {preset_desc}",
                f"Angle: {final_angle}°",
                f"Direction: orbit {final_direction}",
                f"Distance: {final_distance}",
                f"Reveal: {final_reveal}",
                f"Target: {target}",
                f"Identity Preservation: {'✅ ENABLED (Required!)' if preserve_identity else '❌ DISABLED (Not Recommended!)'}",
                f"Multi-step: {final_multi_step} ({final_steps} frames)" if final_multi_step else "Multi-step: No (single frame)",
                f"Scene Context: {scene_context[:80]}..." if len(scene_context) > 80 else f"Scene Context: {scene_context}",
                "=" * 70,
                "Generated Prompt:",
                prompt,
                "=" * 70,
            ]

            if final_multi_step:
                debug_lines.extend([
                    f"Multi-Step Prompts ({final_steps} frames):",
                    multi_step_prompts,
                    "=" * 70,
                ])

            debug_lines.extend([
                "⭐ System Prompt (v5.0.0 - Research-Validated):",
                system_prompt,
                "=" * 70,
                "",
                "💡 TIP: Per QWEN Guide - 'Orbit around' is ⭐⭐⭐⭐⭐ Most Reliable!",
                "  - Use this node to ROTATE AROUND person (turntable/orbital)",
                "  - For CHANGING ANGLE, use Person_View_Control",
                "  - For MOVING CLOSER/FARTHER, use Person_Position_Control",
                "  - For DRAMATIC EFFECTS, use Person_Perspective_Control",
                "  - Multi-step mode: Break 360° into smooth video frames",
                "  - Standard turnaround: 8 frames = professional showcase",
                "  - ⚠️ ALWAYS keep 'preserve_identity' TRUE for people!",
                "",
                "🎬 PERFECT FOR ASCENDRA PROJECT:",
                "  - Use 'ascendra_menacing_45' preset for hooded man shots!",
                "  - Preset optimized for 45° menacing reveal",
                "  - Close distance shows threatening expression",
                "  - Identity preserved for character consistency",
                "=" * 70,
            ])
            print("\n".join(debug_lines))

        # Step 6: Return
        return io.NodeOutput(prompt, multi_step_prompts, frame_count, preset_desc, system_prompt)


# ============================================================================
# EXTENSION REGISTRATION
# ============================================================================

class PersonRotationControlExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [ArchAi3D_Qwen_Person_Rotation_Control]


async def comfy_entrypoint():
    return PersonRotationControlExtension()
