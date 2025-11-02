# -*- coding: utf-8 -*-
"""
ArchAi3D Qwen Object Rotation Control Node (v1)
Author: Amir Ferdos (ArchAi3d)
Email: Amir84ferdos@gmail.com
LinkedIn: https://www.linkedin.com/in/archai3d/

Description:
    Rotate camera around product/object (turntable effect).
    Perfect for 360° product showcases and e-commerce.

    Perfect for:
    - Product turntables (360° rotation)
    - E-commerce showcases (show all sides)
    - Video sequences (multi-frame rotations)
    - Professional product presentations

Based on research:
    - QWEN_PROMPT_GUIDE Function 1: Object Rotation (Most Reliable ⭐⭐⭐⭐⭐)
    - "orbit around [TARGET_OBJECT] [DISTANCE] showing [WHAT_TO_REVEAL]"
    - Multi-step approach for smooth 360° rotations
    - Product photography best practices from Object_Rotation_V2
"""

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

# ============================================================================
# OBJECT ROTATION SYSTEM PROMPT - v5.0.0 Research-Validated
# ============================================================================

OBJECT_ROTATION_SYSTEM_PROMPT = """You are a product photographer. Execute smooth camera orbit around product. Keep product perfectly centered in frame throughout rotation. Maintain consistent distance and lighting. Professional e-commerce quality rotation. Preserve all product details, materials, textures, and colors exactly."""

# ============================================================================
# ROTATION PRESETS - 12 Product Rotation Patterns
# ============================================================================

OBJECT_ROTATION_PRESETS = {
    "custom": {
        "description": "Manual control of rotation parameters",
        "angle": 90,
        "direction": "right",
        "distance": "close",
        "reveal": "all sides",
    },

    # E-COMMERCE TURNTABLES (4) - Product showcases
    "product_turntable_360": {
        "description": "360° smooth turntable - full product rotation, 8 frames",
        "angle": 360,
        "steps": 8,
        "direction": "right",
        "distance": "close",
        "reveal": "all sides",
    },
    "product_four_sides": {
        "description": "360° showing 4 sides - front, right, back, left, 4 frames",
        "angle": 360,
        "steps": 4,
        "direction": "right",
        "distance": "close",
        "reveal": "four main sides",
    },
    "product_six_angles": {
        "description": "360° showing 6 angles - comprehensive view, 6 frames",
        "angle": 360,
        "steps": 6,
        "direction": "right",
        "distance": "close",
        "reveal": "all angles",
    },
    "product_twelve_views": {
        "description": "360° detailed inspection - 12 views, 12 frames",
        "angle": 360,
        "steps": 12,
        "direction": "right",
        "distance": "close",
        "reveal": "detailed inspection views",
    },

    # QUICK ROTATIONS (3) - Partial rotations
    "quarter_turn_90": {
        "description": "90° quarter turn - show side view, 1 frame",
        "angle": 90,
        "steps": 1,
        "direction": "right",
        "distance": "medium",
        "reveal": "the side",
    },
    "half_turn_180": {
        "description": "180° half turn - show opposite side, 1 frame",
        "angle": 180,
        "steps": 1,
        "direction": "right",
        "distance": "medium",
        "reveal": "the back",
    },
    "three_quarter_270": {
        "description": "270° three-quarter rotation - extensive view, 9 frames",
        "angle": 270,
        "steps": 9,
        "direction": "right",
        "distance": "medium",
        "reveal": "three quarters",
    },

    # CINEMATIC (2) - Professional video
    "slow_cinematic_360": {
        "description": "360° ultra-smooth cinema quality - 24 frames",
        "angle": 360,
        "steps": 24,
        "direction": "right",
        "distance": "medium",
        "reveal": "all sides smoothly",
    },
    "hero_reveal_180": {
        "description": "180° dramatic hero reveal - 4 frames",
        "angle": 180,
        "steps": 4,
        "direction": "right",
        "distance": "wide",
        "reveal": "dramatic reveal",
    },

    # SPECIAL (2)
    "inspection_detail": {
        "description": "180° close inspection - detail focus, 6 frames",
        "angle": 180,
        "steps": 6,
        "direction": "right",
        "distance": "close",
        "reveal": "detailed features",
    },
    "social_media_spin": {
        "description": "360° social media optimized - 15 frames (~7.5sec @2fps)",
        "angle": 360,
        "steps": 15,
        "direction": "right",
        "distance": "medium",
        "reveal": "all angles",
    },

    # ARCHITECTURAL (1)
    "architectural_walkaround": {
        "description": "360° building exterior - wide establishing, 8 frames",
        "angle": 360,
        "steps": 8,
        "direction": "right",
        "distance": "wide",
        "reveal": "building in context",
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

def build_object_rotation_prompt(
    scene_context: str,
    direction: str,
    angle: int,
    distance: str,
    reveal: str,
    target: str,
) -> str:
    """Build object rotation prompt.

    Formula from QWEN Guide Function 1 (Lines 71-74):
    [SCENE_CONTEXT], orbit around [TARGET_OBJECT] [DISTANCE] showing [WHAT_TO_REVEAL]

    Args:
        scene_context: Description of the product/object
        direction: Direction to orbit (left/right)
        angle: Rotation angle in degrees
        distance: Orbit distance (close/medium/wide)
        reveal: What to reveal (all sides, details, etc.)
        target: The target object

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

    return ", ".join(parts)


def generate_multi_step_prompts(
    scene_context: str,
    direction: str,
    total_angle: int,
    steps: int,
    distance: str,
    reveal: str,
    target: str,
) -> str:
    """Generate multi-step rotation prompts for video sequences.

    Args:
        scene_context: Description of the product/object
        direction: Direction to orbit
        total_angle: Total rotation angle
        steps: Number of frames
        distance: Orbit distance
        reveal: What to reveal
        target: The target object

    Returns:
        Multi-line string with one prompt per frame
    """
    step_angle = total_angle // steps
    multi_steps = []

    for i in range(steps):
        # Build single step prompt
        step_prompt = build_object_rotation_prompt(
            scene_context=scene_context,
            direction=direction,
            angle=step_angle,
            distance=distance,
            reveal=reveal,
            target=target,
        )

        # Add frame number
        multi_steps.append(f"Frame {i+1}/{steps}: {step_prompt}")

    return "\n".join(multi_steps)


# ============================================================================
# NODE DEFINITION
# ============================================================================

class ArchAi3D_Qwen_Object_Rotation_Control(io.ComfyNode):
    """Object Rotation Control: Rotate camera around product (turntable effect).

    This node uses orbit pattern (QWEN Guide Function 1 - Most Reliable ⭐⭐⭐⭐⭐):
    - 360° product turntables
    - E-commerce showcases
    - Multi-frame video sequences
    - Professional product presentations

    Key Features:
    - 12 preset rotation patterns
    - Research-validated orbit formula (most reliable)
    - Multi-step mode for smooth videos
    - E-commerce standards (4, 6, 8, 12 frame turntables)
    - Automatic system prompt output

    Perfect For:
    - E-commerce product listings (360° turntable)
    - Amazon/Shopify showcases (show all sides)
    - Video sequences (smooth rotations)
    - Professional catalogs (consistent rotation)
    - Social media content (engaging spins)

    Based on research:
    - Object Rotation function (QWEN_PROMPT_GUIDE.md Function 1)
    - "orbit around [TARGET_OBJECT] [DISTANCE] showing [WHAT_TO_REVEAL]"
    - ⭐⭐⭐⭐⭐ Most Reliable (per guide)
    - Multi-step approach for 360° rotations

    USE THIS NODE WHEN:
    - You have a product/object in the image
    - You want to ROTATE AROUND it (turntable)
    - Need 360° product showcase

    SCENE TYPE: Object/Product/Item
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ArchAi3D_Qwen_Object_Rotation_Control",
            category="ArchAi3d/Camera/Object",
            inputs=[
                # Group 1: Preset Selection
                io.Combo.Input(
                    "preset",
                    options=list(OBJECT_ROTATION_PRESETS.keys()),
                    default="product_turntable_360",
                    tooltip="Select rotation preset. Product turntable is standard 360° rotation in 8 frames."
                ),

                # Group 2: Scene Context
                io.String.Input(
                    "scene_context",
                    multiline=True,
                    default="",
                    tooltip="Optional: Product description. Example: 'wooden chair on white background', 'coffee mug'. Improves consistency."
                ),

                # Group 3: Target
                io.String.Input(
                    "target",
                    default="the product",
                    tooltip="Target object to orbit around. Example: 'the product', 'the chair', 'the building'"
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
                    default=90,
                    min=1,
                    max=360,
                    step=1,
                    tooltip="Rotation angle in degrees (1-360°). For 360° turntables, use multi-step mode."
                ),
                io.Combo.Input(
                    "distance",
                    options=["close", "medium", "wide"],
                    default="close",
                    tooltip="Orbit distance: close=product detail, medium=standard, wide=context/environment"
                ),
                io.String.Input(
                    "reveal",
                    default="all sides",
                    tooltip="What to reveal: 'all sides', 'details', 'features', 'back', etc."
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

                # Group 6: Options
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
                    tooltip="⭐ v5.0: Research-validated system prompt for object rotation! Connect to encoder's system_prompt input."
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
        debug_mode,
    ) -> io.NodeOutput:
        """Execute the Object Rotation Control node.

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
            preset_data = OBJECT_ROTATION_PRESETS.get(preset, {})
            final_angle = preset_data.get("angle", angle)
            final_steps = preset_data.get("steps", 1)
            final_direction = preset_data.get("direction", direction)
            final_distance = preset_data.get("distance", distance)
            final_reveal = preset_data.get("reveal", reveal)
            final_multi_step = final_steps > 1  # Presets with steps>1 are multi-step
            preset_desc = preset_data.get("description", "")

        # Step 2: Build single rotation prompt
        prompt = build_object_rotation_prompt(
            scene_context=scene_context,
            direction=final_direction,
            angle=final_angle if not final_multi_step else (final_angle // final_steps),
            distance=final_distance,
            reveal=final_reveal,
            target=target,
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
            )
            frame_count = final_steps

        # Step 4: Get system prompt (v5.0.0 feature - research-validated)
        system_prompt = OBJECT_ROTATION_SYSTEM_PROMPT

        # Step 5: Debug output
        if debug_mode:
            debug_lines = [
                "=" * 70,
                "ArchAi3D_Qwen_Object_Rotation_Control - Generated Prompt (v5.0.0)",
                "=" * 70,
                f"Preset: {preset}",
                f"Description: {preset_desc}",
                f"Angle: {final_angle}°",
                f"Direction: orbit {final_direction}",
                f"Distance: {final_distance}",
                f"Reveal: {final_reveal}",
                f"Target: {target}",
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
                "  - Use this node to ROTATE AROUND product (turntable)",
                "  - For CHANGING ANGLE, use Object_View_Control",
                "  - For MOVING CLOSER/FARTHER, use Object_Position_Control",
                "  - Multi-step mode: Break 360° into smooth video frames",
                "  - E-commerce standard: 8 frames = professional turntable",
                "=" * 70,
            ])
            print("\n".join(debug_lines))

        # Step 6: Return
        return io.NodeOutput(prompt, multi_step_prompts, frame_count, preset_desc, system_prompt)


# ============================================================================
# EXTENSION REGISTRATION
# ============================================================================

class ObjectRotationControlExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [ArchAi3D_Qwen_Object_Rotation_Control]


async def comfy_entrypoint():
    return ObjectRotationControlExtension()
