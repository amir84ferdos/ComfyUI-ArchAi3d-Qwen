# -*- coding: utf-8 -*-
"""
ArchAi3D Qwen Object Position Control Node (v1)
Author: Amir Ferdos (ArchAi3d)
Email: Amir84ferdos@gmail.com
LinkedIn: https://www.linkedin.com/in/archai3d/

Description:
    Position camera for product/object photography using dolly movements.
    Move closer for details or farther for context.

    Perfect for:
    - Product closeup shots (zoom in on details)
    - Context reveals (zoom out to show environment)
    - Detail emphasis (focus on specific features)
    - Professional product framing

Based on research:
    - QWEN_PROMPT_GUIDE Function 6: Dolly (Zoom In/Out)
    - "change the view dolly [DIRECTION] [PREPOSITION] the [TARGET]"
    - Most consistent method for zoom according to guide
    - Product photography best practices
"""

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

# ============================================================================
# OBJECT POSITION SYSTEM PROMPT - v5.0.0 Research-Validated
# ============================================================================

OBJECT_POSITION_SYSTEM_PROMPT = """You are a product photographer. Move camera closer to or farther from product along viewing axis. Frame product at requested distance while maintaining same viewing angle. Preserve all product details, materials, textures, colors, and lighting exactly. Professional product photography composition with proper framing and depth."""

# ============================================================================
# PROMPT STYLE PATTERNS - Dual System (User-Tested + Doc-Verified)
# ============================================================================

# User-tested pattern: "edit camera view, zoom the camera on [target], [framing]"
# Real-world test: ✅ WORKS! "edit camera view, zoom the camera on sink, closeup shot"

# Doc-verified pattern: "change the view dolly [in/out] [towards/from] the [target]"
# QWEN Guide Function 6: ⭐⭐⭐⭐⭐ Most Consistent for Zoom

FRAMING_DESCRIPTORS = {
    "macro": "macro extreme closeup",
    "extreme": "extreme closeup shot",
    "closeup": "closeup shot",
    "detail": "detail view",
    "medium": "medium shot",
    "standard": "standard view",
    "wide": "wide shot",
    "establishing": "establishing shot",
    "flatlay": "overhead flat-lay view",
}

# ============================================================================
# POSITION PRESETS - 18 Product Photography Distances (12 original + 6 new)
# ============================================================================

OBJECT_POSITION_PRESETS = {
    "custom": {
        "description": "Manual control of camera position",
        "movement": "in",
        "target": "",
    },

    # ZOOM IN / DOLLY IN (5) - Getting closer
    "extreme_closeup": {
        "description": "Extreme closeup - reveal finest details, textures, materials",
        "movement": "in",
        "target": "the product",
        "speed": "smooth",
    },
    "detail_shot": {
        "description": "Detail shot - emphasize specific product feature or element",
        "movement": "in",
        "target": "the product detail",
        "speed": "smooth",
    },
    "closeup": {
        "description": "Closeup - frame product tightly, fill frame",
        "movement": "in",
        "target": "the product",
        "speed": "smooth",
    },
    "medium_close": {
        "description": "Medium closeup - product prominent with some breathing room",
        "movement": "in",
        "target": "the product",
        "speed": "smooth",
    },
    "product_focus": {
        "description": "Product focus - isolate product, minimize distractions",
        "movement": "in",
        "target": "the product",
        "speed": "smooth",
    },

    # ZOOM OUT / DOLLY OUT (4) - Getting farther
    "medium_shot": {
        "description": "Medium shot - product in context, balanced composition",
        "movement": "out",
        "target": "the product",
        "speed": "smooth",
    },
    "context_reveal": {
        "description": "Context reveal - show product in environment, tell story",
        "movement": "out",
        "target": "the product",
        "speed": "smooth",
    },
    "full_product": {
        "description": "Full product shot - entire product visible with margins",
        "movement": "out",
        "target": "the product",
        "speed": "smooth",
    },
    "environment_wide": {
        "description": "Wide environment - product in full setting, establish scene",
        "movement": "out",
        "target": "the product",
        "speed": "smooth",
    },

    # SPECIAL POSITIONS (2)
    "hero_framing": {
        "description": "Hero framing - optimal product showcase distance",
        "movement": "in",
        "target": "the product",
        "speed": "cinematic",
    },
    "ecommerce_standard": {
        "description": "E-commerce standard - professional product listing distance",
        "movement": "in",
        "target": "the product",
        "speed": "smooth",
    },

    # NEW PRESETS (6) - Enhanced framing options based on research + user testing
    "macro_extreme": {
        "description": "⭐ NEW: Macro extreme closeup - finest texture details, materials (<0.5m)",
        "movement": "in",
        "target": "the product",
        "speed": "smooth",
        "framing": "macro",
    },
    "closeup_shot": {
        "description": "⭐ NEW: Closeup shot - user-tested pattern! Frame tightly (1-1.5m)",
        "movement": "in",
        "target": "the product",
        "speed": "smooth",
        "framing": "closeup",
    },
    "detail_view": {
        "description": "⭐ NEW: Detail view - emphasize features and elements (1.5-2m)",
        "movement": "in",
        "target": "the product",
        "speed": "smooth",
        "framing": "detail",
    },
    "wide_shot": {
        "description": "⭐ NEW: Wide shot - product with surrounding environment (3-5m)",
        "movement": "out",
        "target": "the product",
        "speed": "smooth",
        "framing": "wide",
    },
    "establishing_shot": {
        "description": "⭐ NEW: Establishing shot - context-setting intro view (5m+)",
        "movement": "out",
        "target": "the product",
        "speed": "smooth",
        "framing": "establishing",
    },
    "overhead_flatlay": {
        "description": "⭐ NEW: Overhead flat-lay - top-down product photography",
        "movement": "in",
        "target": "the product",
        "speed": "smooth",
        "framing": "flatlay",
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

def build_object_position_prompt(
    scene_context: str,
    movement: str,
    target: str,
    speed: str,
    prompt_style: str = "dolly",
    framing: str = None,
) -> str:
    """Build object position control prompt with dual-pattern system.

    Two supported patterns:

    1. DOLLY pattern (QWEN Guide Function 6 - ⭐⭐⭐⭐⭐):
       [SCENE_CONTEXT], change the view dolly [DIRECTION] [PREPOSITION] the [TARGET] [SPEED]

    2. ZOOM pattern (User-tested - ✅ Real-world proven):
       edit camera view, zoom the camera on [TARGET], [FRAMING_DESCRIPTOR]
       Example: "edit camera view, zoom the camera on sink, closeup shot"

    Args:
        scene_context: Description of the product/object
        movement: "in" (closer) or "out" (farther)
        target: The subject (e.g., "the product", "the watch face")
        speed: Movement speed modifier
        prompt_style: "dolly" (doc-verified) or "zoom" (user-tested)
        framing: Framing descriptor key for zoom pattern (e.g., "closeup", "wide")

    Returns:
        Complete prompt string
    """
    parts = []

    # Scene context (for dolly pattern only)
    if prompt_style == "dolly" and scene_context.strip():
        parts.append(scene_context.strip())

    # Build camera command based on selected pattern
    if prompt_style == "zoom":
        # User-tested pattern: "edit camera view, zoom the camera on [target], [framing]"
        camera_command = f"edit camera view, zoom the camera on {target}"

        # Add framing descriptor if available
        if framing and framing in FRAMING_DESCRIPTORS:
            camera_command += f", {FRAMING_DESCRIPTORS[framing]}"

        parts.append(camera_command)

    else:  # dolly (default)
        # Doc-verified pattern: "change the view dolly [in/out] [towards/from] the [target]"
        movement_data = MOVEMENT_MAP.get(movement, MOVEMENT_MAP["in"])
        dolly_command = f"change the view dolly {movement_data['direction']} {movement_data['preposition']} {target}"

        # Add speed modifier
        speed_text = SPEED_MAP.get(speed, "")
        if speed_text:
            dolly_command += f" {speed_text}"

        parts.append(dolly_command)

    return ", ".join(parts)


# ============================================================================
# NODE DEFINITION
# ============================================================================

class ArchAi3D_Qwen_Object_Position_Control(io.ComfyNode):
    """Object Position Control: Move camera closer/farther for product photography.

    This node uses dolly movement (QWEN Guide Function 6 - most consistent for zoom):
    - Dolly in (zoom in): Move closer to product
    - Dolly out (zoom out): Move farther from product
    - Professional framing presets
    - E-commerce standards

    Key Features:
    - 12 preset position/distance options
    - Research-validated dolly pattern (most reliable for zoom)
    - Smooth camera movements
    - Professional product photography framing
    - Automatic system prompt output

    Perfect For:
    - Product detail shots (extreme closeup, detail shot)
    - E-commerce listings (standard product framing)
    - Context reveals (show product in environment)
    - Professional catalogs (consistent framing)
    - Hero shots (optimal showcase distance)

    Based on research:
    - Dolly function (QWEN_PROMPT_GUIDE.md Function 6)
    - "change the view dolly [in/out] [towards/from] the [TARGET]"
    - ⭐⭐⭐⭐⭐ Most Consistent for Zoom (per guide)
    - Product photography composition rules

    USE THIS NODE WHEN:
    - You have a product/object in the image
    - You want to move CLOSER or FARTHER
    - Need professional product framing

    SCENE TYPE: Object/Product/Item
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ArchAi3D_Qwen_Object_Position_Control",
            category="ArchAi3d/Camera/Object",
            inputs=[
                # Group 1: Preset Selection
                io.Combo.Input(
                    "preset",
                    options=list(OBJECT_POSITION_PRESETS.keys()),
                    default="ecommerce_standard",
                    tooltip="Select camera position preset. E-commerce standard is professional product listing distance."
                ),

                # Group 2: Prompt Style (DUAL PATTERN SYSTEM)
                io.Combo.Input(
                    "prompt_style",
                    options=["dolly", "zoom"],
                    default="dolly",
                    tooltip="⭐ Pattern choice: 'dolly' (QWEN Guide ⭐⭐⭐⭐⭐ verified) OR 'zoom' (user-tested real-world pattern). Both work!"
                ),

                # Group 3: Scene Context
                io.String.Input(
                    "scene_context",
                    multiline=True,
                    default="",
                    tooltip="Optional: Product description. Example: 'stainless steel watch', 'coffee mug on wooden table'. Improves consistency."
                ),

                # Group 4: Custom Settings (Custom Mode)
                io.Combo.Input(
                    "movement",
                    options=["in", "out"],
                    default="in",
                    tooltip="Dolly direction: 'in'=move closer (zoom in), 'out'=move farther (zoom out)"
                ),
                io.String.Input(
                    "custom_target",
                    default="the product",
                    tooltip="Custom target description. Example: 'the watch face', 'the logo', 'the handle'"
                ),
                io.Combo.Input(
                    "speed",
                    options=["none", "smooth", "slow", "fast", "cinematic"],
                    default="smooth",
                    tooltip="Movement quality: smooth=standard, cinematic=high-end, none=no speed modifier"
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
                    tooltip="⭐ v5.0: Research-validated system prompt for object position control! Connect to encoder's system_prompt input."
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        preset,
        prompt_style,
        scene_context,
        movement,
        custom_target,
        speed,
        debug_mode,
    ) -> io.NodeOutput:
        """Execute the Object Position Control node.

        Steps:
        1. Apply preset or use custom parameters
        2. Build position control prompt (with selected pattern style)
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
            final_framing = None
            preset_desc = "Custom position control"
        else:
            # Use preset values
            preset_data = OBJECT_POSITION_PRESETS.get(preset, {})
            final_movement = preset_data.get("movement", movement)
            final_target = preset_data.get("target", custom_target)
            final_speed = preset_data.get("speed", speed)
            final_framing = preset_data.get("framing", None)  # New: framing descriptor
            preset_desc = preset_data.get("description", "")

        # Step 2: Build position control prompt with selected pattern
        prompt = build_object_position_prompt(
            scene_context=scene_context,
            movement=final_movement,
            target=final_target,
            speed=final_speed,
            prompt_style=prompt_style,
            framing=final_framing,
        )

        # Step 3: Get system prompt (v5.0.0 feature - research-validated)
        system_prompt = OBJECT_POSITION_SYSTEM_PROMPT

        # Step 4: Debug output
        if debug_mode:
            pattern_info = ""
            if prompt_style == "zoom":
                pattern_info = "⭐ Using ZOOM pattern (user-tested real-world proven)"
                if final_framing:
                    pattern_info += f"\n  Framing: {FRAMING_DESCRIPTORS.get(final_framing, final_framing)}"
            else:
                pattern_info = "⭐ Using DOLLY pattern (QWEN Guide ⭐⭐⭐⭐⭐ verified)"

            debug_lines = [
                "=" * 70,
                "ArchAi3D_Qwen_Object_Position_Control - Generated Prompt (v5.0.0+)",
                "=" * 70,
                f"Preset: {preset}",
                f"Description: {preset_desc}",
                f"Prompt Style: {prompt_style.upper()}",
                pattern_info,
                f"Movement: {final_movement}",
                f"Target: {final_target}",
                f"Speed: {final_speed}",
                f"Scene Context: {scene_context[:80]}..." if len(scene_context) > 80 else f"Scene Context: {scene_context}",
                "=" * 70,
                "Generated Prompt:",
                prompt,
                "=" * 70,
                "⭐ System Prompt (v5.0.0 - Research-Validated):",
                system_prompt,
                "=" * 70,
                "",
                "💡 DUAL PATTERN SYSTEM:",
                "  - DOLLY pattern: QWEN Guide ⭐⭐⭐⭐⭐ verified (technical term)",
                "  - ZOOM pattern: User-tested ✅ real-world proven (natural language)",
                "  - Both patterns work! Choose what feels more natural to you.",
                "",
                "💡 NODE USAGE:",
                "  - Use this node to MOVE CLOSER/FARTHER along viewing axis",
                "  - For CHANGING ANGLE, use Object_View_Control",
                "  - For ROTATING PRODUCT (turntable), use Object_Rotation_Control",
                "=" * 70,
            ]
            print("\n".join(debug_lines))

        # Step 5: Return
        return io.NodeOutput(prompt, preset_desc, system_prompt)


# ============================================================================
# EXTENSION REGISTRATION
# ============================================================================

class ObjectPositionControlExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [ArchAi3D_Qwen_Object_Position_Control]


async def comfy_entrypoint():
    return ObjectPositionControlExtension()
