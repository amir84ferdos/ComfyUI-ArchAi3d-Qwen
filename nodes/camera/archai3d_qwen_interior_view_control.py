# -*- coding: utf-8 -*-
"""
ArchAi3D Qwen Interior View Control Node (v5.1.0)
Author: Amir Ferdos (ArchAi3d)
Email: Amir84ferdos@gmail.com
LinkedIn: https://www.linkedin.com/in/archai3d/

Description:
    Change viewing angle for interior room photography.
    Perfect for showing rooms from different perspectives while keeping camera position fixed.

    Perfect for:
    - Real estate photography (room corners, straight-on views)
    - Interior design visualization (showcase different angles)
    - Architectural interiors (ceiling details, floor patterns)
    - Room documentation (systematic coverage)

Based on research:
    - View angle change pattern (QWEN_PROMPT_GUIDE)
    - "change the view to [ANGLE_DESCRIPTION]"
    - Camera position stays same, only rotation changes
    - Natural language descriptions work best

Version History:
    - v5.1.0: Added preservation constraints system (dropdown presets + custom)
    - v5.0.1: Added target_description parameter for better Qwen understanding
    - v5.0.0: Research-validated system prompt
    - v1.0.0: Initial release with 30 preset angles
"""

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

# ============================================================================
# INTERIOR VIEW SYSTEM PROMPT - v5.0.0 Research-Validated
# ============================================================================

INTERIOR_VIEW_SYSTEM_PROMPT = """You are an interior photographer. Change viewing angle within interior room. Show room from different direction or height. Keep camera position same, only rotate view direction. Preserve all furniture, decor, lighting fixtures, wall features, flooring, ceiling details, and spatial relationships exactly. Maintain natural interior perspective and room proportions."""

# ============================================================================
# VIEW ANGLE PRESETS - 30 Interior Photography Angles (12 original + 8 NEW + 10 TEST)
# ============================================================================

INTERIOR_VIEW_PRESETS = {
    "custom": "Manual control of viewing angle",

    # CORNER VIEWS (4) - Most common for interior real estate
    "corner_front_left": "Front-left corner view - shows two walls meeting",
    "corner_front_right": "Front-right corner view - shows two walls meeting",
    "corner_back_left": "Back-left corner view - shows two walls meeting",
    "corner_back_right": "Back-right corner view - shows two walls meeting",

    # WALL VIEWS (4) - Straight-on perspectives
    "wall_frontal": "Straight-on front wall view - symmetric composition, perpendicular to wall",
    "wall_left": "Straight-on left wall view - symmetric composition, directly facing left side",
    "wall_right": "Straight-on right wall view - symmetric composition, directly facing right side",
    "wall_back": "Straight-on back wall view - symmetric composition, perpendicular to back wall",

    # SPECIAL ANGLES (3)
    "ceiling_up": "Looking up at ceiling - shows ceiling details and lighting",
    "floor_down": "Looking down at floor - shows flooring pattern and layout",
    "entrance_view": "View from entrance/doorway - welcoming perspective",

    # DIAGONAL (1)
    "diagonal_across": "Diagonal view across room - maximum depth",

    # NEW ANGLES (8) - Enhanced views based on QWEN research + user testing
    "closeup_shot_angle": "⭐ NEW: Closeup shot view - tight framing of room details",
    "macro_detail_angle": "⭐ NEW: Macro extreme closeup - finest interior textures and materials",
    "wide_contextual_view": "⭐ NEW: Wide contextual view - complete room with all elements",
    "overhead_birds_eye": "⭐ NEW: Overhead bird's eye view - looking straight down into room",
    "ground_level_dramatic": "⭐ NEW: Ground level dramatic view - looking up from floor level",
    "profile_silhouette": "⭐ NEW: Side profile view - clean room silhouette and depth",
    "establishing_intro": "⭐ NEW: Establishing intro view - room introduction and context",
    "aerial_high_angle": "⭐ NEW: Aerial high angle view - elevated perspective of entire room",

    # TEST PRESETS (10) - Wall-specific variations for testing ⭐ FOR YOU TO TEST!
    "test_left_wall_v1": "🧪 TEST: Left wall - facing left wall directly, perpendicular view",
    "test_left_wall_v2": "🧪 TEST: Left wall - looking at left side wall head-on",
    "test_left_wall_v3": "🧪 TEST: Left wall - centered on left wall, square to surface",
    "test_right_wall_v1": "🧪 TEST: Right wall - facing right wall directly, perpendicular view",
    "test_right_wall_v2": "🧪 TEST: Right wall - looking at right side wall head-on",
    "test_front_wall_v1": "🧪 TEST: Front wall - facing front wall directly, centered composition",
    "test_front_wall_v2": "🧪 TEST: Front wall - looking straight ahead at front wall",
    "test_back_wall_v1": "🧪 TEST: Back wall - facing back wall directly, perpendicular view",
    "test_back_wall_v2": "🧪 TEST: Back wall - looking at rear wall head-on",
    "test_wall_comparison": "🧪 TEST: Wall comparison - try all 4 walls to find best pattern",
}

# ============================================================================
# ANGLE DESCRIPTIONS
# ============================================================================

ANGLE_MAP = {
    # Corner views
    "corner_front_left": "front-left corner view showing two adjacent walls",
    "corner_front_right": "front-right corner view showing two adjacent walls",
    "corner_back_left": "back-left corner view showing two adjacent walls",
    "corner_back_right": "back-right corner view showing two adjacent walls",

    # Wall views - ENHANCED with clearer directional language
    "wall_frontal": "view directly facing the front wall head-on, camera perpendicular to the wall surface, centered and symmetric composition",
    "wall_left": "view directly facing the left wall head-on, camera perpendicular to the left side wall surface, centered and symmetric composition",
    "wall_right": "view directly facing the right wall head-on, camera perpendicular to the right side wall surface, centered and symmetric composition",
    "wall_back": "view directly facing the back wall head-on, camera perpendicular to the rear wall surface, centered and symmetric composition",

    # Special angles
    "ceiling_up": "upward view showing ceiling details and lighting fixtures",
    "floor_down": "downward view showing flooring pattern, furniture placement, and room layout",
    "entrance_view": "view from entrance or doorway showing welcoming room perspective",

    # Diagonal
    "diagonal_across": "diagonal view across room showing maximum depth and spatial relationships",

    # NEW angles (8) - Based on QWEN research + user testing
    "closeup_shot_angle": "closeup shot view with tight framing of room details",
    "macro_detail_angle": "macro extreme closeup view revealing finest interior textures and materials",
    "wide_contextual_view": "wide shot showing complete room with all elements and context",
    "overhead_birds_eye": "overhead bird's eye view looking straight down into room",
    "ground_level_dramatic": "ground level dramatic view looking up from floor level",
    "profile_silhouette": "side profile view showing clean room silhouette and depth",
    "establishing_intro": "establishing shot view introducing room and context",
    "aerial_high_angle": "aerial high angle view from elevated perspective of entire room",

    # TEST variations (10) - Different wall prompt patterns for testing
    "test_left_wall_v1": "view facing the left wall directly with camera perpendicular to the left wall surface",
    "test_left_wall_v2": "view looking at the left side wall head-on",
    "test_left_wall_v3": "view centered on the left wall, camera square to the wall surface",
    "test_right_wall_v1": "view facing the right wall directly with camera perpendicular to the right wall surface",
    "test_right_wall_v2": "view looking at the right side wall head-on",
    "test_front_wall_v1": "view facing the front wall directly with centered composition",
    "test_front_wall_v2": "view looking straight ahead at the front wall",
    "test_back_wall_v1": "view facing the back wall directly with camera perpendicular to the rear wall surface",
    "test_back_wall_v2": "view looking at the rear wall head-on",
    "test_wall_comparison": "view of the room showing clear wall orientation",
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def insert_target_description(angle_desc: str, target_desc: str, view_angle: str) -> str:
    """Intelligently insert target description into angle description.

    Based on Qwen prompt guide best practices:
    - Golden Rule #1: "Mention What You Want in Frame"
    - Golden Rule #2: "Use Spatial Relationships"
    - Qwen responds better to target objects than abstract movements

    Args:
        angle_desc: The base angle description from ANGLE_MAP
        target_desc: User's target/feature description
        view_angle: The preset key (to determine insertion strategy)

    Returns:
        Angle description with target intelligently inserted

    Examples:
        >>> insert_target_description("right wall head-on", "that has refrigerator", "wall_right")
        "right wall that has refrigerator head-on"

        >>> insert_target_description("ceiling details and lighting fixtures", "no molding", "ceiling_up")
        "ceiling details and lighting fixtures, no molding"

        >>> insert_target_description("closeup shot view", "oak dining table", "closeup_shot_angle")
        "closeup shot of oak dining table"
    """
    if not target_desc or not target_desc.strip():
        return angle_desc

    target_desc = target_desc.strip()

    # Strategy 1: Wall views - insert before "head-on" or "perpendicular"
    # Pattern: "wall X head-on" → "wall X {target} head-on"
    if "wall" in view_angle and "wall" in angle_desc.lower():
        # Insert before "head-on"
        if "head-on" in angle_desc:
            # Handle case: "view directly facing the right wall head-on, camera perpendicular..."
            parts = angle_desc.split("head-on", 1)
            # Insert target before "head-on"
            return f"{parts[0].rstrip()} {target_desc} head-on{parts[1]}"

        # Insert before "perpendicular to"
        elif "perpendicular to" in angle_desc:
            parts = angle_desc.split("perpendicular to", 1)
            return f"{parts[0].rstrip()} {target_desc}, camera perpendicular to{parts[1]}"

        # Fallback: insert after "wall"
        else:
            return angle_desc.replace("wall", f"wall {target_desc}", 1)

    # Strategy 2: Ceiling/Floor views - add as additional feature description
    # Pattern: "ceiling details..." → "ceiling {target}, details..."
    elif "ceiling" in view_angle or "floor" in view_angle:
        if "," in angle_desc:
            # Insert after first major phrase, before comma
            parts = angle_desc.split(",", 1)
            return f"{parts[0].rstrip()}, {target_desc},{parts[1]}"
        else:
            # Append at end
            return f"{angle_desc} {target_desc}"

    # Strategy 3: Closeup/Macro views - insert as target subject
    # Pattern: "closeup shot view" → "closeup shot of {target}"
    elif "closeup" in view_angle or "macro" in view_angle:
        if " view " in angle_desc:
            # Replace "view" with "of {target}"
            return angle_desc.replace(" view ", f" of {target_desc} ", 1)
        elif " view" in angle_desc:
            return angle_desc.replace(" view", f" of {target_desc}", 1)
        else:
            # Fallback: append
            return f"{angle_desc} of {target_desc}"

    # Strategy 4: Entrance/Profile views - add descriptive context
    elif "entrance" in view_angle or "profile" in view_angle:
        if " showing " in angle_desc:
            # Insert before "showing"
            return angle_desc.replace(" showing ", f" {target_desc} showing ", 1)
        else:
            # Append
            return f"{angle_desc} {target_desc}"

    # Strategy 5: Corner views - add spatial context
    elif "corner" in view_angle:
        if " showing " in angle_desc:
            # Replace generic "showing two adjacent walls" with specific
            parts = angle_desc.split(" showing ", 1)
            return f"{parts[0].rstrip()} {target_desc} showing {parts[1]}"
        else:
            # Append
            return f"{angle_desc} {target_desc}"

    # Default strategy: Append naturally at the end
    else:
        return f"{angle_desc} {target_desc}"


def build_interior_view_prompt(
    view_angle: str,
    scene_context: str,
    custom_angle: str,
    target_description: str = "",
    preservation_preset: str = "none",
    custom_preservation: str = ""
) -> str:
    """Build interior view change prompt.

    Formula from research (v5.1.0 enhanced):
    [SCENE_CONTEXT], change the view to [ANGLE_DESCRIPTION] [TARGET_DESCRIPTION], [PRESERVATION_CONSTRAINT]

    Enhanced with Qwen best practices:
    - Golden Rule #1: "Mention What You Want in Frame"
    - Golden Rule #2: "Use Spatial Relationships"
    - Golden Rule #3: "Preserve Identity & Context"
    - Target objects help Qwen understand better than abstract directions
    - Preservation clauses force layout consistency during camera changes

    Args:
        view_angle: Selected preset view angle
        scene_context: Description of the interior room
        custom_angle: Custom angle description (if preset is "custom")
        target_description: Optional target/feature description (helps Qwen identify the view)
        preservation_preset: Preservation constraint preset (NEW v5.1.0)
        custom_preservation: Custom preservation text (used when preset is "custom")

    Returns:
        Complete prompt string

    Examples:
        >>> build_interior_view_prompt("wall_right", "modern kitchen", "", "that has refrigerator",
        ...                            "keep all furniture and room layout identical")
        "modern kitchen, change the view to right wall that has refrigerator head-on, keep all furniture and room layout identical"

        >>> build_interior_view_prompt("ceiling_up", "bedroom", "", "no molding", "keep everything else identical")
        "bedroom, change the view to ceiling details and lighting fixtures, no molding, keep everything else identical"
    """
    parts = []

    # Scene context
    if scene_context.strip():
        parts.append(scene_context.strip())

    # View angle with optional target description
    if view_angle == "custom":
        if custom_angle.strip():
            angle_text = custom_angle.strip()
            # Add target description if provided for custom angles
            if target_description and target_description.strip():
                angle_text = f"{angle_text} {target_description.strip()}"
            parts.append(f"change the view to {angle_text}")
    else:
        angle_desc = ANGLE_MAP.get(view_angle, "")
        if angle_desc:
            # Insert target description intelligently based on preset type
            if target_description and target_description.strip():
                angle_desc = insert_target_description(angle_desc, target_description, view_angle)

            parts.append(f"change the view to {angle_desc}")

    # Preservation constraint (NEW v5.1.0)
    # Append at end of prompt based on Qwen research patterns
    if preservation_preset and preservation_preset != "none":
        if preservation_preset == "custom":
            # Use custom preservation text
            if custom_preservation and custom_preservation.strip():
                parts.append(custom_preservation.strip())
        else:
            # Use preset preservation clause
            parts.append(preservation_preset)

    return ", ".join(parts)


# ============================================================================
# NODE DEFINITION
# ============================================================================

class ArchAi3D_Qwen_Interior_View_Control(io.ComfyNode):
    """Interior View Control: Change viewing angle within interior rooms.

    This node changes the camera's viewing direction for interior photography:
    - Corner views (show two adjacent walls)
    - Wall views (straight-on symmetric compositions)
    - Special angles (ceiling, floor, entrance, overhead, ground-level)
    - Diagonal views (maximum depth)
    - Detail views (closeup, macro, wide contextual)
    - Dramatic angles (aerial, establishing)

    Key Features:
    - 20 preset viewing angles for interior photography (12 original + 8 NEW)
    - Research-validated angle descriptions + user-tested patterns
    - Camera position stays same (only rotation changes)
    - Natural language, no technical jargon
    - Automatic system prompt output

    Perfect For:
    - Real estate photography (show room from best angles)
    - Interior design presentations (multiple perspectives)
    - Room documentation (systematic coverage)
    - Virtual tours (navigate viewing angles)

    Based on research:
    - View angle change formula (QWEN_PROMPT_GUIDE.md)
    - "change the view to [ANGLE_DESCRIPTION]"
    - Natural language descriptions most reliable
    - Preserve all room elements

    USE THIS NODE WHEN:
    - You want to change viewing ANGLE within a room
    - Camera position stays same, only look direction changes
    - Need different perspectives of same interior space

    SCENE TYPE: Interior/Room/Indoor
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ArchAi3D_Qwen_Interior_View_Control",
            category="ArchAi3d/Camera/Interior",
            inputs=[
                # Group 1: View Angle Selection
                io.Combo.Input(
                    "preset",
                    options=list(INTERIOR_VIEW_PRESETS.keys()),
                    default="corner_front_left",
                    tooltip="Select viewing angle or use 'custom' for manual control. Corner views are most common for real estate."
                ),

                # Group 2: Scene Context
                io.String.Input(
                    "scene_context",
                    multiline=True,
                    default="",
                    tooltip="Optional: Room description. Example: 'modern living room with gray sofa and wooden floor'. Improves consistency."
                ),

                # Group 3: Target/Feature Description (NEW - v5.0.1)
                io.String.Input(
                    "target_description",
                    default="",
                    tooltip="Optional: Describe what's on the wall or target feature. Examples: 'that has refrigerator', 'with large window', 'no molding', 'oak dining table'. Helps Qwen identify the view better. Based on Qwen research: mentioning target objects improves understanding."
                ),

                # Group 4: Preservation Constraints (NEW - v5.1.0)
                io.Combo.Input(
                    "preservation_preset",
                    options=[
                        "none",
                        "keep all furniture and room layout identical",
                        "keep everything else identical",
                        "keep all furniture positions identical",
                        "keep all materials and colors identical",
                        "maintaining distance, keeping camera level",
                        "custom"
                    ],
                    default="none",
                    tooltip="Optional: Add preservation constraint to force layout consistency during camera changes. Based on Qwen best practices. Select 'custom' to write your own constraint."
                ),

                io.String.Input(
                    "custom_preservation",
                    default="",
                    tooltip="Custom preservation constraint (only used when preset is 'custom'). Examples: 'keep refrigerator visible in frame', 'maintain spatial relationships between objects'."
                ),

                # Group 5: Custom Angle (Custom Mode)
                io.String.Input(
                    "custom_angle",
                    default="",
                    tooltip="Custom angle description. Example: 'corner view emphasizing the fireplace', 'view from kitchen towards dining area'"
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
                    "view_description",
                    tooltip="Description of the selected viewing angle"
                ),
                io.String.Output(
                    "system_prompt",
                    tooltip="⭐ v5.0: Research-validated system prompt for interior view control! Connect to encoder's system_prompt input."
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        preset,
        scene_context,
        target_description,
        preservation_preset,
        custom_preservation,
        custom_angle,
        debug_mode,
    ) -> io.NodeOutput:
        """Execute the Interior View Control node.

        Steps:
        1. Get preset description
        2. Build view change prompt (with optional target + preservation)
        3. Get system prompt
        4. Debug output if requested
        5. Return outputs
        """

        # Step 1: Get preset description
        preset_desc = INTERIOR_VIEW_PRESETS.get(preset, "")

        # Step 2: Build view change prompt (v5.1.0 - with preservation constraints)
        prompt = build_interior_view_prompt(
            view_angle=preset,
            scene_context=scene_context,
            custom_angle=custom_angle,
            target_description=target_description,
            preservation_preset=preservation_preset,
            custom_preservation=custom_preservation,
        )

        # Step 3: Get system prompt (v5.0.0 feature - research-validated)
        system_prompt = INTERIOR_VIEW_SYSTEM_PROMPT

        # Step 4: Debug output
        if debug_mode:
            # Determine preservation constraint text for display
            if preservation_preset == "none":
                preservation_display = "(none)"
            elif preservation_preset == "custom":
                preservation_display = f"Custom: {custom_preservation}" if custom_preservation else "Custom: (empty)"
            else:
                preservation_display = preservation_preset

            debug_lines = [
                "=" * 70,
                "ArchAi3D_Qwen_Interior_View_Control - Generated Prompt (v5.1.0)",
                "=" * 70,
                f"Preset: {preset}",
                f"Description: {preset_desc}",
                f"Scene Context: {scene_context[:80]}..." if len(scene_context) > 80 else f"Scene Context: {scene_context}",
                f"Target Description: {target_description}" if target_description else "Target Description: (none)",
                f"Preservation Constraint: {preservation_display}",
                f"Custom Angle: {custom_angle}" if preset == "custom" else "",
                "=" * 70,
                "Generated Prompt:",
                prompt,
                "=" * 70,
                "⭐ System Prompt (v5.0.0 - Research-Validated):",
                system_prompt,
                "=" * 70,
                "",
                "💡 TIP: For best results:",
                "  - Use this node to CHANGE VIEW ANGLE only",
                "  - Use target_description to help Qwen identify walls/features",
                "    Examples: 'that has refrigerator', 'with large window', 'no molding'",
                "  - Use preservation_preset to force layout consistency (NEW v5.1.0)",
                "    Examples: 'keep all furniture and room layout identical'",
                "  - Based on Qwen research: mentioning target objects + preservation improves understanding",
                "  - For MOVING through room, use Interior_Navigation",
                "  - For focusing on specific furniture/objects, use Interior_Focus",
                "=" * 70,
            ]
            print("\n".join(debug_lines))

        # Step 5: Return
        return io.NodeOutput(prompt, preset_desc, system_prompt)


# ============================================================================
# EXTENSION REGISTRATION
# ============================================================================

class InteriorViewControlExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [ArchAi3D_Qwen_Interior_View_Control]


async def comfy_entrypoint():
    return InteriorViewControlExtension()
