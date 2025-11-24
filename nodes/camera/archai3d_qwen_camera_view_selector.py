"""
ArchAi3D Qwen Camera View Selector Node
Author: Amir Ferdos (ArchAi3d)
Email: Amir84ferdos@gmail.com
LinkedIn: https://www.linkedin.com/in/archai3d/

Description:
    Quick selection from professional camera viewpoints. Perfect for:
    - Architectural photography (elevation views, sections, perspectives)
    - Product photography (hero shots, detail shots, 360° views)
    - Interior design (room corners, walkthroughs, overhead plans)
    - Portrait photography (eye level, high angle, low angle, worm's eye, bird's eye)

Based on research: Qwen works best with named viewpoints like "front view",
"side view", "top-down view" rather than abstract camera movements.
"""

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override
from typing import Dict

# ============================================================================
# CAMERA VIEW PRESETS - Professional photography viewpoints
# ============================================================================

# System prompts for each camera view (research-validated, v4.0.0)
CAMERA_VIEW_SYSTEM_PROMPTS = {
    # Basic Orthographic Views
    "front_view": "You are a professional camera operator. Maintain front-facing perspective. Keep the camera stationary directly in front of the subject. Do not orbit or rotate around objects. Preserve all scene elements, lighting, and composition.",

    "back_view": "You are a professional camera operator. Maintain rear-facing perspective. Keep the camera stationary directly behind the subject. Do not orbit or rotate around objects. Preserve all scene elements, lighting, and composition.",

    "left_side_view": "You are a professional camera operator. Maintain left side perspective. Keep the camera stationary on the left side of the subject. Do not orbit or rotate around objects. Preserve all scene elements, lighting, and composition.",

    "right_side_view": "You are a professional camera operator. Maintain right side perspective. Keep the camera stationary on the right side of the subject. Do not orbit or rotate around objects. Preserve all scene elements, lighting, and composition.",

    "top_view": "You are a professional camera operator. Maintain top-down bird's eye perspective. Keep the camera stationary looking straight down from above. Do not tilt or rotate. Preserve all scene elements, spatial relationships, and layout.",

    "bottom_view": "You are a professional camera operator. Maintain extreme bottom-up worm's eye perspective. Keep the camera stationary looking straight up from below. Do not tilt or rotate. Preserve all scene elements and dramatic angles.",

    # Isometric and 3/4 Views
    "three_quarter_view": "You are a product photographer. Maintain three-quarter perspective showing both front and side. This is the standard e-commerce angle. Keep camera position consistent. Preserve all product details, lighting, and proportions.",

    "isometric_view": "You are a technical illustrator. Maintain isometric projection with no perspective distortion. Keep parallel lines parallel. This is technical drawing style. Preserve all dimensional relationships and clarity.",

    # Portrait/Person Angles
    "eye_level": "You are a portrait photographer. Maintain eye-level camera angle positioned at subject's eye height. This creates natural, engaging perspective. Preserve facial proportions, natural features, and authentic expression. Professional portrait framing.",

    "high_angle": "You are a portrait photographer. Maintain high-angle perspective looking down at subject. Camera positioned above subject, angled downward. This creates vulnerability or intimacy. Preserve facial features and natural proportions.",

    "low_angle": "You are a portrait photographer. Maintain low-angle perspective looking up at subject. Camera positioned below subject, angled upward. This creates power, dominance, or heroic quality. Preserve facial features and dramatic impact.",

    "birds_eye": "You are a portrait photographer. Maintain ultra-high bird's eye perspective. Camera positioned very high up, looking directly down at subject from above. This creates unique overhead perspective. Preserve subject positioning and spatial context.",

    "worms_eye": "You are a portrait photographer. Maintain ultra-low worm's eye perspective. Camera positioned very low to ground, looking directly up at subject. This creates extreme dramatic angle emphasizing height and power. Preserve subject details and dramatic impact.",

    # Architectural Views
    "section_view": "You are an architectural photographer. Maintain section cut perspective as if building is sliced open. Show interior cross-section view. Preserve architectural details, spatial relationships, and structural clarity.",

    "aerial_view": "You are an architectural photographer. Maintain elevated aerial perspective at approximately 45-degree angle. Show overview of site from elevated position. Preserve building proportions, site context, and spatial relationships.",

    "street_level": "You are an architectural photographer. Maintain street-level human eye height perspective. This is natural pedestrian viewpoint. Preserve building scale, architectural details, and street context. Professional real estate photography quality.",

    # Interior Design Views
    "corner_view": "You are an interior design photographer. Maintain corner perspective capturing where two walls meet. This showcases room depth and space. Preserve room proportions, architectural lines, and interior design elements.",

    "entrance_view": "You are an interior design photographer. Maintain entrance perspective from doorway looking into space. This is first impression viewpoint. Preserve room layout, design flow, and welcoming atmosphere.",

    "ceiling_view": "You are an interior design photographer. Maintain ceiling perspective with camera tilted way up. Showcase ceiling details, lighting fixtures, and vertical space. Preserve architectural ceiling features and room height.",

    # Cinematic Views
    "dutch_angle": "You are a cinematic camera operator. Maintain dutch angle with tilted horizon line for dramatic effect. Camera rotated on axis creating diagonal composition. This creates tension or unease. Preserve scene elements while maintaining dramatic tilt.",

    "overhead_shot": "You are a cinematic camera operator. Maintain overhead shot positioned 90 degrees directly above subject. Camera pointing straight down. This creates unique flat perspective. Preserve subject details and compositional relationships.",

    "ground_level": "You are a cinematic camera operator. Maintain ground-level perspective with camera positioned very low to the ground. This creates intimate, immersive viewpoint. Preserve scene details and dramatic low angle perspective.",
}

CAMERA_VIEWS = {
    # Basic Orthographic Views
    "front_view": {
        "description": "Straight front view (elevation)",
        "prompt": "change the view to a front view of {subject}",
    },
    "back_view": {
        "description": "Rear view (back elevation)",
        "prompt": "change the view to a back view of {subject}",
    },
    "left_side_view": {
        "description": "Left side view (left elevation)",
        "prompt": "change the view to a left side view of {subject}",
    },
    "right_side_view": {
        "description": "Right side view (right elevation)",
        "prompt": "change the view to a right side view of {subject}",
    },
    "top_view": {
        "description": "Top-down view (plan view / bird's eye)",
        "prompt": "change the view to a top-down view of {subject}, bird's eye perspective",
    },
    "bottom_view": {
        "description": "Bottom-up view (worm's eye view)",
        "prompt": "change the view to an extreme bottom-up view of {subject}, worm's eye perspective",
    },

    # Isometric and 3/4 Views
    "three_quarter_view": {
        "description": "3/4 view showing front and side (most common product view)",
        "prompt": "change the view to a three-quarter view of {subject} showing both the front and side",
    },
    "isometric_view": {
        "description": "Isometric view (technical drawing style, no perspective)",
        "prompt": "change the view to an isometric view of {subject}",
    },

    # Portrait/Person Angles (from qwen-2.pdf research)
    "eye_level": {
        "description": "Eye level (neutral, natural perspective)",
        "prompt": "change the view to an eye-level shot of {subject}, with the camera positioned at the same height as the subject's eyes",
    },
    "high_angle": {
        "description": "High angle looking down (creates vulnerability, weakness)",
        "prompt": "change the view to a high-angle shot of {subject}, with the camera positioned above the subject and angled downward",
    },
    "low_angle": {
        "description": "Low angle looking up (creates power, dominance)",
        "prompt": "change the view to a low-angle shot of {subject}, with the camera positioned below the subject and angled upward",
    },
    "birds_eye": {
        "description": "Ultra-high bird's eye view (extreme top-down)",
        "prompt": "change the view to an ultra-high angle shot (bird's eye view) of {subject}, with the camera's point of view positioned very high up, directly looking down at the subject from above",
    },
    "worms_eye": {
        "description": "Ultra-low worm's eye view (extreme bottom-up)",
        "prompt": "change the view to an ultra-low angle shot (worm's eye view) of {subject}, with the camera's point of view positioned very low to the ground, directly looking up at the subject",
    },

    # Architectural Views
    "section_view": {
        "description": "Section cut view (as if building is sliced)",
        "prompt": "change the view to a section view of {subject}, showing the interior cross-section",
    },
    "aerial_view": {
        "description": "Aerial view (elevated perspective, 45° angle)",
        "prompt": "change the view to an aerial view of {subject} from elevated position at 45 degree angle",
    },
    "street_level": {
        "description": "Street level / human eye height view",
        "prompt": "change the view to a street-level view of {subject} at human eye height",
    },

    # Interior Design Views
    "corner_view": {
        "description": "Corner view (captures two walls meeting)",
        "prompt": "change the view to a corner view of {subject}, showing where two walls meet",
    },
    "entrance_view": {
        "description": "View from entrance/doorway looking in",
        "prompt": "change the view to an entrance view looking into {subject}",
    },
    "ceiling_view": {
        "description": "View looking up at ceiling",
        "prompt": "change the view and tilt the camera way up to show the ceiling of {subject}",
    },

    # Cinematic Views
    "dutch_angle": {
        "description": "Dutch angle (tilted horizon for dramatic effect)",
        "prompt": "change the view to a dutch angle of {subject}, with the camera tilted to create a diagonal horizon line for dramatic effect",
    },
    "overhead_shot": {
        "description": "Overhead shot (90° top-down, directly above)",
        "prompt": "change the view to an overhead shot directly above {subject}, camera pointing straight down",
    },
    "ground_level": {
        "description": "Ground level (camera on the ground)",
        "prompt": "change the view to ground level of {subject}, with the camera positioned very low to the ground",
    },
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def build_camera_view_prompt(
    view_name: str,
    subject: str,
    scene_context: str,
) -> str:
    """Build camera view prompt with subject substitution.

    Args:
        view_name: Selected camera view preset name
        subject: Subject to photograph (used in prompt)
        scene_context: Optional scene description

    Returns:
        Complete prompt string
    """
    view_data = CAMERA_VIEWS.get(view_name, CAMERA_VIEWS["front_view"])
    prompt_template = view_data["prompt"]

    # Substitute subject into template
    prompt = prompt_template.format(subject=subject if subject.strip() else "the scene")

    # Add scene context if provided
    if scene_context.strip():
        prompt = f"{scene_context.strip()}, {prompt}"

    return prompt


# ============================================================================
# NODE DEFINITION
# ============================================================================

class ArchAi3D_Qwen_Camera_View_Selector(io.ComfyNode):
    """Camera View Selector: Quick selection from professional viewpoints.

    This node provides instant access to 22 professional camera viewpoints:
    - 6 orthographic views (front, back, left, right, top, bottom)
    - 2 3/4 and isometric views
    - 5 portrait angles (eye level, high, low, bird's eye, worm's eye)
    - 4 architectural views (section, aerial, street level)
    - 3 interior design views (corner, entrance, ceiling)
    - 2 cinematic views (dutch angle, overhead, ground level)

    Perfect For:
    - Quick viewpoint changes without complex parameters
    - Standardized architectural views (elevations, sections, plans)
    - Professional portrait angles
    - Product photography (3/4 view is most common e-commerce angle)
    - Cinematic effects (dutch angle for drama)

    Based on research: Named views work more reliably than abstract camera
    movements. "Change to front view" works better than "rotate camera 180°".
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ArchAi3D_Qwen_Camera_View_Selector",
            category="ArchAi3d/Qwen/Camera",
            inputs=[
                # Group 1: View Selection
                io.Combo.Input(
                    "camera_view",
                    options=list(CAMERA_VIEWS.keys()),
                    default="front_view",
                    tooltip="Select from 22 professional camera viewpoints. Each view is optimized for specific photography needs.",
                ),

                # Group 2: Subject and Scene
                io.String.Input(
                    "subject",
                    default="",
                    tooltip="What to photograph. Example: 'the building', 'the subject', 'the product', 'the room'. Leave empty for 'the scene'.",
                ),
                io.String.Input(
                    "scene_context",
                    multiline=True,
                    default="",
                    tooltip="Optional scene description for consistency. Example: 'modern architectural exterior with glass facade'",
                ),

                # Group 3: Options
                io.Boolean.Input(
                    "debug_mode",
                    default=False,
                    tooltip="Print detailed prompt information to console",
                ),
            ],
            outputs=[
                io.String.Output(
                    "prompt",
                    tooltip="Generated prompt for Qwen Edit 2509",
                ),
                io.String.Output(
                    "view_description",
                    tooltip="Description of the selected camera view",
                ),
                io.String.Output(
                    "system_prompt",
                    tooltip="⭐ NEW v4.0: Perfect system prompt for this view! Connect to encoder's system_prompt input for optimal results.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        camera_view,
        subject,
        scene_context,
        debug_mode,
    ) -> io.NodeOutput:
        """Execute the Camera View Selector node.

        Steps:
        1. Get view data for selected viewpoint
        2. Build prompt with subject substitution
        3. Output debug info if requested
        4. Return prompt and description
        """

        # Step 1: Get view data
        view_data = CAMERA_VIEWS.get(camera_view, CAMERA_VIEWS["front_view"])
        view_description = view_data["description"]

        # Get system prompt for this view (v4.0.0 feature)
        system_prompt = CAMERA_VIEW_SYSTEM_PROMPTS.get(camera_view, CAMERA_VIEW_SYSTEM_PROMPTS["front_view"])

        # Step 2: Build prompt
        prompt = build_camera_view_prompt(
            view_name=camera_view,
            subject=subject,
            scene_context=scene_context,
        )

        # Step 3: Debug output
        if debug_mode:
            debug_lines = [
                "=" * 70,
                "ArchAi3D_Qwen_Camera_View_Selector - Generated Prompt (v4.0.0)",
                "=" * 70,
                f"Camera View: {camera_view}",
                f"Description: {view_description}",
                f"Subject: {subject if subject.strip() else 'the scene'}",
                f"Scene Context: {scene_context[:80]}..." if len(scene_context) > 80 else f"Scene Context: {scene_context}",
                "=" * 70,
                "Generated Prompt:",
                prompt,
                "=" * 70,
                "⭐ System Prompt (NEW v4.0.0):",
                system_prompt,
                "=" * 70,
            ]
            print("\n".join(debug_lines))

        # Step 4: Return (now includes system_prompt)
        return io.NodeOutput(prompt, view_description, system_prompt)


# ============================================================================
# EXTENSION REGISTRATION
# ============================================================================

class CameraViewSelectorExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [ArchAi3D_Qwen_Camera_View_Selector]


async def comfy_entrypoint():
    return CameraViewSelectorExtension()
