"""
ArchAi3D Qwen Material Changer Node
Author: Amir Ferdos (ArchAi3d)
Email: Amir84ferdos@gmail.com
LinkedIn: https://www.linkedin.com/in/archai3d/

Description:
    Change materials and textures for interior design visualization. Perfect for:
    - Kitchen design: Try different countertop materials (marble, granite, quartz)
    - Flooring variations: Wood, tile, carpet, concrete
    - Furniture options: Different upholstery fabrics and finishes
    - Wall treatments: Paint colors, wallpaper, paneling

Based on research: "change [object] material to [material], keep everything
else identical" pattern works best. Preservation clause is critical.
"""

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override
from typing import Dict, List

# ============================================================================
# MATERIAL SYSTEM PROMPTS - v4.0.0 Feature
# ============================================================================

MATERIAL_CATEGORY_SYSTEM_PROMPTS = {
    "stone": "You are an interior designer. Change specified object's material to natural or engineered stone finish. Apply stone texture, color, and veining patterns accurately. Preserve all other scene elements, objects, lighting, and composition exactly. Maintain realistic stone appearance and proper material properties.",

    "wood": "You are an interior designer. Change specified object's material to wood finish. Apply wood grain, color tone, and natural texture accurately. Preserve all other scene elements, objects, lighting, and composition exactly. Maintain realistic wood appearance with proper grain direction and finish quality.",

    "metal": "You are an interior designer. Change specified object's material to metal finish. Apply metal surface properties, reflectivity, and finish accurately. Preserve all other scene elements, objects, lighting, and composition exactly. Maintain realistic metal appearance with appropriate shine, brushing, or patina effects.",

    "fabric": "You are an interior designer. Change specified object's material to fabric or textile. Apply fabric texture, weave pattern, and color accurately. Preserve all other scene elements, objects, lighting, and composition exactly. Maintain realistic fabric drape, texture, and appropriate material characteristics.",

    "paint": "You are an interior designer. Change specified object's surface to painted finish. Apply paint color and finish accurately with appropriate coverage. Preserve all other scene elements, objects, lighting, and composition exactly. Maintain realistic paint appearance with proper color tone and smooth or textured finish.",

    "tile": "You are an interior designer. Change specified object's material to tile finish. Apply tile pattern, grout lines, color, and surface accurately. Preserve all other scene elements, objects, lighting, and composition exactly. Maintain realistic tile appearance with proper layout, grout color, and surface finish.",

    "custom": "You are an interior designer. Change specified object's material as described. Apply material properties, texture, and appearance accurately. Preserve all other scene elements, objects, lighting, and composition exactly. Maintain realistic material characteristics and appropriate visual qualities.",
}

# ============================================================================
# MATERIAL PRESETS - Common interior design materials
# ============================================================================

MATERIAL_CATEGORIES = {
    "stone": {
        "description": "Natural and engineered stone",
        "materials": [
            "white Carrara marble with gray veining",
            "black granite with subtle sparkle",
            "polished concrete with light gray tone",
            "slate gray stone",
            "travertine with natural holes and patterns",
            "white quartz with marble-like pattern",
            "soapstone with dark charcoal color",
            "limestone with beige tones",
        ],
    },
    "wood": {
        "description": "Wood types and finishes",
        "materials": [
            "light oak hardwood",
            "dark walnut with rich grain",
            "butcher block oak",
            "reclaimed barn wood with weathered finish",
            "maple with natural blonde color",
            "cherry wood with reddish-brown tone",
            "mahogany with deep brown color",
            "bamboo with light natural finish",
        ],
    },
    "metal": {
        "description": "Metal finishes",
        "materials": [
            "brushed stainless steel",
            "polished brass with golden finish",
            "matte black steel",
            "copper with warm patina",
            "chrome with mirror finish",
            "brushed bronze",
            "aged iron with dark finish",
            "brushed nickel",
        ],
    },
    "fabric": {
        "description": "Textiles and upholstery",
        "materials": [
            "navy blue velvet",
            "light gray linen",
            "charcoal wool",
            "cream cotton",
            "leather with brown patina",
            "black leather with subtle sheen",
            "beige microfiber",
            "emerald green velvet",
        ],
    },
    "paint": {
        "description": "Wall paint colors",
        "materials": [
            "sage green wall paint",
            "warm white paint",
            "light gray paint",
            "navy blue accent wall",
            "blush pink paint",
            "charcoal gray paint",
            "cream beige paint",
            "soft blue-gray paint",
        ],
    },
    "tile": {
        "description": "Ceramic and porcelain tile",
        "materials": [
            "white subway tile with gray grout",
            "black hexagon tile",
            "wood-look porcelain tile",
            "mosaic glass tile with iridescent finish",
            "large format gray tile",
            "terracotta clay tile",
            "patterned cement tile",
            "white marble-look tile",
        ],
    },
    "custom": {
        "description": "Custom material description",
        "materials": [],
    },
}

# Common objects that can be changed
COMMON_OBJECTS = [
    "the kitchen countertop",
    "the flooring",
    "the wall",
    "the accent wall",
    "the backsplash",
    "the cabinets",
    "the sofa upholstery",
    "the chair fabric",
    "the table surface",
    "the island countertop",
    "the bathroom vanity",
    "the shower tile",
    "the ceiling",
    "the trim and molding",
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def build_material_change_prompt(
    object_to_change: str,
    material_description: str,
    scene_context: str,
    preserve_rest: bool,
) -> str:
    """Build material change prompt.

    Formula from research:
    [SCENE_CONTEXT], change [OBJECT] material to [MATERIAL], keep everything else identical

    Args:
        object_to_change: The object to modify
        material_description: New material description
        scene_context: Optional scene description
        preserve_rest: Whether to preserve everything else

    Returns:
        Complete prompt string
    """
    parts = []

    # Scene context
    if scene_context.strip():
        parts.append(scene_context.strip())

    # Material change instruction
    change_phrase = f"change {object_to_change} material to {material_description}"

    # Preservation clause (critical for maintaining scene consistency)
    if preserve_rest:
        change_phrase += ", keep everything else identical"

    parts.append(change_phrase)

    return ", ".join(parts)


# ============================================================================
# NODE DEFINITION
# ============================================================================

class ArchAi3D_Qwen_Material_Changer(io.ComfyNode):
    """Material Changer: Interior design material visualization.

    This node enables quick material and texture exploration for interior design:
    - 6 material categories with 8 presets each (48 total presets)
    - Stone: Marble, granite, quartz, concrete, slate, travertine, etc.
    - Wood: Oak, walnut, maple, cherry, reclaimed, bamboo, etc.
    - Metal: Stainless steel, brass, bronze, copper, chrome, etc.
    - Fabric: Velvet, linen, wool, leather, cotton, microfiber, etc.
    - Paint: Sage green, navy, gray, beige, blush, charcoal, etc.
    - Tile: Subway, hexagon, wood-look, mosaic, cement, etc.

    Key Features:
    - 48 material presets across 6 categories
    - Custom material description option
    - 14 common objects quick-select
    - Automatic preservation clause (keeps everything else identical)
    - Scene context support for consistency

    Perfect For:
    - Kitchen design: Countertops, backsplash, cabinets, flooring
    - Living room: Furniture upholstery, wall colors, flooring
    - Bathroom: Vanity, tile, flooring
    - Client presentations: Show multiple material options quickly

    Based on research: Preservation clause "keep everything else identical"
    is CRITICAL for maintaining scene consistency during material changes.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ArchAi3D_Qwen_Material_Changer",
            category="ArchAi3d/Qwen/Editing",
            inputs=[
                # Group 1: Object Selection
                io.Combo.Input(
                    "object_preset",
                    options=["custom"] + COMMON_OBJECTS,
                    default="custom",
                    tooltip="Select common object or use 'custom' to specify your own. Example: 'the dining table', 'the bedroom walls'",
                ),
                io.String.Input(
                    "custom_object",
                    default="",
                    tooltip="Custom object description (used if object_preset is 'custom'). Example: 'the dining room table surface'",
                ),

                # Group 2: Material Selection
                io.Combo.Input(
                    "material_category",
                    options=list(MATERIAL_CATEGORIES.keys()),
                    default="stone",
                    tooltip="Material category: stone, wood, metal, fabric, paint, tile, or custom",
                ),
                io.Combo.Input(
                    "material_preset",
                    options=["custom"],  # Will be dynamically populated based on category
                    default="custom",
                    tooltip="Select material preset from category or use 'custom' for your own description",
                ),
                io.String.Input(
                    "custom_material",
                    default="white Carrara marble with gray veining",
                    tooltip="Custom material description. Be specific about color, texture, pattern, finish. Example: 'dark walnut with rich grain and matte finish'",
                ),

                # Group 3: Scene Context
                io.String.Input(
                    "scene_context",
                    multiline=True,
                    default="",
                    tooltip="Optional scene description for consistency. Example: 'modern kitchen with white cabinets and stainless appliances'",
                ),

                # Group 4: Options
                io.Boolean.Input(
                    "preserve_rest",
                    default=True,
                    tooltip="Add 'keep everything else identical' clause (recommended for consistency)",
                ),
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
                    "material_description",
                    tooltip="The material description being applied",
                ),
                io.String.Output(
                    "system_prompt",
                    tooltip="⭐ NEW v4.0: Perfect system prompt for this material category! Connect to encoder's system_prompt input for optimal results.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        object_preset,
        custom_object,
        material_category,
        material_preset,
        custom_material,
        scene_context,
        preserve_rest,
        debug_mode,
    ) -> io.NodeOutput:
        """Execute the Material Changer node.

        Steps:
        1. Determine object to change (preset or custom)
        2. Determine material description (preset or custom)
        3. Build material change prompt
        4. Output debug info if requested
        5. Return prompt and material description
        """

        # Step 1: Determine object
        if object_preset == "custom":
            object_to_change = custom_object.strip() if custom_object.strip() else "the object"
        else:
            object_to_change = object_preset

        # Step 2: Determine material
        # Note: In actual implementation, material_preset options would be
        # dynamically populated based on material_category selection.
        # For this version, we'll use custom_material if preset is "custom"
        if material_preset == "custom" or material_category == "custom":
            material_description = custom_material.strip()
        else:
            # Get material from category presets
            category_materials = MATERIAL_CATEGORIES.get(material_category, {}).get("materials", [])
            if material_preset and material_preset != "custom" and material_preset in category_materials:
                material_description = material_preset
            elif category_materials:
                material_description = category_materials[0]  # Default to first in category
            else:
                material_description = custom_material.strip()

        # Get system prompt for this material category (v4.0.0 feature)
        system_prompt = MATERIAL_CATEGORY_SYSTEM_PROMPTS.get(material_category, MATERIAL_CATEGORY_SYSTEM_PROMPTS["custom"])

        # Step 3: Build prompt
        prompt = build_material_change_prompt(
            object_to_change=object_to_change,
            material_description=material_description,
            scene_context=scene_context,
            preserve_rest=preserve_rest,
        )

        # Step 4: Debug output
        if debug_mode:
            debug_lines = [
                "=" * 70,
                "ArchAi3D_Qwen_Material_Changer - Generated Prompt (v4.0.0)",
                "=" * 70,
                f"Object: {object_to_change}",
                f"Material Category: {material_category}",
                f"Material: {material_description}",
                f"Scene Context: {scene_context[:80]}..." if len(scene_context) > 80 else f"Scene Context: {scene_context}",
                f"Preserve Rest: {preserve_rest}",
                "=" * 70,
                "Generated Prompt:",
                prompt,
                "=" * 70,
                "⭐ System Prompt (NEW v4.0.0):",
                system_prompt,
                "=" * 70,
            ]
            print("\n".join(debug_lines))

        # Step 5: Return (now includes system_prompt)
        return io.NodeOutput(prompt, material_description, system_prompt)


# ============================================================================
# EXTENSION REGISTRATION
# ============================================================================

class MaterialChangerExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [ArchAi3D_Qwen_Material_Changer]


async def comfy_entrypoint():
    return MaterialChangerExtension()
