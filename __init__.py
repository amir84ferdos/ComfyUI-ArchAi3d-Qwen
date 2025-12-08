"""
ComfyUI-ArchAi3d-Qwen
Advanced Qwen-VL nodes for ComfyUI with organized submenus

Author: Amir Ferdos (ArchAi3d)
Email: Amir84ferdos@gmail.com
LinkedIn: https://www.linkedin.com/in/archai3d/
GitHub: https://github.com/amir84ferdos
Version: 3.0.0
License: Dual License (Free for personal use, Commercial license required for business use)
"""

# ============================================================================
# CORE ENCODING NODES
# ============================================================================

from .nodes.core.encoders.archai3d_qwen_encoder import ArchAi3D_Qwen_Encoder
from .nodes.core.encoders.archai3d_qwen_encoder_v2 import ArchAi3D_Qwen_Encoder_V2
from .nodes.core.encoders.archai3d_qwen_encoder_simple import ArchAi3D_Qwen_Encoder_Simple
from .nodes.core.encoders.archai3d_qwen_encoder_simple_v2 import ArchAi3dQwenEncoderSimpleV2
from .nodes.core.encoders.archai3d_qwen_encoder_v3 import ArchAi3D_Qwen_Encoder_V3

from .nodes.core.utils.archai3d_qwen_image_scale import ArchAi3D_Qwen_Image_Scale

from .nodes.core.prompts.archai3d_clean_room_prompt import ArchAi3D_Clean_Room_Prompt
from .nodes.core.prompts.archai3d_qwen_system_prompt import ArchAi3D_Qwen_System_Prompt

# ============================================================================
# CAMERA CONTROL NODES
# ============================================================================

from .nodes.camera.archai3d_qwen_camera_view_selector import ArchAi3D_Qwen_Camera_View_Selector
from .nodes.camera.archai3d_qwen_object_rotation_v2 import ArchAi3D_Qwen_Object_Rotation_V2
from .nodes.camera.archai3d_qwen_environment_navigator import ArchAi3D_Qwen_Environment_Navigator
from .nodes.camera.archai3d_qwen_person_perspective import ArchAi3D_Qwen_Person_Perspective
from .nodes.camera.archai3d_qwen_scene_photographer import ArchAi3D_Qwen_Scene_Photographer

# v5.0.0 NEW EXTERIOR CATEGORY NODES
from .nodes.camera.archai3d_qwen_exterior_view_control import ArchAi3D_Qwen_Exterior_View_Control
from .nodes.camera.archai3d_qwen_exterior_navigation import ArchAi3D_Qwen_Exterior_Navigation
from .nodes.camera.archai3d_qwen_exterior_focus import ArchAi3D_Qwen_Exterior_Focus

# v5.0.0 NEW INTERIOR CATEGORY NODES
from .nodes.camera.archai3d_qwen_interior_view_control import ArchAi3D_Qwen_Interior_View_Control
from .nodes.camera.archai3d_qwen_interior_navigation import ArchAi3D_Qwen_Interior_Navigation
from .nodes.camera.archai3d_qwen_interior_focus import ArchAi3D_Qwen_Interior_Focus

# v5.0.0 NEW OBJECT CATEGORY NODES (WEEK 3)
from .nodes.camera.archai3d_qwen_object_view_control import ArchAi3D_Qwen_Object_View_Control
from .nodes.camera.archai3d_qwen_object_position_control import ArchAi3D_Qwen_Object_Position_Control
from .nodes.camera.archai3d_qwen_object_rotation_control import ArchAi3D_Qwen_Object_Rotation_Control

# v5.0.0 NEW PERSON CATEGORY NODES (WEEK 4)
from .nodes.camera.archai3d_qwen_person_view_control import ArchAi3D_Qwen_Person_View_Control
from .nodes.camera.archai3d_qwen_person_position_control import ArchAi3D_Qwen_Person_Position_Control
from .nodes.camera.archai3d_qwen_person_perspective_control import ArchAi3D_Qwen_Person_Perspective_Control
from .nodes.camera.archai3d_qwen_person_cinematographer import ArchAi3D_Qwen_Person_Cinematographer

# v5.1.0 SIMPLE CAMERA CONTROL (Unified)
from .nodes.camera.simple_camera_control import ArchAi3D_Qwen_Simple_Camera_Control

# v5.1.0 DX8152 LORA SUPPORT
from .nodes.camera.dx8152_camera_lora import ArchAi3D_Qwen_DX8152_Camera_LoRA

# v5.1.0 OBJECT FOCUS CAMERA (v1 - Chinese prompts with LoRA mode)
from .nodes.camera.object_focus_camera import ArchAi3D_Object_Focus_Camera

# v5.1.0 OBJECT FOCUS CAMERA V2 (Reddit-validated English prompts)
from .nodes.camera.object_focus_camera_v2 import ArchAi3D_Object_Focus_Camera_V2

# v5.1.0 OBJECT FOCUS CAMERA V3 (Ultimate merged - Chinese/English/Hybrid)
from .nodes.camera.object_focus_camera_v3 import ArchAi3D_Object_Focus_Camera_V3

# v5.1.0 OBJECT FOCUS CAMERA V4 (Enhanced - Distance-aware + Environmental Focus)
from .nodes.camera.object_focus_camera_v4 import ArchAi3D_Object_Focus_Camera_V4

# v5.1.0 OBJECT FOCUS CAMERA V5 (Professional Presets - Material + Quality)
from .nodes.camera.object_focus_camera_v5 import ArchAi3D_Object_Focus_Camera_V5

# v5.1.0 OBJECT FOCUS CAMERA V6 (Ultimate - Vantage Point + Presets)
from .nodes.camera.object_focus_camera_v6 import ArchAi3D_Object_Focus_Camera_V6

# v7.0.0 OBJECT FOCUS CAMERA V7 (Professional Cinematography Edition)
from .nodes.camera.object_focus_camera_v7 import ArchAi3D_Object_Focus_Camera_V7

# CINEMATOGRAPHY PROMPT BUILDER (Nanobanan's 5-Ingredient Formula)
from .nodes.camera.cinematography_prompt_builder import ArchAi3D_Cinematography_Prompt_Builder

# ============================================================================
# IMAGE EDITING NODES
# ============================================================================

from .nodes.editing.archai3d_qwen_material_changer import ArchAi3D_Qwen_Material_Changer
from .nodes.editing.archai3d_qwen_watermark_removal import ArchAi3D_Qwen_Watermark_Removal
from .nodes.editing.archai3d_qwen_colorization import ArchAi3D_Qwen_Colorization
from .nodes.editing.archai3d_qwen_style_transfer import ArchAi3D_Qwen_Style_Transfer

# ============================================================================
# UTILITY NODES
# ============================================================================

from .nodes.utils.archai3d_mask_to_position_guide import ArchAi3D_Mask_To_Position_Guide
from .nodes.utils.archai3d_smart_tile_calculator import ArchAi3D_Smart_Tile_Calculator
from .nodes.utils.archai3d_smart_tile_prompter import ArchAi3D_Smart_Tile_Prompter
from .nodes.utils.archai3d_smart_tile_segs import ArchAi3D_Smart_Tile_SEGS
from .nodes.utils.archai3d_smart_tile_conditioning import ArchAi3D_Smart_Tile_Conditioning
from .nodes.utils.archai3d_smart_tile_detailer import ArchAi3D_Smart_Tile_Detailer
from .nodes.utils.archai3d_info import ArchAi3D_Info

# ============================================================================
# INPUT NODES (Web Interface Integration)
# ============================================================================

from .nodes.inputs.archai3d_string_input import ArchAi3D_String_Input
from .nodes.inputs.archai3d_int_input import ArchAi3D_Int_Input
from .nodes.inputs.archai3d_float_input import ArchAi3D_Float_Input
from .nodes.inputs.archai3d_boolean_input import ArchAi3D_Boolean_Input
from .nodes.inputs.archai3d_load_image_url import ArchAi3D_Load_Image_URL
from .nodes.inputs.archai3d_save_image import ArchAi3D_Save_Image
from .nodes.inputs.archai3d_conditioning_balance import ArchAi3D_Conditioning_Balance
from .nodes.inputs.archai3d_gemini_model import ArchAi3D_Gemini_Model
from .nodes.inputs.archai3d_gemini import ArchAi3D_Gemini
from .nodes.inputs.archai3d_qwenvl_gguf import ArchAi3D_QwenVL_GGUF, ArchAi3D_QwenVL_Server_Control
from .nodes.inputs.archai3d_llama_cpp_installer import ArchAi3D_LlamaCpp_Installer
from .nodes.inputs.archai3d_hf_download import ArchAi3D_HF_Download
from .nodes.inputs.archai3d_civitai_download import ArchAi3D_CivitAI_Download
from .nodes.inputs.archai3d_gdrive_download import ArchAi3D_GDrive_Download
from .nodes.utils.archai3d_position_guide_prompt_builder import ArchAi3D_Position_Guide_Prompt_Builder
from .nodes.utils.archai3d_simple_position_prompt import ArchAi3D_Simple_Position_Prompt
from .nodes.utils.archai3d_color_correction_bt709 import ArchAi3D_Color_Correction_BT709
from .nodes.utils.archai3d_color_correction_advanced import ArchAi3D_Color_Correction_Advanced
from .nodes.utils.archai3d_average_color import ArchAi3D_Average_Color
from .nodes.utils.archai3d_solid_color_image import ArchAi3D_Solid_Color_Image
from .nodes.utils.archai3d_mask_crop_rotate import ArchAi3D_Mask_Crop_Rotate
from .nodes.utils.archai3d_panorama_offset import ArchAi3D_Panorama_Offset
from .nodes.utils.nunchaku_installer import ArchAi3D_Nunchaku_Installer
from .nodes.utils.dependency_installer import ArchAi3D_Dependency_Installer

# ============================================================================
# OPTIMIZED LOW VRAM NODES
# ============================================================================

from .nodes.utils.archai3d_sam3_segment import ArchAi3D_SAM3_Segment
from .nodes.utils.archai3d_metric3d_normal import ArchAi3D_Metric3D_Normal, ArchAi3D_Metric3D_Depth

# ============================================================================
# NODE CLASS MAPPINGS
# Organized with submenu structure for ComfyUI
# ============================================================================

NODE_CLASS_MAPPINGS = {
    # Core - Encoders
    "ArchAi3D_Qwen_Encoder": ArchAi3D_Qwen_Encoder,
    "ArchAi3D_Qwen_Encoder_V2": ArchAi3D_Qwen_Encoder_V2,
    "ArchAi3D_Qwen_Encoder_Simple": ArchAi3D_Qwen_Encoder_Simple,
    "ArchAi3dQwenEncoderSimpleV2": ArchAi3dQwenEncoderSimpleV2,
    "ArchAi3D_Qwen_Encoder_V3": ArchAi3D_Qwen_Encoder_V3,

    # Core - Utils
    "ArchAi3D_Qwen_Image_Scale": ArchAi3D_Qwen_Image_Scale,
    "ArchAi3D_Qwen_System_Prompt": ArchAi3D_Qwen_System_Prompt,

    # Core - Prompts
    "ArchAi3D_Clean_Room_Prompt": ArchAi3D_Clean_Room_Prompt,

    # Camera Control (Legacy)
    "ArchAi3D_Qwen_Camera_View_Selector": ArchAi3D_Qwen_Camera_View_Selector,
    "ArchAi3D_Qwen_Object_Rotation_V2": ArchAi3D_Qwen_Object_Rotation_V2,
    "ArchAi3D_Qwen_Environment_Navigator": ArchAi3D_Qwen_Environment_Navigator,
    "ArchAi3D_Qwen_Person_Perspective": ArchAi3D_Qwen_Person_Perspective,
    "ArchAi3D_Qwen_Scene_Photographer": ArchAi3D_Qwen_Scene_Photographer,

    # v5.0.0 NEW Camera Control - Exterior
    "ArchAi3D_Qwen_Exterior_View_Control": ArchAi3D_Qwen_Exterior_View_Control,
    "ArchAi3D_Qwen_Exterior_Navigation": ArchAi3D_Qwen_Exterior_Navigation,
    "ArchAi3D_Qwen_Exterior_Focus": ArchAi3D_Qwen_Exterior_Focus,

    # v5.0.0 NEW Camera Control - Interior
    "ArchAi3D_Qwen_Interior_View_Control": ArchAi3D_Qwen_Interior_View_Control,
    "ArchAi3D_Qwen_Interior_Navigation": ArchAi3D_Qwen_Interior_Navigation,
    "ArchAi3D_Qwen_Interior_Focus": ArchAi3D_Qwen_Interior_Focus,

    # v5.0.0 NEW Camera Control - Object (WEEK 3)
    "ArchAi3D_Qwen_Object_View_Control": ArchAi3D_Qwen_Object_View_Control,
    "ArchAi3D_Qwen_Object_Position_Control": ArchAi3D_Qwen_Object_Position_Control,
    "ArchAi3D_Qwen_Object_Rotation_Control": ArchAi3D_Qwen_Object_Rotation_Control,

    # v5.0.0 NEW Camera Control - Person (WEEK 4)
    "ArchAi3D_Qwen_Person_View_Control": ArchAi3D_Qwen_Person_View_Control,
    "ArchAi3D_Qwen_Person_Position_Control": ArchAi3D_Qwen_Person_Position_Control,
    "ArchAi3D_Qwen_Person_Perspective_Control": ArchAi3D_Qwen_Person_Perspective_Control,
    "ArchAi3D_Qwen_Person_Cinematographer": ArchAi3D_Qwen_Person_Cinematographer,

    # v5.1.0 Simple Camera Control
    "ArchAi3D_Qwen_Simple_Camera_Control": ArchAi3D_Qwen_Simple_Camera_Control,

    # v5.1.0 dx8152 LoRA Support
    "ArchAi3D_Qwen_DX8152_Camera_LoRA": ArchAi3D_Qwen_DX8152_Camera_LoRA,

    # v5.1.0 Object Focus Camera (v1 - Chinese prompts)
    "ArchAi3D_Object_Focus_Camera": ArchAi3D_Object_Focus_Camera,

    # v5.1.0 Object Focus Camera v2 (Reddit-validated English prompts)
    "ArchAi3D_Object_Focus_Camera_V2": ArchAi3D_Object_Focus_Camera_V2,

    # v5.1.0 Object Focus Camera v3 (Ultimate merged)
    "ArchAi3D_Object_Focus_Camera_V3": ArchAi3D_Object_Focus_Camera_V3,

    # v5.1.0 Object Focus Camera v4 (Enhanced - Distance-aware + Environmental Focus)
    "ArchAi3D_Object_Focus_Camera_V4": ArchAi3D_Object_Focus_Camera_V4,

    # v5.1.0 Object Focus Camera v5 (Professional Presets - Material + Quality)
    "ArchAi3D_Object_Focus_Camera_V5": ArchAi3D_Object_Focus_Camera_V5,

    # v5.1.0 Object Focus Camera v6 (Ultimate - Vantage Point + Presets)
    "ArchAi3D_Object_Focus_Camera_V6": ArchAi3D_Object_Focus_Camera_V6,

    # v7.0.0 Object Focus Camera v7 (Professional Cinematography Edition)
    "ArchAi3D_Object_Focus_Camera_V7": ArchAi3D_Object_Focus_Camera_V7,

    # Cinematography Prompt Builder (Nanobanan's 5-Ingredient Formula)
    "ArchAi3D_Cinematography_Prompt_Builder": ArchAi3D_Cinematography_Prompt_Builder,

    # Image Editing
    "ArchAi3D_Qwen_Material_Changer": ArchAi3D_Qwen_Material_Changer,
    "ArchAi3D_Qwen_Watermark_Removal": ArchAi3D_Qwen_Watermark_Removal,
    "ArchAi3D_Qwen_Colorization": ArchAi3D_Qwen_Colorization,
    "ArchAi3D_Qwen_Style_Transfer": ArchAi3D_Qwen_Style_Transfer,

    # Utils
    "ArchAi3D_Mask_To_Position_Guide": ArchAi3D_Mask_To_Position_Guide,
    "ArchAi3D_Smart_Tile_Calculator": ArchAi3D_Smart_Tile_Calculator,
    "ArchAi3D_Smart_Tile_Prompter": ArchAi3D_Smart_Tile_Prompter,
    "ArchAi3D_Smart_Tile_SEGS": ArchAi3D_Smart_Tile_SEGS,
    "ArchAi3D_Smart_Tile_Conditioning": ArchAi3D_Smart_Tile_Conditioning,
    "ArchAi3D_Smart_Tile_Detailer": ArchAi3D_Smart_Tile_Detailer,
    "ArchAi3D_Info": ArchAi3D_Info,
    "ArchAi3D_Position_Guide_Prompt_Builder": ArchAi3D_Position_Guide_Prompt_Builder,
    "ArchAi3D_Simple_Position_Prompt": ArchAi3D_Simple_Position_Prompt,
    "ArchAi3D_Color_Correction_BT709": ArchAi3D_Color_Correction_BT709,
    "ArchAi3D_Color_Correction_Advanced": ArchAi3D_Color_Correction_Advanced,
    "ArchAi3D_Average_Color": ArchAi3D_Average_Color,
    "ArchAi3D_Solid_Color_Image": ArchAi3D_Solid_Color_Image,
    "ArchAi3D_Mask_Crop_Rotate": ArchAi3D_Mask_Crop_Rotate,
    "ArchAi3D_Panorama_Offset": ArchAi3D_Panorama_Offset,

    # Input Nodes (Web Interface Integration)
    "ArchAi3D_String_Input": ArchAi3D_String_Input,
    "ArchAi3D_Int_Input": ArchAi3D_Int_Input,
    "ArchAi3D_Float_Input": ArchAi3D_Float_Input,
    "ArchAi3D_Boolean_Input": ArchAi3D_Boolean_Input,
    "ArchAi3D_Load_Image_URL": ArchAi3D_Load_Image_URL,
    "ArchAi3D_Save_Image": ArchAi3D_Save_Image,
    "ArchAi3D_Conditioning_Balance": ArchAi3D_Conditioning_Balance,
    "ArchAi3D_Gemini_Model": ArchAi3D_Gemini_Model,
    "ArchAi3D_Gemini": ArchAi3D_Gemini,
    "ArchAi3D_QwenVL_GGUF": ArchAi3D_QwenVL_GGUF,
    "ArchAi3D_QwenVL_Server_Control": ArchAi3D_QwenVL_Server_Control,
    "ArchAi3D_LlamaCpp_Installer": ArchAi3D_LlamaCpp_Installer,

    # Download Nodes
    "ArchAi3D_HF_Download": ArchAi3D_HF_Download,
    "ArchAi3D_CivitAI_Download": ArchAi3D_CivitAI_Download,
    "ArchAi3D_GDrive_Download": ArchAi3D_GDrive_Download,

    # Low VRAM Optimized Nodes
    "ArchAi3D_SAM3_Segment": ArchAi3D_SAM3_Segment,
    "ArchAi3D_Metric3D_Normal": ArchAi3D_Metric3D_Normal,
    "ArchAi3D_Metric3D_Depth": ArchAi3D_Metric3D_Depth,

    # Installer Nodes
    "ArchAi3D_Nunchaku_Installer": ArchAi3D_Nunchaku_Installer,
    "ArchAi3D_Dependency_Installer": ArchAi3D_Dependency_Installer,
}

# ============================================================================
# DISPLAY NAMES
# Organized with emoji prefixes for visual clarity
# ============================================================================

NODE_DISPLAY_NAME_MAPPINGS = {
    # Core - Encoders
    "ArchAi3D_Qwen_Encoder": "🎨 Qwen Encoder",
    "ArchAi3D_Qwen_Encoder_V2": "🎨 Qwen Encoder V2",
    "ArchAi3D_Qwen_Encoder_Simple": "🎨 Qwen Encoder Simple",
    "ArchAi3dQwenEncoderSimpleV2": "🎨 Qwen Encoder Simple V2",
    "ArchAi3D_Qwen_Encoder_V3": "⭐ Qwen Encoder V3 (Preset Balance + CFG)",

    # Core - Utils
    "ArchAi3D_Qwen_Image_Scale": "📏 Qwen Image Scale",
    "ArchAi3D_Qwen_System_Prompt": "💬 Qwen System Prompt",

    # Core - Prompts
    "ArchAi3D_Clean_Room_Prompt": "🏗️ Clean Room Prompt",

    # Camera Control (Legacy)
    "ArchAi3D_Qwen_Camera_View_Selector": "🎬 Camera View Selector",
    "ArchAi3D_Qwen_Object_Rotation_V2": "🔄 Object Rotation V2",
    "ArchAi3D_Qwen_Environment_Navigator": "🚶 Environment Navigator",
    "ArchAi3D_Qwen_Person_Perspective": "👤 Person Perspective",
    "ArchAi3D_Qwen_Scene_Photographer": "📸 Scene Photographer",

    # v5.0.0 NEW Camera Control - Exterior
    "ArchAi3D_Qwen_Exterior_View_Control": "🏢 Exterior View Control",
    "ArchAi3D_Qwen_Exterior_Navigation": "🏢 Exterior Navigation",
    "ArchAi3D_Qwen_Exterior_Focus": "🏢 Exterior Focus",

    # v5.0.0 NEW Camera Control - Interior
    "ArchAi3D_Qwen_Interior_View_Control": "🏠 Interior View Control",
    "ArchAi3D_Qwen_Interior_Navigation": "🏠 Interior Navigation",
    "ArchAi3D_Qwen_Interior_Focus": "🏠 Interior Focus",

    # v5.0.0 NEW Camera Control - Object (WEEK 3)
    "ArchAi3D_Qwen_Object_View_Control": "📦 Object View Control",
    "ArchAi3D_Qwen_Object_Position_Control": "📦 Object Position Control",
    "ArchAi3D_Qwen_Object_Rotation_Control": "📦 Object Rotation Control",

    # v5.0.0 NEW Camera Control - Person (WEEK 4)
    "ArchAi3D_Qwen_Person_View_Control": "👤 Person View Control",
    "ArchAi3D_Qwen_Person_Position_Control": "👤 Person Position Control",
    "ArchAi3D_Qwen_Person_Perspective_Control": "👤 Person Perspective Control",
    "ArchAi3D_Qwen_Person_Cinematographer": "🎬 Person Cinematographer",

    # v5.1.0 Simple Camera Control (v3.0 - Context-Aware)
    "ArchAi3D_Qwen_Simple_Camera_Control": "🎥 Simple Camera Control v3",

    # v5.1.0 dx8152 LoRA Support
    "ArchAi3D_Qwen_DX8152_Camera_LoRA": "📹 dx8152 Camera LoRA",

    # v5.1.0 Object Focus Camera (v1 - Chinese prompts)
    "ArchAi3D_Object_Focus_Camera": "📦 Object Focus Camera",

    # v5.1.0 Object Focus Camera v2 (Reddit-validated prompts)
    "ArchAi3D_Object_Focus_Camera_V2": "📦 Object Focus Camera v2 (Reddit)",

    # v5.1.0 Object Focus Camera v3 (Ultimate merged)
    "ArchAi3D_Object_Focus_Camera_V3": "📦 Object Focus Camera v3 (Ultimate)",

    # v5.1.0 Object Focus Camera v4 (Enhanced)
    "ArchAi3D_Object_Focus_Camera_V4": "📦 Object Focus Camera v4 (Enhanced)",

    # v5.1.0 Object Focus Camera v5 (Professional Presets)
    "ArchAi3D_Object_Focus_Camera_V5": "📦 Object Focus Camera v5 (Professional Presets)",

    # v5.1.0 Object Focus Camera v6 (Ultimate)
    "ArchAi3D_Object_Focus_Camera_V6": "📦 Object Focus Camera v6 (Ultimate)",

    # v7.0.0 Object Focus Camera v7 (Professional Cinematography)
    "ArchAi3D_Object_Focus_Camera_V7": "🎬 Object Focus Camera v7 (Pro Cinema)",

    # v8.0.0 Cinematography Prompt Builder (Nanobanan's 5 Ingredients)
    "ArchAi3D_Cinematography_Prompt_Builder": "📸 Cinematography Prompt Builder",

    # Image Editing
    "ArchAi3D_Qwen_Material_Changer": "🎨 Material Changer",
    "ArchAi3D_Qwen_Watermark_Removal": "🧹 Watermark Removal",
    "ArchAi3D_Qwen_Colorization": "🌈 Colorization",
    "ArchAi3D_Qwen_Style_Transfer": "✨ Style Transfer",

    # Utils
    "ArchAi3D_Mask_To_Position_Guide": "🎯 Mask to Position Guide",
    "ArchAi3D_Smart_Tile_Calculator": "🧮 Smart Tile Calculator",
    "ArchAi3D_Smart_Tile_Prompter": "🧩 Smart Tile Prompter",
    "ArchAi3D_Smart_Tile_SEGS": "🧱 Smart Tile SEGS",
    "ArchAi3D_Smart_Tile_Conditioning": "🎯 Smart Tile Conditioning",
    "ArchAi3D_Smart_Tile_Detailer": "🔧 Smart Tile Detailer",
    "ArchAi3D_Info": "ℹ️ ArchAi3D Info",
    "ArchAi3D_Position_Guide_Prompt_Builder": "📝 Position Guide Prompt Builder",
    "ArchAi3D_Simple_Position_Prompt": "📝 Simple Position Prompt",
    "ArchAi3D_Color_Correction_BT709": "🎨 Color Correction BT.709",
    "ArchAi3D_Color_Correction_Advanced": "✨ Color Correction Advanced",
    "ArchAi3D_Average_Color": "🎨 Average Color",
    "ArchAi3D_Solid_Color_Image": "🎨 Solid Color Image",
    "ArchAi3D_Mask_Crop_Rotate": "✂️ Mask Crop & Rotate",
    "ArchAi3D_Panorama_Offset": "🔄 Panorama Offset",

    # Input Nodes (Web Interface Integration)
    "ArchAi3D_String_Input": "📝 String Input",
    "ArchAi3D_Int_Input": "🔢 Integer Input",
    "ArchAi3D_Float_Input": "🔢 Float Input",
    "ArchAi3D_Boolean_Input": "✅ Boolean Input",
    "ArchAi3D_Load_Image_URL": "🌐 Load Image From URL",
    "ArchAi3D_Save_Image": "💾 Save Image",
    "ArchAi3D_Conditioning_Balance": "⚖️ Conditioning Balance",
    "ArchAi3D_Gemini_Model": "🤖 Gemini Model Selector",
    "ArchAi3D_Gemini": "🤖 Gemini API",
    "ArchAi3D_QwenVL_GGUF": "🚀 QwenVL GGUF (Fast)",
    "ArchAi3D_QwenVL_Server_Control": "🎛️ QwenVL Server Control",
    "ArchAi3D_LlamaCpp_Installer": "⚙️ Llama.cpp Installer",

    # Download Nodes
    "ArchAi3D_HF_Download": "📥 HF Download (Rename)",
    "ArchAi3D_CivitAI_Download": "📥 CivitAI Download (Rename)",
    "ArchAi3D_GDrive_Download": "📥 Google Drive Download",

    # Low VRAM Optimized Nodes
    "ArchAi3D_SAM3_Segment": "🎯 SAM3 Segment (Low VRAM)",
    "ArchAi3D_Metric3D_Normal": "🎯 Metric3D Normal (Low VRAM)",
    "ArchAi3D_Metric3D_Depth": "🎯 Metric3D Depth (Low VRAM)",

    # Installer Nodes
    "ArchAi3D_Nunchaku_Installer": "🔧 Nunchaku Installer",
    "ArchAi3D_Dependency_Installer": "🔧 Dependency Installer",
}

# ============================================================================
# WEB DIRECTORY (for custom UI elements)
# ============================================================================

import os
WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "web")

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
__version__ = "3.0.0"
__author__ = "Amir Ferdos (ArchAi3d)"

# ============================================================================
# STARTUP MESSAGE
# ============================================================================

print("=" * 70)
print(f"[ArchAi3d-Qwen v{__version__}] Loading nodes...")
print(f"  🎨 Core Encoding: 5 nodes (V3 + variants)")
print(f"  📏 Core Utils: 1 node (Image Scale)")
print(f"  💬 Prompt Builders: 3 nodes (Clean Room + Position Guide)")
print(f"  📸 Camera Control: 28 nodes (Object Focus v1-v7 + Simple + dx8152)")
print(f"  🎨 Image Editing: 4 nodes")
print(f"  🎯 Utils: 11 nodes (Smart Tile Prompter + Color Tools + Low VRAM)")
print(f"  🌐 Input Nodes: 9 nodes (String, Int, Float, Boolean, Load URL, Save, Conditioning Balance, Gemini)")
print(f"  🔧 Installer Nodes: 2 nodes (Nunchaku + Dependency)")
print(f"  ✅ Total: {len(NODE_CLASS_MAPPINGS)} nodes loaded!")
print(f"")
print(f"  ⭐ NEW: Dependency Installer - one-click install for all ArchAi3D deps!")
print(f"  ⭐ RTX 5090 Support: PyTorch 2.8+ and CUDA 12.8+ Blackwell compatibility!")
print(f"  📚 Documentation: ./docs/")
print(f"  ⚖️  License: Dual (Free personal, Commercial available)")
print("=" * 70)
