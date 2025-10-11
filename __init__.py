"""
ComfyUI-ArchAi3d-Qwen
Advanced Qwen-VL image scaling and encoding nodes for ComfyUI

Author: Amir Ferdos (ArchAi3d)
Email: Amir84ferdos@gmail.com
LinkedIn: https://www.linkedin.com/in/archai3d/
GitHub: https://github.com/amir84ferdos
Version: 2.0.0
License: MIT
"""

from .archai3d_qwen_encoder import ArchAi3D_Qwen_Encoder
from .archai3d_qwen_encoder_v2 import ArchAi3D_Qwen_Encoder_V2
from .archai3d_qwen_image_scale import ArchAi3D_Qwen_Image_Scale
from .archai3d_qwen_system_prompt import ArchAi3D_Qwen_System_Prompt
from .archai3d_qwen_encoder_simple import ArchAi3D_Qwen_Encoder_Simple
from .archai3d_qwen_encoder_simple_v2 import ArchAi3dQwenEncoderSimpleV2

# Define node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "ArchAi3D_Qwen_Encoder": ArchAi3D_Qwen_Encoder,
    "ArchAi3D_Qwen_Encoder_V2": ArchAi3D_Qwen_Encoder_V2,
    "ArchAi3D_Qwen_Image_Scale": ArchAi3D_Qwen_Image_Scale,
    "ArchAi3D_Qwen_System_Prompt": ArchAi3D_Qwen_System_Prompt,
    "ArchAi3D_Qwen_Encoder_Simple": ArchAi3D_Qwen_Encoder_Simple,
    "ArchAi3dQwenEncoderSimpleV2": ArchAi3dQwenEncoderSimpleV2,
}

# Define display names for the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_Qwen_Encoder": "ArchAi3D Qwen Encoder",
    "ArchAi3D_Qwen_Encoder_V2": "ArchAi3D Qwen Encoder V2",
    "ArchAi3D_Qwen_Image_Scale": "ArchAi3D Qwen Image Scale",
    "ArchAi3D_Qwen_System_Prompt": "ArchAi3D Qwen System Prompt",
    "ArchAi3D_Qwen_Encoder_Simple": "ArchAi3D Qwen Encoder Simple",
    "ArchAi3dQwenEncoderSimpleV2": "ArchAi3D Qwen Encoder Simple V2",
}

# Export the mappings for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
__version__ = "2.0.0"
__author__ = "Amir Ferdos (ArchAi3d)"

print(f"[ArchAi3d-Qwen v{__version__}] Loaded {len(NODE_CLASS_MAPPINGS)} nodes successfully!")
