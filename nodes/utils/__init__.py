"""
Utility nodes for ArchAi3d Qwen workflows

ComfyUI automatically discovers all .py files with comfy_entrypoint() functions.
No explicit imports needed - just having the files in this directory is enough.

UTILITY NODES:
- archai3d_mask_to_position_guide.py - Convert mask to numbered position guide image
  - Detects regions in mask
  - Draws numbered rectangles (left-to-right or top-to-bottom order)
  - Perfect for Qwen position mapping workflow
  - User validated: "it is working"

- archai3d_position_guide_prompt_builder.py - Build formatted prompts for position guide workflow
  - Accepts object descriptions separated by /
  - Auto-formats into Qwen position guide prompt structure
  - 4 template presets (standard, no_removal, minimal, custom)
  - Outputs: (rectangle 1= description), (rectangle 2= description), ...
  - User validated: Working with exact formatting
"""
