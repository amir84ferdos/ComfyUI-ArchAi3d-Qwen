# ArchAi3D Smart Tile Conditioning
#
# Batch encode all tile prompts with single CLIP load
# Optimized for use with Smart Tile Detailer
#
# Author: Amir Ferdos (ArchAi3d)
# Version: 1.2.0 - Strip positional prefixes for cleaner conditioning
# License: Dual License (Free for personal use, Commercial license required for business use)

import re


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _strip_positional_prefix(prompt):
    """
    Remove positional phrases that waste CLIP tokens.

    Examples of phrases removed:
    - "This top-left tile captures..."
    - "In the top-right corner,..."
    - "The bottom-left region shows..."
    - "In the bottom-right quadrant,..."

    The diffusion model doesn't need to know WHERE the tile is,
    only WHAT to generate.
    """
    # Patterns to strip (case insensitive)
    patterns = [
        # "This top-left tile captures/shows/displays..."
        r'^This\s+(top|bottom|center|middle)[-\s]?(left|right|center|middle)?\s+(tile|corner|region|quadrant|area|section|portion)\s+(captures|shows|displays|features|contains|reveals|depicts|presents)?[,\s]*',
        # "In the top-right corner,..."
        r'^In\s+the\s+(top|bottom|center|middle)[-\s]?(left|right|center|middle)?\s+(tile|corner|region|quadrant|area|section|portion)[,\s]*',
        # "The bottom-left region shows..."
        r'^The\s+(top|bottom|center|middle)[-\s]?(left|right|center|middle)?\s+(tile|corner|region|quadrant|area|section|portion)\s+(captures|shows|displays|features|contains|reveals|depicts|presents)?[,\s]*',
        # "Top-left:" or "Bottom-right tile:"
        r'^(top|bottom|center|middle)[-\s]?(left|right|center|middle)?\s*(tile|corner|region|quadrant|area|section|portion)?[:\s]+',
    ]

    cleaned = prompt
    for pattern in patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

    # Capitalize first letter after stripping
    cleaned = cleaned.strip()
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]

    return cleaned


# ============================================================================
# NODE CLASS
# ============================================================================

class ArchAi3D_Smart_Tile_Conditioning:
    """
    Batch encode tile prompts with single CLIP load.

    Takes comma-separated prompts from Smart Tile Prompter and encodes them all
    at once, returning a list of conditionings that can be matched to SEGs.

    Features:
    - CLIP loaded once instead of N times
    - Strips positional prefixes ("This top-left tile...") for cleaner conditioning
    - Significant time savings for multi-tile workflows
    - Direct integration with Smart Tile Detailer
    - Bundle support for one-wire connection from Prompter
    """

    CATEGORY = "ArchAi3d/Utils"
    RETURN_TYPES = ("CONDITIONING_LIST", "CONDITIONING", "STRING")
    RETURN_NAMES = ("conditionings", "negative", "debug_info")
    FUNCTION = "encode_batch"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {
                    "tooltip": "CLIP model for text encoding"
                }),
                "tile_prompts": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Comma-separated tile prompts from Smart Tile Prompter (tile_prompts_labels output)"
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "blurry, low quality, artifacts, distorted",
                    "tooltip": "Global negative prompt applied to all tiles"
                }),
            },
            "optional": {
                "bundle": ("SMART_TILE_BUNDLE", {
                    "tooltip": "Connect bundle from Smart Tile Prompter (auto-fills tile_prompts)"
                }),
                "strip_position_prefix": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Remove positional phrases like 'This top-left tile...' for cleaner conditioning"
                }),
            },
        }

    def encode_batch(self, clip, tile_prompts, negative_prompt, bundle=None, strip_position_prefix=True):
        """
        Encode all tile prompts with single CLIP load.

        Args:
            clip: CLIP model
            tile_prompts: Comma-separated prompts (from Smart Tile Prompter)
            negative_prompt: Global negative prompt
            bundle: Optional bundle from Smart Tile Prompter
            strip_position_prefix: Remove positional phrases before encoding

        Returns:
            conditionings: List of CONDITIONING (one per tile)
            negative: Single CONDITIONING for negative prompt
            debug_info: Debug string showing each tile's prompt
        """
        if clip is None:
            raise RuntimeError("ERROR: clip input is invalid: None")

        # If bundle provided, extract tile_prompts_labels
        tile_prompts_list = None  # For debug info with positions
        if bundle is not None:
            tile_prompts = bundle.get("tile_prompts_labels", tile_prompts)
            tile_prompts_list = bundle.get("tile_prompts_list", None)
            tiles_x = bundle.get("tiles_x", "?")
            tiles_y = bundle.get("tiles_y", "?")
            print(f"[Smart Tile Conditioning v1.2] Using bundle: {tiles_x}x{tiles_y} tiles")

        # Parse prompts - Smart Tile Prompter uses semicolons inside prompts, commas between
        prompts = [p.strip() for p in tile_prompts.split(',') if p.strip()]

        if not prompts:
            raise RuntimeError("ERROR: No prompts provided. Connect tile_prompts_labels from Smart Tile Prompter or bundle.")

        strip_status = "ON" if strip_position_prefix else "OFF"
        print(f"\n[Smart Tile Conditioning v1.2] Encoding {len(prompts)} prompts (strip_position_prefix: {strip_status})...")

        # Build debug info
        debug_lines = [
            "=" * 60,
            "Smart Tile Conditioning v1.2.0 (Debug Output)",
            "=" * 60,
            f"Total tiles: {len(prompts)}",
            f"Strip position prefix: {strip_status}",
            "",
        ]

        # Encode all prompts with single CLIP load
        conditionings = []
        for i, prompt in enumerate(prompts):
            # Replace semicolons back to commas (Smart Tile Prompter escapes them)
            prompt_decoded = prompt.replace(';', ',')

            # Strip positional prefix if enabled
            if strip_position_prefix:
                prompt_for_clip = _strip_positional_prefix(prompt_decoded)
            else:
                prompt_for_clip = prompt_decoded

            # Encode with CLIP
            tokens = clip.tokenize(prompt_for_clip)
            cond = clip.encode_from_tokens_scheduled(tokens)
            conditionings.append(cond)

            # Get position info from bundle if available
            if tile_prompts_list and i < len(tile_prompts_list):
                tile_info = tile_prompts_list[i]
                position = tile_info.get("position", f"tile_{i}")
                tile_num = tile_info.get("tile", i + 1)
            else:
                position = f"tile_{i}"
                tile_num = i + 1

            # Add to debug info - show what was ACTUALLY encoded
            debug_lines.append(f"--- Tile {tile_num} ({position}) ---")
            if strip_position_prefix and prompt_for_clip != prompt_decoded:
                debug_lines.append(f"[STRIPPED] Original started with positional phrase")
            debug_lines.append(prompt_for_clip)
            debug_lines.append("")

            # Progress indicator
            if (i + 1) % 5 == 0 or i == len(prompts) - 1:
                print(f"  Encoded {i + 1}/{len(prompts)} prompts")

        # Encode negative prompt once
        neg_tokens = clip.tokenize(negative_prompt)
        negative = clip.encode_from_tokens_scheduled(neg_tokens)

        debug_lines.append("--- Negative Prompt ---")
        debug_lines.append(negative_prompt)
        debug_lines.append("=" * 60)

        debug_info = "\n".join(debug_lines)

        print(f"[Smart Tile Conditioning v1.2] Done! {len(conditionings)} conditionings ready.")

        return (conditionings, negative, debug_info)


# ============================================================================
# NODE REGISTRATION
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "ArchAi3D_Smart_Tile_Conditioning": ArchAi3D_Smart_Tile_Conditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_Smart_Tile_Conditioning": "🎯 Smart Tile Conditioning",
}
