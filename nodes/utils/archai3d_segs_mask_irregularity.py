# ArchAi3D SEGS Mask Irregularity
#
# Add irregularity/wobble to SEGS mask edges
# Makes tile seams less visible by adding organic variation
#
# Author: Amir Ferdos (ArchAi3d)
# Version: 1.0.0
# License: Dual License (Free for personal use, Commercial license required for business use)

import numpy as np
from collections import namedtuple

# Define SEG namedtuple (compatible with Impact Pack)
SEG = namedtuple("SEG",
                 ['cropped_image', 'cropped_mask', 'confidence', 'crop_region', 'bbox', 'label', 'control_net_wrapper'],
                 defaults=[None])


def add_edge_irregularity(mask, amount=0.5, seed=None):
    """
    Add irregularity to mask edges by random erosion/dilation.

    Args:
        mask: numpy array (H, W) with values 0-1
        amount: How much irregularity (0=none, 1=maximum)
        seed: Random seed for reproducibility

    Returns:
        Modified mask with irregular edges
    """
    if amount <= 0:
        return mask

    rng = np.random.RandomState(seed)
    h, w = mask.shape

    # Find edges (where mask transitions from 0 to 1 or 1 to 0)
    # Use gradient to find edge pixels
    grad_x = np.abs(np.diff(mask, axis=1, prepend=mask[:, :1]))
    grad_y = np.abs(np.diff(mask, axis=0, prepend=mask[:1, :]))
    edges = (grad_x > 0.5) | (grad_y > 0.5)

    # Get edge pixel coordinates
    edge_y, edge_x = np.where(edges)

    if len(edge_x) == 0:
        return mask

    # Create output mask
    result = mask.copy()

    # Amount controls how many edge pixels to modify and by how much
    num_to_modify = int(len(edge_x) * amount * 0.5)
    if num_to_modify == 0:
        return mask

    # Randomly select edge pixels to modify
    indices = rng.choice(len(edge_x), size=min(num_to_modify, len(edge_x)), replace=False)

    # For each selected edge pixel, randomly expand or contract
    wobble_range = max(2, int(min(h, w) * amount * 0.05))

    for idx in indices:
        y, x = edge_y[idx], edge_x[idx]

        # Random wobble direction and distance
        wobble_dist = rng.randint(1, wobble_range + 1)

        # Randomly choose to expand (add 1s) or contract (add 0s)
        if rng.random() < 0.5:
            # Expand: set nearby pixels to 1
            y1 = max(0, y - wobble_dist)
            y2 = min(h, y + wobble_dist + 1)
            x1 = max(0, x - wobble_dist)
            x2 = min(w, x + wobble_dist + 1)

            # Only expand into 0 areas
            if mask[y, x] > 0.5:
                result[y1:y2, x1:x2] = np.maximum(result[y1:y2, x1:x2],
                                                   rng.random((y2-y1, x2-x1)) * amount)
        else:
            # Contract: set nearby pixels to lower values
            y1 = max(0, y - wobble_dist)
            y2 = min(h, y + wobble_dist + 1)
            x1 = max(0, x - wobble_dist)
            x2 = min(w, x + wobble_dist + 1)

            # Only contract from 1 areas
            if mask[y, x] > 0.5:
                reduction = rng.random((y2-y1, x2-x1)) * amount * 0.5
                result[y1:y2, x1:x2] = np.maximum(0, result[y1:y2, x1:x2] - reduction)

    # Smooth the result slightly to avoid harsh edges
    # Simple box blur
    kernel_size = max(3, int(wobble_range * 0.5))
    if kernel_size % 2 == 0:
        kernel_size += 1

    from scipy.ndimage import uniform_filter
    result = uniform_filter(result, size=kernel_size)

    # Ensure we don't go below 0 or above 1
    result = np.clip(result, 0.0, 1.0)

    return result


class ArchAi3D_SEGS_Mask_Irregularity:
    """
    Add irregularity/wobble to SEGS mask edges.

    This makes tile seams less visible by adding organic variation
    to the mask boundaries. Works on any SEGS input.
    """

    CATEGORY = "ArchAi3d/Utils"
    RETURN_TYPES = ("SEGS",)
    RETURN_NAMES = ("segs",)
    FUNCTION = "add_irregularity"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segs": ("SEGS", {
                    "tooltip": "Input SEGS to modify"
                }),
                "irregularity": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Amount of edge irregularity (0=none, 1=maximum)"
                }),
            },
            "optional": {
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffff,
                    "tooltip": "Random seed for reproducibility (0=random)"
                }),
            }
        }

    def add_irregularity(self, segs, irregularity, seed=0):
        """
        Add irregularity to all masks in SEGS.
        """
        # Unpack SEGS
        (ih, iw), seg_list = segs

        if irregularity <= 0:
            print(f"[SEGS Mask Irregularity v1.0] No changes (irregularity=0)")
            return (segs,)

        print(f"[SEGS Mask Irregularity v1.0] Processing {len(seg_list)} segments with irregularity={irregularity}")

        new_segs = []
        for i, seg in enumerate(seg_list):
            # Generate unique seed for each segment
            seg_seed = seed + i if seed > 0 else None

            # Modify mask
            new_mask = add_edge_irregularity(seg.cropped_mask, irregularity, seg_seed)

            # Create new SEG with modified mask
            new_seg = SEG(
                cropped_image=seg.cropped_image,
                cropped_mask=new_mask,
                confidence=seg.confidence,
                crop_region=seg.crop_region,
                bbox=seg.bbox,
                label=seg.label,
                control_net_wrapper=seg.control_net_wrapper
            )
            new_segs.append(new_seg)

        result = ((ih, iw), new_segs)
        print(f"[SEGS Mask Irregularity v1.0] Done - modified {len(new_segs)} masks")

        return (result,)


# ============================================================================
# NODE REGISTRATION
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "ArchAi3D_SEGS_Mask_Irregularity": ArchAi3D_SEGS_Mask_Irregularity,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_SEGS_Mask_Irregularity": "🎭 SEGS Mask Irregularity",
}
