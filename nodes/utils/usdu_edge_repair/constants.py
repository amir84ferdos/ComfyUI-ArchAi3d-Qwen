"""
USDU Edge Repair - Constants
=============================

Shared constants used across the USDU Edge Repair module.
"""

from .usdu_patch import usdu

# Maximum supported image dimension
MAX_RESOLUTION = 8192

# Redraw mode mappings (string -> enum)
MODES = {
    "Linear": usdu.USDUMode.LINEAR,    # Process tiles row-by-row
    "Chess": usdu.USDUMode.CHESS,      # Process tiles in checkerboard pattern
    "None": usdu.USDUMode.NONE,        # Skip redraw pass
}

# Seam fix mode mappings (string -> enum)
SEAM_FIX_MODES = {
    "None": usdu.USDUSFMode.NONE,
    "Band Pass": usdu.USDUSFMode.BAND_PASS,
    "Half Tile": usdu.USDUSFMode.HALF_TILE,
    "Half Tile + Intersections": usdu.USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS,
}
