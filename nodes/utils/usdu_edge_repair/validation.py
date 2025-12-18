"""
USDU Edge Repair - Validation
==============================

Input validation functions for the USDU Edge Repair node.
"""


def validate_safeguard(image, output_width, output_height,
                       tile_width, tile_height, tile_padding,
                       tiles_x, tiles_y):
    """
    Validate all values from Matrix Search match exactly.

    This is a CRITICAL safety check that ensures:
    1. Image dimensions match expected output size
    2. Tile coverage is sufficient (tiles actually cover the image)

    Raises ValueError if ANY mismatch detected (even 1 pixel difference).
    This ensures USDU uses exactly what Matrix Search calculated.

    Args:
        image: Input image tensor in BHWC format [batch, height, width, channels]
        output_width: Expected width from Matrix Search (pixels)
        output_height: Expected height from Matrix Search (pixels)
        tile_width: Tile width from Matrix Search (pixels)
        tile_height: Tile height from Matrix Search (pixels)
        tile_padding: Overlap/padding from Matrix Search (pixels)
        tiles_x: Number of tile columns from Matrix Search
        tiles_y: Number of tile rows from Matrix Search

    Returns:
        True if all checks pass

    Raises:
        ValueError: Detailed error message if any mismatch detected
    """
    errors = []

    # 1. Check image dimensions match expected output
    img_h, img_w = image.shape[1], image.shape[2]
    if img_w != output_width:
        errors.append(f"output_width: expected {output_width}, got {img_w} (diff: {abs(img_w - output_width)}px)")
    if img_h != output_height:
        errors.append(f"output_height: expected {output_height}, got {img_h} (diff: {abs(img_h - output_height)}px)")

    # 2. Verify coverage formula: coverage = tile * n - padding * (n - 1)
    if tiles_x > 1:
        coverage_w = tile_width * tiles_x - tile_padding * (tiles_x - 1)
    else:
        coverage_w = tile_width

    if tiles_y > 1:
        coverage_h = tile_height * tiles_y - tile_padding * (tiles_y - 1)
    else:
        coverage_h = tile_height

    if coverage_w < output_width:
        errors.append(f"coverage_width: {coverage_w}px < output_width {output_width}px")
    if coverage_h < output_height:
        errors.append(f"coverage_height: {coverage_h}px < output_height {output_height}px")

    # 3. If any errors, raise exception with detailed message
    if errors:
        error_msg = "\n".join([
            "",
            "=" * 60,
            "SAFEGUARD ERROR: Mismatch detected!",
            "=" * 60,
            "",
            "ERRORS:",
        ] + [f"  - {e}" for e in errors] + [
            "",
            f"EXPECTED: {output_width}x{output_height}",
            f"ACTUAL:   {img_w}x{img_h}",
            f"COVERAGE: {coverage_w}x{coverage_h}",
            "=" * 60,
        ])
        raise ValueError(error_msg)

    return True
