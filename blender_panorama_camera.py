"""
Blender Panorama Camera Setup Script
=====================================
Run this in Blender's Scripting tab to create a panorama/360° camera.

Author: ArchAi3d
Usage: Copy and paste into Blender > Scripting > New > Paste > Run Script
"""

import bpy
import math

def create_panorama_camera(
    name="Panorama_Camera",
    location=(0, 0, 1.7),  # Eye level height
    rotation=(90, 0, 0),   # Degrees (looking forward)
    panorama_type='EQUIRECTANGULAR',  # 'EQUIRECTANGULAR', 'FISHEYE_EQUIDISTANT', 'FISHEYE_EQUISOLID', 'MIRRORBALL'
    make_active=True
):
    """
    Create a panorama camera in Blender.

    Args:
        name: Camera name
        location: (x, y, z) position in meters
        rotation: (x, y, z) rotation in degrees
        panorama_type: Type of panorama projection
            - 'EQUIRECTANGULAR': Full 360° x 180° spherical (VR/360 photos)
            - 'FISHEYE_EQUIDISTANT': Fisheye with equal angle mapping
            - 'FISHEYE_EQUISOLID': Fisheye with equal area mapping
            - 'MIRRORBALL': Mirror ball reflection mapping
        make_active: Set as active camera

    Returns:
        The created camera object
    """

    # Create camera data
    cam_data = bpy.data.cameras.new(name=name)

    # Set to panoramic type (requires Cycles)
    cam_data.type = 'PANO'
    cam_data.cycles.panorama_type = panorama_type

    # For equirectangular, set full 360° coverage
    if panorama_type == 'EQUIRECTANGULAR':
        cam_data.cycles.latitude_min = math.radians(-90)
        cam_data.cycles.latitude_max = math.radians(90)
        cam_data.cycles.longitude_min = math.radians(-180)
        cam_data.cycles.longitude_max = math.radians(180)

    # Create camera object
    cam_obj = bpy.data.objects.new(name, cam_data)

    # Link to scene
    bpy.context.scene.collection.objects.link(cam_obj)

    # Set location
    cam_obj.location = location

    # Set rotation (convert degrees to radians)
    cam_obj.rotation_euler = (
        math.radians(rotation[0]),
        math.radians(rotation[1]),
        math.radians(rotation[2])
    )

    # Make active camera
    if make_active:
        bpy.context.scene.camera = cam_obj

    # Ensure Cycles render engine is set (required for panoramic cameras)
    bpy.context.scene.render.engine = 'CYCLES'

    # Select the camera
    bpy.ops.object.select_all(action='DESELECT')
    cam_obj.select_set(True)
    bpy.context.view_layer.objects.active = cam_obj

    print(f"✓ Created panorama camera '{name}'")
    print(f"  Type: {panorama_type}")
    print(f"  Location: {location}")
    print(f"  Rotation: {rotation}°")
    print(f"  Render engine set to: CYCLES")

    return cam_obj


def setup_panorama_render_settings(
    resolution_x=4096,
    resolution_y=2048,
    samples=128
):
    """
    Configure render settings optimized for panorama output.

    Args:
        resolution_x: Width (4096 for 4K, 8192 for 8K)
        resolution_y: Height (should be half of width for equirectangular)
        samples: Render samples
    """
    scene = bpy.context.scene

    # Resolution (2:1 aspect ratio for equirectangular)
    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y
    scene.render.resolution_percentage = 100

    # Cycles settings
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True

    # Output format
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'

    print(f"✓ Panorama render settings configured:")
    print(f"  Resolution: {resolution_x} x {resolution_y}")
    print(f"  Samples: {samples}")
    print(f"  Format: PNG")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Create the panorama camera
    camera = create_panorama_camera(
        name="Panorama_Camera",
        location=(0, 0, 1.7),      # 1.7m = average eye height
        rotation=(90, 0, 0),       # Looking forward
        panorama_type='EQUIRECTANGULAR',  # Full 360° spherical
        make_active=True
    )

    # Setup render settings for panorama output
    setup_panorama_render_settings(
        resolution_x=4096,   # 4K width
        resolution_y=2048,   # 2:1 aspect ratio
        samples=128
    )

    print("\n" + "="*50)
    print("PANORAMA CAMERA READY!")
    print("="*50)
    print("Press F12 to render, or use:")
    print("  View > Cameras > Panorama_Camera")
    print("="*50)
