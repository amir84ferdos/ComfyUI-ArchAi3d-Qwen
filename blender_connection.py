"""
Blender MCP Connection Helper
=============================
Connect to Blender via the blender-mcp socket server (port 9876).

This works when blender-mcp addon is running in Blender.

Usage:
    from blender_connection import BlenderConnection

    blender = BlenderConnection()
    blender.connect()

    # Get scene info
    info = blender.get_scene_info()
    print(info)

    # Execute code in Blender
    result = blender.execute_code("import bpy; print(bpy.data.objects[:])")
    print(result)

    blender.disconnect()

Author: ArchAi3d
"""

import socket
import json
from typing import Optional, Dict, Any


class BlenderConnection:
    """Connect to Blender via blender-mcp socket server."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9876, timeout: int = 10):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock: Optional[socket.socket] = None

    def connect(self) -> bool:
        """Connect to Blender MCP server."""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(self.timeout)
            self.sock.connect((self.host, self.port))
            print(f"✓ Connected to Blender at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"✗ Failed to connect: {e}")
            return False

    def disconnect(self):
        """Disconnect from Blender."""
        if self.sock:
            self.sock.close()
            self.sock = None
            print("✓ Disconnected from Blender")

    def _send_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Send a command and receive response."""
        if not self.sock:
            raise ConnectionError("Not connected to Blender")

        self.sock.send(json.dumps(command).encode() + b'\n')
        response = self.sock.recv(65536).decode()
        return json.loads(response)

    def get_scene_info(self) -> Dict[str, Any]:
        """Get information about the current Blender scene."""
        result = self._send_command({"type": "get_scene_info"})
        if result.get("status") == "success":
            return result["result"]
        else:
            raise RuntimeError(result.get("message", "Unknown error"))

    def execute_code(self, code: str) -> Dict[str, Any]:
        """Execute Python code in Blender.

        Args:
            code: Python code to execute in Blender

        Returns:
            Dict with 'executed' status and 'result' output
        """
        result = self._send_command({
            "type": "execute_code",
            "params": {"code": code}
        })
        if result.get("status") == "success":
            return result["result"]
        else:
            raise RuntimeError(result.get("message", "Unknown error"))

    def create_panorama_camera(
        self,
        name: str = "Panorama_Camera",
        location: tuple = (0, 0, 1.7),
        set_active: bool = True,
        resolution: tuple = (4096, 2048)
    ) -> Dict[str, Any]:
        """Create a 360° panorama camera in Blender.

        Args:
            name: Camera name
            location: (x, y, z) position
            set_active: Set as active scene camera
            resolution: (width, height) render resolution

        Returns:
            Execution result
        """
        code = f'''
import bpy
import math

# Create panorama camera
cam_data = bpy.data.cameras.new(name="{name}")
cam_data.type = 'PANO'

# Panorama settings (Blender 4.0+ / 5.0 API)
cam_data.panorama_type = 'EQUIRECTANGULAR'
cam_data.latitude_min = math.radians(-90)
cam_data.latitude_max = math.radians(90)
cam_data.longitude_min = math.radians(-180)
cam_data.longitude_max = math.radians(180)

# Create camera object
cam_obj = bpy.data.objects.new("{name}", cam_data)
bpy.context.scene.collection.objects.link(cam_obj)

# Set position
cam_obj.location = {location}
cam_obj.rotation_euler = (math.radians(90), 0, 0)

# Set as active camera
{"bpy.context.scene.camera = cam_obj" if set_active else "pass"}

# Ensure Cycles render engine
bpy.context.scene.render.engine = 'CYCLES'

# Set resolution
bpy.context.scene.render.resolution_x = {resolution[0]}
bpy.context.scene.render.resolution_y = {resolution[1]}

# Select camera
bpy.ops.object.select_all(action='DESELECT')
cam_obj.select_set(True)
bpy.context.view_layer.objects.active = cam_obj

print(f"Panorama camera '{name}' created at {{cam_obj.location[:]}}")
'''
        return self.execute_code(code)

    def list_objects(self, limit: int = 50) -> list:
        """List objects in the scene."""
        info = self.get_scene_info()
        return info.get("objects", [])[:limit]

    def get_object_count(self) -> int:
        """Get total number of objects in scene."""
        info = self.get_scene_info()
        return info.get("object_count", 0)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_connect() -> BlenderConnection:
    """Quick connect to Blender and return connection object."""
    blender = BlenderConnection()
    if blender.connect():
        return blender
    else:
        raise ConnectionError("Could not connect to Blender. Is blender-mcp running?")


def create_panorama_camera_quick(
    location: tuple = (0, 0, 1.7),
    name: str = "Panorama_Camera"
) -> bool:
    """Quick function to create a panorama camera.

    Usage:
        from blender_connection import create_panorama_camera_quick
        create_panorama_camera_quick(location=(5, 0, 2))
    """
    try:
        blender = quick_connect()
        blender.create_panorama_camera(name=name, location=location)
        blender.disconnect()
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    # Test connection
    blender = BlenderConnection()

    if blender.connect():
        # Get scene info
        print("\n--- Scene Info ---")
        info = blender.get_scene_info()
        print(f"Scene: {info['name']}")
        print(f"Objects: {info['object_count']}")

        # List first 10 objects
        print("\n--- First 10 Objects ---")
        for obj in info['objects'][:10]:
            print(f"  - {obj['name']} ({obj['type']})")

        blender.disconnect()
    else:
        print("Make sure Blender is open with blender-mcp addon running!")
