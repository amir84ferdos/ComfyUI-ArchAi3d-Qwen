# ArchAi3D Save Splat Scene Node
#
# Saves 3D Gaussian Splat data with camera information for webapps
# Supports: PLY, SPZ (compressed), JSON camera sidecar
#
# Author: Amir Ferdos (ArchAi3d)
# Email: Amir84ferdos@gmail.com
# Version: 1.0.0

import json
import os
import shutil
import struct
import gzip
import time
from pathlib import Path

import numpy as np

try:
    import folder_paths
    OUTPUT_DIR = folder_paths.get_output_directory()
except ImportError:
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "output")


def ply_to_spz(ply_path: str, spz_path: str, coordinate_system: str = "RUB") -> bool:
    """Convert PLY to SPZ format.

    Tries to use Niantic's SPZ library if available, otherwise uses built-in converter.

    Args:
        ply_path: Path to input PLY file
        spz_path: Path to output SPZ file
        coordinate_system: Coordinate system of input (RUB, RDF, LUF, RUF)

    Returns:
        True if successful, False otherwise
    """
    # Try Niantic's SPZ library first (C++ - fastest)
    try:
        import spz
        import time as _time

        # Get file size for diagnostics
        file_size_mb = os.path.getsize(ply_path) / (1024 * 1024)
        print(f"[SaveSplatScene] Input PLY: {file_size_mb:.1f} MB")

        start = _time.time()

        # Load PLY using SPZ library
        print(f"[SaveSplatScene] Loading PLY with Niantic library...")
        cloud = spz.load_splat_from_ply(ply_path)
        load_time = _time.time() - start
        print(f"[SaveSplatScene] PLY loaded in {load_time:.2f}s ({cloud.num_points if hasattr(cloud, 'num_points') else 'N/A'} points)")

        # Save as SPZ with coordinate system conversion
        pack_options = spz.PackOptions()
        pack_options.from_coord = getattr(spz, coordinate_system, spz.RDF)

        save_start = _time.time()
        success = spz.save_spz(cloud, pack_options, spz_path)
        save_time = _time.time() - save_start
        total_time = _time.time() - start

        if success:
            out_size_mb = os.path.getsize(spz_path) / (1024 * 1024)
            print(f"[SaveSplatScene] SPZ saved in {save_time:.2f}s ({out_size_mb:.1f} MB)")
            print(f"[SaveSplatScene] Total: {total_time:.2f}s (Niantic C++)")
            return True
        else:
            print(f"[SaveSplatScene] Niantic SPZ save failed")

    except ImportError as e:
        print(f"[SaveSplatScene] Niantic SPZ not installed: {e}")
    except Exception as e:
        print(f"[SaveSplatScene] Niantic SPZ error: {e}")

    # Try gsconverter (3dgsconverter) - use direct Python API
    try:
        from gsconverter import Converter
        import time as _time

        print(f"[SaveSplatScene] Trying gsconverter...")
        start = _time.time()
        converter = Converter(ply_path, spz_path, "spz")
        converter.run()
        elapsed = _time.time() - start

        out_size_mb = os.path.getsize(spz_path) / (1024 * 1024) if os.path.exists(spz_path) else 0
        print(f"[SaveSplatScene] Total: {elapsed:.2f}s (gsconverter) -> {out_size_mb:.1f} MB")
        return True
    except ImportError:
        print(f"[SaveSplatScene] gsconverter not installed")
    except Exception as e:
        print(f"[SaveSplatScene] gsconverter error: {e}")

    # Built-in basic SPZ converter (simplified version)
    try:
        return _builtin_ply_to_spz(ply_path, spz_path)
    except Exception as e:
        print(f"[SaveSplatScene] Built-in converter failed: {e}")
        return False


def _read_ply_gaussians(ply_path: str) -> dict:
    """Read Gaussian splat data from PLY file."""
    with open(ply_path, 'rb') as f:
        # Read header
        header_lines = []
        while True:
            line = f.readline().decode('ascii').strip()
            header_lines.append(line)
            if line == 'end_header':
                break

        # Parse header
        num_vertices = 0
        properties = []
        for line in header_lines:
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            elif line.startswith('property float'):
                prop_name = line.split()[-1]
                properties.append(prop_name)

        # Read binary data
        # Standard Gaussian splat PLY has: x, y, z, nx, ny, nz, f_dc_0..2, f_rest_0..44, opacity, scale_0..2, rot_0..3
        dtype = np.dtype([(p, '<f4') for p in properties])
        data = np.frombuffer(f.read(), dtype=dtype)

    return {
        'num_gaussians': num_vertices,
        'properties': properties,
        'data': data
    }


def _builtin_ply_to_spz(ply_path: str, spz_path: str) -> bool:
    """Built-in PLY to SPZ converter (basic implementation).

    SPZ format: gzipped stream with 16-byte header + 64 bytes per Gaussian
    """
    gaussians = _read_ply_gaussians(ply_path)
    data = gaussians['data']
    num_gaussians = gaussians['num_gaussians']

    # SPZ Header (16 bytes)
    # Magic (4), Version (4), NumGaussians (4), Flags (4)
    magic = b'SPZ\x00'
    version = struct.pack('<I', 2)  # Version 2.0
    num_pts = struct.pack('<I', num_gaussians)
    flags = struct.pack('<I', 0)

    header = magic + version + num_pts + flags

    # Pack Gaussians (64 bytes each)
    # Position: 3x float16 (6 bytes)
    # Scale: 3x uint8 (3 bytes)
    # Rotation: packed 32-bit (4 bytes)
    # Opacity: uint8 (1 byte)
    # SH DC: 3x uint8 (3 bytes)
    # SH Rest: 45x uint8 (45 bytes) - simplified: we'll zero-pad
    # Padding: 2 bytes

    packed_data = bytearray()

    for i in range(num_gaussians):
        # Position (3x float16 = 6 bytes)
        if 'x' in data.dtype.names:
            pos = np.array([data['x'][i], data['y'][i], data['z'][i]], dtype=np.float16)
        else:
            pos = np.zeros(3, dtype=np.float16)
        packed_data.extend(pos.tobytes())

        # Scale (3x uint8, log-encoded = 3 bytes)
        if 'scale_0' in data.dtype.names:
            scales = np.array([data['scale_0'][i], data['scale_1'][i], data['scale_2'][i]])
            # Quantize log-scale to uint8
            log_scales = np.log(np.abs(scales) + 1e-8)
            scale_bytes = np.clip((log_scales + 10) * 12.75, 0, 255).astype(np.uint8)
        else:
            scale_bytes = np.array([128, 128, 128], dtype=np.uint8)
        packed_data.extend(scale_bytes.tobytes())

        # Rotation (quaternion packed into 32 bits = 4 bytes)
        if 'rot_0' in data.dtype.names:
            quat = np.array([data['rot_0'][i], data['rot_1'][i], data['rot_2'][i], data['rot_3'][i]])
            quat = quat / (np.linalg.norm(quat) + 1e-8)  # Normalize
        else:
            quat = np.array([1.0, 0.0, 0.0, 0.0])

        # Pack quaternion: find largest component, store 3 smallest as 10-bit signed ints
        abs_quat = np.abs(quat)
        largest_idx = np.argmax(abs_quat)
        sign = 1 if quat[largest_idx] >= 0 else -1
        quat = quat * sign

        # Get 3 smallest components
        mask = np.ones(4, dtype=bool)
        mask[largest_idx] = False
        smallest_3 = quat[mask]

        # Quantize to 10-bit signed (-512 to 511)
        quant = np.clip(smallest_3 * 724, -511, 511).astype(np.int16)

        # Pack: 2 bits for index, 3x 10 bits for components
        packed_rot = (largest_idx & 0x3)
        packed_rot |= ((quant[0] & 0x3FF) << 2)
        packed_rot |= ((quant[1] & 0x3FF) << 12)
        packed_rot |= ((quant[2] & 0x3FF) << 22)
        packed_data.extend(struct.pack('<I', packed_rot & 0xFFFFFFFF))

        # Opacity (1 byte)
        if 'opacity' in data.dtype.names:
            opacity = 1.0 / (1.0 + np.exp(-data['opacity'][i]))  # Sigmoid
            opacity_byte = np.clip(opacity * 255, 0, 255).astype(np.uint8)
        else:
            opacity_byte = np.uint8(255)
        packed_data.extend(bytes([opacity_byte]))

        # SH DC coefficients (RGB, 3 bytes)
        if 'f_dc_0' in data.dtype.names:
            sh_dc = np.array([data['f_dc_0'][i], data['f_dc_1'][i], data['f_dc_2'][i]])
            # Convert SH to RGB and quantize
            rgb = (sh_dc * 0.28209479177387814 + 0.5)  # C0 coefficient
            rgb_bytes = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        else:
            rgb_bytes = np.array([128, 128, 128], dtype=np.uint8)
        packed_data.extend(rgb_bytes.tobytes())

        # SH Rest (45 bytes) - simplified: store zeros
        packed_data.extend(bytes(45))

        # Padding (2 bytes to reach 64 bytes total)
        packed_data.extend(bytes(2))

    # Gzip compress
    full_data = header + bytes(packed_data)
    with gzip.open(spz_path, 'wb') as f:
        f.write(full_data)

    compressed_size = os.path.getsize(spz_path)
    original_size = os.path.getsize(ply_path)
    ratio = original_size / compressed_size if compressed_size > 0 else 0

    print(f"[SaveSplatScene] Built-in SPZ conversion: {original_size/1024/1024:.1f}MB -> {compressed_size/1024/1024:.1f}MB ({ratio:.1f}x compression)")
    return True


def matrix_to_list(matrix) -> list:
    """Convert numpy array or torch tensor to nested list."""
    if hasattr(matrix, 'cpu'):
        matrix = matrix.cpu().numpy()
    if hasattr(matrix, 'tolist'):
        return matrix.tolist()
    return list(matrix)


def extract_camera_params(intrinsics, extrinsics, image_width: int = 1024, image_height: int = 768) -> dict:
    """Extract camera parameters from intrinsics/extrinsics matrices.

    Args:
        intrinsics: 4x4 intrinsic matrix or dict
        extrinsics: 4x4 extrinsic matrix (camera-to-world) or dict
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        Dict with camera parameters for Three.js/web
    """
    # Handle torch tensors
    if hasattr(intrinsics, 'cpu'):
        intrinsics = intrinsics.cpu().numpy()
    if hasattr(extrinsics, 'cpu'):
        extrinsics = extrinsics.cpu().numpy()

    # Extract focal length and principal point
    if isinstance(intrinsics, np.ndarray) and intrinsics.shape == (4, 4):
        fx = float(intrinsics[0, 0])
        fy = float(intrinsics[1, 1])
        cx = float(intrinsics[0, 2])
        cy = float(intrinsics[1, 2])
    elif isinstance(intrinsics, dict):
        fx = intrinsics.get('fx', intrinsics.get('focal_x', 1000))
        fy = intrinsics.get('fy', intrinsics.get('focal_y', 1000))
        cx = intrinsics.get('cx', image_width / 2)
        cy = intrinsics.get('cy', image_height / 2)
    else:
        fx = fy = 1000
        cx = image_width / 2
        cy = image_height / 2

    # Calculate FOV
    fov_x = 2 * np.arctan(image_width / (2 * fx)) * 180 / np.pi
    fov_y = 2 * np.arctan(image_height / (2 * fy)) * 180 / np.pi

    # Extract position and rotation from extrinsics
    if isinstance(extrinsics, np.ndarray) and extrinsics.shape == (4, 4):
        position = extrinsics[:3, 3].tolist()
        rotation_matrix = extrinsics[:3, :3].tolist()
        camera_to_world = extrinsics.tolist()
    elif isinstance(extrinsics, dict):
        position = extrinsics.get('position', [0, 0, 3])
        rotation_matrix = extrinsics.get('rotation', [[1,0,0],[0,1,0],[0,0,1]])
        camera_to_world = extrinsics.get('matrix', np.eye(4).tolist())
    else:
        position = [0, 0, 3]
        rotation_matrix = [[1,0,0],[0,1,0],[0,0,1]]
        camera_to_world = np.eye(4).tolist()

    return {
        "intrinsics": {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "width": image_width,
            "height": image_height,
            "fov_x_deg": fov_x,
            "fov_y_deg": fov_y,
            "matrix": matrix_to_list(intrinsics) if isinstance(intrinsics, np.ndarray) else None
        },
        "extrinsics": {
            "position": position,
            "rotation_matrix": rotation_matrix,
            "camera_to_world": camera_to_world
        }
    }


class ArchAi3D_SaveSplatScene:
    """Save 3D Gaussian Splat scene with camera data for webapps.

    Takes PLY path and camera matrices from SHARP or similar nodes,
    and saves:
    - SPZ file (compressed splat, ~10x smaller than PLY)
    - JSON file with camera intrinsics/extrinsics

    Perfect for Three.js, Babylon.js, or other web 3D viewers.
    """

    CATEGORY = "ArchAi3d/3D/Export"
    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("spz_path", "json_path", "scene_info",)
    FUNCTION = "save"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ply_path": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Path to input PLY file from SharpPredict or similar"
                }),
                "output_name": ("STRING", {
                    "default": "scene",
                    "tooltip": "Base name for output files (scene.spz, scene_camera.json)"
                }),
            },
            "optional": {
                "extrinsics": ("EXTRINSICS", {
                    "tooltip": "Camera extrinsics (position/rotation) from SHARP"
                }),
                "intrinsics": ("INTRINSICS", {
                    "tooltip": "Camera intrinsics (focal length, FOV) from SHARP"
                }),
                "image_width": ("INT", {
                    "default": 1024,
                    "min": 1,
                    "max": 8192,
                    "tooltip": "Original image width for FOV calculation"
                }),
                "image_height": ("INT", {
                    "default": 768,
                    "min": 1,
                    "max": 8192,
                    "tooltip": "Original image height for FOV calculation"
                }),
                "output_format": (["spz", "ply_copy", "both"], {
                    "default": "spz",
                    "tooltip": "Output format: spz (compressed), ply_copy, or both"
                }),
                "coordinate_system": (["RDF", "RUB", "LUF", "RUF"], {
                    "default": "RDF",
                    "tooltip": "Input PLY coordinate system (RDF=PLY default, RUB=OpenGL/Three.js)"
                }),
                "include_source_image": ("IMAGE", {
                    "tooltip": "Optional: include source image path in JSON"
                }),
            }
        }

    def save(self, ply_path: str, output_name: str,
             extrinsics=None, intrinsics=None,
             image_width: int = 1024, image_height: int = 768,
             output_format: str = "spz", coordinate_system: str = "RDF",
             include_source_image=None):
        """Save splat scene with camera data."""

        if not os.path.exists(ply_path):
            return ("", "", f"ERROR: PLY file not found: {ply_path}")

        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = int(time.time() * 1000)

        base_name = f"{output_name}_{timestamp}"

        # Output paths
        spz_path = os.path.join(OUTPUT_DIR, f"{base_name}.spz")
        ply_copy_path = os.path.join(OUTPUT_DIR, f"{base_name}.ply")
        json_path = os.path.join(OUTPUT_DIR, f"{base_name}_camera.json")

        results = []

        # Convert/copy PLY
        if output_format in ["spz", "both"]:
            success = ply_to_spz(ply_path, spz_path, coordinate_system)
            if success:
                results.append(f"SPZ: {os.path.basename(spz_path)}")
            else:
                # Fallback to PLY copy if SPZ fails
                shutil.copy2(ply_path, ply_copy_path)
                spz_path = ply_copy_path
                results.append(f"PLY (SPZ failed): {os.path.basename(ply_copy_path)}")

        if output_format in ["ply_copy", "both"]:
            shutil.copy2(ply_path, ply_copy_path)
            results.append(f"PLY: {os.path.basename(ply_copy_path)}")

        # Extract camera parameters
        camera_data = extract_camera_params(
            intrinsics if intrinsics is not None else np.eye(4),
            extrinsics if extrinsics is not None else np.eye(4),
            image_width, image_height
        )

        # Build scene JSON
        scene_json = {
            "version": "1.0",
            "generator": "ArchAi3D ComfyUI",
            "timestamp": timestamp,
            "splat_file": os.path.basename(spz_path if output_format == "spz" else ply_copy_path),
            "splat_format": "spz" if output_format == "spz" else "ply",
            "coordinate_system": coordinate_system,
            "camera": camera_data,
            "source": {
                "ply_path": ply_path,
            }
        }

        # Save JSON
        with open(json_path, 'w') as f:
            json.dump(scene_json, f, indent=2)
        results.append(f"JSON: {os.path.basename(json_path)}")

        # Summary
        scene_info = f"Saved: {', '.join(results)}"
        print(f"[SaveSplatScene] {scene_info}")

        final_splat_path = spz_path if output_format in ["spz", "both"] else ply_copy_path

        return (final_splat_path, json_path, scene_info)


NODE_CLASS_MAPPINGS = {
    "ArchAi3D_SaveSplatScene": ArchAi3D_SaveSplatScene,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchAi3D_SaveSplatScene": "Save Splat Scene (SPZ + Camera)",
}
