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
    # TODO: Pass coordinate_system to _builtin_ply_to_spz and implement
    # coordinate conversion (flipP, flipQ, flipSh arrays). Currently assumes
    # input PLY is in RDF (standard 3DGS output) which works for most cases.
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
    """Built-in PLY to SPZ v3 converter.

    SPZ v3 format: gzipped stream with 16-byte header followed by data organized by attribute:
    positions, alphas, colors, scales, rotations, spherical harmonics.

    Header structure (16 bytes):
        uint32_t magic (0x5053474e)
        uint32_t version (3)
        uint32_t numPoints
        uint8_t shDegree (0-3)
        uint8_t fractionalBits (for fixed-point positions)
        uint8_t flags
        uint8_t reserved

    Data layout (by attribute, not per-gaussian):
        positions: numPoints * 9 bytes (3x 24-bit fixed-point)
        alphas: numPoints * 1 byte
        colors: numPoints * 3 bytes (RGB)
        scales: numPoints * 3 bytes (log-encoded)
        rotations: numPoints * 4 bytes (2-bit index + 3x 10-bit smallest-three)
        sh_coeffs: numPoints * (shDegree > 0 ? 3 * ((shDegree+1)^2 - 1) : 0) bytes
    """
    gaussians = _read_ply_gaussians(ply_path)
    data = gaussians['data']
    num_gaussians = gaussians['num_gaussians']

    # Determine SH degree from available properties
    sh_degree = 0
    if 'f_rest_0' in data.dtype.names:
        # Count f_rest properties to determine SH degree
        # degree 1: 9 coeffs (3 DC + 6 rest) -> 3 f_rest per channel
        # degree 2: 24 coeffs -> 21 rest
        # degree 3: 45 coeffs -> 42 rest
        f_rest_count = sum(1 for p in data.dtype.names if p.startswith('f_rest_'))
        if f_rest_count >= 45:
            sh_degree = 3
        elif f_rest_count >= 24:
            sh_degree = 2
        elif f_rest_count >= 9:
            sh_degree = 1

    # Calculate fractional bits for position encoding
    # Find position bounds to determine optimal fractional bits
    if 'x' in data.dtype.names:
        positions = np.column_stack([data['x'], data['y'], data['z']])
        pos_min = positions.min()
        pos_max = positions.max()
        pos_range = max(abs(pos_min), abs(pos_max))
        # 24-bit signed = 23 bits for magnitude, need to fit pos_range
        # fractional_bits = 23 - ceil(log2(pos_range + 1))
        if pos_range > 0:
            int_bits_needed = max(1, int(np.ceil(np.log2(pos_range + 1))))
            fractional_bits = min(23 - int_bits_needed, 16)  # Cap at 16
            fractional_bits = max(fractional_bits, 0)
        else:
            fractional_bits = 12  # Default
    else:
        positions = np.zeros((num_gaussians, 3))
        fractional_bits = 12

    # SPZ v3 Header (16 bytes)
    # Magic: 0x5053474e (little-endian: bytes 4e 47 53 50)
    magic = struct.pack('<I', 0x5053474e)
    version = struct.pack('<I', 3)
    num_pts = struct.pack('<I', num_gaussians)
    sh_degree_byte = struct.pack('B', sh_degree)
    fractional_bits_byte = struct.pack('B', fractional_bits)
    flags_byte = struct.pack('B', 0)
    reserved_byte = struct.pack('B', 0)

    header = magic + version + num_pts + sh_degree_byte + fractional_bits_byte + flags_byte + reserved_byte

    # Pack data by attribute (not per-gaussian)
    packed_data = bytearray()

    # 1. Positions: 24-bit fixed-point signed integers (3 bytes each, 9 bytes per point)
    scale_factor = float(1 << fractional_bits)
    for i in range(num_gaussians):
        for axis in range(3):
            val = positions[i, axis]
            fixed_val = int(np.clip(val * scale_factor, -8388608, 8388607))  # 24-bit signed range
            # Pack as 3 bytes, little-endian (handle negative values with 2's complement)
            if fixed_val < 0:
                fixed_val = fixed_val & 0xFFFFFF  # Convert to 24-bit unsigned representation
            packed_data.append(fixed_val & 0xFF)
            packed_data.append((fixed_val >> 8) & 0xFF)
            packed_data.append((fixed_val >> 16) & 0xFF)

    # 2. Alphas (opacity): 1 byte per point
    for i in range(num_gaussians):
        if 'opacity' in data.dtype.names:
            opacity = 1.0 / (1.0 + np.exp(-float(data['opacity'][i])))  # Sigmoid
            alpha_byte = int(np.clip(opacity * 255, 0, 255))
        else:
            alpha_byte = 255
        packed_data.append(alpha_byte)

    # 3. Colors (raw SH DC coefficients): 3 bytes per point
    # SPZ stores raw SH DC values (NOT converted to RGB)
    # Encoding: byte = shDC * colorScale * 255 + 0.5 * 255, where colorScale = 0.15
    # This allows representing out-of-range values for SH compensation
    COLOR_SCALE = 0.15
    for i in range(num_gaussians):
        if 'f_dc_0' in data.dtype.names:
            sh_dc = np.array([data['f_dc_0'][i], data['f_dc_1'][i], data['f_dc_2'][i]])
            # SPZ color encoding: raw SH DC * 0.15 * 255 + 127.5
            rgb_bytes = np.clip(sh_dc * COLOR_SCALE * 255 + 0.5 * 255, 0, 255).astype(np.uint8)
        else:
            rgb_bytes = np.array([128, 128, 128], dtype=np.uint8)
        packed_data.extend(rgb_bytes.tobytes())

    # 4. Scales: 3 bytes per point (log-encoded)
    # SPZ encoding: scale_byte = (scale + 10) * 16
    for i in range(num_gaussians):
        if 'scale_0' in data.dtype.names:
            # PLY stores scales as log-scale already
            scales = np.array([data['scale_0'][i], data['scale_1'][i], data['scale_2'][i]])
            scale_bytes = np.clip((scales + 10.0) * 16.0, 0, 255).astype(np.uint8)
        else:
            scale_bytes = np.array([128, 128, 128], dtype=np.uint8)
        packed_data.extend(scale_bytes.tobytes())

    # 5. Rotations: 4 bytes per point (SPZ v3 smallest-three encoding)
    # Format: bits 30-31 = largest index, bits 0-29 = 3x 10-bit signed components
    # Each 10-bit component: 9-bit magnitude + 1 sign bit
    # Decoder reads components in reverse order (i=3,2,1,0 excluding largest)
    SQRT1_2 = 0.7071067811865475  # 1/sqrt(2)
    C_MASK = 511  # (1 << 9) - 1
    for i in range(num_gaussians):
        if 'rot_0' in data.dtype.names:
            quat = np.array([data['rot_0'][i], data['rot_1'][i], data['rot_2'][i], data['rot_3'][i]])
            norm = np.linalg.norm(quat)
            if norm > 1e-8:
                quat = quat / norm
            else:
                quat = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            quat = np.array([1.0, 0.0, 0.0, 0.0])

        # Find largest component
        abs_quat = np.abs(quat)
        largest_idx = int(np.argmax(abs_quat))

        # Ensure largest component is positive (quaternion sign invariance)
        if quat[largest_idx] < 0:
            quat = -quat

        # Build packed rotation: largest index in bits 30-31
        packed_rot = (largest_idx & 0x3) << 30

        # Pack components in reverse order (3,2,1,0 excluding largest)
        # This matches decoder which reads from LSB first for highest indices
        bit_offset = 0
        for j in range(3, -1, -1):  # 3, 2, 1, 0
            if j == largest_idx:
                continue
            val = quat[j]
            # Quantize: magnitude = C_MASK * (|val| / sqrt(2))
            magnitude = int(min(C_MASK, abs(val) / SQRT1_2 * C_MASK + 0.5))
            sign_bit = 1 if val < 0 else 0
            # Pack: 9-bit magnitude, then 1 sign bit
            component_10bit = (magnitude & 0x1FF) | (sign_bit << 9)
            packed_rot |= (component_10bit << bit_offset)
            bit_offset += 10

        packed_data.extend(struct.pack('<I', packed_rot))

    # 6. Spherical Harmonics (if sh_degree > 0)
    # PLY stores channel-major: f_rest_0..14 = R, f_rest_15..29 = G, f_rest_30..44 = B
    # SPZ expects coefficient-major interleaved: [R0, G0, B0, R1, G1, B1, ...]
    # Quantization: degree 1 uses 5 bits (bucket=8), higher uses 4 bits (bucket=16)
    if sh_degree > 0:
        # Number of SH coefficients per channel (excluding DC)
        # degree 1: 3, degree 2: 8, degree 3: 15
        num_sh_per_channel = (sh_degree + 1) ** 2 - 1

        # Bucket sizes for quantization
        SH1_BUCKET = 8    # 5-bit precision for degree 1 (3 coeffs)
        SH_REST_BUCKET = 16  # 4-bit precision for higher degrees

        def quantize_sh(val, bucket_size):
            """Quantize SH coefficient with bucket centering."""
            q = int(round(val * 128.0) + 128.0)
            q = ((q + bucket_size // 2) // bucket_size) * bucket_size
            return max(0, min(255, q))

        for i in range(num_gaussians):
            sh_bytes = []
            # Reorganize from PLY channel-major to SPZ coefficient-major interleaved
            for coeff_idx in range(num_sh_per_channel):
                # Determine bucket size: first 3 coeffs (degree 1) use finer quantization
                bucket = SH1_BUCKET if coeff_idx < 3 else SH_REST_BUCKET

                for channel in range(3):  # R, G, B
                    # PLY organization: channel * num_sh_per_channel + coeff_idx
                    ply_idx = channel * num_sh_per_channel + coeff_idx
                    prop_name = f'f_rest_{ply_idx}'
                    if prop_name in data.dtype.names:
                        val = float(data[prop_name][i])
                        quant_val = quantize_sh(val, bucket)
                    else:
                        quant_val = 128  # Neutral value
                    sh_bytes.append(quant_val)
            packed_data.extend(sh_bytes)

    # Gzip compress
    full_data = header + bytes(packed_data)
    with gzip.open(spz_path, 'wb') as f:
        f.write(full_data)

    compressed_size = os.path.getsize(spz_path)
    original_size = os.path.getsize(ply_path)
    ratio = original_size / compressed_size if compressed_size > 0 else 0

    print(f"[SaveSplatScene] Built-in SPZ v3 conversion: {original_size/1024/1024:.1f}MB -> {compressed_size/1024/1024:.1f}MB ({ratio:.1f}x compression)")
    print(f"[SaveSplatScene] SH degree: {sh_degree}, fractional bits: {fractional_bits}, points: {num_gaussians}")
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
        if output_format == "both":
            splat_file_info = {
                "spz": os.path.basename(spz_path),
                "ply": os.path.basename(ply_copy_path)
            }
            splat_format = "both"
        elif output_format == "spz":
            splat_file_info = os.path.basename(spz_path)
            splat_format = "spz"
        else:
            splat_file_info = os.path.basename(ply_copy_path)
            splat_format = "ply"

        scene_json = {
            "version": "1.0",
            "generator": "ArchAi3D ComfyUI",
            "timestamp": timestamp,
            "splat_file": splat_file_info,
            "splat_format": splat_format,
            "coordinate_system": coordinate_system,
            "camera": camera_data,
            "source": {
                "ply_path": ply_path,
            }
        }

        # Add source image info if provided
        if include_source_image is not None:
            scene_json["source"]["has_image"] = True
            scene_json["source"]["image_shape"] = list(include_source_image.shape) if hasattr(include_source_image, 'shape') else None

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
