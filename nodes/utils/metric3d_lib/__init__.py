"""
Metric3D Library - Standalone version for ArchAi3D

Based on comfyui_controlnet_aux/src/custom_controlnet_aux/metric3d
Auto-downloads models from HuggingFace.
"""

import os
import sys
import warnings
import tempfile
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import cv2
import re
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

# Add parent directory for custom_mmpkg import
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PARENT_DIR = CURRENT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

CODE_SPACE = CURRENT_DIR

# Import from local custom_mmpkg
from custom_mmpkg.custom_mmcv.utils import Config, DictAction

# Import from local mono module
from .mono.model.monodepth_model import get_configured_monodepth_model
from .mono.utils.running import load_ckpt
from .mono.utils.do_test import transform_test_data_scalecano, get_prediction
from .mono.utils.visualization import vis_surface_normal

# Model download settings
METRIC3D_MODEL_NAME = "JUGGHM/Metric3D"
MODELS_DIR = os.path.join(Path(__file__).parents[4], "models", "metric3d")
os.makedirs(MODELS_DIR, exist_ok=True)


def custom_hf_download(pretrained_model_or_path, filename):
    """Download model from HuggingFace Hub."""
    local_path = os.path.join(MODELS_DIR, filename)

    if os.path.exists(local_path):
        print(f"[Metric3D] Model found: {local_path}")
        return local_path

    print(f"[Metric3D] Downloading {filename} from {pretrained_model_or_path}...")

    try:
        downloaded_path = hf_hub_download(
            repo_id=pretrained_model_or_path,
            filename=filename,
            local_dir=MODELS_DIR,
            resume_download=True,
        )
        print(f"[Metric3D] Download complete: {downloaded_path}")
        return downloaded_path
    except Exception as e:
        print(f"[Metric3D] Download failed: {e}")
        raise


def HWC3(x):
    """Ensure image is in HWC format with 3 channels."""
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def safer_memory(x):
    """Fix many MAC/AMD problems."""
    return np.ascontiguousarray(x.copy()).copy()


UPSCALE_METHODS = ["INTER_NEAREST", "INTER_LINEAR", "INTER_AREA", "INTER_CUBIC", "INTER_LANCZOS4"]


def get_upscale_method(method_str):
    assert method_str in UPSCALE_METHODS, f"Method {method_str} not found in {UPSCALE_METHODS}"
    return getattr(cv2, method_str)


def pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)


def resize_image_with_pad(input_image, resolution, upscale_method="", skip_hwc3=False, mode='edge'):
    """Resize image with padding to multiple of 64."""
    if skip_hwc3:
        img = input_image
    else:
        img = HWC3(input_image)
    H_raw, W_raw, _ = img.shape
    if resolution == 0:
        return img, lambda x: x
    k = float(resolution) / float(min(H_raw, W_raw))
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    img = cv2.resize(img, (W_target, H_target), interpolation=get_upscale_method(upscale_method) if k > 1 else cv2.INTER_AREA)
    H_pad, W_pad = pad64(H_target), pad64(W_target)
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode=mode)

    def remove_pad(x):
        return safer_memory(x[:H_target, :W_target, ...])

    return safer_memory(img_padded), remove_pad


def common_input_validate(input_image, output_type, **kwargs):
    """Validate and convert input image."""
    if "img" in kwargs:
        warnings.warn("img is deprecated, please use `input_image=...` instead.", DeprecationWarning)
        input_image = kwargs.pop("img")

    if "return_pil" in kwargs:
        warnings.warn("return_pil is deprecated. Use output_type instead.", DeprecationWarning)
        output_type = "pil" if kwargs["return_pil"] else "np"

    if type(output_type) is bool:
        warnings.warn("Passing `True` or `False` to `output_type` is deprecated")
        if output_type:
            output_type = "pil"

    if input_image is None:
        raise ValueError("input_image must be defined.")

    if not isinstance(input_image, np.ndarray):
        input_image = np.array(input_image, dtype=np.uint8)
        output_type = output_type or "pil"
    else:
        output_type = output_type or "np"

    return (input_image, output_type)


def load_model(model_selection, model_path):
    """Load Metric3D model with specified backbone."""
    if model_selection == "vit-small":
        cfg = Config.fromfile(CODE_SPACE / 'mono/configs/HourglassDecoder/vit.raft5.small.py')
    elif model_selection == "vit-large":
        cfg = Config.fromfile(CODE_SPACE / 'mono/configs/HourglassDecoder/vit.raft5.large.py')
    elif model_selection == "vit-giant2":
        cfg = Config.fromfile(CODE_SPACE / 'mono/configs/HourglassDecoder/vit.raft5.giant2.py')
    else:
        raise NotImplementedError(f"metric3d model: {model_selection}")
    model = get_configured_monodepth_model(cfg)
    model, _, _, _ = load_ckpt(model_path, model, strict_match=False)
    model.eval()
    return model, cfg


def gray_to_colormap(img, cmap='rainbow'):
    """Transfer gray map to matplotlib colormap."""
    assert img.ndim == 2

    img[img < 0] = 0
    mask_invalid = img < 1e-10
    img = img / (img.max() + 1e-8)
    norm = plt.Normalize(vmin=0, vmax=1.1)
    cmap_m = plt.get_cmap(cmap)
    map = plt.cm.ScalarMappable(norm=norm, cmap=cmap_m)
    colormap = (map.to_rgba(img)[:, :, :3] * 255).astype(np.uint8)
    colormap[mask_invalid] = 0
    return colormap


def predict_depth_normal(model, cfg, np_img, fx=1000.0, fy=1000.0, state_cache={}):
    """Run Metric3D prediction for depth and normal."""
    intrinsic = [fx, fy, np_img.shape[1]/2, np_img.shape[0]/2]
    rgb_input, cam_models_stacks, pad, label_scale_factor = transform_test_data_scalecano(
        np_img, intrinsic, cfg.data_basic, device=next(model.parameters()).device
    )

    with torch.no_grad():
        pred_depth, confidence, output = get_prediction(
            model=model,
            input=rgb_input.unsqueeze(0),
            cam_model=cam_models_stacks,
            pad_info=pad,
            scale_info=label_scale_factor,
            gt_depth=None,
            normalize_scale=cfg.data_basic.depth_range[1],
            ori_shape=[np_img.shape[0], np_img.shape[1]],
        )

        pred_normal = output['normal_out_list'][0][:, :3, :, :]
        H, W = pred_normal.shape[2:]
        pred_normal = pred_normal[:, :, pad[0]:H-pad[1], pad[2]:W-pad[3]]
        pred_depth = pred_depth[:, :, pad[0]:H-pad[1], pad[2]:W-pad[3]]

    pred_depth = pred_depth.squeeze().cpu().numpy()
    pred_color = gray_to_colormap(pred_depth, 'Greys')

    pred_normal = torch.nn.functional.interpolate(
        pred_normal, [np_img.shape[0], np_img.shape[1]], mode='bilinear'
    ).squeeze()
    pred_normal = pred_normal.permute(1, 2, 0)
    pred_color_normal = vis_surface_normal(pred_normal)
    pred_normal = pred_normal.cpu().numpy()

    # Storing depth and normal map in state for potential 3D reconstruction
    state_cache['depth'] = pred_depth
    state_cache['normal'] = pred_normal
    state_cache['img'] = np_img
    state_cache['intrinsic'] = intrinsic
    state_cache['confidence'] = confidence

    return pred_color, pred_color_normal, state_cache


class Metric3DDetector:
    """Metric3D Depth and Normal Map Detector."""

    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.device = "cpu"

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path=METRIC3D_MODEL_NAME, filename="metric_depth_vit_small_800k.pth"):
        """Load pretrained model from HuggingFace."""
        model_path = custom_hf_download(pretrained_model_or_path, filename)
        backbone = re.findall(r"metric_depth_vit_(\w+)_", model_path)[0]
        model, cfg = load_model(f'vit-{backbone}', model_path)
        return cls(model, cfg)

    def to(self, device):
        """Move model to device."""
        self.model.to(device)
        self.device = device
        return self

    def __call__(self, input_image, detect_resolution=512, fx=1000, fy=1000,
                 output_type=None, upscale_method="INTER_CUBIC", depth_and_normal=True, **kwargs):
        """Run inference on input image."""
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)

        depth_map, normal_map, _ = predict_depth_normal(self.model, self.cfg, input_image, fx=fx, fy=fy)
        # ControlNet uses inverse depth and normal
        depth_map, normal_map = depth_map, 255 - normal_map
        depth_map, remove_pad = resize_image_with_pad(depth_map, detect_resolution, upscale_method)
        normal_map, _ = resize_image_with_pad(normal_map, detect_resolution, upscale_method)
        depth_map, normal_map = remove_pad(depth_map), remove_pad(normal_map)

        if output_type == "pil":
            depth_map = Image.fromarray(depth_map)
            normal_map = Image.fromarray(normal_map)

        if depth_and_normal:
            return depth_map, normal_map
        else:
            return depth_map
