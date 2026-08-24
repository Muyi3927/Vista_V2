"""
VISTA 配置加载器与校验模块：
从 YAML 读取全局超参数，自动解析计算设备与绝对路径。
"""
from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Dict, Optional

import torch
import yaml

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_ROOT = os.path.abspath(os.path.join(_SRC_DIR, ".."))
_DEFAULT_YAML = os.path.join(_DEFAULT_ROOT, "config", "default.yaml")

_CFG: Dict[str, Any] = {}


def _deep_update(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _resolve_device(spec: str) -> torch.device:
    """自动解析计算设备：优先 GPU，不可用时回退至 CPU。"""
    if not spec or spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        print("[设备] CUDA 不可用，自动切换至 CPU 运算。")
        return torch.device("cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(f"[设备] 请求使用 {spec} 但 CUDA 不可用，自动切换至 CPU。")
        return torch.device("cpu")
    return device


def _abspath(root: str, path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(root, path))


def load_config(path: Optional[str] = None, overrides: Optional[dict] = None) -> Dict[str, Any]:
    """加载 YAML 配置文件，解析路径，应用动态覆盖参数。"""
    global _CFG
    cfg_path = path or _DEFAULT_YAML
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"未找到配置文件: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = deepcopy(raw)
    if overrides:
        _deep_update(cfg, overrides)

    root = (cfg.get("project") or {}).get("root") or _DEFAULT_ROOT
    root = os.path.abspath(root)
    cfg.setdefault("project", {})["root"] = root

    paths = cfg.setdefault("paths", {})
    if not paths.get("input") and paths.get("data"):
        paths["input"] = paths["data"]

    for key in ("input", "data", "output", "sam_checkpoint"):
        if key in paths and paths[key]:
            paths[key] = _abspath(root, paths[key])

    run = cfg.setdefault("run", {})
    run.setdefault("mode", "auto")

    # SAM 默认参数
    sam = cfg.setdefault("sam", {})
    sam.setdefault("enabled", True)
    sam.setdefault("model_type", "vit_h")
    sam.setdefault("pred_iou_thresh", 0.88)
    sam.setdefault("stability_score_thresh", 0.95)
    sam.setdefault("crop_n_layers", 1)
    sam.setdefault("points_per_side", 64)
    sam.setdefault("min_mask_region_area", 0)

    # SLIC 默认参数
    slic = cfg.setdefault("slic", {})
    slic.setdefault("enabled", True)
    slic.setdefault("cell_size", 14.0)
    slic.setdefault("min_segments", 800)
    slic.setdefault("max_segments", 8000)
    slic.setdefault("n_segments", 2000)
    slic.setdefault("compactness", 15.0)
    slic.setdefault("dbscan_eps", 6.0)
    slic.setdefault("use_lab", True)
    slic.setdefault("enforce_connectivity", True)

    # 预处理与掩码融合默认参数
    pre = cfg.setdefault("preprocess", {})
    pre.setdefault("target_size", 0)
    pre.setdefault("denoise", False)
    pre.setdefault("denoise_sigma_color", 35.0)
    pre.setdefault("denoise_sigma_space", 35.0)
    pre.setdefault("composite_bg_color", None)
    pre.setdefault("use_alpha_mask", True)
    pre.setdefault("transparent_svg", None)
    pre.setdefault("min_area", 0.00015)
    pre.setdefault("iou_sam_slic_thresh", 0.85)
    pre.setdefault("iou_threshold", 0.90)
    pre.setdefault("sam_pure_suppress_std", 10.0)
    pre.setdefault("slic_contain_suppress_thresh", 0.85)
    pre.setdefault("inclusion_threshold", 0.90)
    pre.setdefault("self_pure_std_thresh", 15.0)
    pre.setdefault("parent_pure_std_thresh", 3.0)
    pre.setdefault("color_diff_thresh", 5.0)

    # 路径拟合默认参数
    pf = cfg.setdefault("path_fit", {})
    pf.setdefault("bezier_max_error", 0.003)
    pf.setdefault("line_threshold", 0.004)
    if pf.get("poly_epsilon") is None:
        pf["poly_epsilon"] = None
    pf.setdefault("contour_min_dist", 0.004)

    # 优化超参数
    opt = cfg.setdefault("optimize", {})
    opt.setdefault("learning_rate", 0.1)
    opt.setdefault("color_lr", 0.01)
    opt.setdefault("stroke_width_lr", 0.05)
    opt.setdefault("stroke_color_lr", 0.01)
    opt.setdefault("num_iters", 1000)
    opt.setdefault("is_stroke", False)
    opt.setdefault("early_stopping_patience", 20)
    opt.setdefault("early_stopping_delta", 5.0e-5)
    opt.setdefault("collinear_scale", 0.01)
    opt.setdefault("collinear_cos_threshold", 0.5)
    opt.setdefault("save_every", 5)
    opt.setdefault("frame_every", 5)

    # 剪枝与精修参数
    prune = cfg.setdefault("prune", {})
    prune.setdefault("enabled", True)
    prune.setdefault("rm_color_threshold", 0.02)
    prune.setdefault("inclusion_threshold", 0.8)
    prune.setdefault("refine_iters", 100)
    prune.setdefault("raster_threshold", 0.5)

    # 输出保存开关 (三大阶段: raw -> origin -> pre)
    save_opts = cfg.setdefault("save_options", {})
    save_opts.setdefault("save_raw_masks", True)
    save_opts.setdefault("save_origin_masks", True)
    save_opts.setdefault("save_pre_masks", True)
    save_opts.setdefault("save_color_mask", True)
    save_opts.setdefault("save_overview_images", True)
    save_opts.setdefault("save_step_svgs", True)

    temp = (cfg.get("project") or {}).get("temp_outputs", "temp_outputs")
    cfg["project"]["temp_outputs"] = _abspath(root, temp)

    cfg["_resolved_device"] = _resolve_device(cfg.get("device", "auto"))
    _CFG = cfg
    _export_legacy_constants(cfg)
    return cfg


def get_config() -> Dict[str, Any]:
    """获取当前已加载的配置字典。"""
    if not _CFG:
        return load_config()
    return _CFG


def _export_legacy_constants(cfg: Dict[str, Any]) -> None:
    """向全局变量导出常量以兼容既有导入代码。"""
    g = globals()
    root = cfg["project"]["root"]
    paths = cfg["paths"]
    sam = cfg["sam"]
    pre = cfg["preprocess"]
    pf = cfg["path_fit"]
    opt = cfg["optimize"]
    prune = cfg["prune"]

    g["PROJECT_PATH"] = root if root.endswith(os.sep) else root + os.sep
    g["DATA_PATH"] = paths.get("input", "")
    g["TEMP_OUTPUTS_DIR"] = cfg["project"]["temp_outputs"]
    g["CHECKPOINT_PATH"] = paths.get("sam_checkpoint", "")
    g["MODEL_TYPE"] = sam.get("model_type", "vit_h")
    g["DEVICE"] = cfg["_resolved_device"]
    g["TARGET_SIZE"] = int(pre.get("target_size", 0))
    g["PREDICTION_IOU_THRESHOLD"] = float(sam.get("pred_iou_thresh", 0.88))
    g["STABILITY_SCORE_THRESHOLD"] = float(sam.get("stability_score_thresh", 0.95))
    g["CROP_N_LAYERS"] = int(sam.get("crop_n_layers", 1))
    g["MIN_AREA"] = pre.get("min_area", 0.00015)
    g["BEZIER_MAX_ERROR"] = pf.get("bezier_max_error", 0.003)
    g["LINE_THRESHOLD"] = pf.get("line_threshold", 0.004)
    g["LEARNING_RATE"] = float(opt.get("learning_rate", 0.1))
    g["NUM_ITERS"] = int(opt.get("num_iters", 1000))
    g["IS_STROKE"] = bool(opt.get("is_stroke", False))


# 导入模块时自动加载默认配置
load_config()
