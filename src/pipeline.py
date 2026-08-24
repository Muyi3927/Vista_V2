"""
VISTA 端到端矢量化流水线调度模块 (pipeline.py)
阶段解耦：
1. create_job       : 图像加载与预处理 (utils.py) -> 运行目录与 target_img/ 准备
2. stage_segment    : 语义与超像素混合分割融合 (segmentation.py) -> pre_masks/
3. stage_vectorize  : 贝塞尔拟合与 DiffVG 可微渲染优化 (vectorize.py) -> final.svg / animation.gif
"""
from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import get_config, load_config
from segmentation import run_segmentation, run_segmentation_late_fill
from utils import load_and_resize, save_target_image
from vectorize import run_vectorization


def _deep_update(base: dict, override: dict) -> dict:
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def merge_cfg(overrides: Optional[dict] = None, base: Optional[dict] = None) -> dict:
    cfg = deepcopy(base or get_config())
    if overrides:
        _deep_update(cfg, overrides)
    return cfg


# ==============================================================================
# 流水线阶段定义
# ==============================================================================

def create_job(
    image_path: str,
    base_out_dir: Optional[str] = None,
    cfg: Optional[Dict[str, Any]] = None,
    unique_subdir: bool = True,
) -> Dict[str, Any]:
    """
    阶段 0：初始化运行目录并进行图像预处理与尺寸调整。
    """
    cfg = cfg or get_config()
    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_root = base_out_dir or cfg.get("paths", {}).get("output") or os.path.join(cfg["project"]["root"], "out", "run")

    if unique_subdir:
        uid = uuid.uuid4().hex[:6]
        run_dir = os.path.join(out_root, f"{stem}_{uid}")
    else:
        run_dir = os.path.join(out_root, stem)

    os.makedirs(run_dir, exist_ok=True)
    target_img_dir = os.path.join(run_dir, "target_img")
    os.makedirs(target_img_dir, exist_ok=True)

    pre_cfg = cfg.get("preprocess", {})
    target_size = int(pre_cfg.get("target_size", 0))
    denoise = bool(pre_cfg.get("denoise", False))
    denoise_sigma_color = float(pre_cfg.get("denoise_sigma_color", 35.0))
    denoise_sigma_space = float(pre_cfg.get("denoise_sigma_space", 35.0))
    bg_color_cfg = pre_cfg.get("composite_bg_color", None)
    if bg_color_cfg is not None and isinstance(bg_color_cfg, (list, tuple)) and len(bg_color_cfg) >= 3:
        bg_color = tuple([int(c) for c in bg_color_cfg[:3]])
    else:
        bg_color = None

    target_pil, alpha_mask = load_and_resize(
        image_path,
        target_size=target_size,
        bg_color=bg_color,
        denoise=denoise,
        denoise_sigma_color=denoise_sigma_color,
        denoise_sigma_space=denoise_sigma_space,
        return_alpha_mask=True,
    )
    target_path = save_target_image(target_pil, target_img_dir, os.path.basename(image_path))
    target_rgb = np.array(target_pil)
    has_alpha = (alpha_mask is not None and np.sum(alpha_mask < 250) > 0)

    return {
        "run_dir": run_dir,
        "image_path": image_path,
        "target_path": target_path,
        "target_rgb": target_rgb,
        "image_name": stem,
        "alpha_mask": alpha_mask,
        "has_alpha": has_alpha,
    }


def stage_segment(
    run_dir: str,
    target_rgb: np.ndarray,
    cfg: Optional[Dict[str, Any]] = None,
    preloaded_model=None,
    alpha_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    阶段 1：运行 SAM + SLIC 候选分割、孔洞提取、连通拆分与分层融合。
    """
    cfg = cfg or get_config()
    device = cfg.get("_resolved_device")

    st = time.time()
    pre_mask_paths, bg_color = run_segmentation(
        image_rgb=target_rgb,
        output_dir=run_dir,
        cfg=cfg,
        preloaded_model=preloaded_model,
        device=device,
        alpha_mask=alpha_mask,
    )
    elapsed = time.time() - st

    return {
        "stage": "segment",
        "pre_masks": pre_mask_paths,
        "num_masks": len(pre_mask_paths),
        "bg_color": bg_color,
        "time_sec": round(elapsed, 4),
    }


def stage_vectorize(
    run_dir: str,
    target_rgb: np.ndarray,
    pre_mask_paths: List[str],
    bg_color: Tuple[int, int, int],
    cfg: Optional[Dict[str, Any]] = None,
    final_out_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    阶段 2：初始 SVG 路径拟合与 DiffVG 可微渲染优化。
    """
    cfg = cfg or get_config()
    device = cfg.get("_resolved_device")

    vec_result = run_vectorization(
        target_rgb=target_rgb,
        pre_mask_paths=pre_mask_paths,
        bg_color=bg_color,
        output_dir=run_dir,
        cfg=cfg,
        device=device,
    )

    svg_path = vec_result.get("svg_path")
    if svg_path and os.path.isfile(svg_path):
        top_svg = os.path.join(run_dir, "final.svg")
        if not os.path.isfile(top_svg):
            shutil.copy2(svg_path, top_svg)
        if final_out_dir:
            os.makedirs(final_out_dir, exist_ok=True)
            final_dst = os.path.join(final_out_dir, f"{os.path.basename(run_dir)}.svg")
            shutil.copy2(svg_path, final_dst)

    gif_path = vec_result.get("gif_path")
    if gif_path and os.path.isfile(gif_path) and final_out_dir:
        final_gif = os.path.join(final_out_dir, f"{os.path.basename(run_dir)}.gif")
        shutil.copy2(gif_path, final_gif)

    return vec_result


def process_single_image(
    image_path: str,
    base_out_dir: Optional[str] = None,
    final_out_dir: Optional[str] = None,
    cfg: Optional[Dict[str, Any]] = None,
    preloaded_model=None,
) -> Dict[str, Any]:
    """
    单张图像完整端到端矢量化流程：
    预处理 -> SAM+SLIC 分割与融合 -> DiffVG 矢量化优化 -> 结果归档
    """
    cfg = cfg or get_config()
    job = create_job(image_path, base_out_dir=base_out_dir, cfg=cfg)
    run_dir = job["run_dir"]
    target_rgb = job["target_rgb"]

    # 1. 语义与超像素分割与融合
    seg = stage_segment(
        run_dir,
        target_rgb=target_rgb,
        cfg=cfg,
        preloaded_model=preloaded_model,
        alpha_mask=job.get("alpha_mask"),
    )

    # 2. 贝塞尔拟合与可微优化
    # 若配置未强制指定 transparent_svg (即为 auto/null)，则根据原图是否为透明图自适应决定：
    # 输入有透明底(无背景) -> 导出无背景透明 SVG；输入是实心RGB(有背景) -> 导出保留背景 SVG
    vec_cfg = deepcopy(cfg)
    user_trans = (cfg.get("preprocess") or {}).get("transparent_svg")
    if user_trans is None or str(user_trans).lower() == "auto":
        vec_cfg.setdefault("preprocess", {})["transparent_svg"] = bool(job.get("has_alpha", False))
    else:
        vec_cfg.setdefault("preprocess", {})["transparent_svg"] = bool(user_trans)

    vec = stage_vectorize(
        run_dir,
        target_rgb=target_rgb,
        pre_mask_paths=seg["pre_masks"],
        bg_color=seg["bg_color"],
        cfg=vec_cfg,
        final_out_dir=final_out_dir,
    )

    summary = {
        "run_dir": run_dir,
        "image_path": image_path,
        "segment": {
            "num_masks": seg["num_masks"],
            "time_sec": seg["time_sec"],
        },
        "vectorize": {
            "shapes": vec.get("shapes"),
            "path_point_nums": vec.get("path_point_nums"),
            "mse_loss": vec.get("mse_loss"),
            "time_sec": vec.get("time_consuming"),
            "svg_path": vec.get("svg_path"),
            "gif_path": vec.get("gif_path"),
        },
        "total_time_sec": round(seg["time_sec"] + vec.get("time_consuming", 0), 4),
    }

    result_json_path = os.path.join(run_dir, "result.json")
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary
