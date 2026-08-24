"""
VISTA Web API 后端封装模块 (app_main.py)
提供 `img_to_svg_full` 接口，支持 Web 端实时参数覆盖与全景矢量化处理，返回各阶段全景图与图层列表。
"""
from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from config import get_config, load_config
from pipeline import process_single_image


def img_to_svg_full(
    image_path: str,
    # 基础与预处理
    target_size: Optional[int] = None,
    # SLIC 超像素
    slic_n_segments: Optional[int] = None,
    slic_compactness: Optional[float] = None,
    dbscan_eps: Optional[float] = None,
    # SAM 语义分割
    use_sam: Optional[bool] = None,
    pred_iou_thresh: Optional[float] = None,
    stability_score_thresh: Optional[float] = None,
    crop_n_layers: Optional[int] = None,
    points_per_side: Optional[int] = None,
    # 层级融合与吸收
    min_area: Optional[float] = None,
    iou_sam_slic_thresh: Optional[float] = None,
    iou_sam_internal_thresh: Optional[float] = None,
    parent_contain_thresh: Optional[float] = None,
    self_pure_std_thresh: Optional[float] = None,
    parent_pure_std_thresh: Optional[float] = None,
    color_diff_thresh: Optional[float] = None,
    # 几何拟合
    bzer_max_error: Optional[float] = None,
    line_threshold: Optional[float] = None,
    poly_epsilon: Optional[float] = None,
    # DiffVG 优化
    learning_rate: Optional[float] = None,
    color_lr: Optional[float] = None,
    num_iters: Optional[int] = None,
    early_stopping_patience: Optional[int] = None,
    collinear_scale: Optional[float] = None,
    is_stroke: Optional[bool] = None,
    # 剪枝与精修
    prune_enabled: Optional[bool] = None,
    rm_color_threshold: Optional[float] = None,
    refine_iters: Optional[int] = None,
) -> Dict[str, Any]:
    """
    单张图像 Web 端矢量化入口：支持全部阶段超参数灵活覆盖。
    """
    cfg = get_config()
    overrides: Dict[str, Any] = {}

    if target_size is not None:
        overrides.setdefault("preprocess", {})["target_size"] = int(target_size)
    if min_area is not None:
        overrides.setdefault("preprocess", {})["min_area"] = float(min_area)
    if color_diff_thresh is not None:
        overrides.setdefault("preprocess", {})["color_diff_thresh"] = float(color_diff_thresh)
    if iou_sam_slic_thresh is not None:
        overrides.setdefault("preprocess", {})["iou_sam_slic_thresh"] = float(iou_sam_slic_thresh)
    if iou_sam_internal_thresh is not None:
        overrides.setdefault("preprocess", {})["iou_threshold"] = float(iou_sam_internal_thresh)
    if parent_contain_thresh is not None:
        overrides.setdefault("preprocess", {})["inclusion_threshold"] = float(parent_contain_thresh)
    if self_pure_std_thresh is not None:
        overrides.setdefault("preprocess", {})["self_pure_std_thresh"] = float(self_pure_std_thresh)
    if parent_pure_std_thresh is not None:
        overrides.setdefault("preprocess", {})["parent_pure_std_thresh"] = float(parent_pure_std_thresh)

    if slic_n_segments is not None:
        overrides.setdefault("slic", {})["n_segments"] = int(slic_n_segments)
    if slic_compactness is not None:
        overrides.setdefault("slic", {})["compactness"] = float(slic_compactness)
    if dbscan_eps is not None:
        overrides.setdefault("slic", {})["dbscan_eps"] = float(dbscan_eps)

    if use_sam is not None:
        overrides.setdefault("sam", {})["enabled"] = bool(use_sam)
    if pred_iou_thresh is not None:
        overrides.setdefault("sam", {})["pred_iou_thresh"] = float(pred_iou_thresh)
    if stability_score_thresh is not None:
        overrides.setdefault("sam", {})["stability_score_thresh"] = float(stability_score_thresh)
    if crop_n_layers is not None:
        overrides.setdefault("sam", {})["crop_n_layers"] = int(crop_n_layers)
    if points_per_side is not None:
        overrides.setdefault("sam", {})["points_per_side"] = int(points_per_side)

    if bzer_max_error is not None:
        overrides.setdefault("path_fit", {})["bezier_max_error"] = float(bzer_max_error)
    if line_threshold is not None:
        overrides.setdefault("path_fit", {})["line_threshold"] = float(line_threshold)
    if poly_epsilon is not None:
        overrides.setdefault("path_fit", {})["poly_epsilon"] = float(poly_epsilon)

    if learning_rate is not None:
        overrides.setdefault("optimize", {})["learning_rate"] = float(learning_rate)
    if color_lr is not None:
        overrides.setdefault("optimize", {})["color_lr"] = float(color_lr)
    if is_stroke is not None:
        overrides.setdefault("optimize", {})["is_stroke"] = bool(is_stroke)
    if num_iters is not None:
        overrides.setdefault("optimize", {})["num_iters"] = int(num_iters)
    if early_stopping_patience is not None:
        overrides.setdefault("optimize", {})["early_stopping_patience"] = int(early_stopping_patience)
    if collinear_scale is not None:
        overrides.setdefault("optimize", {})["collinear_scale"] = float(collinear_scale)

    if prune_enabled is not None:
        overrides.setdefault("prune", {})["enabled"] = bool(prune_enabled)
    if rm_color_threshold is not None:
        overrides.setdefault("prune", {})["rm_color_threshold"] = float(rm_color_threshold)
    if refine_iters is not None:
        overrides.setdefault("prune", {})["refine_iters"] = int(refine_iters)

    temp_dir = cfg["project"]["temp_outputs"]
    os.makedirs(temp_dir, exist_ok=True)

    run_cfg = load_config(overrides=overrides) if overrides else cfg

    summary = process_single_image(
        image_path,
        base_out_dir=temp_dir,
        cfg=run_cfg,
    )

    run_dir = summary["run_dir"]
    vec = summary.get("vectorize", {})
    seg = summary.get("segment", {})

    svg_path = vec.get("svg_path", os.path.join(run_dir, "final.svg"))
    gif_path = vec.get("gif_path", os.path.join(run_dir, "animation.gif"))

    # 收集各阶段生成的可视化全景图 (三大阶段: raw -> origin -> pre)
    slic_ov = os.path.join(run_dir, "slic_overview_colored.png")
    sam_ov = os.path.join(run_dir, "sam_overview_colored.png")
    origin_ov = os.path.join(run_dir, "origin_overview_colored.png")
    pre_ov = os.path.join(run_dir, "pre_overview_colored.png")

    overviews = {
        "slic": slic_ov if os.path.isfile(slic_ov) else "",
        "sam": sam_ov if os.path.isfile(sam_ov) else "",
        "origin": origin_ov if os.path.isfile(origin_ov) else "",
        "pre": pre_ov if os.path.isfile(pre_ov) else "",
    }

    # 收集 pre_masks 列表
    pre_masks_dir = os.path.join(run_dir, "pre_masks")
    pre_colored_dir = os.path.join(run_dir, "pre_colored_masks")
    layer_items = []
    if os.path.isdir(pre_masks_dir):
        for fname in sorted(os.listdir(pre_masks_dir)):
            if fname.endswith(".png"):
                bw_p = os.path.join(pre_masks_dir, fname)
                col_p = os.path.join(pre_colored_dir, fname) if os.path.isdir(pre_colored_dir) else ""
                layer_items.append({
                    "name": fname,
                    "bw_path": bw_p,
                    "colored_path": col_p if os.path.isfile(col_p) else bw_p,
                })

    meta_file = os.path.join(run_dir, "pre_masks_meta.json")
    meta_info = {}
    if os.path.isfile(meta_file):
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta_info = json.load(f)
        except Exception:
            pass

    return {
        "run_dir": run_dir,
        "svg_path": svg_path,
        "gif_path": gif_path,
        "overviews": overviews,
        "layers": layer_items,
        "meta": meta_info,
        "stats": {
            "total_time_sec": summary.get("total_time_sec", 0.0),
            "segment_time_sec": seg.get("time_sec", 0.0),
            "vectorize_time_sec": vec.get("time_sec", 0.0),
            "shapes": vec.get("shapes", 0),
            "path_point_nums": vec.get("path_point_nums", 0),
            "mse_loss": vec.get("mse_loss", 0.0),
            "num_masks": seg.get("num_masks", 0),
        },
    }
