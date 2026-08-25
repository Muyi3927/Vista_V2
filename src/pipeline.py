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
import torch

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
    has_alpha = (alpha_mask is not None and np.sum(alpha_mask < 128) > 50)

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
    pre_mask_paths, bg_color, stage3_removal_logs = run_segmentation(
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
        "stage3_removal_logs": stage3_removal_logs,
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
    
    # 释放分阶段 GPU 显存，避免高分辨率下 DiffVG OOM 雪崩导致全部 Mask 为空
    if torch.cuda.is_available():
        import gc
        gc.collect()
        torch.cuda.empty_cache()

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

    # 3. 记录全流程决策日志 (decision_log.json & decision_log.md)：详细记录被移除的掩码、同色吸收原因与阶段4几何剪枝信息
    stage3_logs = seg.get("stage3_removal_logs", [])
    stage4_logs = vec.get("prune_logs", [])

    sam_dedup_logs = [item for item in stage3_logs if item.get("stage") == "stage3_sam_self_dedup"]
    slic_suppress_logs = [item for item in stage3_logs if item.get("stage") == "stage3_slic_suppression"]
    color_absorption_logs = [item for item in stage3_logs if item.get("stage") == "stage3_color_absorption"]

    decision_log = {
        "image_path": image_path,
        "run_dir": run_dir,
        "statistics": {
            "total_stage3_removed": len(stage3_logs),
            "sam_self_dedup_count": len(sam_dedup_logs),
            "slic_suppressed_count": len(slic_suppress_logs),
            "color_absorbed_count": len(color_absorption_logs),
            "stage4_pruned_paths_count": len(stage4_logs),
            "final_retained_shapes": vec.get("shapes", 0),
        },
        "stage3_deduplication_removals": {
            "sam_self_dedup": sam_dedup_logs,
            "slic_cross_suppression": slic_suppress_logs,
            "color_absorption": color_absorption_logs,
        },
        "stage4_geometry_pruning_removals": stage4_logs,
    }
    decision_log_path = os.path.join(run_dir, "decision_log.json")
    with open(decision_log_path, "w", encoding="utf-8") as f:
        json.dump(decision_log, f, indent=2, ensure_ascii=False)

    # 导出便于查阅的 Markdown 决策报告
    try:
        md_lines = [
            f"# VISTA 掩码去重与剪枝决策日志",
            f"- **图像路径**: `{image_path}`",
            f"- **输出目录**: `{run_dir}`",
            f"- **阶段3 去重移除掩码总数**: {len(stage3_logs)} (SAM自去重: {len(sam_dedup_logs)}, SLIC跨界压制: {len(slic_suppress_logs)}, 父级同色吸收: {len(color_absorption_logs)})",
            f"- **阶段4 几何剪枝移除数**: {len(stage4_logs)}",
            f"- **最终保留矢量图层数**: {vec.get('shapes', 0)}",
            "",
            "## 阶段 3 掩码预处理去重详情",
            "",
            "### 1. SAM 内部高重合自去重 (SAM Self-Dedup)",
        ]
        if sam_dedup_logs:
            md_lines.append("| 被移除图层 (Target) | 保留图层 (Kept) | 面积 (px) | IoU | 阈值 | 判定原因 |")
            md_lines.append("|---|---|---|---|---|---|")
            for item in sam_dedup_logs:
                md_lines.append(f"| `{item.get('target_mask_file')}` | `{item.get('kept_mask_file')}` | {item.get('area')} | {item.get('iou')} | {item.get('iou_threshold')} | {item.get('reason')} |")
        else:
            md_lines.append("_无 SAM 自去重移除图层_")
        md_lines.append("")

        md_lines.append("### 2. SAM 纯度感知跨界压制 SLIC (SLIC Cross Suppression)")
        if slic_suppress_logs:
            md_lines.append("| 被移除 SLIC 图层 | 压制方 SAM 图层 | 面积 (px) | IoU / 包含率 | SAM 色彩方差 | 判定原因 |")
            md_lines.append("|---|---|---|---|---|---|")
            for item in slic_suppress_logs:
                metric = f"IoU={item['iou']}" if "iou" in item else f"包含率={item.get('contain_ratio',0)*100:.1f}%"
                md_lines.append(f"| `{item.get('target_mask_file')}` | `{item.get('kept_mask_file')}` | {item.get('area')} | {metric} | {item.get('sam_pure_std')} | {item.get('reason')} |")
        else:
            md_lines.append("_无 SLIC 跨界压制图层_")
        md_lines.append("")

        md_lines.append("### 3. CIELAB 纯度双锁直接父级同色吸收 (Color Absorption)")
        if color_absorption_logs:
            md_lines.append("| 被吸收图层 (Target) | 直接父级 (Parent) | 包含率 | CIELAB 色差 ΔE | ΔE 阈值 | 判定原因 |")
            md_lines.append("|---|---|---|---|---|---|")
            for item in color_absorption_logs:
                md_lines.append(f"| `{item.get('target_mask_file')}` | `{item.get('parent_mask_file')}` | {item.get('contain_ratio',0)*100:.1f}% | {item.get('delta_e')} | {item.get('delta_e_threshold')} | {item.get('reason')} |")
        else:
            md_lines.append("_无同色吸收图层_")
        md_lines.append("")

        md_lines.append("## 阶段 4 矢量化几何剪枝详情")
        if stage4_logs:
            md_lines.append("| Shape ID | 父级 ID | 总面积 (px) | 可见面积 (px) | 色差 ΔE / 透明度 | 剪枝原因 |")
            md_lines.append("|---|---|---|---|---|---|")
            for item in stage4_logs:
                s_id = f"#{item.get('shape_id')}"
                p_id = f"#{item.get('parent_id')}" if item.get('parent_id') is not None else "-"
                area = item.get('total_area', '-')
                vis = item.get('visible_area', '-')
                val = f"Alpha={item['alpha']}" if 'alpha' in item else (f"ΔE={item['delta_e']}" if 'delta_e' in item else "-")
                md_lines.append(f"| {s_id} | {p_id} | {area} | {vis} | {val} | {item.get('reason')} |")
        else:
            md_lines.append("_无阶段 4 剪枝图层_")
        md_lines.append("")

        decision_md_path = os.path.join(run_dir, "decision_log.md")
        with open(decision_md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
    except Exception as e:
        print(f"[Warning] 生成 decision_log.md 异常: {e}")

    # 导出交互式可视化 HTML 报告 (decision_log.html)，支持直接在浏览器中图文对照查看掩码图与裁决依据
    try:
        def _render_mask_img_tag(fname, label):
            if not fname:
                return "<span style='color:#999'>无</span>"
            if fname == "000_bg.png":
                return "<span class='badge bg-secondary'>000_bg (画布背景)</span>"
            return f"""
            <div class='mask-card'>
                <div class='mask-tag'>{label}</div>
                <img src='origin_colored_masks/{fname}' onerror="this.src='origin_masks/{fname}'; this.onerror=null;" alt='{fname}'>
                <div class='mask-name'>{fname}</div>
            </div>
            """

        html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VISTA 去重与剪枝决策可视化报告 - {os.path.basename(image_path)}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f8fafc; color: #1e293b; margin: 0; padding: 24px; }}
        .container {{ max-width: 1280px; margin: 0 auto; background: #ffffff; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); padding: 32px; }}
        h1 {{ font-size: 24px; margin-top: 0; color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 16px; display: flex; align-items: center; justify-content: space-between; }}
        .badge {{ display: inline-block; padding: 4px 10px; border-radius: 6px; font-size: 13px; font-weight: 600; background: #e0e7ff; color: #3730a3; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin: 24px 0; }}
        .stat-card {{ background: #f1f5f9; padding: 16px; border-radius: 8px; border-left: 4px solid #3b82f6; }}
        .stat-card.orange {{ border-left-color: #f97316; }}
        .stat-card.green {{ border-left-color: #10b981; }}
        .stat-card.purple {{ border-left-color: #8b5cf6; }}
        .stat-val {{ font-size: 24px; font-weight: 700; color: #0f172a; margin-top: 4px; }}
        .stat-lbl {{ font-size: 13px; color: #64748b; font-weight: 500; }}
        
        .section-title {{ font-size: 18px; font-weight: 700; margin: 32px 0 16px 0; color: #1e293b; display: flex; align-items: center; gap: 8px; }}
        .section-title::before {{ content: ""; display: inline-block; width: 6px; height: 18px; background: #3b82f6; border-radius: 3px; }}
        
        table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 14px; }}
        th {{ background: #f8fafc; color: #475569; font-weight: 600; text-align: left; padding: 12px 16px; border-bottom: 2px solid #e2e8f0; }}
        td {{ padding: 14px 16px; border-bottom: 1px solid #f1f5f9; vertical-align: middle; }}
        tr:hover td {{ background: #f8fafc; }}
        
        .mask-card {{ display: inline-flex; flex-direction: column; align-items: center; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 6px; width: 140px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
        .mask-card img {{ width: 120px; height: 80px; object-fit: contain; background: #0f172a; border-radius: 4px; }}
        .mask-tag {{ font-size: 10px; font-weight: 700; color: #64748b; margin-bottom: 4px; text-transform: uppercase; }}
        .mask-name {{ font-size: 11px; color: #334155; font-family: monospace; word-break: break-all; margin-top: 4px; text-align: center; }}
        .vs-container {{ display: flex; align-items: center; gap: 12px; }}
        .vs-badge {{ font-size: 12px; font-weight: 700; color: #94a3b8; background: #f1f5f9; padding: 4px 8px; border-radius: 4px; }}
        
        .reason-box {{ background: #fffbeb; border-left: 3px solid #f59e0b; padding: 8px 12px; border-radius: 4px; font-size: 13px; color: #92400e; }}
        .reason-box.blue {{ background: #eff6ff; border-left-color: #3b82f6; color: #1e40af; }}
        .reason-box.purple {{ background: #faf5ff; border-left-color: #a855f7; color: #6b21a8; }}
        .empty-hint {{ color: #94a3b8; font-style: italic; padding: 12px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>
            <span>VISTA 去重与剪枝决策可视化报告</span>
            <span class="badge">{os.path.basename(image_path)}</span>
        </h1>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-lbl">阶段3 预处理去重总数</div>
                <div class="stat-val">{len(stage3_logs)}</div>
            </div>
            <div class="stat-card orange">
                <div class="stat-lbl">SAM 自去重 / SLIC 跨界压制</div>
                <div class="stat-val">{len(sam_dedup_logs)} / {len(slic_suppress_logs)}</div>
            </div>
            <div class="stat-card purple">
                <div class="stat-lbl">阶段3 父级同色吸收</div>
                <div class="stat-val">{len(color_absorption_logs)}</div>
            </div>
            <div class="stat-card green">
                <div class="stat-lbl">阶段4 几何剪枝 / 最终保留图层</div>
                <div class="stat-val">{len(stage4_logs)} / {vec.get('shapes', 0)}</div>
            </div>
        </div>

        <!-- 阶段 3.1 SAM 自去重 -->
        <div class="section-title">1. SAM 内部高重合自去重 (SAM Self-Dedup: {len(sam_dedup_logs)})</div>
        """
        if sam_dedup_logs:
            html_content += """
            <table>
                <thead>
                    <tr>
                        <th style="width:360px">图层对照 (被移除 vs 保留)</th>
                        <th>图层面积</th>
                        <th>IoU 重合度 / 阈值</th>
                        <th>裁决依据与判定原因</th>
                    </tr>
                </thead>
                <tbody>
            """
            for itm in sam_dedup_logs:
                t_tag = _render_mask_img_tag(itm.get("target_mask_file"), "Target (移除)")
                k_tag = _render_mask_img_tag(itm.get("kept_mask_file"), "Kept (保留)")
                html_content += f"""
                    <tr>
                        <td>
                            <div class="vs-container">
                                {t_tag}
                                <span class="vs-badge">VS</span>
                                {k_tag}
                            </div>
                        </td>
                        <td><b>{itm.get('area')}</b> px</td>
                        <td><span class="badge">IoU = {itm.get('iou')}</span> <span style="color:#64748b">(>{itm.get('iou_threshold')})</span></td>
                        <td><div class="reason-box blue">{itm.get('reason')}</div></td>
                    </tr>
                """
            html_content += "</tbody></table>"
        else:
            html_content += "<div class='empty-hint'>无 SAM 自去重移除图层</div>"

        # 阶段 3.2 SLIC 跨界压制
        html_content += f"""
        <!-- 阶段 3.2 SLIC 跨界压制 -->
        <div class="section-title">2. SAM 纯度感知跨界压制 SLIC (SLIC Cross Suppression: {len(slic_suppress_logs)})</div>
        """
        if slic_suppress_logs:
            html_content += """
            <table>
                <thead>
                    <tr>
                        <th style="width:360px">图层对照 (SLIC 被压制 vs SAM 压制方)</th>
                        <th>SLIC 面积</th>
                        <th>重合指标 (IoU / 包含率)</th>
                        <th>SAM 纯度 & 裁决原因</th>
                    </tr>
                </thead>
                <tbody>
            """
            for itm in slic_suppress_logs:
                t_tag = _render_mask_img_tag(itm.get("target_mask_file"), "SLIC (移除)")
                k_tag = _render_mask_img_tag(itm.get("kept_mask_file"), "SAM (保留)")
                metric_str = f"<span class='badge'>IoU = {itm['iou']}</span>" if "iou" in itm else f"<span class='badge'>包含率 = {itm.get('contain_ratio',0)*100:.1f}%</span>"
                html_content += f"""
                    <tr>
                        <td>
                            <div class="vs-container">
                                {t_tag}
                                <span class="vs-badge">VS</span>
                                {k_tag}
                            </div>
                        </td>
                        <td><b>{itm.get('area')}</b> px</td>
                        <td>{metric_str}</td>
                        <td><div class="reason-box">{itm.get('reason')}</div></td>
                    </tr>
                """
            html_content += "</tbody></table>"
        else:
            html_content += "<div class='empty-hint'>无 SLIC 跨界压制图层</div>"

        # 阶段 3.3 同色吸收
        html_content += f"""
        <!-- 阶段 3.3 同色吸收 -->
        <div class="section-title">3. CIELAB 纯度双锁直接父级同色吸收 (Color Absorption: {len(color_absorption_logs)})</div>
        """
        if color_absorption_logs:
            html_content += """
            <table>
                <thead>
                    <tr>
                        <th style="width:360px">图层对照 (被吸收图层 vs 直接父级)</th>
                        <th>包含率</th>
                        <th>CIELAB 色差 ΔE / 阈值</th>
                        <th>裁决原因</th>
                    </tr>
                </thead>
                <tbody>
            """
            for itm in color_absorption_logs:
                t_tag = _render_mask_img_tag(itm.get("target_mask_file"), "Child (被吸收)")
                k_tag = _render_mask_img_tag(itm.get("parent_mask_file"), "Parent (父级)")
                html_content += f"""
                    <tr>
                        <td>
                            <div class="vs-container">
                                {t_tag}
                                <span class="vs-badge">VS</span>
                                {k_tag}
                            </div>
                        </td>
                        <td><span class="badge">{itm.get('contain_ratio',0)*100:.1f}%</span></td>
                        <td><b>ΔE = {itm.get('delta_e')}</b> <span style="color:#64748b">(&lt;{itm.get('delta_e_threshold')})</span></td>
                        <td><div class="reason-box purple">{itm.get('reason')}</div></td>
                    </tr>
                """
            html_content += "</tbody></table>"
        else:
            html_content += "<div class='empty-hint'>无同色吸收图层</div>"

        # 阶段 4 几何剪枝
        def _render_shape_svg_tag(svg_rel_path, label):
            if not svg_rel_path:
                return "<span style='color:#999'>无</span>"
            return f"""
            <div class='mask-card'>
                <div class='mask-tag'>{label}</div>
                <img src='{svg_rel_path}' alt='{label}' style='background: #0f172a;'>
                <div class='mask-name'>{os.path.basename(svg_rel_path)}</div>
            </div>
            """

        html_content += f"""
        <!-- 阶段 4 几何剪枝 -->
        <div class="section-title">4. 阶段 4 矢量化几何剪枝详情 (Stage 4 Geometry Pruning: {len(stage4_logs)})</div>
        """
        if stage4_logs:
            html_content += """
            <table>
                <thead>
                    <tr>
                        <th style="width:360px">图层几何预览 (剪枝 Shape vs 参考父级)</th>
                        <th>Shape 编号</th>
                        <th>总面积 / 可见面积</th>
                        <th>指标 (Alpha / ΔE)</th>
                        <th>剪枝判定原因</th>
                    </tr>
                </thead>
                <tbody>
            """
            for itm in stage4_logs:
                s_id = f"#{itm.get('shape_id')}"
                p_id = f"#{itm.get('parent_id')}" if itm.get('parent_id') is not None else "-"
                area = itm.get('total_area', '-')
                vis = itm.get('visible_area', '-')
                val = f"Alpha={itm['alpha']}" if 'alpha' in itm else (f"ΔE={itm['delta_e']}" if 'delta_e' in itm else "-")
                
                s_svg = _render_shape_svg_tag(itm.get("shape_svg"), f"Shape {s_id}")
                if itm.get("parent_svg"):
                    p_svg = _render_shape_svg_tag(itm.get("parent_svg"), f"Parent {p_id}")
                    shape_preview = f"""
                    <div class="vs-container">
                        {s_svg}
                        <span class="vs-badge">VS</span>
                        {p_svg}
                    </div>
                    """
                else:
                    shape_preview = s_svg

                html_content += f"""
                    <tr>
                        <td>{shape_preview}</td>
                        <td><b>{s_id}</b></td>
                        <td>{area} px / <span style='color:#3b82f6;'>{vis} px</span></td>
                        <td><span class="badge">{val}</span></td>
                        <td><div class="reason-box blue">{itm.get('reason')}</div></td>
                    </tr>
                """
            html_content += "</tbody></table>"
        else:
            html_content += "<div class='empty-hint'>无阶段 4 剪枝图层</div>"

        html_content += """
    </div>
</body>
</html>
        """
        decision_html_path = os.path.join(run_dir, "decision_log.html")
        with open(decision_html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
    except Exception as e:
        print(f"[Warning] 生成 decision_log.html 异常: {e}")

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
        "decision_log": decision_log,
        "total_time_sec": round(seg["time_sec"] + vec.get("time_consuming", 0), 4),
    }

    result_json_path = os.path.join(run_dir, "result.json")
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary
