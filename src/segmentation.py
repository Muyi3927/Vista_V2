"""
VISTA 语义与超像素混合分割模块 (segmentation.py)
包含：
- 第一阶段：SAM 与 SLIC 原始分割候选提取（严格保留内部空洞）
- 第二阶段：闭合孔洞实心化、自适应形态学去毛刺与单连通组件拆解
- 第三阶段：SAM 内部 NMS 去重、SAM 跨界压制 SLIC、原生掩码压制空洞、纯度保护机制下的直接父级同色吸收
"""
from __future__ import annotations

from collections import deque
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from skimage import segmentation
from sklearn.cluster import DBSCAN
import torch

from utils import (
    get_canvas_background_color,
    resolve_min_area,
)

# ==============================================================================
# SAM 模型加载与 NMS 设备补丁
# ==============================================================================
import segment_anything.automatic_mask_generator as _sam_amg

_orig_batched_nms = _sam_amg.batched_nms


def _batched_nms_same_device(boxes, scores, idxs, iou_threshold):
    """确保在调用 torchvision batched_nms 时所有输入张量位于同一设备。"""
    device = boxes.device
    if scores.device != device:
        scores = scores.to(device)
    if idxs.device != device:
        idxs = idxs.to(device)
    return _orig_batched_nms(boxes, scores, idxs, iou_threshold)


_sam_amg.batched_nms = _batched_nms_same_device

_SAM_MODEL_CACHE: Dict[Tuple[str, str, str], Any] = {}


def get_sam_model(model_type: str = "vit_h", checkpoint_path: str = "", device: Optional[torch.device] = None):
    """SAM 模型单例缓存管理器，避免重复加载权重至显存。"""
    from segment_anything import sam_model_registry

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif not isinstance(device, torch.device):
        device = torch.device(device)

    key = (model_type, os.path.abspath(checkpoint_path) if checkpoint_path else "", str(device))
    if key not in _SAM_MODEL_CACHE:
        if not checkpoint_path or not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"未找到 SAM 权重文件: {checkpoint_path}")
        print(f"[SAM] 正在加载 SAM ({model_type}) 模型权重至 {device}...")
        model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        model.to(device=device)
        model.eval()
        _SAM_MODEL_CACHE[key] = model
        print("[SAM] 模型加载成功并已缓存。")
    return _SAM_MODEL_CACHE[key]


# ==============================================================================
# 掩码基础分析与辅助保存
# ==============================================================================

def _compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """计算两个二值掩码之间的交并比 (IoU)。"""
    m1 = mask1 > 0
    m2 = mask2 > 0
    inter = np.logical_and(m1, m2).sum()
    union = m1.sum() + m2.sum() - inter
    return float(inter) / float(union) if union > 0 else 0.0


def _get_dominant_color_and_homogeneity(image_rgb: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, float]:
    """计算掩码区域内的中位数代表色与各通道平均标准差 std（纯度）。"""
    pixels = image_rgb[mask > 0].astype(np.float32)
    if len(pixels) == 0:
        return np.zeros(3, dtype=np.uint8), 999.0
    median_color = np.median(pixels, axis=0)
    std_val = float(pixels.std(axis=0).mean())
    return median_color.astype(np.uint8), std_val


def _fill_holes(mask_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    带边框保护的孔洞填充算法：
    外围增加 1-pixel padding 确保边界处开放式背景连通至 (0,0)；
    只有内部真正封闭的空洞才会被识别为 holes。
    返回：(填实后的掩码 filled, 被提取的孔洞掩码 holes)
    """
    h, w = mask_u8.shape[:2]
    padded = np.pad(mask_u8, pad_width=1, mode="constant", constant_values=0)
    flood = padded.copy()
    fill_mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(flood, fill_mask, (0, 0), 255)
    flood_cropped = flood[1:-1, 1:-1]
    holes = (flood_cropped == 0).astype(np.uint8) * 255
    filled = cv2.bitwise_or(mask_u8, holes)
    return filled, holes


def _smart_morphology_preprocess(region_u8: np.ndarray, orig_area: int) -> Tuple[np.ndarray, int]:
    """自适应形态学平滑：根据区域面积动态选择核大小去毛刺，保护微小细节。"""
    if orig_area <= 150:
        return region_u8, 0
    elif orig_area < 500:
        k_size = 3
    elif orig_area < 5000:
        k_size = 5
    elif orig_area < 25000:
        k_size = 7
    else:
        k_size = 9

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    smoothed = cv2.morphologyEx(region_u8, cv2.MORPH_CLOSE, kernel)
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)
    smoothed = cv2.GaussianBlur(smoothed, (k_size, k_size), 0)
    _, smoothed = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)

    if int((smoothed > 0).sum()) <= 20:
        return region_u8, 0
    return smoothed, k_size


def _save_mask_item(
    mask_u8: np.ndarray,
    color_rgb: List[int],
    out_dir: Optional[Path],
    colored_dir: Optional[Path],
    filename: str,
    export_size: int,
    native_shape: Tuple[int, int],
    save_color_mask: bool,
):
    """保存单张黑白二值掩码以及对应纯色真彩掩码。"""
    h, w = native_shape
    out_w, out_h = w, h
    if export_size and export_size > 0:
        scale = export_size / max(w, h)
        out_w, out_h = int(w * scale), int(h * scale)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        pil_bw = Image.fromarray(mask_u8)
        if (out_w, out_h) != (w, h):
            pil_bw = pil_bw.resize((out_w, out_h), Image.Resampling.NEAREST)
        pil_bw.save(out_dir / filename)

    if save_color_mask and colored_dir is not None:
        colored_dir.mkdir(parents=True, exist_ok=True)
        color_canvas = np.zeros((h, w, 3), dtype=np.uint8)
        color_canvas[mask_u8 > 0] = color_rgb
        pil_col = Image.fromarray(color_canvas)
        if (out_w, out_h) != (w, h):
            pil_col = pil_col.resize((out_w, out_h), Image.Resampling.LANCZOS)
        pil_col.save(colored_dir / filename)


def _save_overview_colored(
    proposals: List[Dict],
    out_path: Path,
    native_shape: Tuple[int, int],
    export_size: int = 0,
):
    """按面积从大到小绘制所有图层叠加的全景彩色效果图。"""
    h, w = native_shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    sorted_p = sorted(proposals, key=lambda x: x["area"], reverse=True)
    for p in sorted_p:
        canvas[p["mask_image"] > 0] = p["fill_color"]

    out_w, out_h = w, h
    if export_size and export_size > 0:
        scale = export_size / max(w, h)
        out_w, out_h = int(w * scale), int(h * scale)

    pil_img = Image.fromarray(canvas)
    if (out_w, out_h) != (w, h):
        pil_img = pil_img.resize((out_w, out_h), Image.Resampling.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(out_path)


# ==============================================================================
# 第一阶段：原始候选提取（SAM + SLIC）
# ==============================================================================

def _close_contour_defects(mask_u8: np.ndarray, max_defect_depth: float = 25.0) -> np.ndarray:
    """
    边缘多边形大坑/深凹槽几何平滑修复：
    利用轮廓凸包缺陷 (Convexity Defects) 找出由于超像素缺失造成的非自然锐角大坑并连接填平。
    """
    h, w = mask_u8.shape[:2]
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fixed_mask = mask_u8.copy()

    for cnt in contours:
        if len(cnt) < 5 or cv2.contourArea(cnt) < 200:
            continue
        hull = cv2.convexHull(cnt, returnPoints=False)
        if hull is None or len(hull) < 3:
            continue
        try:
            defects = cv2.convexityDefects(cnt, hull)
            if defects is None:
                continue
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                depth = d / 256.0  # 真实物理像素深度
                # 如果是狭长且具有一定深度的多边形掉块缺口（典型超像素掉块）
                if 4.0 <= depth <= max_defect_depth:
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[f][0])
                    # 计算开口跨度
                    span = np.linalg.norm(np.array(start) - np.array(end))
                    if span < depth * 3.5:  # 具有坑洞形态特征
                        pts = np.array([start, far, end], dtype=np.int32)
                        cv2.fillPoly(fixed_mask, [pts], 255)
        except Exception:
            pass

    return fixed_mask


def get_raw_slic_proposals(
    image_rgb: np.ndarray,
    output_sub_dir: Optional[Path],
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    提取 SLIC 超像素候选：
    1. 依据图像分辨率自适应计算超像素密度（保持单个网格物理尺寸在合适范围）；
    2. 使用 CIELAB 均匀感知色彩空间进行 DBSCAN 聚类；
    3. 结合超像素级大坑几何修复与自适应闭运算填平边缘缝隙。
    """
    h, w = image_rgb.shape[:2]
    slic_cfg = cfg.get("slic", {})
    cell_size = float(slic_cfg.get("cell_size", 14.0))
    min_segments = int(slic_cfg.get("min_segments", 800))
    max_segments = int(slic_cfg.get("max_segments", 8000))

    if cell_size > 0:
        auto_segments = int((h * w) / (cell_size ** 2))
        n_segments = int(np.clip(auto_segments, min_segments, max_segments))
    else:
        n_segments = int(cfg.get("slic_n_segments", slic_cfg.get("n_segments", 2000)))

    compactness = float(cfg.get("slic_compactness", slic_cfg.get("compactness", 15.0)))
    dbscan_eps = float(cfg.get("dbscan_eps", slic_cfg.get("dbscan_eps", 6.0)))
    use_lab = bool(slic_cfg.get("use_lab", True))
    enforce_conn = bool(slic_cfg.get("enforce_connectivity", True))
    min_area = cfg.get("min_area", cfg.get("preprocess", {}).get("min_area", 0.00015))
    save_raw_masks = bool(cfg.get("save_raw_masks", cfg.get("save_options", {}).get("save_raw_masks", True)))
    save_color_mask = bool(cfg.get("save_color_mask", cfg.get("save_options", {}).get("save_color_mask", True)))

    print(f"[Stage 1 - SLIC] 自适应分辨率 SLIC (分辨率={w}x{h}, n_segments={n_segments}, compactness={compactness})...")
    
    # 转换为 CIELAB 空间以获得符合人类视觉一致性的色彩感知距离
    if use_lab:
        img_for_slic = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    else:
        img_for_slic = image_rgb

    segments = segmentation.slic(
        img_for_slic,
        n_segments=n_segments,
        compactness=compactness,
        start_label=1,
        enforce_connectivity=enforce_conn,
    )
    labels = np.unique(segments)
    
    # 在 LAB 色彩空间下计算各超像素块均值并进行 DBSCAN 聚类
    avg_colors_lab = np.array([img_for_slic[segments == label].mean(axis=0) for label in labels])

    print(f"[Stage 1 - SLIC] CIELAB DBSCAN 聚类 (eps={dbscan_eps})...")
    db = DBSCAN(eps=dbscan_eps, min_samples=1).fit(avg_colors_lab)
    cluster_labels = db.labels_
    unique_clusters = np.unique(cluster_labels)

    proposals = []
    min_area_px = resolve_min_area(min_area, h * w)

    # 依据分辨率动态计算闭运算填坑核大小 (例如 512 分辨率下为 3，2000 分辨率下为 7)
    k_close = max(3, int(max(h, w) / 300) | 1)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))

    for cluster_id in unique_clusters:
        if cluster_id == -1:
            continue
        mask = np.zeros((h, w), dtype=np.uint8)
        for i, label_val in enumerate(labels):
            if cluster_labels[i] == cluster_id:
                mask[segments == label_val] = 255

        # 1. 形态学闭运算：填平超像素网格拼合带来的毛刺和微小缝隙
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        # 2. 几何凹陷修复：填平多边形超像素掉块产生的大坑
        mask = _close_contour_defects(mask, max_defect_depth=max(15.0, cell_size * 1.5))

        area = int((mask > 0).sum())
        if area >= min_area_px:
            mean_col, std_val = _get_dominant_color_and_homogeneity(image_rgb, mask)
            proposals.append({
                "mask_image": mask,
                "area": area,
                "fill_color": mean_col.tolist(),
                "homogeneity_std": std_val,
                "source": "slic",
            })

    proposals = sorted(proposals, key=lambda x: x["area"], reverse=True)
    for idx, item in enumerate(proposals):
        item["orig_idx"] = idx

    if output_sub_dir is not None:
        raw_dir = output_sub_dir / "raw_slic_masks" if save_raw_masks else None
        col_dir = output_sub_dir / "raw_slic_colored_masks" if (save_raw_masks and save_color_mask) else None

        for idx, item in enumerate(proposals):
            if save_raw_masks:
                out_name = f"{idx:03d}_slic.png"
                _save_mask_item(
                    item["mask_image"], item["fill_color"], raw_dir, col_dir,
                    out_name, 0, (h, w), save_color_mask
                )

        _save_overview_colored(proposals, output_sub_dir / "slic_overview_colored.png", (h, w), 0)
        print(f"  --> [Stage 1 - SLIC] 原始留洞候选数: {len(proposals)} | 预览已保存至: {output_sub_dir / 'slic_overview_colored.png'}")

    return proposals


def get_raw_sam_proposals(
    image_rgb: np.ndarray,
    output_sub_dir: Optional[Path],
    cfg: Dict[str, Any],
    preloaded_model=None,
    device: Optional[torch.device] = None,
) -> List[Dict[str, Any]]:
    """提取 SAM 自动分割候选（严格保留内部空洞）。"""
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    sam_checkpoint = cfg.get("sam_checkpoint", cfg.get("paths", {}).get("sam_checkpoint", ""))
    sam_model_type = cfg.get("sam_model_type", cfg.get("sam", {}).get("model_type", "vit_h"))
    min_area = cfg.get("min_area", cfg.get("preprocess", {}).get("min_area", 0.00015))
    save_raw_masks = bool(cfg.get("save_raw_masks", cfg.get("save_options", {}).get("save_raw_masks", True)))
    save_color_mask = bool(cfg.get("save_color_mask", cfg.get("save_options", {}).get("save_color_mask", True)))

    if preloaded_model is not None:
        sam_model = preloaded_model
    else:
        if not sam_checkpoint or not os.path.isfile(sam_checkpoint):
            print(f"[提示] 未找到 SAM 权重 ({sam_checkpoint})，跳过 SAM 路由...")
            return []
        print(f"[Stage 1 - SAM] 加载模型 ({sam_model_type})...")
        sam_model = get_sam_model(model_type=sam_model_type, checkpoint_path=sam_checkpoint, device=device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam_model,
        pred_iou_thresh=float(cfg.get("pred_iou_thresh", cfg.get("sam", {}).get("pred_iou_thresh", 0.88))),
        stability_score_thresh=float(cfg.get("stability_score_thresh", cfg.get("sam", {}).get("stability_score_thresh", 0.95))),
        crop_n_layers=int(cfg.get("crop_n_layers", cfg.get("sam", {}).get("crop_n_layers", 1))),
        points_per_side=int(cfg.get("points_per_side", cfg.get("sam", {}).get("points_per_side", 64))),
        min_mask_region_area=0,
    )
    raw_masks = mask_generator.generate(image_rgb)
    h, w = image_rgb.shape[:2]

    proposals = []
    for item in raw_masks:
        m_bool = item["segmentation"]
        mask = m_bool.astype(np.uint8) * 255

        area = int((mask > 0).sum())
        min_area_px = resolve_min_area(min_area, h * w)
        if area >= min_area_px:
            mean_col, std_val = _get_dominant_color_and_homogeneity(image_rgb, mask)
            proposals.append({
                "mask_image": mask,
                "area": area,
                "fill_color": mean_col.tolist(),
                "homogeneity_std": std_val,
                "source": "sam",
            })

    proposals = sorted(proposals, key=lambda x: x["area"], reverse=True)
    for idx, item in enumerate(proposals):
        item["orig_idx"] = idx

    if output_sub_dir is not None:
        raw_dir = output_sub_dir / "raw_sam_masks" if save_raw_masks else None
        col_dir = output_sub_dir / "raw_sam_colored_masks" if (save_raw_masks and save_color_mask) else None

        for idx, item in enumerate(proposals):
            if save_raw_masks:
                out_name = f"{idx:03d}_sam.png"
                _save_mask_item(
                    item["mask_image"], item["fill_color"], raw_dir, col_dir,
                    out_name, 0, (h, w), save_color_mask
                )

        _save_overview_colored(proposals, output_sub_dir / "sam_overview_colored.png", (h, w), 0)
        print(f"  --> [Stage 1 - SAM] 原始留洞候选数: {len(proposals)} | 预览已保存至: {output_sub_dir / 'sam_overview_colored.png'}")

    return proposals


# ==============================================================================
# 第二阶段：孔洞填充、形态学平滑与单连通组件拆解
# ==============================================================================

def process_hole_queue_and_morphology(
    image_rgb: np.ndarray,
    raw_proposals: List[Dict[str, Any]],
    output_sub_dir: Optional[Path],
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """对原始候选进行内部闭合孔洞填实，并拆分为单连通组件。"""
    print(f"\n==================================================================")
    print(f" [Stage 2 - Hole Fill & CC] 执行内部孔洞实心化与连通拆分(不把孔洞加入候选队列)...")
    print(f"==================================================================")
    h, w = image_rgb.shape[:2]
    min_area = resolve_min_area(cfg.get("min_area", cfg.get("preprocess", {}).get("min_area", 0.00015)), h * w)
    save_original_masks = bool(cfg.get("save_original_masks", cfg.get("save_origin_masks", cfg.get("save_options", {}).get("save_origin_masks", True))))
    save_color_mask = bool(cfg.get("save_color_mask", cfg.get("save_options", {}).get("save_color_mask", True)))

    mask_queue = deque()
    for idx, p in enumerate(raw_proposals):
        mask_queue.append({
            "source": p["source"],
            "orig_idx": p["orig_idx"],
            "depth": 0,
            "mask_image": p["mask_image"],
        })

    solid_regions = []
    hole_count = 0

    while mask_queue:
        curr = mask_queue.popleft()
        filled, holes = _fill_holes(curr["mask_image"])

        # 检测并记录闭合孔洞
        hole_area = int((holes > 0).sum())
        if hole_area >= min_area:
            hole_count += 1
            # print(f"  [Hole-Fill] {curr['source'].upper()}#{curr['orig_idx']:02d} 发现并填补内部闭合孔洞(面积={hole_area:6d})")

        smoothed_m, _ = _smart_morphology_preprocess(filled, orig_area=int((filled > 0).sum()))
        num_labels, labels = cv2.connectedComponents(smoothed_m)

        for comp_id in range(1, num_labels):
            comp_mask = (labels == comp_id).astype(np.uint8) * 255
            comp_area = int((comp_mask > 0).sum())
            if comp_area >= min_area:
                mean_col, std_val = _get_dominant_color_and_homogeneity(image_rgb, comp_mask)
                solid_regions.append({
                    "mask_image": comp_mask,
                    "area": comp_area,
                    "fill_color": mean_col.tolist(),
                    "homogeneity_std": std_val,
                    "source": curr["source"],
                    "orig_idx": curr["orig_idx"],
                    "depth": curr["depth"],
                    "cc_idx": comp_id,
                })

    solid_regions = sorted(solid_regions, key=lambda x: x["area"], reverse=True)
    for idx, item in enumerate(solid_regions):
        item["orig_sort_idx"] = idx
    print(f"  --> [Stage 2] 完成！由初始 {len(raw_proposals)} 个候选，实心化闭合空洞 {hole_count} 处后，生成实心连通块共 {len(solid_regions)} 个")

    if output_sub_dir is not None:
        orig_dir = output_sub_dir / "origin_masks" if save_original_masks else None
        orig_col_dir = output_sub_dir / "origin_colored_masks" if (save_original_masks and save_color_mask) else None

        for idx, item in enumerate(solid_regions):
            if save_original_masks:
                out_name = f"{idx:03d}_{item['source']}_m{item['orig_idx']:02d}_d{item['depth']}_cc{item['cc_idx']}.png"
                _save_mask_item(
                    item["mask_image"], item["fill_color"], orig_dir, orig_col_dir,
                    out_name, 0, (h, w), save_color_mask
                )

        _save_overview_colored(solid_regions, output_sub_dir / "origin_overview_colored.png", (h, w), 0)
        print(f"  --> [Stage 2] 已生成 origin 预览图: {output_sub_dir / 'origin_overview_colored.png'}")

    return solid_regions


# ==============================================================================
# 第三阶段：层级去重与纯度感知直接父级同色吸收 (pre_masks/)
# ==============================================================================

def perform_fusion_and_save(
    image_rgb: np.ndarray,
    candidates: List[Dict[str, Any]],
    output_sub_dir: Optional[Path],
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    第三阶段 - 跨界去重与纯度感知直接父级同色吸收（完全一致原版算法）：
    3.1 SAM 自去重与压制 SLIC
    3.2 原生掩码压制重复空洞
    3.3 构建全局背景 Mask (序号0)
    3.4 纯度感知直接父级同色吸收 (Smallest Direct Enclosing Parent)
    """
    print(f"\n==================================================================")
    print(f" [Stage 3 - Fusion Engine] 跨算法去重 & 纯度感知父级同色吸收")
    print(f"==================================================================")
    h, w = image_rgb.shape[:2]

    iou_sam_internal_thresh = float(cfg.get("iou_sam_internal_thresh", cfg.get("iou_threshold", cfg.get("preprocess", {}).get("iou_threshold", 0.90))))
    iou_sam_slic_thresh = float(cfg.get("iou_sam_slic_thresh", cfg.get("preprocess", {}).get("iou_sam_slic_thresh", 0.90)))
    parent_contain_thresh = float(cfg.get("parent_contain_thresh", cfg.get("inclusion_threshold", cfg.get("preprocess", {}).get("inclusion_threshold", 0.90))))
    iou_native_hole_thresh = float(cfg.get("iou_native_hole_thresh", cfg.get("preprocess", {}).get("iou_native_hole_thresh", 0.90)))
    self_thresh = float(cfg.get("self_pure_std_thresh", cfg.get("max_pure_std_thresh", cfg.get("preprocess", {}).get("self_pure_std_thresh", 15.0))))
    parent_thresh = float(cfg.get("parent_pure_std_thresh", cfg.get("preprocess", {}).get("parent_pure_std_thresh", 3.0)))
    color_diff_thresh = float(cfg.get("color_diff_thresh", cfg.get("preprocess", {}).get("color_diff_thresh", 5.0)))
    save_nms_masks = bool(cfg.get("save_nms_masks", cfg.get("save_options", {}).get("save_nms_masks", True)))
    save_color_mask = bool(cfg.get("save_color_mask", cfg.get("save_options", {}).get("save_color_mask", True)))
    export_size = int(cfg.get("export_size", cfg.get("preprocess", {}).get("target_size", 0)))

    # 拆解来源
    sam_list = [c for c in candidates if c["source"] == "sam"]
    slic_list = [c for c in candidates if c["source"] == "slic"]

    # 3.1 & 3.2 SAM 自去重与纯度感知跨界压制 SLIC
    kept_sam = []
    for s_item in sorted(sam_list, key=lambda x: x["area"], reverse=True):
        is_dup = False
        for k_item in kept_sam:
            iou = _compute_iou(s_item["mask_image"], k_item["mask_image"])
            if iou > iou_sam_internal_thresh:
                is_dup = True
                break
        if not is_dup:
            kept_sam.append(s_item)

    kept_slic = []
    # 纯度保护阈值：SAM 的色彩方差低于此值说明是纯色图层，允许压制；若高于此值说明是复合容器，严禁压制内部 SLIC 细节
    sam_pure_suppress_std = float(cfg.get("sam_pure_suppress_std", cfg.get("preprocess", {}).get("sam_pure_suppress_std", 10.0)))
    slic_contain_suppress_thresh = float(cfg.get("slic_contain_suppress_thresh", cfg.get("preprocess", {}).get("slic_contain_suppress_thresh", 0.85)))

    for sl_item in sorted(slic_list, key=lambda x: x["area"], reverse=True):
        suppressed = False
        sl_mask = sl_item["mask_image"]
        sl_area = float(sl_item["area"])
        sl_col = np.array(sl_item["fill_color"], dtype=float)

        for k_sam in kept_sam:
            sam_std = k_sam.get("homogeneity_std", 0.0)
            sam_mask = k_sam["mask_image"]
            sam_area = float(k_sam["area"])

            inter = np.logical_and(sl_mask > 0, sam_mask > 0).sum()
            if inter == 0:
                continue

            union = sl_area + sam_area - inter
            iou = inter / float(union) if union > 0 else 0.0
            contain_ratio = inter / sl_area if sl_area > 0 else 0.0

            sam_col = np.array(k_sam["fill_color"], dtype=float)
            c_diff = np.linalg.norm(sl_col - sam_col)

            # 【纯度感知立体压制规则】：
            # 条件 1：高对称 IoU 重合 (>=0.85)，且 SAM 本身是纯色（不是多色容器） -> SAM 取代 SLIC
            if iou >= iou_sam_slic_thresh and sam_std <= sam_pure_suppress_std:
                suppressed = True
                break

            # 条件 2：包含度极高 (SLIC 85% 以上都在 SAM 内部) 且两者颜色极其接近 (色差 < 10) 且 SAM 纯净
            # 说明 SLIC 只是 SAM 大纯色块内因超像素聚类分裂出的多边形残片/坑洼副本 -> 坚决压制
            if contain_ratio >= slic_contain_suppress_thresh and c_diff < color_diff_thresh * 2.0 and sam_std <= sam_pure_suppress_std:
                suppressed = True
                break

        if not suppressed:
            kept_slic.append(sl_item)

    temp_list = kept_sam + kept_slic
    native_masks = [m for m in temp_list if m["depth"] == 0]
    hole_masks = [m for m in temp_list if m["depth"] > 0]

    kept_holes = []
    for h_item in sorted(hole_masks, key=lambda x: x["area"], reverse=True):
        suppressed = False
        for n_item in native_masks:
            iou = _compute_iou(h_item["mask_image"], n_item["mask_image"])
            if iou > iou_native_hole_thresh:
                suppressed = True
                # print(f"  [原生压制空洞] orig#{h_item.get('orig_sort_idx', 0):03d} 与 原生掩码 orig#{n_item.get('orig_sort_idx', 0):03d} IoU={iou:.3f} -> 移除重复空洞")
                break
        if not suppressed:
            kept_holes.append(h_item)

    surviving_cands = sorted(native_masks + kept_holes, key=lambda x: x["area"], reverse=True)

    # 3.2.5 去重后增加一个背景 Mask (命名 bg, 序号为 0)，其背景色由 get_canvas_background_color 准确提取
    bg_color = [int(x) for x in get_canvas_background_color(image_rgb)]
    print(f"  [背景构造] 提取画布背景色 RGB={bg_color}，构建全局背景掩码(命名bg,序号0)")
    bg_item = {
        "source": "bg",
        "orig_idx": 0,
        "depth": 0,
        "cc_idx": 0,
        "area": int(h * w),
        "mask_image": np.ones((h, w), dtype=np.uint8) * 255,
        "fill_color": bg_color,
        "orig_sort_idx": 0,
        "nms_idx": 0,
        "homogeneity_std": 0.0,
    }
    surviving = [bg_item] + surviving_cands
    for idx, item in enumerate(surviving):
        item["nms_idx"] = idx
    n = len(surviving)
    to_remove = set()

    def _get_mask_filename(idx: int, item: Dict) -> str:
        if item["source"] == "bg":
            return f"{idx:03d}_bg.png"
        return f"{idx:03d}_{item['source']}_m{item['orig_idx']:02d}_d{item['depth']}_cc{item['cc_idx']}.png"

    # 按照面积大小另存 IoU 去重后、纯度吸收前的中间掩码与彩色掩码
    if output_sub_dir is not None and save_nms_masks:
        nms_dir = output_sub_dir / "nms_masks"
        nms_col_dir = output_sub_dir / "nms_colored_masks" if save_color_mask else None
        for idx, item in enumerate(surviving):
            out_name = _get_mask_filename(idx, item)
            _save_mask_item(
                item["mask_image"], item["fill_color"], nms_dir, nms_col_dir,
                out_name, 0, (h, w), save_color_mask
            )
        _save_overview_colored(surviving, output_sub_dir / "nms_overview_colored.png", (h, w), 0)
        print(f"  --> [Stage 3.1 & 3.2] 完成！IoU去重后留存 {len(surviving)} 个掩码已保存在: {nms_dir}")

    # 3.3 检查所有精选候选的直接父级同色从属关系（保留自身纯色和父级纯色判断，全面升级为 CIELAB Delta E 色差）
    # 默认 color_diff_thresh = 6.0 对应 CIELAB Delta E 约为 6.0 (标准同色系无感容差)
    cielab_diff_thresh = float(color_diff_thresh) if color_diff_thresh is not None else 6.0

    for i in range(n - 1, -1, -1):
        if i in to_remove or surviving[i]["source"] == "bg":
            continue
        cur = surviving[i]
        # 【自身纯度锁】：自身不纯（如包含多个色块/纹理）则独立保留，绝不被吸收
        if cur.get("homogeneity_std", 0.0) > self_thresh:
            continue
        cur_mask = cur["mask_image"] > 0
        cur_color_u8 = np.clip(np.array(cur["fill_color"], dtype=float), 0, 255).astype(np.uint8).reshape(1, 1, 3)
        cur_lab = cv2.cvtColor(cur_color_u8, cv2.COLOR_RGB2LAB).astype(float)[0, 0]
        cur_area = cur["area"]

        for j in range(i - 1, -1, -1):
            if j in to_remove:
                continue
            parent = surviving[j]
            parent_mask = parent["mask_image"] > 0

            inter = np.logical_and(cur_mask, parent_mask).sum()
            inc_ratio = inter / float(cur_area) if cur_area > 0 else 0.0

            if inc_ratio >= parent_contain_thresh:
                # 【父级纯度锁】：父级如果不是纯色（包含多物体/杂色容器），停止向该父级吸收，防止误杀
                if parent.get("homogeneity_std", 0.0) > parent_thresh:
                    break

                parent_color_u8 = np.clip(np.array(parent["fill_color"], dtype=float), 0, 255).astype(np.uint8).reshape(1, 1, 3)
                parent_lab = cv2.cvtColor(parent_color_u8, cv2.COLOR_RGB2LAB).astype(float)[0, 0]

                # 高精度 CIELAB 感知色差
                delta_e = float(np.linalg.norm(cur_lab - parent_lab))

                if delta_e < cielab_diff_thresh:
                    to_remove.add(i)
                    # print(f"  [同色吸收] #{i:03d} -> 被直接父级 #{j:03d} 吸收 (包含 {inc_ratio*100:.1f}%, CIELAB DeltaE={delta_e:.2f} < {cielab_diff_thresh})")
                break

    final_masks = [surviving[idx] for idx in range(n) if idx not in to_remove]
    print(f"  --> [Stage 3] 融合完成！终极保留优质图层: {len(final_masks)} 个")

    if output_sub_dir is not None:
        pre_dir = output_sub_dir / "pre_masks"
        pre_col_dir = output_sub_dir / "pre_colored_masks" if save_color_mask else None
        pre_dir.mkdir(parents=True, exist_ok=True)

        meta_info = {
            "num_masks": len(final_masks),
            "output_dir": str(pre_dir),
            "image_width": w,
            "image_height": h,
            "background_color": bg_color,
            "masks": [],
        }

        for idx, item in enumerate(final_masks):
            out_name = _get_mask_filename(idx, item)
            _save_mask_item(
                item["mask_image"], item["fill_color"], pre_dir, pre_col_dir,
                out_name, export_size, (h, w), save_color_mask
            )
            meta_info["masks"].append({
                "id": idx,
                "filename": out_name,
                "area": item["area"],
                "fill_color": item["fill_color"],
                "homogeneity_std": round(item["homogeneity_std"], 2),
                "source": item["source"],
                "orig_idx": item["orig_idx"],
                "depth": item["depth"],
                "cc_idx": item["cc_idx"],
            })

        _save_overview_colored(final_masks, output_sub_dir / "pre_overview_colored.png", (h, w), export_size)
        with open(output_sub_dir / "pre_masks_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta_info, f, ensure_ascii=False, indent=2)

        print(f"[OK] 最终精简分层成功保存至: {pre_dir}")
        print(f"[OK] 最终效果全彩预览已生成: {output_sub_dir / 'pre_overview_colored.png'}")

    return final_masks


# ==============================================================================
# 实验性接口：后置填洞流水线 (Late Hole-Filling Pipeline)
# 阶段 2: 保持中空拓扑 + 真实物理色彩采样 + 形态学平滑 + 单连通拆分 (不填洞)
# 阶段 3: 基于 100% 真实纯度进行跨界去重与父子同色吸收
# 阶段 3 尾声: 对最终筛选出的存活图层执行几何填洞实心化
# ==============================================================================

def process_morphology_keep_holes(
    image_rgb: np.ndarray,
    raw_proposals: List[Dict[str, Any]],
    output_sub_dir: Optional[Path],
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    【实验性阶段 2】：暂不填洞，仅执行形态学平滑与单连通组件拆解。
    保持原生中空拓扑，保证采集到的色彩方差 (homogeneity_std) 100% 真实，不被洞内异色污染。
    """
    print(f"\n==================================================================")
    print(f" [Stage 2 - Pure Morphology (Keep Holes)] 保持中空拓扑形态平滑 & 单连通拆解...")
    print(f"==================================================================")
    h, w = image_rgb.shape[:2]
    min_area = resolve_min_area(cfg.get("min_area", cfg.get("preprocess", {}).get("min_area", 0.00015)), h * w)
    save_original_masks = bool(cfg.get("save_original_masks", cfg.get("save_origin_masks", cfg.get("save_options", {}).get("save_origin_masks", True))))
    save_color_mask = bool(cfg.get("save_color_mask", cfg.get("save_options", {}).get("save_color_mask", True)))

    pure_regions = []

    for idx, p in enumerate(raw_proposals):
        raw_mask = p["mask_image"]
        # 直接对原始带洞 Mask 进行形态学平滑
        smoothed_m, _ = _smart_morphology_preprocess(raw_mask, orig_area=int((raw_mask > 0).sum()))
        num_labels, labels = cv2.connectedComponents(smoothed_m)

        for comp_id in range(1, num_labels):
            comp_mask = (labels == comp_id).astype(np.uint8) * 255
            comp_area = int((comp_mask > 0).sum())
            if comp_area >= min_area:
                # 仅对真实前景像素进行纯度采样，绝不污染洞内背景色
                mean_col, std_val = _get_dominant_color_and_homogeneity(image_rgb, comp_mask)
                pure_regions.append({
                    "mask_image": comp_mask,
                    "area": comp_area,
                    "fill_color": mean_col.tolist(),
                    "homogeneity_std": std_val,
                    "source": p["source"],
                    "orig_idx": p["orig_idx"],
                    "depth": 0,
                    "cc_idx": comp_id,
                })

    pure_regions = sorted(pure_regions, key=lambda x: x["area"], reverse=True)
    for idx, item in enumerate(pure_regions):
        item["orig_sort_idx"] = idx
    print(f"  --> [Stage 2 Pure] 完成！生成保留原生纯度的单连通图层共 {len(pure_regions)} 个")

    if output_sub_dir is not None:
        orig_dir = output_sub_dir / "origin_masks" if save_original_masks else None
        orig_col_dir = output_sub_dir / "origin_colored_masks" if (save_original_masks and save_color_mask) else None

        for idx, item in enumerate(pure_regions):
            if save_original_masks:
                out_name = f"{idx:03d}_{item['source']}_m{item['orig_idx']:02d}_d{item['depth']}_cc{item['cc_idx']}.png"
                _save_mask_item(
                    item["mask_image"], item["fill_color"], orig_dir, orig_col_dir,
                    out_name, 0, (h, w), save_color_mask
                )

        _save_overview_colored(pure_regions, output_sub_dir / "origin_overview_colored.png", (h, w), 0)

    return pure_regions


def run_segmentation(
    image_rgb: np.ndarray,
    output_dir: str,
    cfg: Dict[str, Any],
    preloaded_model=None,
    device: Optional[torch.device] = None,
    alpha_mask: Optional[np.ndarray] = None,
) -> Tuple[List[str], Tuple[int, int, int]]:
    """
    【VISTA 标准分割流水线 - 后置填洞方案】：
    阶段 1：SLIC 自适应超像素 + SAM 语义候选提取 (保持中空拓扑)；
    阶段 2：形态学平滑 + 单连通拆解 (保持中空真实采样，确保色彩方差 100% 真实)；
    阶段 3：跨算法立体压制 (SAM 自去重、SAM 压制 SLIC、基于 CIELAB Delta E 与纯度感知的父子同色吸收)；
    阶段 3 尾声：统一执行闭合孔洞几何填实 (为阶段 4 贝塞尔闭合轮廓提供实心几何底盘)。
    """
    return run_segmentation_late_fill(
        image_rgb=image_rgb,
        output_dir=output_dir,
        cfg=cfg,
        preloaded_model=preloaded_model,
        device=device,
        alpha_mask=alpha_mask,
    )


def run_segmentation_late_fill(
    image_rgb: np.ndarray,
    output_dir: str,
    cfg: Dict[str, Any],
    preloaded_model=None,
    device: Optional[torch.device] = None,
    alpha_mask: Optional[np.ndarray] = None,
) -> Tuple[List[str], Tuple[int, int, int]]:
    """
    【新实验性顶层分割接口】：后置填洞 (Late Hole-Filling)。
    阶段 2 保持中空真实采样 -> 阶段 3 纯度融合去重 -> 最终结果实心化输出至 pre_masks。
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    use_sam = bool(cfg.get("use_sam", cfg.get("sam", {}).get("enabled", True)))
    use_slic = bool(cfg.get("use_slic", cfg.get("slic", {}).get("enabled", True)))
    use_alpha_mask = bool(cfg.get("use_alpha_mask", cfg.get("preprocess", {}).get("use_alpha_mask", True)))

    raw_proposals = []

    if use_alpha_mask and alpha_mask is not None:
        fg_area = int(np.sum(alpha_mask > 127))
        h, w = image_rgb.shape[:2]
        if 0 < fg_area < h * w:
            alpha_item = {
                "source": "alpha",
                "orig_idx": 0,
                "depth": 0,
                "cc_idx": 0,
                "area": fg_area,
                "mask_image": alpha_mask.copy(),
                "fill_color": list(get_canvas_background_color(image_rgb)),
                "orig_sort_idx": 0,
                "homogeneity_std": 0.0,
            }
            raw_proposals.append(alpha_item)

    if use_slic:
        slic_props = get_raw_slic_proposals(image_rgb, output_path, cfg)
        raw_proposals.extend(slic_props)

    if use_sam:
        sam_props = get_raw_sam_proposals(image_rgb, output_path, cfg, preloaded_model=preloaded_model, device=device)
        raw_proposals.extend(sam_props)

    # 1. 阶段 2: 保持中空进行形态学平滑与单连通拆解 (不填洞)
    pure_proposals = process_morphology_keep_holes(image_rgb, raw_proposals, output_path, cfg)

    # 2. 阶段 3: 在 100% 真实纯度下执行跨界压制与父子同色吸收
    fused_masks = perform_fusion_and_save(image_rgb, pure_proposals, output_path, cfg)

    # 3. 最终几何实心化：对最终存活的各图层填洞，为阶段 4 贝塞尔闭合轮廓拟合提供实心几何底盘
    print("  [后置几何实心化] 对最终精简图层执行闭合孔洞填实...")
    save_color_mask = bool(cfg.get("save_color_mask", cfg.get("save_options", {}).get("save_color_mask", True)))
    pre_dir = output_path / "pre_masks"
    pre_col_dir = output_path / "pre_colored_masks" if save_color_mask else None
    pre_paths = []

    for idx, item in enumerate(fused_masks):
        if item["source"] != "bg":
            filled_m, _ = _fill_holes(item["mask_image"])
            item["mask_image"] = filled_m
            out_name = f"{idx:03d}_{item['source']}_m{item['orig_idx']:02d}_d{item['depth']}_cc{item['cc_idx']}.png"
            out_file = pre_dir / out_name
            _save_mask_item(
                item["mask_image"], item["fill_color"], pre_dir, pre_col_dir,
                out_name, 0, image_rgb.shape[:2], save_color_mask
            )
            pre_paths.append(str(out_file))

    # 同步重新生成实心化后的全景预览彩图
    _save_overview_colored(fused_masks, output_path / "pre_overview_colored.png", image_rgb.shape[:2], 0)

    bg_color = tuple(get_canvas_background_color(image_rgb))
    return pre_paths, bg_color
