"""
VISTA 几何拟合与 DiffVG 可微矢量优化模块 (vectorize.py)
包含：
- 初始多图层 SVG 构建与分步帧渲染 (generate_init_svg)
- 基于光栅化几何与颜色相似度的分层剪枝算法 (prune_shapes_by_rendered_masks)
- 三阶段 DiffVG 可微渲染优化循环 (svg_optimize: 主优化 -> 几何剪枝 -> 短迭代精修)
- 矢量化顶层调度接口 (run_vectorization)
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import imageio
import numpy as np
import pydiffvg
from PIL import Image
import torch

from utils import (
    collinear_handle_loss,
    color_similarity,
    compute_path_point_nums,
    is_mask_included,
    mask_edge_color_Kmeans,
    mask_to_path,
)


def _save_svg_with_viewbox(svg_path: str, canvas_width: int, canvas_height: int, shapes, shape_groups):
    """保存 SVG 并保证根元素具备标准的 viewBox，避免浏览器裁剪。"""
    pydiffvg.save_svg(svg_path, canvas_width, canvas_height, shapes, shape_groups)
    try:
        if os.path.isfile(svg_path):
            with open(svg_path, "r", encoding="utf-8") as f:
                content = f.read()
            if "<svg" in content and "viewBox" not in content:
                content = content.replace("<svg", f'<svg viewBox="0 0 {canvas_width} {canvas_height}"', 1)
                with open(svg_path, "w", encoding="utf-8") as f:
                    f.write(content)
    except Exception:
        pass


def generate_init_svg(
    shapes: List[pydiffvg.Path],
    shape_groups: List[pydiffvg.ShapeGroup],
    device: torch.device,
    pre_mask_path_list: List[str],
    bg_color: Tuple[int, int, int],
    target_image: np.ndarray,
    frames: List[np.ndarray],
    out_svg_path: str,
    max_error: float = 0.003,
    line_threshold: float = 0.004,
    is_stroke: bool = False,
    poly_epsilon: Optional[float] = None,
    contour_min_dist: float = 0.004,
) -> Tuple[List[pydiffvg.Path], List[pydiffvg.ShapeGroup], List[np.ndarray], Dict[int, np.ndarray]]:
    """
    根据预处理后的 mask 生成初始 SVG；每个 mask 一条 path。
    """
    print("初始化 SVG...")
    st = time.time()
    os.makedirs(out_svg_path, exist_ok=True)
    height, width, _ = target_image.shape
    index_mask_dict = {}

    # 自适应分辨率缩放：输入 < 0.1 则按图像长边比例换算（0.003/0.004）；>= 0.1 则以 512 为基线等比缩放
    long_side = max(width, height)
    eff_max_error = (max_error * long_side) if (max_error < 0.1) else (max_error * (long_side / 512.0))
    eff_line_thresh = (line_threshold * long_side) if (line_threshold < 0.1) else (line_threshold * (long_side / 512.0))
    if poly_epsilon is not None:
        eff_poly_eps = (poly_epsilon * long_side) if (poly_epsilon < 0.1) else (poly_epsilon * (long_side / 512.0))
    else:
        eff_poly_eps = None
    if contour_min_dist is not None:
        eff_min_dist = (contour_min_dist * long_side) if (contour_min_dist < 0.1) else (contour_min_dist * (long_side / 512.0))
    else:
        eff_min_dist = 2.0

    bg_points = torch.tensor([
        [0.0, 0.0],
        [float(width), 0.0],
        [float(width), float(height)],
        [0.0, float(height)],
    ])
    bg_path = pydiffvg.Path(
        num_control_points=torch.LongTensor([0, 0, 0, 0]),
        points=bg_points,
        stroke_width=torch.tensor(0.0),
        is_closed=True,
    )

    if bg_color is None:
        print("警告: 未接收到背景色，默认使用纯白背景")
        bg_color = (255, 255, 255)
    else:
        print(f"初始化画布，使用全局背景色: {bg_color}")

    color = torch.zeros(4, device=device)
    color[:3] = torch.tensor(bg_color, device=device) / 255.0
    color[3] = 1.0
    bg_group = pydiffvg.ShapeGroup(
        shape_ids=torch.tensor([0]),
        fill_color=color,
        stroke_color=torch.tensor([0.0, 0.0, 0.0, 0.0]),
    )
    shapes.append(bg_path)
    shape_groups.append(bg_group)

    i = 1
    for mask_path in pre_mask_path_list:
        base_name = os.path.basename(mask_path).lower()
        if base_name == "bg_auto.png" or base_name.startswith("bg_auto"):
            # print(f"跳过自动背景 mask: {mask_path}")
            continue
        mask_image = Image.open(mask_path).convert("L")
        if mask_image.size != (width, height):
            mask_image = mask_image.resize((width, height), Image.Resampling.NEAREST)

        arr_chk = np.array(mask_image)
        fill_ratio = float((arr_chk > 127).sum()) / float(arr_chk.size + 1e-6)
        if fill_ratio >= 0.98:
            # print(f"跳过近全图 mask: {mask_path}")
            continue
        path = mask_to_path(
            mask_image,
            max_error=eff_max_error,
            line_threshold=eff_line_thresh,
            poly_epsilon=eff_poly_eps,
            contour_min_dist=eff_min_dist,
        )
        if path is None:
            # print(f"跳过空轮廓 mask: {mask_path}")
            continue

        path.points = path.points.to(device)
        rgb_color, _ = mask_edge_color_Kmeans(
            target_image, mask_image, shrink=9, thickness=7
        )
        color = torch.zeros(4, device=device)
        color[:3] = torch.tensor(rgb_color, device=device) / 255.0
        color[3] = 1.0

        shapes_t = [path]
        shape_groups_t = [
            pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([0]),
                fill_color=color,
                stroke_color=torch.tensor([0.0, 0.0, 0.0, 1.0]),
            )
        ]

        if is_stroke:
            group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([i], device=device),
                fill_color=color,
                stroke_color=torch.tensor([0.0, 0.0, 0.0, 1.0], device=device),
            )
        else:
            group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([i], device=device),
                fill_color=color,
                stroke_color=torch.tensor([0.0, 0.0, 0.0, 0.0], device=device),
            )
        shapes.append(path)
        shape_groups.append(group)
        index_mask_dict[i] = np.array(mask_image)

        file_name = os.path.basename(mask_path).split(".")[0]
        _save_svg_with_viewbox(
            os.path.join(out_svg_path, f"{file_name}.svg"),
            width, height, shapes, shape_groups,
        )
        _save_svg_with_viewbox(
            os.path.join(out_svg_path, f"single_{file_name}.svg"),
            width, height, shapes_t, shape_groups_t,
        )

        scene_args = pydiffvg.RenderFunction.serialize_scene(
            width, height, shapes, shape_groups
        )
        img_render = pydiffvg.RenderFunction.apply(
            width, height, 2, 2, 0, None, *scene_args
        )
        frame = (img_render[:, :, :3].detach().cpu().numpy() * 255).astype(np.uint8)
        frames.append(frame)
        i += 1

    _save_svg_with_viewbox(
        os.path.join(out_svg_path, "init.svg"), width, height, shapes, shape_groups
    )
    print(f"SVG 初始化耗时--------------->: {time.time() - st:.2f} s")
    return shapes, shape_groups, frames, index_mask_dict


# ==============================================================================
# 渲染与指标统计辅助函数
# ==============================================================================

def _render_rgb(shapes: List[pydiffvg.Path], shape_groups: List[pydiffvg.ShapeGroup], w: int, h: int, seed: int = 0) -> torch.Tensor:
    scene_args = pydiffvg.RenderFunction.serialize_scene(w, h, shapes, shape_groups)
    img = pydiffvg.RenderFunction.apply(w, h, 2, 2, seed, None, *scene_args)
    return img[:, :, :3]


@torch.no_grad()
def _mse_and_stats(shapes: List[pydiffvg.Path], shape_groups: List[pydiffvg.ShapeGroup], image_target: torch.Tensor, w: int, h: int, device: torch.device, seed: int = 0) -> Tuple[float, int, int]:
    """返回 (mse_float, n_paths, n_points)。"""
    img = _render_rgb(shapes, shape_groups, w, h, seed=seed).to(device)
    mse = float(torch.mean((img - image_target) ** 2).item())
    n_paths = len(shapes)
    n_points = int(compute_path_point_nums(shapes))
    return mse, n_paths, n_points


def _snapshot_metrics(stage: str, shapes: List[pydiffvg.Path], shape_groups: List[pydiffvg.ShapeGroup], image_target: torch.Tensor, w: int, h: int, device: torch.device) -> Dict[str, Any]:
    mse, n_paths, n_points = _mse_and_stats(
        shapes, shape_groups, image_target, w, h, device
    )
    rec = {
        "stage": stage,
        "mse": round(mse, 6),
        "paths": n_paths,
        "points": n_points,
        "fg_paths": max(0, n_paths - 1),
    }
    print(
        f"[阶段指标:{stage}] MSE={rec['mse']:.6f} "
        f"形状数={rec['paths']} (前景={rec['fg_paths']}) 路径点数={rec['points']}"
    )
    return rec


@torch.no_grad()
def rasterize_layer_masks(shapes: List[pydiffvg.Path], shape_groups: List[pydiffvg.ShapeGroup], w: int, h: int, device: torch.device, threshold: float = 0.5) -> Dict[int, np.ndarray]:
    """优化后按实际几何重新光栅化每层占用掩码。"""
    render = pydiffvg.RenderFunction.apply
    masks = {}
    for i, path in enumerate(shapes):
        pts = path.points.detach()
        ncp = path.num_control_points
        sw = path.stroke_width.detach() if hasattr(path.stroke_width, "detach") else path.stroke_width
        is_closed = path.is_closed
        single = pydiffvg.Path(
            num_control_points=ncp,
            points=pts,
            stroke_width=sw if torch.is_tensor(sw) else torch.tensor(float(sw)),
            is_closed=is_closed,
        )
        group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([0]),
            fill_color=torch.tensor([1.0, 1.0, 1.0, 1.0]),
            stroke_color=torch.tensor([0.0, 0.0, 0.0, 0.0]),
        )
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            w, h, [single], [group]
        )
        img = render(w, h, 2, 2, 0, None, *scene_args)
        alpha = img[:, :, 3] if img.shape[2] >= 4 else img[:, :, 0]
        occ = (img[:, :, :3].mean(dim=2) > threshold) | (alpha > threshold)
        mask = (occ.detach().cpu().numpy().astype(np.uint8)) * 255
        masks[i] = mask
    return masks


def _enable_grads(shapes: List[pydiffvg.Path], shape_groups: List[pydiffvg.ShapeGroup], is_stroke: bool, device: torch.device):
    points_vars, stroke_width_vars, color_vars, stroke_color_vars = [], [], [], []
    for path in shapes:
        path.points = path.points.to(device).detach().requires_grad_(True)
        points_vars.append(path.points)
        if is_stroke:
            path.stroke_width = path.stroke_width.to(device).detach().requires_grad_(True)
            stroke_width_vars.append(path.stroke_width)
    for group in shape_groups:
        group.fill_color = group.fill_color.to(device).detach().requires_grad_(True)
        color_vars.append(group.fill_color)
        if is_stroke:
            group.stroke_color = group.stroke_color.to(device).detach().requires_grad_(True)
            stroke_color_vars.append(group.stroke_color)
    return points_vars, stroke_width_vars, color_vars, stroke_color_vars


def _build_optimizer(
    points_vars,
    color_vars,
    stroke_width_vars,
    stroke_color_vars,
    points_lr,
    color_lr,
    stroke_width_lr,
    stroke_color_lr,
    is_stroke,
):
    optim_params = [
        {"params": points_vars, "lr": points_lr},
        {"params": color_vars, "lr": color_lr},
    ]
    if is_stroke:
        optim_params += [
            {"params": stroke_width_vars, "lr": stroke_width_lr},
            {"params": stroke_color_vars, "lr": stroke_color_lr},
        ]
    optimizer = torch.optim.Adam(optim_params, betas=(0.9, 0.9), eps=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.6, patience=10
    )
    return optimizer, scheduler


def _run_optimize_loop(
    shapes: List[pydiffvg.Path],
    shape_groups: List[pydiffvg.ShapeGroup],
    image_target: torch.Tensor,
    device: torch.device,
    canvas_width: int,
    canvas_height: int,
    num_iters: int,
    points_lr: float,
    color_lr: float = 0.01,
    stroke_width_lr: float = 0.05,
    stroke_color_lr: float = 0.01,
    is_stroke: bool = False,
    early_stopping_patience: int = 20,
    early_stopping_delta: float = 5e-5,
    collinear_scale: float = 0.01,
    collinear_cos_threshold: float = 0.5,
    save_every: int = 5,
    frame_every: int = 5,
    frames: Optional[List[np.ndarray]] = None,
    svg_out_path: Optional[str] = None,
    result_path: Optional[str] = None,
    tag: str = "opt",
    seed_offset: int = 0,
) -> Tuple[float, float, Optional[torch.Tensor]]:
    """执行可微优化循环。"""
    if svg_out_path:
        os.makedirs(svg_out_path, exist_ok=True)
    if result_path:
        os.makedirs(result_path, exist_ok=True)
    points_vars, stroke_width_vars, color_vars, stroke_color_vars = _enable_grads(
        shapes, shape_groups, is_stroke, device
    )
    optimizer, scheduler = _build_optimizer(
        points_vars, color_vars, stroke_width_vars, stroke_color_vars,
        points_lr, color_lr, stroke_width_lr, stroke_color_lr, is_stroke,
    )

    best_loss = float("inf")
    no_improve = 0
    last_loss = float("inf")
    last_mse = float("inf")
    img_render = None

    for it in range(num_iters):
        optimizer.zero_grad()
        img_render = _render_rgb(
            shapes, shape_groups, canvas_width, canvas_height, seed=seed_offset + it
        ).to(device)
        mse_loss = torch.mean((img_render - image_target) ** 2)
        collinear_val = collinear_handle_loss(
            shapes, scale=collinear_scale, cos_threshold=collinear_cos_threshold
        )
        loss = mse_loss + collinear_val
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        for group in shape_groups:
            group.fill_color.data.clamp_(0.0, 1.0)
            if is_stroke:
                group.stroke_color.data.clamp_(0.0, 1.0)

        last_loss = float(loss.item())
        last_mse = float(mse_loss.item())
        if last_loss + early_stopping_delta < best_loss:
            best_loss = last_loss
            no_improve = 0
        else:
            no_improve += 1

        if save_every > 0 and it % save_every == 0:
            lr0 = optimizer.param_groups[0]["lr"]
            print(
                f"[{tag}] 迭代 {it}, 损失={last_loss:.4f}, MSE={last_mse:.4f}, "
                f"共线惩罚={float(collinear_val):.4f}, 学习率={lr0:.4g}"
            )
            if svg_out_path is not None:
                _save_svg_with_viewbox(
                    os.path.join(svg_out_path, f"{tag}_iter_{it}.svg"),
                    canvas_width, canvas_height, shapes, shape_groups,
                )

        if frames is not None and frame_every > 0 and it % frame_every == 0:
            frames.append(
                (img_render.detach().cpu().numpy() * 255).astype(np.uint8)
            )

        if no_improve >= early_stopping_patience:
            print(f"[{tag}] 早停 @ iter {it}")
            if result_path is not None:
                _save_svg_with_viewbox(
                    os.path.join(result_path, f"{tag}_early.svg"),
                    canvas_width, canvas_height, shapes, shape_groups,
                )
            break

    return last_loss, last_mse, img_render


# ==============================================================================
# 基于光栅化占用掩码的同色包含图层剪枝
# ==============================================================================

def prune_shapes_by_rendered_masks(
    shapes: List[pydiffvg.Path],
    shape_groups: List[pydiffvg.ShapeGroup],
    layer_masks: Dict[int, np.ndarray],
    device: torch.device,
    rm_color_threshold: float = 0.01,
    inclusion_threshold: float = 0.8,
) -> Tuple[List[pydiffvg.Path], List[pydiffvg.ShapeGroup], Dict[int, np.ndarray], int]:
    """
    用当前渲染 mask 做层级剪枝。返回 (shapes, shape_groups, layer_masks, n_removed)。
    背景层 index 0 永不删除。
    """
    to_remove = set()
    n = len(shape_groups)
    print("移除多余 path（基于优化后 rasterize mask 与颜色相似度）...")

    base_color_thresh = float(rm_color_threshold) if (rm_color_threshold is not None and rm_color_threshold > 0) else 0.02
    effective_inclusion = float(inclusion_threshold) if inclusion_threshold is not None else 0.80

    total_area = 512 * 512
    for m in layer_masks.values():
        if m is not None and hasattr(m, "shape") and len(m.shape) >= 2:
            total_area = int(m.shape[0] * m.shape[1])
            break
    max_bg_remove_area = total_area * 0.003

    for i in range(n - 1, 0, -1):
        if i in to_remove:
            continue
        current_mask = layer_masks.get(i)
        if current_mask is None:
            to_remove.add(i)
            continue
        current_area = int((current_mask > 0).sum())
        if current_area < 5:
            to_remove.add(i)
            # print(f"移除 shape {i} (area={current_area})：极小/不可见图层")
            continue

        area_ratio = float(current_area) / float(max(total_area, 1))
        # 面积较大时严格使用用户设定的 base_color_thresh；微小碎片适度允许容差
        if area_ratio > 0.001:
            color_thresh_i = base_color_thresh
        else:
            color_thresh_i = base_color_thresh * 1.5

        current_color = shape_groups[i].fill_color[:3]

        for j in range(i - 1, -1, -1):
            if j in to_remove:
                continue
            if j == 0:
                existing_color = shape_groups[0].fill_color[:3]
                dist = color_similarity(existing_color, current_color, device)
                if dist < color_thresh_i:
                    if current_area <= max_bg_remove_area:
                        to_remove.add(i)
                        # print(f"移除 shape {i} (area={current_area}): 贴背景层小碎片且颜色相近")
                break

            existing_mask = layer_masks.get(j)
            if existing_mask is None:
                continue
            if is_mask_included(
                current_mask, existing_mask, inclusion_threshold=effective_inclusion
            ):
                existing_color = shape_groups[j].fill_color[:3]
                dist = color_similarity(existing_color, current_color, device)
                if dist < color_thresh_i:
                    to_remove.add(i)
                    # print(f"移除 shape {i} (area={current_area}): 被直接父级 shape {j} 包含且颜色相近")
                break

    for idx in sorted(list(to_remove), reverse=True):
        del shapes[idx]
        del shape_groups[idx]

    new_masks = {}
    kept_old = [k for k in range(n) if k not in to_remove]
    for new_i, old_i in enumerate(kept_old):
        shape_groups[new_i].shape_ids = torch.tensor([new_i], device=device)
        if old_i in layer_masks:
            new_masks[new_i] = layer_masks[old_i]

    print(f"共移除 {len(to_remove)} 个 path，剩余 {len(shapes)}")
    return shapes, shape_groups, new_masks, len(to_remove)


# ==============================================================================
# DiffVG 可微渲染优化主控
# ==============================================================================

def svg_optimize(
    shapes: List[pydiffvg.Path],
    shape_groups: List[pydiffvg.ShapeGroup],
    target_image: np.ndarray,
    device: torch.device,
    svg_out_path: str,
    frames: List[np.ndarray],
    index_mask_dict: Dict[int, np.ndarray],
    Points_lr: float = 0.1,
    num_iters: int = 1000,
    early_stopping_patience: int = 20,
    early_stopping_delta: float = 5e-5,
    is_stroke: bool = False,
    rm_color_threshold: float = 0.02,
    color_lr: float = 0.01,
    stroke_width_lr: float = 0.05,
    stroke_color_lr: float = 0.01,
    collinear_scale: float = 0.01,
    collinear_cos_threshold: float = 0.5,
    save_every: int = 5,
    frame_every: int = 5,
    prune_enabled: bool = True,
    prune_inclusion_threshold: float = 0.8,
    refine_iters: int = 80,
    raster_threshold: float = 0.5,
) -> Tuple[str, str, List[pydiffvg.Path], List[pydiffvg.ShapeGroup], float, List[Dict[str, Any]]]:
    """主优化 -> 几何光栅化剪枝 -> 短迭代精修。"""
    st = time.time()
    print("开始 SVG 优化...")
    result_path = os.path.dirname(svg_out_path)
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(svg_out_path, exist_ok=True)
    image_target = torch.from_numpy(target_image).float().to(device) / 255.0
    canvas_height, canvas_width = target_image.shape[0], target_image.shape[1]
    _save_svg_with_viewbox(
        os.path.join(result_path, "init.svg"),
        canvas_width, canvas_height, shapes, shape_groups,
    )

    metrics = []
    common_kw = dict(
        image_target=image_target,
        device=device,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        points_lr=Points_lr,
        color_lr=color_lr,
        stroke_width_lr=stroke_width_lr,
        stroke_color_lr=stroke_color_lr,
        is_stroke=is_stroke,
        early_stopping_patience=early_stopping_patience,
        early_stopping_delta=early_stopping_delta,
        collinear_scale=collinear_scale,
        collinear_cos_threshold=collinear_cos_threshold,
        save_every=save_every,
        frame_every=frame_every,
        frames=frames,
        svg_out_path=svg_out_path,
        result_path=result_path,
    )

    # 阶段 1：主优化
    _last_loss, _last_mse, img_render = _run_optimize_loop(
        shapes, shape_groups, num_iters=num_iters, tag="opt", seed_offset=0, **common_kw
    )
    _save_svg_with_viewbox(
        os.path.join(result_path, "op_final.svg"),
        canvas_width, canvas_height, shapes, shape_groups,
    )
    metrics.append(
        _snapshot_metrics(
            "before_prune", shapes, shape_groups, image_target,
            canvas_width, canvas_height, device,
        )
    )

    n_removed = 0
    do_prune = prune_enabled and rm_color_threshold is not None and rm_color_threshold > 0

    if do_prune:
        # 阶段 2：光栅化各层掩码并剪枝
        print("Rasterize 各层占用 mask（优化后几何）...")
        layer_masks = rasterize_layer_masks(
            shapes, shape_groups, canvas_width, canvas_height, device,
            threshold=raster_threshold,
        )
        index_mask_dict.clear()
        index_mask_dict.update(layer_masks)

        shapes, shape_groups, layer_masks, n_removed = prune_shapes_by_rendered_masks(
            shapes, shape_groups, layer_masks, device,
            rm_color_threshold=rm_color_threshold,
            inclusion_threshold=prune_inclusion_threshold,
        )
        index_mask_dict.clear()
        index_mask_dict.update(layer_masks)

        metrics.append(
            _snapshot_metrics(
                "after_prune", shapes, shape_groups, image_target,
                canvas_width, canvas_height, device,
            )
        )
        _save_svg_with_viewbox(
            os.path.join(result_path, "after_prune.svg"),
            canvas_width, canvas_height, shapes, shape_groups,
        )

        # 阶段 3：短迭代精修
        if refine_iters > 0 and n_removed > 0:
            print(f"剪枝后短 refine {refine_iters} iters...")
            _last_loss, _last_mse, img_render = _run_optimize_loop(
                shapes, shape_groups,
                num_iters=refine_iters,
                tag="refine",
                seed_offset=10000,
                early_stopping_patience=max(10, early_stopping_patience // 2),
                **{k: v for k, v in common_kw.items()
                   if k not in ("early_stopping_patience",)},
            )
        elif refine_iters > 0 and n_removed == 0:
            print("未移除 path，跳过 refine")

        metrics.append(
            _snapshot_metrics(
                "after_refine", shapes, shape_groups, image_target,
                canvas_width, canvas_height, device,
            )
        )
    else:
        m = metrics[-1]
        metrics.append({**m, "stage": "after_prune"})
        metrics.append({**m, "stage": "after_refine"})

    final_mse, _, _ = _mse_and_stats(
        shapes, shape_groups, image_target, canvas_width, canvas_height, device
    )
    if metrics:
        metrics[-1]["mse"] = round(final_mse, 6)

    svg_path = os.path.join(result_path, "final.svg")
    _save_svg_with_viewbox(svg_path, canvas_width, canvas_height, shapes, shape_groups)

    if img_render is not None:
        frames.append(
            (img_render.detach().cpu().numpy() * 255).astype(np.uint8)
        )
    else:
        img = _render_rgb(shapes, shape_groups, canvas_width, canvas_height)
        frames.append((img.detach().cpu().numpy() * 255).astype(np.uint8))

    gif_path = os.path.join(result_path, "animation.gif")
    if frames:
        imageio.mimsave(gif_path, frames, duration=15, loop=1)

    print("---- Pareto / stage metrics ----")
    for m in metrics:
        print(
            f"  {m['stage']:14s}  mse={m['mse']:.6f}  "
            f"paths={m['paths']:3d}  points={m['points']:5d}"
        )
    print(f"SVG 优化耗时--------------->: {time.time() - st:.2f} s")
    print(f"final MSE (reported): {final_mse:.6f}")

    return svg_path, gif_path, shapes, shape_groups, final_mse, metrics


# ==============================================================================
# 矢量化阶段顶层入口
# ==============================================================================

def run_vectorization(
    target_rgb: np.ndarray,
    pre_mask_paths: List[str],
    bg_color: Tuple[int, int, int],
    output_dir: str,
    cfg: Dict[str, Any],
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """运行完整矢量化阶段（初始贝塞尔拟合 + DiffVG 可微渲染优化与精修）。"""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pydiffvg.set_device(device)
    pydiffvg.set_use_gpu(device.type == "cuda")

    init_svg_dir = os.path.join(output_dir, "init_svgs")
    optim_svg_dir = os.path.join(output_dir, "optim_svgs")
    os.makedirs(init_svg_dir, exist_ok=True)
    os.makedirs(optim_svg_dir, exist_ok=True)

    pf_cfg = cfg.get("path_fit", {})
    opt_cfg = cfg.get("optimize", {})
    prune_cfg = cfg.get("prune", {})

    st = time.time()
    shapes, shape_groups, frames = [], [], []

    shapes, shape_groups, frames, index_mask_dict = generate_init_svg(
        shapes=shapes,
        shape_groups=shape_groups,
        device=device,
        pre_mask_path_list=pre_mask_paths,
        bg_color=bg_color,
        target_image=target_rgb,
        frames=frames,
        out_svg_path=init_svg_dir,
        max_error=float(pf_cfg.get("bezier_max_error", 0.003)),
        line_threshold=float(pf_cfg.get("line_threshold", 0.004)),
        is_stroke=bool(opt_cfg.get("is_stroke", False)),
        poly_epsilon=pf_cfg.get("poly_epsilon"),
        contour_min_dist=float(pf_cfg.get("contour_min_dist", 0.004)),
    )

    svg_path, gif_path, shapes, shape_groups, final_loss, metrics = svg_optimize(
        shapes=shapes,
        shape_groups=shape_groups,
        target_image=target_rgb,
        device=device,
        svg_out_path=optim_svg_dir,
        frames=frames,
        index_mask_dict=index_mask_dict,
        Points_lr=float(opt_cfg.get("learning_rate", 0.10)),
        num_iters=int(opt_cfg.get("num_iters", 1000)),
        early_stopping_patience=int(opt_cfg.get("early_stopping_patience", 20)),
        early_stopping_delta=float(opt_cfg.get("early_stopping_delta", 5e-5)),
        is_stroke=bool(opt_cfg.get("is_stroke", False)),
        rm_color_threshold=float(prune_cfg.get("rm_color_threshold", 0.02)),
        color_lr=float(opt_cfg.get("color_lr", 0.01)),
        stroke_width_lr=float(opt_cfg.get("stroke_width_lr", 0.05)),
        stroke_color_lr=float(opt_cfg.get("stroke_color_lr", 0.01)),
        collinear_scale=float(opt_cfg.get("collinear_scale", 0.01)),
        collinear_cos_threshold=float(opt_cfg.get("collinear_cos_threshold", 0.5)),
        save_every=int(opt_cfg.get("save_every", 5)),
        frame_every=int(opt_cfg.get("frame_every", 5)),
        prune_enabled=bool(prune_cfg.get("enabled", True)),
        prune_inclusion_threshold=float(prune_cfg.get("inclusion_threshold", 0.8)),
        refine_iters=int(prune_cfg.get("refine_iters", 80)),
        raster_threshold=float(prune_cfg.get("raster_threshold", 0.5)),
    )
    elapsed = time.time() - st

    path_point_nums = compute_path_point_nums(shapes)
    return {
        "svg_path": svg_path,
        "gif_path": gif_path,
        "shapes": len(shapes),
        "path_point_nums": path_point_nums,
        "mse_loss": round(final_loss, 6),
        "time_consuming": round(elapsed, 4),
        "metrics": metrics,
    }
