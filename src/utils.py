from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import pydiffvg
from scipy.optimize import least_squares
from sklearn.cluster import KMeans
import torch
from PIL import Image
from collections import Counter

def mask_edge_color_Kmeans(image, mask, shrink=5, thickness=5, n_clusters=2,
                           edge_thr=10, min_mask_area=100):
    """
    边缘环形聚类 + 主色调比较，返回最合适的颜色。

    流程：
      1. mask 面积太小 → 直接返回主色调（颜色纯）
      2. 边缘环形采样 → K-means → 边缘色 edge
      3. mask 区域采样 → K-means → 主色调 dominant
      4. dist(edge, dominant) > edge_thr → 返回 dominant（颜色不纯）
         否则 → 返回 edge（颜色纯）

    返回值：
      (rgb, used_dominant)
      - rgb: (R, G, B) 元组
      - used_dominant: True 表示返回主色调（颜色不纯），False 表示返回边缘色（颜色纯）

    参数：
      image    : HxWx3 uint8 原图
      mask     : 二值 mask（PIL Image 或 ndarray）
      shrink   : 边缘环向外收缩像素（奇数）
      thickness: 边缘环厚度（奇数）
      n_clusters: K-means 聚类数
      edge_thr : 边缘色与主色调的欧氏距离阈值（0-255 RGB），
                 默认 40，越大越倾向返回边缘色
      min_mask_area : mask 最小面积，低于此值直接返回主色调（默认 100）
    """
    mask_np = np.array(mask)
    if mask_np.dtype != np.uint8:
        mask_np = mask_np.astype(np.uint8)
    if mask_np.shape[:2] != image.shape[:2]:
        mask_np = cv2.resize(mask_np, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # ---------- 0. 面积太小，直接返回主色调（颜色纯） ----------
    fg = mask_np > 127
    mask_area = int(fg.sum())
    if mask_area < min_mask_area:
        dom_pixels = image[fg].reshape(-1, image.shape[-1])
        if len(dom_pixels) > 5000:
            indices = np.random.choice(len(dom_pixels), size=5000, replace=False)
            dom_pixels = dom_pixels[indices]
        unique_dom = np.unique(dom_pixels, axis=0)
        nc = min(n_clusters, len(unique_dom))
        nc = max(1, nc)
        kmeans = KMeans(n_clusters=nc, random_state=42, n_init=10)
        labels = kmeans.fit_predict(dom_pixels)
        counts = Counter(labels)
        dom_label = counts.most_common(1)[0][0]
        dominant_color = tuple(kmeans.cluster_centers_[dom_label].astype(int)[:3])
        return dominant_color, False  # 面积小，直接返回主色调，颜色纯

    # 强制奇数 kernel
    shrink = shrink if shrink % 2 != 0 else shrink + 1
    thickness = thickness if thickness % 2 != 0 else thickness + 1

    # ---------- 1. 边缘色 ----------
    kernel_shrink = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (shrink, shrink))
    eroded_outer = cv2.erode(mask_np, kernel_shrink, iterations=1)

    kernel_thick = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
    eroded_inner = cv2.erode(eroded_outer, kernel_thick, iterations=1)

    ring_mask = cv2.bitwise_xor(eroded_outer, eroded_inner)
    edge_pixels = image[ring_mask > 0]

    if len(edge_pixels) < 50:
        # 边缘像素太少，直接用主色调
        dom_pixels = image[fg].reshape(-1, image.shape[-1])
        if len(dom_pixels) > 5000:
            indices = np.random.choice(len(dom_pixels), size=5000, replace=False)
            dom_pixels = dom_pixels[indices]
        unique_dom = np.unique(dom_pixels, axis=0)
        nc = min(n_clusters, len(unique_dom))
        nc = max(1, nc)
        kmeans = KMeans(n_clusters=nc, random_state=42, n_init=10)
        labels = kmeans.fit_predict(dom_pixels)
        counts = Counter(labels)
        dom_label = counts.most_common(1)[0][0]
        dominant_color = tuple(kmeans.cluster_centers_[dom_label].astype(int)[:3])
        return dominant_color, False  # 边缘像素少，返回主色调，颜色纯

    # ---------- 2. 边缘色聚类 ----------
    if len(edge_pixels) > 5000:
        indices = np.random.choice(len(edge_pixels), size=5000, replace=False)
        sample_pixels = edge_pixels[indices]
    else:
        sample_pixels = edge_pixels

    unique_pixels = np.unique(sample_pixels, axis=0)
    nc = min(n_clusters, len(unique_pixels))
    nc = max(1, nc)

    kmeans = KMeans(n_clusters=nc, random_state=42, n_init=10)
    labels = kmeans.fit_predict(sample_pixels)
    counts = Counter(labels)
    dominant_label = counts.most_common(1)[0][0]
    edge_color = tuple(kmeans.cluster_centers_[dominant_label].astype(int)[:3])

    # ---------- 3. 主色调聚类 ----------
    dom_pixels = image[fg].reshape(-1, image.shape[-1])
    if len(dom_pixels) > 5000:
        indices = np.random.choice(len(dom_pixels), size=5000, replace=False)
        dom_sample = dom_pixels[indices]
    else:
        dom_sample = dom_pixels

    unique_dom = np.unique(dom_sample, axis=0)
    nc_dom = min(n_clusters, len(unique_dom))
    nc_dom = max(1, nc_dom)

    kmeans_dom = KMeans(n_clusters=nc_dom, random_state=42, n_init=10)
    labels_dom = kmeans_dom.fit_predict(dom_sample)
    counts_dom = Counter(labels_dom)
    dom_label = counts_dom.most_common(1)[0][0]
    dominant_color = tuple(kmeans_dom.cluster_centers_[dom_label].astype(int)[:3])

    # ---------- 4. 比较选择 ----------
    dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(dominant_color, edge_color)))
    if dist > edge_thr:
        return dominant_color, True   # used_dominant=True 表示颜色不纯
    else:
        return edge_color, False


def get_canvas_background_color(image, n_clusters=2):
    """
    通过提取图像四周边缘多像素宽度的带状区域寻找真正的画布背景色，
    并对最大聚类类别计算中位数 RGB，极大提高背景提取准确度且抗边缘噪点。
    """
    h, w = image.shape[:2]
    bw = max(2, min(h, w) // 40)  # 提取边缘稍宽的带状像素(约占边长的2.5%)，抗单像素框干扰

    top_edge = image[:bw, :].reshape(-1, 3)
    bottom_edge = image[-bw:, :].reshape(-1, 3)
    left_edge = image[:, :bw].reshape(-1, 3)
    right_edge = image[:, -bw:].reshape(-1, 3)

    border_pixels = np.concatenate([top_edge, bottom_edge, left_edge, right_edge], axis=0)

    if len(border_pixels) > 5000:
        idx = np.random.choice(len(border_pixels), size=5000, replace=False)
        border_pixels = border_pixels[idx]

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(border_pixels)

    counts = Counter(labels)
    dominant_label = counts.most_common(1)[0][0]

    dominant_pixels = border_pixels[labels == dominant_label]
    bg_color = np.median(dominant_pixels, axis=0).astype(int)

    return tuple(int(x) for x in bg_color[:3])

def get_image_dominant_color(image, n_clusters=3, max_samples=10000):
    """
    使用 K-means 聚类提取整张图像的主色调，作为默认背景色。
    加入随机采样以确保初始化速度极快。
    """
    # 将图像展平为像素列表 (H*W, C)
    pixels = image.reshape(-1, image.shape[-1])

    # 随机采样，避免全图聚类导致的严重耗时
    if len(pixels) > max_samples:
        indices = np.random.choice(len(pixels), size=max_samples, replace=False)
        sample_pixels = pixels[indices]
    else:
        sample_pixels = pixels

    # 运行 K-means 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(sample_pixels)

    # 统计包含像素最多的类
    counts = Counter(labels)
    dominant_cluster_label = counts.most_common(1)[0][0]

    # 获取该类的颜色中心，并转换为整数
    main_color = kmeans.cluster_centers_[dominant_cluster_label].astype(int)

    # 确保只返回 RGB 三个通道（防止原图带有 Alpha 通道报错）
    return tuple(main_color[:3])

def load_and_resize(
    image_path: str,
    target_size: int = 512,
    bg_color: Optional[Tuple[int, int, int]] = None,
    denoise: bool = False,
    denoise_sigma_color: float = 35.0,
    denoise_sigma_space: float = 35.0,
    return_alpha_mask: bool = False,
):
    """
    加载图像并转换为 RGB，如果有透明度/Alpha 通道则合成填充。
    支持：
    1. 自适应对比色或固定色 Alpha 合成，避免前景亮白被白底吞噬；
    2. 可选保边双边滤波去噪 (Bilateral Filter)，抑制 JPEG 压缩噪点；
    3. 可选返回原图提取的二值 Alpha 前景掩码。
    """
    print("预处理目标图像...")
    image = Image.open(image_path)

    # 1. 检查是否存在 Alpha 通道或透明度信息 (RGBA, LA, PA, P with transparency, 等)
    has_alpha = False
    if image.mode in ("RGBA", "LA", "PA"):
        has_alpha = True
    elif image.mode == "P" and "transparency" in image.info:
        has_alpha = True
    elif "A" in image.getbands():
        has_alpha = True

    alpha_mask_np = None

    if has_alpha:
        # 转为标准的 RGBA 进行真实 Alpha 混合
        rgba_img = image.convert("RGBA")
        np_rgba = np.array(rgba_img)
        alpha_channel = np_rgba[:, :, 3]

        # 仅当存在真正的透明背景像素 (alpha < 128) 时才认定为透明背景图
        real_transparent_pixels = int(np.sum(alpha_channel < 128))
        if real_transparent_pixels < max(50, int(alpha_channel.size * 0.001)):
            has_alpha = False

        if return_alpha_mask and has_alpha:
            alpha_mask_np = (alpha_channel > 10).astype(np.uint8) * 255
        elif return_alpha_mask:
            alpha_mask_np = None

        # 决定合成底色：若未指定，则进行自适应背景色计算（避免前景整体为纯白时被纯白底吞没）
        if bg_color is None:
            fg_pixels = np_rgba[alpha_channel > 20, :3]
            if len(fg_pixels) > 0:
                mean_fg = np.mean(fg_pixels, axis=0)
                # 若前景主体为极高亮白色（平均亮度 > 235），则使用黑色底/深色底提供对比
                if np.mean(mean_fg) > 235:
                    chosen_bg = (0, 0, 0)
                else:
                    chosen_bg = (255, 255, 255)
            else:
                chosen_bg = (255, 255, 255)
        else:
            chosen_bg = bg_color

        # 创建指定纯色背景并合成
        bg = Image.new("RGBA", rgba_img.size, (chosen_bg[0], chosen_bg[1], chosen_bg[2], 255))
        image = Image.alpha_composite(bg, rgba_img).convert("RGB")
    else:
        image = image.convert("RGB")

    # 2. 缩放图像，保持宽高比 (target_size <= 0 表示保持原图大小不缩放)
    if target_size > 0:
        w, h = image.size
        scale = target_size / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        if alpha_mask_np is not None:
            alpha_mask_pil = Image.fromarray(alpha_mask_np).resize(new_size, Image.Resampling.NEAREST)
            alpha_mask_np = np.array(alpha_mask_pil)

    # 3. 可选保边双边滤波去噪 (Bilateral Filter)
    if denoise:
        img_np = np.array(image)
        denoised_np = cv2.bilateralFilter(
            img_np,
            d=5,
            sigmaColor=float(denoise_sigma_color),
            sigmaSpace=float(denoise_sigma_space),
        )
        image = Image.fromarray(denoised_np)

    if return_alpha_mask:
        return image, alpha_mask_np
    return image

  
def resolve_min_area(min_area, target_area: int) -> int:
    """
    将最小面积阈值(min_area)解析为真实像素数：
    - 若 0 < min_area < 1 (例如默认的 0.00015)，按目标图像总像素面积(target_area)的该比例计算；
    - 若 min_area >= 1，则直接作为像素数；
    - 若 min_area <= 0，返回 0。
    """
    if min_area is None:
        return max(1, int(target_area * 0.00015))
    try:
        val = float(min_area)
    except (ValueError, TypeError):
        return max(1, int(target_area * 0.00015))
    if val <= 0:
        return 0
    if val < 1.0:
        return max(1, int(target_area * val))
    return int(val)


def save_target_image(image, out_dir, file_name):
    out_file = os.path.join(out_dir, file_name)
    if not os.path.splitext(out_file)[1]:  # 检查是否有扩展名
        out_file += '.png'  # 如果没有，添加默认扩展名（PNG 无损）
    image.save(out_file)
    return out_file

def find_background_seed(image):
    """
    寻找漫水填充的背景种子点。
    严格要求该点必须是黑色 (0)。
    """
    h, w = image.shape[:2]
    
    # 1. 优先找四个角落
    corners = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    for pt in corners:
        if image[pt[1], pt[0]] == 0:
            return pt
            
    # 2. 如果四个角落全被物体挡住了（极少见），沿着四条边界寻找黑点
    for x in range(w):
        if image[0, x] == 0: return (x, 0)
        if image[h - 1, x] == 0: return (x, h - 1)
    for y in range(h):
        if image[y, 0] == 0: return (0, y)
        if image[y, w - 1] == 0: return (w - 1, y)
        
    # 兜底：如果整个边框全是白的，说明这可能真的是个全屏图
    return (0, 0)

def bezier_curve(t, P0, P1, P2, P3):
    """
    计算三阶贝塞尔曲线上的点，t 为 [0,1] 间的参数数组
    """
    t = np.array(t).reshape(-1, 1)
    return (1 - t)**3 * P0 + 3*(1 - t)**2 * t * P1 + 3*(1 - t) * t**2 * P2 + t**3 * P3

def fit_bezier_segment(points):
    """
    拟合单段贝塞尔曲线，返回四个控制点
    """
    n = len(points)
    t = np.linspace(0, 1, n)
    P0 = points[0]
    P3 = points[-1]
    P1_guess = points[int(n/3)]
    P2_guess = points[int(2*n/3)]
    def residuals(params):
        P1 = params[:2]
        P2 = params[2:]
        curve_points = bezier_curve(t, P0, P1, P2, P3)
        return (curve_points - points).flatten()
    result = least_squares(residuals, np.concatenate([P1_guess, P2_guess]))
    P1 = result.x[:2]
    P2 = result.x[2:]
    return P0, P1, P2, P3

def compute_error(points, P0, P1, P2, P3):
    """
    计算贝塞尔曲线与点集的最大误差
    """
    t = np.linspace(0, 1, len(points))
    curve_points = bezier_curve(t, P0, P1, P2, P3)
    errors = np.linalg.norm(curve_points - points, axis=1)
    return np.max(errors)

def point_to_line_distance(points, p1, p2):
    """
    计算点集到直线的距离
    """
    p1, p2, points = map(np.array, [p1, p2, points])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return np.zeros(len(points))
    point_vec = points - p1
    t = np.clip(np.dot(point_vec, line_vec) / (line_len**2), 0, 1)
    projections = p1 + t[:, None] * line_vec
    distances = np.linalg.norm(points - projections, axis=1)
    return distances

def fit_contour(contour_simple, contour, max_error=1.0, line_threshold=1.0):
    """
    使用三阶贝塞尔曲线拟合多边形点集。
    每个分段通过最小二乘法拟合 P1 与 P2 控制点，确保初始路径具备充足的可微变形自由度。
    """
    structured_points = [contour_simple[0]]
    i = 0
    while i < len(contour_simple) - 1:
        # 首段至少跨过一个中间点，防止单段曲线首尾直接闭合
        start_j = len(contour_simple) - 1
        if i == 0 and start_j > 1:
            start_j -= 1
            
        for j in range(start_j, i, -1):
            p1 = contour_simple[i]
            p2 = contour_simple[j]
            idx1 = np.where((contour == p1).all(axis=1))[0][0]
            idx2 = np.where((contour == p2).all(axis=1))[0][0]
            if idx2 >= idx1:
                segment = contour[idx1:idx2+1]
            else:
                segment = np.concatenate((contour[idx1:], contour[:idx2+1]))

            P0, P1, P2, P3 = fit_bezier_segment(segment)
            error = compute_error(segment, P0, P1, P2, P3)
            if error <= max_error:
                structured_points.append([P1, P2, P3])
                i = j
                break

        if i != j:
            # 最小分段拟合保底
            P0, P1, P2, P3 = fit_bezier_segment(segment)
            structured_points.append([P1, P2, P3])
            i = j
        
    return structured_points

def points_to_path(structured_points, closed=True):
    """
    将结构化的点列表转换为 pydiffvg.Path 对象
    """
    if not structured_points or len(structured_points) < 2:
        return None
    points = []
    num_control_points = []
    start_point = structured_points[0]
    points.append(torch.tensor(start_point, dtype=torch.float32))
    for i, item in enumerate(structured_points[1:]):
        if isinstance(item, np.ndarray):
            points.append(torch.tensor(item, dtype=torch.float32))
            num_control_points.append(0)
        elif isinstance(item, list) and len(item) == 3:
            c1, c2, p2 = item
            points.extend([
                torch.tensor(c1, dtype=torch.float32),
                torch.tensor(c2, dtype=torch.float32),
                torch.tensor(p2, dtype=torch.float32)
            ])
            num_control_points.append(2)
        else:
            raise ValueError("列表元素必须是 numpy 数组或包含三个 numpy 数组的列表")
    if len(points) <= 1:
        return None
    points.pop()  # 移除最后一个多余点
    if len(points) == 0:
        return None
    path = pydiffvg.Path(
        num_control_points=torch.tensor(num_control_points),
        points=torch.stack(points),
        stroke_width=torch.tensor(1.0),
        is_closed=closed
    )
    return path


def mask_to_path(
    mask,
    max_error=1.0,
    line_threshold=1.0,
    poly_epsilon=None,
    contour_min_dist=2.0,
    all_bezier=True,
    adaptive_area_scale=True,
    adaptive_base_area_ratio=0.05,
    adaptive_scale_range=(0.5, 1.2),
):
    """
    根据二值 mask 提取轮廓并拟合生成 pydiffvg.Path。

    all_bezier: 是否初始化为全三阶贝塞尔曲线（默认 True，保证最大可微优化自由度）。
    adaptive_area_scale: 是否基于 mask 面积比例自适应缩放误差容差。
    adaptive_base_area_ratio: 基准面积占比（面积为此比例时缩放为 1.0）。
    adaptive_scale_range: 面积自适应缩放上下限 [min_scale, max_scale]。
    poly_epsilon: approxPolyDP 容差（像素）。None 时使用 max_error。
    contour_min_dist: 轮廓点最小间距，抑制控制点扎堆。
    """
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask, dtype=np.uint8)
    else:
        mask = mask.astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    best_contour = max(contours, key=len)
    contour = best_contour.squeeze()
    if contour.ndim == 1:
        contour = contour[None, :]

    unique_contour, unique_indices = np.unique(contour, axis=0, return_index=True)
    contour = contour[np.sort(unique_indices)]
    if len(contour) < 3:
        return None

    # 根据 mask 区域面积占比自适应缩放误差容差（小面片分配更精细的像素容差，避免小特征失真）
    if adaptive_area_scale:
        mask_bool = (mask > 127)
        mask_area = float(np.sum(mask_bool))
        canvas_total_area = float(mask.shape[0] * mask.shape[1])
        area_ratio = mask_area / max(canvas_total_area, 1.0)
        
        base_ratio = float(adaptive_base_area_ratio) if adaptive_base_area_ratio else 0.05
        min_s, max_s = adaptive_scale_range if adaptive_scale_range else (0.5, 1.2)
        area_scale = float(np.clip(np.sqrt(area_ratio / max(base_ratio, 1e-5)), min_s, max_s))
    else:
        area_scale = 1.0

    # contour_min_dist 也自适应跟随面积缩放，并保底不过度压缩小图元
    eff_min_dist = float(contour_min_dist) * area_scale
    eff_min_dist = max(eff_min_dist, 1.0)

    if len(contour) > 5:
        filtered_contour = [contour[0]]
        for pt in contour[1:]:
            if np.linalg.norm(pt - filtered_contour[-1]) > eff_min_dist:
                filtered_contour.append(pt)
        if len(filtered_contour) >= 2 and np.linalg.norm(filtered_contour[-1] - filtered_contour[0]) <= eff_min_dist:
            filtered_contour.pop()
        
        # 点数保护：若抽样后点数少于 4 个但原轮廓有效点充足，按均匀步长保留顶点
        if len(filtered_contour) < 4 and len(contour) >= 4:
            step = max(1, len(contour) // 4)
            filtered_contour = list(contour[::step])
            if len(filtered_contour) > 4:
                filtered_contour = filtered_contour[:4]
        
        contour = np.array(filtered_contour)

    if len(contour) < 3:
        return None

    eff_error = float(max_error) * area_scale
    eff_line = float(line_threshold) * area_scale

    # approxPolyDP epsilon 可配；默认跟随 bezier max_error 对齐
    eps = float(eff_error if poly_epsilon is None else (poly_epsilon * area_scale))
    eps = max(eps, 0.2)
    simplified = cv2.approxPolyDP(contour, eps, closed=True).squeeze()

    if simplified.ndim == 1:
        simplified = simplified[None, :]

    if len(simplified) < 3:
        step = max(1, len(contour) // 4)
        simplified = contour[::step]
        if len(simplified) > 4:
            simplified = simplified[:4]

    matches = np.where((contour == simplified[0]).all(axis=1))[0]
    if len(matches) == 0:
        return None
    idx1 = matches[0]
    if (contour[idx1 - 1] != simplified[-1]).any():
        simplified = np.vstack((simplified, contour[idx1 - 1]))

    structured_points = fit_contour(simplified, contour, eff_error, eff_line)
    return points_to_path(structured_points, closed=True)


def straighten_shapes_to_lines(
    shapes: List[pydiffvg.Path],
    straighten_threshold: float = 1.5,
    adaptive_scale: bool = True,
    base_size: float = 100.0,
    relative_ratio: float = 0.015,
) -> Tuple[List[pydiffvg.Path], int]:
    """
    优化后置直线坍缩重构：
    在主可微优化收敛后，检测所有三阶贝塞尔曲线段（P0 -> P1 -> P2 -> P3）。
    若控制点 P1、P2 到端点连线 (P0-P3) 的垂直偏差小于阈值 (straighten_threshold)，
    说明该边在图像中物理上为直线，将其坍缩重构为严格直线段 (移除 P1、P2，num_control_points 从 2 变为 0)。
    
    straighten_threshold: 直线判定垂直距离容差（像素）。
    adaptive_scale: 是否根据 Shape 尺寸比例自适应缩放容差。
    base_size: 自适应基准尺寸（像素）。
    relative_ratio: 相对弦长偏差比例上限（最大垂直偏差/弦长 <= relative_ratio 判定为直线）。
    返回：(重构后的 shapes 列表, 转换的直线段总数)
    """
    if straighten_threshold <= 0:
        return shapes, 0

    total_straightened = 0
    device = shapes[0].points.device if shapes and len(shapes) > 0 else torch.device("cpu")

    for path_idx, path in enumerate(shapes):
        pts = path.points.detach().cpu().numpy()
        ncps = path.num_control_points.detach().cpu().numpy()
        is_closed = path.is_closed
        
        # 计算该 Shape 的包围盒面积与对角线尺寸
        min_xy = np.min(pts, axis=0)
        max_xy = np.max(pts, axis=0)
        shape_bbox_diag = np.linalg.norm(max_xy - min_xy)
        
        if adaptive_scale:
            b_sz = float(base_size) if base_size else 100.0
            size_scale = float(np.clip(np.sqrt(shape_bbox_diag / max(b_sz, 1e-3)), 0.5, 1.5))
            eff_straight_thresh = straighten_threshold * size_scale
        else:
            eff_straight_thresh = straighten_threshold
        
        new_pts = [pts[0]]
        new_ncps = []
        
        pt_cursor = 1
        n_segments = len(ncps)
        
        for seg_idx, ncp in enumerate(ncps):
            is_last_seg = (seg_idx == n_segments - 1)
            
            if ncp == 2:
                # 三次贝塞尔曲线：P0, P1, P2, P3
                p0 = new_pts[-1]
                p1 = pts[pt_cursor]
                p2 = pts[pt_cursor + 1]
                
                # 如果是 closed path 的最后一段，终点 P3 即为起点 pts[0]
                if is_closed and is_last_seg and pt_cursor + 2 >= len(pts):
                    p3 = pts[0]
                    pt_cursor += 2
                else:
                    p3 = pts[pt_cursor + 2]
                    pt_cursor += 3
                
                # 计算 P1, P2 到弦 P0-P3 的最大垂直距离
                line_vec = p3 - p0
                line_len = np.linalg.norm(line_vec)
                if line_len < 1e-6:
                    max_dev = 0.0
                else:
                    def _pt_dist(p, a, b, l_len):
                        vec = p - a
                        t = np.clip(np.dot(vec, b - a) / (l_len ** 2), 0, 1)
                        proj = a + t * (b - a)
                        return np.linalg.norm(p - proj)
                    
                    dev1 = _pt_dist(p1, p0, p3, line_len)
                    dev2 = _pt_dist(p2, p0, p3, line_len)
                    max_dev = max(dev1, dev2)
                
                # 判定是否为直线：
                # 1. 垂直偏差小于自适应面积阈值 eff_straight_thresh
                # 2. 相对偏差 (max_dev / line_len) 小于 relative_ratio
                is_straight = (max_dev <= eff_straight_thresh) or (line_len > 0 and (max_dev / line_len) < relative_ratio)
                
                if is_straight:
                    # 坍缩为直线段
                    if not (is_closed and is_last_seg):
                        new_pts.append(p3)
                    new_ncps.append(0)
                    total_straightened += 1
                else:
                    # 保持三次贝塞尔曲线
                    new_pts.extend([p1, p2])
                    if not (is_closed and is_last_seg):
                        new_pts.append(p3)
                    new_ncps.append(2)
            else:
                # 原本就是直线段 (ncp == 0)
                if is_closed and is_last_seg and pt_cursor >= len(pts):
                    p_end = pts[0]
                else:
                    p_end = pts[pt_cursor]
                    pt_cursor += 1
                    new_pts.append(p_end)
                new_ncps.append(0)
                
        # 更新 Path 对象
        path.points = torch.tensor(np.array(new_pts), dtype=torch.float32, device=device).requires_grad_(True)
        path.num_control_points = torch.tensor(new_ncps, dtype=torch.int32, device=device)

    if total_straightened > 0:
        print(f"  [几何后置重构] 检测并成功将 {total_straightened} 段近直线贝塞尔曲线坍缩为直线段 (阈值={straighten_threshold}px)")
    return shapes, total_straightened


def mask_color_Kmeans(image, mask, n_clusters=3, threshold=0.9):
    """
    使用 K-means 聚类从图像对应的 mask 区域中提取主色。
    通过聚类分离噪声（如误分割的鸟/树枝），提取面积最大的颜色作为主色调。
    """
    mask_np = np.array(mask)
    if mask_np.shape[:2] != image.shape[:2]:
        mask_np = cv2.resize(mask_np, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    # 提取 mask 区域的像素
    masked_pixels = image[mask_np > 0]
    
    if len(masked_pixels) == 0:
        return (0, 0, 0) # 防止 mask 为空时报错
    
    # 计算 mask 像素占图像总像素的比例
    total_pixels = image.shape[0] * image.shape[1]
    mask_ratio = len(masked_pixels) / total_pixels
    
    # 如果 mask 占比超过阈值，直接返回白色 (根据你原有的逻辑)
    if mask_ratio > threshold:
        return (255, 255, 255)  
    
    # 【性能优化】：如果像素点太多，K-means 运行会很慢，随机抽取 10000 个点即可准确找到主色
    if len(masked_pixels) > 10000:
        indices = np.random.choice(len(masked_pixels), size=10000, replace=False)
        sample_pixels = masked_pixels[indices]
    else:
        sample_pixels = masked_pixels
        
    # 处理颜色种类极少的情况（例如纯色图像）
    unique_pixels = np.unique(sample_pixels, axis=0)
    if len(unique_pixels) < n_clusters:
        n_clusters = len(unique_pixels)
        
    # 运行 K-means 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(sample_pixels)
    
    # 【核心改进】：统计每个聚类的像素数量，找到包含像素最多的类
    counts = Counter(labels)
    dominant_cluster_label = counts.most_common(1)[0][0]
    
    # 获取该类的颜色中心，并转换为整数
    main_color = kmeans.cluster_centers_[dominant_cluster_label].astype(int)
    
    return tuple(main_color)

def color_similarity_lab(color1: torch.Tensor, color2: torch.Tensor, device: torch.device) -> float:
    """
    计算两个颜色之间的 CIELAB 感知色差 Delta E。
    结合 CIELAB 与人类加权感知，消灭单纯 RGB 欧氏距离在蓝色/深色区域的感知盲区。
    返回：标量 Delta E (0~100 scale，通常 < 8.0 属于紧密同色系，< 2.3 属于人眼不可分辨)
    """
    c1 = color1.detach().cpu().numpy().reshape(-1)[:3]
    c2 = color2.detach().cpu().numpy().reshape(-1)[:3]

    # 若输入为 [0, 1] 浮点，映射至 [0, 255]
    if np.max(c1) <= 1.05 and np.max(c2) <= 1.05:
        rgb1_u8 = np.clip(c1 * 255.0, 0, 255).astype(np.uint8).reshape(1, 1, 3)
        rgb2_u8 = np.clip(c2 * 255.0, 0, 255).astype(np.uint8).reshape(1, 1, 3)
    else:
        rgb1_u8 = np.clip(c1, 0, 255).astype(np.uint8).reshape(1, 1, 3)
        rgb2_u8 = np.clip(c2, 0, 255).astype(np.uint8).reshape(1, 1, 3)

    # 转换至 CIELAB 空间
    lab1 = cv2.cvtColor(rgb1_u8, cv2.COLOR_RGB2LAB).astype(float)[0, 0]
    lab2 = cv2.cvtColor(rgb2_u8, cv2.COLOR_RGB2LAB).astype(float)[0, 0]

    # CIELAB Delta E (CIE76)
    delta_e = float(np.linalg.norm(lab1 - lab2))
    return delta_e


def color_similarity(color1, color2, device):
    """
    兼容既有接口：调用高精度 CIELAB 感知距离 (返回归一化至 0~1 的度量)。
    """
    delta_e = color_similarity_lab(color1, color2, device)
    return delta_e / 100.0

def is_mask_included(current_mask, existing_mask, inclusion_threshold=0.8):
    """
    判断 current_mask 的大部分是否被 existing_mask 包含，基于交集和最小面积的比值进行判断。
    如果交集和较小 mask 的比值大于 inclusion_threshold，则认为 current_mask 被包含。
    
    参数：
        current_mask: 当前 mask，二值化后的 numpy 数组。
        existing_mask: 已有 mask，二值化后的 numpy 数组。
        inclusion_threshold: 包含判断的阈值，交集和较小面积的比值大于此值时，认为当前 mask 被包含。
    
    返回：
        bool: 如果 current_mask 被 existing_mask 完全包含，返回 True，否则返回 False。
    """
    # 将当前 mask 和已有 mask 转换为二值图
    current_mask_binary = (current_mask > 0).astype(np.uint8)
    existing_mask_binary = (existing_mask > 0).astype(np.uint8)
    if current_mask_binary.shape != existing_mask_binary.shape:
        existing_mask_binary = cv2.resize(existing_mask_binary, (current_mask_binary.shape[1], current_mask_binary.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 计算交集区域
    intersection = cv2.bitwise_and(current_mask_binary, existing_mask_binary)

    # 计算交集的面积
    intersection_area = np.sum(intersection)

    # 计算当前 mask 和已有 mask 的面积
    current_area = np.sum(current_mask_binary)
    existing_area = np.sum(existing_mask_binary)

    # 获取较小的 mask 面积
    smaller_area = min(current_area, existing_area)

    # 防止除零错误
    if smaller_area == 0:
        return False

    # 计算交集和较小面积的比值
    inclusion_ratio = intersection_area / smaller_area

    # 判断比值是否大于阈值
    return inclusion_ratio >= inclusion_threshold


def render_svg_to_jpg(svg_path, output_jpg_path, width, height, background_color=(255, 255, 255), preserve_aspect_ratio=True):
    """
    Render an SVG file to a JPG image with specified width and height, scaling content to fill the canvas.
    
    Args:
        svg_path (str): Path to the input SVG file.
        output_jpg_path (str): Path to save the output JPG file.
        width (int): Desired width of the output image in pixels.
        height (int): Desired height of the output image in pixels.
        background_color (tuple): RGB color for the background (default: white, (255, 255, 255)).
        preserve_aspect_ratio (bool): If True, scale SVG content to preserve aspect ratio; if False, stretch to fill.
    
    Returns:
        bool: True if rendering is successful, False otherwise.
    """
    try:
        # Set device
        pydiffvg.set_use_gpu(torch.cuda.is_available())
        
        # Load SVG file
        canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(svg_path)
        
        # Calculate scaling factors
        scale_x = width / canvas_width
        scale_y = height / canvas_height
        
        if preserve_aspect_ratio:
            # Use the smaller scale to avoid stretching
            scale = min(scale_x, scale_y)
            scale_x = scale
            scale_y = scale
            # Center the content
            offset_x = (width - canvas_width * scale_x) / 2
            offset_y = (height - canvas_height * scale_y) / 2
        else:
            # Stretch to fill the canvas
            offset_x = 0
            offset_y = 0
        
        # Scale shapes and paths
        for shape in shapes:
            if hasattr(shape, 'points'):
                # Scale path points
                shape.points[:, 0] = shape.points[:, 0] * scale_x + offset_x
                shape.points[:, 1] = shape.points[:, 1] * scale_y + offset_y
            if hasattr(shape, 'stroke_width'):
                # Scale stroke width (optional)
                shape.stroke_width *= min(scale_x, scale_y)
        
        # Create rendering scene with target resolution
        scene = pydiffvg.RenderFunction.serialize_scene(
            width, height, shapes, shape_groups
        )
        
        # Initialize renderer
        render = pydiffvg.RenderFunction.apply
        
        # Render SVG to tensor
        img = render(
            width,           # render width
            height,          # render height
            2,               # num_samples_x
            2,               # num_samples_y
            0,               # seed
            None,            # background_image
            *scene
        )
        
        # Convert tensor to numpy array
        img = img[:, :, :3].cpu().numpy()  # Remove alpha channel, keep RGB
        img = (img * 255).astype(np.uint8)  # Scale to 0-255
        
        # Create background image
        background = np.ones((height, width, 3), dtype=np.uint8) * np.array(background_color, dtype=np.uint8)
        
        # Blend image with background (handle transparency)
        alpha = img[:, :, 3:4] / 255.0 if img.shape[-1] == 4 else np.ones((height, width, 1))
        blended_img = (img[:, :, :3] * alpha + background * (1 - alpha)).astype(np.uint8)
        
        # Save as JPG
        pil_img = Image.fromarray(blended_img)
        pil_img.save(output_jpg_path, "JPEG", quality=95)
        
        print(f"Rendered SVG to JPG: {output_jpg_path}")
        return True
    
    except Exception as e:
        print(f"Error rendering SVG to JPG: {str(e)}")
        return False  

def compute_path_point_nums(shapes) :
    cnt = 0
    for path in  shapes :
        cnt += len(path.points)
    
    return cnt

def add_to_file(data_to_add, timing_file):
    data = {}
    # read all data that you have so far:
    if os.path.exists(timing_file):
        with open(timing_file, 'r') as f:
            data = json.load(f)
    # update dict:
    for k in data_to_add:
        data[k] = data_to_add[k]
    # write dict to file:
    with open(timing_file, 'w') as f:
        json.dump(data, f, indent=2)


def mixed_path_xing_loss(shapes, scale=1e-3, eps=1e-8):
    """
    针对直线和三阶贝塞尔混合路径的向量化 Xing Loss
    
    参数:
        shapes: pydiffvg.Path 对象的列表
        scale: 损失的缩放权重
        eps: 防止除以 0 的微小常量
    返回:
        标量 Tensor，表示所有贝塞尔曲线的平均 Xing Loss
    """
    beziers = []

    # 1. 解析所有 path，提取出所有的三阶贝塞尔控制多边形 (4个点)
    for path in shapes:
        points = path.points
        n_points = points.shape[0]
        num_control_points = path.num_control_points
        
        # 确保数据在正确的设备上
        device = points.device

        idx = 0
        for n in num_control_points:
            # 如果是三阶贝塞尔曲线
            if n == 2: 
                # 提取 4 个点: 起点, 控制点1, 控制点2, 终点
                # 使用 % n_points 是为了完美兼容 closed=True 时的首尾相接
                p0 = points[idx]
                p1 = points[(idx + 1) % n_points]
                p2 = points[(idx + 2) % n_points]
                p3 = points[(idx + 3) % n_points]
                beziers.append(torch.stack([p0, p1, p2, p3]))
                idx += 3
            # 如果是直线段，跳过（因为直线不可能自身打结）
            elif n == 0: 
                idx += 1
            # 兼容其他可能的格式，如二阶贝塞尔
            elif n == 1:
                idx += 2

    # 如果画面中全是直线，没有贝塞尔曲线，直接返回 0 梯度
    if not beziers:
        # 获取任意一个 device
        default_device = shapes[0].points.device if shapes else torch.device('cpu')
        return torch.tensor(0.0, device=default_device, requires_grad=True)

    # 2. 拼接所有的贝塞尔曲线进行并行计算
    # beziers 形状: [B, 4, 2], B 是所有 path 中三阶贝塞尔段的总数
    beziers = torch.stack(beziers)

    # 提取控制多边形的三条线段向量 [B, 2]
    v1 = beziers[:, 1, :] - beziers[:, 0, :]
    v2 = beziers[:, 2, :] - beziers[:, 1, :]
    v3 = beziers[:, 3, :] - beziers[:, 2, :]

    # 3. 向量化的角度正弦计算
    def compute_sin(vec_a, vec_b):
        """
        计算两个向量夹角的正弦值：
        $ \sin(\theta) = \frac{\vec{v_a} \times \vec{v_b}}{|\vec{v_a}| |\vec{v_b}|} $
        """
        # 2D 向量叉乘: a_x * b_y - a_y * b_x
        cross = vec_a[:, 0] * vec_b[:, 1] - vec_a[:, 1] * vec_b[:, 0]
        norm_a = torch.norm(vec_a, dim=1)
        norm_b = torch.norm(vec_b, dim=1)
        return cross / (norm_a * norm_b + eps)

    # 计算 v1 与 v2，以及 v1 与 v3 的正弦值
    sin_12 = compute_sin(v1, v2)
    sin_13 = compute_sin(v1, v3)

    # 4. 判断折叠方向并计算 Loss
    # 如果 v1->v2 是逆时针 (sin >= 0)，direct=1；否则顺时针 opst=1
    direct = (sin_12 >= 0).float()
    opst = 1.0 - direct

    # 惩罚项：如果 v1->v3 发生了与 v1->v2 相反方向的弯折，则产生 loss
    loss = direct * torch.relu(-sin_13) + opst * torch.relu(sin_13)

    # 返回平均 loss 并缩放
    return loss.mean() * scale

def laplacian_loss(shapes, scale=1.0):
    """
    拉普拉斯平滑损失：惩罚控制点的剧烈抖动，促使线条柔和顺滑。
    """
    loss = 0.0
    for path in shapes:
        points = path.points
        if len(points) < 3:
            continue
        
        # 计算相邻三个点的二阶差分 (P_{i-1} - 2P_i + P_{i+1})
        laplacian = points[:-2] - 2 * points[1:-1] + points[2:]
        
        # 考虑到闭合路径，首尾也要计算
        if path.is_closed:
            lap_first = points[-1:] - 2 * points[0:1] + points[1:2]
            lap_last = points[-2:-1] - 2 * points[-1:] + points[0:1]
            laplacian = torch.cat([lap_first, laplacian, lap_last], dim=0)
            
        # 使用 L2 范数
        loss += torch.mean(laplacian ** 2)
        
    return (loss / len(shapes)) * scale

def arc_length_loss(shapes, scale=1.0):
    """
    弧长惩罚：促使算法用最短的路径完成拟合，抑制无意义的蜿蜒曲折。
    """
    loss = 0.0
    for path in shapes:
        points = path.points
        if len(points) < 2:
            continue
            
        # 计算相邻点之间的距离
        segments = points[1:] - points[:-1]
        
        if path.is_closed:
            seg_closed = points[0:1] - points[-1:]
            segments = torch.cat([segments, seg_closed], dim=0)
            
        # 计算总长度
        length = torch.norm(segments, dim=1)
        loss += torch.mean(length)  # 使用 mean 防止节点越多的路径 loss 越大
        
    return (loss / len(shapes)) * scale

def uniform_spacing_loss(shapes, scale=1.0):
    """
    均匀间距损失：防止控制点在局部过度扎堆，提高曲线的参数化优雅度。
    """
    loss = 0.0
    for path in shapes:
        points = path.points
        if len(points) < 3:
            continue
            
        segments = points[1:] - points[:-1]
        if path.is_closed:
            seg_closed = points[0:1] - points[-1:]
            segments = torch.cat([segments, seg_closed], dim=0)
            
        # 计算每段的长度
        lengths = torch.norm(segments, dim=1)
        
        # 惩罚长度的方差
        mean_length = torch.mean(lengths)
        variance = torch.mean((lengths - mean_length) ** 2)
        
        # 为了尺度不变性，除以均值的平方
        loss += variance / (mean_length ** 2 + 1e-8)
        
    return (loss / len(shapes)) * scale

def angle_smoothness_loss(shapes, scale=1.0):
    """
    锐角惩罚损失：防止出现极端的折返角。
    """
    loss = 0.0
    valid_paths = 0
    for path in shapes:
        points = path.points
        if len(points) < 3:
            continue
            
        # 获取线段向量
        v = points[1:] - points[:-1]
        if path.is_closed:
            v_closed = points[0:1] - points[-1:]
            v = torch.cat([v, v_closed], dim=0)
            
        # 归一化向量
        v_norm = v / (torch.norm(v, dim=1, keepdim=True) + 1e-8)
        
        # 计算相邻向量的点积 (余弦值)
        # cosine = 1 表示完全平滑(直线)，cosine = -1 表示 180度折返
        cosine = torch.sum(v_norm[:-1] * v_norm[1:], dim=1)
        
        if path.is_closed:
            cos_closed = torch.sum(v_norm[-1:] * v_norm[0:1], dim=1)
            cosine = torch.cat([cosine, cos_closed], dim=0)
            
        # 我们希望 cosine 尽量接近 1。惩罚 (1 - cosine)
        loss += torch.mean(1.0 - cosine)
        valid_paths += 1
        
    if valid_paths == 0: return 0.0
    return (loss / valid_paths) * scale

def collinear_handle_loss(shapes, scale=1e-2, cos_threshold=0.5, eps=1e-8):
    """
    控制点共线损失 (G1 连续性)：
    促使锚点两侧的控制点与锚点三点一线，形成丝滑的曲线。
    同时保护刻意的锐角转折不被圆滑化。
    """
    loss = 0.0
    valid_count = 0

    for path in shapes:
        points = path.points
        n_points = points.shape[0]
        num_control_points = path.num_control_points

        triplets = []
        idx = 0

        # 遍历寻找相邻的贝塞尔曲线段
        for i in range(len(num_control_points)):
            n = num_control_points[i]
            # 获取下一段的控制点数量 (处理闭合曲线的首尾相连)
            next_n = num_control_points[(i + 1) % len(num_control_points)]

            # 只有当当前段和下一段都是三阶贝塞尔 (n==2) 时，才存在平滑过渡的可能
            if n == 2 and next_n == 2:
                # 在 pydiffvg 中，闭合曲线的最后一个控制点连接回 points[0]
                # 这里利用取余操作完美解决跨越首尾的索引问题
                anchor_idx = (idx + 3) % n_points
                c_in_idx = (idx + 2) % n_points
                c_out_idx = (idx + 4) % n_points

                c_in = points[c_in_idx]
                anchor = points[anchor_idx]
                c_out = points[c_out_idx]

                triplets.append(torch.stack([c_in, anchor, c_out]))

            # 更新索引指针
            if n == 2: idx += 3
            elif n == 0: idx += 1
            elif n == 1: idx += 2

        if not triplets:
            continue

        triplets = torch.stack(triplets) # 形状: (K, 3, 2)
        c_in = triplets[:, 0, :]
        anchor = triplets[:, 1, :]
        c_out = triplets[:, 2, :]

        # 1. 计算出入向量
        v_in = anchor - c_in
        v_out = c_out - anchor

        norm_in = torch.norm(v_in, dim=1)
        norm_out = torch.norm(v_out, dim=1)

        # 2. 过滤掉重合的点 (避免除以 0 导致 NaN)
        valid_mask = (norm_in > eps) & (norm_out > eps)
        if not valid_mask.any():
            continue

        v_in = v_in[valid_mask]
        v_out = v_out[valid_mask]
        norm_in = norm_in[valid_mask]
        norm_out = norm_out[valid_mask]

        # 3. 计算余弦相似度 [-1, 1]
        cos_sim = torch.sum(v_in * v_out, dim=1) / (norm_in * norm_out)

        # 4. 【智能保护逻辑】
        # 只有当 cos_sim 大于阈值 (例如 0.5，即角度小于 60 度) 时，
        # 才认为它是一条"尝试平滑但没做好"的曲线，对其施加惩罚拉直至 1.0。
        # 如果 cos_sim 小于阈值，说明是刻意的锐角，loss 为 0。
        smooth_mask = (cos_sim > cos_threshold).float()
        
        current_loss = smooth_mask * (1.0 - cos_sim)
        
        loss += current_loss.sum()
        valid_count += current_loss.size(0)

    # 兜底：如果全图没有符合条件的平滑连接点
    if valid_count == 0:
        default_device = shapes[0].points.device if shapes else torch.device('cpu')
        return torch.tensor(0.0, device=default_device, requires_grad=True)

    return (loss / valid_count) * scale