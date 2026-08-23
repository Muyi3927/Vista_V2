#!/usr/bin/env python3
"""
VISTA Studio — Gradio 独立可视化工作台 (studio.py)
包含：
- Tab 1: 图像输入与预处理 (Target Size, 尺寸设置)
- Tab 2: SAM + SLIC 混合分割与分层融合 (超参数调节、origin_masks与pre_masks彩图看板、图层画廊)
- Tab 3: 几何拟合与 DiffVG 优化 (贝塞尔误差、学习率、迭代轮数、剪枝、最终 SVG 与 GIF 对比展示)

启动方式：
  conda activate vista && cd src && python studio.py
"""
from __future__ import annotations

import os

# 清除 socks 代理避免 httpx/gradio 启动报错
for p_key in ("all_proxy", "ALL_PROXY", "socks_proxy", "SOCKS_PROXY"):
    os.environ.pop(p_key, None)

import shutil
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
from PIL import Image as PILImage

from config import get_config, load_config
from pipeline import create_job, stage_segment, stage_vectorize
from utils import compute_path_point_nums, load_and_resize

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _allowed_paths() -> List[str]:
    """返回允许 Gradio 读取与缓存文件的所有路径列表。"""
    cfg = get_config()
    paths = [
        _PROJECT_ROOT,
        os.path.abspath(cfg["project"]["temp_outputs"]),
        tempfile.gettempdir(),
    ]
    out = (cfg.get("paths") or {}).get("output")
    if out:
        paths.append(os.path.abspath(out))
    return paths


def _load_overview_img(path: str) -> Optional[np.ndarray]:
    if os.path.isfile(path):
        bgr = cv2.imread(path)
        if bgr is not None:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return None


def _make_mask_gallery(run_dir: str) -> List[np.ndarray]:
    pre_dir = os.path.join(run_dir, "pre_colored_masks")
    if not os.path.isdir(pre_dir):
        pre_dir = os.path.join(run_dir, "pre_masks")
    if not os.path.isdir(pre_dir):
        return []
    thumbs = []
    for f in sorted(os.listdir(pre_dir)):
        if f.endswith(".png"):
            p = os.path.join(pre_dir, f)
            img = cv2.imread(p)
            if img is not None:
                thumbs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return thumbs


import re


def _format_svg_for_web(svg_text: str) -> str:
    """对 SVG 标签注入 viewBox、100% 相对尺寸与 preserveAspectRatio，保证在各种容器内完整展示。"""
    w_m = re.search(r'width="([^"]+)"', svg_text)
    h_m = re.search(r'height="([^"]+)"', svg_text)
    w = float(w_m.group(1)) if w_m else 512.0
    h = float(h_m.group(1)) if h_m else 512.0

    def repl_svg_tag(match):
        attrs = match.group(0)
        if "viewBox" not in attrs:
            attrs = attrs.replace("<svg", f'<svg viewBox="0 0 {w} {h}"', 1)
        attrs = re.sub(r'\bwidth="[^"]*"', 'width="100%"', attrs)
        attrs = re.sub(r'\bheight="[^"]*"', 'height="100%"', attrs)
        if "preserveAspectRatio" not in attrs:
            attrs = attrs.replace("<svg", '<svg preserveAspectRatio="xMidYMid meet"', 1)
        if "style=" not in attrs:
            attrs = attrs.replace("<svg", '<svg style="width:100%;height:100%;max-width:100%;max-height:100%;object-fit:contain;display:block;"', 1)
        return attrs

    return re.sub(r"<svg[^>]*>", repl_svg_tag, svg_text, count=1)


def _svg_html(svg_path: Optional[str]) -> str:
    if not svg_path or not os.path.isfile(svg_path):
        return "<div style='height:380px;display:flex;align-items:center;justify-content:center;background:#f8fafc;color:#94a3b8;border:1px dashed #cbd5e1;border-radius:8px;'>等待矢量化生成 SVG...</div>"
    try:
        with open(svg_path, "r", encoding="utf-8") as f:
            svg_content = f.read()

        formatted_svg = _format_svg_for_web(svg_content)
        return f"""
        <div style="width:100%;height:380px;display:flex;align-items:center;justify-content:center;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;padding:8px;box-sizing:border-box;background-image:linear-gradient(45deg, #f1f5f9 25%, transparent 25%), linear-gradient(-45deg, #f1f5f9 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #f1f5f9 75%), linear-gradient(-45deg, transparent 75%, #f1f5f9 75%);background-size:16px 16px;">
            <div style="width:100%;height:100%;display:flex;align-items:center;justify-content:center;overflow:hidden;">
                {formatted_svg}
            </div>
        </div>
        """
    except Exception as e:
        return f"<div style='color:red;'>读取 SVG 失败: {e}</div>"


def _gif_html(gif_path: Optional[str]) -> str:
    if not gif_path or not os.path.isfile(gif_path):
        return "<div style='padding:24px;color:#94a3b8;text-align:center;background:#f8fafc;border:1px dashed #cbd5e1;border-radius:8px;'>暂无动画（完成矢量化后显示，点击画面可重播）</div>"
    abs_p = os.path.abspath(gif_path)
    return f"""
    <div style="text-align:center;padding:12px;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;background-image:linear-gradient(45deg, #f1f5f9 25%, transparent 25%), linear-gradient(-45deg, #f1f5f9 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #f1f5f9 75%), linear-gradient(-45deg, transparent 75%, #f1f5f9 75%);background-size:16px 16px;">
      <p style="margin:0 0 8px;color:#64748b;font-size:13px;font-weight:500;">点击画面重新播放 DiffVG 优化演化过程动画</p>
      <img src="/gradio_api/file={abs_p}" alt="Evolution Animation"
           style="max-width:100%;max-height:420px;border-radius:6px;border:1px solid #cbd5e1;cursor:pointer;box-shadow:0 2px 4px rgba(0,0,0,0.06);"
           onclick="const b=this.src.split('?')[0]; this.src=''; this.src=b+'?t='+Date.now();" />
    </div>
    """


# ---------------------------------------------------------------------------
# UI Callbacks
# ---------------------------------------------------------------------------

def _composite_white_bg(src_path: str) -> str:
    """
    若图像含 Alpha 通道（或 Gradio 将 RGBA 保存为黑底 RGB），
    将其与纯白底板合成后保存为临时 PNG，返回新路径。
    对于真正的不透明 RGB 图像直接返回原路径。
    """
    try:
        im = PILImage.open(src_path)
    except Exception:
        return src_path

    # 判断是否需要合成：RGBA/LA/PA 以及带 transparency 的 P 模式
    has_alpha = im.mode in ("RGBA", "LA", "PA") or (
        im.mode == "P" and "transparency" in im.info
    ) or "A" in im.getbands()

    # 即使已经是 RGB，若四角接近黑色且原文件是 PNG（可能 Gradio 丢失了 alpha），
    # 仍尝试从 Gradio 缓存中找到原始 RGBA 文件重新合成
    if not has_alpha and im.mode == "RGB":
        # 仅当文件名后缀是 png/webp 时才检查四角
        import numpy as _np
        _arr = _np.array(im)
        _corners = [_arr[0, 0], _arr[0, -1], _arr[-1, 0], _arr[-1, -1]]
        _all_black = all((_c < 10).all() for _c in _corners)
        if not _all_black:
            return src_path  # 正常不透明图像，直接返回

    # 做白底 Alpha 合成
    rgba = im.convert("RGBA")
    white_bg = PILImage.new("RGBA", rgba.size, (255, 255, 255, 255))
    composited = PILImage.alpha_composite(white_bg, rgba).convert("RGB")

    # 保存到临时文件（使用原文件名后缀以便 save_target_image 正确命名）
    import uuid as _uuid
    orig_stem = os.path.splitext(os.path.basename(src_path))[0]
    tmp_path = os.path.join(
        tempfile.gettempdir(),
        f"{orig_stem}_wb_{_uuid.uuid4().hex[:6]}.png",
    )
    composited.save(tmp_path, format="PNG")
    return tmp_path


def ui_create_job(image, target_size: int):
    if image is None:
        raise gr.Error("请先上传目标图像！")
    cfg = get_config()
    temp_parent = cfg["project"]["temp_outputs"]
    os.makedirs(temp_parent, exist_ok=True)

    run_cfg = load_config(overrides={"preprocess": {"target_size": int(target_size)}})
    # 在传给 create_job 之前，先对透明/黑底图像做白色背景合成
    src_path = _composite_white_bg(str(image))
    job = create_job(src_path, base_out_dir=temp_parent, cfg=run_cfg)
    run_dir = job["run_dir"]
    target_rgb = job["target_rgb"]

    state = {
        "run_dir": run_dir,
        "target_rgb": target_rgb,
        "target_path": job["target_path"],
    }
    info = f"✅ 已成功创建任务工作区\nrun_dir = {run_dir}\n图像尺寸 = {target_rgb.shape[1]}×{target_rgb.shape[0]}"
    return state, target_rgb, run_dir, info


def ui_run_segment(
    state,
    use_sam: bool,
    slic_n_segments: int,
    slic_compactness: float,
    dbscan_eps: float,
    pred_iou_thresh: float,
    stability_score_thresh: float,
    points_per_side: int,
    min_area: float,
    iou_sam_slic_thresh: float,
    iou_sam_internal_thresh: float,
    parent_contain_thresh: float,
    self_pure_std_thresh: float,
    parent_pure_std_thresh: float,
    color_diff_thresh: float,
):
    if not state or not state.get("run_dir"):
        raise gr.Error("请先在 Tab 1 创建或加载 Job！")
    run_dir = state["run_dir"]
    target_rgb = state["target_rgb"]

    overrides = {
        "slic": {
            "n_segments": int(slic_n_segments),
            "compactness": float(slic_compactness),
            "dbscan_eps": float(dbscan_eps),
        },
        "sam": {
            "enabled": bool(use_sam),
            "pred_iou_thresh": float(pred_iou_thresh),
            "stability_score_thresh": float(stability_score_thresh),
            "points_per_side": int(points_per_side),
        },
        "preprocess": {
            "min_area": float(min_area),
            "iou_sam_slic_thresh": float(iou_sam_slic_thresh),
            "iou_threshold": float(iou_sam_internal_thresh),
            "inclusion_threshold": float(parent_contain_thresh),
            "self_pure_std_thresh": float(self_pure_std_thresh),
            "parent_pure_std_thresh": float(parent_pure_std_thresh),
            "color_diff_thresh": float(color_diff_thresh),
        },
    }
    run_cfg = load_config(overrides=overrides)

    st_time = time.time()
    seg = stage_segment(run_dir, target_rgb=target_rgb, cfg=run_cfg)
    elapsed = time.time() - st_time

    slic_ov = _load_overview_img(os.path.join(run_dir, "slic_overview_colored.png"))
    sam_ov = _load_overview_img(os.path.join(run_dir, "sam_overview_colored.png"))
    orig_ov = _load_overview_img(os.path.join(run_dir, "origin_overview_colored.png"))
    pre_ov = _load_overview_img(os.path.join(run_dir, "pre_overview_colored.png"))

    gallery = _make_mask_gallery(run_dir)
    state["pre_masks"] = seg["pre_masks"]
    state["bg_color"] = seg["bg_color"]

    info = f"✅ 分割与融合完成！耗时: {elapsed:.2f}s | 提取优质图层: {seg['num_masks']} 个 | 背景底色: {seg['bg_color']}"
    return state, slic_ov, sam_ov, orig_ov, pre_ov, gallery, info


def ui_run_vectorize(
    state,
    bezier_max_error: float,
    line_threshold: float,
    learning_rate: float,
    num_iters: int,
    early_stopping_patience: int,
    prune_enabled: bool,
    rm_color_threshold: float,
    refine_iters: int,
    is_stroke: bool,
):
    if not state or not state.get("pre_masks"):
        raise gr.Error("请先执行 Tab 2 的分割与融合！")
    run_dir = state["run_dir"]
    target_rgb = state["target_rgb"]

    overrides = {
        "path_fit": {
            "bezier_max_error": float(bezier_max_error),
            "line_threshold": float(line_threshold),
        },
        "optimize": {
            "learning_rate": float(learning_rate),
            "num_iters": int(num_iters),
            "early_stopping_patience": int(early_stopping_patience),
            "is_stroke": bool(is_stroke),
        },
        "prune": {
            "enabled": bool(prune_enabled),
            "rm_color_threshold": float(rm_color_threshold),
            "refine_iters": int(refine_iters),
        },
    }
    run_cfg = load_config(overrides=overrides)

    vec = stage_vectorize(
        run_dir,
        target_rgb=target_rgb,
        pre_mask_paths=state["pre_masks"],
        bg_color=state.get("bg_color", (255, 255, 255)),
        cfg=run_cfg,
    )

    svg_p = vec.get("svg_path")
    gif_p = vec.get("gif_path")

    info = (
        f"✅ 矢量化与 DiffVG 优化完成！耗时: {vec.get('time_consuming', 0):.2f}s\n"
        f"矢量形状数 (Shapes): {vec.get('shapes')} | 控制点总数: {vec.get('path_point_nums')} | 最终 MSE: {vec.get('mse_loss')}\n"
        f"SVG 文件路径: {svg_p}"
    )

    svg_view = _svg_html(svg_p)
    gif_view = _gif_html(gif_p)
    return target_rgb, svg_view, gif_view, svg_p, gif_p, info


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------

def build_gradio_app():
    load_config()
    cfg = get_config()
    sam_cfg = cfg.get("sam", {})
    slic_cfg = cfg.get("slic", {})
    pre_cfg = cfg.get("preprocess", {})
    pf_cfg = cfg.get("path_fit", {})
    opt_cfg = cfg.get("optimize", {})
    prune_cfg = cfg.get("prune", {})

    with gr.Blocks(title="VISTA Studio — 交互工作台", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
# 🎨 VISTA Studio: 语义分割与可微渲染图像矢量化工作台
基于 **SAM 语义先验 + SLIC 超像素聚类** 结合 **纯度感知同色吸收** 与 **DiffVG 可微优化** 的端到端 SVG 矢量化系统。
"""
        )
        state = gr.State({})

        with gr.Tabs():
            # ----------------- Tab 1: Input -----------------
            with gr.Tab("1. 目标图像输入与预处理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(label="上传目标图像", type="filepath")
                        target_size = gr.Slider(0, 1536, value=int(pre_cfg.get("target_size", 0)), step=128, label="target_size 缩放长边 (0=保持原图分辨率)")
                        create_btn = gr.Button("📁 初始化任务 (Create Job)", variant="primary")
                    with gr.Column(scale=1):
                        target_preview = gr.Image(label="目标图预览 (Target RGB)", type="numpy")
                        run_dir_txt = gr.Textbox(label="任务工作区目录 (run_dir)", interactive=False)
                        job_info = gr.Textbox(label="任务状态日志", lines=3)

                create_btn.click(
                    ui_create_job,
                    inputs=[image_input, target_size],
                    outputs=[state, target_preview, run_dir_txt, job_info],
                )

            # ----------------- Tab 2: Segmentation -----------------
            with gr.Tab("2. SAM+SLIC 分割与层级融合"):
                gr.Markdown("### 分割与融合超参数调节 (默认读取 config/default.yaml)")
                with gr.Row():
                    use_sam = gr.Checkbox(value=bool(sam_cfg.get("enabled", True)), label="启用 SAM 语义分割先验")
                    slic_n = gr.Slider(500, 5000, value=int(slic_cfg.get("n_segments", 2000)), step=100, label="SLIC 超像素数")
                    slic_c = gr.Slider(1.0, 20.0, value=float(slic_cfg.get("compactness", 5.0)), step=0.5, label="SLIC 紧凑度")
                    dbscan_eps = gr.Slider(1.0, 20.0, value=float(slic_cfg.get("dbscan_eps", 5.0)), step=0.5, label="DBSCAN 聚类色差 eps")
                with gr.Row():
                    pred_iou = gr.Slider(0.5, 1.0, value=float(sam_cfg.get("pred_iou_thresh", 0.88)), step=0.01, label="SAM 预测 IoU 阈值")
                    stability = gr.Slider(0.5, 1.0, value=float(sam_cfg.get("stability_score_thresh", 0.95)), step=0.01, label="SAM 稳定性评分阈值")
                    points_per_side = gr.Slider(8, 64, value=int(sam_cfg.get("points_per_side", 64)), step=4, label="SAM 采样点密度")
                with gr.Row():
                    min_area = gr.Number(value=float(pre_cfg.get("min_area", 0.00015)), label="min_area 最小面积占比", precision=5)
                    iou_sam_slic = gr.Slider(0.7, 1.0, value=float(pre_cfg.get("iou_sam_slic_thresh", 0.90)), step=0.01, label="SAM 压制 SLIC 阈值 (IoU)")
                    iou_sam_internal = gr.Slider(0.7, 1.0, value=float(pre_cfg.get("iou_threshold", 0.90)), step=0.01, label="SAM 自去重阈值 (IoU)")
                    parent_contain = gr.Slider(0.7, 1.0, value=float(pre_cfg.get("inclusion_threshold", 0.90)), step=0.01, label="直接父级最小包含率")
                with gr.Row():
                    self_pure_std = gr.Number(value=float(pre_cfg.get("self_pure_std_thresh", 15.0)), label="自身纯度上限方差 (self_std)")
                    parent_pure_std = gr.Number(value=float(pre_cfg.get("parent_pure_std_thresh", 3.0)), label="父级纯度上限方差 (parent_std)")
                    color_diff = gr.Number(value=float(pre_cfg.get("color_diff_thresh", 5.0)), label="父子同色吸收色差 (color_diff)")

                seg_btn = gr.Button("▶ 运行分割与层级融合 (Stage 1-3)", variant="primary")
                seg_info = gr.Textbox(label="分割阶段汇总日志", lines=3)

                gr.Markdown("### 各阶段全景图看板")
                with gr.Row():
                    ov_slic = gr.Image(label="1. SLIC 超像素留洞全景", type="numpy")
                    ov_sam = gr.Image(label="2. SAM 语义先验全景", type="numpy")
                with gr.Row():
                    ov_orig = gr.Image(label="3. 孔洞实心化连通拆分 (origin)", type="numpy")
                    ov_pre = gr.Image(label="4. 终版纯度融合全景 (pre_masks)", type="numpy")

                gr.Markdown("### 最终图层分解画廊 (pre_masks)")
                gallery = gr.Gallery(label="pre_masks 图层列表", columns=4, height=320, object_fit="contain")

                seg_btn.click(
                    ui_run_segment,
                    inputs=[
                        state, use_sam, slic_n, slic_c, dbscan_eps,
                        pred_iou, stability, points_per_side, min_area,
                        iou_sam_slic, iou_sam_internal, parent_contain,
                        self_pure_std, parent_pure_std, color_diff,
                    ],
                    outputs=[state, ov_slic, ov_sam, ov_orig, ov_pre, gallery, seg_info],
                )

            # ----------------- Tab 3: Vectorize -----------------
            with gr.Tab("3. 几何拟合与 DiffVG 优化"):
                gr.Markdown("### 优化与拟合超参数调节 (默认读取 config/default.yaml)")
                with gr.Row():
                    bezier_max_error = gr.Slider(0.0005, 0.0100, value=float(pf_cfg.get("bezier_max_error", 0.003)), step=0.0005, label="贝塞尔拟合误差容差比例 (max_error)")
                    line_threshold = gr.Slider(0.0005, 0.0150, value=float(pf_cfg.get("line_threshold", 0.004)), step=0.0005, label="直线判断阈值比例 (line_thresh)")
                    learning_rate = gr.Slider(0.01, 0.30, value=float(opt_cfg.get("learning_rate", 0.10)), step=0.01, label="控制点坐标学习率 (lr)")
                    num_iters = gr.Slider(100, 2000, value=int(opt_cfg.get("num_iters", 1000)), step=50, label="DiffVG 优化最大轮数")
                with gr.Row():
                    patience = gr.Slider(5, 50, value=int(opt_cfg.get("early_stopping_patience", 20)), step=5, label="早停耐心轮数 (patience)")
                    prune_enabled = gr.Checkbox(value=bool(prune_cfg.get("enabled", True)), label="启用几何同色剪枝")
                    rm_color_threshold = gr.Slider(0.005, 0.05, value=float(prune_cfg.get("rm_color_threshold", 0.02)), step=0.005, label="剪枝同色阈值")
                    refine_iters = gr.Slider(0, 200, value=int(prune_cfg.get("refine_iters", 80)), step=10, label="剪枝后精修轮数 (refine_iters)")
                    is_stroke = gr.Checkbox(value=bool(opt_cfg.get("is_stroke", False)), label="包含描边优化 (is_stroke)")

                vec_btn = gr.Button("🚀 开始矢量化优化 (Run Vectorize)", variant="primary")
                vec_info = gr.Textbox(label="矢量化优化结果日志", lines=3)

                gr.Markdown("### 最终矢量成果比对")
                with gr.Row():
                    final_target = gr.Image(label="原始目标图像", type="numpy", height=380)
                    svg_html_box = gr.HTML(label="生成矢量图 (final.svg)", value=_svg_html(None))

                gr.Markdown("### DiffVG 优化演化过程动画 (点击画面重播)")
                with gr.Row():
                    gif_html_box = gr.HTML(label="优化过程动画 (animation.gif)", value=_gif_html(None))

                with gr.Row():
                    svg_download = gr.File(label="下载 final.svg")
                    gif_download = gr.File(label="下载 animation.gif")

                vec_btn.click(
                    ui_run_vectorize,
                    inputs=[
                        state, bezier_max_error, line_threshold, learning_rate,
                        num_iters, patience, prune_enabled, rm_color_threshold,
                        refine_iters, is_stroke,
                    ],
                    outputs=[final_target, svg_html_box, gif_html_box, svg_download, gif_download, vec_info],
                )

    return demo


if __name__ == "__main__":
    app = build_gradio_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        allowed_paths=_allowed_paths(),
    )
