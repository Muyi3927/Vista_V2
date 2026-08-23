"""
VISTA FastAPI Web 交互服务 (app.py)
提供静态前端展示与 `/process` 图像矢量化 REST API 接口（开放全面可调参数，支持多视图全景与图层查看）。
"""
from __future__ import annotations

import os

# 清除 socks 代理避免 httpx/fastapi 报错
for p_key in ("all_proxy", "ALL_PROXY", "socks_proxy", "SOCKS_PROXY"):
    os.environ.pop(p_key, None)

import logging
import shutil
import tempfile
import time
import uuid
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app_main import img_to_svg_full
from config import get_config

app = FastAPI(title="VISTA Web Studio", version="1.0.0")

# CORS 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("VISTA_Server")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(console_handler)
logger.propagate = False

_CFG = get_config()
TEMP_OUTPUTS_DIR = os.path.abspath(_CFG["project"]["temp_outputs"])
os.makedirs(TEMP_OUTPUTS_DIR, exist_ok=True)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

app.mount("/temp_outputs", StaticFiles(directory=TEMP_OUTPUTS_DIR), name="temp_outputs")


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """主页渲染。"""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.isfile(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>VISTA 矢量化 Web 服务正在运行</h1>"


@app.get("/style.css")
async def serve_root_css():
    """样式表根路由兼容。"""
    css_file = os.path.join(STATIC_DIR, "style.css")
    if os.path.isfile(css_file):
        return FileResponse(css_file, media_type="text/css")
    return HTMLResponse(content="", media_type="text/css")


@app.get("/app.js")
async def serve_root_js():
    """脚本根路由兼容。"""
    js_file = os.path.join(STATIC_DIR, "app.js")
    if os.path.isfile(js_file):
        return FileResponse(js_file, media_type="application/javascript")
    return HTMLResponse(content="", media_type="application/javascript")


@app.get("/logo.svg")
async def serve_root_logo():
    """Logo 根路由兼容。"""
    logo_file = os.path.join(STATIC_DIR, "logo.svg")
    if os.path.isfile(logo_file):
        return FileResponse(logo_file, media_type="image/svg+xml")
    return HTMLResponse(content="", media_type="image/svg+xml")


def delete_file_after_delay(file_path: str, delay: int = 1800):
    """延迟清理临时生成的文件或目录（默认30分钟）。"""
    time.sleep(delay)
    try:
        if os.path.isdir(file_path):
            shutil.rmtree(file_path, ignore_errors=True)
        elif os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        logger.warning(f"清理临时文件失败 {file_path}: {e}")


@app.get("/config/defaults")
async def get_default_config():
    """获取 YAML 默认配置字典。"""
    cfg = get_config()
    sam_c = cfg.get("sam", {})
    slic_c = cfg.get("slic", {})
    pre_c = cfg.get("preprocess", {})
    pf_c = cfg.get("path_fit", {})
    opt_c = cfg.get("optimize", {})
    prune_c = cfg.get("prune", {})

    return JSONResponse(content={
        "target_size": pre_c.get("target_size", 0),
        "slic_n_segments": slic_c.get("n_segments", 2000),
        "slic_compactness": slic_c.get("compactness", 5.0),
        "dbscan_eps": slic_c.get("dbscan_eps", 5.0),
        "use_sam": sam_c.get("enabled", True),
        "pred_iou_thresh": sam_c.get("pred_iou_thresh", 0.88),
        "stability_score_thresh": sam_c.get("stability_score_thresh", 0.95),
        "points_per_side": sam_c.get("points_per_side", 64),
        "min_area": pre_c.get("min_area", 0.00015),
        "iou_sam_slic_thresh": pre_c.get("iou_sam_slic_thresh", 0.90),
        "iou_sam_internal_thresh": pre_c.get("iou_threshold", 0.90),
        "parent_contain_thresh": pre_c.get("inclusion_threshold", 0.90),
        "self_pure_std_thresh": pre_c.get("self_pure_std_thresh", 15.0),
        "parent_pure_std_thresh": pre_c.get("parent_pure_std_thresh", 3.0),
        "color_diff_thresh": pre_c.get("color_diff_thresh", 5.0),
        "bzer_max_error": pf_c.get("bezier_max_error", 1.5),
        "line_threshold": pf_c.get("line_threshold", 2.0),
        "learning_rate": opt_c.get("learning_rate", 0.10),
        "num_iters": opt_c.get("num_iters", 1000),
        "early_stopping_patience": opt_c.get("early_stopping_patience", 20),
        "prune_enabled": prune_c.get("enabled", True),
        "rm_color_threshold": prune_c.get("rm_color_threshold", 0.02),
        "refine_iters": prune_c.get("refine_iters", 80),
        "is_stroke": opt_c.get("is_stroke", False),
    })


def _to_web_url(abs_path: str) -> str:
    """将输出路径转换为可访问的静态 URL。"""
    if not abs_path:
        return ""
    norm_path = os.path.abspath(abs_path)
    if norm_path.startswith(TEMP_OUTPUTS_DIR):
        rel = os.path.relpath(norm_path, TEMP_OUTPUTS_DIR).replace("\\", "/")
        return f"/temp_outputs/{rel}"
    return ""


@app.post("/process")
async def process_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    # 1. 基础与预处理
    target_size: int = Form(0),
    # 2. SLIC 超像素
    slic_n_segments: int = Form(2000),
    slic_compactness: float = Form(5.0),
    dbscan_eps: float = Form(5.0),
    # 3. SAM 语义先验
    use_sam: bool = Form(True),
    pred_iou_thresh: float = Form(0.88),
    stability_score_thresh: float = Form(0.95),
    crop_n_layers: int = Form(1),
    points_per_side: int = Form(64),
    # 4. 层级融合与同色吸收
    min_area: float = Form(0.00015),
    iou_sam_slic_thresh: float = Form(0.90),
    iou_sam_internal_thresh: float = Form(0.90),
    parent_contain_thresh: float = Form(0.90),
    self_pure_std_thresh: float = Form(15.0),
    parent_pure_std_thresh: float = Form(3.0),
    color_diff_thresh: float = Form(5.0),
    # 5. 几何拟合
    bzer_max_error: float = Form(0.003),
    line_threshold: float = Form(0.004),
    poly_epsilon: Optional[float] = Form(None),
    # 6. DiffVG 优化与剪枝
    learning_rate: float = Form(0.10),
    color_lr: float = Form(0.01),
    num_iters: int = Form(1000),
    early_stopping_patience: int = Form(20),
    collinear_scale: float = Form(0.01),
    is_stroke: bool = Form(False),
    prune_enabled: bool = Form(True),
    rm_color_threshold: float = Form(0.02),
    refine_iters: int = Form(80),
):
    """接收图像上传并执行矢量化处理的 HTTP POST 接口。"""
    logger.info(f"收到图像上传请求: {file.filename}")
    original_filename = os.path.splitext(file.filename)[0]
    original_ext = os.path.splitext(file.filename)[1] or ".png"
    unique_id = str(uuid.uuid4())[:8]
    base_filename = f"{original_filename}_{unique_id}"

    temp_dir = tempfile.gettempdir()
    temp_img_path = os.path.join(temp_dir, f"{base_filename}{original_ext}")

    try:
        with open(temp_img_path, "wb") as temp_img:
            shutil.copyfileobj(file.file, temp_img)
    except Exception as e:
        logger.error(f"保存上传临时文件失败: {e}")
        raise HTTPException(status_code=500, detail=f"保存文件失败: {str(e)}")

    try:
        res = img_to_svg_full(
            temp_img_path,
            target_size=target_size,
            slic_n_segments=slic_n_segments,
            slic_compactness=slic_compactness,
            dbscan_eps=dbscan_eps,
            use_sam=use_sam,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            crop_n_layers=crop_n_layers,
            points_per_side=points_per_side,
            min_area=min_area,
            iou_sam_slic_thresh=iou_sam_slic_thresh,
            iou_sam_internal_thresh=iou_sam_internal_thresh,
            parent_contain_thresh=parent_contain_thresh,
            self_pure_std_thresh=self_pure_std_thresh,
            parent_pure_std_thresh=parent_pure_std_thresh,
            color_diff_thresh=color_diff_thresh,
            bzer_max_error=bzer_max_error,
            line_threshold=line_threshold,
            poly_epsilon=poly_epsilon,
            learning_rate=learning_rate,
            color_lr=color_lr,
            num_iters=num_iters,
            early_stopping_patience=early_stopping_patience,
            collinear_scale=collinear_scale,
            is_stroke=is_stroke,
            prune_enabled=prune_enabled,
            rm_color_threshold=rm_color_threshold,
            refine_iters=refine_iters,
        )

        svg_url = _to_web_url(res["svg_path"])
        gif_url = _to_web_url(res["gif_path"])

        overviews_url = {
            k: _to_web_url(v) for k, v in res.get("overviews", {}).items()
        }

        layers_url = []
        for item in res.get("layers", []):
            layers_url.append({
                "name": item["name"],
                "bw_url": _to_web_url(item["bw_path"]),
                "colored_url": _to_web_url(item["colored_path"]),
            })

        # 注册后台清理任务
        background_tasks.add_task(delete_file_after_delay, res["run_dir"], delay=1800)
        background_tasks.add_task(delete_file_after_delay, temp_img_path, delay=1800)

        return JSONResponse(content={
            "svg_url": svg_url,
            "gif_url": gif_url,
            "overviews": overviews_url,
            "layers": layers_url,
            "stats": res["stats"],
            "meta": res.get("meta", {}),
        })
    except Exception as e:
        logger.error(f"矢量化执行失败: {e}")
        return JSONResponse(status_code=500, content={"detail": f"矢量化处理失败: {str(e)}"})


@app.get("/download")
async def download_file(file_url: str):
    """文件下载路由。"""
    if file_url.startswith("/temp_outputs/"):
        rel = file_url[len("/temp_outputs/"):]
        target = os.path.join(TEMP_OUTPUTS_DIR, rel)
        if os.path.isfile(target):
            return FileResponse(target, filename=os.path.basename(target))
    raise HTTPException(status_code=404, detail="文件不存在或已过期")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)