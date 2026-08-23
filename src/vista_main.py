"""
VISTA 主批处理入口（基于 YAML 配置文件驱动）
使用方法：
  1. 修改 config/default.yaml（paths.input / paths.output / sam_checkpoint 等）
  2. 运行：
       cd src && python vista_main.py
"""
from __future__ import annotations

import json
import os
import shutil
import time
from typing import List, Tuple

from config import load_config
from pipeline import process_single_image

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _list_images(folder: str) -> List[str]:
    """列出目录下所有支持格式的图像文件路径。"""
    files = []
    for f in sorted(os.listdir(folder)):
        if os.path.splitext(f)[1].lower() in VALID_EXT:
            files.append(os.path.join(folder, f))
    return files


def _resolve_jobs(cfg: dict) -> Tuple[str, List[str]]:
    """根据配置解析单图模式或文件夹批量处理任务。"""
    paths = cfg.get("paths", {})
    input_path = paths.get("input") or paths.get("data")
    mode = (cfg.get("run") or {}).get("mode", "auto").strip().lower()

    if not input_path:
        raise ValueError("config paths.input 为空，请在 config/default.yaml 中指定输入路径")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"paths.input 路径不存在: {input_path}")

    is_dir = os.path.isdir(input_path)
    is_file = os.path.isfile(input_path)

    if mode == "auto":
        mode = "folder" if is_dir else "image"
    elif mode in ("image", "file", "single"):
        mode = "image"
    elif mode in ("folder", "dir", "directory", "batch"):
        mode = "folder"
    else:
        raise ValueError(f"未知运行模式 run.mode={mode!r}，请使用 auto / image / folder")

    if mode == "image":
        if is_dir:
            raise ValueError(f"run.mode=image 但 paths.input 是文件夹: {input_path}")
        if os.path.splitext(input_path)[1].lower() not in VALID_EXT:
            raise ValueError(f"不支持的图像格式: {input_path}")
        return mode, [input_path]

    # 文件夹批量模式
    if is_file:
        raise ValueError(f"run.mode=folder 但 paths.input 是单张文件: {input_path}")
    images = _list_images(input_path)
    if not images:
        raise FileNotFoundError(f"文件夹中未找到支持的图像: {input_path}")
    return mode, images


def main():
    cfg = load_config()
    paths = cfg["paths"]
    output_dir = paths.get("output") or os.path.join(cfg["project"]["root"], "out", "run")
    final_out_dir = os.path.join(output_dir, "final_out")
    os.makedirs(final_out_dir, exist_ok=True)

    mode, image_files = _resolve_jobs(cfg)

    print("=" * 50)
    print("VISTA 图像矢量化主程序")
    print(f"  运行模式 : {mode}")
    print(f"  输入路径 : {paths.get('input')}")
    print(f"  输出路径 : {output_dir}")
    print(f"  计算设备 : {cfg['_resolved_device']}")
    print(f"  待处理数 : {len(image_files)}")
    print("=" * 50)

    start_all = time.time()
    success_count = 0

    for i, img_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] 正在处理: {img_path}")
        try:
            summary = process_single_image(
                img_path,
                base_out_dir=output_dir,
                final_out_dir=final_out_dir,
                cfg=cfg,
            )
            vec = summary.get("vectorize", {})
            print(f"  --> 处理成功，耗时: {summary.get('total_time_sec', 0):.2f}s | 形状数: {vec.get('shapes')} | 路径点数: {vec.get('path_point_nums')} | MSE损失: {vec.get('mse_loss')}")
            success_count += 1
        except Exception as e:
            print(f"  --> [失败] 处理 {img_path} 出错: {e}")
            if mode == "image":
                raise

    print("\n" + "=" * 50)
    print(f"全部任务完成！成功处理 {success_count}/{len(image_files)} 张图像，总耗时: {time.time() - start_all:.2f}s")
    print(f"最终 SVG 成果目录: {final_out_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
