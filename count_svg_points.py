#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVG 复杂度与路径统计分析工具 (SVG Path & Control Point Analyzer)

支持：
1. 分析单个 SVG 文件
2. 批量扫描文件夹（支持递归子目录与通配符筛选）
3. 统计每个 SVG 的：路径数 (Path Count)、总控制点数 (Total Control Points)、平均控制点数 (Avg Points/Path)、直线段与贝塞尔曲线段数
4. 支持以整齐美观的终端表格呈现，并可一键导出为 CSV 或 JSON 统计报表。
"""

import os
import sys
import glob
import re
import json
import csv
import argparse
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional, Tuple


def parse_svg_path_d(d_str: str) -> Tuple[int, int, int]:
    """
    精确解析 SVG path 中的 d 属性。
    返回: (控制点总数, 贝塞尔曲线段数, 直线段数)
    """
    if not d_str:
        return 0, 0, 0

    tokens = re.findall(r'([a-df-zA-DF-Z]|[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', d_str)
    if not tokens:
        return 0, 0, 0

    coords = []
    n_bezier = 0
    n_lines = 0

    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token.isalpha():
            cmd = token
            idx += 1
            if cmd in ('C', 'c'):
                n_bezier += 1
            elif cmd in ('L', 'l', 'H', 'h', 'V', 'v'):
                n_lines += 1
        else:
            coords.append(float(token))
            idx += 1

    total_points = len(coords) // 2
    return total_points, n_bezier, n_lines


def analyze_single_svg(svg_path: str) -> Optional[Dict[str, Any]]:
    """
    分析单个 SVG 文件中的路径与控制点数据。
    """
    if not os.path.isfile(svg_path):
        return None

    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
    except Exception as e:
        print(f"[Warning] 解析 SVG 失败: {svg_path} | 错误: {e}", file=sys.stderr)
        return None

    # 清理命名空间前缀
    for el in root.iter():
        if '}' in el.tag:
            el.tag = el.tag.split('}', 1)[1]

    paths = root.findall('.//path')
    polygons = root.findall('.//polygon')
    polylines = root.findall('.//polyline')
    circles = root.findall('.//circle')
    rects = root.findall('.//rect')

    total_shapes = len(paths) + len(polygons) + len(polylines) + len(circles) + len(rects)
    path_count = len(paths)
    
    total_points = 0
    total_bezier_segs = 0
    total_line_segs = 0
    path_point_list = []

    for p in paths:
        d = p.attrib.get('d', '')
        pts_count, n_bez, n_line = parse_svg_path_d(d)
        total_points += pts_count
        total_bezier_segs += n_bez
        total_line_segs += n_line
        path_point_list.append(pts_count)

    for pg in polygons:
        pts_str = pg.attrib.get('points', '')
        nums = [float(x) for x in re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', pts_str)]
        n_p = len(nums) // 2
        total_points += n_p
        total_line_segs += max(1, n_p)
        path_point_list.append(n_p)

    for pl in polylines:
        pts_str = pl.attrib.get('points', '')
        nums = [float(x) for x in re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', pts_str)]
        n_p = len(nums) // 2
        total_points += n_p
        total_line_segs += max(1, n_p - 1)
        path_point_list.append(n_p)

    for _ in rects:
        total_points += 4
        total_line_segs += 4
        path_point_list.append(4)
    for _ in circles:
        total_points += 12
        total_bezier_segs += 4
        path_point_list.append(12)

    avg_points = (total_points / max(total_shapes, 1)) if total_shapes > 0 else 0.0
    min_points = min(path_point_list) if path_point_list else 0
    max_points = max(path_point_list) if path_point_list else 0

    return {
        "file_name": os.path.basename(svg_path),
        "file_path": svg_path,
        "total_shapes": total_shapes,
        "path_count": path_count,
        "total_points": total_points,
        "avg_points_per_path": round(avg_points, 2),
        "min_points_in_path": min_points,
        "max_points_in_path": max_points,
        "bezier_segments": total_bezier_segs,
        "line_segments": total_line_segs,
        "file_size_kb": round(os.path.getsize(svg_path) / 1024.0, 2),
    }


def analyze_svg_batch(
    target_path: str,
    pattern: str = "*.svg",
    recursive: bool = True
) -> List[Dict[str, Any]]:
    """
    批量扫描目录或单个文件。
    """
    svg_files = []
    if os.path.isfile(target_path):
        svg_files = [target_path]
    elif os.path.isdir(target_path):
        if recursive:
            svg_files = sorted(glob.glob(os.path.join(target_path, "**", pattern), recursive=True))
        else:
            svg_files = sorted(glob.glob(os.path.join(target_path, pattern)))
    else:
        svg_files = sorted(glob.glob(target_path, recursive=recursive))

    results = []
    for fpath in svg_files:
        if not fpath.lower().endswith(".svg"):
            continue
        info = analyze_single_svg(fpath)
        if info is not None:
            results.append(info)

    return results


def print_summary_table(results: List[Dict[str, Any]]):
    """
    格式化打印终端对齐表格。
    """
    if not results:
        print("\n未找到有效的 SVG 文件。")
        return

    print("\n" + "=" * 110)
    print(f"  {'SVG 文件名':<42} | {'路径数':<8} | {'总控制点':<10} | {'平均点数/条':<12} | {'曲线/直线段':<14} | {'文件大小':<10}")
    print("-" * 110)

    total_all_paths = 0
    total_all_points = 0
    total_all_bezier = 0
    total_all_lines = 0

    for r in results:
        fname = r["file_name"]
        if len(fname) > 40:
            fname = fname[:37] + "..."
        paths_str = str(r["total_shapes"])
        pts_str = str(r["total_points"])
        avg_str = f"{r['avg_points_per_path']:.2f}"
        seg_str = f"{r['bezier_segments']}曲 / {r['line_segments']}直"
        size_str = f"{r['file_size_kb']} KB"

        total_all_paths += r["total_shapes"]
        total_all_points += r["total_points"]
        total_all_bezier += r["bezier_segments"]
        total_all_lines += r["line_segments"]

        print(f"  {fname:<42} | {paths_str:<8} | {pts_str:<10} | {avg_str:<12} | {seg_str:<14} | {size_str:<10}")

    print("=" * 110)
    avg_total_pts = (total_all_points / max(total_all_paths, 1)) if total_all_paths > 0 else 0.0
    print(f"  【全量统计汇总】")
    print(f"    - 处理 SVG 文件总数 : {len(results)} 个")
    print(f"    - 矢量路径总数     : {total_all_paths} 条")
    print(f"    - 控制点总数量     : {total_all_points} 个")
    print(f"    - 全局平均控制点数 : {avg_total_pts:.2f} 个/路径")
    print(f"    - 全局段数构成     : {total_all_bezier} 贝塞尔曲线段 | {total_all_lines} 直线段")
    print("=" * 110 + "\n")


def export_csv(results: List[Dict[str, Any]], out_csv_path: str):
    """
    导出统计数据到 CSV 文件。
    """
    if not results:
        return
    keys = list(results[0].keys())
    with open(out_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"--> CSV 报表已成功导出至: {out_csv_path}")


def export_json(results: List[Dict[str, Any]], out_json_path: str):
    """
    导出统计数据到 JSON 文件。
    """
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"--> JSON 报表已成功导出至: {out_json_path}")


def main():
    parser = argparse.ArgumentParser(description="SVG 路径数与控制点统计分析工具 (单个/批量)")
    parser.add_argument("input_path", type=str, help="SVG 文件路径或 SVG 文件夹路径")
    parser.add_argument("--pattern", "-p", type=str, default="*.svg", help="文件匹配规则（默认: *.svg）")
    parser.add_argument("--no-recursive", "-nr", action="store_true", help="不递归搜索子文件夹")
    parser.add_argument("--csv", type=str, default=None, help="导出统计结果到指定 CSV 文件")
    parser.add_argument("--json", type=str, default=None, help="导出统计结果到指定 JSON 文件")

    args = parser.parse_args()

    results = analyze_svg_batch(
        target_path=args.input_path,
        pattern=args.pattern,
        recursive=not args.no_recursive,
    )

    print_summary_table(results)

    if args.csv:
        export_csv(results, args.csv)
    if args.json:
        export_json(results, args.json)


if __name__ == "__main__":
    main()
