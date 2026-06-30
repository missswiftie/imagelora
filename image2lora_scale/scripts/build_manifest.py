#!/usr/bin/env python3
"""
从 dataset/ 目录扫描 styleized/ 并生成 OmniStyle manifest JSON。

输出格式与 sampled_100styles_150pairs.json 一致:
  metadata + samples[].pairs[]

用法:
  run_python scripts/build_manifest.py \\
      --data_root ../dataset \\
      --output ../dataset/omnistyle_manifest.json
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime


STYLIZED_DIR = "styleized"
STYLE_DIR = "style"
CONTENT_DIR = "content"


def parse_stylized_name(filename: str):
    """解析 styleized/{content}&&{style}.png 文件名。"""
    base = filename
    for ext in (".png", ".jpg", ".jpeg"):
        if base.lower().endswith(ext):
            base = base[: -len(ext)]
            break
    if "&&" not in base:
        return None
    content_name, style_name = base.split("&&", 1)
    return content_name, style_name


def extract_caption(content_name: str) -> str:
    """从 content 文件名提取 language_instruction (Category_Description)。"""
    if "_" in content_name:
        return content_name.split("_", 1)[1]
    return content_name


def extract_category(content_name: str) -> str:
    if "_" in content_name:
        return content_name.split("_", 1)[0]
    return "unknown"


def find_file(root: str, subdir: str, name: str) -> str:
    """在 subdir 下查找文件，自动尝试常见扩展名。"""
    for ext in ("", ".png", ".jpg", ".jpeg", ".jpg.jpg", ".jpeg.jpg", ".png.jpg"):
        candidate = os.path.join(root, subdir, name + ext if ext else name)
        if os.path.isfile(candidate):
            return f"{subdir}/{os.path.basename(candidate)}"
    direct = os.path.join(root, subdir, name)
    if os.path.isfile(direct):
        return f"{subdir}/{name}"
    return f"{subdir}/{name}"


def build_manifest(data_root: str, output: str, min_pairs_per_style: int = 0):
    stylized_root = os.path.join(data_root, STYLIZED_DIR)
    if not os.path.isdir(stylized_root):
        raise FileNotFoundError(f"styleized directory not found: {stylized_root}")

    style_pairs = defaultdict(list)
    missing_content = 0
    missing_style = 0
    scanned = 0

    for fname in os.listdir(stylized_root):
        if fname.startswith("."):
            continue
        parsed = parse_stylized_name(fname)
        if parsed is None:
            continue
        content_name, style_name = parsed
        scanned += 1

        content_rel = find_file(data_root, CONTENT_DIR, content_name)
        style_rel = find_file(data_root, STYLE_DIR, style_name)
        stylized_rel = f"{STYLIZED_DIR}/{fname}"

        if not os.path.isfile(os.path.join(data_root, content_rel)):
            missing_content += 1
        if not os.path.isfile(os.path.join(data_root, style_rel)):
            missing_style += 1

        style_pairs[style_name].append({
            "content_image": content_rel,
            "style_image": style_rel,
            "stylized_image": stylized_rel,
            "category": extract_category(content_name),
            "language_instruction": extract_caption(content_name),
        })

    samples = []
    for style_name, pairs in sorted(style_pairs.items()):
        if len(pairs) < min_pairs_per_style:
            continue
        style_rel = pairs[0]["style_image"]
        categories = {p["category"] for p in pairs}
        samples.append({
            "style_name": style_name,
            "style_image": style_rel,
            "num_pairs": len(pairs),
            "num_unique_categories": len(categories),
            "pairs": pairs,
        })

    total_pairs = sum(s["num_pairs"] for s in samples)
    manifest = {
        "metadata": {
            "dataset": "OmniStyle-150K",
            "description": "Full OmniStyle-150K manifest built from dataset/styleized/",
            "created_at": datetime.now().isoformat(),
            "path_mapping": {
                "content/": "content/",
                "style/": "style/",
                "OmniStyle-150K/": f"{STYLIZED_DIR}/",
                "stylized/": f"{STYLIZED_DIR}/",
            },
            "data_root": os.path.abspath(data_root),
            "total_pairs": total_pairs,
            "num_styles": len(samples),
            "scanned_stylized_files": scanned,
            "missing_content_refs": missing_content,
            "missing_style_refs": missing_style,
        },
        "samples": samples,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Wrote {total_pairs} pairs / {len(samples)} styles -> {output}")
    print(f"  scanned={scanned}, missing_content={missing_content}, missing_style={missing_style}")


def main():
    parser = argparse.ArgumentParser(description="Build OmniStyle manifest JSON from dataset/")
    parser.add_argument("--data_root", type=str, default="../dataset")
    parser.add_argument("--output", type=str, default="../dataset/omnistyle_manifest.json")
    parser.add_argument("--min_pairs_per_style", type=int, default=0)
    args = parser.parse_args()

    scale_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_root = os.path.dirname(scale_root)

    data_root = args.data_root
    if not os.path.isabs(data_root):
        data_root = os.path.normpath(os.path.join(scale_root, data_root))

    output = args.output
    if not os.path.isabs(output):
        output = os.path.normpath(os.path.join(scale_root, output))

    build_manifest(data_root, output, args.min_pairs_per_style)


if __name__ == "__main__":
    main()
