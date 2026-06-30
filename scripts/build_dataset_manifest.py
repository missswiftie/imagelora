#!/usr/bin/env python3
"""
从 dataset/ 扫描 styleized/、style/、content/，生成与
sampled_100styles_150pairs.json 相同格式的索引文件。

默认输出: dataset/omnistyle_150k.json

用法:
  source env.sh
  run_python scripts/build_dataset_manifest.py
  run_python scripts/build_dataset_manifest.py --output dataset/omnistyle_150k.json
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime


STYLIZED_DIR = "styleized"
STYLE_DIR = "style"
CONTENT_DIR = "content"
DEFAULT_OUTPUT = "dataset/omnistyle_150k.json"


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
    if "_" in content_name:
        caption = content_name.split("_", 1)[1]
    else:
        caption = content_name
    for ext in (".png", ".jpg", ".jpeg"):
        if caption.lower().endswith(ext):
            caption = caption[: -len(ext)]
    return caption


def extract_category(content_name: str) -> str:
    if "_" in content_name:
        return content_name.split("_", 1)[0]
    return "unknown"


def resolve_file(data_root: str, subdir: str, name: str) -> tuple[str, bool]:
    """返回 (相对路径, 是否存在)。"""
    for ext in ("", ".png", ".jpg", ".jpeg", ".jpg.jpg", ".jpeg.jpg", ".png.jpg"):
        fname = name + ext if ext else name
        rel = f"{subdir}/{fname}"
        if os.path.isfile(os.path.join(data_root, rel)):
            return rel, True
    rel = f"{subdir}/{name}"
    return rel, os.path.isfile(os.path.join(data_root, rel))


def build_manifest(
    data_root: str,
    output: str,
    only_complete: bool = True,
    min_pairs_per_style: int = 0,
):
    stylized_root = os.path.join(data_root, STYLIZED_DIR)
    if not os.path.isdir(stylized_root):
        raise FileNotFoundError(f"styleized directory not found: {stylized_root}")

    style_pairs: dict[str, list] = defaultdict(list)
    missing_content = 0
    missing_style = 0
    missing_stylized = 0
    scanned = 0
    skipped_incomplete = 0

    for fname in sorted(os.listdir(stylized_root)):
        if fname.startswith("."):
            continue
        parsed = parse_stylized_name(fname)
        if parsed is None:
            continue
        content_name, style_name = parsed
        scanned += 1

        content_rel, content_ok = resolve_file(data_root, CONTENT_DIR, content_name)
        style_rel, style_ok = resolve_file(data_root, STYLE_DIR, style_name)
        stylized_rel = f"{STYLIZED_DIR}/{fname}"
        stylized_ok = os.path.isfile(os.path.join(data_root, stylized_rel))

        if not content_ok:
            missing_content += 1
        if not style_ok:
            missing_style += 1
        if not stylized_ok:
            missing_stylized += 1

        if only_complete and not (content_ok and style_ok and stylized_ok):
            skipped_incomplete += 1
            continue

        style_pairs[style_name].append({
            "content_image": content_rel,
            "style_image": style_rel,
            "stylized_image": stylized_rel,
            "category": extract_category(content_name),
            "language_instruction": extract_caption(content_name),
        })

    samples = []
    pair_counts = []
    for style_name in sorted(style_pairs.keys()):
        pairs = style_pairs[style_name]
        if len(pairs) < min_pairs_per_style:
            continue
        pair_counts.append(len(pairs))
        categories = {p["category"] for p in pairs}
        samples.append({
            "style_name": style_name,
            "style_image": pairs[0]["style_image"],
            "num_pairs": len(pairs),
            "num_unique_categories": len(categories),
            "pairs": pairs,
        })

    total_pairs = sum(s["num_pairs"] for s in samples)
    max_pairs = max(pair_counts) if pair_counts else 0
    styles_with_max = sum(1 for c in pair_counts if c == max_pairs) if max_pairs else 0

    num_content = len([
        f for f in os.listdir(os.path.join(data_root, CONTENT_DIR))
        if not f.startswith(".")
    ]) if os.path.isdir(os.path.join(data_root, CONTENT_DIR)) else 0
    num_style = len([
        f for f in os.listdir(os.path.join(data_root, STYLE_DIR))
        if not f.startswith(".")
    ]) if os.path.isdir(os.path.join(data_root, STYLE_DIR)) else 0
    num_stylized = len([
        f for f in os.listdir(stylized_root) if not f.startswith(".")
    ])

    manifest = {
        "metadata": {
            "dataset": "OmniStyle-150K",
            "description": "OmniStyle-150K 全量索引，由 dataset/styleized/ 扫描生成",
            "created_at": datetime.now().isoformat(),
            "note": "语言指令 (language_instruction) 提取自 content 图片文件名，格式为 Category_Description",
            "package_dir": "dataset",
            "path_mapping": {
                "content/": "content/",
                "style/": "style/",
                "OmniStyle-150K/": f"{STYLIZED_DIR}/",
                "stylized/": f"{STYLIZED_DIR}/",
            },
            "copied_files": {
                "content": num_content,
                "style": num_style,
                "styleized": num_stylized,
            },
            "num_styles": len(samples),
            "total_pairs": total_pairs,
            "max_pairs_per_style": max_pairs,
            "styles_with_max_pairs": styles_with_max,
            "scanned_stylized_files": scanned,
            "skipped_incomplete_pairs": skipped_incomplete,
            "missing_content_refs": missing_content,
            "missing_style_refs": missing_style,
            "missing_stylized_refs": missing_stylized,
            "missing_files": skipped_incomplete if only_complete else (
                missing_content + missing_style + missing_stylized
            ),
            "only_complete_triplets": only_complete,
        },
        "samples": samples,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Wrote {output}")
    print(f"  styles={len(samples)}, pairs={total_pairs}")
    print(f"  scanned={scanned}, skipped_incomplete={skipped_incomplete}")
    print(f"  missing: content={missing_content}, style={missing_style}, stylized={missing_stylized}")


def main():
    parser = argparse.ArgumentParser(description="Build OmniStyle dataset index JSON")
    parser.add_argument("--data_root", type=str, default="dataset")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--include_incomplete", action="store_true",
                        help="包含 content/style 缺失的不完整三元组")
    parser.add_argument("--min_pairs_per_style", type=int, default=0)
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = args.data_root if os.path.isabs(args.data_root) else os.path.join(repo_root, args.data_root)
    output = args.output if os.path.isabs(args.output) else os.path.join(repo_root, args.output)

    build_manifest(
        data_root=data_root,
        output=output,
        only_complete=not args.include_incomplete,
        min_pairs_per_style=args.min_pairs_per_style,
    )


if __name__ == "__main__":
    main()
