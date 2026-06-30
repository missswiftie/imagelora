#!/usr/bin/env python3
"""批量推理：用多张差异大的 style ref 验证风格是否分化。"""

import argparse
import json
import os
import subprocess
import sys

SCALE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(SCALE_ROOT)


DEFAULT_CASES = [
    {
        "name": "style_0000",
        "ref": "dataset/style/s0000____0912_01_query_2_img_000079_1683294877098_05408690224086452.jpeg.jpg",
        "prompt": "a cat sitting on a windowsill, highly detailed",
    },
    {
        "name": "style_0042",
        "ref": "dataset/style/s0042____0911_01_query_0_img_000034_1682330269283_04434056690766527.jpg.jpg",
        "prompt": "a mountain landscape at sunset, highly detailed",
    },
    {
        "name": "style_0053",
        "ref": "dataset/style/s0053____1113_01_query_0_img_000118_1682347962693_03292243121968712.jpg.jpg",
        "prompt": "a portrait of a young woman, studio lighting",
    },
    {
        "name": "style_0112",
        "ref": "dataset/style/s0112____1025_01_query_1_img_000156_1683664012862_022263192689277955.jpeg.jpg",
        "prompt": "a futuristic city skyline at night",
    },
    {
        "name": "style_0117",
        "ref": "dataset/style/s0117____1017_01_query_2_img_000037_1684095175403_0951688619316563.jpg.jpg",
        "prompt": "a bowl of fruit on a wooden table",
    },
]


def parse_args():
    p = argparse.ArgumentParser(description="Validate style diversity with batch inference")
    p.add_argument("--checkpoint_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--cases_json", type=str, default=None, help="Optional JSON list of {name, ref, prompt}")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resolution", type=int, default=1024)
    return p.parse_args()


def main():
    args = parse_args()
    cases = DEFAULT_CASES
    if args.cases_json:
        with open(args.cases_json, encoding="utf-8") as f:
            cases = json.load(f)

    out_dir = args.output_dir or os.path.join(
        args.checkpoint_dir, "validation",
    )
    os.makedirs(out_dir, exist_ok=True)

    infer_script = os.path.join(SCALE_ROOT, "scripts", "infer.py")
    python_wrapper = os.path.join(REPO_ROOT, "bin", "python")

    for case in cases:
        name = case["name"]
        ref = case["ref"]
        prompt = case["prompt"]
        output = os.path.join(out_dir, f"{name}.png")
        cmd = [
            python_wrapper, infer_script,
            "--checkpoint_dir", args.checkpoint_dir,
            "--ref_image", os.path.join(REPO_ROOT, ref) if not os.path.isabs(ref) else ref,
            "--prompt", prompt,
            "--output", output,
            "--seed", str(args.seed),
            "--resolution", str(args.resolution),
        ]
        print(f"[validate] {name} -> {output}")
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)

    print(f"Validation images saved to {out_dir}")
    print("Compare outputs: styles should look visually distinct if training is working.")


if __name__ == "__main__":
    main()
