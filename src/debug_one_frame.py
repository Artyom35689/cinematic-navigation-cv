#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal debug: load unpacked 3DGS PLY, render ONE frame with gsplat,
save it as PNG and dump basic metadata.

Run inside container from /app:

  python3 -m src.debug_one_frame \
    --scene scenes/ConferenceHall.ply \
    --outdir output/debug_conference \
    --width 1280 --height 720 --fov 70
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .gsplat_scene import load_gaussians_from_ply, build_camera
from .render_utils import render_one_frame


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Debug single-frame gsplat rendering.")
    ap.add_argument("--scene", required=True, help="Path to unpacked 3DGS .ply")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fov", type=float, default=70.0)
    ap.add_argument("--max_splats", type=int, default=2_000_000)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Scene:", args.scene)
    print("[INFO] Outdir:", outdir)

    gauss = load_gaussians_from_ply(args.scene, max_points=args.max_splats)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for gsplat; torch.cuda.is_available() == False")

    view, K = build_camera(gauss.means, args.fov, args.width, args.height)
    device = torch.device("cuda")

    img_bgr = render_one_frame(
        gauss,
        view,
        K,
        width=args.width,
        height=args.height,
        device=device,
    )

    out_png = outdir / "debug_frame.png"
    cv2.imwrite(str(out_png), img_bgr)
    print("[INFO] Saved PNG:", out_png)

    meta = {
        "scene": args.scene,
        "width": args.width,
        "height": args.height,
        "fov_deg": args.fov,
        "bbox_min": [float(x) for x in gauss.means.min(axis=0)],
        "bbox_max": [float(x) for x in gauss.means.max(axis=0)],
    }
    meta_path = outdir / "debug_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("[INFO] Saved meta:", meta_path)


if __name__ == "__main__":
    main()
