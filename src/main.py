#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple cinematic navigation pipeline:

  - load unpacked 3DGS PLY into GaussiansNP
  - generate simple straight camera path through given waypoints (XZ plane, Y up)
  - render frames with gsplat (streaming into ffmpeg)
  - write MP4 via ffmpeg
  - optionally run YOLO detection on rendered video (on the fly)


Пример запуска:

python3 -m src.main \
     --scene scenes/ConferenceHall.ply \
     --outdir output/ConferenceHall_demo \
     --seconds 35 --fps 30 --fov 70 --res 1280x720 \
     --max-splats 20000000
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from .analysis.research_scene import save_camera_path_on_density_xz
from .gsplat_scene import load_gaussians_from_ply, generate_camera_poses_straight_path, generate_camera_poses_spline
from .render_utils import render_gsplat_to_video_streaming


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Simple cinematic navigation renderer with gsplat + YOLO."
    )
    p.add_argument(
        "--scene",
        type=str,
        default="scenes/ConferenceHall.ply",
        help="Path to unpacked 3DGS .ply scene.",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default="output/demo_gsplat",
        help="Output directory for video and JSON.",
    )
    p.add_argument(
        "--seconds",
        type=float,
        default=30.0,
        help="Duration of the fly-through in seconds.",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second.",
    )
    p.add_argument(
        "--fov",
        type=float,
        default=80.0,
        help="Horizontal field of view in degrees.",
    )
    p.add_argument(
        "--res",
        type=str,
        default="1280x720",
        help="Resolution as WxH, e.g. 1280x720.",
    )
    p.add_argument(
        "--max-splats",
        type=int,
        default=20_000_000,
        help="Max number of Gaussians to keep (random downsampling if exceeded).",
    )
    p.add_argument(
        "--detect",
        action="store_true",
        default=False,
        help="Run YOLO detection on rendered frames.",
    )
    p.add_argument(
        "--model",
        type=str,
        default="models/yolo12n.pt",
        help="YOLO model path & name for detection.",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="YOLO confidence threshold.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    args = parse_args()

    scene_path = Path(args.scene)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        w_str, h_str = args.res.lower().split("x")
        width, height = int(w_str), int(h_str)
    except Exception as e:
        raise SystemExit(f"Invalid --res '{args.res}', expected WxH, got error: {e}")

    num_frames = max(1, int(round(args.seconds * args.fps)))
    logger.info(
        "[MAIN] Frames to render: %d at %d FPS, %dx%d, FOV=%.1f",
        num_frames,
        args.fps,
        width,
        height,
        args.fov,
    )

    # Load Gaussians
    gauss = load_gaussians_from_ply(str(scene_path), max_points=args.max_splats)

    # Camera poses: прямой путь по XZ через заданные точки
    waypoints_xz = [
        [28.0, 30.0],
        [20.0, 22.0],
        [10.0, 24.0],
        [0.0,24.0],
        [0.0,-5.0],
        [2.0,-5.0],
        [2.0,0.0],
        [5.1,0.0],
        [5.1,25.0],
    ]

    poses = generate_camera_poses_spline(
        gauss.means,
        num_frames=num_frames,
        waypoints_xz=waypoints_xz,
        height_fraction=0.0,      # можно поднять, например 0.1
        lookahead_fraction=0.05,  # как далеко вперёд по пути смотрим
        samples_per_segment=64,   # при необходимости можно увеличить
    )
    cam_path_plot = outdir / "camera_path_on_density_xz.png"
    save_camera_path_on_density_xz(
        gauss.means,   # или scene.gaussians.means, если работаешь с классом
        poses,
        cam_path_plot,
        grid_res=256,
        arrow_stride=10,
    )

    # Device
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available in this container, but gsplat requires CUDA. "
            "Check that you run docker with --gpus and NVIDIA runtime."
        )
    device = torch.device("cuda")

    # Output video path
    out_mp4 = outdir / "panorama_tour.mp4"

    # Render + stream to video (и опциональный YOLO)
    dets = render_gsplat_to_video_streaming(
        gauss=gauss,
        poses=poses,
        width=width,
        height=height,
        fov_deg=args.fov,
        device=device,
        out_path=out_mp4,
        fps=args.fps,
        detect=args.detect,
        yolo_model_path=args.model,
        yolo_conf=args.conf,
        draw_boxes=args.detect,  # если детектим — сразу и рисуем
    )

    logger.info("[MAIN] Video written: %s", out_mp4)

    # Camera path JSON
    cam_json = outdir / "camera_path.json"
    meta_frames: List[Dict[str, Any]] = []
    for i, pose in enumerate(poses):
        meta_frames.append(
            {
                "index": i,
                "eye": pose["eye"],
                "center": pose["center"],
                "view": np.asarray(pose["view"], dtype=float).tolist(),
            }
        )

    meta = {
        "scene": str(scene_path),
        "resolution": [width, height],
        "fps": args.fps,
        "seconds": args.seconds,
        "fov_deg": args.fov,
        "frames": meta_frames,
        "render_backend": "gsplat_rgb",
        "path_type": "straight_polyline_xz",
        "waypoints_xz": waypoints_xz,
    }
    with cam_json.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info("[MAIN] Camera path JSON written: %s", cam_json)

    # Optional YOLO detections (уже посчитаны по кадрам на лету)
    if args.detect and dets is not None:
        det_json = outdir / "detections_yolo.json"
        with det_json.open("w", encoding="utf-8") as f:
            json.dump({"detections": dets}, f, indent=2)
        logger.info("[MAIN] YOLO detections JSON written: %s", det_json)


if __name__ == "__main__":
    main()
