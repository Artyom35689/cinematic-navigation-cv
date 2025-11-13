#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple cinematic navigation pipeline:

  - load unpacked 3DGS PLY into GaussiansNP
  - generate simple orbit camera path around bbox
  - render frames with gsplat
  - write MP4 via ffmpeg
  - optionally run YOLO detection on rendered video


python3 -m src.main \
     --scene scenes/ConferenceHall.ply \
     --outdir output/ConferenceHall_demo \
     --seconds 4 --fps 24 --fov 70 --res 1280x720 \
     --max-splats 200000000

"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from .gsplat_scene import load_gaussians_from_ply, generate_camera_poses
from .render_utils import render_frames_gsplat, write_video, run_detection


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
        required=True,
        help="Path to unpacked 3DGS .ply scene.",
    )
    p.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output directory for video and JSON.",
    )
    p.add_argument(
        "--seconds",
        type=float,
        default=4.0,
        help="Duration of the fly-through in seconds.",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Frames per second.",
    )
    p.add_argument(
        "--fov",
        type=float,
        default=70.0,
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
        default=2_000_000,
        help="Max number of Gaussians to keep (random downsampling if exceeded).",
    )
    p.add_argument(
        "--detect",
        action="store_true",
        help="Run YOLO detection on rendered frames.",
    )
    p.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model path/name for detection.",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.25,
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
    logging.info(
        "[INFO] Frames to render: %d at %d FPS, %dx%d, FOV=%.1f",
        num_frames,
        args.fps,
        width,
        height,
        args.fov,
    )

    # Load Gaussians
    gauss = load_gaussians_from_ply(str(scene_path), max_points=args.max_splats)

    # Camera poses (simple orbit)
    poses = generate_camera_poses(gauss.means, num_frames)

    # Device
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available in this container, but gsplat requires CUDA. "
            "Check that you run docker with --gpus and NVIDIA runtime."
        )
    device = torch.device("cuda")

    # Render frames
    frames_bgr = render_frames_gsplat(
        gauss,
        poses,
        width=width,
        height=height,
        fov_deg=args.fov,
        device=device,
    )

    # Video
    out_mp4 = outdir / "panorama_tour.mp4"
    write_video(frames_bgr, out_mp4, fps=args.fps)
    logging.info("[INFO] Video written: %s", out_mp4)

    # Camera path JSON
    cam_json = outdir / "camera_path.json"
    meta = {
        "scene": str(scene_path),
        "resolution": [width, height],
        "fps": args.fps,
        "seconds": args.seconds,
        "fov_deg": args.fov,
        "frames": [
            {
                "index": i,
                "eye": poses[i]["eye"],
                "center": poses[i]["center"],
                "view": np.asarray(poses[i]["view"], dtype=float).tolist(),
            }
            for i in range(len(poses))
        ],
        "render_backend": "gsplat_rgb",
    }
    with cam_json.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logging.info("[INFO] Camera path JSON written: %s", cam_json)

    # Optional YOLO
    if args.detect:
        dets = run_detection(frames_bgr, args.model, args.conf)
        det_json = outdir / "detections_yolo.json"
        with det_json.open("w", encoding="utf-8") as f:
            json.dump({"detections": dets}, f, indent=2)
        logging.info("[INFO] YOLO detections JSON written: %s", det_json)


if __name__ == "__main__":
    main()
