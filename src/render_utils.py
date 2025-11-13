#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rendering utilities: gsplat rasterization, video writing, YOLO detection.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
from gsplat.rendering import rasterization
import ffmpeg  # ffmpeg-python
import logging

from .gsplat_scene import GaussiansNP, build_intrinsics


# ---------------------------------------------------------------------
# gsplat rendering
# ---------------------------------------------------------------------

def render_one_frame(
    gauss: GaussiansNP,
    view: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    device: torch.device,
) -> np.ndarray:
    """
    Render a single frame given explicit view (4x4) and intrinsics K (3x3).

    Returns BGR uint8 frame [H, W, 3].
    """
    means_t = torch.from_numpy(gauss.means).to(device)
    quats_t = torch.from_numpy(gauss.quats).to(device)
    scales_t = torch.from_numpy(gauss.scales).to(device)
    opac_t = torch.from_numpy(gauss.opacities).to(device)
    colors_t = torch.from_numpy(gauss.colors).to(device)

    view_t = torch.from_numpy(view.astype(np.float32)).to(device)[None, :, :]  # [1,4,4]
    K_t = torch.from_numpy(K.astype(np.float32)).to(device)[None, :, :]       # [1,3,3]

    logging.info(
        "[INFO] [gsplat] one-frame: N=%d, HxW=%dx%d",
        gauss.means.shape[0],
        height,
        width,
    )

    images, _, _ = rasterization(
        means_t,
        quats_t,
        scales_t,
        opac_t,
        colors_t,
        view_t,
        K_t,
        width,
        height,
        near_plane=0.01,
        far_plane=1e4,
        sh_degree=None,
        packed=True,
        render_mode="RGB",
        rasterize_mode="classic",
    )

    img = images[0].clamp(0.0, 1.0).detach().cpu().numpy()  # [H,W,3]
    logging.info(
        "[DEBUG] img stats: min=%.6f max=%.6f mean=%.6f",
        float(img.min()),
        float(img.max()),
        float(img.mean()),
    )

    img_bgr = (img[..., ::-1] * 255.0 + 0.5).astype(np.uint8)
    return img_bgr


def render_frames_gsplat(
    gauss: GaussiansNP,
    poses: List[Dict[str, Any]],
    width: int,
    height: int,
    fov_deg: float,
    device: torch.device,
) -> List[np.ndarray]:
    """
    Render multiple frames for a list of poses (each with 'view').

    Uses a single intrinsics matrix for all frames (same FOV & resolution).
    Returns list of BGR uint8 frames.
    """
    logging.info("[INFO] Trying gsplat on device: %s", device)

    means_t = torch.from_numpy(gauss.means).to(device)
    quats_t = torch.from_numpy(gauss.quats).to(device)
    scales_t = torch.from_numpy(gauss.scales).to(device)
    opac_t = torch.from_numpy(gauss.opacities).to(device)
    colors_t = torch.from_numpy(gauss.colors).to(device)

    K_np = build_intrinsics(fov_deg, width, height)
    Ks = torch.from_numpy(K_np).to(device)[None, :, :]  # [1,3,3]

    frames_bgr: List[np.ndarray] = []
    num_frames = len(poses)

    for i, pose in enumerate(poses):
        view_np = np.asarray(pose["view"], dtype=np.float32)
        view_t = torch.from_numpy(view_np).to(device)[None, :, :]  # [1,4,4]

        logging.info("[INFO] [gsplat] frame %d/%d", i + 1, num_frames)

        colors, alphas, meta = rasterization(
            means_t,
            quats_t,
            scales_t,
            opac_t,
            colors_t,
            view_t,
            Ks,
            width,
            height,
            near_plane=0.01,
            far_plane=1e4,
            sh_degree=None,
            packed=True,
            render_mode="RGB",
            rasterize_mode="classic",
        )

        frame_rgb = colors[0].clamp(0.0, 1.0).detach().cpu().numpy()
        if frame_rgb.shape != (height, width, 3):
            raise RuntimeError(
                f"Unexpected frame shape from gsplat: {frame_rgb.shape}, "
                f"expected ({height}, {width}, 3)"
            )

        frame_u8 = (frame_rgb * 255.0 + 0.5).astype(np.uint8)
        frame_bgr = frame_u8[..., ::-1]
        frames_bgr.append(frame_bgr)

    return frames_bgr


# ---------------------------------------------------------------------
# Video writer
# ---------------------------------------------------------------------

def write_video(frames_bgr: List[np.ndarray], out_path: Path, fps: int) -> None:
    """
    Write BGR frames to MP4 using ffmpeg via rawvideo pipe.
    """
    if not frames_bgr:
        raise RuntimeError("No frames to write")

    h, w, c = frames_bgr[0].shape
    if c != 3:
        raise RuntimeError(f"First frame has unexpected channels: {c}")

    for idx, f in enumerate(frames_bgr):
        if f.shape != (h, w, 3):
            raise RuntimeError(
                f"Frame {idx} has shape {f.shape}, expected {(h, w, 3)}"
            )
        if f.dtype != np.uint8:
            raise RuntimeError(
                f"Frame {idx} has dtype {f.dtype}, expected uint8"
            )

    logging.info(
        "[INFO] Writing video: %s (%d frames, %dx%d @ %d FPS)",
        out_path,
        len(frames_bgr),
        w,
        h,
        fps,
    )

    stack = np.stack(frames_bgr, axis=0)  # [T,H,W,3]
    frames_bytes = stack.tobytes()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    process = (
        ffmpeg
        .input(
            "pipe:",
            format="rawvideo",
            pix_fmt="bgr24",
            s=f"{w}x{h}",
            r=fps,
        )
        .output(
            str(out_path),
            vcodec="libx264",
            pix_fmt="yuv420p",
        )
        .overwrite_output()
        .run_async(pipe_stdin=True, quiet=True)
    )

    process.stdin.write(frames_bytes)
    process.stdin.close()
    ret = process.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg returned non-zero exit code: {ret}")


# ---------------------------------------------------------------------
# YOLO detection on rendered frames
# ---------------------------------------------------------------------

def run_detection(
    frames_bgr: List[np.ndarray],
    model_path: str,
    conf: float,
) -> List[Dict[str, Any]]:
    """
    Run Ultralytics YOLO on BGR frames and collect simple bounding boxes.
    """
    from ultralytics import YOLO

    logging.info("[INFO] Loading YOLO model: %s", model_path)
    model = YOLO(model_path)

    all_dets: List[Dict[str, Any]] = []
    for i, frame_bgr in enumerate(frames_bgr):
        logging.info("[INFO] [YOLO] frame %d/%d", i + 1, len(frames_bgr))
        frame_rgb = frame_bgr[..., ::-1]
        res = model.predict(frame_rgb, conf=conf, verbose=False)[0]

        frame_list = []
        if res.boxes is not None:
            for b in res.boxes:
                cls_id = int(b.cls.item())
                score = float(b.conf.item())
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                frame_list.append(
                    {
                        "cls": cls_id,
                        "conf": score,
                        "xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    }
                )

        all_dets.append(
            {
                "frame": i,
                "detections": frame_list,
            }
        )

    return all_dets
