#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rendering utilities: gsplat rasterization, video writing, YOLO detection.

ВАЖНО: для длинных видео используйте render_gsplat_to_video_streaming(),
который пишет кадры в ffmpeg по одному и не накапливает их в памяти.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional

import logging
import numpy as np
import torch
from gsplat.rendering import rasterization
import ffmpeg  # ffmpeg-python
import cv2
from .gsplat_scene import GaussiansNP, build_intrinsics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Single-frame render (debug)
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

    logger.info(
        "[GSPLAT] one-frame: N=%d, HxW=%dx%d",
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
    logger.info(
        "[GSPLAT] img stats: min=%.6f max=%.6f mean=%.6f",
        float(img.min()),
        float(img.max()),
        float(img.mean()),
    )

    img_bgr = (img[..., ::-1] * 255.0 + 0.5).astype(np.uint8)
    return img_bgr


# ---------------------------------------------------------------------
# LEGACY: in-memory render + write (можно оставить для коротких роликов)
# ---------------------------------------------------------------------

def render_frames_gsplat(
    gauss: GaussiNP,
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

    ВНИМАНИЕ: для длинных видео хранит все кадры в памяти -> OOM.
    Для продакшен-пайплайна используйте render_gsplat_to_video_streaming().
    """
    logger.info("[GSPLAT] Trying gsplat on device: %s", device)

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

        logger.info("[GSPLAT] frame %d/%d", i + 1, num_frames)

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


def write_video(frames_bgr: List[np.ndarray], out_path: Path, fps: int) -> None:
    """
    Write BGR frames to MP4 using ffmpeg via rawvideo pipe.

    ВНИМАНИЕ: ожидает список всех кадров -> большой расход памяти.
    Для больших роликов лучше использовать потоковую запись.
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

    logger.info(
        "[VIDEO] Writing video: %s (%d frames, %dx%d @ %d FPS)",
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
# STREAMING: рендер + запись (и опциональный YOLO) без накопления кадров
# ---------------------------------------------------------------------

# render_utils.py (фрагмент)

def render_gsplat_to_video_streaming(
    gauss: GaussiansNP,
    poses: List[Dict[str, Any]],
    width: int,
    height: int,
    fov_deg: float,
    device: torch.device,
    out_path: Path,
    fps: int,
    detect: bool = False,
    yolo_model_path: str = "yolov8n.pt",
    yolo_conf: float = 0.25,
    draw_boxes: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    """
    Render frames with gsplat and stream them directly into ffmpeg *в формате RGB*.

    - gsplat -> float32 RGB [0,1]
    - конвертим в uint8 RGB
    - при detect=True:
        - YOLO работает по RGB,
        - res.plot() возвращает RGB,
        - если draw_boxes=True — в видео идёт аннотированная RGB-картинка.
    - ffmpeg ожидает rawvideo pix_fmt=rgb24.

    Возвращает список детекций, если detect=True, иначе None.
    """
    logger.info("[STREAM] Device: %s", device)
    logger.info(
        "[STREAM] Writing video: %s (%dx%d @ %d FPS, frames=%d)",
        out_path,
        width,
        height,
        fps,
        len(poses),
    )

    # Gaussians на GPU
    means_t = torch.from_numpy(gauss.means).to(device)
    quats_t = torch.from_numpy(gauss.quats).to(device)
    scales_t = torch.from_numpy(gauss.scales).to(device)
    opac_t = torch.from_numpy(gauss.opacities).to(device)
    colors_t = torch.from_numpy(gauss.colors).to(device)

    # Интринсики
    K_np = build_intrinsics(fov_deg, width, height)
    Ks = torch.from_numpy(K_np).to(device)[None, :, :]  # [1,3,3]

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ffmpeg: ждёт RAW RGB
    process = (
        ffmpeg
        .input(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
            s=f"{width}x{height}",
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

    # YOLO (опционально)
    yolo_model = None
    detections: Optional[List[Dict[str, Any]]] = None
    if detect:
        from ultralytics import YOLO
        logger.info("[YOLO] Loading model: %s", yolo_model_path)
        yolo_model = YOLO(yolo_model_path)
        detections = []

    num_frames = len(poses)

    try:
        for i, pose in enumerate(poses):
            view_np = np.asarray(pose["view"], dtype=np.float32)
            view_t = torch.from_numpy(view_np).to(device)[None, :, :]  # [1,4,4]

            logger.info("[STREAM] [gsplat] frame %d/%d", i + 1, num_frames)

            colors_out, alphas, meta = rasterization(
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

            # gsplat даёт RGB float32 [0,1]
            frame_rgb = colors_out[0].clamp(0.0, 1.0).detach().cpu().numpy()
            if frame_rgb.shape != (height, width, 3):
                raise RuntimeError(
                    f"Unexpected frame shape from gsplat: {frame_rgb.shape}, "
                    f"expected ({height}, {width}, 3)"
                )

            frame_u8 = (frame_rgb * 255.0 + 0.5).astype(np.uint8)
            frame_rgb_u8 = np.ascontiguousarray(frame_u8)  # [H,W,3], RGB, C-contiguous

            frame_out = frame_rgb_u8

            # YOLO + оверлей
            if detect and yolo_model is not None and detections is not None:
                logger.info("[STREAM] [YOLO] frame %d/%d", i + 1, num_frames)

                # YOLO v8/11 ожидает RGB, см. оф. обсуждение
                # results = model.predict(img_rgb, ...)
                res = yolo_model.predict(
                    frame_rgb_u8,
                    conf=yolo_conf,
                    verbose=False,
                )[0]

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

                detections.append({"frame": i, "detections": frame_list})

                if draw_boxes:
                    # res.plot() по документации работает с RGB и возвращает RGB ndarray
                    annotated_rgb = res.plot()
                    # На всякий случай подрежем до нужного размера
                    if annotated_rgb.shape[0] != height or annotated_rgb.shape[1] != width:
                        annotated_rgb = annotated_rgb[:height, :width, :]
                    frame_out = np.ascontiguousarray(annotated_rgb)

            # Пишем кадр (RGB) в ffmpeg
            process.stdin.write(frame_out.tobytes())

    finally:
        logger.info("[STREAM] Closing ffmpeg stdin")
        process.stdin.close()
        ret = process.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg returned non-zero exit code: {ret}")

    return detections
