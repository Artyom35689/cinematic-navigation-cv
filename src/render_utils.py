#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rendering utilities: gsplat rasterization, video writing, YOLO detection.

IMPORTANT:
    - For long videos, prefer `render_gsplat_to_video_streaming()`, which
      streams frames directly into ffmpeg without keeping them all in memory.
    - This module now supports an optional "revisit" pass: after the initial
      scan with YOLO, the camera runs the same rail again and stops to re-look
      at a few selected objects in 3D.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import logging
import numpy as np
import torch
from gsplat.rendering import rasterization
import ffmpeg  # ffmpeg-python
import cv2  # kept for compatibility; not strictly required here

from .gsplat_scene import GaussiansNP, build_intrinsics, look_at

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

    Returns:
        img_bgr: uint8 frame [H, W, 3] in BGR order (OpenCV-friendly).
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
# LEGACY: in-memory render + write (OK for short clips)
# ---------------------------------------------------------------------

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

    WARNING:
        For long videos this keeps all frames in memory -> possible OOM.
        For production use, prefer `render_gsplat_to_video_streaming()`.
    """
    logger.info("[GSPLAT] Device: %s", device)

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

    WARNING:
        Expects all frames in memory. For large clips prefer streaming API.
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
# Helpers for 3D object centers and revisit path
# ---------------------------------------------------------------------

def _backproject_box_center_to_world(
    bbox_xyxy: List[float],
    depth: np.ndarray,
    K_np: np.ndarray,
    view_np: np.ndarray,
    num_samples_per_axis: int = 3,
    margin_frac: float = 0.2,
    depth_quantile: float = 0.2,
) -> Optional[Tuple[np.ndarray, float]]:
    """
    Back-project an object's bounding box into 3D world coordinates using depth.

    Compared to a naive center-sample, this tries to be more robust and less
    biased towards floor / background:

        - we shrink the box to a central region (margin_frac) to avoid edges,
        - we collect *all* valid depth samples in that patch,
        - we keep only the central depth distribution
          (between quantiles [q, 1-q]),
        - then back-project these pixels into world and take mean + avg radius.

    Args:
        bbox_xyxy: [x1, y1, x2, y2] in pixel coordinates.
        depth: [H, W] expected depth in camera coordinates (z_cam).
        K_np: [3,3] intrinsics.
        view_np: [4,4] world->camera matrix (same as used in gsplat).
        num_samples_per_axis: kept for backward compatibility, not used anymore.
        margin_frac: fractional margin to cut off from each side of the box
                     (e.g. 0.2 -> use central 60% region).
        depth_quantile: how much of depth extremes to discard (e.g. 0.2 ->
                        keep [0.2, 0.8] depth quantiles).

    Returns:
        (center_world, radius_world) if enough valid depth samples are found,
        otherwise None.
    """
    H, W = depth.shape
    x1, y1, x2, y2 = bbox_xyxy

    # Clamp bbox to image bounds
    x1_i = int(np.floor(max(0.0, min(W - 1.0, x1))))
    x2_i = int(np.ceil(min(W - 1.0, max(0.0, x2))))
    y1_i = int(np.floor(max(0.0, min(H - 1.0, y1))))
    y2_i = int(np.ceil(min(H - 1.0, max(0.0, y2))))

    if x2_i <= x1_i or y2_i <= y1_i:
        return None

    # Optionally shrink to a central patch to avoid big floor/ceiling areas
    box_w = x2_i - x1_i + 1
    box_h = y2_i - y1_i + 1
    if margin_frac > 0.0 and box_w > 4 and box_h > 4:
        dx = int(round(box_w * margin_frac))
        dy = int(round(box_h * margin_frac))
        if dx * 2 < box_w and dy * 2 < box_h:
            x1_i += dx
            x2_i -= dx
            y1_i += dy
            y2_i -= dy

    if x2_i <= x1_i or y2_i <= y1_i:
        return None

    depth_patch = depth[y1_i : y2_i + 1, x1_i : x2_i + 1]  # [h,w]
    h_p, w_p = depth_patch.shape
    if h_p == 0 or w_p == 0:
        return None

    # Valid depths
    valid_mask = np.isfinite(depth_patch) & (depth_patch > 0.0)
    if not np.any(valid_mask):
        return None

    z_vals = depth_patch[valid_mask]
    if z_vals.size < 4:
        return None

    # Keep central depth band to reject foreground/background outliers
    q = float(np.clip(depth_quantile, 0.0, 0.45))
    z_low, z_high = np.quantile(z_vals, [q, 1.0 - q])
    depth_band_mask = valid_mask & (depth_patch >= z_low) & (depth_patch <= z_high)
    if not np.any(depth_band_mask):
        # Fallback: use all valid depths
        depth_band_mask = valid_mask

    # Collect pixel coordinates and depths in image space
    ys_local, xs_local = np.nonzero(depth_band_mask)  # indices in patch
    z_keep = depth_patch[ys_local, xs_local]

    if z_keep.size == 0:
        return None

    # Convert patch indices to full image coordinates
    us = x1_i + xs_local
    vs = y1_i + ys_local

    fx, fy = K_np[0, 0], K_np[1, 1]
    cx, cy = K_np[0, 2], K_np[1, 2]

    # Camera space
    z_cam = z_keep.astype(np.float32)
    x_cam = (us.astype(np.float32) - cx) / fx * z_cam
    y_cam = (vs.astype(np.float32) - cy) / fy * z_cam

    ones = np.ones_like(z_cam, dtype=np.float32)
    p_cam = np.stack([x_cam, y_cam, z_cam, ones], axis=1)  # [K,4]

    # Camera -> world
    world_from_cam = np.linalg.inv(view_np).astype(np.float32)
    p_world = (world_from_cam @ p_cam.T).T  # [K,4]

    w_comp = p_world[:, 3]
    good = np.abs(w_comp) > 1e-6
    if not np.any(good):
        return None

    p_world3 = (p_world[good, :3] / w_comp[good, None]).astype(np.float32)  # [K',3]
    if p_world3.shape[0] == 0:
        return None

    center_world = p_world3.mean(axis=0)
    radius_world = float(
        np.linalg.norm(p_world3 - center_world[None, :], axis=1).mean()
    )

    return center_world, radius_world


def _select_revisit_objects(
    objects: List[Dict[str, Any]],
    top_k: int,
    min_world_dist: float,
) -> List[Dict[str, Any]]:
    """
    Select up to top_k objects for revisit, using purely random sampling.

    Strategy:
        - filter only objects that have a valid 3D center ("center_world");
        - shuffle candidates randomly;
        - greedily pick objects, optionally enforcing a minimum 3D separation
          of `min_world_dist` between selected centers.

    Returns:
        List of selected object dicts (subset of input).
    """
    if top_k <= 0 or not objects:
        return []

    # Consider only objects with 3D center
    candidates = [o for o in objects if "center_world" in o]
    if not candidates:
        logger.info("[REVISIT] No candidates with 'center_world', skipping selection.")
        return []

    # Random shuffle (fixed seed for reproducibility, change if needed)
    rng = np.random.default_rng(0)
    rng.shuffle(candidates)

    selected: List[Dict[str, Any]] = []
    for obj in candidates:
        center = np.asarray(obj["center_world"], dtype=np.float32)

        if min_world_dist > 0.0 and selected:
            too_close = False
            for s in selected:
                center_s = np.asarray(s["center_world"], dtype=np.float32)
                if np.linalg.norm(center - center_s) < min_world_dist:
                    too_close = True
                    break
            if too_close:
                continue

        selected.append(obj)
        if len(selected) >= top_k:
            break

    logger.info(
        "[REVISIT] Randomly selected %d/%d objects (top_k=%d, min_world_dist=%.3f)",
        len(selected),
        len(candidates),
        top_k,
        min_world_dist,
    )
    return selected


def _draw_selected_objects_boxes(
    frame_rgb: np.ndarray,
    view_np: np.ndarray,
    K_np: np.ndarray,
    selected_objects: List[Dict[str, Any]],
    max_box_px: int = 80,
    min_box_px: int = 10,
) -> np.ndarray:
    """
    Draw bounding boxes for selected 3D objects on an RGB frame.

    - Each object is described by "center_world" and optional "radius_world".
    - We project center_world into the current camera using view_np & K_np.
    - If the projection is inside the image and z_cam > 0 (in front of camera),
      we draw a square box centered at the projected position.
    - Box size is estimated from radius_world and depth; clamped to
      [min_box_px, max_box_px].
    - Additionally, we draw a caption with class id and confidence
      (taken from 'class_id' and 'score' fields if present).

    Args:
        frame_rgb: [H,W,3] uint8 RGB frame (will be copied).
        view_np: [4,4] world->camera matrix.
        K_np: [3,3] intrinsics.
        selected_objects: list of objects with "center_world" (and optionally
                          "radius_world", "class_id", "score").
    Returns:
        New RGB frame with drawn boxes and captions (RGB).
    """
    if not selected_objects:
        return frame_rgb

    H, W, _ = frame_rgb.shape
    fx, fy = K_np[0, 0], K_np[1, 1]
    cx, cy = K_np[0, 2], K_np[1, 2]

    # Work in BGR for OpenCV drawing
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    for obj in selected_objects:
        center_world = np.asarray(obj["center_world"], dtype=np.float32)
        radius_world = float(obj.get("radius_world", 0.5))

        # Optional metadata
        class_id = obj.get("class_id", None)
        score = obj.get("score", None)

        # World -> camera
        p_world = np.concatenate(
            [center_world, np.array([1.0], dtype=np.float32)],
            axis=0,
        )
        p_cam = view_np @ p_world  # [4]
        if abs(p_cam[3]) < 1e-6:
            continue
        p_cam3 = p_cam[:3] / p_cam[3]

        z_cam = float(p_cam3[2])
        if z_cam <= 0.0:
            # Behind camera, skip
            continue

        # Camera -> image
        x_cam, y_cam = float(p_cam3[0]), float(p_cam3[1])
        u = fx * x_cam / z_cam + cx
        v = fy * y_cam / z_cam + cy

        if not (0.0 <= u < W and 0.0 <= v < H):
            # Outside image bounds
            continue

        # Estimate approximate box size in pixels from radius_world and depth
        # proj_radius ≈ radius_world * fx / z
        pix_r = radius_world * fx / max(z_cam, 1e-3)
        pix_r = float(np.clip(pix_r, min_box_px, max_box_px))

        x1 = int(round(u - pix_r))
        y1 = int(round(v - pix_r))
        x2 = int(round(u + pix_r))
        y2 = int(round(v + pix_r))

        # Clip to image bounds
        x1 = max(0, min(W - 1, x1))
        x2 = max(0, min(W - 1, x2))
        y1 = max(0, min(H - 1, y1))
        y2 = max(0, min(H - 1, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        # Draw rectangle
        color = (0, 255, 0)  # BGR (green)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thickness=2)

        # Prepare caption: "cls: <id> conf: <score>"
        caption_parts = []
        if class_id is not None:
            caption_parts.append(f"cls:{class_id}")
        if score is not None:
            caption_parts.append(f"conf:{score:.2f}")
        caption = " ".join(caption_parts)

        if caption:
            # Put caption above the box (clipped to image)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1

            (text_w, text_h), baseline = cv2.getTextSize(
                caption, font, font_scale, thickness
            )

            text_x = x1
            text_y = max(0, y1 - 5)

            # Background rectangle for readability
            bg_x1 = text_x
            bg_y1 = max(0, text_y - text_h - baseline)
            bg_x2 = min(W - 1, text_x + text_w)
            bg_y2 = min(H - 1, text_y + baseline)

            cv2.rectangle(
                frame_bgr,
                (bg_x1, bg_y1),
                (bg_x2, bg_y2),
                (0, 0, 0),
                thickness=-1,
            )
            cv2.putText(
                frame_bgr,
                caption,
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                lineType=cv2.LINE_AA,
            )

        # Mark exact projected center
        cv2.drawMarker(
            frame_bgr,
            (int(round(u)), int(round(v))),
            color,
            markerType=cv2.MARKER_CROSS,
            markerSize=10,
            thickness=2,
        )

    frame_rgb_out = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return frame_rgb_out


def _attach_best_frame_indices(
    objects: List[Dict[str, Any]],
    poses: List[Dict[str, Any]],
) -> None:
    """
    For each object, find the frame index where the camera eye is closest to
    the object's 3D center (in world coordinates).

    Modifies objects in-place, adding:
        - "best_frame_idx"
        - "best_dist"
    """
    if not objects or not poses:
        return

    eyes = np.asarray([p["eye"] for p in poses], dtype=np.float32)  # [T,3]

    for obj in objects:
        center = np.asarray(obj["center_world"], dtype=np.float32)
        dists = np.linalg.norm(eyes - center[None, :], axis=1)  # [T]
        best_idx = int(np.argmin(dists))
        best_dist = float(dists[best_idx])
        obj["best_frame_idx"] = best_idx
        obj["best_dist"] = best_dist

    logger.info("[REVISIT] Attached best_frame_idx for %d objects", len(objects))


def build_revisit_path_with_stops(
    base_poses: List[Dict[str, Any]],
    revisit_objects: List[Dict[str, Any]],
    fps: int,
    stop_seconds: float = 0.5,
    interp_seconds: float = 0.25,
) -> List[Dict[str, Any]]:
    """
    Build a second-pass camera path over the same rail with stops for objects.

    The camera:
        - runs the same sequence of positions (eyes) as in base_poses
          (from frame 0 to the last frame),
        - at frames where objects are assigned ("best_frame_idx"), we:
            * smoothly rotate the view towards the object center,
            * hold gaze on the object for `stop_seconds`,
            * then continue along the rail.

    NOTE:
        - This function *does not* affect the first pass. It builds only the
          "revisit" segment that is appended after the initial scan.
        - We assume world up axis: +Y.

    Returns:
        List of poses (each {"eye", "center", "view"}) for the revisit pass.
    """
    if not base_poses or not revisit_objects:
        return []

    stop_frames = max(1, int(stop_seconds * fps))
    interp_frames = max(1, int(interp_seconds * fps))

    # Map: frame_idx -> single object (pick highest score if multiple)
    by_frame: Dict[int, Dict[str, Any]] = {}
    for obj in revisit_objects:
        fi = int(obj["best_frame_idx"])
        prev = by_frame.get(fi)
        if prev is None or float(obj.get("score", 0.0)) > float(prev.get("score", 0.0)):
            by_frame[fi] = obj

    logger.info(
        "[REVISIT] Building revisit path: stop_frames=%d, interp_frames=%d, frames_with_objects=%d",
        stop_frames,
        interp_frames,
        len(by_frame),
    )

    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    poses_revisit: List[Dict[str, Any]] = []

    # Run along the same rail again from frame 0 to last
    for i, base_pose in enumerate(base_poses):
        eye = np.asarray(base_pose["eye"], dtype=np.float32)
        base_center = np.asarray(base_pose["center"], dtype=np.float32)

        # Always add the base pose first
        poses_revisit.append(
            {
                "eye": eye.tolist(),
                "center": base_center.tolist(),
                "view": np.asarray(base_pose["view"], dtype=np.float32),
            }
        )

        if i not in by_frame:
            continue

        obj = by_frame[i]
        obj_center = np.asarray(obj["center_world"], dtype=np.float32)

        # Rotate IN towards the object
        for k in range(interp_frames):
            alpha = float(k + 1) / float(interp_frames + 1)
            center_interp = (1.0 - alpha) * base_center + alpha * obj_center
            view_interp = look_at(eye, center_interp, up)
            poses_revisit.append(
                {
                    "eye": eye.tolist(),
                    "center": center_interp.tolist(),
                    "view": view_interp,
                }
            )

        # HOLD on the object
        view_obj = look_at(eye, obj_center, up)
        for _ in range(stop_frames):
            poses_revisit.append(
                {
                    "eye": eye.tolist(),
                    "center": obj_center.tolist(),
                    "view": view_obj,
                }
            )

        # We do NOT explicitly rotate back here; the next base pose will gradually
        # take over when the second pass proceeds along the rail.

    logger.info(
        "[REVISIT] Revisit path length: %d poses (base=%d)",
        len(poses_revisit),
        len(base_poses),
    )
    return poses_revisit


# ---------------------------------------------------------------------
# STREAMING: render + write (and optional YOLO + revisit) without buffering
# ---------------------------------------------------------------------

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
    # NEW: revisit configuration
    revisit_top_k: int = 0,
    revisit_min_world_dist: float = 0.5,
    revisit_stop_seconds: float = 0.5,
    revisit_interp_seconds: float = 0.25,
) -> Optional[List[Dict[str, Any]]]:
    """
    Render frames with gsplat and stream them directly into ffmpeg *as RGB*.

    Pipeline:
        - First pass:
            * gsplat -> float32 RGB (+ optional depth) in [0,1]
            * convert to uint8 RGB
            * if detect=True:
                - run YOLO on RGB
                - optionally draw boxes
                - compute 3D centers in world coords using depth (plan A)
                - store all per-frame detections
            * write frame to ffmpeg (RGB rawvideo)
        - Optional second pass (revisit):
            * only if detect=True AND revisit_top_k > 0 AND there are objects
            * select up to revisit_top_k objects in 3D
            * build a "revisit" path with stops on the same rail
            * render these poses (RGB only, no YOLO)
            * append to the same video.

    gsplat details:
        - render_mode:
            * if detect and revisit_top_k > 0 -> "RGB+ED" (RGB + expected depth)
            * else if detect -> "RGB" (we only need RGB for YOLO)
            * else -> "RGB"
        - ffmpeg expects rawvideo pix_fmt=rgb24.

    Returns:
        detections: list of frame-level dicts if detect=True, otherwise None.

        Each detection entry has the form:
            {
              "frame": int,
              "detections": [
                {
                  "cls": int,
                  "conf": float,
                  "xyxy": [x1,y1,x2,y2],
                  # NEW (if depth-based 3D center is available):
                  "center_world": [x,y,z],
                  "radius_world": float
                },
                ...
              ]
            }
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

    num_frames = len(poses)
    if num_frames == 0:
        raise RuntimeError("No poses provided for rendering")

    # Gaussians on GPU
    means_t = torch.from_numpy(gauss.means).to(device)
    quats_t = torch.from_numpy(gauss.quats).to(device)
    scales_t = torch.from_numpy(gauss.scales).to(device)
    opac_t = torch.from_numpy(gauss.opacities).to(device)
    colors_t = torch.from_numpy(gauss.colors).to(device)

    # Intrinsics
    K_np = build_intrinsics(fov_deg, width, height)
    Ks = torch.from_numpy(K_np).to(device)[None, :, :]  # [1,3,3]

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ffmpeg: consumes RAW RGB
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

    # YOLO (optional)
    yolo_model = None
    detections: Optional[List[Dict[str, Any]]] = None
    all_objects_3d: List[Dict[str, Any]] = []

    # Need depth only if we plan to do a 3D-based revisit
    need_depth = bool(detect and revisit_top_k > 0)

    if detect:
        from ultralytics import YOLO
        logger.info("[YOLO] Loading model: %s", yolo_model_path)
        yolo_model = YOLO(yolo_model_path)
        detections = []

    # Determine render_mode for the first pass
    if need_depth:
        render_mode = "RGB+ED"
    else:
        render_mode = "RGB"

    logger.info(
        "[STREAM] First pass render_mode=%s (need_depth=%s, detect=%s, revisit_top_k=%d)",
        render_mode,
        need_depth,
        detect,
        revisit_top_k,
    )

    try:
        # -------------------------------------------------------------
        # First pass: scan with YOLO (+depth if enabled) and write to video
        # -------------------------------------------------------------
        for i, pose in enumerate(poses):
            view_np = np.asarray(pose["view"], dtype=np.float32)
            view_t = torch.from_numpy(view_np).to(device)[None, :, :]  # [1,4,4]

            logger.info("[STREAM] [gsplat] pass1 frame %d/%d", i + 1, num_frames)

            renders, alphas, meta = rasterization(
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
                render_mode=render_mode,
                rasterize_mode="classic",
            )

            # gsplat returns [1, H, W, C]
            renders_np = renders[0].detach().cpu().numpy()  # [H,W,C]
            if renders_np.shape[0] != height or renders_np.shape[1] != width:
                raise RuntimeError(
                    f"Unexpected frame shape from gsplat: {renders_np.shape}, "
                    f"expected ({height}, {width}, C)"
                )

            if render_mode == "RGB+ED":
                rgb = np.clip(renders_np[..., :3], 0.0, 1.0)
                depth = renders_np[..., 3]  # [H,W], expected depth
            else:
                rgb = np.clip(renders_np[..., :3], 0.0, 1.0)
                depth = None

            frame_rgb_u8 = (rgb * 255.0 + 0.5).astype(np.uint8)
            frame_rgb_u8 = np.ascontiguousarray(frame_rgb_u8)  # [H,W,3]

            frame_out = frame_rgb_u8

            # --- YOLO + detection bookkeeping ---
            if detect and yolo_model is not None and detections is not None:
                logger.info("[STREAM] [YOLO] pass1 frame %d/%d", i + 1, num_frames)

                # YOLO expects RGB image
                res = yolo_model.predict(
                    frame_rgb_u8,
                    conf=yolo_conf,
                    verbose=False,
                )[0]

                frame_list: List[Dict[str, Any]] = []
                frame_eye = np.asarray(pose["eye"], dtype=np.float32)

                if res.boxes is not None:
                    for b in res.boxes:
                        cls_id = int(b.cls.item())
                        score = float(b.conf.item())
                        x1, y1, x2, y2 = b.xyxy[0].tolist()
                        det: Dict[str, Any] = {
                            "cls": cls_id,
                            "conf": score,
                            "xyxy": [float(x1), float(y1), float(x2), float(y2)],
                        }

                        # If we have depth, compute 3D center in world coords
                        if depth is not None:
                            maybe_center = _backproject_box_center_to_world(
                                bbox_xyxy=[x1, y1, x2, y2],
                                depth=depth,
                                K_np=K_np,
                                view_np=view_np,
                                num_samples_per_axis=3,
                            )
                            if maybe_center is not None:
                                center_world, radius_world = maybe_center
                                det["center_world"] = center_world.tolist()
                                det["radius_world"] = float(radius_world)

                                all_objects_3d.append(
                                    {
                                        "frame_idx": i,
                                        "class_id": cls_id,
                                        "score": score,
                                        "bbox": det["xyxy"],
                                        "center_world": center_world.tolist(),
                                        "radius_world": float(radius_world),
                                        "cam_eye_at_detection": frame_eye.tolist(),
                                    }
                                )

                        frame_list.append(det)

                detections.append(
                    {
                        "frame": i,
                        "detections": frame_list,
                    }
                )

                if draw_boxes:
                    # res.plot() returns an RGB numpy array with annotations
                    annotated_rgb = res.plot()
                    if (
                        annotated_rgb.shape[0] != height
                        or annotated_rgb.shape[1] != width
                    ):
                        annotated_rgb = annotated_rgb[:height, :width, :]
                    frame_out = np.ascontiguousarray(annotated_rgb)

            # --- write frame (RGB) to ffmpeg ---
            process.stdin.write(frame_out.tobytes())

        # -------------------------------------------------------------
        # Second pass (optional): revisit selected objects in 3D
        # -------------------------------------------------------------
        do_revisit = bool(
            detect and revisit_top_k > 0 and len(all_objects_3d) > 0
        )

        if do_revisit:
            logger.info(
                "[REVISIT] Enabled. collected_objects=%d, top_k=%d",
                len(all_objects_3d),
                revisit_top_k,
            )

            # Pick up to revisit_top_k objects in 3D
            selected_objects = _select_revisit_objects(
                all_objects_3d,
                top_k=revisit_top_k,
                min_world_dist=revisit_min_world_dist,
            )

            if selected_objects:
                _attach_best_frame_indices(selected_objects, poses)
                revisit_poses = build_revisit_path_with_stops(
                    base_poses=poses,
                    revisit_objects=selected_objects,
                    fps=fps,
                    stop_seconds=revisit_stop_seconds,
                    interp_seconds=revisit_interp_seconds,
                )

                if revisit_poses:
                    logger.info(
                        "[REVISIT] Starting second pass: %d poses (drawing boxes for selected objects)",
                        len(revisit_poses),
                    )
                    for j, pose in enumerate(revisit_poses):
                        view_np = np.asarray(pose["view"], dtype=np.float32)
                        view_t = torch.from_numpy(view_np).to(device)[None, :, :]

                        logger.info(
                            "[STREAM] [gsplat] pass2 frame %d/%d",
                            j + 1,
                            len(revisit_poses),
                        )

                        renders2, alphas2, meta2 = rasterization(
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

                        rgb2 = (
                            renders2[0]
                            .clamp(0.0, 1.0)
                            .detach()
                            .cpu()
                            .numpy()
                        )  # [H,W,3], float32 in [0,1]
                        frame2_u8 = (rgb2 * 255.0 + 0.5).astype(np.uint8)
                        frame2_rgb = np.ascontiguousarray(frame2_u8)

                        # Draw boxes ONLY for selected objects that are actually visible
                        frame2_rgb = _draw_selected_objects_boxes(
                            frame_rgb=frame2_rgb,
                            view_np=view_np,
                            K_np=K_np,
                            selected_objects=selected_objects,
                        )

                        process.stdin.write(frame2_rgb.tobytes())
                else:
                    logger.info("[REVISIT] No revisit poses generated, skipping pass2.")

            else:
                logger.info("[REVISIT] No objects selected after filtering, skipping pass2.")
        else:
            logger.info(
                "[REVISIT] Disabled or no 3D objects available "
                "(detect=%s, top_k=%d, all_objects_3d=%d).",
                detect,
                revisit_top_k,
                len(all_objects_3d),
            )

    finally:
        logger.info("[STREAM] Closing ffmpeg stdin")
        process.stdin.close()
        ret = process.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg returned non-zero exit code: {ret}")

    return detections
