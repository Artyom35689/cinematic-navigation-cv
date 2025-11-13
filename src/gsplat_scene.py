#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scene utilities for 3D Gaussian Splatting (unpacked .ply):

Expected vertex layout:

  element vertex N
    property float x
    property float y
    property float z
    property float f_dc_0
    property float f_dc_1
    property float f_dc_2
    property float opacity
    property float scale_0
    property float scale_1
    property float scale_2
    property float rot_0
    property float rot_1
    property float rot_2
    property float rot_3

This is the "classic" 3DGS format (e.g. после конвертации через SplatTransform).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from plyfile import PlyData

# SH DC constant (same as in original 3DGS code)
SH_C0 = 0.28209479177387814


@dataclass
class GaussiansNP:
    """Gaussian cloud stored as NumPy arrays (host memory)."""
    means: np.ndarray      # [N, 3]
    quats: np.ndarray      # [N, 4]
    scales: np.ndarray     # [N, 3]
    opacities: np.ndarray  # [N]
    colors: np.ndarray     # [N, 3] RGB in [0,1]


# ---------------------------------------------------------------------
# PLY loading
# ---------------------------------------------------------------------

def load_gaussians_from_ply(path: str, max_points: Optional[int] = 2_000_000) -> GaussiansNP:
    """
    Load Gaussians from a full 3DGS PLY (unpacked, см. описание сверху).

    Heuristics:
      - scales: если много отрицательных значений -> считаем log-stddev и делаем exp().
      - opacity: если вне [0,1] -> считаем логитами и делаем sigmoid().
      - f_dc_*: DC-сферические гармоники → RGB через 0.5 + SH_C0 * f_dc, потом clamp.

    Если max_points не None и N > max_points, выполняется рандомный даунсэмплинг.
    """
    print(f"[DEBUG] load_gaussians_from_ply: {path}")
    ply = PlyData.read(path)

    elem_names = [e.name for e in ply.elements]
    print(f"[DEBUG] elements: {elem_names}")

    try:
        v = ply["vertex"]
    except KeyError:
        raise RuntimeError(f"PLY has no 'vertex' element; available: {elem_names}")

    names = v.data.dtype.names
    print(f"[DEBUG] vertex fields: {names}")

    required = [
        "x", "y", "z",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
        "opacity",
        "f_dc_0", "f_dc_1", "f_dc_2",
    ]
    missing = [f for f in required if f not in names]
    if missing:
        raise RuntimeError(f"PLY missing required fields {missing}. Available: {names}")

    # --- positions ---
    x = np.asarray(v["x"], np.float32)
    y = np.asarray(v["y"], np.float32)
    z = np.asarray(v["z"], np.float32)
    means = np.stack([x, y, z], axis=1)  # (N,3)

    # --- raw scales & opacities ---
    s0 = np.asarray(v["scale_0"], np.float32)
    s1 = np.asarray(v["scale_1"], np.float32)
    s2 = np.asarray(v["scale_2"], np.float32)
    scales_raw = np.stack([s0, s1, s2], axis=1)

    op_raw = np.asarray(v["opacity"], np.float32)

    scale_min, scale_max = float(scales_raw.min()), float(scales_raw.max())
    frac_neg_scale = float((scales_raw < 0.0).mean())
    op_min, op_max = float(op_raw.min()), float(op_raw.max())

    print(f"[DEBUG] scale_raw min/max={scale_min:.4f}/{scale_max:.4f}, neg_frac={frac_neg_scale:.3f}")
    print(f"[DEBUG] opacity_raw min/max={op_min:.4f}/{op_max:.4f}")

    # scales: heuristic – treat as log-stddev if many negatives
    if frac_neg_scale > 0.1:
        scales = np.exp(scales_raw)
        print("[DEBUG] treating scales as log-stddev, applying exp()")
    else:
        scales = scales_raw
        print("[DEBUG] treating scales as already linear")

    # opacities: heuristic – logits → sigmoid if values outside [0,1]
    if op_min < 0.0 or op_max > 1.0:
        opacities = 1.0 / (1.0 + np.exp(-op_raw))
        print("[DEBUG] treating opacities as logits, applying sigmoid()")
    else:
        opacities = op_raw
        print("[DEBUG] treating opacities as already in [0,1]")

    opacities = opacities.astype(np.float32)

    # --- quaternions ---
    q0 = np.asarray(v["rot_0"], np.float32)
    q1 = np.asarray(v["rot_1"], np.float32)
    q2 = np.asarray(v["rot_2"], np.float32)
    q3 = np.asarray(v["rot_3"], np.float32)
    quats = np.stack([q0, q1, q2, q3], axis=1)  # (N,4)

    # normalize quats
    norm = np.linalg.norm(quats, axis=1, keepdims=True)
    norm[norm == 0.0] = 1.0
    quats = quats / norm

    # --- color from DC SH ---
    fdc0 = np.asarray(v["f_dc_0"], np.float32)
    fdc1 = np.asarray(v["f_dc_1"], np.float32)
    fdc2 = np.asarray(v["f_dc_2"], np.float32)
    f_dc = np.stack([fdc0, fdc1, fdc2], axis=1)  # (N,3)

    colors = 0.5 + SH_C0 * f_dc
    colors = np.clip(colors, 0.0, 1.0).astype(np.float32)

    N = means.shape[0]
    print(f"[DEBUG] loaded {N} splats")

    if max_points is not None and N > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(N, size=max_points, replace=False)
        means = means[idx]
        quats = quats[idx]
        scales = scales[idx]
        opacities = opacities[idx]
        colors = colors[idx]
        print(f"[DEBUG] downsampled {N} -> {max_points}")

    return GaussiansNP(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
    )


# ---------------------------------------------------------------------
# Camera utilities
# ---------------------------------------------------------------------

def build_intrinsics(fov_deg: float, width: int, height: int) -> np.ndarray:
    """
    Build simple pinhole intrinsics for given *horizontal* FOV in degrees.

    fx = fy = 0.5 * width / tan(FOV/2)
    """
    fov_rad = np.deg2rad(float(fov_deg))
    fx = 0.5 * width / np.tan(0.5 * fov_rad)
    fy = fx
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5

    K = np.array(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return K


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n < 1e-8 else v / n


def look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    """
    World -> camera view matrix.

    Камера ориентирована так, что ось +Z камеры смотрит от eye к center.
    """
    forward = _normalize(center - eye)              # +Z_cam
    right = _normalize(np.cross(forward, up))       # +X_cam
    new_up = np.cross(right, forward)               # +Y_cam

    R = np.stack([right, new_up, forward], axis=0).astype(np.float32)  # (3,3)
    t = -R @ eye.astype(np.float32)                                     # (3,)

    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = R
    view[:3, 3] = t
    return view


def build_camera(means: np.ndarray,
                 fov_deg: float,
                 width: int,
                 height: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Один "средний" кадр: камера ставится перед сценой по -Z и смотрит на центр bbox.

    Возвращает:
      view: (4,4) world->camera
      K:    (3,3) intrinsics
    """
    bbox_min = means.min(axis=0)
    bbox_max = means.max(axis=0)
    center = 0.5 * (bbox_min + bbox_max)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    if diag < 1e-6:
        diag = 1.0

    cam_pos = center + np.array([0.0, 0.0, -0.7 * diag], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    view = look_at(cam_pos, center, up)
    K = build_intrinsics(fov_deg, width, height)

    print("[DEBUG] camera center:", center)
    print("[DEBUG] camera pos:", cam_pos)
    print("[DEBUG] bbox_min:", bbox_min, "bbox_max:", bbox_max, "diag:", diag)
    print("[DEBUG] K:", K)

    # depth sanity-check
    M = min(4096, means.shape[0])
    idx = np.random.choice(means.shape[0], size=M, replace=False)
    pts = means[idx]
    pts_h = np.concatenate([pts, np.ones((M, 1), dtype=np.float32)], axis=1)
    cam_pts = (view @ pts_h.T).T
    z_vals = cam_pts[:, 2]
    print(
        "[DEBUG] depth stats: "
        f"min={z_vals.min():.3f}, max={z_vals.max():.3f}, mean={z_vals.mean():.3f}"
    )

    return view, K


def generate_camera_poses(
    means: np.ndarray,
    num_frames: int,
    orbit_angle_deg: float = 60.0,
    height_fraction: float = 0.1,
    distance_scale: float = 1.6,
) -> List[Dict[str, Any]]:
    """
    Простейшая орбита камеры вокруг сцены:

      - bbox → центр и диагональ
      - радиус = distance_scale * diag
      - орбита в XZ-плоскости, угол от -orbit/2 до +orbit/2
      - лёгкий подъём по Y = height_fraction * diag

    Возвращает список словарей {"view": 4x4, "eye": [...], "center": [...]}.
    """
    mins = means.min(axis=0)
    maxs = means.max(axis=0)
    center = 0.5 * (mins + maxs)
    diag = float(np.linalg.norm(maxs - mins)) + 1e-6

    radius = distance_scale * diag
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    poses: List[Dict[str, Any]] = []
    for i in range(num_frames):
        t = 0.0 if num_frames == 1 else i / (num_frames - 1)
        angle = np.deg2rad((t - 0.5) * orbit_angle_deg)
        offset = np.array([np.sin(angle), 0.0, np.cos(angle)], dtype=np.float32)
        eye = center + offset * radius
        eye[1] += height_fraction * diag

        view = look_at(eye, center, up)
        poses.append(
            {
                "view": view,
                "eye": eye.tolist(),
                "center": center.tolist(),
            }
        )

    print(
        "[DEBUG] generate_camera_poses: N=%d, center=%s, radius=%.3f"
        % (num_frames, np.round(center, 3).tolist(), radius)
    )
    return poses
