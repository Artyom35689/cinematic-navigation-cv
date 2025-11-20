#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scene utilities for 3D Gaussian Splatting (unpacked .ply) + camera path planners.

Expected vertex layout (classic 3DGS PLY, e.g. after SplatTransform):

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

Conventions:
  - World up axis: Y.
  - Camera motion plane for path planners: XZ.
  - All logging via Python's logging module (no prints).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import logging

import numpy as np
from plyfile import PlyData

logger = logging.getLogger(__name__)

# SH DC constant (same as in original 3DGS code)
SH_C0 = 0.28209479177387814


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------


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
    Load Gaussians from a full 3DGS PLY.

    Heuristics:
      - scales: if many negative values -> treat as log-stddev and apply exp().
      - opacity: if outside [0,1] -> treat as logits and apply sigmoid().
      - f_dc_*: DC spherical harmonics -> RGB via 0.5 + SH_C0 * f_dc, then clamp.

    If max_points is not None and N > max_points, performs random downsampling.
    """
    logger.info("[PLY] Loading 3DGS PLY: %s", path)
    ply = PlyData.read(path)

    elem_names = [e.name for e in ply.elements]
    logger.debug("[PLY] Elements: %s", elem_names)

    try:
        v = ply["vertex"]
    except KeyError:
        raise RuntimeError(f"PLY has no 'vertex' element; available: {elem_names}")

    names = v.data.dtype.names
    logger.debug("[PLY] Vertex fields: %s", names)

    required = [
        "x",
        "y",
        "z",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
        "opacity",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
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

    logger.info(
        "[PLY] scale_raw min/max=%.4f/%.4f, neg_frac=%.3f",
        scale_min,
        scale_max,
        frac_neg_scale,
    )
    logger.info("[PLY] opacity_raw min/max=%.4f/%.4f", op_min, op_max)

    # scales: heuristic – treat as log-stddev if many negatives
    if frac_neg_scale > 0.1:
        scales = np.exp(scales_raw)
        logger.info("[PLY] Treating scales as log-stddev, applying exp().")
    else:
        scales = scales_raw
        logger.info("[PLY] Treating scales as already linear.")

    # opacities: heuristic – logits → sigmoid if values outside [0,1]
    if op_min < 0.0 or op_max > 1.0:
        opacities = 1.0 / (1.0 + np.exp(-op_raw))
        logger.info("[PLY] Treating opacities as logits, applying sigmoid().")
    else:
        opacities = op_raw
        logger.info("[PLY] Treating opacities as already in [0,1].")

    opacities = opacities.astype(np.float32)

    # --- quaternions ---
    q0 = np.asarray(v["rot_0"], np.float32)
    q1 = np.asarray(v["rot_1"], np.float32)
    q2 = np.asarray(v["rot_2"], np.float32)
    q3 = np.asarray(v["rot_3"], np.float32)
    quats = np.stack([q0, q1, q2, q3], axis=1)  # (N,4)

    # normalize quats for safety
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
    logger.info("[PLY] Loaded %d splats.", N)

    if max_points is not None and N > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(N, size=max_points, replace=False)
        means = means[idx]
        quats = quats[idx]
        scales = scales[idx]
        opacities = opacities[idx]
        colors = colors[idx]
        logger.info("[PLY] Downsampled %d -> %d splats.", N, max_points)

    return GaussiansNP(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
    )


# ---------------------------------------------------------------------
# Camera utilities (intrinsics, look-at)
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
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return K


def _normalize(v: np.ndarray) -> np.ndarray:
    """Safe vector normalization (returns zero vector if norm is tiny)."""
    n = np.linalg.norm(v)
    return v if n < 1e-8 else v / n


def look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    """
    World -> camera view matrix.

    Camera is oriented so that camera +Z axis looks from eye to center.
    """
    forward = _normalize(center - eye)        # +Z_cam
    right = _normalize(np.cross(forward, up)) # +X_cam
    new_up = np.cross(right, forward)         # +Y_cam

    R = np.stack([right, new_up, forward], axis=0).astype(np.float32)  # (3,3)
    t = -R @ eye.astype(np.float32)                                     # (3,)

    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = R
    view[:3, 3] = t
    return view


def build_camera(
    means: np.ndarray,
    fov_deg: float,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience helper: single "average" camera:

      - bbox → center and diagonal
      - camera placed in front of scene along -Z
      - looks at bbox center

    Returns:
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

    # Light-weight depth sanity-check for debugging
    M = min(4096, means.shape[0])
    idx = np.random.choice(means.shape[0], size=M, replace=False)
    pts = means[idx]
    pts_h = np.concatenate([pts, np.ones((M, 1), dtype=np.float32)], axis=1)
    cam_pts = (view @ pts_h.T).T
    z_vals = cam_pts[:, 2]

    logger.info(
        "[CAM] BBox center=%s, cam_pos=%s, bbox_min=%s, bbox_max=%s, diag=%.3f",
        np.round(center, 3),
        np.round(cam_pos, 3),
        np.round(bbox_min, 3),
        np.round(bbox_max, 3),
        diag,
    )
    logger.info(
        "[CAM] Depth stats: min=%.3f, max=%.3f, mean=%.3f",
        float(z_vals.min()),
        float(z_vals.max()),
        float(z_vals.mean()),
    )

    return view, K


# ---------------------------------------------------------------------
# Simple orbit camera (baseline)
# ---------------------------------------------------------------------


def generate_camera_poses(
    means: np.ndarray,
    num_frames: int,
    orbit_angle_deg: float = 60.0,
    height_fraction: float = 0.1,
    distance_scale: float = 1.6,
) -> List[Dict[str, Any]]:
    """
    Simple orbit camera around scene:

      - bbox → center and diagonal
      - radius = distance_scale * diag
      - orbit in XZ plane, angle from -orbit/2 to +orbit/2
      - camera Y = center_y + height_fraction * diag

    Returns list of dicts {"view": 4x4, "eye": [...], "center": [...]}.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be > 0")

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
                "eye": eye.astype(np.float32).tolist(),
                "center": center.astype(np.float32).tolist(),
            }
        )

    logger.info(
        "[PATH/ORBIT] Generated %d poses, center=%s, radius=%.3f",
        num_frames,
        np.round(center, 3),
        radius,
    )
    return poses


# ---------------------------------------------------------------------
# Path planners (straight line / spline) with segment behaviors
# ---------------------------------------------------------------------

# We specify the path (in XZ plane) together with per-segment camera behavior:
#
#   path_with_behaviors = [
#       ([x0, z0], beh0),
#       ([x1, z1], beh1),
#       ...
#       ([xM, zM], behM),  # behavior for the last waypoint is ignored
#   ]
#
# Segment i runs from waypoint i to waypoint i+1 and uses behavior beh_i.
#
# Behavior dicts (per segment):
#
#   None or {"mode": "default"}:
#       - follow path and look forward (base behavior).
#
#   {"mode": "look_at_point",
#    "target": [tx, ty, tz],
#    "strength": 1.0}
#       - camera moves along the path,
#       - look direction is smoothly pulled toward target in the middle of the segment:
#           - at segment start/end camera looks along path direction (smooth glue),
#           - in the middle effect is strongest (strength in [0,1]).
#
#   {"mode": "extra_yaw",
#    "angle_deg": 360.0}
#       - camera moves along the path,
#       - additional yaw (around world Y) is applied over the segment:
#           - extra yaw is 0 at start and end of segment,
#           - max yaw ~= angle_deg in the middle of segment.
#
#   {"mode": "height_arc",
#    "height_offset_fraction": 0.2}
#       - camera moves along the path,
#       - camera follows an "arc" in height on this segment:
#           - at start/end: same height as base path,
#           - in the middle: lifted by height_offset_fraction * scene_diag.
#
# All behaviors are implemented so that orientation/height are equal to base path
# at segment endpoints, which guarantees smooth stitching between segments.


def _smooth_bump(u: float) -> float:
    """
    Smooth bump in [0,1] with value 0 at endpoints and 1 in the middle:

        f(u) = 4 * u * (1 - u)

    Used as an envelope for per-segment effects so that they start and end at 0.
    """
    u = float(np.clip(u, 0.0, 1.0))
    return 4.0 * u * (1.0 - u)


def _apply_segment_behavior(
    behavior: Optional[Dict[str, Any]],
    eye: np.ndarray,
    center: np.ndarray,
    forward: np.ndarray,
    up: np.ndarray,
    diag: float,
    u: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply per-segment behavior to given base eye/center for local segment
    coordinate u in [0,1].

    All modes guarantee that:
      - for u=0 and u=1 the look direction is equal to the base path direction,
      - for u in (0,1) behavior may modify orientation and/or height.

    Args:
        behavior: behavior dict or None.
        eye: base camera position [3].
        center: base look-at point [3].
        forward: normalized base forward vector (center - eye).
        up: world up (0,1,0).
        diag: scene diagonal (used for height scaling).
        u: local segment parameter in [0,1].

    Returns:
        (eye_out, center_out) as np.float32 arrays.
    """
    if behavior is None:
        return eye, center

    mode = behavior.get("mode", "default")
    if mode == "default":
        return eye, center

    # Ensure we work with float32 copies
    eye_out = eye.astype(np.float32).copy()
    center_out = center.astype(np.float32).copy()
    base_dir = forward.astype(np.float32).copy()
    base_dir = _normalize(base_dir)

    # Distance from eye to center: we keep it constant when rotating view
    base_dist = float(np.linalg.norm(center - eye) + 1e-8)

    # Envelope in [0,1], zero at endpoints
    w = _smooth_bump(u)

    if mode == "look_at_point":
        target = np.asarray(behavior.get("target", center_out), dtype=np.float32)
        strength = float(behavior.get("strength", 1.0))
        strength = float(np.clip(strength, 0.0, 1.0))

        # Direction towards target
        dir_target = _normalize(target - eye_out)
        # Blend base_dir and target direction with envelope
        alpha = w * strength
        dir_mix = _normalize((1.0 - alpha) * base_dir + alpha * dir_target)
        center_out = eye_out + dir_mix * base_dist
        return eye_out, center_out

    if mode == "extra_yaw":
        # Additional yaw applied around world Y, zero at segment endpoints.
        angle_deg = behavior.get("angle_deg", None)
        turns = behavior.get("turns", None)

        if angle_deg is not None:
            angle_deg = float(angle_deg)
        elif turns is not None:
            angle_deg = float(turns) * 360.0
        else:
            angle_deg = 360.0  # default full spin-like behavior

        angle_rad_max = np.deg2rad(angle_deg)
        # extra yaw is symmetric and zero at endpoints
        extra_yaw = angle_rad_max * np.sin(np.pi * float(np.clip(u, 0.0, 1.0)))

        # Rotate base_dir in XZ plane around world Y
        dx, dy, dz = base_dir[0], base_dir[1], base_dir[2]
        cos_y = np.cos(extra_yaw)
        sin_y = np.sin(extra_yaw)
        new_dx = dx * cos_y + dz * sin_y
        new_dz = -dx * sin_y + dz * cos_y
        new_dir = _normalize(np.array([new_dx, dy, new_dz], dtype=np.float32))

        center_out = eye_out + new_dir * base_dist
        return eye_out, center_out

    if mode == "height_arc":
        # Move camera along a vertical arc on this segment.
        # Height offset is zero at endpoints and maximal in the middle.
        offset_frac = float(behavior.get("height_offset_fraction", 0.0))
        delta_y = offset_frac * diag * w

        eye_out[1] += delta_y
        center_out[1] += delta_y
        return eye_out, center_out

    logger.warning("[PATH/BEHAV] Unknown mode '%s', using default.", mode)
    return eye, center


# Type alias for clarity
PathWithBehaviors = Sequence[Tuple[Sequence[float], Optional[Dict[str, Any]]]]


def generate_camera_poses_straight_path(
    means: np.ndarray,
    num_frames: int,
    path_with_behaviors: PathWithBehaviors,
    height_fraction: float = 0.0,
    lookahead_fraction: float = 0.05,
) -> List[Dict[str, Any]]:
    """
    Generate camera poses that move along a polyline in the XZ plane
    with per-segment camera behavior.

    - World up axis: Y.
    - Camera moves along given waypoints in XZ: [(x0, z0), (x1, z1), ...].
    - Camera height (Y) is constant and derived from scene bbox
      (height_fraction * scene_diag above center) unless modified by behaviors.
    - Base behavior: camera looks forward along the path (look-ahead).
    - Per-segment behaviors modify orientation/height but always start and end
      aligned with base path direction for smooth stitching.

    Args:
        means: (N,3) Gaussian means, used only to estimate bbox/height.
        num_frames: total frames to generate.
        path_with_behaviors: sequence of ( [x, z], behavior_dict_or_None ).
                             Segment i goes from waypoint i to waypoint i+1
                             and uses behavior from element i.
        height_fraction: base camera Y offset as fraction of scene diagonal.
        lookahead_fraction: fraction of total path length used as look-ahead
                            distance when computing base "forward" direction.

    Returns:
        List of dicts {"view": 4x4, "eye": [3], "center": [3]}.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be > 0")
    if len(path_with_behaviors) < 2:
        raise ValueError("Need at least two waypoints in path_with_behaviors")

    # Extract waypoints and per-segment behaviors
    waypoints_xz = np.asarray([p for (p, _) in path_with_behaviors], dtype=np.float32)  # [M,2]
    seg_behaviors: List[Optional[Dict[str, Any]]] = [b for (_, b) in path_with_behaviors]
    M = waypoints_xz.shape[0]
    if M < 2:
        raise ValueError("Need at least two waypoints for a straight path")

    # Scene bbox / center for height
    bbox_min = means.min(axis=0)
    bbox_max = means.max(axis=0)
    center = 0.5 * (bbox_min + bbox_max)
    diag = float(np.linalg.norm(bbox_max - bbox_min)) + 1e-6

    center_y = float(center[1])
    base_cam_y = center_y + float(height_fraction) * diag

    # Build 3D waypoints in XZ plane with constant Y
    waypoints = np.stack(
        [
            waypoints_xz[:, 0],
            np.full_like(waypoints_xz[:, 0], base_cam_y),
            waypoints_xz[:, 1],
        ],
        axis=1,
    )  # [M,3]

    # Precompute segment lengths and cumulative distances
    seg_vecs = waypoints[1:] - waypoints[:-1]  # [M-1,3]
    seg_lens = np.linalg.norm(seg_vecs, axis=1)  # [M-1]
    if np.any(seg_lens <= 0):
        raise ValueError("Found zero-length segment in waypoints")

    cum_lens = np.concatenate([[0.0], np.cumsum(seg_lens)])  # [M]
    total_len = float(cum_lens[-1])

    logger.info(
        "[PATH/STRAIGHT] total_len=%.3f, num_segments=%d, num_frames=%d",
        total_len,
        len(seg_lens),
        num_frames,
    )

    def point_at_s(s: float) -> Tuple[np.ndarray, int, float]:
        """
        Return (point, seg_idx, u_local) along polyline at arc length s.

        s is clamped to [0, total_len].
        seg_idx in [0, M-2], u_local in [0,1] is local parameter in that segment.
        """
        s = np.clip(s, 0.0, total_len)
        j = np.searchsorted(cum_lens, s, side="right") - 1
        j = max(0, min(j, len(seg_lens) - 1))
        s0 = cum_lens[j]
        seg_len = seg_lens[j]
        if seg_len <= 0:
            return waypoints[j], j, 0.0
        u_local = float((s - s0) / seg_len)
        u_local = float(np.clip(u_local, 0.0, 1.0))
        point = waypoints[j] * (1.0 - u_local) + waypoints[j + 1] * u_local
        return point, j, u_local

    # Generate poses
    poses: List[Dict[str, Any]] = []
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    if num_frames == 1:
        # Degenerate: place camera at first point, look to second
        eye = waypoints[0]
        look_target = waypoints[1]
        forward = _normalize(look_target - eye)
        eye, center = _apply_segment_behavior(
            seg_behaviors[0] if seg_behaviors else None,
            eye,
            look_target,
            forward,
            up,
            diag,
            u=0.0,
        )
        view = look_at(eye, center, up)
        poses.append(
            {
                "view": view,
                "eye": eye.astype(np.float32).tolist(),
                "center": center.astype(np.float32).tolist(),
            }
        )
        logger.info("[PATH/STRAIGHT] Generated single-frame pose.")
        return poses

    lookahead_dist = float(lookahead_fraction) * total_len

    for i in range(num_frames):
        # global position along path by arc length
        t = i / float(num_frames - 1)
        s = t * total_len
        s_ahead = min(s + lookahead_dist, total_len)

        eye, seg_idx, u_local = point_at_s(s)
        target_ahead, _, _ = point_at_s(s_ahead)

        forward = _normalize(target_ahead - eye)
        base_center = eye + forward * np.linalg.norm(target_ahead - eye)

        behavior = seg_behaviors[min(seg_idx, len(seg_behaviors) - 1)]
        eye_mod, center_mod = _apply_segment_behavior(
            behavior,
            eye,
            base_center,
            forward,
            up,
            diag,
            u_local,
        )

        view = look_at(eye_mod, center_mod, up)

        poses.append(
            {
                "view": view,
                "eye": eye_mod.astype(np.float32).tolist(),
                "center": center_mod.astype(np.float32).tolist(),
            }
        )

    logger.info(
        "[PATH/STRAIGHT] Generated %d poses along polyline through %d waypoints",
        len(poses),
        waypoints_xz.shape[0],
    )
    return poses


def generate_camera_poses_spline(
    means: np.ndarray,
    num_frames: int,
    path_with_behaviors: PathWithBehaviors,
    height_fraction: float = 0.0,
    lookahead_fraction: float = 0.05,
    samples_per_segment: int = 64,
) -> List[Dict[str, Any]]:
    """
    Generate camera poses along a Catmull-Rom spline in the XZ plane
    with per-segment camera behavior.

    - World up axis: Y.
    - Camera moves in XZ through given waypoints: [(x0, z0), (x1, z1), ...].
    - Camera height (Y) is constant and derived from scene bbox (unless modified).
    - Path is smooth (Catmull-Rom spline), not piecewise linear.
    - Speed is approximately constant via arc-length parameterization.
    - Base behavior: camera looks forward along the spline (look-ahead).
    - Per-segment behaviors modify orientation/height but always start and end
      aligned with base path direction for smooth stitching.

    Args:
        means: (N,3) Gaussian means, used only to estimate bbox/height.
        num_frames: total frames to generate.
        path_with_behaviors: sequence of ( [x, z], behavior_dict_or_None ).
                             Segment i goes from waypoint i to waypoint i+1
                             and uses behavior from element i.
        height_fraction: base camera Y offset as fraction of scene diagonal.
        lookahead_fraction: fraction of total path length used as look-ahead
                            distance when computing base "forward" direction.
        samples_per_segment: number of samples per spline segment when building
                             arc-length table (more -> smoother speed, slower).

    Returns:
        List of dicts {"view": 4x4, "eye": [3], "center": [3]}.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be > 0")
    if len(path_with_behaviors) < 2:
        raise ValueError("Need at least two waypoints in path_with_behaviors")

    waypoints_xz = np.asarray([p for (p, _) in path_with_behaviors], dtype=np.float32)  # [M,2]
    seg_behaviors: List[Optional[Dict[str, Any]]] = [b for (_, b) in path_with_behaviors]
    M = waypoints_xz.shape[0]
    if M < 2:
        raise ValueError("Need at least two waypoints for spline path")

    # --- Scene bbox / camera height ---
    bbox_min = means.min(axis=0)
    bbox_max = means.max(axis=0)
    center = 0.5 * (bbox_min + bbox_max)
    diag = float(np.linalg.norm(bbox_max - bbox_min)) + 1e-6

    center_y = float(center[1])
    base_cam_y = center_y + float(height_fraction) * diag

    logger.info(
        "[PATH/SPLINE] bbox_min=%s bbox_max=%s diag=%.3f base_cam_y=%.3f",
        np.round(bbox_min, 3),
        np.round(bbox_max, 3),
        diag,
        base_cam_y,
    )

    # --- Catmull-Rom helpers ---

    pts = waypoints_xz  # [M,2]
    n_segments = M - 1

    def get_ctrl(idx: int) -> np.ndarray:
        """Clamp index to [0, M-1] and return the control point [2]."""
        idx = max(0, min(idx, M - 1))
        return pts[idx]

    def catmull_rom_segment(seg_idx: int, t: float) -> np.ndarray:
        """
        Catmull-Rom position on segment seg_idx in [0,1].

        p0--p1--p2--p3, the curve segment goes from p1 to p2.
        """
        p0 = get_ctrl(seg_idx - 1)
        p1 = get_ctrl(seg_idx)
        p2 = get_ctrl(seg_idx + 1)
        p3 = get_ctrl(seg_idx + 2)

        t2 = t * t
        t3 = t2 * t
        # Classic uniform Catmull-Rom spline formula:
        return 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )

    # --- Arc-length sampling (approximate constant speed) ---

    if samples_per_segment < 2:
        samples_per_segment = 2

    total_segments = n_segments
    total_samples = total_segments * samples_per_segment + 1
    sample_s = np.linspace(0.0, float(total_segments), total_samples, dtype=np.float32)

    sample_pos = np.zeros((total_samples, 2), dtype=np.float32)
    for idx, s in enumerate(sample_s):
        seg_idx = int(np.clip(np.floor(s), 0, total_segments - 1))
        t_local = float(s - seg_idx)
        sample_pos[idx] = catmull_rom_segment(seg_idx, t_local)

    seg_vecs = sample_pos[1:] - sample_pos[:-1]
    seg_len = np.linalg.norm(seg_vecs, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])  # [total_samples]
    total_len = float(arc[-1]) + 1e-9

    logger.info(
        "[PATH/SPLINE] n_segments=%d, samples=%d, total_len=%.3f",
        total_segments,
        total_samples,
        total_len,
    )

    def s_from_arc(target_len: float) -> float:
        """Inverse mapping: arc length -> global param s in [0, total_segments]."""
        target_len = np.clip(target_len, 0.0, total_len)
        k = int(np.searchsorted(arc, target_len, side="right"))
        if k <= 0:
            return float(sample_s[0])
        if k >= len(arc):
            return float(sample_s[-1])

        l0 = arc[k - 1]
        l1 = arc[k]
        s0 = sample_s[k - 1]
        s1 = sample_s[k]

        if l1 <= l0 + 1e-9:
            return float(s0)

        alpha = float((target_len - l0) / (l1 - l0))
        return float(s0 + alpha * (s1 - s0))

    def pos_from_arc(target_len: float) -> Tuple[np.ndarray, int, float]:
        """
        Return (pos, seg_idx, u_local) for a given arc length.

        - target_len is clamped to [0, total_len].
        - s_global in [0, n_segments] is obtained via s_from_arc.
        - seg_idx = floor(s_global) in [0, n_segments-1].
        - u_local = s_global - seg_idx in [0,1] is local param on that segment.
        """
        s_global = s_from_arc(target_len)
        seg_idx = int(np.clip(np.floor(s_global), 0, total_segments - 1))
        u_local = float(s_global - seg_idx)
        u_local = float(np.clip(u_local, 0.0, 1.0))
        pos = catmull_rom_segment(seg_idx, u_local)
        return pos, seg_idx, u_local

    # --- Generate poses along spline ---

    poses: List[Dict[str, Any]] = []
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    if num_frames == 1:
        p0, seg_idx, u_local = pos_from_arc(0.0)
        p1, _, _ = pos_from_arc(min(0.05 * total_len, total_len))

        eye = np.array([p0[0], base_cam_y, p0[1]], dtype=np.float32)
        center_base = np.array([p1[0], base_cam_y, p1[1]], dtype=np.float32)
        forward = _normalize(center_base - eye)

        behavior = seg_behaviors[min(seg_idx, len(seg_behaviors) - 1)]
        eye_mod, center_mod = _apply_segment_behavior(
            behavior,
            eye,
            center_base,
            forward,
            up,
            diag,
            u_local,
        )

        view = look_at(eye_mod, center_mod, up)
        poses.append(
            {
                "view": view,
                "eye": eye_mod.astype(np.float32).tolist(),
                "center": center_mod.astype(np.float32).tolist(),
            }
        )
        logger.info("[PATH/SPLINE] Generated single-frame pose.")
        return poses

    lookahead_len = float(lookahead_fraction) * total_len

    for i in range(num_frames):
        t = i / float(num_frames - 1)
        cur_len = t * total_len
        ahead_len = min(cur_len + lookahead_len, total_len)

        p_cur, seg_idx, u_local = pos_from_arc(cur_len)
        p_ahead, _, _ = pos_from_arc(ahead_len)

        eye = np.array([p_cur[0], base_cam_y, p_cur[1]], dtype=np.float32)
        center_base = np.array([p_ahead[0], base_cam_y, p_ahead[1]], dtype=np.float32)
        forward = _normalize(center_base - eye)

        behavior = seg_behaviors[min(seg_idx, len(seg_behaviors) - 1)]
        eye_mod, center_mod = _apply_segment_behavior(
            behavior,
            eye,
            center_base,
            forward,
            up,
            diag,
            u_local,
        )

        view = look_at(eye_mod, center_mod, up)
        poses.append(
            {
                "view": view,
                "eye": eye_mod.astype(np.float32).tolist(),
                "center": center_mod.astype(np.float32).tolist(),
            }
        )

    logger.info(
        "[PATH/SPLINE] Generated %d poses along spline through %d waypoints",
        len(poses),
        waypoints_xz.shape[0],
    )
    return poses
