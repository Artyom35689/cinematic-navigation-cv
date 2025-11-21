#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scene normalization utilities.

Goal:
    - Keep semantic world up axis as +Y.
    - Optionally:
        1) Level the floor: rotate scene so that floor plane normal == +Y
           (removes pitch/roll).
        2) Align yaw: rotate around Y so that dominant horizontal direction
           aligns with +Z.
        3) Optionally shift floor so that min Y ≈ 0.

Usage:
    from src.analysis.scene_normalizer import normalize_scene_axes

    gauss_norm, meta = normalize_scene_axes(gauss)
"""

from __future__ import annotations

from typing import Dict, Tuple
import logging
import numpy as np

from ..gsplat_scene import GaussiansNP

logger = logging.getLogger(__name__)


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n < eps:
        return v
    return v / n


def _rot_y(deg: float) -> np.ndarray:
    """Rotation matrix around Y axis by given angle in degrees."""
    rad = np.deg2rad(float(deg))
    c, s = float(np.cos(rad)), float(np.sin(rad))
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float32,
    )


def _axis_angle_to_R(axis: np.ndarray, angle_rad: float, eps: float = 1e-8) -> np.ndarray:
    """
    Rodrigues' rotation formula: build 3x3 rotation for given axis (3,) and angle in radians.
    """
    axis = _normalize(axis, eps=eps)
    if float(np.linalg.norm(axis)) < eps or abs(angle_rad) < eps:
        return np.eye(3, dtype=np.float32)

    x, y, z = float(axis[0]), float(axis[1]), float(axis[2])
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    C = 1.0 - c

    R = np.array(
        [
            [c + x * x * C,      x * y * C - z * s,  x * z * C + y * s],
            [y * x * C + z * s,  c + y * y * C,      y * z * C - x * s],
            [z * x * C - y * s,  z * y * C + x * s,  c + z * z * C],
        ],
        dtype=np.float32,
    )
    return R


def _fit_floor_plane_normal(
    means: np.ndarray,
    floor_band_frac: float = 0.1,
) -> np.ndarray:
    """
    Estimate floor plane normal from the lowest band of points in Y.

    Args:
        means: (N,3) points.
        floor_band_frac: fraction of scene height near min Y to treat as "floor band".

    Returns:
        normal: (3,) unit vector, pointing roughly upward; if estimation fails,
                returns [0,1,0].
    """
    xyz = np.asarray(means, dtype=np.float32)
    y = xyz[:, 1]

    y_min, y_max = float(y.min()), float(y.max())
    height = y_max - y_min
    if height <= 1e-6:
        logger.warning("[NORM] Scene has near-zero height; using default up axis.")
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)

    band = max(0.0, min(1.0, float(floor_band_frac)))
    if band <= 0.0:
        band = 0.1

    y_thresh = y_min + band * height
    mask = y <= y_thresh
    pts_floor = xyz[mask]

    if pts_floor.shape[0] < 10:
        logger.warning(
            "[NORM] Too few points in floor band (got %d), using default up axis.",
            pts_floor.shape[0],
        )
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # PCA / plane fit: normal is eigenvector with smallest eigenvalue
    center = pts_floor.mean(axis=0, keepdims=True)
    pts_centered = pts_floor - center
    cov = np.cov(pts_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx_min = int(np.argmin(eigvals))
    normal = eigvecs[:, idx_min].astype(np.float32)  # (3,)

    normal = _normalize(normal)
    # Make it point "upward" (positive Y)
    if normal[1] < 0.0:
        normal = -normal

    return normal


def normalize_scene_axes(
    gauss: GaussiansNP,
    shift_y_to_zero: bool = True,
    align_floor: bool = True,
    floor_band_frac: float = 0.1,
    align_yaw: bool = True,
    max_floor_tilt_deg: float = 60.0,
    max_yaw_deg: float = 90.0,
) -> Tuple[GaussiansNP, Dict[str, object]]:
    """
    Normalize scene orientation, preserving semantic Y-up:

        1) (optional) Floor leveling (pitch/roll):
           - take lowest band of points in Y;
           - fit floor plane normal n;
           - if angle(n, +Y) <= max_floor_tilt_deg:
                 rotate scene so that n -> +Y
             else:
                 skip floor leveling (to avoid crazy 90+ deg spins).
        2) (optional) Yaw alignment:
           - PCA in XZ;
           - compute yaw so that dominant dir -> +Z;
           - wrap yaw into [-max_yaw_deg, max_yaw_deg];
           - rotate around +Y by this clamped yaw.
        3) (optional) Vertical shift:
           - move scene so that min Y ≈ 0.

    Args:
        gauss: GaussiansNP with `means` at least.
        shift_y_to_zero: if True, shift all points after rotation so that min Y = 0.
        align_floor: if True, estimate floor plane and (maybe) level it.
        floor_band_frac: fraction of scene height near min Y to use as "floor band".
        align_yaw: if True, rotate around Y so that dominant XZ direction -> +Z.
        max_floor_tilt_deg: max allowed pitch/roll correction; if angle between
                            estimated floor normal and +Y is larger, skip leveling.
        max_yaw_deg: clamp yaw so that |yaw| <= max_yaw_deg (in degrees).

    Returns:
        gauss_out: new GaussiansNP with transformed `means`.
        meta: dict with diagnostic info.
    """
    means = np.asarray(gauss.means, dtype=np.float32)
    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError(f"GaussiansNP.means has invalid shape: {means.shape}")

    bbox_min_before = means.min(axis=0)
    bbox_max_before = means.max(axis=0)

    R_total = np.eye(3, dtype=np.float32)

    # -------------------------------------------------------------
    # 1) Floor leveling (pitch/roll)
    # -------------------------------------------------------------
    floor_normal_before = None
    floor_normal_after = None
    R_floor = np.eye(3, dtype=np.float32)
    floor_angle_deg = 0.0

    if align_floor:
        n = _fit_floor_plane_normal(means, floor_band_frac=floor_band_frac)
        floor_normal_before = n.copy()

        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        n = _normalize(n)
        dot_ = float(np.clip(np.dot(n, up), -1.0, 1.0))
        angle = np.arccos(dot_)  # radians
        angle_deg = float(np.rad2deg(angle))

        if angle_deg < 1e-3:
            logger.info("[NORM] Floor already aligned with +Y (angle≈0).")
        elif angle_deg > max_floor_tilt_deg:
            logger.warning(
                "[NORM] Estimated floor tilt %.2f° > max_floor_tilt_deg=%.2f°, "
                "skipping floor leveling.",
                angle_deg,
                max_floor_tilt_deg,
            )
        else:
            axis = np.cross(n, up)  # rotate n -> up
            if np.linalg.norm(axis) < 1e-8:
                axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            R_floor = _axis_angle_to_R(axis, angle)
            means = (R_floor @ means.T).T
            R_total = R_floor @ R_total
            floor_angle_deg = angle_deg
            logger.info(
                "[NORM] Floor leveling applied: angle_deg=%.3f, axis=%s",
                angle_deg,
                np.round(_normalize(axis), 4),
            )

        # Проверка нормали после (даже если не вертели — просто для отладки)
        n_after = _fit_floor_plane_normal(means, floor_band_frac=floor_band_frac)
        floor_normal_after = n_after
    else:
        logger.info("[NORM] Floor leveling disabled; keeping original pitch/roll.")

    # -------------------------------------------------------------
    # 2) Yaw alignment (around +Y), with clamping
    # -------------------------------------------------------------
    yaw_deg_applied = 0.0
    main_dir_before = None
    main_dir_after = None
    R_yaw = np.eye(3, dtype=np.float32)

    max_yaw_deg = float(abs(max_yaw_deg))
    if align_yaw and max_yaw_deg > 0.0:
        xz = means[:, [0, 2]]
        xz_c = xz - xz.mean(axis=0, keepdims=True)
        cov_xz = np.cov(xz_c, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov_xz)
        idx_max = int(np.argmax(eigvals))
        dir_xz = eigvecs[:, idx_max]  # (2,)

        main_dir_before = np.array([dir_xz[0], 0.0, dir_xz[1]], dtype=np.float32)
        main_dir_before = _normalize(main_dir_before)

        x, z = float(main_dir_before[0]), float(main_dir_before[2])
        if abs(x) < 1e-8 and abs(z) < 1e-8:
            logger.info("[NORM] Dominant XZ direction is degenerate; skipping yaw alignment.")
        else:
            # Полный yaw, который бы отправил main_dir -> +Z
            yaw_rad = np.arctan2(-x, z)
            yaw_deg = float(np.rad2deg(yaw_rad))

            # Приводим yaw в диапазон [-180, 180]
            while yaw_deg <= -180.0:
                yaw_deg += 360.0
            while yaw_deg > 180.0:
                yaw_deg -= 360.0

            # Клэмпим к [-max_yaw_deg, max_yaw_deg]
            if abs(yaw_deg) > max_yaw_deg:
                yaw_deg = float(np.sign(yaw_deg) * max_yaw_deg)

            yaw_deg_applied = yaw_deg
            R_yaw = _rot_y(yaw_deg_applied)
            means = (R_yaw @ means.T).T
            R_total = R_yaw @ R_total

            # Повторно оценим главную ось в XZ
            xz_after = means[:, [0, 2]]
            xz_after_c = xz_after - xz_after.mean(axis=0, keepdims=True)
            cov_xz_after = np.cov(xz_after_c, rowvar=False)
            eigvals_a, eigvecs_a = np.linalg.eigh(cov_xz_after)
            idx_max_a = int(np.argmax(eigvals_a))
            dir_xz_a = eigvecs_a[:, idx_max_a]
            main_dir_after = np.array([dir_xz_a[0], 0.0, dir_xz_a[1]], dtype=np.float32)
            main_dir_after = _normalize(main_dir_after)

            logger.info(
                "[NORM] Yaw alignment: yaw_deg_applied=%.3f (|yaw|<=%.1f), "
                "main_dir_before=%s, main_dir_after=%s",
                yaw_deg_applied,
                max_yaw_deg,
                np.round(main_dir_before, 4),
                np.round(main_dir_after, 4),
            )
    else:
        logger.info("[NORM] Yaw alignment disabled; keeping original heading.")

    # -------------------------------------------------------------
    # 3) Optional vertical shift: min Y -> 0
    # -------------------------------------------------------------
    bbox_min_after = means.min(axis=0)
    bbox_max_after = means.max(axis=0)

    y_shift = 0.0
    if shift_y_to_zero:
        y_min_after = float(bbox_min_after[1])
        y_shift = -y_min_after
        means[:, 1] += y_shift
        bbox_min_after = means.min(axis=0)
        bbox_max_after = means.max(axis=0)

    logger.info(
        "[NORM] Final bbox_min=%s, bbox_max=%s, y_shift=%.3f",
        np.round(bbox_min_after, 3),
        np.round(bbox_max_after, 3),
        y_shift,
    )

    gauss_out = GaussiansNP(
        means=means.astype(np.float32),
        quats=gauss.quats,
        scales=gauss.scales,
        opacities=gauss.opacities,
        colors=gauss.colors,
    )

    meta: Dict[str, object] = {
        "R_total": R_total.astype(np.float32).tolist(),
        "R_floor": R_floor.astype(np.float32).tolist(),
        "R_yaw": R_yaw.astype(np.float32).tolist(),
        "floor_normal_before": (
            floor_normal_before.astype(np.float32).tolist()
            if floor_normal_before is not None
            else None
        ),
        "floor_normal_after": (
            floor_normal_after.astype(np.float32).tolist()
            if floor_normal_after is not None
            else None
        ),
        "floor_level_angle_deg": floor_angle_deg,
        "yaw_deg_applied": yaw_deg_applied,
        "main_dir_xz_before": (
            main_dir_before.astype(np.float32).tolist()
            if main_dir_before is not None
            else None
        ),
        "main_dir_xz_after": (
            main_dir_after.astype(np.float32).tolist()
            if main_dir_after is not None
            else None
        ),
        "bbox_min_before": bbox_min_before.astype(np.float32).tolist(),
        "bbox_max_before": bbox_max_before.astype(np.float32).tolist(),
        "bbox_min_after": bbox_min_after.astype(np.float32).tolist(),
        "bbox_max_after": bbox_max_after.astype(np.float32).tolist(),
        "y_shift": y_shift,
    }

    return gauss_out, meta
