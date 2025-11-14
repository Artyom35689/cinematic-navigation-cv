#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plan_and_render_centerline.py

End-to-end:
  - load unpacked 3DGS PLY
  - build 2.5D height grid over horizontal plane
  - compute centerline path through free space
  - build smooth camera poses along this path
  - render video with gsplat
  - write MP4 + camera path JSON + debug grid images

Пример запуска:

  python3 -m src.plan_and_render_centerline \
    --scene scenes/ConferenceHall.ply \
    --outdir output/ConferenceHall_centerline \
    --cell 0.5 \
    --margin-cells 2 \
    --height-fraction 0.3 \
    --bin-size 1.0 \
    --frames-per-meter 40 \
    --fps 24 --fov 70 --res 1280x720 \
    --max-splats 2000000
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from .scene_grid import (
    load_3dgs_unpacked_ply,
    build_height_grid,
    compute_free_mask,
    save_debug_images,
    SceneGrid,
)

from .gsplat_scene import GaussiansNP, build_intrinsics
from .render_utils import (
    render_frames_gsplat,
    write_video,
    render_path_gsplat_to_video,
)

try:
    import cv2
except ImportError:
    cv2 = None


# ---------------------------------------------------------------------
# 1) Камера и движение по polyline
# ---------------------------------------------------------------------


def look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    """
    Build world->camera view matrix (4x4, row-major) для gsplat.

    Совместим с build_camera из debug_one_frame:
      +Z_cam смотрит вперёд (к center),
      +X_cam вправо,
      +Y_cam вверх.
    """
    eye = np.asarray(eye, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    # направление вперёд: от камеры к цели
    forward = center - eye
    fn = np.linalg.norm(forward)
    if fn < 1e-8:
        forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    else:
        forward = forward / fn

    un = np.linalg.norm(up)
    if un < 1e-8:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        up = up / un

    # правая ось
    right = np.cross(forward, up)
    rn = np.linalg.norm(right)
    if rn < 1e-8:
        up = np.array([0.0, 1.0, 0.001], dtype=np.float32)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
    else:
        right /= rn

    new_up = np.cross(right, forward)

    # R: строки — оси камеры в мировых координатах
    R = np.stack([right, new_up, forward], axis=0).astype(np.float32)  # (3,3)
    t = -R @ eye  # (3,)

    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = R
    view[:3, 3] = t
    return view


def _polyline_arclength(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    points: (N,3)
    Возвращает:
      s: (N,) накопленная длина
      L: полная длина
    """
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] < 2:
        s = np.zeros((points.shape[0],), dtype=np.float32)
        return s, 0.0

    diffs = points[1:] - points[:-1]
    seg_len = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    return s, float(s[-1])


def _sample_polyline(points: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Равномерная по длине выборка вдоль polyline.

    points: (N,3)
    num_samples: K
    -> (K,3)
    """
    points = np.asarray(points, dtype=np.float32)
    N = points.shape[0]
    if N == 0:
        raise ValueError("Empty polyline")
    if N == 1 or num_samples == 1:
        return np.repeat(points[:1], num_samples, axis=0)

    s, L = _polyline_arclength(points)
    if L < 1e-6:
        return np.repeat(points[:1], num_samples, axis=0)

    ts = np.linspace(0.0, L, num_samples, dtype=np.float32)

    sampled = np.zeros((num_samples, 3), dtype=np.float32)
    for i, t in enumerate(ts):
        k = np.searchsorted(s, t, side="right") - 1
        k = int(np.clip(k, 0, N - 2))

        s0, s1 = s[k], s[k + 1]
        if s1 <= s0 + 1e-8:
            alpha = 0.0
        else:
            alpha = float((t - s0) / (s1 - s0))

        sampled[i] = (1.0 - alpha) * points[k] + alpha * points[k + 1]

    return sampled


def _smooth_positions(pts: np.ndarray, window: int) -> np.ndarray:
    """
    Простое скользящее среднее по позициям (для сглаживания поворотов).

    pts: (N,3)
    window: нечётное число, >=1
    """
    pts = np.asarray(pts, dtype=np.float32)
    N = pts.shape[0]
    if window <= 1 or N <= 1:
        return pts.copy()

    if window % 2 == 0:
        window += 1
    half = window // 2

    out = np.zeros_like(pts)
    for i in range(N):
        i0 = max(0, i - half)
        i1 = min(N - 1, i + half)
        out[i] = pts[i0 : i1 + 1].mean(axis=0)
    return out


def build_poses_along_path(
    polyline_xyz: np.ndarray,
    num_frames: int,
    up_vec: np.ndarray | None = None,
    smooth_window: int = 7,
    back_offset: float = 0.2,
) -> List[Dict[str, Any]]:
    """
    Полёт вдоль polyline с примерно постоянной скоростью.

    polyline_xyz: (N,3) — центр-линия в мировых координатах
    num_frames: количество кадров в видео
    up_vec: глобальный up (по умолчанию Y-вверх)
    smooth_window: окно сглаживания позиций (в кадрах, нечётное)
    back_offset: насколько камера отстоит назад по направлению движения (м)

    -> список поз [{"view", "eye", "center"}, ...].
    """
    pts = np.asarray(polyline_xyz, dtype=np.float32)
    if pts.shape[0] < 2:
        raise RuntimeError("Path polyline must have at least 2 points")

    if up_vec is None:
        up_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        up_vec = np.asarray(up_vec, dtype=np.float32)

    # Равномерная выборка вдоль пути
    centers = _sample_polyline(pts, num_frames)
    # Сглаживаем траекторию центров
    centers_sm = _smooth_positions(centers, smooth_window)

    poses: List[Dict[str, Any]] = []
    for i in range(num_frames):
        center = centers_sm[i]

        # Направление вперёд по центральной разности
        if 0 < i < num_frames - 1:
            forward = centers_sm[i + 1] - centers_sm[i - 1]
        elif i == 0 and num_frames > 1:
            forward = centers_sm[1] - centers_sm[0]
        elif i == num_frames - 1 and num_frames > 1:
            forward = centers_sm[i] - centers_sm[i - 1]
        else:
            forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)

        norm = np.linalg.norm(forward)
        if norm < 1e-5:
            forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        else:
            forward = forward / norm

        # Камера чуть позади точки пути
        eye = center - forward * back_offset

        view = look_at(eye, center, up_vec)
        poses.append(
            {
                "view": view,
                "eye": eye.tolist(),
                "center": center.tolist(),
            }
        )

    return poses


# ---------------------------------------------------------------------
# 2) Центр-линия по 2.5D-сетке
# ---------------------------------------------------------------------


def erode_free_mask(free_mask: np.ndarray, margin_cells: int) -> np.ndarray:
    """
    Простейшая "эрозия" free-mask: ячейка безопасна, если
    все клетки в квадрате радиуса margin_cells вокруг неё тоже свободны.
    """
    if margin_cells <= 0:
        return free_mask.copy()

    ny, nx = free_mask.shape
    safe = np.zeros_like(free_mask, dtype=bool)

    for iy in range(ny):
        y0 = max(0, iy - margin_cells)
        y1 = min(ny - 1, iy + margin_cells)
        for ix in range(nx):
            if not free_mask[iy, ix]:
                continue
            x0 = max(0, ix - margin_cells)
            x1 = min(nx - 1, ix + margin_cells)
            if free_mask[y0 : y1 + 1, x0 : x1 + 1].all():
                safe[iy, ix] = True

    return safe


def build_centerline_from_grid(
    grid: SceneGrid,
    free_mask: np.ndarray,
    cam_height_rel: float,
    margin_cells: int,
    bin_size_m: float,
    density_quantile: float = 0.3,
    min_dense_cells: int = 100,
) -> np.ndarray:
    """
    Строим 3D центр-линию из 2.5D-сетки.

    Шаги:
      1) erode_free_mask(free_mask, margin_cells) -> safe_mask
         (безопасные клетки с запасом до стен).
      2) По safe_mask берём распределение count (сколько точек в ячейке).
      3) Оставляем только наиболее плотные клетки:
           count >= quantile(count, density_quantile)
         (если density_quantile <= 0, используем все safe-клетки).
      4) По этим плотным safe-клеткам берём 3D-точки камерного центра:
           cell_to_world(ix, iy, z_offset=cam_height_rel)
      5) PCA по горизонтали, бинирование вдоль главной оси,
         усреднение в каждом бине -> centerline_xyz.
    """
    ny, nx = free_mask.shape

    # 1) базовая эрозия по margin_cells
    safe_mask = erode_free_mask(free_mask, margin_cells)
    n_safe = int(safe_mask.sum())
    logging.info(
        "[INFO] safe cells after margin=%d: %d",
        margin_cells,
        n_safe,
    )
    if n_safe < 10:
        logging.warning(
            "[WARN] Too few safe cells (%d), falling back to free_mask",
            n_safe,
        )
        safe_mask = free_mask
        n_safe = int(safe_mask.sum())
        if n_safe < 5:
            raise RuntimeError("Too few free cells to build a meaningful path")

    # 2) фильтр по плотности (по count)
    mask = safe_mask.copy()
    counts_safe = grid.count[safe_mask]
    if density_quantile > 0.0 and counts_safe.size > 0:
        thr = float(np.quantile(counts_safe, density_quantile))
        dense_mask = grid.count >= thr
        cand_mask = safe_mask & dense_mask
        n_dense = int(cand_mask.sum())
        logging.info(
            "[INFO] density filter: quantile=%.2f thr=%d -> dense cells=%d",
            density_quantile,
            int(round(thr)),
            n_dense,
        )
        if n_dense >= min_dense_cells:
            mask = cand_mask
        else:
            logging.warning(
                "[WARN] dense cells (%d) < min_dense_cells=%d, "
                "falling back to safe_mask",
                n_dense,
                min_dense_cells,
            )

    n_mask = int(mask.sum())
    if n_mask < 10:
        logging.warning(
            "[WARN] mask cells (%d) too small, fallback to free_mask", n_mask
        )
        mask = free_mask
        n_mask = int(mask.sum())
        if n_mask < 5:
            raise RuntimeError("Too few usable cells after density filtering")

    # 3) собираем 3D точки (центры ячеек) по mask
    pts_xy = []
    pts_xyz = []
    for iy in range(ny):
        for ix in range(nx):
            if not mask[iy, ix]:
                continue
            p = grid.cell_to_world(ix, iy, z_offset=cam_height_rel)
            pts_xy.append(p[:2])
            pts_xyz.append(p)

    pts_xy = np.asarray(pts_xy, dtype=np.float32)
    pts_xyz = np.asarray(pts_xyz, dtype=np.float32)
    logging.info("[INFO] PCA samples (dense safe cells): %d", pts_xy.shape[0])

    # 4) PCA по XY
    center_xy = pts_xy.mean(axis=0)
    X = pts_xy - center_xy
    cov = np.cov(X, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal_axis = eigvecs[:, np.argmax(eigvals)]  # (2,)

    # координата вдоль главной оси
    s_vals = X @ principal_axis
    s_min = float(s_vals.min())
    s_max = float(s_vals.max())
    logging.info("[DEBUG] s_min=%.3f s_max=%.3f along principal axis", s_min, s_max)

    span = s_max - s_min
    if span < 1e-3:
        logging.warning("[WARN] span along principal axis is tiny (%.4f)", span)
        return pts_xyz.mean(axis=0, keepdims=True)

    if bin_size_m <= 0.0:
        bin_size_m = max(1e-3, span / 10.0)

    num_bins = max(2, int(math.ceil(span / bin_size_m)))
    logging.info("[DEBUG] bin_size_m=%.3f num_bins=%d", bin_size_m, num_bins)

    bins_xyz: List[List[np.ndarray]] = [list() for _ in range(num_bins)]

    for p3, s in zip(pts_xyz, s_vals):
        b = int(math.floor((s - s_min) / bin_size_m))
        if b < 0:
            b = 0
        elif b >= num_bins:
            b = num_bins - 1
        bins_xyz[b].append(p3)

    centerline: List[np.ndarray] = []
    for b, bucket in enumerate(bins_xyz):
        if not bucket:
            continue
        arr = np.stack(bucket, axis=0)
        centerline.append(arr.mean(axis=0))

    if len(centerline) < 2:
        raise RuntimeError(
            f"Centerline too short after binning (len={len(centerline)}); "
            f"try smaller --bin-size or lower --density-quantile."
        )

    centerline_xyz = np.stack(centerline, axis=0).astype(np.float32)
    logging.info("[INFO] centerline points: %d", centerline_xyz.shape[0])
    return centerline_xyz


def save_grid_with_path(
    grid: SceneGrid,
    free_mask: np.ndarray,
    centerline_xyz: np.ndarray,
    out_png: Path,
    scale: int = 4,
) -> None:
    """
    Топ-даун визуализация сетки:
      - фон: тёмный
      - blocked: красный
      - free:   зелёный
      - path:   жёлтый
    """
    if cv2 is None:
        logging.warning("[WARN] cv2 not available, skip grid+path image")
        return

    ny, nx = free_mask.shape

    img = np.zeros((ny, nx, 3), dtype=np.uint8)
    img[:, :] = (20, 20, 20)

    occupied = (~np.isnan(grid.floor_z)) & (~free_mask)
    img[occupied] = (0, 0, 255)
    img[free_mask] = (0, 255, 0)

    # центр-линия -> индексы ячеек
    for p in centerline_xyz:
        x, y = float(p[0]), float(p[1])
        ix = int((x - grid.origin_xy[0]) / grid.cell_size)
        iy = int((y - grid.origin_xy[1]) / grid.cell_size)
        if 0 <= ix < nx and 0 <= iy < ny:
            img[iy, ix] = (0, 255, 255)  # BGR: жёлтый

    img_big = cv2.resize(
        img, (nx * scale, ny * scale), interpolation=cv2.INTER_NEAREST
    )
    cv2.imwrite(str(out_png), img_big)
    logging.info("[INFO] grid+path image saved: %s", out_png)


# ---------------------------------------------------------------------
# 2.5) Оверлей пути на кадры (режим debug / короткие ролики)
# ---------------------------------------------------------------------


def overlay_path_on_frames(
    frames_bgr: List[np.ndarray],
    poses: List[Dict[str, Any]],
    centerline_xyz: np.ndarray,
    width: int,
    height: int,
    fov_deg: float,
) -> List[np.ndarray]:
    """
    Проецируем точки центр-линии в изображение и рисуем их
    на каждом кадре как маленькие жёлтые кружки.
    """
    if cv2 is None:
        logging.warning("[WARN] cv2 not available, skip path overlay")
        return frames_bgr

    K = build_intrinsics(fov_deg, width, height)  # (3,3)

    pts3 = np.asarray(centerline_xyz, dtype=np.float32)
    if pts3.ndim != 2 or pts3.shape[1] != 3:
        raise ValueError(f"centerline_xyz must be (N,3), got {pts3.shape}")

    N = pts3.shape[0]
    pts_h = np.concatenate(
        [pts3, np.ones((N, 1), dtype=np.float32)],
        axis=1,
    )  # (N,4)

    frames_out: List[np.ndarray] = []
    for i, frame in enumerate(frames_bgr):
        view = np.asarray(poses[i]["view"], dtype=np.float32)  # (4,4)
        cam_pts = (view @ pts_h.T).T  # (N,4)
        xyz = cam_pts[:, :3]
        z = xyz[:, 2]

        mask = z > 0.1  # только точки перед камерой
        xyz_vis = xyz[mask]
        if xyz_vis.shape[0] == 0:
            frames_out.append(frame)
            continue

        proj = (K @ xyz_vis.T).T  # (M,3)
        u = proj[:, 0] / proj[:, 2]
        v = proj[:, 1] / proj[:, 2]

        img = frame.copy()
        for uu, vv in zip(u, v):
            uu_i = int(round(float(uu)))
            vv_i = int(round(float(vv)))
            if 0 <= uu_i < width and 0 <= vv_i < height:
                cv2.circle(img, (uu_i, vv_i), 3, (0, 255, 255), -1)  # жёлтые точки

        frames_out.append(img)

    return frames_out


# ---------------------------------------------------------------------
# 3) CLI и основной пайплайн
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build centerline path via 2.5D grid and render cinematic flythrough "
            "with gsplat."
        )
    )
    p.add_argument("--scene", type=str, required=True, help="Path to unpacked 3DGS PLY")
    p.add_argument("--outdir", type=str, required=True, help="Output directory")

    # Рендер + скорость
    p.add_argument(
        "--seconds",
        type=float,
        default=10.0,
        help="Legacy mode: video length in seconds (used if --frames-per-meter <= 0).",
    )
    p.add_argument("--fps", type=int, default=24, help="Frames per second")
    p.add_argument("--fov", type=float, default=70.0, help="Vertical FOV in degrees")
    p.add_argument(
        "--res",
        type=str,
        default="1280x720",
        help="Resolution as WxH (e.g. 1280x720).",
    )
    p.add_argument(
        "--frames-per-meter",
        type=float,
        default=0.0,
        help=(
            "If >0, number of frames per meter along path. "
            "Overrides --seconds. "
            "Duration = (frames_per_meter * path_length) / fps."
        ),
    )
    p.add_argument(
        "--smooth-window",
        type=int,
        default=7,
        help="Odd window size for smoothing camera positions along path.",
    )
    p.add_argument(
        "--back-offset",
        type=float,
        default=0.2,
        help="Distance (m) to place camera behind the path point along direction of motion.",
    )
    p.add_argument(
        "--max-splats",
        type=int,
        default=2_000_000,
        help="Max number of Gaussians (random downsampling).",
    )

    # grid / planner parameters
    p.add_argument(
        "--density-quantile",
        type=float,
        default=0.3,
        help=(
            "Quantile (0..1) of per-cell point count used to keep only dense "
            "free cells when building the centerline. "
            "0 disables density filtering."
        ),
    )
    p.add_argument(
        "--cell",
        type=float,
        default=0.5,
        help="2D grid cell size in world units (horizontal plane).",
    )
    p.add_argument(
        "--margin-cells",
        type=int,
        default=2,
        help="Safety margin in cells from obstacles.",
    )
    p.add_argument(
        "--height-fraction",
        type=float,
        default=0.3,
        help=(
            "Camera height as fraction of typical (ceil_z - floor_z). "
            "Used only to estimate cam_height_rel."
        ),
    )
    p.add_argument(
        "--bin-size",
        type=float,
        default=1.0,
        help="Bin size (m) along principal axis for centerline sampling.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for gsplat rendering.",
    )
    p.add_argument(
        "--overlay-path",
        action="store_true",
        help=(
            "If set: render frames into memory, overlay centerline on each frame "
            "and then write video. Useful for debugging, but uses more RAM. "
            "If not set: use streaming render directly to ffmpeg (no overlay)."
        ),
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    args = parse_args()

    scene_path = Path(args.scene)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # resolution
    try:
        w_str, h_str = args.res.lower().split("x")
        width, height = int(w_str), int(h_str)
    except Exception as e:
        raise SystemExit(f"Invalid --res '{args.res}', expected WxH, error: {e}")

    # -----------------------------------------------------------------
    # 3.1 load Gaussians (unpacked 3DGS PLY)
    # -----------------------------------------------------------------
    means, quats, scales, opacities, colors = load_3dgs_unpacked_ply(
        str(scene_path),
        max_points=args.max_splats,
    )
    gauss = GaussiansNP(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
    )
    logging.info("[INFO] Loaded Gaussians: %d", gauss.means.shape[0])

    # -----------------------------------------------------------------
    # 3.2 build 2.5D height grid and free mask
    # -----------------------------------------------------------------
    logging.info(
        "[INFO] Building 2.5D grid: cell_size=%.3f",
        args.cell,
    )
    grid = build_height_grid(
        gauss.means,
        cell_size=args.cell,
    )

    # Оценим типичную высоту (ceil_z - floor_z) по валидным клеткам
    floor_z = grid.floor_z
    ceil_z = grid.ceil_z
    valid = ~np.isnan(floor_z) & ~np.isnan(ceil_z)
    if valid.any():
        height_span = (ceil_z - floor_z)[valid]
        typical_span = float(np.median(height_span))
    else:
        typical_span = 2.0

    cam_height_rel = float(args.height_fraction * typical_span)
    cam_height_rel = max(0.2, cam_height_rel)
    logging.info(
        "[INFO] typical_span=%.3f, height_fraction=%.3f -> cam_height_rel=%.3f",
        typical_span,
        args.height_fraction,
        cam_height_rel,
    )

    free_mask = compute_free_mask(
        grid,
        cam_height_rel=cam_height_rel,
        min_headroom=0.5,
    )

    # debug картинки сетки
    save_debug_images(
        grid,
        free_mask,
        outdir=outdir,
        prefix="grid",
    )

    # -----------------------------------------------------------------
    # 3.3 centerline in 3D
    # -----------------------------------------------------------------
    logging.info(
        "[INFO] Computing centerline: margin_cells=%d, bin_size=%.3f",
        args.margin_cells,
        args.bin_size,
    )
    centerline_xyz = build_centerline_from_grid(
        grid,
        free_mask=free_mask,
        cam_height_rel=cam_height_rel,
        margin_cells=args.margin_cells,
        bin_size_m=args.bin_size,
        density_quantile=args.density_quantile,
        min_dense_cells=100,
    )

    # картинка сетки с траекторией (top-down)
    save_grid_with_path(
        grid,
        free_mask=free_mask,
        centerline_xyz=centerline_xyz,
        out_png=outdir / "grid_path.png",
        scale=4,
    )

    # -----------------------------------------------------------------
    # 3.4 path length + number of frames
    # -----------------------------------------------------------------
    _, path_length = _polyline_arclength(centerline_xyz)

    if args.frames_per_meter > 0.0:
        num_frames = max(1, int(round(path_length * args.frames_per_meter)))
        seconds = num_frames / args.fps
        mode = "frames_per_meter"
    else:
        seconds = args.seconds
        num_frames = max(1, int(round(seconds * args.fps)))
        mode = "seconds"

    logging.info(
        "[INFO] Path length L=%.3f m, mode=%s, frames=%d, fps=%d, duration=%.2f s",
        path_length,
        mode,
        num_frames,
        args.fps,
        seconds,
    )

    # -----------------------------------------------------------------
    # 3.5 camera poses along centerline
    # -----------------------------------------------------------------
    logging.info(
        "[INFO] Building camera poses: smooth_window=%d, back_offset=%.3f",
        args.smooth_window,
        args.back_offset,
    )
    poses = build_poses_along_path(
        centerline_xyz,
        num_frames=num_frames,
        up_vec=np.array([0.0, 1.0, 0.0], dtype=np.float32),  # Y-up
        smooth_window=args.smooth_window,
        back_offset=args.back_offset,
    )
    logging.info("[INFO] Camera poses: %d", len(poses))

    # -----------------------------------------------------------------
    # 3.6 select device
    # -----------------------------------------------------------------
    if args.device == "cuda":
        if not torch.cuda.is_available():
            logging.warning(
                "[WARN] CUDA requested but not available, switching to CPU"
            )
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logging.info("[INFO] Rendering with gsplat on device: %s", device)
    out_mp4 = outdir / "centerline_tour.mp4"

    # -----------------------------------------------------------------
    # 3.7 render video
    # -----------------------------------------------------------------
    if args.overlay_path:
        # Debug / короткие ролики: рендерим кадры в память, рисуем путь,
        # пишем видео обычным write_video.
        frames_bgr = render_frames_gsplat(
            gauss,
            poses,
            width=width,
            height=height,
            fov_deg=args.fov,
            device=device,
        )
        frames_bgr = overlay_path_on_frames(
            frames_bgr,
            poses,
            centerline_xyz,
            width=width,
            height=height,
            fov_deg=args.fov,
        )
        write_video(frames_bgr, out_mp4, fps=args.fps)
        logging.info("[INFO] Video written (overlay_path): %s", out_mp4)
    else:
        # Основной режим: без накопления кадров, сразу пишем в ffmpeg.
        render_path_gsplat_to_video(
            gauss,
            poses,
            width=width,
            height=height,
            fov_deg=args.fov,
            device=device,
            out_path=out_mp4,
            fps=args.fps,
        )
        logging.info("[INFO] Video written (stream): %s", out_mp4)

    # -----------------------------------------------------------------
    # 3.8 camera path JSON
    # -----------------------------------------------------------------
    cam_json = outdir / "centerline_path.json"
    meta = {
        "scene": str(scene_path),
        "resolution": [width, height],
        "fps": args.fps,
        "seconds": seconds,
        "fov_deg": args.fov,
        "path_length_m": path_length,
        "grid": grid.to_json_meta(),
        "planner": {
            "cell_size": float(args.cell),
            "margin_cells": int(args.margin_cells),
            "height_fraction": float(args.height_fraction),
            "bin_size": float(args.bin_size),
            "cam_height_rel": float(cam_height_rel),
            "frames_per_meter": float(args.frames_per_meter),
            "smooth_window": int(args.smooth_window),
            "back_offset": float(args.back_offset),
            "mode": mode,
            "density_quantile": float(args.density_quantile),
        },
        "path": {
            "points": centerline_xyz.astype(float).tolist(),
        },
        "frames": [
            {
                "index": i,
                "eye": poses[i]["eye"],
                "center": poses[i]["center"],
                "view": np.asarray(poses[i]["view"], dtype=float).tolist(),
            }
            for i in range(len(poses))
        ],
        "render_backend": (
            "gsplat_rgb_centerline_overlay" if args.overlay_path
            else "gsplat_rgb_centerline_stream"
        ),
    }
    cam_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logging.info("[INFO] Camera path JSON written: %s", cam_json)


if __name__ == "__main__":
    main()
