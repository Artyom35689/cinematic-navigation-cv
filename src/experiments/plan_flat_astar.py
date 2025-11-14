#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plan_flat_astar.py

End-to-end (для «плоской» сцены):

  - load unpacked 3DGS PLY (x, y, z, scale_*, rot_*, opacity, f_dc_*)
  - build 2.5D height grid over XY (floor_z / ceil_z по Z)
  - compute free_mask (ячейки, где камера может пролететь на заданной высоте)
  - построить 2D occupancy grid (True = можно лететь)
  - загрузить waypoint'ы (world или cell coords) из JSON
  - между соседними waypoint'ами построить путь A* по сетке
  - конвертировать путь в 3D (x, y, z_cam = floor_z + cam_height_rel)
  - построить плавные позы камеры вдоль пути (Z-up)
  - отрендерить видео gsplat'ом
  - нарисовать путь поверх кадра
  - сохранить MP4 + JSON с путём

Формат JSON с waypoint'ами (один из двух вариантов):

1) В мировых координатах (x, y):

{
  "waypoints_world": [
    [x1, y1],
    [x2, y2],
    ...
  ]
}

2) В координатах ячеек:

{
  "waypoints_cells": [
    [ix1, iy1],
    [ix2, iy2],
    ...
  ]
}

Запуск (пример для ConferenceHall):

  python3 -m src.plan_flat_astar \
    --scene scenes/ConferenceHall.ply \
    --outdir output/ConferenceHall_flat_astar \
    --waypoints-json configs/ConferenceHall_waypoints.json \
    --cell 0.5 \
    --margin-cells 2 \
    --height-fraction 0.3 \
    --frames-per-meter 25 \
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

import heapq
import numpy as np
import torch
from .scene_grid import save_grid_with_axes
from .scene_grid import (
    load_3dgs_unpacked_ply,
    build_height_grid,
    compute_free_mask,
    save_debug_images,
    SceneGrid,
)
from .gsplat_scene import GaussiansNP, build_intrinsics
from .render_utils import render_frames_gsplat, write_video

try:
    import cv2
except ImportError:
    cv2 = None


# ---------------------------------------------------------------------
# 1) Вспомогательные функции для координат и A*
# ---------------------------------------------------------------------


def world_to_cell(grid: SceneGrid, x: float, y: float) -> Tuple[int, int]:
    """
    Перевод мировых координат (x, y) в индексы ячеек (ix, iy).
    XY-плоскость та же, что и в build_height_grid (origin_xy + cell_size).
    """
    ix = int((x - float(grid.origin_xy[0])) / float(grid.cell_size))
    iy = int((y - float(grid.origin_xy[1])) / float(grid.cell_size))

    ix = max(0, min(grid.nx - 1, ix))
    iy = max(0, min(grid.ny - 1, iy))
    return ix, iy


def snap_to_nearest_free(
    occ: np.ndarray,
    ix: int,
    iy: int,
    max_radius: int = 5,
) -> Tuple[int, int]:
    """
    Если (ix, iy) не свободная ячейка, ищем ближайшую свободную
    в квадрате радиуса <= max_radius.
    """
    ny, nx = occ.shape
    if 0 <= ix < nx and 0 <= iy < ny and occ[iy, ix]:
        return ix, iy

    best = None
    for r in range(1, max_radius + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                jx = ix + dx
                jy = iy + dy
                if 0 <= jx < nx and 0 <= jy < ny and occ[jy, jx]:
                    best = (jx, jy)
                    break
            if best is not None:
                break
        if best is not None:
            break

    if best is None:
        raise RuntimeError(
            f"Cannot snap waypoint ({ix},{iy}) to free cell within radius {max_radius}"
        )
    return best


def erode_mask(mask: np.ndarray, margin_cells: int) -> np.ndarray:
    """
    Простейшая "эрозия": ячейка остаётся True, если в квадрате
    радиуса margin_cells вокруг неё все ячейки тоже True.
    """
    if margin_cells <= 0:
        return mask.copy()

    ny, nx = mask.shape
    out = np.zeros_like(mask, dtype=bool)

    for iy in range(ny):
        y0 = max(0, iy - margin_cells)
        y1 = min(ny - 1, iy + margin_cells)
        for ix in range(nx):
            if not mask[iy, ix]:
                continue
            x0 = max(0, ix - margin_cells)
            x1 = min(nx - 1, ix + margin_cells)
            if mask[y0 : y1 + 1, x0 : x1 + 1].all():
                out[iy, ix] = True

    return out


def build_occupancy_from_grid(
    grid: SceneGrid,
    free_mask: np.ndarray,
    margin_cells: int,
) -> np.ndarray:
    """
    Строим 2D occupancy grid для A*:
      True  = можно лететь
      False = стена / препятствие / неизвестно

    По сути это free_mask с учётом отступа margin_cells от стен.
    """
    occ = free_mask.copy()
    if margin_cells > 0:
        eroded = erode_mask(occ, margin_cells)
        if eroded.any():
            logging.info(
                "[INFO] Erosion with margin_cells=%d reduced free cells from %d to %d",
                margin_cells,
                int(occ.sum()),
                int(eroded.sum()),
            )
            occ = eroded
        else:
            logging.warning(
                "[WARN] Erosion removed all free cells, using original free_mask"
            )
    return occ


def load_waypoints_cells(
    json_path: Path,
    grid: SceneGrid,
    occ: np.ndarray,
    max_snap_radius: int = 5,
) -> List[Tuple[int, int]]:
    """
    Загружаем waypoint'ы из JSON в виде индексов ячеек (ix, iy).

    Поддерживаем два формата:
      - "waypoints_cells": [[ix,iy], ...]
      - "waypoints_world": [[x,y], ...] (мировые coords -> ячейки)
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))

    cells: List[Tuple[int, int]] = []

    if "waypoints_cells" in data:
        for item in data["waypoints_cells"]:
            if len(item) != 2:
                raise ValueError("waypoints_cells entries must be [ix, iy]")
            ix = int(round(item[0]))
            iy = int(round(item[1]))
            ix = max(0, min(grid.nx - 1, ix))
            iy = max(0, min(grid.ny - 1, iy))
            ix, iy = snap_to_nearest_free(occ, ix, iy, max_snap_radius=max_snap_radius)
            cells.append((ix, iy))
    elif "waypoints_world" in data:
        for item in data["waypoints_world"]:
            if len(item) != 2:
                raise ValueError("waypoints_world entries must be [x, y]")
            wx = float(item[0])
            wy = float(item[1])
            ix, iy = world_to_cell(grid, wx, wy)
            ix, iy = snap_to_nearest_free(occ, ix, iy, max_snap_radius=max_snap_radius)
            cells.append((ix, iy))
    else:
        raise ValueError(
            "Waypoint JSON must contain either 'waypoints_cells' or 'waypoints_world'"
        )

    if len(cells) < 2:
        raise ValueError("Need at least 2 waypoints")

    logging.info("[INFO] Loaded %d waypoints (cells)", len(cells))
    return cells


NEIGHBORS_8 = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
]


def _heuristic(ix: int, iy: int, gx: int, gy: int) -> float:
    return math.hypot(gx - ix, gy - iy)


def astar_path(
    occ: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """
    2D A* по occupancy grid.

      occ[iy,ix] = True  -> можно
                   False -> нельзя

    start, goal: (ix, iy)
    Возвращает список (ix,iy) от start до goal включительно.
    """
    ny, nx = occ.shape
    sx, sy = start
    gx, gy = goal

    if not (0 <= sx < nx and 0 <= sy < ny):
        raise ValueError("start out of bounds")
    if not (0 <= gx < nx and 0 <= gy < ny):
        raise ValueError("goal out of bounds")
    if not occ[sy, sx]:
        raise ValueError("start cell is not free")
    if not occ[gy, gx]:
        raise ValueError("goal cell is not free")

    start_key = (sx, sy)
    goal_key = (gx, gy)

    open_heap: List[Tuple[float, float, Tuple[int, int]]] = []
    heapq.heappush(open_heap, (_heuristic(sx, sy, gx, gy), 0.0, start_key))

    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    gscore: Dict[Tuple[int, int], float] = {start_key: 0.0}

    while open_heap:
        f, g, (ix, iy) = heapq.heappop(open_heap)
        if (ix, iy) == goal_key:
            # восстановить путь
            path: List[Tuple[int, int]] = [(ix, iy)]
            while (ix, iy) in came_from:
                ix, iy = came_from[(ix, iy)]
                path.append((ix, iy))
            path.reverse()
            return path

        if g > gscore.get((ix, iy), float("inf")) + 1e-8:
            # устаревшая запись
            continue

        for dx, dy in NEIGHBORS_8:
            jx = ix + dx
            jy = iy + dy
            if not (0 <= jx < nx and 0 <= jy < ny):
                continue
            if not occ[jy, jx]:
                continue

            step_cost = math.hypot(dx, dy)
            tentative_g = g + step_cost
            old_g = gscore.get((jx, jy), float("inf"))
            if tentative_g + 1e-8 < old_g:
                gscore[(jx, jy)] = tentative_g
                came_from[(jx, jy)] = (ix, iy)
                f_new = tentative_g + _heuristic(jx, jy, gx, gy)
                heapq.heappush(open_heap, (f_new, tentative_g, (jx, jy)))

    raise RuntimeError("A* failed to find a path between waypoints")


# ---------------------------------------------------------------------
# 2) Геометрия пути и позы камеры (Z-up)
# ---------------------------------------------------------------------


def _polyline_arclength(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    points: (N,3)
    -> s: (N,) накопленная длина
       L: полная длина.
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


def look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    """
    Строим world->camera (4x4), Z-up в мировых координатах.

    +X_cam — вправо, +Y_cam — вверх, +Z_cam — вперёд (к center).
    """
    eye = np.asarray(eye, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    forward = center - eye
    fn = np.linalg.norm(forward)
    if fn < 1e-8:
        forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        forward = forward / fn

    un = np.linalg.norm(up)
    if un < 1e-8:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        up = up / un

    right = np.cross(forward, up)
    rn = np.linalg.norm(right)
    if rn < 1e-8:
        up = np.array([0.0, 0.0, 1.0 + 1e-3], dtype=np.float32)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
    else:
        right /= rn

    new_up = np.cross(right, forward)

    R = np.stack([right, new_up, forward], axis=0).astype(np.float32)  # (3,3)
    t = -R @ eye

    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = R
    view[:3, 3] = t
    return view


def build_poses_along_path(
    polyline_xyz: np.ndarray,
    num_frames: int,
    up_vec: np.ndarray | None = None,
    smooth_window: int = 7,
    back_offset: float = 0.2,
) -> List[Dict[str, Any]]:
    """
    Строим позы камеры вдоль 3D-пути:

      - равномерно по длине (примерно постоянная скорость),
      - сглаживаем центры,
      - направление вперёд вдоль траектории,
      - камера чуть позади точки пути (back_offset).

    Z-up: up_vec по умолчанию = [0, 0, 1].
    """
    pts = np.asarray(polyline_xyz, dtype=np.float32)
    if pts.shape[0] < 2:
        raise RuntimeError("Path polyline must have at least 2 points")

    if up_vec is None:
        up_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        up_vec = np.asarray(up_vec, dtype=np.float32)

    # равномерная выборка по длине
    centers = _sample_polyline(pts, num_frames)
    centers_sm = _smooth_positions(centers, smooth_window)

    poses: List[Dict[str, Any]] = []
    for i in range(num_frames):
        center = centers_sm[i]

        if 0 < i < num_frames - 1:
            forward = centers_sm[i + 1] - centers_sm[i - 1]
        elif i == 0 and num_frames > 1:
            forward = centers_sm[1] - centers_sm[0]
        elif i == num_frames - 1 and num_frames > 1:
            forward = centers_sm[i] - centers_sm[i - 1]
        else:
            forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        norm = np.linalg.norm(forward)
        if norm < 1e-5:
            forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            forward = forward / norm

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
# 3) Визуализация сетки и пути
# ---------------------------------------------------------------------


def save_grid_with_path(
    grid: SceneGrid,
    occ: np.ndarray,
    path_cells: List[Tuple[int, int]],
    out_png: Path,
    scale: int = 4,
) -> None:
    """
    Топ-даун визуализация occupancy:
      - фон: тёмный
      - blocked: красный
      - free:   зелёный
      - путь:   жёлтый
    """
    if cv2 is None:
        logging.warning("[WARN] cv2 not available, skip grid+path image")
        return

    ny, nx = occ.shape
    img = np.zeros((ny, nx, 3), dtype=np.uint8)
    img[:, :] = (20, 20, 20)

    # ячейки, где вообще есть пол/потолок
    occupied_geo = ~np.isnan(grid.floor_z)
    img[occupied_geo & ~occ] = (0, 0, 255)   # красный = блок
    img[occ] = (0, 255, 0)                   # зелёный = свободно

    # путь
    for ix, iy in path_cells:
        if 0 <= ix < nx and 0 <= iy < ny:
            img[iy, ix] = (0, 255, 255)      # жёлтый

    img_big = cv2.resize(
        img, (nx * scale, ny * scale), interpolation=cv2.INTER_NEAREST
    )
    cv2.imwrite(str(out_png), img_big)
    logging.info("[INFO] grid+path image saved: %s", out_png)


def overlay_path_on_frames(
    frames_bgr: List[np.ndarray],
    poses: List[Dict[str, Any]],
    path_xyz: np.ndarray,
    width: int,
    height: int,
    fov_deg: float,
) -> List[np.ndarray]:
    """
    Проецируем точки пути в изображение и рисуем маленькие жёлтые кружки
    на каждом кадре.
    """
    if cv2 is None:
        logging.warning("[WARN] cv2 not available, skip path overlay")
        return frames_bgr

    K = build_intrinsics(fov_deg, width, height)  # (3,3)

    pts3 = np.asarray(path_xyz, dtype=np.float32)
    if pts3.ndim != 2 or pts3.shape[1] != 3:
        raise ValueError(f"path_xyz must be (N,3), got {pts3.shape}")

    N = pts3.shape[0]
    pts_h = np.concatenate(
        [pts3, np.ones((N, 1), dtype=np.float32)],
        axis=1,
    )  # (N,4)

    frames_out: List[np.ndarray] = []
    for i, frame in enumerate(frames_bgr):
        view = np.asarray(poses[i]["view"], dtype=np.float32)  # (4,4)
        cam_pts = (view @ pts_h.T).T   # (N,4)
        xyz = cam_pts[:, :3]
        z = xyz[:, 2]

        mask = z > 0.1
        xyz_vis = xyz[mask]
        if xyz_vis.shape[0] == 0:
            frames_out.append(frame)
            continue

        proj = (K @ xyz_vis.T).T
        u = proj[:, 0] / proj[:, 2]
        v = proj[:, 1] / proj[:, 2]

        img = frame.copy()
        for uu, vv in zip(u, v):
            uu_i = int(round(float(uu)))
            vv_i = int(round(float(vv)))
            if 0 <= uu_i < width and 0 <= vv_i < height:
                cv2.circle(img, (uu_i, vv_i), 2, (0, 255, 255), -1)

        frames_out.append(img)

    return frames_out


# ---------------------------------------------------------------------
# 4) CLI и основной пайплайн
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Flat A* planner over 2D grid + cinematic fly-through with gsplat "
            "(Z-up, unpacked 3DGS PLY)."
        )
    )
    p.add_argument("--scene", type=str, required=True, help="Path to unpacked 3DGS PLY")
    p.add_argument("--outdir", type=str, required=True, help="Output directory")
    p.add_argument(
        "--waypoints-json",
        type=str,
        required=True,
        help="JSON file with 'waypoints_world' or 'waypoints_cells'.",
    )

    # Рендер + скорость
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
        default=25.0,
        help=(
            "Number of frames per meter along path. "
            "Duration = (frames_per_meter * path_length) / fps."
        ),
    )
    p.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help=(
            "Fallback: if frames-per-meter <= 0, use fixed duration in seconds "
            "instead."
        ),
    )
    p.add_argument(
        "--smooth-window",
        type=int,
        default=9,
        help="Odd window size for smoothing camera positions along path.",
    )
    p.add_argument(
        "--back-offset",
        type=float,
        default=0.5,
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
        "--cell",
        type=float,
        default=0.5,
        help="2D grid cell size in world units (XY plane).",
    )
    p.add_argument(
        "--margin-cells",
        type=int,
        default=2,
        help="Safety margin in cells from obstacles (grid erosion).",
    )
    p.add_argument(
        "--height-fraction",
        type=float,
        default=0.3,
        help=(
            "Camera height as fraction of typical (ceil_z - floor_z). "
            "Used to set z_cam = floor_z + cam_height_rel."
        ),
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for gsplat rendering.",
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

    way_json_path = Path(args.waypoints_json)

    # resolution
    try:
        w_str, h_str = args.res.lower().split("x")
        width, height = int(w_str), int(h_str)
    except Exception as e:
        raise SystemExit(f"Invalid --res '{args.res}', expected WxH, error: {e}")

    # -----------------------------------------------------------------
    # 4.1 load Gaussians (unpacked 3DGS PLY)
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
    # 4.2 build 2.5D height grid and free mask
    # -----------------------------------------------------------------
    logging.info("[INFO] Building 2.5D grid: cell_size=%.3f", args.cell)
    grid = build_height_grid(
        gauss.means,
        cell_size=args.cell,
    )

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

    # debug картинки сетки (floor_z + free_mask)
    save_grid_with_axes(
        grid,
        free_mask,
        out_png=outdir / "grid_free_axes.png",
        scale=8,           # можно 6–10
        tick_step_world=5.0,  # шаг подписей по координатам, в метрах/юнитах
    )

    # -----------------------------------------------------------------
    # 4.3 occupancy grid для A* и waypoint'ы
    # -----------------------------------------------------------------
    occ = build_occupancy_from_grid(
        grid,
        free_mask=free_mask,
        margin_cells=args.margin_cells,
    )
    logging.info(
        "[INFO] occupancy free cells: %d / %d",
        int(occ.sum()),
        int(grid.nx * grid.ny),
    )

    way_cells = load_waypoints_cells(
        way_json_path,
        grid=grid,
        occ=occ,
        max_snap_radius=5,
    )

    # -----------------------------------------------------------------
    # 4.4 A* по отрезкам между waypoint'ами
    # -----------------------------------------------------------------
    path_cells: List[Tuple[int, int]] = []
    for i in range(len(way_cells) - 1):
        start = way_cells[i]
        goal = way_cells[i + 1]
        logging.info(
            "[INFO] A* segment %d: start=%s goal=%s",
            i,
            start,
            goal,
        )
        seg = astar_path(occ, start, goal)
        if i > 0:
            # убрать дублирование первой точки сегмента
            seg = seg[1:]
        path_cells.extend(seg)

    logging.info("[INFO] A* total path cells: %d", len(path_cells))

    if len(path_cells) < 2:
        raise RuntimeError("Resulting A* path is too short")

    # -----------------------------------------------------------------
    # 4.5 конвертация пути в 3D (x, y, z_cam)
    # -----------------------------------------------------------------
    path_xyz = []
    for ix, iy in path_cells:
        p = grid.cell_to_world(ix, iy, z_offset=cam_height_rel)
        path_xyz.append(p)
    path_xyz = np.asarray(path_xyz, dtype=np.float32)

    # картинка сетки с траекторией (top-down)
    save_grid_with_path(
        grid,
        occ=occ,
        path_cells=path_cells,
        out_png=outdir / "grid_path_flat_astar.png",
        scale=4,
    )

    # -----------------------------------------------------------------
    # 4.6 длина пути + число кадров
    # -----------------------------------------------------------------
    _, path_length = _polyline_arclength(path_xyz)

    if args.frames_per_meter > 0.0:
        num_frames = max(1, int(round(path_length * args.frames_per_meter)))
        seconds = num_frames / args.fps
        mode = "frames_per_meter"
    else:
        seconds = max(1e-3, float(args.seconds))
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
    # 4.7 позы камеры вдоль пути (Z-up)
    # -----------------------------------------------------------------
    logging.info(
        "[INFO] Building camera poses: smooth_window=%d, back_offset=%.3f",
        args.smooth_window,
        args.back_offset,
    )
    poses = build_poses_along_path(
        path_xyz,
        num_frames=num_frames,
        up_vec=np.array([0.0, 0.0, 1.0], dtype=np.float32),  # Z-up
        smooth_window=args.smooth_window,
        back_offset=args.back_offset,
    )
    logging.info("[INFO] Camera poses: %d", len(poses))

    # -----------------------------------------------------------------
    # 4.8 рендер с gsplat
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
    frames_bgr = render_frames_gsplat(
        gauss,
        poses,
        width=width,
        height=height,
        fov_deg=args.fov,
        device=device,
    )

    # -----------------------------------------------------------------
    # 4.9 оверлей пути на кадры
    # -----------------------------------------------------------------
    frames_bgr = overlay_path_on_frames(
        frames_bgr,
        poses,
        path_xyz,
        width=width,
        height=height,
        fov_deg=args.fov,
    )

    # -----------------------------------------------------------------
    # 4.10 запись видео
    # -----------------------------------------------------------------
    out_mp4 = outdir / "flat_astar_tour.mp4"
    write_video(frames_bgr, out_mp4, fps=args.fps)
    logging.info("[INFO] Video written: %s", out_mp4)

    # -----------------------------------------------------------------
    # 4.11 JSON с путём и метаданными
    # -----------------------------------------------------------------
    cam_json = outdir / "flat_astar_path.json"
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
            "cam_height_rel": float(cam_height_rel),
            "frames_per_meter": float(args.frames_per_meter),
            "smooth_window": int(args.smooth_window),
            "back_offset": float(args.back_offset),
            "mode": mode,
        },
        "waypoints_cells": [list(p) for p in way_cells],
        "path": {
            "cells": [list(p) for p in path_cells],
            "points": path_xyz.astype(float).tolist(),
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
        "render_backend": "gsplat_rgb_flat_astar",
    }
    cam_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logging.info("[INFO] Camera path JSON written: %s", cam_json)


if __name__ == "__main__":
    main()
