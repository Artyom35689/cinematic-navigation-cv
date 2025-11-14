#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
path_runner.py

Высокоуровневый раннер: берёт YAML-конфиг, строит плоский путь по сцене
(waypoints + A* по 2D-сетке XZ) и рендерит ролик с gsplat при Y-up.

Ожидаемая структура cfg (см. пример conferencehall.yaml):

scene:
  ply: "scenes/ConferenceHall.ply"
  max_splats: 2000000

output:
  dir: "output/ConferenceHall_manual"

render:
  fps: 24
  fov_deg: 70
  res: [1280, 720]  # или "1280x720"
  seconds: 10       # fallback, если frames_per_meter <= 0
  device: "cuda"    # или "cpu"

path:
  mode: "astar_waypoints"
  # ВАЖНО: при Y-up здесь [x, z], а не [x, y]
  waypoints_world:
    - [28.0, 30.0]
    - [20.0, 22.0]
    - [10.0, 25.0]
    - [0.0, 25.0]
    - [0.0, -5.0]
  cell_size: 0.5
  margin_cells: 2
  height_fraction: 0.3   # используется, если cam_height_rel не задан
  cam_height_rel: 1.7    # фиксированная высота над полом (по Y)
  min_headroom: 0.5      # запас до потолка (по Y)
  frames_per_meter: 25
  smooth_window: 9
  back_offset: 0.5
  snap_radius: 10
  overlay_path: false    # true — рисовать путь поверх кадра (debug)

detection:
  enabled: false
  model: "yolov8n.pt"
  conf: 0.25
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from .scene_grid import (
    SceneGrid,
    load_3dgs_unpacked_ply,
    build_height_grid,
    compute_free_mask,
)
from .render_utils import (
    render_frames_gsplat,
    write_video,
    run_detection,
    render_path_gsplat_to_video,
)
from .gsplat_scene import GaussiansNP, build_intrinsics

try:
    import cv2
except ImportError:
    cv2 = None


# ---------------------------------------------------------------------
# 1) Waypoints: world <-> grid (XZ-плоскость) + snapping к свободным клеткам
# ---------------------------------------------------------------------


def world_to_cell(grid: SceneGrid, x: float, z: float) -> tuple[int, int]:
    """
    Перевод мировых координат (x, z) в индексы ячеек (ix, iy)
    в XZ-плоскости, с которой работает build_height_grid (при Y-up).
    """
    cx = float(grid.origin_xy[0])  # min_x
    cz = float(grid.origin_xy[1])  # min_z
    s = float(grid.cell_size)

    ix = int((x - cx) / s)
    iy = int((z - cz) / s)

    ix = max(0, min(grid.nx - 1, ix))
    iy = max(0, min(grid.ny - 1, iy))
    return ix, iy


def snap_to_nearest_free(
    occ: np.ndarray,
    ix: int,
    iy: int,
    max_radius: int = 10,
) -> tuple[int, int]:
    """
    Если (ix, iy) не свободная ячейка, ищем ближайшую свободную
    в квадрате радиуса <= max_radius. Если не нашли — кидаем RuntimeError.
    """
    ny, nx = occ.shape
    if 0 <= ix < nx and 0 <= iy < ny and occ[iy, ix]:
        return ix, iy

    best: tuple[int, int] | None = None
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

    logging.info(
        "[path_runner] snapped (%d,%d) -> (%d,%d) within radius=%d",
        ix,
        iy,
        best[0],
        best[1],
        max_radius,
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
    Строим 2D occupancy grid для A* в XZ-плоскости:
      True  = можно лететь
      False = стена / препятствие / неизвестно

    По сути это free_mask с учётом отступа margin_cells от стен.
    """
    occ = free_mask.copy()
    if margin_cells > 0:
        eroded = erode_mask(occ, margin_cells)
        if eroded.any():
            logging.info(
                "[path_runner] Erosion with margin_cells=%d reduced free cells from %d to %d",
                margin_cells,
                int(occ.sum()),
                int(eroded.sum()),
            )
            occ = eroded
        else:
            logging.warning(
                "[path_runner] Erosion removed all free cells, using original free_mask"
            )
    return occ


def load_waypoints_from_config(
    path_cfg: dict,
    grid: SceneGrid,
    occ: np.ndarray,
    max_snap_radius: int = 10,
) -> list[tuple[int, int]]:
    """
    Загружаем waypoints из path_cfg и приводим к индексам ячеек (ix, iy).

    При Y-up:
      path.waypoints_world: [[x, z], ...]   # XZ-плоскость
      или
      path.waypoints_cells: [[ix, iy], ...]
    """
    cells: list[tuple[int, int]] = []

    if "waypoints_world" in path_cfg:
        for item in path_cfg["waypoints_world"]:
            if len(item) != 2:
                raise ValueError("waypoints_world entries must be [x, z]")
            wx = float(item[0])
            wz = float(item[1])
            ix, iy = world_to_cell(grid, wx, wz)
            ix, iy = snap_to_nearest_free(occ, ix, iy, max_radius=max_snap_radius)
            cells.append((ix, iy))

    elif "waypoints_cells" in path_cfg:
        for item in path_cfg["waypoints_cells"]:
            if len(item) != 2:
                raise ValueError("waypoints_cells entries must be [ix, iy]")
            ix = int(round(item[0]))
            iy = int(round(item[1]))
            ix = max(0, min(grid.nx - 1, ix))
            iy = max(0, min(grid.ny - 1, iy))
            ix, iy = snap_to_nearest_free(occ, ix, iy, max_radius=max_snap_radius)
            cells.append((ix, iy))
    else:
        raise ValueError(
            "Config.path must contain either 'waypoints_world' or 'waypoints_cells'"
        )

    if len(cells) < 2:
        raise ValueError("Need at least 2 waypoints")

    logging.info("[path_runner] Loaded %d waypoints (cells)", len(cells))
    return cells


# ---------------------------------------------------------------------
# 2) A* по occupancy-grid
# ---------------------------------------------------------------------

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
    return float(np.hypot(gx - ix, gy - iy))


def astar_path(
    occ: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """
    2D A* по occupancy grid (XZ-плоскость).

      occ[iy,ix] = True  -> можно
                   False -> нельзя

    start, goal: (ix, iy)
    Возвращает список (ix,iy) от start до goal включительно.
    """
    import heapq

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
            path: List[Tuple[int, int]] = [(ix, iy)]
            while (ix, iy) in came_from:
                ix, iy = came_from[(ix, iy)]
                path.append((ix, iy))
            path.reverse()
            return path

        if g > gscore.get((ix, iy), float("inf")) + 1e-8:
            continue

        for dx, dy in NEIGHBORS_8:
            jx = ix + dx
            jy = iy + dy
            if not (0 <= jx < nx and 0 <= jy < ny):
                continue
            if not occ[jy, jx]:
                continue

            step_cost = float(np.hypot(dx, dy))
            tentative_g = g + step_cost
            old_g = gscore.get((jx, jy), float("inf"))
            if tentative_g + 1e-8 < old_g:
                gscore[(jx, jy)] = tentative_g
                came_from[(jx, jy)] = (ix, iy)
                f_new = tentative_g + _heuristic(jx, jy, gx, gy)
                heapq.heappush(open_heap, (f_new, tentative_g, (jx, jy)))

    raise RuntimeError("A* failed to find a path between waypoints")


# ---------------------------------------------------------------------
# 3) Геометрия пути и позы камеры (Y-up)
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
        k = int(np.clip(np.searchsorted(s, t, side="right") - 1, 0, len(s) - 2))
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
    Строим world->camera (4x4) при Y-up.

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
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        up = up / un

    right = np.cross(forward, up)
    rn = np.linalg.norm(right)
    if rn < 1e-8:
        up = np.array([0.0, 1.001, 0.0], dtype=np.float32)
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
    Строим позы камеры вдоль 3D-пути (при Y-up):

      - равномерно по длине (примерно постоянная скорость),
      - сглаживаем центры,
      - направление вперёд вдоль траектории,
      - камера чуть позади точки пути (back_offset).

    up_vec по умолчанию = [0, 1, 0].
    """
    pts = np.asarray(polyline_xyz, dtype=np.float32)
    if pts.shape[0] < 2:
        raise RuntimeError("Path polyline must have at least 2 points")

    if up_vec is None:
        up_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        up_vec = np.asarray(up_vec, dtype=np.float32)

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
                "view": view.astype(np.float32),
                "eye": eye.tolist(),
                "center": center.tolist(),
            }
        )

    return poses


# ---------------------------------------------------------------------
# 4) Визуализация сетки и пути (XZ-плоскость)
# ---------------------------------------------------------------------


def save_grid_with_path(
    grid: SceneGrid,
    occ: np.ndarray,
    path_cells: List[Tuple[int, int]],
    out_png: Path,
    scale: int = 4,
) -> None:
    """
    Топ-даун визуализация occupancy в XZ-плоскости:
      - фон: тёмный
      - blocked: красный
      - free:   зелёный
      - путь:   жёлтый
    """
    if cv2 is None:
        logging.warning("[path_runner] cv2 not available, skip grid+path image")
        return

    ny, nx = occ.shape
    img = np.zeros((ny, nx, 3), dtype=np.uint8)
    img[:, :] = (20, 20, 20)

    occupied_geo = ~np.isnan(grid.floor_z)  # floor_y известна
    img[occupied_geo & ~occ] = (0, 0, 255)   # красный = блок
    img[occ] = (0, 255, 0)                   # зелёный = свободно

    for ix, iy in path_cells:
        if 0 <= ix < nx and 0 <= iy < ny:
            img[iy, ix] = (0, 255, 255)      # жёлтый

    img_big = cv2.resize(
        img, (nx * scale, ny * scale), interpolation=cv2.INTER_NEAREST
    )
    cv2.imwrite(str(out_png), img_big)
    logging.info("[path_runner] grid+path image saved: %s", out_png)


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
        logging.warning("[path_runner] cv2 not available, skip path overlay")
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
# 5) Основной раннер
# ---------------------------------------------------------------------


def run_path_from_config(cfg: Dict[str, Any], det_fn=None) -> None:
    """
    Основной пайплайн: берёт cfg (dict из YAML), строит путь и рендерит видео.
    det_fn оставлен для совместимости, но не используется: детекция
    управляется cfg["detection"].

    Координаты сцены: (x, y, z) с Y-up.
    Плоскость навигации: XZ.
    """
    logging.info("[path_runner] run_path_from_config()")

    scene_cfg = cfg.get("scene", {})
    out_cfg = cfg.get("output", {})
    render_cfg = cfg.get("render", {})
    path_cfg = cfg.get("path", {})
    det_cfg = cfg.get("detection", {})

    scene_path = Path(scene_cfg["ply"])
    outdir = Path(out_cfg.get("dir", "output/path_runner"))
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Render params ---
    res = render_cfg.get("res", [1280, 720])
    if isinstance(res, str):
        w_str, h_str = res.lower().split("x")
        width, height = int(w_str), int(h_str)
    else:
        width, height = int(res[0]), int(res[1])

    fps = int(render_cfg.get("fps", 24))
    fov_deg = float(render_cfg.get("fov_deg", 70.0))
    seconds_fallback = float(render_cfg.get("seconds", 8.0))
    device_name = render_cfg.get("device", "cuda")

    max_splats = int(scene_cfg.get("max_splats", 2_000_000))

    logging.info("[path_runner] Scene: %s", scene_path)
    logging.info("[path_runner] Outdir: %s", outdir)
    logging.info(
        "[path_runner] Resolution: %dx%d, fps=%d, fov=%.1f",
        width,
        height,
        fps,
        fov_deg,
    )

    # --- Load Gaussians ---
    means, quats, scales, opacities, colors = load_3dgs_unpacked_ply(
        str(scene_path),
        max_points=max_splats,
    )
    gauss = GaussiansNP(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
    )
    logging.info("[path_runner] Loaded Gaussians: %d", gauss.means.shape[0])

    # --- Build 2.5D grid (XZ-плоскость, высота по Y) ---
    cell_size = float(path_cfg.get("cell_size", 0.5))
    logging.info("[path_runner] Building 2.5D grid (Y-up): cell_size=%.3f", cell_size)
    grid = build_height_grid(
        gauss.means,
        cell_size=cell_size,
    )

    # --- Compute typical vertical span and camera height (по Y) ---
    floor_y = grid.floor_z
    ceil_y = grid.ceil_z
    valid = ~np.isnan(floor_y) & ~np.isnan(ceil_y)
    if valid.any():
        height_span = (ceil_y - floor_y)[valid]
        typical_span = float(np.median(height_span))
    else:
        typical_span = 2.0

    height_fraction = float(path_cfg.get("height_fraction", 0.3))
    default_cam_height_rel = max(0.2, height_fraction * typical_span)
    cam_height_rel = float(path_cfg.get("cam_height_rel", default_cam_height_rel))
    min_headroom = float(path_cfg.get("min_headroom", 0.5))

    logging.info(
        "[path_runner] typical_span=%.3f, height_fraction=%.3f, "
        "default_cam_height_rel=%.3f, cam_height_rel=%.3f, min_headroom=%.3f",
        typical_span,
        height_fraction,
        default_cam_height_rel,
        cam_height_rel,
        min_headroom,
    )

    free_mask = compute_free_mask(
        grid,
        cam_height_rel=cam_height_rel,
        min_headroom=min_headroom,
    )

    # --- Occupancy for A* ---
    margin_cells = int(path_cfg.get("margin_cells", 2))
    occ = build_occupancy_from_grid(
        grid,
        free_mask=free_mask,
        margin_cells=margin_cells,
    )
    logging.info(
        "[INFO] [path_runner] Occupancy free cells: %d / %d",
        int(occ.sum()),
        int(grid.nx * grid.ny),
    )

    # --- Waypoints ---
    snap_radius = int(path_cfg.get("snap_radius", 10))
    way_cells = load_waypoints_from_config(
        path_cfg,
        grid=grid,
        occ=occ,
        max_snap_radius=snap_radius,
    )

    # --- A* между соседними waypoint'ами ---
    path_cells: List[Tuple[int, int]] = []
    for i in range(len(way_cells) - 1):
        start = way_cells[i]
        goal = way_cells[i + 1]
        logging.info(
            "[path_runner] A* segment %d: start=%s goal=%s",
            i,
            start,
            goal,
        )
        seg = astar_path(occ, start, goal)
        if i > 0:
            seg = seg[1:]
        path_cells.extend(seg)

    logging.info("[path_runner] total A* path cells: %d", len(path_cells))
    if len(path_cells) < 2:
        raise RuntimeError("Resulting A* path is too short")

    # --- 3D путь (x, y, z) при Y-up ---
    path_xyz = np.stack(
        [grid.cell_to_world(ix, iy, z_offset=cam_height_rel) for (ix, iy) in path_cells],
        axis=0,
    ).astype(np.float32)

    # Отладочная картинка сетки с путём (XZ-плоскость)
    save_grid_with_path(
        grid,
        occ=occ,
        path_cells=path_cells,
        out_png=outdir / "grid_path.png",
        scale=4,
    )

    # --- Длина пути и число кадров ---
    _, path_length = _polyline_arclength(path_xyz)
    frames_per_meter = float(path_cfg.get("frames_per_meter", 25.0))

    if frames_per_meter > 0.0 and path_length > 1e-6:
        num_frames = max(1, int(round(path_length * frames_per_meter)))
        seconds = num_frames / fps
        mode = "frames_per_meter"
    else:
        seconds = seconds_fallback
        num_frames = max(1, int(round(seconds * fps)))
        mode = "seconds"

    logging.info(
        "[path_runner] Path length L=%.3f, mode=%s, frames=%d, duration=%.2f s",
        path_length,
        mode,
        num_frames,
        seconds,
    )

    # --- Позы камеры вдоль пути (Y-up) ---
    smooth_window = int(path_cfg.get("smooth_window", 9))
    back_offset = float(path_cfg.get("back_offset", 0.5))
    logging.info(
        "[path_runner] Building camera poses: smooth_window=%d, back_offset=%.3f",
        smooth_window,
        back_offset,
    )
    poses = build_poses_along_path(
        path_xyz,
        num_frames=num_frames,
        up_vec=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        smooth_window=smooth_window,
        back_offset=back_offset,
    )
    logging.info("[path_runner] Camera poses: %d", len(poses))

    # --- Выбор устройства ---
    if device_name == "cuda":
        if not torch.cuda.is_available():
            logging.warning("[path_runner] CUDA requested but not available, using CPU")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logging.info("[path_runner] Rendering on device: %s", device)

    # --- Флаги overlay / detection ---
    overlay_flag = bool(path_cfg.get("overlay_path", False))
    det_enabled = bool(det_cfg.get("enabled", False))

    out_mp4 = outdir / "path_tour.mp4"

    # Можно ли рендерить в потоковом режиме (без хранения кадров)?
    can_stream = (not overlay_flag) and (not det_enabled)

    if can_stream:
        # Память-дружелюбный путь: сразу в ffmpeg, без списка frames_bgr
        render_path_gsplat_to_video(
            gauss=gauss,
            poses=poses,
            width=width,
            height=height,
            fov_deg=fov_deg,
            device=device,
            out_path=out_mp4,
            fps=fps,
        )
        logging.info("[path_runner] Video written (stream): %s", out_mp4)

    else:
        # Старый путь: кадры нужны в памяти (overlay или детекция)
        frames_bgr = render_frames_gsplat(
            gauss,
            poses,
            width=width,
            height=height,
            fov_deg=fov_deg,
            device=device,
        )

        # --- Оверлей пути (debug) ---
        if overlay_flag:
            frames_bgr = overlay_path_on_frames(
                frames_bgr,
                poses,
                path_xyz,
                width=width,
                height=height,
                fov_deg=fov_deg,
            )

        # --- Видео ---
        write_video(frames_bgr, out_mp4, fps=fps)
        logging.info("[path_runner] Video written: %s", out_mp4)

        # --- Детекция (опционально) ---
        if det_enabled:
            det_model = det_cfg.get("model", "yolov8n.pt")
            det_conf = float(det_cfg.get("conf", 0.25))
            dets = run_detection(frames_bgr, det_model, det_conf)
            det_json = outdir / "detections_yolo.json"
            det_json.write_text(
                json.dumps({"detections": dets}, indent=2),
                encoding="utf-8",
            )
            logging.info("[path_runner] YOLO detections JSON written: %s", det_json)

    # --- Meta JSON ---
    meta = {
        "scene": str(scene_path),
        "output_dir": str(outdir),
        "resolution": [width, height],
        "fps": fps,
        "seconds": seconds,
        "fov_deg": fov_deg,
        "path_length_m": path_length,
        "path_mode": mode,
        "grid": {
            "origin_xy": grid.origin_xy.tolist(),  # [min_x, min_z]
            "cell_size": float(grid.cell_size),
            "nx": int(grid.nx),
            "ny": int(grid.ny),
        },
        "planner": {
            "cell_size": cell_size,
            "margin_cells": margin_cells,
            "height_fraction": height_fraction,
            "cam_height_rel": cam_height_rel,
            "min_headroom": min_headroom,
            "frames_per_meter": frames_per_meter,
            "smooth_window": smooth_window,
            "back_offset": back_offset,
            "snap_radius": snap_radius,
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
    }
    meta_json = outdir / "path_runner_meta.json"
    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logging.info("[path_runner] Meta JSON written: %s", meta_json)
