# src/scene_grid.py

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from plyfile import PlyData

try:
    import cv2  # only for debug images
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)

# SH DC constant (как в 3DGS)
SH_C0 = 0.28209479177387814


# ---------------------------------------------------------------------
# 1) Загрузка unpacked 3DGS PLY -> NumPy массивы
# ---------------------------------------------------------------------
def load_3dgs_unpacked_ply(
    path: str, max_points: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load full 3DGS PLY (unpacked format):

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

    Returns:
      means:     (N,3) float32
      quats:     (N,4) float32
      scales:    (N,3) float32  (уже exp, если были лог-stddev)
      opacities: (N,) float32   (в [0,1])
      colors:    (N,3) float32  (RGB в [0,1])
    """
    logger.info("[scene_grid] load_3dgs_unpacked_ply: %s", path)
    ply = PlyData.read(path)

    elem_names = [e.name for e in ply.elements]
    logger.debug("[scene_grid] elements: %s", elem_names)

    try:
        v = ply["vertex"]
    except KeyError:
        raise RuntimeError(f"PLY has no 'vertex' element; available elements: {elem_names}")

    names = v.data.dtype.names
    logger.debug("[scene_grid] vertex fields: %s", names)

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

    # positions
    x = np.asarray(v["x"], np.float32)
    y = np.asarray(v["y"], np.float32)
    z = np.asarray(v["z"], np.float32)
    means = np.stack([x, y, z], axis=1)  # (N,3)

    # raw scales & opacities
    s0 = np.asarray(v["scale_0"], np.float32)
    s1 = np.asarray(v["scale_1"], np.float32)
    s2 = np.asarray(v["scale_2"], np.float32)
    scales_raw = np.stack([s0, s1, s2], axis=1)

    op_raw = np.asarray(v["opacity"], np.float32)

    scale_min, scale_max = float(scales_raw.min()), float(scales_raw.max())
    frac_neg_scale = float((scales_raw < 0.0).mean())
    op_min, op_max = float(op_raw.min()), float(op_raw.max())

    logger.info(
        "[scene_grid] scales raw min/max=%.4f/%.4f, neg_frac=%.3f",
        scale_min, scale_max, frac_neg_scale
    )
    logger.info(
        "[scene_grid] opacity raw min/max=%.4f/%.4f",
        op_min, op_max
    )

    # scales: log-stddev -> exp, if many negatives
    if frac_neg_scale > 0.1:
        scales = np.exp(scales_raw)
        logger.info("[scene_grid] treating scales as log-stddev, applying exp()")
    else:
        scales = scales_raw
        logger.info("[scene_grid] treating scales as already linear")

    # opacities: logits -> sigmoid if outside [0,1]
    if op_min < 0.0 or op_max > 1.0:
        opacities = 1.0 / (1.0 + np.exp(-op_raw))
        logger.info("[scene_grid] treating opacities as logits, applying sigmoid()")
    else:
        opacities = op_raw
        logger.info("[scene_grid] treating opacities as already in [0,1]")

    opacities = opacities.astype(np.float32)

    # quaternions
    q0 = np.asarray(v["rot_0"], np.float32)
    q1 = np.asarray(v["rot_1"], np.float32)
    q2 = np.asarray(v["rot_2"], np.float32)
    q3 = np.asarray(v["rot_3"], np.float32)
    quats = np.stack([q0, q1, q2, q3], axis=1)  # (N,4)

    norm = np.linalg.norm(quats, axis=1, keepdims=True)
    norm[norm == 0.0] = 1.0
    quats = quats / norm

    # colors from DC SH
    fdc0 = np.asarray(v["f_dc_0"], np.float32)
    fdc1 = np.asarray(v["f_dc_1"], np.float32)
    fdc2 = np.asarray(v["f_dc_2"], np.float32)
    f_dc = np.stack([fdc0, fdc1, fdc2], axis=1)  # (N,3)

    colors = 0.5 + SH_C0 * f_dc
    colors = np.clip(colors, 0.0, 1.0).astype(np.float32)

    N = means.shape[0]
    logger.info("[scene_grid] loaded %d splats", N)

    if max_points is not None and N > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(N, size=max_points, replace=False)
        idx.sort()
        means = means[idx]
        quats = quats[idx]
        scales = scales[idx]
        opacities = opacities[idx]
        colors = colors[idx]
        logger.info("[scene_grid] downsampled %d -> %d", N, max_points)

    return means, quats, scales, opacities, colors


# ---------------------------------------------------------------------
# 2) 2.5D grid (XZ-плоскость + высота по Y)
# ---------------------------------------------------------------------
@dataclass
class SceneGrid:
    # origin_xy теперь на самом деле origin_xz: [min_x, min_z]
    origin_xy: np.ndarray   # (2,) [min_x, min_z]
    cell_size: float
    nx: int
    ny: int
    count: np.ndarray       # (ny, nx) int32, how many points in cell
    # floor_z / ceil_z на самом деле floor_y / ceil_y (высоты по оси Y)
    floor_z: np.ndarray     # (ny, nx) float32, NaN if empty
    ceil_z: np.ndarray      # (ny, nx) float32, NaN if empty

    def cell_to_world(self, ix: int, iy: int, z_offset: float = 0.0) -> np.ndarray:
        """
        Центр ячейки (ix, iy) в мировых координатах при Y-up.

        Плоскость сетки: (X,Z), высота: Y.
        origin_xy = [min_x, min_z]
        floor_z[iy, ix] = floor_y в этой ячейке.

        z_offset трактуем как cam_height_rel по оси Y.
        """
        x = self.origin_xy[0] + (ix + 0.5) * self.cell_size   # X
        z = self.origin_xy[1] + (iy + 0.5) * self.cell_size   # Z

        fy = self.floor_z[iy, ix]  # "floor_y" в этой ячейке
        if np.isnan(fy):
            y = 0.0
        else:
            y = fy + z_offset

        # Мировая система: (x, y, z) с Y-up
        return np.array([x, y, z], dtype=np.float32)

    def to_json_meta(self) -> dict:
        return {
            "origin_xy": self.origin_xy.tolist(),  # на самом деле (x,z)
            "cell_size": float(self.cell_size),
            "nx": int(self.nx),
            "ny": int(self.ny),
        }


def build_height_grid(
    means: np.ndarray,
    cell_size: float = 0.5,
    min_points_per_cell: int = 50,
    floor_percentile: float = 5.0,
    ceil_percentile: float = 95.0,
) -> SceneGrid:
    """
    Build 2.5D grid при Y-up:

    - плоскость сетки: (X,Z),
    - высота берётся по оси Y.

    Для каждой ячейки (ix,iy) собираем y-значения точек, попавших в неё, и считаем:

      floor_y = percentile(y, floor_percentile)
      ceil_y  = percentile(y, ceil_percentile)

    Эти значения храним в массивах floor_z / ceil_z, чтобы не ломать остальной код.
    """
    assert means.shape[1] == 3, "means must be (N,3)"

    # Плоскость навигации = (X,Z), высота = Y
    x = means[:, 0]
    y = means[:, 1]  # высота
    z = means[:, 2]

    plane = np.stack([x, z], axis=1)  # (N,2) => (X,Z)

    mins = plane.min(axis=0)  # [min_x, min_z]
    maxs = plane.max(axis=0)
    size = maxs - mins

    nx = int(np.ceil(size[0] / cell_size))
    ny = int(np.ceil(size[1] / cell_size))

    logger.info(
        "[scene_grid] build_height_grid (Y-up): cell_size=%.3f, nx=%d, ny=%d",
        cell_size, nx, ny
    )

    # Lists of y-values per cell (но храним как floor_z/ceil_z)
    y_lists = [[[] for _ in range(nx)] for _ in range(ny)]

    xs = plane[:, 0]  # x
    zs = plane[:, 1]  # z

    ix_all = ((xs - mins[0]) / cell_size).astype(np.int32)
    iy_all = ((zs - mins[1]) / cell_size).astype(np.int32)

    valid_mask = (
        (ix_all >= 0) & (ix_all < nx) &
        (iy_all >= 0) & (iy_all < ny)
    )

    ix_all = ix_all[valid_mask]
    iy_all = iy_all[valid_mask]
    y_valid = y[valid_mask]

    for ix, iy, yy in zip(ix_all, iy_all, y_valid):
        y_lists[iy][ix].append(float(yy))

    floor_z = np.full((ny, nx), np.nan, dtype=np.float32)
    ceil_z = np.full((ny, nx), np.nan, dtype=np.float32)
    count = np.zeros((ny, nx), dtype=np.int32)

    for iy in range(ny):
        for ix in range(nx):
            ys = y_lists[iy][ix]
            if len(ys) < min_points_per_cell:
                continue
            arr = np.asarray(ys, np.float32)
            count[iy, ix] = arr.size
            floor_z[iy, ix] = np.percentile(arr, floor_percentile)  # floor_y
            ceil_z[iy, ix] = np.percentile(arr, ceil_percentile)    # ceil_y

    origin_xy = mins.astype(np.float32)  # [min_x, min_z]
    return SceneGrid(
        origin_xy=origin_xy,
        cell_size=float(cell_size),
        nx=nx,
        ny=ny,
        count=count,
        floor_z=floor_z,  # на самом деле floor_y
        ceil_z=ceil_z,    # на самом деле ceil_y
    )


def compute_free_mask(
    grid: SceneGrid,
    cam_height_rel: float = 1.7,
    min_headroom: float = 0.8,
) -> np.ndarray:
    """
    Compute which cells are free для камеры при Y-up:

        y_cam = floor_y + cam_height_rel

    Ячейка свободна, если:
      - floor_y конечен
      - ceil_y конечен
      - ceil_y - floor_y >= cam_height_rel + min_headroom
    """
    floor_y = grid.floor_z   # так названы поля
    ceil_y = grid.ceil_z

    valid = ~np.isnan(floor_y) & ~np.isnan(ceil_y)
    headroom = ceil_y - floor_y

    free = valid & (headroom >= (cam_height_rel + min_headroom))

    logger.info(
        "[scene_grid] free cells: %d / %d",
        int(free.sum()),
        int(grid.nx * grid.ny),
    )
    return free


def save_debug_images(
    grid: SceneGrid,
    free_mask: np.ndarray,
    outdir: Path,
    prefix: str = "grid",
) -> None:
    """
    Save simple top-down debug images при Y-up:
      - floor_z (на самом деле floor_y) heatmap в XZ-проекции
      - free_mask visualization
    """
    outdir.mkdir(parents=True, exist_ok=True)

    if cv2 is None:
        logger.warning("[scene_grid] cv2 not available, skip debug images")
        return

    ny, nx = grid.floor_z.shape

    # floor_y normalized heatmap
    fz = grid.floor_z.copy()
    mask_valid = ~np.isnan(fz)
    if mask_valid.any():
        vmin = float(fz[mask_valid].min())
        vmax = float(fz[mask_valid].max())
        if vmax > vmin:
            fz_norm = (fz - vmin) / (vmax - vmin)
        else:
            fz_norm = np.zeros_like(fz)
        fz_norm[~mask_valid] = 0.0
    else:
        fz_norm = np.zeros_like(fz)

    img_floor = (fz_norm * 255.0).astype(np.uint8)
    img_floor = cv2.applyColorMap(img_floor, cv2.COLORMAP_PLASMA)

    # free mask
    img_free = np.zeros((ny, nx, 3), dtype=np.uint8)
    img_free[:, :] = (20, 20, 20)  # background
    occupied = (~np.isnan(grid.floor_z)) & (~free_mask)
    img_free[occupied] = (0, 0, 255)   # red = blocked
    img_free[free_mask] = (0, 255, 0)  # green = free

    # upscale a bit for readability
    scale = 4
    img_floor_big = cv2.resize(img_floor, (nx * scale, ny * scale), interpolation=cv2.INTER_NEAREST)
    img_free_big = cv2.resize(img_free, (nx * scale, ny * scale), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(str(outdir / f"{prefix}_floor_z.png"), img_floor_big)
    cv2.imwrite(str(outdir / f"{prefix}_free.png"), img_free_big)
    logger.info("[scene_grid] saved debug images: %s_*", prefix)

    # also meta json
    meta = grid.to_json_meta()
    meta["free_cells"] = int(free_mask.sum())
    meta["total_cells"] = int(grid.nx * grid.ny)
    (outdir / f"{prefix}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def save_grid_with_axes(
    grid: SceneGrid,
    free_mask: np.ndarray,
    out_png: Path,
    scale: int = 8,
    tick_step_world: float = 5.0,
) -> None:
    """
    Рисует occupancy-карту (как grid_free.png), но с подписями мировых координат.

    При Y-up:
      - плоскость картинки = (X,Z),
      - подписи по осям: x и z.
    """
    if cv2 is None:
        logger.warning("[scene_grid] cv2 not available, skip axes image")
        return

    ny, nx = free_mask.shape

    # базовая occupancy-картинка
    img = np.zeros((ny, nx, 3), dtype=np.uint8)
    img[:, :] = (20, 20, 20)

    occupied = (~np.isnan(grid.floor_z)) & (~free_mask)
    img[occupied] = (0, 0, 255)   # красный = занято
    img[free_mask] = (0, 255, 0)  # зелёный = свободно

    img_big = cv2.resize(
        img,
        (nx * scale, ny * scale),
        interpolation=cv2.INTER_NEAREST,
    )

    x0, z0 = float(grid.origin_xy[0]), float(grid.origin_xy[1])
    s = float(grid.cell_size)
    x1 = x0 + nx * s
    z1 = z0 + ny * s

    def cell_center(ix: int, iy: int) -> tuple[float, float]:
        wx = x0 + (ix + 0.5) * s
        wz = z0 + (iy + 0.5) * s
        return wx, wz

    def cell_to_pix(ix: int, iy: int) -> tuple[int, int]:
        # iy=0 вверху
        px = int((ix + 0.5) * scale)
        py = int((iy + 0.5) * scale)
        return px, py

    font = cv2.FONT_HERSHEY_SIMPLEX

    # подписи углов (координаты центров крайних клеток)
    corners = [
        (0,        0,        "TL"),  # top-left
        (nx - 1,   0,        "TR"),  # top-right
        (0,        ny - 1,   "BL"),  # bottom-left
        (nx - 1,   ny - 1,   "BR"),  # bottom-right
    ]
    for ix, iy, label in corners:
        wx, wz = cell_center(ix, iy)
        px, py = cell_to_pix(ix, iy)
        text = f"{label} (x={wx:.1f}, z={wz:.1f})"
        cv2.putText(
            img_big,
            text,
            (px + 5, py - 5),
            font,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # несколько вертикальных засечек по X
    x_min_tick = math.ceil(x0 / tick_step_world) * tick_step_world
    x_vals = np.arange(x_min_tick, x1, tick_step_world)
    for xv in x_vals:
        ix_f = (xv - x0) / s
        if ix_f < 0 or ix_f >= nx:
            continue
        px = int((ix_f + 0.5) * scale)
        cv2.line(
            img_big,
            (px, 0),
            (px, ny * scale - 1),
            (80, 80, 80),
            1,
        )
        cv2.putText(
            img_big,
            f"x={xv:.1f}",
            (px + 2, 12),
            font,
            0.4,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    # несколько горизонтальных засечек по Z
    z_min_tick = math.ceil(z0 / tick_step_world) * tick_step_world
    z_vals = np.arange(z_min_tick, z1, tick_step_world)
    for zv in z_vals:
        iy_f = (zv - z0) / s
        if iy_f < 0 or iy_f >= ny:
            continue
        py = int((iy_f + 0.5) * scale)
        cv2.line(
            img_big,
            (0, py),
            (nx * scale - 1, py),
            (80, 80, 80),
            1,
        )
        cv2.putText(
            img_big,
            f"z={zv:.1f}",
            (5, py - 3),
            font,
            0.4,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    # маленькая легенда
    cv2.putText(img_big, "+x →", (10, 20), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img_big, "+z ↓", (10, 40), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_png), img_big)
    logger.info("[scene_grid] saved axes image: %s", out_png)
