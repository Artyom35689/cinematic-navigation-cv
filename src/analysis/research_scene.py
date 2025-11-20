#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scene research utilities.

Запуск (из корня проекта, где src — пакет):

  python3 -m src.analysis.research_scene \
      --scene scenes/ConferenceHall.ply \
      --outdir output/scene_research \
      --grid-res 256 \
      --slice-thickness-frac 0.15 \
      --num-slices 10

Требуется:
  - существующий src/gsplat_scene.py с load_gaussians_from_ply;
  - matplotlib (pip install matplotlib).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

import numpy as np

from ..gsplat_scene import load_gaussians_from_ply  # используем уже рабочий код

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Quaternion helper (WXYZ, как в 3DGS PLY / SPZ спецификации)
# ---------------------------------------------------------------------


def quat_forward_axis_wxyz(quats: np.ndarray) -> np.ndarray:
    """
    Для каждого кватерниона (w,x,y,z) возвращает направление локальной оси Z
    в мировых координатах.

    По документации 3DGS PLY / SPZ:
      rot_0, rot_1, rot_2, rot_3 = W, X, Y, Z (нормализованный кватернион):contentReference[oaicite:1]{index=1}

    Возвращает:
      dirs: [N, 3] float32, единичные векторы.
    """
    q = np.asarray(quats, dtype=np.float32)
    if q.shape[1] != 4:
        raise ValueError(f"Expected quats shape (N,4), got {q.shape}")

    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    # Третья колонка матрицы вращения (R @ [0,0,1]^T)
    # R по стандартной формуле для WXYZ:
    # R[0,2] = 2(xz + wy)
    # R[1,2] = 2(yz - wx)
    # R[2,2] = 1 - 2(x^2 + y^2)
    dir_x = 2.0 * (x * z + w * y)
    dir_y = 2.0 * (y * z - w * x)
    dir_z = 1.0 - 2.0 * (x * x + y * y)

    dirs = np.stack([dir_x, dir_y, dir_z], axis=1)
    # нормализация для численной устойчивости
    norm = np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8
    dirs = dirs / norm
    return dirs.astype(np.float32)


def forward_angles_y_up(dirs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Для направлений (единичные векторы) считает:
      - azimuth_deg: угол в плоскости XZ относительно +Z (yaw), [-180, 180]
      - elevation_deg: угол относительно плоскости XZ (наклон), [-90, 90]

    dirs: [N,3], предполагается Y — вверх.
    """
    dirs = np.asarray(dirs, dtype=np.float32)
    dx = dirs[:, 0]
    dy = dirs[:, 1]
    dz = dirs[:, 2]

    # азимут: atan2(X, Z), чтобы yaw=0 смотрел по +Z
    azimuth = np.arctan2(dx, dz)

    # наклон: относительно XZ плоскости
    horiz_len = np.sqrt(dx * dx + dz * dz) + 1e-8
    elevation = np.arctan2(dy, horiz_len)

    azimuth_deg = np.rad2deg(azimuth)
    elevation_deg = np.rad2deg(elevation)
    return azimuth_deg, elevation_deg


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------


def _ensure_matplotlib() -> None:
    if plt is None:
        raise RuntimeError(
            "matplotlib is not available. Install it with 'pip install matplotlib'."
        )


def save_histogram(
    data: np.ndarray,
    out_path: Path,
    title: str,
    xlabel: str,
    bins: int = 128,
) -> None:
    _ensure_matplotlib()
    data = np.asarray(data, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(data, bins=bins, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("[PLOT] Saved histogram: %s", out_path)


def save_density_xz(
    means: np.ndarray,
    out_path: Path,
    grid_res: int,
    title: str,
) -> None:
    """
    Глобальная карта плотности проекции на XZ (вид сверху).
    """
    _ensure_matplotlib()
    xyz = np.asarray(means, dtype=np.float32)
    x = xyz[:, 0]
    z = xyz[:, 2]

    x_min, x_max = float(x.min()), float(x.max())
    z_min, z_max = float(z.min()), float(z.max())

    H, _, _ = np.histogram2d(
        x,
        z,
        bins=grid_res,
        range=[[x_min, x_max], [z_min, z_max]],
    )  # [grid_res, grid_res]

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(
        H.T,
        origin="lower",
        extent=[x_min, x_max, z_min, z_max],
        aspect="equal",
    )
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    fig.colorbar(im, ax=ax, label="point density")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("[PLOT] Saved XZ density map: %s", out_path)


def save_y_slices_xz(
    means: np.ndarray,
    outdir: Path,
    grid_res: int,
    thickness_frac: float,
    num_slices: int,
) -> None:
    """
    Серия горизонтальных срезов по Y в XZ-плоскости.

    Диапазон по Y задаётся относительно размеров сцены:
      - сцена по Y: [y_min, y_max]
      - высота сцены: H = y_max - y_min

    Толщина слоя задаётся как доля высоты сцены:
      thickness = thickness_frac * H

    Количество слоёв задаётся явно (num_slices).
    Положение слоёв равномерно распределяется от y_min до y_max так, чтобы:
      - если num_slices == 1: центр одного слоя в середине,
      - если num_slices > 1: первый слой начинается у y_min,
        последний не выходит за y_max.

    Args:
        means: (N,3) точки гауссиан.
        outdir: куда сохранять PNG.
        grid_res: разрешение карты плотности (grid_res x grid_res).
        thickness_frac: толщина слоя в долях высоты сцены (0..1).
        num_slices: количество слоёв (>=1).
    """
    _ensure_matplotlib()
    xyz = np.asarray(means, dtype=np.float32)
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    y_min, y_max = float(y.min()), float(y.max())
    x_min, x_max = float(x.min()), float(x.max())
    z_min, z_max = float(z.min()), float(z.max())

    height = y_max - y_min
    if height <= 0:
        logger.warning("[SLICES] Height along Y is non-positive, skipping slices.")
        return

    if num_slices <= 0:
        logger.warning("[SLICES] num_slices <= 0, skipping slices.")
        return

    # Толщина слоя как доля высоты сцены
    thickness = thickness_frac * height
    if thickness <= 0:
        logger.warning("[SLICES] Non-positive slice thickness, skipping slices.")
        return
    if thickness > height:
        logger.warning(
            "[SLICES] thickness_frac=%.3f -> thickness=%.3f > scene height=%.3f; "
            "clamping to full height.",
            thickness_frac,
            thickness,
            height,
        )
        thickness = height

    logger.info(
        "[SLICES] Y range [%.3f, %.3f], height=%.3f, thickness=%.3f, num_slices=%d",
        y_min,
        y_max,
        height,
        thickness,
        num_slices,
    )

    # Расчёт стартов слоёв:
    # - num_slices == 1: один слой по центру
    # - num_slices > 1: первый слой начинается у y_min, последний не выходит за y_max
    starts = []
    if num_slices == 1:
        center_y = 0.5 * (y_min + y_max)
        y0 = center_y - 0.5 * thickness
        y1 = center_y + 0.5 * thickness
        y0 = max(y_min, y0)
        y1 = min(y_max, y1)
        starts.append((y0, y1))
    else:
        # Общий доступный интервал для "центров" слоёв:
        # y_min .. (y_max - thickness)
        max_start = y_max - thickness
        if max_start <= y_min:
            # Тогда все слои фактически перекроют весь диапазон Y
            y0 = y_min
            y1 = y_max
            starts = [(y0, y1)] * num_slices
        else:
            step = (max_start - y_min) / float(num_slices - 1)
            for i in range(num_slices):
                y0 = y_min + i * step
                y1 = y0 + thickness
                starts.append((y0, y1))

    for slice_idx, (y0, y1) in enumerate(starts):
        mask = (y >= y0) & (y < y1)
        pts = xyz[mask]
        if pts.shape[0] == 0:
            logger.info(
                "[SLICES] slice %d: Y [%.3f, %.3f] -> 0 points, skipping.",
                slice_idx,
                y0,
                y1,
            )
            continue

        H, _, _ = np.histogram2d(
            pts[:, 0],
            pts[:, 2],
            bins=grid_res,
            range=[[x_min, x_max], [z_min, z_max]],
        )

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(
            H.T,
            origin="lower",
            extent=[x_min, x_max, z_min, z_max],
            aspect="equal",
        )
        ax.set_title(f"Y slice [{y0:.2f}, {y1:.2f}]")
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        fig.colorbar(im, ax=ax, label="point density")
        fig.tight_layout()

        out_path = outdir / f"slice_y_{slice_idx:03d}_{y0:.2f}_{y1:.2f}.png"
        fig.savefig(out_path)
        plt.close(fig)
        logger.info(
            "[PLOT] Saved Y-slice %d: [%.3f, %.3f], pts=%d -> %s",
            slice_idx,
            y0,
            y1,
            pts.shape[0],
            out_path,
        )



# ---------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Research / analysis of 3DGS scene (.ply): "
                    "densities, slices, orientation stats.",
    )
    ap.add_argument(
        "--scene",
        required=True,
        help="Path to unpacked 3DGS .ply scene.",
    )
    ap.add_argument(
        "--outdir",
        required=True,
        help="Output directory for plots and stats.",
    )
    ap.add_argument(
        "--max-splats",
        type=int,
        default=2_000_000,
        help="Max number of Gaussians to keep (random downsampling if exceeded).",
    )
    ap.add_argument(
        "--grid-res",
        type=int,
        default=256,
        help="Resolution of 2D grids (density maps, slices).",
    )
    ap.add_argument(
        "--slice-thickness-frac",
        type=float,
        default=0.1,
        help=(
            "Slice thickness as fraction of scene height along Y "
            "(e.g. 0.1 = 1/10 of scene height)."
        ),
    )
    ap.add_argument(
        "--num-slices",
        type=int,
        default=5,
        help="Number of Y-slices to generate.",
    )

    ap.add_argument(
        "--angle-bins",
        type=int,
        default=180,
        help="Number of bins for orientation histograms.",
    )
    return ap.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if plt is None:
        logger.error("matplotlib is not installed; cannot generate plots.")
        raise SystemExit(1)

    logger.info("[MAIN] Scene: %s", args.scene)
    logger.info("[MAIN] Outdir: %s", outdir)

    # 1) Загрузка сцены
    gauss = load_gaussians_from_ply(args.scene, max_points=args.max_splats)
    xyz = gauss.means

    bbox_min = xyz.min(axis=0)
    bbox_max = xyz.max(axis=0)
    diag = float(np.linalg.norm(bbox_max - bbox_min)) + 1e-6

    logger.info(
        "[SCENE] N=%d, bbox_min=%s, bbox_max=%s, diag=%.3f",
        xyz.shape[0],
        np.round(bbox_min, 3),
        np.round(bbox_max, 3),
        diag,
    )

    # 2) 1D гистограммы по координатам
    save_histogram(
        xyz[:, 0],
        outdir / "hist_x.png",
        title="X distribution",
        xlabel="X",
    )
    save_histogram(
        xyz[:, 1],
        outdir / "hist_y.png",
        title="Y distribution",
        xlabel="Y",
    )
    save_histogram(
        xyz[:, 2],
        outdir / "hist_z.png",
        title="Z distribution",
        xlabel="Z",
    )

    # 3) Глобальная XZ карта плотности (вид сверху)
    save_density_xz(
        xyz,
        outdir / "density_xz.png",
        grid_res=args.grid_res,
        title="XZ density (top-down view)",
    )

    # 4) Y-срезы в XZ (поиск пола/потолка/слоёв)
    slices_dir = outdir / "slices_y_xz"
    slices_dir.mkdir(parents=True, exist_ok=True)
    save_y_slices_xz(
        xyz,
        slices_dir,
        grid_res=args.grid_res,
        thickness_frac=args.slice_thickness_frac,
        num_slices=args.num_slices,
    )


    # 5) Ориентации (кватернионы -> направление локальной оси Z -> углы)
    dirs = quat_forward_axis_wxyz(gauss.quats)
    azimuth_deg, elevation_deg = forward_angles_y_up(dirs)

    save_histogram(
        azimuth_deg,
        outdir / "orient_azimuth_deg.png",
        title="Forward azimuth (deg, Y-up, XZ plane)",
        xlabel="azimuth (deg, -180..180)",
        bins=args.angle_bins,
    )
    save_histogram(
        elevation_deg,
        outdir / "orient_elevation_deg.png",
        title="Forward elevation (deg, vs XZ plane)",
        xlabel="elevation (deg, -90..90)",
        bins=args.angle_bins // 2,
    )

    logger.info("[MAIN] Analysis finished. Outputs in: %s", outdir)


if __name__ == "__main__":
    main()


def save_camera_path_on_density_xz(
    means: np.ndarray,
    poses: List[Dict[str, Any]],
    out_path: Path,
    grid_res: int = 256,
    arrow_stride: int = 10,
) -> None:
    """
    Рисует маршрут камеры и направление взгляда на XZ-карте плотности сцены.

    Допущения:
      - Верхняя ось мира: Y.
      - Камера ходит по XZ, eye = [x, y, z].
      - Плотность строится по проекции всех гауссиан на XZ.

    Args:
        means: (N,3) координаты гауссиан.
        poses: список словарей с ключами "eye" и "center"
               (как в твоём camera_path.json / generate_camera_poses_*).
        out_path: путь до PNG, куда сохранить картинку.
        grid_res: разрешение карты плотности (grid_res x grid_res).
        arrow_stride: как часто рисовать стрелки направления (каждый k-й кадр).
    """
    _ensure_matplotlib()

    xyz = np.asarray(means, dtype=np.float32)
    x = xyz[:, 0]
    z = xyz[:, 2]

    x_min, x_max = float(x.min()), float(x.max())
    z_min, z_max = float(z.min()), float(z.max())

    # --- карта плотности по XZ ---
    H, _, _ = np.histogram2d(
        x,
        z,
        bins=grid_res,
        range=[[x_min, x_max], [z_min, z_max]],
    )  # [grid_res, grid_res]

    # --- траектория камеры ---
    if not poses:
        logger.warning("[CAM-PATH] No poses provided, skipping camera path plot.")
        return

    eyes = np.asarray([p["eye"] for p in poses], dtype=np.float32)  # [T,3]
    centers = np.asarray([p["center"] for p in poses], dtype=np.float32)  # [T,3]

    cam_x = eyes[:, 0]
    cam_z = eyes[:, 2]

    # направления взгляда в XZ
    dir_x = centers[:, 0] - eyes[:, 0]
    dir_z = centers[:, 2] - eyes[:, 2]
    dirs = np.stack([dir_x, dir_z], axis=1)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8
    dirs_norm = dirs / norms  # единичные векторы в XZ

    # масштаб стрелок относительно размера сцены
    x_span = max(1e-6, x_max - x_min)
    z_span = max(1e-6, z_max - z_min)
    arrow_len = 0.05 * max(x_span, z_span)  # 5% размера сцены

    logger.info(
        "[CAM-PATH] Plotting %d poses on XZ density (grid_res=%d), arrow_stride=%d",
        len(poses),
        grid_res,
        arrow_stride,
    )

    # --- рисуем ---
    fig, ax = plt.subplots(figsize=(7, 7))

    # плотность
    im = ax.imshow(
        H.T,
        origin="lower",
        extent=[x_min, x_max, z_min, z_max],
        aspect="equal",
        cmap="gray",
    )
    fig.colorbar(im, ax=ax, label="point density")

    # маршрут
    ax.plot(cam_x, cam_z, "r-", linewidth=2.0, label="camera path")

    # рисуем стрелки через stride, чтобы не захламлять
    idxs = np.arange(0, len(poses), max(1, arrow_stride))
    ax.quiver(
        cam_x[idxs],
        cam_z[idxs],
        dirs_norm[idxs, 0] * arrow_len,
        dirs_norm[idxs, 1] * arrow_len,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="cyan",
        width=0.003,
        label="view direction (sampled)",
    )

    # отметим старт/финиш
    ax.scatter(cam_x[0], cam_z[0], c="green", s=60, marker="o", label="start")
    ax.scatter(cam_x[-1], cam_z[-1], c="magenta", s=60, marker="x", label="end")

    ax.set_title("Camera path & view direction on XZ density")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    logger.info("[CAM-PATH] Saved camera path plot: %s", out_path)