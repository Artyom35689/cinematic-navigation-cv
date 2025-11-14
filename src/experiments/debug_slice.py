#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
height_sweep.py

Визуальный высотный анализ:

  - грузим unpacked 3DGS PLY
  - считаем диапазон высот вдоль выбранной оси (vertical-axis)
  - выбираем N уровней по высоте (между min_fraction и max_fraction)
  - на каждом уровне ставим камеру:
        * по двум горизонтальным координатам — в центр сцены
        * по вертикальной координате — соответствующая высота
        * смотрим в центр сцены
  - рендерим один кадр через gsplat на каждом уровне
  - сохраняем PNG + JSON

Пример для ConferenceHall (Z = высота):

  python3 -m src.height_sweep \\
    --scene scenes/ConferenceHall.ply \\
    --outdir output/ConferenceHall_height_sweep \\
    --res 1280x720 \\
    --fov 70 \\
    --num-views 10 \\
    --min-fraction 0.05 \\
    --max-fraction 0.95 \\
    --vertical-axis 2 \\
    --max-splats 2000000
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
import cv2

from .scene_grid import load_3dgs_unpacked_ply
from .gsplat_scene import GaussiansNP, build_intrinsics
from .render_utils import render_one_frame


# ---------------------------------------------------------------------
# Камера на заданной высоте вдоль произвольной оси
# ---------------------------------------------------------------------

def _unit_axis(axis: int) -> np.ndarray:
    """Единичный вектор вдоль оси 0/1/2."""
    e = np.zeros(3, dtype=np.float32)
    e[axis] = 1.0
    return e


def build_view_at_height_axis(
    means: np.ndarray,
    h_val: float,
    vertical_axis: int,
    fov_deg: float,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Строим простую камеру:

      - вертикальная ось = vertical_axis (0,1,2)
      - берём bbox и центр сцены
      - eye: смещаемся вдоль некоторой горизонтальной оси,
             вертикальную компоненту (= по vertical_axis) задаём как h_val
      - up = единичный вдоль vertical_axis
      - смотрим в центр сцены
    """
    means = np.asarray(means, dtype=np.float32)
    bbox_min = means.min(axis=0)  # (3,)
    bbox_max = means.max(axis=0)  # (3,)
    center = 0.5 * (bbox_min + bbox_max)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    if diag < 1e-6:
        diag = 1.0

    # вертикальный вектор
    up = _unit_axis(vertical_axis)

    # выберем горизонтальное направление "сзади сцены"
    # горизонтальная плоскость = все оси кроме vertical_axis
    if vertical_axis == 2:
        # Z = высота → горизонт XY, смотрим с отрицательного Y
        horiz_dir = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    elif vertical_axis == 1:
        # Y = высота → горизонт XZ, смотрим с отрицательного Z
        horiz_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    else:
        # X = высота → горизонт YZ, смотрим с отрицательного Z
        horiz_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    # гарантируем, что горизонтальное направление действительно в горизонтальной плоскости
    horiz_dir[vertical_axis] = 0.0
    n = np.linalg.norm(horiz_dir)
    if n < 1e-6:
        horiz_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        horiz_dir[vertical_axis] = 0.0
        horiz_dir /= (np.linalg.norm(horiz_dir) + 1e-8)
    else:
        horiz_dir /= n

    # базовая позиция камеры: "сзади" сцены по горизонтали
    eye = center + (-0.7 * diag) * horiz_dir
    # задаём высоту вдоль vertical_axis
    eye[vertical_axis] = float(h_val)

    # look-at: +Z_cam вперёд, +X_cam вправо, +Y_cam вверх
    forward = center - eye
    fn = np.linalg.norm(forward)
    if fn < 1e-8:
        forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    else:
        forward = forward / fn

    up_n = up / (np.linalg.norm(up) + 1e-8)

    right = np.cross(forward, up_n)
    rn = np.linalg.norm(right)
    if rn < 1e-8:
        # на всякий случай подкорректируем up
        up_n = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        up_n[vertical_axis] = 1.0
        up_n /= (np.linalg.norm(up_n) + 1e-8)
        right = np.cross(forward, up_n)
        right /= (np.linalg.norm(right) + 1e-8)
    else:
        right /= rn

    new_up = np.cross(right, forward)

    R = np.stack([right, new_up, forward], axis=0).astype(np.float32)  # (3,3)
    t = -R @ eye  # (3,)

    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = R
    view[:3, 3] = t

    K = build_intrinsics(fov_deg, width, height)  # (3,3)
    return view, K


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Render N snapshots from one horizontal position at different "
            "heights along chosen vertical axis for unpacked 3DGS PLY."
        )
    )
    p.add_argument("--scene", type=str, required=True, help="Path to unpacked 3DGS PLY")
    p.add_argument("--outdir", type=str, required=True, help="Output directory")

    p.add_argument(
        "--res",
        type=str,
        default="1280x720",
        help="Resolution WxH, e.g. 1280x720.",
    )
    p.add_argument(
        "--fov",
        type=float,
        default=70.0,
        help="Vertical field of view (degrees).",
    )
    p.add_argument(
        "--num-views",
        type=int,
        default=10,
        help="Number of heights / snapshots to render.",
    )
    p.add_argument(
        "--min-fraction",
        type=float,
        default=0.05,
        help="Lower fraction of [H_min, H_max] range for camera heights.",
    )
    p.add_argument(
        "--max-fraction",
        type=float,
        default=0.95,
        help="Upper fraction of [H_min, H_max] range for camera heights.",
    )
    p.add_argument(
        "--vertical-axis",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="Index of vertical axis in scene coordinates (0=X, 1=Y, 2=Z).",
    )
    p.add_argument(
        "--max-splats",
        type=int,
        default=2_000_000,
        help="Max number of Gaussians to keep (random downsampling).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for gsplat rendering.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------
# Основной пайплайн
# ---------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    args = parse_args()

    scene_path = Path(args.scene)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    logging.info("[INFO] Scene: %s", scene_path)
    logging.info("[INFO] Outdir: %s", outdir)
    logging.info("[INFO] vertical_axis=%d", args.vertical_axis)

    # Разрешение
    try:
        w_str, h_str = args.res.lower().split("x")
        width, height = int(w_str), int(h_str)
    except Exception as e:
        raise SystemExit(f"Invalid --res '{args.res}', expected WxH, error: {e}")

    # Загружаем распакованный 3DGS PLY
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

    # Диапазон высот вдоль выбранной оси (обрезаем выбросы по квантилям)
    axis = int(args.vertical_axis)
    h_vals = means[:, axis]
    h_lo = float(np.quantile(h_vals, 0.01))
    h_hi = float(np.quantile(h_vals, 0.99))
    if h_hi <= h_lo + 1e-6:
        h_lo = float(h_vals.min())
        h_hi = float(h_vals.max())

    logging.info(
        "[INFO] H range along axis %d (1–99%% quantile): [%.3f, %.3f]",
        axis, h_lo, h_hi,
    )

    # Выбираем N высот в этом диапазоне
    if args.num_views <= 0:
        raise SystemExit("--num-views must be > 0")

    min_f = float(args.min_fraction)
    max_f = float(args.max_fraction)
    if max_f <= min_f:
        raise SystemExit("--max-fraction must be > min-fraction")

    fracs = np.linspace(min_f, max_f, args.num_views, dtype=np.float32)
    heights = h_lo + fracs * (h_hi - h_lo)  # (N,)

    logging.info(
        "[INFO] Heights fractions: min=%.3f max=%.3f num=%d",
        min_f,
        max_f,
        args.num_views,
    )

    # Выбор устройства
    if args.device == "cuda":
        if not torch.cuda.is_available():
            logging.warning("[WARN] CUDA requested but not available, using CPU")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logging.info("[INFO] Using device: %s", device)

    snapshots_info: List[dict] = []

    for idx, (f, h_val) in enumerate(zip(fracs, heights)):
        logging.info(
            "[INFO] View %2d/%2d: fraction=%.3f -> h=%.3f (axis=%d)",
            idx + 1,
            args.num_views,
            float(f),
            float(h_val),
            axis,
        )

        view, K = build_view_at_height_axis(
            gauss.means,
            h_val=float(h_val),
            vertical_axis=axis,
            fov_deg=args.fov,
            width=width,
            height=height,
        )

        frame_bgr = render_one_frame(
            gauss,
            view=view,
            K=K,
            width=width,
            height=height,
            device=device,
        )

        out_png = outdir / f"view_{idx:02d}_h{h_val:+.2f}_axis{axis}.png"
        cv2.imwrite(str(out_png), frame_bgr)
        logging.info("[INFO] Saved PNG: %s", out_png)

        snapshots_info.append(
            {
                "index": idx,
                "fraction": float(f),
                "h_value": float(h_val),
                "vertical_axis": axis,
                "png": str(out_png),
            }
        )

    # JSON с информацией по кадрам
    meta = {
        "scene": str(scene_path),
        "resolution": [width, height],
        "fov_deg": float(args.fov),
        "vertical_axis": axis,
        "h_range_quantiles": {
            "q01": h_lo,
            "q99": h_hi,
        },
        "min_fraction": float(min_f),
        "max_fraction": float(max_f),
        "num_views": int(args.num_views),
        "snapshots": snapshots_info,
    }
    meta_path = outdir / "height_sweep_views.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logging.info("[INFO] Meta JSON written: %s", meta_path)


if __name__ == "__main__":
    main()
