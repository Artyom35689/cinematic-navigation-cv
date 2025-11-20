#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scene research utilities.

Usage (from project root, where src is a package):

  python3 -m src.analysis.research_scene \
      --scene scenes/ConferenceHall.ply \
      --outdir output/scene_research \
      --grid-res 512 \
      --slice-thickness-frac 0.15 \
      --num-slices 10 \
      --plot-3d

Requirements:
  - existing src/gsplat_scene.py with load_gaussians_from_ply;
  - matplotlib (pip install matplotlib).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

import numpy as np

from ..gsplat_scene import load_gaussians_from_ply  # reuse existing loader

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
except ImportError:
    plt = None


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Quaternion helpers (WXYZ, as in 3DGS PLY / SPZ spec)
# ---------------------------------------------------------------------


def quat_forward_axis_wxyz(quats: np.ndarray) -> np.ndarray:
    """
    For each quaternion (w,x,y,z) returns the direction of its local Z axis
    in world coordinates.

    According to 3DGS PLY / SPZ:
      rot_0, rot_1, rot_2, rot_3 = W, X, Y, Z (normalized quaternion).

    Returns:
      dirs: [N, 3] float32, unit vectors.
    """
    q = np.asarray(quats, dtype=np.float32)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"Expected quats shape (N,4), got {q.shape}")

    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    # Third column of rotation matrix (R @ [0,0,1]^T)
    # R for WXYZ:
    # R[0,2] = 2(xz + wy)
    # R[1,2] = 2(yz - wx)
    # R[2,2] = 1 - 2(x^2 + y^2)
    dir_x = 2.0 * (x * z + w * y)
    dir_y = 2.0 * (y * z - w * x)
    dir_z = 1.0 - 2.0 * (x * x + y * y)

    dirs = np.stack([dir_x, dir_y, dir_z], axis=1)
    # Normalize for numerical stability
    norm = np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8
    dirs = dirs / norm
    return dirs.astype(np.float32)


def forward_angles_y_up(dirs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    For unit direction vectors returns:
      - azimuth_deg: angle in XZ plane relative to +Z (yaw), [-180, 180]
      - elevation_deg: angle above XZ plane (pitch), [-90, 90]

    Assumes world up axis is Y.

    Args:
        dirs: [N,3] float32, unit vectors.

    Returns:
        (azimuth_deg, elevation_deg) as float32 arrays.
    """
    dirs = np.asarray(dirs, dtype=np.float32)
    dx = dirs[:, 0]
    dy = dirs[:, 1]
    dz = dirs[:, 2]

    # Azimuth: atan2(X, Z) so that yaw=0 looks along +Z
    azimuth = np.arctan2(dx, dz)

    # Elevation: angle w.r.t. XZ plane
    horiz_len = np.sqrt(dx * dx + dz * dz) + 1e-8
    elevation = np.arctan2(dy, horiz_len)

    azimuth_deg = np.rad2deg(azimuth)
    elevation_deg = np.rad2deg(elevation)
    return azimuth_deg, elevation_deg


# ---------------------------------------------------------------------
# Plot helpers (2D)
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
    """
    Save a simple 1D histogram for a given array.
    """
    _ensure_matplotlib()
    data = np.asarray(data, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.hist(data, bins=bins, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
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
    Global density map of the scene in XZ plane (top-down view).

    Args:
        means: (N,3) Gaussian means.
        out_path: where to save PNG.
        grid_res: resolution of the 2D grid (grid_res x grid_res).
        title: plot title.
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

    fig, ax = plt.subplots(figsize=(16, 9))
    im = ax.imshow(
        H.T,
        origin="lower",
        extent=[x_min, x_max, z_min, z_max],
        aspect="equal",
        cmap="viridis",
    )
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    fig.colorbar(im, ax=ax, label="point density")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
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
    Generate a series of horizontal slices over Y, visualized in XZ plane.

    Scene bounds along Y:
      - [y_min, y_max]
      - scene height: H = y_max - y_min

    Slice thickness is specified as a fraction of the scene height:
      thickness = thickness_frac * H

    The number of slices is specified directly (num_slices).
    Slice positions are evenly distributed over [y_min, y_max], with clamping
    so that slices do not go out of the scene bounds.

    Args:
        means: (N,3) Gaussian means.
        outdir: directory where slice PNGs will be saved.
        grid_res: resolution of the 2D grid for each slice.
        thickness_frac: slice thickness as fraction of scene height (0..1).
        num_slices: number of slices (>=1).
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

    outdir.mkdir(parents=True, exist_ok=True)

    # Slice ranges:
    # - num_slices == 1: center slice.
    # - num_slices > 1: first starts at y_min, last does not exceed y_max.
    ranges: List[Tuple[float, float]] = []
    if num_slices == 1:
        center_y = 0.5 * (y_min + y_max)
        y0 = center_y - 0.5 * thickness
        y1 = center_y + 0.5 * thickness
        y0 = max(y_min, y0)
        y1 = min(y_max, y1)
        ranges.append((y0, y1))
    else:
        max_start = y_max - thickness
        if max_start <= y_min:
            # All slices effectively cover full Y
            ranges = [(y_min, y_max)] * num_slices
        else:
            step = (max_start - y_min) / float(num_slices - 1)
            for i in range(num_slices):
                y0 = y_min + i * step
                y1 = y0 + thickness
                ranges.append((y0, y1))

    for slice_idx, (y0, y1) in enumerate(ranges):
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

        fig, ax = plt.subplots(figsize=(16, 9))
        im = ax.imshow(
            H.T,
            origin="lower",
            extent=[x_min, x_max, z_min, z_max],
            aspect="equal",
            cmap="viridis",
        )
        ax.set_title(f"Y slice [{y0:.2f}, {y1:.2f}]")
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        fig.colorbar(im, ax=ax, label="point density")
        fig.tight_layout()

        out_path = outdir / f"slice_y_{slice_idx:03d}_{y0:.2f}_{y1:.2f}.png"
        fig.savefig(out_path, dpi=150)
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
# Camera path visualization on XZ density (with height)
# ---------------------------------------------------------------------


def save_camera_path_on_density_xz(
    means: np.ndarray,
    poses: List[Dict[str, Any]],
    out_path: Path,
    grid_res: int = 512,
    arrow_stride: int = 10,
) -> None:
    """
    Plot camera path and view direction on XZ density map of the scene.
    Camera height (Y) is visualized via color.

    Assumptions:
      - World up axis: Y.
      - Camera moves on XZ plane, eye = [x, y, z].
      - Density map is built from all splats projected onto XZ.

    Args:
        means: (N,3) Gaussian means.
        poses: list of dicts with keys "eye" and "center"
               (as produced by generate_camera_poses_*).
        out_path: path to PNG output.
        grid_res: resolution of the XZ density grid.
        arrow_stride: draw an arrow every k-th frame to avoid clutter.
    """
    _ensure_matplotlib()

    xyz = np.asarray(means, dtype=np.float32)
    x = xyz[:, 0]
    z = xyz[:, 2]

    x_min, x_max = float(x.min()), float(x.max())
    z_min, z_max = float(z.min()), float(z.max())

    # Scene density map in XZ
    H, _, _ = np.histogram2d(
        x,
        z,
        bins=grid_res,
        range=[[x_min, x_max], [z_min, z_max]],
    )

    if not poses:
        logger.warning("[CAM-PATH] No poses provided, skipping camera path plot.")
        return

    eyes = np.asarray([p["eye"] for p in poses], dtype=np.float32)      # [T,3]
    centers = np.asarray([p["center"] for p in poses], dtype=np.float32)  # [T,3]

    cam_x = eyes[:, 0]
    cam_y = eyes[:, 1]
    cam_z = eyes[:, 2]

    # Forward direction in XZ
    dir_x = centers[:, 0] - eyes[:, 0]
    dir_z = centers[:, 2] - eyes[:, 2]
    dirs = np.stack([dir_x, dir_z], axis=1)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8
    dirs_norm = dirs / norms  # unit vectors in XZ

    # Arrow length relative to scene span
    x_span = max(1e-6, x_max - x_min)
    z_span = max(1e-6, z_max - z_min)
    arrow_len = 0.05 * max(x_span, z_span)

    # Height-based coloring (Y)
    y_min, y_max = float(cam_y.min()), float(cam_y.max())
    y_span = max(1e-6, y_max - y_min)
    norm_y = (cam_y - y_min) / y_span  # [0,1]
    cmap = plt.cm.plasma

    logger.info(
        "[CAM-PATH] Plotting %d poses on XZ density (grid_res=%d), arrow_stride=%d",
        len(poses),
        grid_res,
        arrow_stride,
    )

    fig, ax = plt.subplots(figsize=(16, 9))

    # Density background
    im = ax.imshow(
        H.T,
        origin="lower",
        extent=[x_min, x_max, z_min, z_max],
        aspect="equal",
        cmap="gray",
    )
    fig.colorbar(im, ax=ax, label="point density")

    # Path outline (light line)
    ax.plot(cam_x, cam_z, color="white", linewidth=1.0, alpha=0.6, label="camera path")

    # Height-colored points along path
    sc = ax.scatter(
        cam_x,
        cam_z,
        c=norm_y,
        cmap=cmap,
        s=10,
        alpha=0.9,
        label="camera samples (height)",
    )
    cbar = fig.colorbar(sc, ax=ax, label="camera height (normalized Y)")

    # Arrows for view direction (subsampled)
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

    # Mark start / end
    ax.scatter(cam_x[0], cam_z[0], c="green", s=60, marker="o", label="start")
    ax.scatter(cam_x[-1], cam_z[-1], c="magenta", s=60, marker="x", label="end")

    ax.set_title("Camera path & view direction on XZ density (height-colored)")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    logger.info("[CAM-PATH] Saved camera path plot: %s", out_path)


# ---------------------------------------------------------------------
# 3D visualization helpers
# ---------------------------------------------------------------------


def save_scene_points_3d(
    means: np.ndarray,
    out_path: Path,
    max_points: int = 100_000,
) -> None:
    """
    Save a coarse 3D scatter plot of the scene.

    Args:
        means: (N,3) Gaussian means.
        out_path: PNG path.
        max_points: maximum number of points to render (random subsample).
    """
    _ensure_matplotlib()
    xyz = np.asarray(means, dtype=np.float32)
    N = xyz.shape[0]

    if N == 0:
        logger.warning("[PLOT-3D] Scene has zero points, skipping 3D scatter.")
        return

    if N > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(N, size=max_points, replace=False)
        pts = xyz[idx]
        logger.info(
            "[PLOT-3D] Downsampling scene points %d -> %d for 3D scatter.",
            N,
            max_points,
        )
    else:
        pts = xyz

    xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        xs,
        ys,
        zs,
        s=1,
        c=zs,
        cmap="viridis",
        alpha=0.6,
    )

    ax.set_title("Scene points (3D view)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20.0, azim=60.0)  # some reasonable default view
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    logger.info("[PLOT-3D] Saved 3D scene scatter: %s", out_path)


def save_camera_path_3d(
    means: np.ndarray,
    poses: List[Dict[str, Any]],
    out_path: Path,
    max_points: int = 50_000,
    arrow_stride: int = 10,
) -> None:
    """
    Save a 3D plot of the scene and camera path.

    - Points: coarse subsample of the scene.
    - Camera path: polyline through eye positions.
    - View direction: arrows at every arrow_stride frames.

    Args:
        means: (N,3) Gaussian means.
        poses: list of dicts with keys "eye" and "center".
        out_path: PNG path.
        max_points: maximum number of scene points to show.
        arrow_stride: draw viewing direction arrows every k frames.
    """
    _ensure_matplotlib()
    if not poses:
        logger.warning("[PLOT-3D] No poses provided, skipping camera path 3D plot.")
        return

    xyz = np.asarray(means, dtype=np.float32)
    N = xyz.shape[0]

    if N > max_points:
        rng = np.random.default_rng(1)
        idx = rng.choice(N, size=max_points, replace=False)
        pts = xyz[idx]
        logger.info(
            "[PLOT-3D] Downsampling scene points %d -> %d for 3D cam path plot.",
            N,
            max_points,
        )
    else:
        pts = xyz

    xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]

    eyes = np.asarray([p["eye"] for p in poses], dtype=np.float32)      # [T,3]
    centers = np.asarray([p["center"] for p in poses], dtype=np.float32)  # [T,3]

    cam_x = eyes[:, 0]
    cam_y = eyes[:, 1]
    cam_z = eyes[:, 2]

    # Forward vectors
    fwd = centers - eyes
    norms = np.linalg.norm(fwd, axis=1, keepdims=True) + 1e-8
    fwd_norm = fwd / norms

    # Arrow length as fraction of scene diagonal
    bbox_min = xyz.min(axis=0)
    bbox_max = xyz.max(axis=0)
    diag = float(np.linalg.norm(bbox_max - bbox_min)) + 1e-6
    arrow_len = 0.1 * diag

    logger.info(
        "[PLOT-3D] Plotting 3D camera path with %d poses, arrow_stride=%d",
        len(poses),
        arrow_stride,
    )

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Scene scatter (light)
    ax.scatter(
        xs,
        ys,
        zs,
        s=1,
        c=zs,
        cmap="gray",
        alpha=0.3,
    )

    # Camera path
    ax.plot(
        cam_x,
        cam_y,
        cam_z,
        color="red",
        linewidth=2.0,
        label="camera path",
    )

    # Arrows for view direction (subsampled)
    idxs = np.arange(0, len(poses), max(1, arrow_stride))
    ax.quiver(
        cam_x[idxs],
        cam_y[idxs],
        cam_z[idxs],
        fwd_norm[idxs, 0] * arrow_len,
        fwd_norm[idxs, 1] * arrow_len,
        fwd_norm[idxs, 2] * arrow_len,
        length=1.0,
        normalize=False,
        color="cyan",
    )

    # Mark start / end
    ax.scatter(cam_x[0], cam_y[0], cam_z[0], c="green", s=40, marker="o", label="start")
    ax.scatter(
        cam_x[-1],
        cam_y[-1],
        cam_z[-1],
        c="magenta",
        s=40,
        marker="x",
        label="end",
    )

    ax.set_title("Camera path in 3D")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper right")
    ax.view_init(elev=25.0, azim=60.0)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    logger.info("[PLOT-3D] Saved 3D camera path plot: %s", out_path)


# ---------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Research / analysis of 3DGS scene (.ply): "
        "densities, slices, orientation stats, optional 3D views.",
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
        default=512,
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
    ap.add_argument(
        "--plot-3d",
        action="store_true",
        help="Also generate coarse 3D views of scene and camera path (if path is provided).",
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

    # 1) Load scene
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

    # 2) 1D histograms of coordinates
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

    # 3) Global XZ density map (top-down)
    save_density_xz(
        xyz,
        outdir / "density_xz.png",
        grid_res=args.grid_res,
        title="XZ density (top-down view)",
    )

    # 4) Y-slices projected to XZ (floor / ceiling / layers)
    slices_dir = outdir / "slices_y_xz"
    save_y_slices_xz(
        xyz,
        slices_dir,
        grid_res=args.grid_res,
        thickness_frac=args.slice_thickness_frac,
        num_slices=args.num_slices,
    )

    # 5) Orientation stats (quaternions -> forward axis -> angles)
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
        bins=max(1, args.angle_bins // 2),
    )

    # 6) Optional coarse 3D views
    if args.plot_3d:
        save_scene_points_3d(
            xyz,
            outdir / "scene_points_3d.png",
            max_points=100_000,
        )
        # 3D camera path is only meaningful if you call save_camera_path_3d
        # from the main cinematic pipeline with actual poses.
        logger.info(
            "[MAIN] 3D scene points saved; camera path 3D plot should be "
            "called from your main pipeline using save_camera_path_3d()."
        )

    logger.info("[MAIN] Analysis finished. Outputs in: %s", outdir)


if __name__ == "__main__":
    main()
