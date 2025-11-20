#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cinematic navigation pipeline (config-driven):

  - load unpacked 3DGS PLY into GaussiansNP
  - generate camera path in XZ plane (Y up) using configured waypoints + behaviors
  - render frames with gsplat and stream directly into ffmpeg (MP4)
  - optionally run YOLO detection during rendering
  - optionally run scene analysis (histograms, slices, orientation, camera path plot)

Usage (default configs inside this file):

  python3 -m src.main

If you want to customize behavior, edit GLOBAL_CONFIG and SCENE_CONFIG_CONF_HALL
or call main(scene_cfg=..., global_cfg=...) from your own script.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .gsplat_scene import (
    load_gaussians_from_ply,
    generate_camera_poses_spline,
    generate_camera_poses_straight_path,
)
from .render_utils import render_gsplat_to_video_streaming
from .analysis import research_scene as rs  # reuse analysis helpers

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------

# Scene config: only scene path + path definition/behavior.
SCENE_CONFIG_CONF_HALL: Dict[str, Any] = {
    "name": "ConferenceHall_demo",
    "scene_path": "scenes/ConferenceHall.ply",
    # Path type: "spline" or "straight"
    "path_type": "spline",
    # Waypoints + behaviors; behavior applies from this waypoint to the next.
    # Behavior is optional (None) or a dict with a known "mode".
    #
    # NOTE:
    #   - World up: Y
    #   - Camera moves in XZ plane (waypoint is [x, z]).
    "path_with_behaviors": [
        # segment 0: [28,30] -> [20,22], no special behavior
        {"waypoint": [28.0, 30.0], "behavior": None},

        # segment 1: [20,22] -> [10,24], look at a specific point
        {
            "waypoint": [20.0, 24.0],
            "behavior": {
                "mode": "look_at_point",
                "target": [20.0, -3.0, 25.0],
                "strength": 0.5,
            },
        },

        # segment 2: [10,24] -> [0,24], height arc (e.g. under ceiling)
        {
            "waypoint": [10.0, 24.0],
            "behavior": {
                "mode": "height_arc",
                # fraction of scene diagonal used as vertical bump at segment center
                "height_offset_fraction": 0.01,
            },
        },

        # segment 3: [0,24] -> [0,-5], extra yaw twist along the segment
        {
            "waypoint": [0.0, 24.0],
            "behavior": {
                "mode": "extra_yaw",
                "angle_deg": -100.0,  # full spin over the segment
            },
        },

        # remaining segments with default behavior
        {"waypoint": [0.0, -5.0], "behavior": None},
        {"waypoint": [2.0, -5.0], "behavior": None},
        {"waypoint": [2.0, 0.0], "behavior": None},
        {"waypoint": [5.1, 0.0], "behavior": None},
        {"waypoint": [5.1, 25.0], "behavior": None},  # behavior for last waypoint ignored
    ],
    # For straight path mode (if you switch path_type="straight"),
    # only waypoints_xz are used:
    "straight_path_waypoints_xz": [
        [28.0, 30.0],
        [20.0, 22.0],
        [10.0, 24.0],
        [0.0, 24.0],
        [0.0, -5.0],
        [2.0, -5.0],
        [2.0, 0.0],
        [5.1, 0.0],
        [5.1, 25.0],
    ],
}

# Global config: everything else (render, video, detection, analysis).
GLOBAL_CONFIG: Dict[str, Any] = {
    "outdir": "output/ConferenceHall_demo",
    "seconds": 60.0,
    "fps": 60,
    "fov_deg": 75.0,
    "resolution": [1280, 720],  # [width, height]
    "max_splats": 20_000_000,
    "detect": False,
    "yolo_model": "models/yolo12n.pt",
    "yolo_conf": 0.5,
    "draw_boxes": True,  # draw YOLO boxes directly on frames when detect=True

    # Scene analysis: if True, in addition to video we generate analysis plots.
    "analyze_scene": True,
    "analysis": {
        "grid_res": 512,
        "slice_thickness_frac": 0.1,
        "num_slices": 6,
        "angle_bins": 180,
        # Where to put analysis results (inside outdir)
        "subdir": "analysis",
    },

    # Camera path plotting on density map (also useful for debugging)
    "plot_camera_path": True,
    "camera_path_plot": "camera_path_on_density_xz.png",
    "camera_path_grid_res": 512,
    "camera_path_arrow_stride": 10,
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _compute_num_frames(seconds: float, fps: int) -> int:
    """Compute number of frames from duration and FPS."""
    return max(1, int(round(seconds * fps)))


def _get_resolution(res_value: Any) -> Tuple[int, int]:
    """
    Accepts resolution in forms:
      - [width, height]
      - (width, height)
      - "WxH" string
    Returns (width, height) as ints.
    """
    if isinstance(res_value, (list, tuple)) and len(res_value) == 2:
        return int(res_value[0]), int(res_value[1])

    if isinstance(res_value, str):
        try:
            w_str, h_str = res_value.lower().split("x")
            return int(w_str), int(h_str)
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"Invalid resolution string '{res_value}': {e}") from e

    raise ValueError(f"Unsupported resolution format: {res_value!r}")


def _maybe_run_scene_analysis(
    gauss,
    scene_path: Path,
    outdir: Path,
    global_cfg: Dict[str, Any],
) -> None:
    """
    Optionally run scene analysis (histograms, slices, orientation) using
    helpers from src.analysis.research_scene.
    """
    if not global_cfg.get("analyze_scene", False):
        logger.info("[ANALYSIS] Scene analysis disabled in GLOBAL_CONFIG.")
        return

    analysis_cfg = global_cfg.get("analysis", {}) or {}
    grid_res = int(analysis_cfg.get("grid_res", 512))
    slice_thickness_frac = float(analysis_cfg.get("slice_thickness_frac", 0.1))
    num_slices = int(analysis_cfg.get("num_slices", 5))
    angle_bins = int(analysis_cfg.get("angle_bins", 180))

    analysis_subdir = analysis_cfg.get("subdir", "analysis")
    analysis_dir = outdir / analysis_subdir
    analysis_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "[ANALYSIS] Running scene analysis for '%s' into %s",
        scene_path,
        analysis_dir,
    )

    xyz = gauss.means
    bbox_min = xyz.min(axis=0)
    bbox_max = xyz.max(axis=0)
    diag = float(np.linalg.norm(bbox_max - bbox_min)) + 1e-6

    logger.info(
        "[ANALYSIS] N=%d, bbox_min=%s, bbox_max=%s, diag=%.3f",
        xyz.shape[0],
        np.round(bbox_min, 3),
        np.round(bbox_max, 3),
        diag,
    )

    # 1D histograms for X, Y, Z
    rs.save_histogram(
        xyz[:, 0],
        analysis_dir / "hist_x.png",
        title="X distribution",
        xlabel="X",
    )
    rs.save_histogram(
        xyz[:, 1],
        analysis_dir / "hist_y.png",
        title="Y distribution",
        xlabel="Y",
    )
    rs.save_histogram(
        xyz[:, 2],
        analysis_dir / "hist_z.png",
        title="Z distribution",
        xlabel="Z",
    )

    # Global density in XZ
    rs.save_density_xz(
        xyz,
        analysis_dir / "density_xz.png",
        grid_res=grid_res,
        title="XZ density (top-down view)",
    )

    # Y slices in XZ
    slices_dir = analysis_dir / "slices_y_xz"
    rs.save_y_slices_xz(
        xyz,
        slices_dir,
        grid_res=grid_res,
        thickness_frac=slice_thickness_frac,
        num_slices=num_slices,
    )

    # Orientations (quaternions -> forward axis -> yaw/pitch)
    dirs = rs.quat_forward_axis_wxyz(gauss.quats)
    azimuth_deg, elevation_deg = rs.forward_angles_y_up(dirs)

    rs.save_histogram(
        azimuth_deg,
        analysis_dir / "orient_azimuth_deg.png",
        title="Forward azimuth (deg, Y-up, XZ plane)",
        xlabel="azimuth (deg, -180..180)",
        bins=angle_bins,
    )
    rs.save_histogram(
        elevation_deg,
        analysis_dir / "orient_elevation_deg.png",
        title="Forward elevation (deg, vs XZ plane)",
        xlabel="elevation (deg, -90..90)",
        bins=max(1, angle_bins // 2),
    )

    logger.info("[ANALYSIS] Scene analysis finished.")


def _maybe_plot_camera_path(
    gauss,
    poses: List[Dict[str, Any]],
    outdir: Path,
    global_cfg: Dict[str, Any],
) -> None:
    """Optionally plot camera path on XZ density map."""
    if not global_cfg.get("plot_camera_path", True):
        logger.info("[CAM-PATH] Camera path plotting disabled in GLOBAL_CONFIG.")
        return

    rel_path = global_cfg.get("camera_path_plot", "camera_path_on_density_xz.png")
    grid_res = int(global_cfg.get("camera_path_grid_res", 512))
    arrow_stride = int(global_cfg.get("camera_path_arrow_stride", 10))

    out_path = outdir / rel_path
    rs.save_camera_path_on_density_xz(
        means=gauss.means,
        poses=poses,
        out_path=out_path,
        grid_res=grid_res,
        arrow_stride=arrow_stride,
    )


def _build_path_with_behaviors(scene_cfg: Dict[str, Any]) -> List[Tuple[List[float], Optional[Dict[str, Any]]]]:
    """
    Convert scene config 'path_with_behaviors' (list of dicts) into
    list of (waypoint_xz, behavior) tuples for generate_camera_poses_spline.
    """
    path_items = scene_cfg.get("path_with_behaviors", [])
    if not path_items:
        raise ValueError("Scene config has empty 'path_with_behaviors' list.")

    out: List[Tuple[List[float], Optional[Dict[str, Any]]]] = []
    for item in path_items:
        wp = item.get("waypoint")
        behavior = item.get("behavior", None)
        if not isinstance(wp, (list, tuple)) or len(wp) != 2:
            raise ValueError(f"Invalid waypoint in scene config: {item!r}")
        out.append(([float(wp[0]), float(wp[1])], behavior))
    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main(
    scene_cfg: Optional[Dict[str, Any]] = None,
    global_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Main entry point.

    If scene_cfg / global_cfg are None, built-in SCENE_CONFIG_CONF_HALL / GLOBAL_CONFIG
    are used. You can override them by importing this module and calling:

        from src.main import main

        my_scene_cfg = {...}
        my_global_cfg = {...}
        main(scene_cfg=my_scene_cfg, global_cfg=my_global_cfg)
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    if scene_cfg is None:
        scene_cfg = SCENE_CONFIG_CONF_HALL
    if global_cfg is None:
        global_cfg = GLOBAL_CONFIG

    scene_path = Path(scene_cfg["scene_path"]).resolve()
    outdir = Path(global_cfg["outdir"]).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    width, height = _get_resolution(global_cfg.get("resolution", [1280, 720]))
    seconds = float(global_cfg.get("seconds", 30.0))
    fps = int(global_cfg.get("fps", 30))
    fov_deg = float(global_cfg.get("fov_deg", 80.0))
    max_splats = int(global_cfg.get("max_splats", 2_000_000))

    detect = bool(global_cfg.get("detect", False))
    yolo_model = str(global_cfg.get("yolo_model", "models/yolo12n.pt"))
    yolo_conf = float(global_cfg.get("yolo_conf", 0.5))
    draw_boxes = bool(global_cfg.get("draw_boxes", True))

    num_frames = _compute_num_frames(seconds, fps)
    logger.info(
        "[MAIN] Scene: %s",
        scene_path,
    )
    logger.info(
        "[MAIN] Output dir: %s",
        outdir,
    )
    logger.info(
        "[MAIN] Frames to render: %d @ %d FPS, %dx%d, FOV=%.1f, max_splats=%d",
        num_frames,
        fps,
        width,
        height,
        fov_deg,
        max_splats,
    )

    # ------------------------------------------------------------------
    # 1) Load Gaussians
    # ------------------------------------------------------------------
    gauss = load_gaussians_from_ply(str(scene_path), max_points=max_splats)

    # ------------------------------------------------------------------
    # 2) Camera poses
    # ------------------------------------------------------------------
    path_type = str(scene_cfg.get("path_type", "spline")).lower()

    if path_type == "spline":
        path_with_behaviors = _build_path_with_behaviors(scene_cfg)
        poses = generate_camera_poses_spline(
            means=gauss.means,
            num_frames=num_frames,
            path_with_behaviors=path_with_behaviors,
            height_fraction=0.0,
            lookahead_fraction=0.05,
            samples_per_segment=64,
        )
        path_type_meta = "spline_xz_with_behaviors"

    elif path_type == "straight":
        waypoints_xz = scene_cfg.get("straight_path_waypoints_xz")
        if not waypoints_xz:
            raise ValueError(
                "Scene config path_type='straight' but 'straight_path_waypoints_xz' is empty."
            )
        poses = generate_camera_poses_straight_path(
            means=gauss.means,
            num_frames=num_frames,
            waypoints_xz=waypoints_xz,
            height_fraction=0.0,
            lookahead_fraction=0.05,
        )
        path_type_meta = "straight_polyline_xz"

    else:
        raise ValueError(f"Unsupported path_type in scene config: {path_type!r}")

    # Plot camera path on density map (if enabled)
    _maybe_plot_camera_path(gauss, poses, outdir, global_cfg)

    # ------------------------------------------------------------------
    # 3) Device
    # ------------------------------------------------------------------
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available in this container, but gsplat requires CUDA. "
            "Check that you run docker with --gpus and NVIDIA runtime."
        )
    device = torch.device("cuda")
    logger.info("[MAIN] Using device: %s", device)

    # ------------------------------------------------------------------
    # 4) Render + stream to video (and optional YOLO)
    # ------------------------------------------------------------------
    out_mp4 = outdir / "panorama_tour.mp4"

    logger.info("[MAIN] Starting rendering to video: %s", out_mp4)
    dets = render_gsplat_to_video_streaming(
        gauss=gauss,
        poses=poses,
        width=width,
        height=height,
        fov_deg=fov_deg,
        device=device,
        out_path=out_mp4,
        fps=fps,
        detect=detect,
        yolo_model_path=yolo_model,
        yolo_conf=yolo_conf,
        draw_boxes=(detect and draw_boxes),
    )
    logger.info("[MAIN] Video written: %s", out_mp4)

    # ------------------------------------------------------------------
    # 5) Camera path JSON meta
    # ------------------------------------------------------------------
    cam_json = outdir / "camera_path.json"
    meta_frames: List[Dict[str, Any]] = []
    for i, pose in enumerate(poses):
        meta_frames.append(
            {
                "index": i,
                "eye": pose["eye"],
                "center": pose["center"],
                "view": np.asarray(pose["view"], dtype=float).tolist(),
            }
        )

    meta = {
        "scene": str(scene_path),
        "resolution": [width, height],
        "fps": fps,
        "seconds": seconds,
        "fov_deg": fov_deg,
        "frames": meta_frames,
        "render_backend": "gsplat_rgb",
        "path_type": path_type_meta,
    }
    with cam_json.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info("[MAIN] Camera path JSON written: %s", cam_json)

    # ------------------------------------------------------------------
    # 6) Optional YOLO detections JSON (already computed during rendering)
    # ------------------------------------------------------------------
    if detect and dets is not None:
        det_json = outdir / "detections_yolo.json"
        with det_json.open("w", encoding="utf-8") as f:
            json.dump({"detections": dets}, f, indent=2)
        logger.info("[MAIN] YOLO detections JSON written: %s", det_json)

    # ------------------------------------------------------------------
    # 7) Optional scene analysis (histograms, slices, orientations)
    # ------------------------------------------------------------------
    _maybe_run_scene_analysis(gauss, scene_path, outdir, global_cfg)

    logger.info("[MAIN] Done.")


if __name__ == "__main__":
    main()
