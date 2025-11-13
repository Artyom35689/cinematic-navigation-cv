#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal debug: load 3DGS-style PLY (unpacked), render ONE frame with gsplat,
save it as PNG and print stats.

Run inside container from /app:

  python3 -m src.debug_one_frame \
    --scene scenes/ConferenceHall.ply \
    --outdir output/debug_conference \
    --width 1280 --height 720 --fov 70
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import cv2
from plyfile import PlyData
import torch
from gsplat.rendering import rasterization

SH_C0 = 0.28209479177387814  # SH DC constant


# ---------------------------------------------------------------------
# 1) Load full 3DGS PLY (unpacked format)
# ---------------------------------------------------------------------
def load_3dgs_unpacked_ply(path: str, max_points: int | None = None):
    """
    Load full 3DGS PLY (unpacked):

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

    Returns numpy arrays:
      means:     (N,3)
      quats:     (N,4)
      scales:    (N,3)
      opacities: (N,)
      colors:    (N,3)
    """
    print(f"[DEBUG] load_3dgs_unpacked_ply: {path}")
    ply = PlyData.read(path)

    elem_names = [e.name for e in ply.elements]
    print(f"[DEBUG] elements: {elem_names}")

    try:
        v = ply["vertex"]
    except KeyError:
        raise RuntimeError(f"PLY has no 'vertex' element; available elements: {elem_names}")

    names = v.data.dtype.names
    print(f"[DEBUG] vertex fields: {names}")

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

    print(f"[DEBUG] scale_raw min/max={scale_min:.4f}/{scale_max:.4f}, neg_frac={frac_neg_scale:.3f}")
    print(f"[DEBUG] opacity_raw min/max={op_min:.4f}/{op_max:.4f}")

    # scales: heuristic – treat as log-stddev if many negatives
    if frac_neg_scale > 0.1:
        scales = np.exp(scales_raw)
        print("[DEBUG] treating scales as log-stddev, applying exp()")
    else:
        scales = scales_raw
        print("[DEBUG] treating scales as already linear")

    # opacities: heuristic – logits → sigmoid if values outside [0,1]
    if op_min < 0.0 or op_max > 1.0:
        opacities = 1.0 / (1.0 + np.exp(-op_raw))
        print("[DEBUG] treating opacities as logits, applying sigmoid()")
    else:
        opacities = op_raw
        print("[DEBUG] treating opacities as already in [0,1]")

    opacities = opacities.astype(np.float32)

    # --- quaternions ---
    q0 = np.asarray(v["rot_0"], np.float32)
    q1 = np.asarray(v["rot_1"], np.float32)
    q2 = np.asarray(v["rot_2"], np.float32)
    q3 = np.asarray(v["rot_3"], np.float32)
    quats = np.stack([q0, q1, q2, q3], axis=1)  # (N,4)

    # normalize quats
    norm = np.linalg.norm(quats, axis=1, keepdims=True)
    norm[norm == 0.0] = 1.0
    quats = quats / norm

    # --- color from DC SH ---
    fdc0 = np.asarray(v["f_dc_0"], np.float32)
    fdc1 = np.asarray(v["f_dc_1"], np.float32)
    fdc2 = np.asarray(v["f_dc_2"], np.float32)
    f_dc = np.stack([fdc0, fdc1, fdc2], axis=1)  # (N,3)

    # Approximate RGB: 0.5 + SH_C0 * f_dc, then clamp
    colors = 0.5 + SH_C0 * f_dc
    colors = np.clip(colors, 0.0, 1.0).astype(np.float32)

    N = means.shape[0]
    print(f"[DEBUG] loaded {N} splats")

    # downsample if needed
    if max_points is not None and N > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(N, size=max_points, replace=False)
        means = means[idx]
        quats = quats[idx]
        scales = scales[idx]
        opacities = opacities[idx]
        colors = colors[idx]
        print(f"[DEBUG] downsampled {N} -> {max_points}")

    return means, quats, scales, opacities, colors


# ---------------------------------------------------------------------
# 2) Simple camera from bbox
# ---------------------------------------------------------------------
# debug_one_frame.py (и аналогично в main.py)
import numpy as np
import torch


def build_camera(means: np.ndarray,
                 fov_deg: float,
                 width: int,
                 height: int):
    """
    means: (N,3) numpy, уже из распакованного 3DGS PLY.
    Строим простую pinhole-камеру, смотрящую на центр сцены.

    Возвращаем:
      view: (4,4) world->camera
      K:    (3,3) intrinsics
    """

    # --- bbox и центр сцены ---
    bbox_min = means.min(axis=0)
    bbox_max = means.max(axis=0)
    center   = 0.5 * (bbox_min + bbox_max)
    diag     = float(np.linalg.norm(bbox_max - bbox_min))
    if diag < 1e-6:
        diag = 1.0

    # Ставим камеру "перед" сценой по -Z, чтобы смотреть в сторону +Z
    cam_pos = center + np.array([0.0, 0.0, -0.7 * diag], dtype=np.float32)

    print("[DEBUG] camera center:", center)
    print("[DEBUG] camera pos:", cam_pos)
    print("[DEBUG] bbox_min:", bbox_min, "bbox_max:", bbox_max, "diag:", diag)

    def normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v if n < 1e-8 else v / n

    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # ВАЖНО:
    # forward = направление от камеры к центру сцены → ось +Z камеры
    forward = normalize(center - cam_pos)          # +Z_cam
    right   = normalize(np.cross(forward, up))     # +X_cam
    new_up  = np.cross(right, forward)             # +Y_cam

    # R: строки — оси камеры в мировых координатах
    R = np.stack([right, new_up, forward], axis=0).astype(np.float32)  # (3,3)
    t = -R @ cam_pos.astype(np.float32)                                 # (3,)

    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = R
    view[:3,  3] = t

    # --- pinhole-интринсики под заданный FOV (горизонтальный) ---
    fov_rad = np.deg2rad(float(fov_deg))
    fx = 0.5 * width / np.tan(0.5 * fov_rad)
    fy = fx  # пусть FOV по вертикали получается из соотношения сторон
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5

    K = np.array(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    print("[DEBUG] K:", K)

    # --- быстрая проверка глубин ---
    M = min(4096, means.shape[0])
    idx = np.random.choice(means.shape[0], size=M, replace=False)
    pts = means[idx]  # (M,3)
    pts_h = np.concatenate([pts, np.ones((M, 1), dtype=np.float32)], axis=1)  # (M,4)
    cam_pts = (view @ pts_h.T).T  # (M,4)
    z_vals = cam_pts[:, 2]
    print(
        "[DEBUG] depth stats: "
        f"min={z_vals.min():.3f}, max={z_vals.max():.3f}, mean={z_vals.mean():.3f}"
    )

    return view, K



# ---------------------------------------------------------------------
# 3) One-frame gsplat rendering
# ---------------------------------------------------------------------
def render_one_frame(
    means: np.ndarray,
    quats: np.ndarray,
    scales: np.ndarray,
    opacities: np.ndarray,
    colors: np.ndarray,
    view: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    device: str = "cuda",
):
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] cuda requested but not available, fallback to cpu")
        device = "cpu"

    dev = torch.device(device)

    means_t = torch.from_numpy(means).to(dev)
    quats_t = torch.from_numpy(quats).to(dev)
    scales_t = torch.from_numpy(scales).to(dev)
    opac_t = torch.from_numpy(opacities).to(dev)
    colors_t = torch.from_numpy(colors).to(dev)

    view_t = torch.from_numpy(view).to(dev).unsqueeze(0)  # (1,4,4)
    K_t = torch.from_numpy(K).to(dev).unsqueeze(0)        # (1,3,3)

    print(
        "[DEBUG] shapes:",
        "means", means_t.shape,
        "scales", scales_t.shape,
        "quats", quats_t.shape,
        "opac", opac_t.shape,
        "colors", colors_t.shape,
        "view", view_t.shape,
        "K", K_t.shape,
    )

    images, depths, meta = rasterization(
        means=means_t,
        quats=quats_t,
        scales=scales_t,
        opacities=opac_t,
        colors=colors_t,
        viewmats=view_t,
        Ks=K_t,
        width=width,
        height=height,
        near_plane=0.01,
        far_plane=1e10,
        render_mode="RGB",
        camera_model="pinhole",
    )

    # images: (B,H,W,3) in [0,1]
    img = images[0].clamp(0.0, 1.0).detach().cpu().numpy()
    print("[DEBUG] img stats: min=%.6f max=%.6f mean=%.6f"
          % (img.min(), img.max(), img.mean()))

    # RGB->BGR for OpenCV
    img_bgr = (img[..., ::-1] * 255.0).astype(np.uint8)
    return img_bgr


# ---------------------------------------------------------------------
# 4) CLI
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fov", type=float, default=70.0)
    ap.add_argument("--max_splats", type=int, default=2_000_000)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Scene:", args.scene)
    print("[INFO] Outdir:", outdir)

    means, quats, scales, opacities, colors = load_3dgs_unpacked_ply(
        args.scene,
        max_points=args.max_splats,
    )

    view, K = build_camera(means, args.fov, args.width, args.height)

    img_bgr = render_one_frame(
        means,
        quats,
        scales,
        opacities,
        colors,
        view,
        K,
        width=args.width,
        height=args.height,
        device=args.device,
    )

    out_png = outdir / "debug_frame.png"
    cv2.imwrite(str(out_png), img_bgr)
    print("[INFO] Saved PNG:", out_png)

    meta = {
        "scene": args.scene,
        "width": args.width,
        "height": args.height,
        "fov_deg": args.fov,
        "bbox_min": [float(x) for x in means.min(axis=0)],
        "bbox_max": [float(x) for x in means.max(axis=0)],
    }
    meta_path = outdir / "debug_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print("[INFO] Saved meta:", meta_path)


if __name__ == "__main__":
    main()
