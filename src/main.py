# src/main.py

import argparse
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from plyfile import PlyData

import torch
from gsplat.rendering import rasterization

import ffmpeg  # ffmpeg-python


# ---- Constants for 3DGS decoding ----

# SH constant for DC band (C0)
SH_C0 = 0.28209479177387814  # standard value used in 3D Gaussian Splatting viewers


# ---- Data containers ----

@dataclass
class GaussiansNP:
    means: np.ndarray      # [N, 3]
    quats: np.ndarray      # [N, 4] (wxyz)
    scales: np.ndarray     # [N, 3]
    opacities: np.ndarray  # [N]
    colors: np.ndarray     # [N, 3] RGB in [0,1]


# ---- PLY loading ----

def load_gaussians_from_ply(path: str, max_points: int = 2_000_000) -> GaussiansNP:
    """
    Load Gaussians from a 3DGS-style PLY file.

    Supports two cases:
      1) "classic" 3D Gaussian Splatting PLY:
         vertex has fields: x,y,z, scale_0..2, rot_0..3, opacity, f_dc_0..2, ...
      2) packed_* format (SuperSplat-like): packed_position, packed_rotation,
         packed_scale, packed_color.

    Returns NumPy arrays; device (CPU/CUDA) is decided later.
    """
    logging.info("[INFO] Loading PLY: %s", path)
    ply = PlyData.read(path)

    # CORRECT: find element by name in the list of PlyElement
    vertex_elem = next((e for e in ply.elements if e.name == "vertex"), None)
    if vertex_elem is None:
        raise RuntimeError("PLY has no 'vertex' element")

    v = vertex_elem.data
    fields = list(v.dtype.names or [])
    logging.info("[DEBUG] vertex fields: %s", fields)

    # Case 1: classic 3DGS PLY
    required_3dgs = [
        "x", "y", "z",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
        "opacity",
        "f_dc_0", "f_dc_1", "f_dc_2",
    ]
    if all(f in fields for f in required_3dgs):
        n = len(v)
        logging.info("[INFO] Detected classic 3DGS PLY with %d vertices", n)

        xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

        # scales are stored in log-space -> exp
        scales = np.stack(
            [v["scale_0"], v["scale_1"], v["scale_2"]],
            axis=1,
        ).astype(np.float32)
        scales = np.exp(scales)

        # quaternions (wxyz), may be unnormalized
        quats = np.stack(
            [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]],
            axis=1,
        ).astype(np.float32)
        norms = np.linalg.norm(quats, axis=1, keepdims=True) + 1e-8
        quats /= norms

        # opacity is stored as logit -> sigmoid
        op = np.asarray(v["opacity"], dtype=np.float32)
        op = 1.0 / (1.0 + np.exp(-op))

        # f_dc_0..2 -> RGB in [0,1]
        dc = np.stack(
            [v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]],
            axis=1,
        ).astype(np.float32)
        rgb = 0.5 + SH_C0 * dc
        rgb = np.clip(rgb, 0.0, 1.0)

    # Case 2: packed_* format (heuristic unpack)
    elif "packed_position" in fields:
        n = len(v)
        logging.info("[INFO] Detected packed_* PLY, heuristic unpack, N=%d", n)

        pos_raw = np.frombuffer(v["packed_position"].tobytes(), dtype=np.float32)
        if pos_raw.size < n * 3:
            raise RuntimeError(
                f"packed_position size {pos_raw.size} < expected {n*3}"
            )
        xyz = pos_raw.reshape(-1, 3)[:n].astype(np.float32)

        col_raw = np.frombuffer(v["packed_color"].tobytes(), dtype=np.float32)
        if col_raw.size >= n * 3:
            col = col_raw.reshape(-1, 3)[:n].astype(np.float32)
            rgb = np.clip(col, 0.0, 1.0)
        else:
            rgb = np.ones((n, 3), dtype=np.float32) * 0.8  # fallback

        scales = np.ones((n, 3), dtype=np.float32) * 0.02
        quats = np.zeros((n, 4), dtype=np.float32)
        quats[:, 3] = 1.0  # identity quaternion
        op = np.ones((n,), dtype=np.float32) * 0.8

    else:
        raise RuntimeError(
            f"Unsupported vertex format in PLY. Fields: {fields}. "
            "Prefer 3DGS PLY with x,y,z,scale_*,rot_*,opacity,f_dc_* or use "
            "SplatTransform to convert from packed to standard PLY."
        )

    # Optional downsampling
    n_all = xyz.shape[0]
    if n_all > max_points:
        logging.info("[INFO] Downsampling: %d -> %d", n_all, max_points)
        idx = np.random.choice(n_all, size=max_points, replace=False)
        idx.sort()
        xyz = xyz[idx]
        scales = scales[idx]
        quats = quats[idx]
        op = op[idx]
        rgb = rgb[idx]

    logging.info(
        "[INFO] Loaded splats: %d (means/scales/quats/opacities/colors)", xyz.shape[0]
    )

    return GaussiansNP(
        means=xyz,
        quats=quats,
        scales=scales,
        opacities=op,
        colors=rgb,
    )



# ---- Camera path ----

def look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    """
    Build world-to-camera view matrix (4x4, row-major) for gsplat.

    eye, center, up are 3D vectors.
    """
    f = center - eye
    f = f / (np.linalg.norm(f) + 1e-8)
    up_n = up / (np.linalg.norm(up) + 1e-8)
    s = np.cross(f, up_n)
    s = s / (np.linalg.norm(s) + 1e-8)
    u = np.cross(s, f)

    # Rotation (camera space axes as rows)
    R = np.stack([s, u, -f], axis=0)  # 3x3
    t = -R @ eye  # 3

    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = R
    view[:3, 3] = t
    return view


def generate_camera_poses(
    means: np.ndarray,
    num_frames: int,
    orbit_angle_deg: float = 60.0,
    height_fraction: float = 0.1,
    distance_scale: float = 1.6,
) -> List[Dict[str, Any]]:
    """
    Very simple cinematic path:
      - Compute scene bounding box and center.
      - Place camera on an arc around the scene (orbit_angle_deg total).
      - Slightly elevated by height_fraction * scene_size.

    Returns list of dicts with 'view' (4x4), 'eye', 'center'.
    """
    mins = means.min(axis=0)
    maxs = means.max(axis=0)
    center = 0.5 * (mins + maxs)
    diag = np.linalg.norm(maxs - mins) + 1e-6

    radius = distance_scale * diag
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    poses = []
    for i in range(num_frames):
        t = 0.0 if num_frames == 1 else i / (num_frames - 1)
        # angle from -orbit/2 to +orbit/2
        angle = math.radians((t - 0.5) * orbit_angle_deg)
        # orbit in XZ plane
        offset = np.array([math.sin(angle), 0.0, math.cos(angle)], dtype=np.float32)
        eye = center + offset * radius
        eye[1] += height_fraction * diag  # small elevation

        view = look_at(eye, center, up)
        poses.append(
            {
                "view": view,
                "eye": eye.tolist(),
                "center": center.tolist(),
            }
        )

    logging.info(
        "[INFO] Generated %d camera poses around center=%s, radius=%.3f",
        num_frames,
        np.round(center, 3).tolist(),
        radius,
    )
    return poses


# ---- Rendering with gsplat ----

def render_frames_gsplat(
    gauss: GaussiansNP,
    poses: List[Dict[str, Any]],
    width: int,
    height: int,
    fov_deg: float,
    device: torch.device,
) -> List[np.ndarray]:
    """
    Render frames using gsplat.rendering.rasterization, following the official API:

        colors, alphas, meta = rasterization(
            means, quats, scales, opacities, colors,
            viewmats, Ks, width, height, ...
        )

    Shapes:
      means:    [N, 3]
      quats:    [N, 4]
      scales:   [N, 3]
      opacities:[N]
      colors:   [N, 3]  (RGB post-activation)
      viewmats: [C, 4, 4] with C=1
      Ks:       [C, 3, 3] with C=1
      output colors: [C, H, W, 3] -> [H, W, 3]
    """
    logging.info("[INFO] Trying gsplat on device: %s", device)

    means_t = torch.from_numpy(gauss.means).to(device)
    quats_t = torch.from_numpy(gauss.quats).to(device)
    scales_t = torch.from_numpy(gauss.scales).to(device)
    opac_t = torch.from_numpy(gauss.opacities).to(device)
    colors_t = torch.from_numpy(gauss.colors).to(device)

    # Intrinsics from vertical FOV
    fov_rad = math.radians(fov_deg)
    fy = 0.5 * height / math.tan(0.5 * fov_rad)
    fx = fy
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5

    Ks = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )[None, :, :]  # [1, 3, 3]

    frames_bgr: List[np.ndarray] = []

    num_frames = len(poses)
    for i, pose in enumerate(poses):
        view = torch.from_numpy(np.asarray(pose["view"], dtype=np.float32)).to(device)
        view = view[None, :, :]  # [1, 4, 4]

        logging.info("[INFO] [gsplat] frame %d/%d", i + 1, num_frames)

        colors, alphas, meta = rasterization(
            means_t,
            quats_t,
            scales_t,
            opac_t,
            colors_t,
            view,
            Ks,
            width,
            height,
            near_plane=0.01,
            far_plane=1e4,
            sh_degree=None,       # we pass post-activation RGB
            packed=True,          # default, memory-efficient
            render_mode="RGB",
            rasterize_mode="classic",
        )

        # colors: [1, H, W, 3], values in [0,1]
        frame_rgb = colors[0].clamp(0.0, 1.0).detach().cpu().numpy()  # [H, W, 3]
        if frame_rgb.shape != (height, width, 3):
            raise RuntimeError(
                f"Unexpected frame shape from gsplat: {frame_rgb.shape}, "
                f"expected ({height}, {width}, 3)"
            )

        frame_u8 = (frame_rgb * 255.0 + 0.5).astype(np.uint8)
        frame_bgr = frame_u8[..., ::-1]  # RGB -> BGR
        frames_bgr.append(frame_bgr)

    return frames_bgr


# ---- Video writer (rawvideo -> ffmpeg) ----

def write_video(frames_bgr: List[np.ndarray], out_path: Path, fps: int) -> None:
    """
    Write BGR frames to MP4 using ffmpeg via rawvideo pipe.

    This avoids PNG / libpng entirely and enforces a consistent frame size.
    """
    if not frames_bgr:
        raise RuntimeError("No frames to write")

    # Validate shapes
    h, w, c = frames_bgr[0].shape
    if c != 3:
        raise RuntimeError(f"First frame has unexpected channels: {c}")

    for idx, f in enumerate(frames_bgr):
        if f.shape != (h, w, 3):
            raise RuntimeError(
                f"Frame {idx} has shape {f.shape}, expected {(h, w, 3)}"
            )
        if f.dtype != np.uint8:
            raise RuntimeError(
                f"Frame {idx} has dtype {f.dtype}, expected uint8"
            )

    logging.info(
        "[INFO] Writing video: %s (%d frames, %dx%d @ %d FPS)",
        out_path,
        len(frames_bgr),
        w,
        h,
        fps,
    )

    stack = np.stack(frames_bgr, axis=0)  # [T, H, W, 3]
    frames_bytes = stack.tobytes()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    process = (
        ffmpeg
        .input(
            "pipe:",
            format="rawvideo",
            pix_fmt="bgr24",
            s=f"{w}x{h}",
            r=fps,
        )
        .output(
            str(out_path),
            vcodec="libx264",
            pix_fmt="yuv420p",
        )
        .overwrite_output()
        .run_async(pipe_stdin=True, quiet=True)
    )

    process.stdin.write(frames_bytes)
    process.stdin.close()
    ret = process.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg returned non-zero exit code: {ret}")


# ---- (Optional) YOLO detection on rendered frames ----

def run_detection(
    frames_bgr: List[np.ndarray],
    model_path: str,
    conf: float,
) -> List[Dict[str, Any]]:
    """
    Run Ultralytics YOLO on BGR frames and collect simple bounding boxes.

    Result format:
      [
        {
          "frame": i,
          "detections": [
            {"cls": int, "conf": float, "xyxy": [x1,y1,x2,y2]},
            ...
          ]
        },
        ...
      ]
    """
    from ultralytics import YOLO

    logging.info("[INFO] Loading YOLO model: %s", model_path)
    model = YOLO(model_path)

    all_dets: List[Dict[str, Any]] = []
    for i, frame_bgr in enumerate(frames_bgr):
        logging.info("[INFO] [YOLO] frame %d/%d", i + 1, len(frames_bgr))
        # YOLO expects RGB by default
        frame_rgb = frame_bgr[..., ::-1]
        res = model.predict(frame_rgb, conf=conf, verbose=False)[0]

        frame_list = []
        if res.boxes is not None:
            for b in res.boxes:
                cls_id = int(b.cls.item())
                score = float(b.conf.item())
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                frame_list.append(
                    {
                        "cls": cls_id,
                        "conf": score,
                        "xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    }
                )

        all_dets.append(
            {
                "frame": i,
                "detections": frame_list,
            }
        )

    return all_dets


# ---- Main CLI ----

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Simple cinematic navigation renderer with gsplat + YOLO."
    )
    p.add_argument(
        "--scene",
        type=str,
        required=True,
        help="Path to .ply scene (3DGS-style or packed_*).",
    )
    p.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output directory for video and JSON.",
    )
    p.add_argument(
        "--seconds",
        type=float,
        default=4.0,
        help="Duration of the fly-through in seconds.",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Frames per second.",
    )
    p.add_argument(
        "--fov",
        type=float,
        default=70.0,
        help="Vertical field of view in degrees.",
    )
    p.add_argument(
        "--res",
        type=str,
        default="1280x720",
        help="Resolution as WxH, e.g. 1280x720.",
    )
    p.add_argument(
        "--max-splats",
        type=int,
        default=2_000_000,
        help="Max number of Gaussians to keep (random downsampling if exceeded).",
    )
    p.add_argument(
        "--detect",
        action="store_true",
        help="Run YOLO detection on rendered frames.",
    )
    p.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model path/name for detection.",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold.",
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

    # Resolution parsing
    try:
        w_str, h_str = args.res.lower().split("x")
        width, height = int(w_str), int(h_str)
    except Exception as e:
        raise SystemExit(f"Invalid --res '{args.res}', expected WxH, got error: {e}")

    # Number of frames
    num_frames = max(1, int(round(args.seconds * args.fps)))
    logging.info(
        "[INFO] Frames to render: %d at %d FPS, %dx%d, FOV=%.1f",
        num_frames,
        args.fps,
        width,
        height,
        args.fov,
    )
    # Load Gaussians from PLY
    gauss = load_gaussians_from_ply(str(scene_path), max_points=4294967295)

    # Camera poses
    poses = generate_camera_poses(gauss.means, num_frames)

    # Choose device for gsplat
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available in this container, but gsplat requires CUDA. "
            "Check that you run docker with --gpus and the NVIDIA runtime."
        )
    device = torch.device("cuda")

    # Render frames with gsplat
    frames_bgr = render_frames_gsplat(
        gauss,
        poses,
        width=width,
        height=height,
        fov_deg=args.fov,
        device=device,
    )

    # Save video
    out_mp4 = outdir / "panorama_tour.mp4"
    write_video(frames_bgr, out_mp4, fps=args.fps)
    logging.info("[INFO] Video written: %s", out_mp4)

    # Save camera path JSON
    cam_json = outdir / "camera_path.json"
    meta = {
        "scene": str(scene_path),
        "resolution": [width, height],
        "fps": args.fps,
        "seconds": args.seconds,
        "fov_deg": args.fov,
        "frames": [
            {
                "index": i,
                "eye": poses[i]["eye"],
                "center": poses[i]["center"],
                "view": np.asarray(poses[i]["view"], dtype=float).tolist(),
            }
            for i in range(len(poses))
        ],
        "render_backend": "gsplat_rgb",
    }
    with cam_json.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logging.info("[INFO] Camera path JSON written: %s", cam_json)

    # Optional YOLO detection
    if args.detect:
        dets = run_detection(frames_bgr, args.model, args.conf)
        det_json = outdir / "detections_yolo.json"
        with det_json.open("w", encoding="utf-8") as f:
            json.dump({"detections": dets}, f, indent=2)
        logging.info("[INFO] YOLO detections JSON written: %s", det_json)


if __name__ == "__main__":
    main()
