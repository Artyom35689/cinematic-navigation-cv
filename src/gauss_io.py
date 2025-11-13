# src/gauss_io.py (или прямо в main.py / debug_one_frame.py)
from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
import torch
from plyfile import PlyData

SH_C0 = 0.28209479177387814  # 1 / (2 * sqrt(pi))

def _colors_from_f_dc(f_dc: np.ndarray) -> np.ndarray:
    """
    f_dc: [N, 3] – DC-компонента SH.
    Возвращаем приблизительный RGB в [0,1].
    """
    rgb = 0.5 + SH_C0 * f_dc
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    return rgb

def load_3dgs_unpacked_ply(
    path: str,
    max_points: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Загружает полноценный 3DGS PLY (x,y,z, scale_*, rot_*, opacity, f_dc_*).
    Возвращает:
      means:     [N,3] float32
      quats:     [N,4] float32 (нормализованные)
      scales:    [N,3] float32
      opacities: [N]   float32 в [0,1]
      colors:    [N,3] float32 в [0,1]
    """
    print(f"[DEBUG] load_3dgs_unpacked_ply: {path}")
    ply = PlyData.read(path)

    elem_names = [e.name for e in ply.elements]
    print(f"[DEBUG] elements: {elem_names}")

    # корректный способ достать vertex
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
        raise RuntimeError(
            f"PLY missing required fields {missing}. "
            f"Available fields: {names}"
        )

    # --- позиции ---
    x = np.asarray(v["x"], np.float32)
    y = np.asarray(v["y"], np.float32)
    z = np.asarray(v["z"], np.float32)
    means = np.stack([x, y, z], axis=1)  # [N,3]

    # --- raw scale & opacity ---
    s0 = np.asarray(v["scale_0"], np.float32)
    s1 = np.asarray(v["scale_1"], np.float32)
    s2 = np.asarray(v["scale_2"], np.float32)
    scales_raw = np.stack([s0, s1, s2], axis=1)  # [N,3]

    op_raw = np.asarray(v["opacity"], np.float32)

    scale_min, scale_max = float(scales_raw.min()), float(scales_raw.max())
    frac_neg_scale = float((scales_raw < 0.0).mean())
    op_min, op_max = float(op_raw.min()), float(op_raw.max())

    print(
        f"[DEBUG] scale_raw min/max={scale_min:.4f}/{scale_max:.4f}, "
        f"neg_frac={frac_neg_scale:.3f}"
    )
    print(f"[DEBUG] opacity_raw min/max={op_min:.4f}/{op_max:.4f}")

    # --- эвристика для scale ---
    if frac_neg_scale > 0.1:
        # похоже на лог-скейлы → exp
        scales = np.exp(scales_raw)
        print("[DEBUG] treating scales as log-stddev, applying exp()")
    else:
        scales = scales_raw
        print("[DEBUG] treating scales as already linear")

    # --- эвристика для opacity ---
    if op_min < 0.0 or op_max > 1.0:
        opacities = 1.0 / (1.0 + np.exp(-op_raw))
        print("[DEBUG] treating opacities as logits, applying sigmoid()")
    else:
        opacities = op_raw
        print("[DEBUG] treating opacities as already in [0,1]")

    opacities = opacities.astype(np.float32)

    # --- кватернионы ---
    q0 = np.asarray(v["rot_0"], np.float32)
    q1 = np.asarray(v["rot_1"], np.float32)
    q2 = np.asarray(v["rot_2"], np.float32)
    q3 = np.asarray(v["rot_3"], np.float32)
    quats = np.stack([q0, q1, q2, q3], axis=1)  # [N,4]

    # нормализация
    norm = np.linalg.norm(quats, axis=1, keepdims=True)
    norm[norm == 0.0] = 1.0
    quats = quats / norm

    # --- цвет из DC SH ---
    fdc0 = np.asarray(v["f_dc_0"], np.float32)
    fdc1 = np.asarray(v["f_dc_1"], np.float32)
    fdc2 = np.asarray(v["f_dc_2"], np.float32)
    f_dc = np.stack([fdc0, fdc1, fdc2], axis=1)  # [N,3]

    colors = _colors_from_f_dc(f_dc)  # [N,3] в [0,1]

    N = means.shape[0]
    print(f"[DEBUG] loaded {N} splats")

    # --- даунсэмплинг при необходимости ---
    if max_points is not None and N > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(N, size=max_points, replace=False)
        means = means[idx]
        quats = quats[idx]
        scales = scales[idx]
        opacities = opacities[idx]
        colors = colors[idx]
        print(f"[DEBUG] downsampled {N} -> {max_points}")

    # в torch
    means_t = torch.from_numpy(means)
    quats_t = torch.from_numpy(quats)
    scales_t = torch.from_numpy(scales)
    opacities_t = torch.from_numpy(opacities)
    colors_t = torch.from_numpy(colors)

    return means_t, quats_t, scales_t, opacities_t, colors_t


def load_gaussians_from_ply(
    path: str,
    max_points: Optional[int] = None,
):
    """
    Обёртка для main/debug:
    - если PLY всё ещё packed_* → просим конвертнуть
    - иначе используем load_3dgs_unpacked_ply().
    """
    ply = PlyData.read(path)
    elem_names = [e.name for e in ply.elements]
    print(f"[DEBUG] load_gaussians_from_ply, elements: {elem_names}")

    # ищем packed-формат
    has_packed = any(
        "packed_position" in (e.data.dtype.names or ())
        for e in ply.elements
    )
    if has_packed:
        raise RuntimeError(
            "This PLY still contains packed_* fields. "
            "Please convert it to full 3DGS PLY (x,y,z, scale_*, rot_*, "
            "opacity, f_dc_*) with SplatTransform или аналогом."
        )

    # если не packed → считаем, что уже «распакованный» PLY
    return load_3dgs_unpacked_ply(path, max_points=max_points)
