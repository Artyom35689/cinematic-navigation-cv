# app/src/detection.py
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def run_yolo_on_frames(
    frames_bgr: List[np.ndarray],
    model_path: str,
    conf: float,
) -> List[Dict[str, Any]]:
    from ultralytics import YOLO

    logger.info("[detection] Loading YOLO model: %s", model_path)
    model = YOLO("models/"+model_path)


    all_dets: List[Dict[str, Any]] = []
    for i, frame_bgr in enumerate(frames_bgr):
        logger.info("[detection] frame %d/%d", i + 1, len(frames_bgr))
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

        all_dets.append({"frame": i, "detections": frame_list})

    return all_dets


def make_det_fn_from_config(det_cfg: Dict[str, Any], outdir: Path):
    if not det_cfg.get("enabled", False):
        return None

    model = det_cfg.get("model", "yolov8n.pt")
    conf = float(det_cfg.get("conf", 0.25))
    out_json_name = det_cfg.get("output_json", "detections_yolo.json")

    def _fn(frames_bgr: List[np.ndarray]):
        dets = run_yolo_on_frames(frames_bgr, model_path=model, conf=conf)
        out_json = outdir / out_json_name
        out_json.write_text(json.dumps({"detections": dets}, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("[detection] wrote detections to %s", out_json)
        return dets

    return _fn
