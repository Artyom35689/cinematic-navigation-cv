#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cli_main.py

Точка входа для запуска path_runner по YAML-конфигу.

Использование (из корня проекта /app):

  python -m src.cli_main                  # возьмёт configs/conferencehall.yaml
  python -m src.cli_main path/to/cfg.yaml # явный путь к конфигу
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

from .path_runner import run_path_from_config


def _init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[{levelname}] [{name}] {message}",
        style="{",
    )


def _load_config(cfg_path: Path) -> Dict[str, Any]:
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError("Top-level YAML config must be a mapping (dict)")
    return cfg


def _normalize_config(cfg: Dict[str, Any], cfg_path: Path) -> Dict[str, Any]:
    """
    Приводим старую структуру к новой:

      - если в cfg["scene"]["output_dir"] есть путь — перенаправляем его
        в cfg["output"]["dir"], чтобы path_runner его использовал;
      - поддерживаем и новый стиль с cfg["output"]["dir"].
    """
    scene_cfg = cfg.get("scene", {})
    if not isinstance(scene_cfg, dict):
        scene_cfg = {}
        cfg["scene"] = scene_cfg

    out_cfg = cfg.get("output")
    if not isinstance(out_cfg, dict):
        out_cfg = {}
        cfg["output"] = out_cfg

    # 1) Если в сцене явно указан output_dir — уважаем его
    if "output_dir" in scene_cfg and "dir" not in out_cfg:
        out_cfg["dir"] = scene_cfg["output_dir"]

    # 2) Если нигде не указан вывод — ставим дефолт
    if "dir" not in out_cfg:
        # папка рядом с конфигом: ./output/<имя_конфига_без_расширения>
        default_dir = (
            cfg_path.parent / "output" / cfg_path.stem
        )
        out_cfg["dir"] = str(default_dir)

    return cfg


def main(config_path: str | None = None) -> None:
    _init_logging()

    # Определяем путь к конфигу:
    #  - если явно передали в main() — используем его,
    #  - иначе смотрим на sys.argv[1],
    #  - иначе падаем на configs/conferencehall.yaml.
    if config_path is not None:
        cfg_path = Path(config_path)
    else:
        if len(sys.argv) > 1:
            cfg_path = Path(sys.argv[1])
        else:
            cfg_path = Path("configs/conferencehall.yaml")

    logging.info("Using config: %s", cfg_path)

    cfg = _load_config(cfg_path)
    cfg = _normalize_config(cfg, cfg_path)

    # Здесь больше не нужен outdir — всё делает сам run_path_from_config.
    run_path_from_config(cfg, det_fn=None)


if __name__ == "__main__":
    main()
