# Dockerfile (CUDA + build toolchain + headless rendering deps)
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ninja-build cmake git ca-certificates curl ffmpeg \
    python3 python3-pip python3-venv python3-dev \
    # headless OpenGL/EGL stack for Open3D:
    libosmesa6 libosmesa6-dev libegl1 libgl1-mesa-glx mesa-utils \
 && rm -rf /var/lib/apt/lists/*

# Python deps (keep wheels small; no seaborn etc.)
RUN python3 -m pip install --no-cache-dir --upgrade pip \
 && python3 -m pip install --no-cache-dir \
    numpy plyfile opencv-python-headless ultralytics ffmpeg-python tqdm pyyaml

# Torch 2.5.1 + cu121 to match container’s toolkit
RUN python3 - <<'PY'
import sys, subprocess
pkgs = [
  "torch==2.5.1+cu121", "torchvision==0.20.1+cu121", "torchaudio==2.5.1+cu121",
  "--index-url", "https://download.pytorch.org/whl/cu121"
]
subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", *pkgs])
PY

# gsplat (builds CUDA extensions against toolkit)
RUN python3 -m pip install --no-cache-dir gsplat

ENV PYTHONUNBUFFERED=1 \
    EGL_PLATFORM=surfaceless \
    OPEN3D_RENDERING_BACKEND=OSMESA
WORKDIR /app

CMD ["/bin/bash"]
