FROM python:3.10-slim

# CUDA is not in the base image: the cu* torch wheels ship their own CUDA
# runtime, so a GPU host only needs the nvidia container runtime.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --index-url REPLACES PyPI, and the torch index carries only torch and its
# runtime deps, so a source dep there (typing_extensions) cannot find its build
# backend and the install dies. PyPI is kept as a fallback for those. The torch
# index stays primary and its wheels carry a +cpu/+cu local version, which
# outranks PyPI's plain build, so the intended variant still wins.
# Installed before requirements-api.txt is copied in, so editing that file
# does not throw away this multi-GB layer (the CUDA build is ~4GB).
RUN pip install --no-cache-dir "torch<3" torchvision \
    --index-url ${TORCH_INDEX_URL} \
    --extra-index-url https://pypi.org/simple

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# The CUDA build of torch runs some Qwen2-VL ops through Triton, which compiles
# a small helper with a C compiler the first time a kernel launches; without
# one, every report fails over to the template. The CPU build never uses
# Triton, so it skips the ~150MB. Placed after the pip layers so changing it
# does not invalidate the multi-GB torch download.
RUN case "${TORCH_INDEX_URL}" in \
      */cpu) ;; \
      *) apt-get update \
         && apt-get install -y --no-install-recommends gcc libc6-dev \
         && rm -rf /var/lib/apt/lists/* ;; \
    esac

COPY . .

EXPOSE 8000
