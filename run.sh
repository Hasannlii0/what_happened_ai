#!/usr/bin/env sh
# Starts the stack on the GPU when Docker can hand a container this machine's
# NVIDIA GPU, and on the CPU everywhere else. Arguments go to docker compose;
# with none, it runs "up". Set WH_DEVICE=cpu to stay on the CPU regardless.
#
#   ./run.sh              start everything
#   ./run.sh up --build   rebuild the images first
#   ./run.sh down         stop
#
# Compose cannot make a GPU request optional -- a reservation on a machine
# without NVIDIA makes "up" fail -- so the choice is made here, before it runs.
set -e
cd "$(dirname "$0")"

[ "$#" -eq 0 ] && set -- up

files="-f docker-compose.yml"

if [ "${WH_DEVICE:-auto}" = "cpu" ]; then
  echo "WH_DEVICE=cpu: using the CPU build."
# Actually asking for the GPU is the only reliable test: Docker Desktop on
# Windows serves GPUs through WSL 2 and lists no "nvidia" runtime in
# `docker info`. The image is the Dockerfile's base, so this pull is not wasted.
elif docker run --rm --gpus all python:3.10-slim true >/dev/null 2>&1; then
  echo "NVIDIA GPU available to Docker: using the CUDA build."
  files="$files -f docker-compose.gpu.yml"
elif command -v nvidia-smi >/dev/null 2>&1; then
  echo "An NVIDIA GPU is present but Docker cannot use it. Install the NVIDIA" \
       "Container Toolkit (Linux) or enable WSL 2 GPU support (Docker Desktop)." \
       "Using the CPU build."
else
  echo "No NVIDIA GPU available to Docker: using the CPU build."
fi

# $files is deliberately unquoted: it holds several fixed flags.
exec docker compose $files "$@"
