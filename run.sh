#!/usr/bin/env sh
# Starts the stack on the GPU when Docker can hand a container this machine's
# NVIDIA GPU, and on the CPU everywhere else, then saves that choice in .env so
# a plain `docker compose up` on this machine keeps making it. Arguments go to
# docker compose; with none, it runs "up". WH_DEVICE=cpu stays on the CPU.
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

compose_files="docker-compose.yml"

if [ "${WH_DEVICE:-auto}" = "cpu" ]; then
  echo "WH_DEVICE=cpu: using the CPU build."
# Actually asking for the GPU is the only reliable test: Docker Desktop on
# Windows serves GPUs through WSL 2 and lists no "nvidia" runtime in
# `docker info`. The image is the Dockerfile's base, so this pull is not wasted.
elif docker run --rm --gpus all python:3.10-slim true >/dev/null 2>&1; then
  echo "NVIDIA GPU available to Docker: using the CUDA build."
  compose_files="$compose_files,docker-compose.gpu.yml"
elif command -v nvidia-smi >/dev/null 2>&1; then
  echo "An NVIDIA GPU is present but Docker cannot use it. Install the NVIDIA" \
       "Container Toolkit (Linux) or enable WSL 2 GPU support (Docker Desktop)." \
       "Using the CPU build."
else
  echo "No NVIDIA GPU available to Docker: using the CPU build."
fi

# Only the two lines this script owns are replaced; the rest of .env is kept.
# A comma separator reads the same on Windows and Linux.
tmp=$(mktemp)
touch .env
grep -v -e '^COMPOSE_FILE=' -e '^COMPOSE_PATH_SEPARATOR=' .env > "$tmp" || true
printf 'COMPOSE_PATH_SEPARATOR=,\nCOMPOSE_FILE=%s\n' "$compose_files" >> "$tmp"
mv "$tmp" .env

exec env COMPOSE_PATH_SEPARATOR=, COMPOSE_FILE="$compose_files" docker compose "$@"
