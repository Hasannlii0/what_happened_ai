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

COPY requirements-api.txt .
RUN pip install --no-cache-dir "torch<3" torchvision --index-url ${TORCH_INDEX_URL}
RUN pip install --no-cache-dir -r requirements-api.txt

COPY . .

EXPOSE 8000
