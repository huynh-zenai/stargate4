# syntax=docker/dockerfile:1.7
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel@sha256:a7103283ea7113e10ae5d014bd2342acebda0bc53164b2f7b1dd6eb7a766bdb6

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0" \
    CUDA_HOME=/usr/local/cuda-12.8 \
    PATH=/usr/local/cuda-12.8/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/lib64:$LD_LIBRARY_PATH \
    PUPPETEER_CACHE_DIR=/opt/puppeteer-cache \
    NODE_MAJOR=20 \
    RENDER_POOL_SIZE=32

ENV FLASHINFER_CACHE_DIR=/root/.cache/huggingface/.engine/flashinfer \
    TORCHINDUCTOR_CACHE_DIR=/root/.cache/huggingface/.engine/inductor \
    TRITON_CACHE_DIR=/root/.cache/huggingface/.engine/triton \
    SGLANG_CACHE_DIR=/root/.cache/huggingface/.engine/sglang

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates curl wget git build-essential bash gnupg \
        ninja-build \
        fonts-liberation fonts-noto-color-emoji \
        libnss3 libatk-bridge2.0-0 libx11-xcb1 libxcomposite1 \
        libxdamage1 libxrandr2 libgbm1 libasound2 \
        libpangocairo-1.0-0 libgtk-3-0 \
        libegl1 libgl1-mesa-dri libgles2 \
        xvfb; \
    curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" | bash -; \
    apt-get install -y --no-install-recommends nodejs; \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    speedtest-cli \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt && \
    python - <<'PY'
import fastapi, httpx, loguru, numpy, openai, PIL, pydantic, requests, torch, transformers, yaml
from transformers import AutoImageProcessor, AutoModel
print(f"[app-env] OK — torch {torch.__version__} | transformers {transformers.__version__} | "
      f"numpy {numpy.__version__} | pydantic {pydantic.VERSION}")
PY

COPY docker/setup_sglang_env.sh /tmp/setup_sglang_env.sh
COPY docker/sglang-env.lock.txt /tmp/sglang-env.lock.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    bash /tmp/setup_sglang_env.sh && \
    rm -rf /opt/rustup /tmp/setup_sglang_env.sh /tmp/sglang-env.lock.txt

ARG GLM_MODEL=zai-org/GLM-4.6V-Flash
ARG GLM_REVISION=411bb4d77144a3f03accbf4b780f5acb8b7cde4e
COPY pipeline_service/scripts/setup_glm_vllm_env.sh /tmp/setup_glm_vllm_env.sh
RUN --mount=type=cache,target=/root/.cache/uv \
    SKIP_CUDA_CHECK=1 \
    VENV=/opt/vllm-glm-env \
    MODEL="$GLM_MODEL" \
    MODEL_REVISION="$GLM_REVISION" \
    bash /tmp/setup_glm_vllm_env.sh && \
    rm -f /tmp/setup_glm_vllm_env.sh

WORKDIR /workspace
COPY docker/package.json docker/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci --no-audit --no-fund

COPY pipeline_service     /workspace
COPY configuration.yaml   /workspace/configuration.yaml
RUN chmod +x /workspace/run.sh

EXPOSE 10006 8001 8002
CMD ["/workspace/run.sh"]
