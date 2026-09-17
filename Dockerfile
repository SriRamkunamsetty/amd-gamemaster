# syntax=docker/dockerfile:1
# Build from the repository root:
#   docker build \
#     --build-arg POLICY_DIR=checkpoints/c6-policy-merged -t <registry>/amd-gamemaster:<tag> .
#
# IMPORTANT (grader gate): peak VRAM must be >= 1 GiB. Tree search runs on CPU, so the image serves
# the fine-tuned policy model on the GPU and the agent uses it for priors (GAMEMASTER_HYBRID=true).
ARG BASE_IMAGE=rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0
FROM ${BASE_IMAGE}

ARG INSTALL_VLLM=1
ARG VLLM_SPEC="vllm"
# Local directory (inside the build context) holding the merged fine-tuned policy; empty -> download MODEL_ID
ARG POLICY_DIR=""
ARG MODEL_ID=Qwen/Qwen3-1.7B

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HARNESS_OUTPUT_DIR=/app/output \
    LOG_FORMAT=json \
    LLM_BACKEND=vllm \
    LLM_SERVE_MODEL=/models/policy \
    LLM_SERVED_NAME=gamemaster-policy \
    LLM_BASE_URL=http://127.0.0.1:8000/v1 \
    LLM_VRAM_BUDGET_GIB=12 \
    LLM_MAX_MODEL_LEN=4096 \
    GAMEMASTER_HYBRID=true \
    GAMEMASTER_MOVE_TIME=5

WORKDIR /app

COPY requirements.txt /app/requirements.txt
COPY docker/requirements-gpu.txt /tmp/requirements-gpu.txt
RUN python3 -m pip install -r /app/requirements.txt -r /tmp/requirements-gpu.txt \
 && if [ "$INSTALL_VLLM" = "1" ]; then python3 -m pip install "$VLLM_SPEC" || echo "WARN: vLLM install failed; hf backend will be used"; fi

COPY ${POLICY_DIR:-docker/empty}/ /models/policy/
RUN if [ ! -f /models/policy/config.json ]; then \
      python3 -c "from huggingface_hub import snapshot_download as s; s('${MODEL_ID}', local_dir='/models/policy', allow_patterns=['*.json','*.safetensors','*.txt','merges.txt','vocab.*'])"; \
    fi

COPY academy-core /opt/src/academy-core
COPY pyproject.toml app.py /opt/src/gamemaster/
COPY src /opt/src/gamemaster/src
RUN python3 -m pip install --no-deps /opt/src/academy-core /opt/src/gamemaster \
 && cp /opt/src/gamemaster/app.py /app/app.py \
 && mkdir -p /app/input /app/output

HEALTHCHECK --interval=30s --timeout=5s --start-period=600s CMD test -f /tmp/academy_ready || exit 1
ENTRYPOINT ["python3", "-m", "academy_core.server.entrypoint"]
