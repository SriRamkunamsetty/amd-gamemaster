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
ARG PIP_EXTRA_ARGS=""
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

# The base image ships a ROCm build of torch. Installing almost any torch-dependent package
# (transformers, accelerate, vLLM, ...) can make pip silently replace it with a CUDA build, which
# then fails at runtime with errors that never mention torch (the official challenge doc measured
# exactly this). Pin the whole torch stack so pip refuses -- loudly, at build time -- instead.
RUN python3 -m pip freeze | grep -iE '^(torch|torchvision|torchaudio|triton|pytorch-triton-rocm)==' > /etc/torch-constraints.txt; \
    cat /etc/torch-constraints.txt
ENV PIP_CONSTRAINT=/etc/torch-constraints.txt

COPY requirements.txt /app/requirements.txt
COPY docker/requirements-gpu.txt /tmp/requirements-gpu.txt
RUN python3 -m pip install -r /app/requirements.txt -r /tmp/requirements-gpu.txt \
 && if [ "$INSTALL_VLLM" = "1" ]; then python3 -m pip install $PIP_EXTRA_ARGS "$VLLM_SPEC" || echo "WARN: vLLM install failed; hf backend will be used"; fi

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

# Fail the build if anything above swapped the ROCm torch for a CUDA build (official checklist item).
RUN python3 -c "import sys, torch; print('torch', torch.__version__, 'hip', torch.version.hip); sys.exit(0 if torch.version.hip else 1)"

HEALTHCHECK --interval=30s --timeout=5s --start-period=600s CMD test -f /tmp/academy_ready || exit 1
ENTRYPOINT ["python3", "-m", "academy_core.server.entrypoint"]
