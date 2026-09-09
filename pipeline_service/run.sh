#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_HARD_NOFILE="$(ulimit -Hn 2>/dev/null || echo 0)"
if [ "$_HARD_NOFILE" = "unlimited" ]; then
    ulimit -Sn 1048576 2>/dev/null || true
elif [ "${_HARD_NOFILE:-0}" -gt "$(ulimit -Sn)" ] 2>/dev/null; then
    ulimit -Sn "$_HARD_NOFILE" 2>/dev/null || true
fi
echo "[run.sh] file descriptors: soft=$(ulimit -Sn) hard=$(ulimit -Hn)"

CONFIG_FILE="${CONFIG_PATH:-/workspace/configuration.yaml}"
export CONFIG_FILE

# Thread pools: the GLM vLLM's CPU-side image preprocessing spawns nproc-sized OpenMP pools; on a small-quota
# host (10 CPUs seen on RunPod) that oversubscription made the judge's cold path 1.9x slower, while OMP=1 costs
# nothing on 20-40 CPU quotas. Inherited by every engine spawned from llm/spawn.py.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# Preflight: network, GPU count, per-GPU bf16 TFLOPS + decode-shaped weight-stream GB/s + power limit.
# Writes /tmp/preflight_metrics.json (picked up by serve.py for the miner-diag header).
if python -m modules.metrics.preflight; then
    echo "=== PRE-FLIGHT OK ==="
else
    echo "=== PRE-FLIGHT FAILED — starting FastAPI in REPLACE mode ==="
fi

echo "=== STAGE 2: FastAPI ==="
python serve.py &
SERVE_PID=$!

GLM_VLLM_BIN="${GLM_VLLM_BIN:-/opt/vllm-glm-env/bin/vllm}"
if [ ! -x "$GLM_VLLM_BIN" ]; then
    echo "=== STAGE 2.5: building GLM vLLM env ($GLM_VLLM_BIN missing) ==="
    GLM_MODEL_INFO="$(GLM_VLLM_BIN="$GLM_VLLM_BIN" python - <<'PY'
import os, yaml
try:
    cfg = yaml.safe_load(open(os.environ["CONFIG_FILE"])) or {}
except Exception:
    raise SystemExit(0)
for spec in (cfg.get("llm_clients") or {}).values():
    if not isinstance(spec, dict):
        continue
    v = spec.get("vllm") or {}
    if str(v.get("vllm_bin") or "") == os.environ["GLM_VLLM_BIN"] and str(v.get("model") or "").strip():
        print(str(v["model"]).strip())
        print(str(v.get("revision") or "").strip())
        break
PY
)"
    GLM_MODEL="$(sed -n 1p <<<"$GLM_MODEL_INFO")"
    GLM_REVISION="$(sed -n 2p <<<"$GLM_MODEL_INFO")"
    [ -n "$GLM_MODEL" ] && echo "[run.sh] GLM env build target: $GLM_MODEL @ ${GLM_REVISION:-main}"
    MODEL="$GLM_MODEL" MODEL_REVISION="$GLM_REVISION" \
        bash "$SCRIPT_DIR/scripts/setup_glm_vllm_env.sh" \
        || echo "[run.sh] GLM env setup failed — judge/critic vLLM skipped, coder continues" >&2
else
    echo "[run.sh] GLM vLLM env present at $GLM_VLLM_BIN — skipping build"
fi

CODER_ENGINE_PYTHON="${CODER_ENGINE_PYTHON:-/opt/sglang-env/bin/python}"
if [ -x "$CODER_ENGINE_PYTHON" ]; then
    echo "[run.sh] coder engine present at $CODER_ENGINE_PYTHON"
    "$CODER_ENGINE_PYTHON" -c 'import sglang; print(f"[run.sh] sglang {sglang.__version__}")' \
        || echo "[run.sh] WARNING: $CODER_ENGINE_PYTHON cannot import sglang" >&2
else
    echo "[run.sh] ERROR: coder engine missing at $CODER_ENGINE_PYTHON — the image " \
         "was built without docker/setup_sglang_env.sh. The coder will not start." >&2
fi

if [ -z "${HF_TOKEN:-}" ] && [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    export HF_TOKEN="$HUGGING_FACE_HUB_TOKEN"
fi
if [ -n "${HF_TOKEN:-}" ]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    echo "[run.sh] HF_TOKEN present (${#HF_TOKEN} chars) — private drafter reachable"
else
    echo "[run.sh] WARNING: no HF_TOKEN/HUGGING_FACE_HUB_TOKEN in the environment. " \
         "The DFlash2 drafter repo is private and the coder will fail to start. " \
         "Run the container with -e HF_TOKEN=hf_..." >&2
fi

echo "=== STAGE 3: engine spawn ==="
python -m llm.spawn || echo "[run.sh] engine spawn returned non-zero — FastAPI continues for diagnostics" >&2

wait $SERVE_PID
