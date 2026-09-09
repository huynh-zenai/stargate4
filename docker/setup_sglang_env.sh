#!/bin/bash
set -euo pipefail

VENV="${VENV:-/opt/sglang-env}"
PYBIN="${PYBIN:-python3.11}"
SGLANG_SHA="${SGLANG_SHA:-1cf2b8c54d81}"
TORCH_BACKEND="${TORCH_BACKEND:-cu129}"
export CARGO_HOME="${CARGO_HOME:-/opt/rustup/cargo}"
export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup/rustup}"

export PATH="$CARGO_HOME/bin:$PATH"
command -v cargo >/dev/null || {
    curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable --no-modify-path
    export PATH="$CARGO_HOME/bin:$PATH"
}

echo "[sglang-env] $VENV | sglang @ $SGLANG_SHA | torch=$TORCH_BACKEND | cargo $(cargo --version | awk '{print $2}')"
"$PYBIN" -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip uv

OVERRIDES="$(mktemp)"
echo "cuda-python<13" > "$OVERRIDES"
LOCK_FILE="${LOCK_FILE:-/tmp/sglang-env.lock.txt}"
CONSTRAINT_ARGS=()
if [ -f "$LOCK_FILE" ]; then
    CONSTRAINT_ARGS=(--constraint "$LOCK_FILE")
    echo "[sglang-env] constraining transitive deps to $LOCK_FILE"
else
    echo "[sglang-env] WARNING: $LOCK_FILE missing — transitive deps will float" >&2
fi
"$VENV/bin/uv" pip install --python "$VENV/bin/python" \
    "sglang[all] @ git+https://github.com/sgl-project/sglang.git@${SGLANG_SHA}#subdirectory=python" \
    --torch-backend "$TORCH_BACKEND" --override "$OVERRIDES" "${CONSTRAINT_ARGS[@]}"

"$VENV/bin/pip" uninstall -y sgl-deep-gemm >/dev/null 2>&1 || true
"$VENV/bin/pip" install --force-reinstall --no-deps sglang-kernel \
    --index-url https://docs.sglang.ai/whl/cu129/

"$VENV/bin/pip" install ninja

"$VENV/bin/python" - <<'PY'
import sglang, torch
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.models.dflash import DFlash2DraftModel, CandidateSelector
from sglang.srt.models.qwen3_5 import Qwen3_5ForConditionalGeneration
assert SpeculativeAlgorithm.from_string("DFLASH").is_dflash()
print(f"[sglang-env] OK — sglang {sglang.__version__} | torch {torch.__version__} | "
      f"DFlash2DraftModel + CandidateSelector + Qwen3_5ForConditionalGeneration present")
PY

echo "[sglang-env] done -> configuration.yaml coder-instance.vllm.engine_python: $VENV/bin/python"
