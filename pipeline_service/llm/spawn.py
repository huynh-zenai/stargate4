from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

import yaml

from modules.metrics.gpu import _detect_all_gpu_ids, _largest_power_of_two_leq

_LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0")
_DEFAULT_VLLM_PORT = 8001
# Default vLLM venv (Qwen-compatible transformers). A client can override it
# with `vllm.vllm_bin` to run from a separate env — e.g. GLM-4.6V needs a newer
# transformers than the coder, so it points at /opt/vllm-glm-env/bin/vllm.
_DEFAULT_VLLM_BIN = "/opt/vllm-env/bin/vllm"
_DEFAULT_SGLANG_PYTHON = "/opt/sglang-env/bin/python"
_DEFAULT_REASONING_PARSER = "qwen3"
_AUTO_GPU_TOKENS = {"", "auto", "all"}


@dataclass(frozen=True)
class SpeculativeSpec:
    """Speculative-decoding flags for an SGLang endpoint."""
    algorithm: str
    draft_model: str
    draft_revision: str | None = None
    num_draft_tokens: int | None = None
    num_steps: int | None = None
    eagle_topk: int | None = None


@dataclass(frozen=True)
class VllmJob:
    name: str
    model: str
    revision: str | None
    port: int
    gpu_ids: str
    tp: int
    gpu_util: float
    max_len: int
    max_seqs: int
    api_key: str
    vllm_bin: str
    trust_remote_code: bool
    reasoning_parser: str | None
    extra_args: tuple[str, ...]
    dp: int = 1
    engine: str = "vllm"
    engine_python: str = _DEFAULT_SGLANG_PYTHON
    speculative: SpeculativeSpec | None = None
    env: tuple[tuple[str, str], ...] = ()

    @property
    def launcher(self) -> str:
        """The executable whose presence gates this job."""
        return self.engine_python if self.engine == "sglang" else self.vllm_bin


@dataclass
class _RawSpec:
    """Per-client spec before cross-client GPU allocation."""
    name: str
    model: str
    revision: str | None
    port: int
    gpu_util: float
    max_len: int
    max_seqs: int
    api_key: str
    explicit_ids: list[str] | None
    explicit_tp: int | None
    vllm_bin: str
    trust_remote_code: bool
    reasoning_parser: str | None
    extra_args: tuple[str, ...]
    data_parallel: int = 1
    engine: str = "vllm"
    engine_python: str = _DEFAULT_SGLANG_PYTHON
    speculative: SpeculativeSpec | None = None
    env: tuple[tuple[str, str], ...] = ()

# Check if the URL is a local host
def _is_local(url: str | None) -> bool:
    u = (url or "").lower()
    return any(h in u for h in _LOCAL_HOSTS)


# Get the port from the base URL
def _port_from_base_url(base_url: str | None, override: Any) -> int:
    if override is not None:
        return int(override)
    m = re.search(r":(\d+)(?:/|$)", base_url or "")
    return int(m.group(1)) if m else _DEFAULT_VLLM_PORT


# Parse the explicit GPU IDs from the YAML
def _parse_explicit_ids(raw: Any) -> list[str] | None:
    """Return list of GPU IDs from YAML, or None if auto/missing."""
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in _AUTO_GPU_TOKENS:
        return None
    return [x.strip() for x in str(raw).split(",") if x.strip()]

# Parse the explicit tensor parallel size from the YAML
def _parse_explicit_tp(raw: Any) -> int | None:
    if raw is None:
        return None
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


# Resolve the reasoning parser: absent -> default ("qwen3"); explicit null/empty -> None (omit flag).
def _parse_reasoning_parser(raw: Any) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip()
    return s or None

def _parse_engine(raw: Any) -> str:
    engine = (str(raw).strip().lower() if raw is not None else "") or "vllm"
    if engine not in ("vllm", "sglang"):
        raise ValueError(f"unknown engine {engine!r}; expected 'vllm' or 'sglang'")
    return engine


def _parse_speculative(raw: Any) -> SpeculativeSpec | None:
    if not isinstance(raw, dict) or not raw:
        return None
    draft_model = str(raw.get("draft_model") or "").strip()
    if not draft_model:
        raise ValueError("speculative.draft_model is required when speculative is set")

    def _opt_int(key: str) -> int | None:
        val = raw.get(key)
        return int(val) if val is not None else None

    revision = raw.get("draft_revision")
    return SpeculativeSpec(
        algorithm=str(raw.get("algorithm") or "DFLASH").strip().upper(),
        draft_model=draft_model,
        draft_revision=(str(revision).strip() or None) if revision is not None else None,
        num_draft_tokens=_opt_int("num_draft_tokens"),
        num_steps=_opt_int("num_steps"),
        eagle_topk=_opt_int("eagle_topk"),
    )


# Collect the raw specifications from the YAML
def _collect_raw_specs(cfg: dict[str, Any]) -> list[_RawSpec]:
    llm = cfg.get("llm_clients") or {}
    specs: list[_RawSpec] = []
    for name, spec in llm.items():
        if not isinstance(spec, dict):
            continue
        if spec.get("enabled", True) is False:
            continue
        v = spec.get("vllm") or {}
        if not isinstance(v, dict):
            continue
        model = (v.get("model") or "").strip()
        if not model:
            continue
        base = spec.get("base_url") or ""
        if not _is_local(base):
            continue

        specs.append(_RawSpec(
            name=name,
            model=model,
            revision=(str(v.get("revision")).strip() or None) if v.get("revision") is not None else None,
            port=_port_from_base_url(base, v.get("port")),
            gpu_util=float(v.get("gpu_memory_utilization", 0.90)),
            max_len=int(v.get("max_model_len", 8192)),
            max_seqs=int(v.get("max_num_seqs", 4)),
            api_key=str(v.get("api_key", "local")),
            explicit_ids=_parse_explicit_ids(v.get("gpu_ids")),
            explicit_tp=_parse_explicit_tp(v.get("tensor_parallel_size")),
            vllm_bin=str(v.get("vllm_bin") or _DEFAULT_VLLM_BIN),
            trust_remote_code=bool(v.get("trust_remote_code", False)),
            reasoning_parser=_parse_reasoning_parser(
                v.get("reasoning_parser", _DEFAULT_REASONING_PARSER)
            ),
            extra_args=tuple(str(x) for x in (v.get("extra_args") or [])),
            data_parallel=int(v.get("data_parallel_size", 1) or 1),
            engine=_parse_engine(v.get("engine")),
            engine_python=str(v.get("engine_python") or _DEFAULT_SGLANG_PYTHON),
            speculative=_parse_speculative(v.get("speculative")),
            env=tuple(
                (str(k), str(val))
                for k, val in (v.get("env") or {}).items()
            ),
        ))
    return specs


# Allocate the GPUs to the specifications
def _allocate_gpus(specs: list[_RawSpec], all_gpus: list[str]) -> dict[str, list[str]]:
    """
    Input:
        specs: list of raw specifications
        all_gpus: list of all visible GPUs
    Output:
        assigned: dictionary of client names to list of GPU IDs
    """
    assigned: dict[str, list[str]] = {}
    used: set[str] = set()

    # Phase 1: explicit gpu_ids
    for s in specs:
        if s.explicit_ids is None:
            continue
        for g in s.explicit_ids:
            if g not in all_gpus:
                raise ValueError(
                    f"{s.name}: gpu_ids includes {g!r} but visible GPUs are {all_gpus}"
                )
            if g in used:
                raise ValueError(
                    f"{s.name}: GPU {g} already reserved by another client"
                )
            used.add(g)
        assigned[s.name] = list(s.explicit_ids)

    # Phase 2: auto gpu_ids + explicit tp
    free = [g for g in all_gpus if g not in used]
    for s in specs:
        if s.explicit_ids is not None:
            continue
        if s.explicit_tp is None:
            continue
        if len(free) < s.explicit_tp:
            raise ValueError(
                f"{s.name}: tensor_parallel_size={s.explicit_tp} but only "
                f"{len(free)} GPU(s) free after explicit reservations"
            )
        assigned[s.name] = free[: s.explicit_tp]
        free = free[s.explicit_tp:]

    # Phase 3: fully auto — split remaining evenly
    auto_specs = [s for s in specs if s.name not in assigned]
    if auto_specs:
        if len(free) < len(auto_specs):
            raise ValueError(
                f"{len(auto_specs)} auto client(s) but only {len(free)} GPU(s) "
                f"free — specify gpu_ids explicitly or remove a client"
            )
        per = len(free) // len(auto_specs)
        remainder = len(free) % len(auto_specs)
        idx = 0
        for i, s in enumerate(auto_specs):
            n = per + (1 if i < remainder else 0)
            assigned[s.name] = free[idx: idx + n]
            idx += n

    return assigned


def _finalize_jobs(specs: list[_RawSpec], assigned: dict[str, list[str]]) -> list[VllmJob]:
    """Decide the tensor parallel size for each job
    Input:
        specs: list of raw specifications
        assigned: dictionary of client names to list of GPU IDs
    Output:
        jobs: list of vLLM jobs
    """
    pow2 = os.environ.get("VLLM_TP_POWER_OF_TWO", "").strip() in ("1", "true", "yes")
    jobs: list[VllmJob] = []
    for s in specs:
        ids = assigned[s.name]
        # With data parallelism the client owns tp * dp GPUs: dp independent
        # replicas, each sharded over tp cards.
        dp = max(1, s.data_parallel)
        if s.explicit_tp is not None:
            if s.explicit_tp * dp != len(ids):
                raise ValueError(
                    f"{s.name}: tensor_parallel_size={s.explicit_tp} * "
                    f"data_parallel_size={dp} = {s.explicit_tp * dp} but "
                    f"{len(ids)} GPU(s) assigned ({ids})"
                )
            tp = s.explicit_tp
        else:
            tp = max(1, len(ids) // dp)
            if pow2:
                tp = _largest_power_of_two_leq(tp)
        jobs.append(VllmJob(
            name=s.name, model=s.model, revision=s.revision, port=s.port,
            gpu_ids=",".join(ids), tp=tp, dp=dp,
            gpu_util=s.gpu_util, max_len=s.max_len,
            max_seqs=s.max_seqs, api_key=s.api_key,
            vllm_bin=s.vllm_bin, trust_remote_code=s.trust_remote_code,
            reasoning_parser=s.reasoning_parser, extra_args=s.extra_args,
            engine=s.engine, engine_python=s.engine_python,
            speculative=s.speculative, env=s.env,
        ))
    return jobs


def _build_jobs(cfg: dict[str, Any]) -> list[VllmJob]:
    """Build the vLLM jobs from the configuration
    Input:
        cfg: dictionary of configuration
    Output:
        jobs: list of vLLM jobs
    """
    # Collect the raw specifications from the YAML
    specs = _collect_raw_specs(cfg)
    if not specs:
        return []
    # Detect all visible GPUs
    all_gpus = _detect_all_gpu_ids()
    # Allocate the GPUs to the specifications
    assigned = _allocate_gpus(specs, all_gpus)
    # Decide the tensor parallel size for each job
    jobs = _finalize_jobs(specs, assigned)

    print(
        f"[vllm-spawn] Visible GPUs: {all_gpus} | Local Clients: "
        f"{[s.name for s in specs]}",
        flush=True,
    )
    return jobs


def _build_cmd(job: VllmJob) -> list[str]:
    """Dispatch to the engine's argv builder."""
    if job.engine == "sglang":
        return _build_sglang_cmd(job)
    return _build_vllm_cmd(job)


def _build_vllm_cmd(job: VllmJob) -> list[str]:
    cmd = [
        job.vllm_bin, "serve", job.model,
        "--port", str(job.port),
        "--api-key", job.api_key,
        "--max-model-len", str(job.max_len),
        "--tensor-parallel-size", str(job.tp),
        *(["--data-parallel-size", str(job.dp)] if job.dp > 1 else []),
        "--gpu-memory-utilization", str(job.gpu_util),
        "--max_num_seqs", str(job.max_seqs),
        "--generation-config", "vllm",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--max-num-batched-tokens", "8192",
    ]
    if job.revision:
        cmd += ["--revision", job.revision, "--tokenizer-revision", job.revision]
    if job.reasoning_parser:
        cmd += ["--reasoning-parser", job.reasoning_parser]
    if job.trust_remote_code:
        cmd.append("--trust-remote-code")
    cmd += list(job.extra_args)
    return cmd


def _build_sglang_cmd(job: VllmJob) -> list[str]:
    """SGLang argv, flag-for-flag with the vLLM builder above.

        vllm serve <model>                ->  --model-path <model>
        --max-model-len N                 ->  --context-length N
        --tensor-parallel-size N          ->  --tp-size N
        --data-parallel-size N            ->  --dp-size N
        --gpu-memory-utilization F        ->  --mem-fraction-static F
        --max_num_seqs N                  ->  --max-running-requests N
        --enable-prefix-caching           ->  RadixAttention, on unless
                                              --disable-radix-cache is passed
        --enable-chunked-prefill
          + --max-num-batched-tokens 8192 ->  --chunked-prefill-size 8192
        --revision / --api-key
          / --reasoning-parser            ->  same names
        --generation-config vllm          ->  no equivalent and none needed:
                                              SGLang does not apply the
                                              checkpoint's generation_config.json
                                              defaults, and the agents send every
                                              sampling param explicitly anyway.

    The bind host stays 127.0.0.1: every consumer is in-container (the pipeline
    talks to localhost:PORT) and the container is normally run with
    --network host, where 0.0.0.0 would expose the engine on the node.
    """
    cmd = [
        job.engine_python, "-m", "sglang.launch_server",
        "--model-path", job.model,
        "--host", "127.0.0.1",
        "--port", str(job.port),
        "--api-key", job.api_key,
        "--context-length", str(job.max_len),
        "--tp-size", str(job.tp),
        *(["--dp-size", str(job.dp)] if job.dp > 1 else []),
        "--mem-fraction-static", str(job.gpu_util),
        "--max-running-requests", str(job.max_seqs),
        "--chunked-prefill-size", "8192",
    ]
    if job.revision:
        cmd += ["--revision", job.revision]
    if job.reasoning_parser:
        cmd += ["--reasoning-parser", job.reasoning_parser]
    if job.trust_remote_code:
        cmd.append("--trust-remote-code")

    spec = job.speculative
    if spec is not None:
        cmd += [
            "--speculative-algorithm", spec.algorithm,
            "--speculative-draft-model-path", spec.draft_model,
        ]
        # SGLang takes one --revision for the whole server, so the drafter's
        # own pin needs its dedicated flag.
        if spec.draft_revision:
            cmd += ["--speculative-draft-model-revision", spec.draft_revision]
        if spec.num_draft_tokens is not None:
            cmd += ["--speculative-num-draft-tokens", str(spec.num_draft_tokens)]
        if spec.num_steps is not None:
            cmd += ["--speculative-num-steps", str(spec.num_steps)]
        if spec.eagle_topk is not None:
            cmd += ["--speculative-eagle-topk", str(spec.eagle_topk)]

    cmd += list(job.extra_args)
    return cmd


def _venv_nvidia_lib_dirs(engine_python: str) -> list[str]:
    """`<venv>/lib/python*/site-packages/nvidia/*/lib` dirs, sorted, or []."""
    venv = os.path.dirname(os.path.dirname(engine_python))
    pattern = os.path.join(venv, "lib", "python*", "site-packages", "nvidia", "*", "lib")
    return sorted(d for d in glob.glob(pattern) if os.path.isdir(d))


def _bin_exists(path: str) -> bool:
    """True if the vLLM binary is runnable (absolute/relative path or on PATH)."""
    if "/" in path:
        return os.path.isfile(path) and os.access(path, os.X_OK)
    return shutil.which(path) is not None


def _spawn_one(job: VllmJob) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = job.gpu_ids
    env.update(dict(job.env))
    if job.engine == "sglang":
        # flashinfer JIT-compiles its sampling kernels at engine init and dies
        # with FileNotFoundError: 'ninja' unless the venv's bin is on PATH.
        env["PATH"] = os.path.dirname(job.engine_python) + os.pathsep + env.get("PATH", "")
        nvidia_libs = _venv_nvidia_lib_dirs(job.engine_python)
        if nvidia_libs:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(
                nvidia_libs + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else [])
            )
    cmd = _build_cmd(job)
    spec = job.speculative
    spec_note = (
        f"{spec.algorithm}:{spec.draft_model}@{spec.draft_revision or 'main'}"
        if spec is not None else "-"
    )
    print(
        f"[vllm-spawn] Starting {job.engine} Client: {job.name} | Model: {job.model} | "
        f"Revision: {job.revision or 'main'} | Port: {job.port} | "
        f"GPUs: {job.gpu_ids} | Tensor Parallel Size: {job.tp} | "
        f"Data Parallel Size: {job.dp} | bin: {job.launcher} | "
        f"speculative: {spec_note} | "
        f"env: {','.join(k for k, _ in job.env) or '-'} | "
        f"reasoning_parser: {job.reasoning_parser or '-'} | trust_remote_code: {job.trust_remote_code}",
        flush=True,
    )
    return subprocess.Popen(cmd, env=env, start_new_session=True)


def main() -> int:
    path = os.environ.get("CONFIG_FILE", "/workspace/configuration.yaml")
    try:
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"[vllm-spawn] Configuration file not found: {path}", file=sys.stderr)
        return 1
    except yaml.YAMLError as e:
        print(f"[vllm-spawn] Invalid YAML in {path}: {e}", file=sys.stderr)
        return 1

    try:
        jobs = _build_jobs(cfg)
    except ValueError as e:
        print(f"[vllm-spawn] Error: {e}", file=sys.stderr)
        return 1

    if not jobs:
        print("[vllm-spawn] No local vLLM clients configured — nothing to start", flush=True)
        return 0

    # Spawn each job independently: a missing binary or launch failure for one
    # client must NOT abort the others (e.g. GLM env not built yet should still
    # let the coder come up).
    failures = 0
    for j in jobs:
        if not _bin_exists(j.launcher):
            hint = (
                "build the env (e.g. `bash docker/setup_sglang_env.sh`) or fix "
                "`vllm.engine_python` in the config"
                if j.engine == "sglang" else
                "build the env (e.g. `bash scripts/setup_glm_vllm_env.sh`) or fix "
                "`vllm.vllm_bin` in the config"
            )
            print(
                f"[vllm-spawn] ERROR {j.name}: {j.engine} launcher not found: "
                f"{j.launcher!r} — {hint}. Skipping this client.",
                file=sys.stderr,
            )
            failures += 1
            continue
        try:
            _spawn_one(j)
        except Exception as e:  # noqa: BLE001 - one client's failure must not kill the rest
            print(f"[vllm-spawn] ERROR {j.name}: spawn failed: {e}", file=sys.stderr)
            failures += 1

    launched = len(jobs) - failures
    print(
        f"[vllm-spawn] {launched}/{len(jobs)} vLLM instances launched in background",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
