#!/usr/bin/env python3
"""Batched generation run with full telemetry capture.

Drives the pipeline service over N batches and records everything a post-mortem
needs: per-batch wall clock, per-prompt timing, per-candidate coder/validate
timings and token usage, every judge duel with the stage that decided it, host
preflight metrics, a GPU utilisation timeline, and the raw service log.

    python tests/run_batches.py r42.txt --out runs/r42 --batch-size 32

Layout written under --out:

    run_summary.json          aggregate stats + per-batch wall clock
    REPORT.md                 human-readable summary
    config.json               /debug/run config snapshot
    prompts.json              the exact stem -> url mapping used
    gpu_samples.csv           nvidia-smi timeline for the whole run
    pipeline.log              service log pulled from the container
    docker.log                container stdout/stderr
    batch_XX/
        batch.json            batch-level timing + /debug/run before & after
        progress.csv          progress timeline (t, done, status)
        results.zip           the service's own /results artifact
        tasks/<stem>.json     full /debug/tasks payload (meta: candidates, duels)
        renders/<stem>.png    winning 2x2 grid
        inputs/<stem>.png     reference image
        candidates/<stem>/k<NN>.png   per-candidate grids (first --candidate-pngs stems)
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import httpx

BOLD, GREEN, RED, YELLOW, CYAN, RESET = "\033[1m", "\033[32m", "\033[31m", "\033[33m", "\033[36m", "\033[0m"


def log(msg: str) -> None:
    print(f"  {CYAN}--{RESET} {msg}", flush=True)


def step(msg: str) -> None:
    print(f"\n{BOLD}[{time.strftime('%H:%M:%S')}] {msg}{RESET}", flush=True)


# --------------------------------------------------------------------------- prompts

def parse_prompts(path: Path) -> list[dict]:
    """Accept `stem url`, or a bare url whose basename becomes the stem."""
    out: list[dict] = []
    seen: set[str] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            stem, url = parts
        else:
            url = parts[0]
            stem = Path(urlparse(url).path).stem
        if stem in seen:
            print(f"{YELLOW}duplicate stem {stem}, skipping{RESET}")
            continue
        seen.add(stem)
        out.append({"stem": stem, "image_url": url})
    return out


# --------------------------------------------------------------------------- gpu sampler

class GPUSampler(threading.Thread):
    """nvidia-smi timeline for the whole run, one CSV row per GPU per tick."""

    FIELDS = ("index", "utilization.gpu", "utilization.memory", "memory.used",
              "memory.total", "power.draw", "temperature.gpu", "clocks.sm")

    def __init__(self, path: Path, interval: float = 5.0) -> None:
        super().__init__(daemon=True)
        self.path, self.interval = path, interval
        self._stop = threading.Event()
        self.rows = 0
        self.phase = "idle"

    def run(self) -> None:
        with self.path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ts", "phase", *self.FIELDS])
            while not self._stop.is_set():
                try:
                    r = subprocess.run(
                        ["nvidia-smi", f"--query-gpu={','.join(self.FIELDS)}",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=10,
                    )
                    ts = time.time()
                    for line in r.stdout.strip().splitlines():
                        w.writerow([f"{ts:.1f}", self.phase, *[c.strip() for c in line.split(",")]])
                        self.rows += 1
                    f.flush()
                except Exception:
                    pass
                self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()


# --------------------------------------------------------------------------- service

def wait_ready(client: httpx.Client, max_wait: int) -> bool:
    step(f"Waiting for the service to leave warming_up (max {max_wait}s)")
    deadline = time.time() + max_wait
    last = ""
    while time.time() < deadline:
        try:
            data = client.get("/status").json()
        except Exception as exc:
            if str(exc) != last:
                log(f"not up yet: {type(exc).__name__}")
                last = str(exc)
            time.sleep(3)
            continue
        status = data["status"]
        if status == "warming_up":
            if status != last:
                log(f"status={status}")
                last = status
            time.sleep(5)
            continue
        if status == "replace":
            print(f"{RED}service asked for REPLACE (pre-flight or probe failed){RESET}")
            return False
        log(f"{GREEN}status={status}{RESET} after {int(time.time() - (deadline - max_wait))}s")
        return True
    print(f"{RED}still warming up after {max_wait}s{RESET}")
    return False


def snapshot(client: httpx.Client, path: str) -> dict:
    try:
        return client.get(path).json()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def run_batch(client: httpx.Client, prompts: list[dict], seed: int, out: Path,
              timeout: int, sampler: GPUSampler, candidate_pngs: int,
              candidate_js: bool = True) -> dict:
    stems = [p["stem"] for p in prompts]
    out.mkdir(parents=True, exist_ok=True)

    before = snapshot(client, "/debug/run")
    step(f"Batch -> {out.name}: {len(prompts)} prompts, seed={seed}")

    t_submit = time.time()
    r = client.post("/generate", json={"prompts": prompts, "seed": seed})
    if r.status_code != 200:
        return {"error": f"POST /generate {r.status_code}: {r.text[:400]}"}
    log(f"accepted={r.json()['accepted']}")

    # Poll, recording the whole progress curve — the shape tells you whether the
    # batch is coder-bound (steady) or tail-bound (a long flat finish).
    prog_path = out / "progress.csv"
    deadline = time.time() + timeout
    last_done = -1
    timed_out = True
    with prog_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "elapsed_s", "done", "total", "status"])
        while time.time() < deadline:
            try:
                d = client.get("/status").json()
            except Exception:
                time.sleep(2)
                continue
            now = time.time()
            w.writerow([f"{now:.1f}", f"{now - t_submit:.1f}", d["progress"], d["total"], d["status"]])
            f.flush()
            if d["progress"] != last_done:
                bar_n = int(30 * d["progress"] / max(d["total"], 1))
                print(f"\r  [{'#' * bar_n}{'.' * (30 - bar_n)}] {d['progress']}/{d['total']} "
                      f"{d['status']} {now - t_submit:.0f}s   ", end="", flush=True)
                last_done = d["progress"]
            if d["status"] == "complete":
                timed_out = False
                break
            time.sleep(2)
    print()
    t_done = time.time()
    wall = t_done - t_submit
    if timed_out:
        print(f"{RED}batch timed out after {timeout}s{RESET}")
    else:
        log(f"{GREEN}complete in {wall:.1f}s ({wall / max(len(prompts),1):.1f}s/prompt){RESET}")

    after = snapshot(client, "/debug/run")

    # The service's own artifact, exactly as a validator would collect it.
    try:
        rz = client.get("/results", timeout=300.0)
        if rz.status_code == 200:
            (out / "results.zip").write_bytes(rz.content)
            log(f"results.zip {len(rz.content)/1024:.0f} KB")
    except Exception as exc:
        log(f"{YELLOW}/results failed: {type(exc).__name__}: {exc}{RESET}")

    # Per-prompt detail.
    tdir, rdir, idir, cdir = out / "tasks", out / "renders", out / "inputs", out / "candidates"
    cjsdir = out / "candidates_js"
    for d in (tdir, rdir, idir):
        d.mkdir(exist_ok=True)
    records = []
    for i, stem in enumerate(stems):
        want_cands = i < candidate_pngs
        try:
            resp = client.get(f"/debug/tasks/{stem}", timeout=300.0,
                              params={"include_candidate_pngs": str(want_cands).lower(),
                                      "include_candidate_js": str(candidate_js).lower()})
        except Exception as exc:
            records.append({"stem": stem, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if resp.status_code != 200:
            records.append({"stem": stem, "error": f"HTTP {resp.status_code}"})
            continue
        data = resp.json()

        png_b64 = data.pop("rendered_png_b64", None)
        if png_b64:
            (rdir / f"{stem}.png").write_bytes(base64.b64decode(png_b64))
        cand_b64 = data.pop("multigen_pngs_b64", None) or []
        if cand_b64 and want_cands:
            sub = cdir / stem
            sub.mkdir(parents=True, exist_ok=True)
            for k, b in enumerate(cand_b64):
                if b:
                    (sub / f"k{k:02d}.png").write_bytes(base64.b64decode(b))
        data.pop("refinement_rendered_pngs_b64", None)

        js = data.pop("js_code", None)
        data["js_chars"] = len(js) if js else 0

        # Losing candidates' source: the bracket discards it, /results only ever
        # carries the winner, so without this there is no way to ask "what did the
        # judge reject, and was it right?" after the fact.
        cjs = data.pop("candidate_js", None)
        if cjs:
            sub = cjsdir / stem
            sub.mkdir(parents=True, exist_ok=True)
            for k, code in sorted(cjs.items(), key=lambda kv: int(kv[0])):
                (sub / f"k{int(k):02d}.js").write_text(code, encoding="utf-8")

        url = data.get("image_url")
        if url and not (idir / f"{stem}.png").exists():
            try:
                ir = httpx.get(url, timeout=30.0, follow_redirects=True)
                if ir.status_code == 200:
                    (idir / f"{stem}.png").write_bytes(ir.content)
            except Exception:
                pass

        (tdir / f"{stem}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))
        records.append(data)

    batch = {
        "name": out.name,
        "n_prompts": len(prompts),
        "seed": seed,
        "submitted_at": t_submit,
        "completed_at": t_done,
        "wall_s": round(wall, 2),
        "s_per_prompt": round(wall / max(len(prompts), 1), 2),
        "timed_out": timed_out,
        "debug_run_before": before,
        "debug_run_after": after,
        "stems": stems,
    }
    (out / "batch.json").write_text(json.dumps(batch, indent=2))
    return {**batch, "records": records}


# --------------------------------------------------------------------------- analysis

def summarise(batches: list[dict]) -> dict:
    recs = [r for b in batches for r in b.get("records", []) if "error" not in r]
    errs = [r for b in batches for r in b.get("records", []) if "error" in r]

    ok = [r for r in recs if not r.get("failed")]
    failed = [r for r in recs if r.get("failed")]

    walls, coder_s, cand_total, cand_dropped = [], [], 0, 0
    drops, decided, dueltimes, tok_c, tok_p = Counter(), Counter(), [], 0, 0
    js_err = Counter()
    uniq_ratio = []

    for r in recs:
        t = (r.get("meta") or {}).get("timing") or {}
        if t.get("wall_s"):
            walls.append(t["wall_s"])
        mg = (r.get("meta") or {}).get("multigen") or {}
        for c in mg.get("candidates", []):
            cand_total += 1
            if c.get("drop_reason"):
                cand_dropped += 1
                drops[c["drop_reason"].split(":")[0]] += 1
            if c.get("coder_s"):
                coder_s.append(c["coder_s"])
            u = c.get("usage") or {}
            tok_c += int(u.get("completion_tokens") or 0)
            tok_p += int(u.get("prompt_tokens") or 0)
            for e in (c.get("js_errors") or []):
                js_err[str(e).split(":")[0][:60]] += 1
        if mg.get("K"):
            uniq_ratio.append(mg.get("unique_programs", 0) / mg["K"])
        for d in (r.get("meta") or {}).get("duels", []):
            decided[(d.get("decided_by") or "?").split("|")[0].strip()] += 1
            dueltimes.append(d.get("compare_s") or 0)

    def stat(xs: list[float]) -> dict:
        if not xs:
            return {}
        return {
            "n": len(xs), "mean": round(statistics.mean(xs), 2),
            "median": round(statistics.median(xs), 2),
            "p90": round(sorted(xs)[int(0.9 * (len(xs) - 1))], 2),
            "min": round(min(xs), 2), "max": round(max(xs), 2),
        }

    total_wall = sum(b.get("wall_s", 0) for b in batches)
    return {
        "prompts_total": len(recs) + len(errs),
        "succeeded": len(ok),
        "failed": len(failed),
        "fetch_errors": len(errs),
        "success_rate": round(len(ok) / max(len(recs) + len(errs), 1), 4),
        "total_wall_s": round(total_wall, 1),
        "batches": [{"name": b["name"], "wall_s": b.get("wall_s"),
                     "s_per_prompt": b.get("s_per_prompt"),
                     "timed_out": b.get("timed_out")} for b in batches],
        "task_wall_s": stat(walls),
        "candidate_coder_s": stat(coder_s),
        "duel_compare_s": stat(dueltimes),
        "candidates_total": cand_total,
        "candidates_dropped": cand_dropped,
        "candidate_drop_rate": round(cand_dropped / max(cand_total, 1), 4),
        "drop_reasons": dict(drops.most_common()),
        "js_error_rules": dict(js_err.most_common(15)),
        "unique_program_ratio": stat([round(x, 3) for x in uniq_ratio]),
        "duels_total": sum(decided.values()),
        "decided_by": dict(decided.most_common()),
        "coder_tokens_completion": tok_c,
        "coder_tokens_prompt": tok_p,
        "coder_output_tok_per_s": (round(tok_c / total_wall, 1) if total_wall else None),
        "failures": [{"stem": r["stem"], "reason": r.get("failure_reason")} for r in failed],
    }


def write_report(out: Path, summary: dict, cfg: dict) -> None:
    b = summary
    lines = [
        "# Generation run report", "",
        f"- prompts: **{b['prompts_total']}** in {len(b['batches'])} batches",
        f"- succeeded: **{b['succeeded']}**  failed: **{b['failed']}**  "
        f"(success rate {b['success_rate']*100:.1f}%)",
        f"- total wall clock: **{b['total_wall_s']:.0f}s**",
        f"- coder output tokens: {b['coder_tokens_completion']:,} "
        f"(~{b['coder_output_tok_per_s']} tok/s over the whole run)",
        "",
        "## Config", "",
        "```json", json.dumps(cfg.get("config", {}), indent=2), "```", "",
        "## Host", "",
        "```json", json.dumps(cfg.get("diag", {}), indent=2), "```", "",
        "## Per batch", "",
        "| batch | wall (s) | s/prompt | timed out |", "|---|---|---|---|",
    ]
    for x in b["batches"]:
        lines.append(f"| {x['name']} | {x['wall_s']} | {x['s_per_prompt']} | {x['timed_out']} |")
    lines += [
        "", "## Timing distributions", "",
        "| metric | n | mean | median | p90 | min | max |", "|---|---|---|---|---|---|---|",
    ]
    for k in ("task_wall_s", "candidate_coder_s", "duel_compare_s"):
        s = b.get(k) or {}
        if s:
            lines.append(f"| {k} | {s['n']} | {s['mean']} | {s['median']} | {s['p90']} | {s['min']} | {s['max']} |")
    lines += [
        "", "## Candidate attrition", "",
        f"- candidates generated: **{b['candidates_total']}**",
        f"- dropped before the bracket: **{b['candidates_dropped']}** "
        f"({b['candidate_drop_rate']*100:.1f}%)", "",
        "| drop reason | count |", "|---|---|",
    ]
    for k, v in b["drop_reasons"].items():
        lines.append(f"| {k} | {v} |")
    if b["js_error_rules"]:
        lines += ["", "### Validator rules that fired", "", "| rule | count |", "|---|---|"]
        for k, v in b["js_error_rules"].items():
            lines.append(f"| {k} | {v} |")
    lines += ["", "## Judge", "", f"- duels: **{b['duels_total']}**", "",
              "| decided by | count |", "|---|---|"]
    for k, v in b["decided_by"].items():
        lines.append(f"| {k or '(none)'} | {v} |")
    if b["failures"]:
        lines += ["", "## Failed prompts", "", "| stem | reason |", "|---|---|"]
        for f in b["failures"]:
            lines.append(f"| {f['stem']} | {str(f['reason'])[:160]} |")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts_file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=10006)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seed-step", type=int, default=0,
                    help="added to --seed for each subsequent batch (0 = same seed)")
    ap.add_argument("--timeout", type=int, default=14400, help="per-batch poll timeout (s)")
    ap.add_argument("--ready-timeout", type=int, default=7200)
    ap.add_argument("--candidate-pngs", type=int, default=4,
                    help="save per-candidate grids for the first N stems of each batch")
    ap.add_argument("--no-candidate-js", action="store_true",
                    help="skip saving the losing candidates' source (saved by default: "
                         "only the winner reaches /results, so this is the only record "
                         "of what the bracket rejected)")
    ap.add_argument("--container", default="", help="docker container to pull logs from")
    ap.add_argument("--gpu-interval", type=float, default=5.0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prompts = parse_prompts(Path(args.prompts_file))
    if not prompts:
        sys.exit(f"{RED}no prompts in {args.prompts_file}{RESET}")

    batches_in = [prompts[i:i + args.batch_size] for i in range(0, len(prompts), args.batch_size)]
    print(f"\n{BOLD}Run{RESET}  {len(prompts)} prompts -> {len(batches_in)} batches "
          f"of {args.batch_size}  out={out}")

    (out / "prompts.json").write_text(json.dumps(prompts, indent=2))

    sampler = GPUSampler(out / "gpu_samples.csv", args.gpu_interval)
    sampler.start()

    client = httpx.Client(base_url=f"http://{args.host}:{args.port}", timeout=60.0)
    run_started = time.time()
    results: list[dict] = []
    try:
        sampler.phase = "warmup"
        if not wait_ready(client, args.ready_timeout):
            sys.exit(1)

        cfg = snapshot(client, "/debug/run")
        (out / "config.json").write_text(json.dumps(cfg, indent=2))
        log(f"host: {cfg.get('diag_header', '?')}")

        for i, chunk in enumerate(batches_in):
            sampler.phase = f"batch_{i:02d}"
            seed = args.seed + i * args.seed_step
            results.append(run_batch(client, chunk, seed, out / f"batch_{i:02d}",
                                     args.timeout, sampler, args.candidate_pngs,
                                     candidate_js=not args.no_candidate_js))
        sampler.phase = "done"
    finally:
        sampler.stop()
        client.close()

    # Service log + container stdout, so a post-mortem has the raw stream too.
    try:
        r = httpx.get(f"http://{args.host}:{args.port}/debug/logs", timeout=300.0)
        if r.status_code == 200:
            (out / "pipeline.log").write_bytes(r.content)
            log(f"pipeline.log {len(r.content)/1024/1024:.1f} MB")
    except Exception as exc:
        log(f"{YELLOW}/debug/logs failed: {exc}{RESET}")
    if args.container:
        try:
            p = subprocess.run(["docker", "logs", args.container],
                               capture_output=True, timeout=300)
            (out / "docker.log").write_bytes(p.stdout + p.stderr)
            log(f"docker.log {(len(p.stdout)+len(p.stderr))/1024/1024:.1f} MB")
        except Exception as exc:
            log(f"{YELLOW}docker logs failed: {exc}{RESET}")

    summary = summarise(results)
    summary["run_wall_s"] = round(time.time() - run_started, 1)
    summary["started_at"] = run_started
    summary["finished_at"] = time.time()
    summary["gpu_samples"] = sampler.rows
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2))
    write_report(out, summary, json.loads((out / "config.json").read_text()))

    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"  {GREEN}{summary['succeeded']}{RESET}/{summary['prompts_total']} succeeded, "
          f"{RED}{summary['failed']}{RESET} failed")
    print(f"  total wall {summary['total_wall_s']:.0f}s | "
          f"per-batch {[b['wall_s'] for b in summary['batches']]}")
    print(f"  candidates {summary['candidates_total']}, dropped {summary['candidates_dropped']} "
          f"({summary['candidate_drop_rate']*100:.1f}%)")
    print(f"  duels {summary['duels_total']} | decided_by {summary['decided_by']}")
    print(f"  out: {out.resolve()}")


if __name__ == "__main__":
    main()
