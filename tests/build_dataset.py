#!/usr/bin/env python3
"""Continuous dataset generation: coder + validator only, no judge.

Walks a prompt-URL file in chunks forever (or until --max-prompts / Ctrl-C),
keeping every candidate program that passes the JS validator. With
`actors.judge.enabled: false` and `ensemble_size: 2`, both candidates per prompt
are generated and validated; the pipeline promotes the first live one, and the
other is recovered through /debug/tasks?include_candidate_js=true.

    python tests/build_dataset.py /home/ubuntu/image_prompts.txt --out /home/ubuntu/datasets/sh

Resumable: `state.json` records which prompts are done, so re-running skips them.
Stopping is safe at any point — each chunk is fully flushed before the next starts.

Layout:
    state.json                    {done: [...], chunks: N, stats: {...}}
    manifest.jsonl                one row per kept program (append-only)
    chunk_0000/
        tasks/<stem>.json         full telemetry for the prompt
        js/<stem>.k00.js          every candidate that validated
        renders/<stem>.png        winner's 2x2 grid
    prompts_index.json            stem -> url for everything attempted
"""
from __future__ import annotations

import argparse
import base64
import json
import signal
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

BOLD, GREEN, RED, YELLOW, CYAN, RESET = "\033[1m", "\033[32m", "\033[31m", "\033[33m", "\033[36m", "\033[0m"

_stop = False


def _handle_sigint(signum, frame):
    global _stop
    if _stop:
        print(f"\n{RED}second interrupt — exiting now{RESET}")
        sys.exit(130)
    _stop = True
    print(f"\n{YELLOW}stop requested — finishing the current chunk, then exiting cleanly{RESET}")


def log(m: str) -> None:
    print(f"  {CYAN}--{RESET} {m}", flush=True)


def step(m: str) -> None:
    print(f"\n{BOLD}[{time.strftime('%H:%M:%S')}] {m}{RESET}", flush=True)


def parse_prompts(path: Path) -> list[dict]:
    out, seen = [], set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        url = parts[1] if len(parts) == 2 else parts[0]
        stem = parts[0] if len(parts) == 2 else Path(urlparse(url).path).stem
        if stem in seen:
            continue
        seen.add(stem)
        out.append({"stem": stem, "image_url": url})
    return out


def wait_ready(client: httpx.Client, max_wait: int) -> bool:
    step(f"Waiting for the service (max {max_wait}s)")
    deadline = time.time() + max_wait
    last = ""
    while time.time() < deadline:
        try:
            st = client.get("/status").json()["status"]
        except Exception as e:
            if type(e).__name__ != last:
                log(f"not up yet: {type(e).__name__}")
                last = type(e).__name__
            time.sleep(5)
            continue
        if st == "warming_up":
            if st != last:
                log("status=warming_up")
                last = st
            time.sleep(10)
            continue
        if st == "replace":
            print(f"{RED}service asked for REPLACE{RESET}")
            return False
        log(f"{GREEN}status={st}{RESET}")
        return True
    return False


def run_chunk(client: httpx.Client, prompts: list[dict], seed: int, out: Path,
              timeout: int) -> dict:
    """Submit one chunk, wait for it, and pull every validated program."""
    stems = [p["stem"] for p in prompts]
    tdir, jdir, rdir = out / "tasks", out / "js", out / "renders"
    for d in (tdir, jdir, rdir):
        d.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    r = client.post("/generate", json={"prompts": prompts, "seed": seed})
    if r.status_code != 200:
        return {"error": f"POST /generate {r.status_code}: {r.text[:300]}"}

    deadline = time.time() + timeout
    last = -1
    timed_out = True
    while time.time() < deadline:
        try:
            d = client.get("/status").json()
        except Exception:
            time.sleep(2)
            continue
        if d["progress"] != last:
            n = int(30 * d["progress"] / max(d["total"], 1))
            print(f"\r  [{'#' * n}{'.' * (30 - n)}] {d['progress']}/{d['total']} "
                  f"{time.time() - t0:.0f}s   ", end="", flush=True)
            last = d["progress"]
        if d["status"] == "complete":
            timed_out = False
            break
        time.sleep(2)
    print()
    wall = time.time() - t0

    kept, failed, rows = 0, 0, []
    for stem in stems:
        try:
            resp = client.get(f"/debug/tasks/{stem}", timeout=300.0,
                              params={"include_candidate_pngs": "false",
                                      "include_candidate_js": "true"})
        except Exception as exc:
            failed += 1
            rows.append({"stem": stem, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if resp.status_code != 200:
            failed += 1
            rows.append({"stem": stem, "error": f"HTTP {resp.status_code}"})
            continue
        data = resp.json()

        png = data.pop("rendered_png_b64", None)
        if png:
            (rdir / f"{stem}.png").write_bytes(base64.b64decode(png))
        data.pop("multigen_pngs_b64", None)
        data.pop("refinement_rendered_pngs_b64", None)
        cjs = data.pop("candidate_js", None) or {}
        winner_js = data.pop("js_code", None)

        mg = (data.get("meta") or {}).get("multigen") or {}
        cand_by_k = {c["k"]: c for c in mg.get("candidates", [])}

        # Keep every candidate the validator passed — that is the dataset.
        # A candidate is usable when it produced code, cleared js_checker, and
        # rendered; drop_reason is None exactly in that case.
        for k_s, code in sorted(cjs.items(), key=lambda kv: int(kv[0])):
            k = int(k_s)
            c = cand_by_k.get(k, {})
            if c.get("drop_reason") is not None:
                continue
            p = jdir / f"{stem}.k{k:02d}.js"
            p.write_text(code, encoding="utf-8")
            kept += 1
            rows.append({
                "stem": stem, "k": k, "seed": c.get("seed"),
                "image_url": data.get("image_url"),
                "js": str(p.relative_to(out.parent)),
                "code_chars": c.get("code_chars"), "code_sha": c.get("code_sha"),
                "coder_s": c.get("coder_s"), "validate_s": c.get("validate_s"),
                "render_ms": c.get("render_ms"),
                "completion_tokens": (c.get("usage") or {}).get("completion_tokens"),
                "js_valid": c.get("js_valid"), "winner": k == mg.get("winner_k"),
            })
        if not cjs and winner_js and not data.get("failed"):
            # Fallback for an image built before include_candidate_js existed.
            p = jdir / f"{stem}.k00.js"
            p.write_text(winner_js, encoding="utf-8")
            kept += 1
            rows.append({"stem": stem, "k": 0, "js": str(p.relative_to(out.parent)),
                         "image_url": data.get("image_url"), "winner": True})

        data["js_chars"] = len(winner_js) if winner_js else 0
        (tdir / f"{stem}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))

    return {"n_prompts": len(prompts), "wall_s": round(wall, 1), "kept": kept,
            "task_errors": failed, "timed_out": timed_out, "rows": rows}


def main() -> None:
    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    ap = argparse.ArgumentParser()
    ap.add_argument("prompts_file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=10006)
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=3987009184)
    ap.add_argument("--seed-step", type=int, default=7919,
                    help="added per chunk so chunks never share a seed stream")
    ap.add_argument("--timeout", type=int, default=14400)
    ap.add_argument("--ready-timeout", type=int, default=7200)
    ap.add_argument("--max-prompts", type=int, default=0, help="0 = no limit")
    ap.add_argument("--max-hours", type=float, default=0.0, help="0 = no limit")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    state_p = out / "state.json"
    state = json.loads(state_p.read_text()) if state_p.exists() else {
        "done": [], "chunks": 0, "kept": 0, "prompts_attempted": 0, "wall_s": 0.0}
    done = set(state["done"])

    allp = parse_prompts(Path(args.prompts_file))
    (out / "prompts_index.json").write_text(
        json.dumps({p["stem"]: p["image_url"] for p in allp}, indent=1))
    todo = [p for p in allp if p["stem"] not in done]
    print(f"\n{BOLD}Dataset build{RESET}  {len(allp)} prompts, {len(done)} already done, "
          f"{len(todo)} to go  chunk={args.chunk_size}  out={out}")

    client = httpx.Client(base_url=f"http://{args.host}:{args.port}", timeout=60.0)
    if not wait_ready(client, args.ready_timeout):
        sys.exit(1)

    manifest = (out / "manifest.jsonl").open("a")
    started = time.time()
    attempted = 0
    try:
        while todo and not _stop:
            if args.max_prompts and attempted >= args.max_prompts:
                log(f"reached --max-prompts {args.max_prompts}")
                break
            if args.max_hours and (time.time() - started) / 3600 >= args.max_hours:
                log(f"reached --max-hours {args.max_hours}")
                break

            chunk, todo = todo[:args.chunk_size], todo[args.chunk_size:]
            idx = state["chunks"]
            seed = args.seed + idx * args.seed_step
            cdir = out / f"chunk_{idx:04d}"
            step(f"chunk {idx} — {len(chunk)} prompts, seed={seed} "
                 f"(kept so far: {state['kept']})")

            res = run_chunk(client, chunk, seed, cdir, args.timeout)
            if "error" in res:
                print(f"{RED}{res['error']}{RESET}")
                break

            for row in res["rows"]:
                manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
            manifest.flush()

            attempted += res["n_prompts"]
            state["chunks"] = idx + 1
            state["kept"] += res["kept"]
            state["prompts_attempted"] += res["n_prompts"]
            state["wall_s"] = round(state["wall_s"] + res["wall_s"], 1)
            state["done"] = sorted(done | {p["stem"] for p in chunk})
            done = set(state["done"])
            state_p.write_text(json.dumps(state))

            rate = res["kept"] / max(res["wall_s"], 1e-9)
            log(f"{GREEN}chunk {idx}: kept {res['kept']} programs from {res['n_prompts']} "
                f"prompts in {res['wall_s']:.0f}s ({rate * 3600:.0f} programs/h){RESET}"
                + (f"  {RED}errors={res['task_errors']}{RESET}" if res["task_errors"] else ""))
    finally:
        manifest.close()
        client.close()
        state_p.write_text(json.dumps(state))

    total_h = (time.time() - started) / 3600
    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"  chunks: {state['chunks']}  prompts: {state['prompts_attempted']}  "
          f"kept programs: {GREEN}{state['kept']}{RESET}")
    print(f"  this session: {total_h:.2f}h  |  cumulative generate wall: {state['wall_s']:.0f}s")
    print(f"  out: {out.resolve()}   (re-run the same command to resume)")


if __name__ == "__main__":
    main()
