#!/usr/bin/env python3
"""Repoint every model reference in this repo at the cont1037 mirrors.

Run after mirror_models.py has created the mirrors — it reads the new commit
SHAs from mirrored_models.json, because a mirror gets its own history and the
upstream revision pins do not carry over.

    python tools/repoint_to_cont.py --map /home/ubuntu/mirrored_models.json

Touches:
    configuration.yaml                              judge/critic model + revision, embedder
    pipeline_service/modules/judge/embedder_settings.py   default model_id + revision
    pipeline_service/modules/judge/multi_stage.py         MODEL served-name constant
    docker/Dockerfile                               ARG GLM_MODEL / GLM_REVISION
    pipeline_service/scripts/setup_glm_vllm_env.sh  MODEL / fallback default
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GLM_SRC = "zai-org/GLM-4.6V-Flash"
GLM_DST = "cont1037/GLM-4.6V-Flash"
GLM_SRC_REV = "411bb4d77144a3f03accbf4b780f5acb8b7cde4e"

DINO_SRC = "computer-vision-ai-lab/dinov3-vits16-pretrain-lvd1689m"
DINO_DST = "cont1037/dinov3-vits16-pretrain-lvd1689m"
DINO_SRC_REV = "e2b5191960331471bf2734d372e6a7151a6079c5"


def patch(path: Path, subs: list[tuple[str, str]], regex: bool = False) -> int:
    text = original = path.read_text()
    for old, new in subs:
        text = re.sub(old, new, text) if regex else text.replace(old, new)
    if text != original:
        path.write_text(text)
        n = sum(1 for a, b in zip(original.splitlines(), text.splitlines()) if a != b)
        print(f"  patched {path.relative_to(ROOT)} ({n} lines)")
        return 1
    print(f"  unchanged {path.relative_to(ROOT)}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="/home/ubuntu/mirrored_models.json",
                    help="output of mirror_models.py: {repo: {revision: sha}}")
    args = ap.parse_args()

    mp = Path(args.map)
    if not mp.exists():
        sys.exit(f"missing {mp} — run mirror_models.py first so the new SHAs are known")
    m = json.loads(mp.read_text())
    try:
        glm_rev = m[GLM_DST]["revision"]
        dino_rev = m[DINO_DST]["revision"]
    except KeyError as e:
        sys.exit(f"{mp} has no entry for {e}")

    print(f"GLM  -> {GLM_DST}@{glm_rev}")
    print(f"DINO -> {DINO_DST}@{dino_rev}\n")

    patch(ROOT / "configuration.yaml", [
        (GLM_SRC, GLM_DST),
        (GLM_SRC_REV, glm_rev),
        (DINO_SRC, DINO_DST),
        (DINO_SRC_REV, dino_rev),
    ])
    patch(ROOT / "pipeline_service/modules/judge/embedder_settings.py", [
        (DINO_SRC, DINO_DST),
        (DINO_SRC_REV, dino_rev),
    ])
    # The judge's MODEL constant is the *served* name the local vLLM answers to,
    # which is whatever `vllm serve <model>` was given — so it tracks the repo id.
    patch(ROOT / "pipeline_service/modules/judge/multi_stage.py", [(GLM_SRC, GLM_DST)])
    patch(ROOT / "docker/Dockerfile", [
        (f"ARG GLM_MODEL={GLM_SRC}", f"ARG GLM_MODEL={GLM_DST}"),
        (f"ARG GLM_REVISION={GLM_SRC_REV}", f"ARG GLM_REVISION={glm_rev}"),
    ])
    patch(ROOT / "pipeline_service/scripts/setup_glm_vllm_env.sh", [
        (f'MODEL="${{MODEL:-{GLM_SRC}}}"', f'MODEL="${{MODEL:-{GLM_DST}}}"'),
    ])

    leftovers = []
    for p in ROOT.rglob("*"):
        if not p.is_file() or ".git/" in str(p) or "/runs/" in str(p):
            continue
        if p.suffix not in (".yaml", ".py", ".sh", ".txt") and p.name != "Dockerfile":
            continue
        try:
            t = p.read_text()
        except Exception:
            continue
        for needle in ("zai-org", "computer-vision-ai-lab"):
            if needle in t:
                leftovers.append(f"{p.relative_to(ROOT)}: {needle}")
    print("\nremaining non-cont references:",
          "\n  " + "\n  ".join(leftovers) if leftovers else " none")


if __name__ == "__main__":
    main()
