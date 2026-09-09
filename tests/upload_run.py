#!/usr/bin/env python3
"""Push a run directory produced by run_batches.py to a private HF dataset repo.

    python tests/upload_run.py runs/r42 --repo cont1037/imback4-r42-<stamp>

The token is read from HF_TOKEN or --token-file (never written into the upload).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def du(path: Path) -> str:
    try:
        return subprocess.run(["du", "-sh", str(path)], capture_output=True,
                              text=True, timeout=120).stdout.split()[0]
    except Exception:
        return "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--repo", required=True, help="e.g. cont1037/imback4-r42-20260907")
    ap.add_argument("--token-file", default="/home/ubuntu/.secrets/hf_token")
    ap.add_argument("--repo-type", default="dataset", choices=["dataset", "model"])
    ap.add_argument("--public", action="store_true", help="default is private")
    ap.add_argument("--ignore", nargs="*", default=[],
                    help="extra glob patterns to exclude (e.g. '*/candidates/*')")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        sys.exit(f"not a directory: {run_dir}")

    token = os.environ.get("HF_TOKEN") or Path(args.token_file).read_text().strip()
    if not token:
        sys.exit("no token (set HF_TOKEN or --token-file)")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    who = api.whoami()
    print(f"authenticated as {who.get('name')} ({who.get('type')})")

    api.create_repo(repo_id=args.repo, repo_type=args.repo_type,
                    private=not args.public, exist_ok=True)
    print(f"repo ready: {args.repo} ({args.repo_type}, "
          f"{'public' if args.public else 'private'})")

    print(f"uploading {run_dir} ({du(run_dir)}) ...")
    url = api.upload_folder(
        folder_path=str(run_dir),
        repo_id=args.repo,
        repo_type=args.repo_type,
        commit_message=f"run {run_dir.name}",
        ignore_patterns=[".git*", "**/__pycache__/**", *args.ignore],
    )
    print(f"done: {url}")
    print(f"https://huggingface.co/{args.repo_type}s/{args.repo}")


if __name__ == "__main__":
    main()
