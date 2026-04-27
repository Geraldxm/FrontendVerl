#!/usr/bin/env python3
"""
Upload selected checkpoint steps to Hugging Face Hub, using one repo per step.

Example:
python scripts/upload_steps_to_hf.py \
  --source-dirs \
    /inspire/hdd/global_user/gexinmu-253108100065/Repos/FrontendVerl/merged_checkpoints/frontend_focal/baseline_v3_Qwen3-1.7B-Base \
    /inspire/hdd/global_user/gexinmu-253108100065/Repos/FrontendVerl/merged_checkpoints/frontend_focal/focal_v3_Qwen3-1.7B-Base \
  --steps 80 120 140 \
  --namespace your-hf-username \
  --repo-prefix frontend-focal \
  --private
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload selected checkpoint steps to Hugging Face Hub (one step -> one repo)."
    )
    parser.add_argument(
        "--source-dirs",
        nargs="+",
        required=True,
        help="Experiment directories containing step folders, e.g. global_step_80.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        required=True,
        help="Step list, e.g. 80 120 or global_step_80 global_step_120.",
    )
    parser.add_argument("--namespace", required=True, help="HF namespace, usually your username or org.")
    parser.add_argument(
        "--repo-prefix",
        required=True,
        help="Prefix for generated repo names, e.g. frontend-focal.",
    )
    parser.add_argument(
        "--repo-type",
        default="model",
        choices=["model", "dataset", "space"],
        help="HF repo type.",
    )
    parser.add_argument("--private", action="store_true", help="Create private repositories.")
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Optional HF endpoint (sets HF_ENDPOINT), e.g. https://hf-mirror.com.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token. If omitted, uses HF_TOKEN / HUGGINGFACE_HUB_TOKEN / cached login.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip missing steps instead of failing.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print planned actions.")
    return parser.parse_args()


def normalize_step(step: str) -> str:
    return step if step.startswith("global_step_") else f"global_step_{step}"


def sanitize_repo_name(name: str) -> str:
    lowered = name.lower()
    cleaned = re.sub(pattern=r"[^a-z0-9._-]+", repl="-", string=lowered)
    return cleaned.strip("-")


def iter_upload_tasks(source_dirs: Iterable[str], steps: Iterable[str]):
    for source_dir in source_dirs:
        source_path = Path(source_dir).expanduser().resolve()
        exp_name = source_path.name
        for step in steps:
            step_dir_name = normalize_step(step=step)
            step_path = source_path / step_dir_name
            step_num = step_dir_name.removeprefix("global_step_")
            yield source_path, exp_name, step_dir_name, step_num, step_path


def main() -> int:
    args = parse_args()

    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint
        print(f"[info] HF_ENDPOINT={args.endpoint}")

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[error] Missing dependency: huggingface_hub")
        print("[hint] Install with: pip install -U huggingface_hub")
        return 2

    api = HfApi(token=args.token)
    normalized_steps = [normalize_step(step=s) for s in args.steps]

    print("[info] Planned uploads:")
    print(f"  source_dirs={len(args.source_dirs)}")
    print(f"  steps={normalized_steps}")
    print(f"  namespace={args.namespace}")
    print(f"  repo_prefix={args.repo_prefix}")
    print(f"  repo_type={args.repo_type}")
    print(f"  private={args.private}")
    print(f"  dry_run={args.dry_run}")

    failures = 0
    total = 0

    for source_path, exp_name, step_dir_name, step_num, step_path in iter_upload_tasks(
        source_dirs=args.source_dirs,
        steps=args.steps,
    ):
        total += 1
        repo_name = sanitize_repo_name(name=f"{args.repo_prefix}-{exp_name}-step-{step_num}")
        repo_id = f"{args.namespace}/{repo_name}"
        commit_message = f"upload {exp_name} {step_dir_name}"

        if not step_path.exists():
            msg = f"[warn] Missing step dir: {step_path}"
            if args.skip_missing:
                print(msg + " (skipped)")
                continue
            print(msg)
            failures += 1
            continue

        print(f"\n[task {total}] {step_path}")
        print(f"  -> repo_id={repo_id}")
        print(f"  -> commit_message={commit_message}")

        if args.dry_run:
            continue

        try:
            api.create_repo(
                repo_id=repo_id,
                repo_type=args.repo_type,
                private=args.private,
                exist_ok=True,
            )
            api.upload_folder(
                folder_path=str(step_path),
                repo_id=repo_id,
                repo_type=args.repo_type,
                commit_message=commit_message,
            )
            print(f"  [ok] Uploaded to {repo_id}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  [error] Upload failed for {repo_id}: {exc}")

    print("\n[summary]")
    print(f"  total_tasks={total}")
    print(f"  failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
