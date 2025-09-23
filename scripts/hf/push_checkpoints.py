#!/usr/bin/env python3
"""
Utility to push a local folder (checkpoints, adapters, or artifacts) to the Hugging Face Hub.

Examples:
    python scripts/hf/push_checkpoints.py \
        --repo-id Junaidi-AI/med-vllm \
        --local-path outputs/ner-checkpoint \
        --path-in-repo checkpoints/ner-2025-09-23 \
        --commit-message "Add NER checkpoint"

If you need to create a PR instead of pushing to main directly, add --create-pr.
To rely on credentials saved via `hf auth login`, do not set HF_TOKEN or unset it for this run.

    env -u HF_TOKEN python scripts/hf/push_checkpoints.py ...
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Push a local folder to the Hugging Face Hub")
    p.add_argument("--repo-id", required=True, help="Target repo id, e.g. org/name or user/name")
    p.add_argument(
        "--local-path",
        required=True,
        help="Local file or folder to upload (can be a single file or a directory)",
    )
    p.add_argument(
        "--path-in-repo",
        default=None,
        help="Destination path in repo (defaults to same name as local-path)",
    )
    p.add_argument(
        "--repo-type",
        default="model",
        choices=["model", "dataset", "space"],
        help="Repo type on the Hub",
    )
    p.add_argument(
        "--commit-message",
        default="Upload artifacts",
        help="Commit title / first line",
    )
    p.add_argument(
        "--commit-description",
        default=None,
        help="Optional commit description (body)",
    )
    p.add_argument(
        "--create-pr",
        action="store_true",
        help="Upload changes as a Pull Request instead of pushing to the default branch",
    )
    p.add_argument(
        "--revision",
        default=None,
        help="Optional branch name or refs/pr/X reference to push against",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    local_path = Path(args.local_path)
    if not local_path.exists():
        print(f"[ERROR] Local path does not exist: {local_path}")
        return 2

    # Respect existing saved credential; avoid env HF_TOKEN overriding by allowing caller to unset it.
    api = HfApi()

    # Use HfApi.upload_folder for directories, or upload_file for single files
    try:
        if local_path.is_dir():
            commit = api.upload_folder(
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                folder_path=str(local_path),
                path_in_repo=(args.path_in_repo or local_path.name),
                commit_message=args.commit_message,
                commit_description=args.commit_description,
                create_pr=args.create_pr,
                revision=args.revision,
            )
        else:
            # Single file
            commit = api.upload_file(
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                path_or_fileobj=str(local_path),
                path_in_repo=(args.path_in_repo or local_path.name),
                commit_message=args.commit_message,
                commit_description=args.commit_description,
                create_pr=args.create_pr,
                revision=args.revision,
            )
    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")
        return 3

    print(commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
