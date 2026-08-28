"""Publish reviewed IOO artifacts to Hugging Face when explicitly requested."""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True)
    p.add_argument("--folder", required=True)
    p.add_argument("--commit-message", default="Add reviewed IOO experiment artifacts")
    args = p.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required and must be supplied through the environment")
    from huggingface_hub import HfApi
    folder = Path(args.folder)
    if not folder.is_dir():
        raise SystemExit(f"artifact folder does not exist: {folder}")
    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=args.repo_id,
                      repo_type="dataset", commit_message=args.commit_message)
    print(f"uploaded reviewed artifacts to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
