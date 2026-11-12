"""List, and with --upload send, the all-layer LoRA weights to a private Hugging Face model repository.

Without --upload nothing leaves the machine: the files, sizes and SHA-256 are printed. The repository is created
private and is never made public here. The token is read from HF_TOKEN or, in Colab under %run, from a secret.
"""

import argparse
import hashlib
import os
from pathlib import Path

NLI = "mix4-logic-teacher-jsem"
TASKS = ("unified", "jcommonsenseqa", "jsts", "jcola")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def plan(runs, seeds):
    """(local file, path in the repository) for every weight file that exists."""
    out = []
    final = runs / "tasks" / "unified-final"
    if final.exists():
        out += [(p, f"unified/{p.relative_to(final).as_posix()}") for p in sorted(final.rglob("*")) if p.is_file()]
    for seed in seeds:
        for task in TASKS:
            out.append(
                (runs / "tasks" / f"{task}-s{seed}" / "adapter.safetensors", f"adapters/{task}-s{seed}.safetensors")
            )
        out.append((runs / "logic" / f"{NLI}-s{seed}" / "adapter.safetensors", f"adapters/nli-s{seed}.safetensors"))
    return [(p, name) for p, name in out if p.exists()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, required=True, help="folder holding tasks/ and logic/")
    ap.add_argument("--card", type=Path, default=Path(__file__).with_name("hub") / "README.md")
    ap.add_argument("--name", default="laya-ja")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--secret", default="HuggingFaceAccessToken")
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    files = plan(args.runs, args.seeds)
    sums = [(sha256(p), name, p.stat().st_size) for p, name in files]
    for digest, name, size in sums:
        print(f"{size / 1e6:9.1f} MB  {digest[:16]}  {name}")
    print(
        f"{len(files)} files, {sum(s for _, _, s in sums) / 1e6:.1f} MB; card: {args.card} exists={args.card.exists()}"
    )
    if not args.upload:
        print("Listed only. Add --upload to send to a private repository.")
        return
    if not files or not args.card.exists():
        raise SystemExit("Nothing to upload, or the model card is missing.")
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from google.colab import userdata

            token = userdata.get(args.secret)
        except Exception as error:
            raise SystemExit(f"Set HF_TOKEN, or the Colab secret {args.secret}: {type(error).__name__}") from None
    api = HfApi(token=token)
    repo = f"{api.whoami()['name']}/{args.name}"
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    listing = "".join(f"{digest}  {name}\n" for digest, name, _ in sums).encode()
    api.upload_file(path_or_fileobj=args.card, path_in_repo="README.md", repo_id=repo)
    api.upload_file(path_or_fileobj=listing, path_in_repo="SHA256SUMS", repo_id=repo)
    for path, name in files:
        api.upload_file(path_or_fileobj=path, path_in_repo=name, repo_id=repo)
        print("sent", name, flush=True)
    info = api.model_info(repo)
    print(f"https://huggingface.co/{repo} private={info.private} revision={info.sha}")


if __name__ == "__main__":
    main()
