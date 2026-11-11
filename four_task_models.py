"""Download the optional quantized judges and verify each upstream artifact."""

import argparse
import hashlib
from pathlib import Path

from huggingface_hub import hf_hub_download

MODELS = {
    "nli": (
        "unsloth/Qwen3.5-9B-GGUF",
        "3885219b6810b007914f3a7950a8d1b469d598a5",
        {
            "Qwen3.5-9B-Q4_K_M.gguf": "03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8"
        },
    ),
    "reason": (
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "a483e9e6cbd595906af30beda3187c2663a1118c",
        {
            "Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf": "9c964e657212fea1f24905dd7b0a89b82fd807d19fab0b41da14251b07b88fbe"
        },
    ),
    "gemma": (
        "unsloth/gemma-4-26B-A4B-it-GGUF",
        "c099eb48e663fd284577b04978a94ffccb261841",
        {
            "gemma-4-26B-A4B-it-UD-Q3_K_M.gguf": "811882f6df81c8ea204a5d382f196d6bd601245091786912d4c70df41cdfa052"
        },
    ),
    "qa": (
        "fukayatti0/jev-japanese-judgment",
        "fdae208ea5e435b1992fecdef8a4cb02c9a6bcaf",
        {
            "gguf/jev-Q4_K_M.gguf": "82aa1bedac6965858f61ea6d56f530153208136a6f0b347a05a521cd9596b827",
            "gguf/head.npz": "f63f3410337893b12ca426ea5e41ba18d7eb642fd244179131cfe597890e2d70",
        },
    ),
}


def download(key, root):
    repo, rev, files = MODELS[key]
    for filename, expected in files.items():
        path = Path(hf_hub_download(repo, filename, revision=rev, local_dir=root / key))
        with path.open("rb") as f:
            actual = hashlib.file_digest(f, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Checksum mismatch for {path}")
        print(path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=[*MODELS, "all"])
    p.add_argument("--root", type=Path, default=Path(".cache/four-models"))
    args = p.parse_args()
    for k in MODELS if args.mode == "all" else [args.mode]:
        download(k, args.root)
