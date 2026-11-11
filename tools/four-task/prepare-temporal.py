"""Reconstruct temporal rows from the two pinned TSV files without exporting labels."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

from prepare import retrieve

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from four_task_prompts import nli_prompt

p = argparse.ArgumentParser()
p.add_argument("--cache", type=Path, default=Path(".cache/four-public"))
p.add_argument("--out", type=Path, required=True)
p.add_argument("--shards", type=int, default=4)
p.add_argument("--shard", type=int, required=True)
a = p.parse_args()
a.cache.mkdir(parents=True, exist_ok=True)
a.out.mkdir(parents=True, exist_ok=True)
base = "tomo-ut/temporalNLI_dataset/f578b5daac7bff3855d238c032a60fd047cddd9c/dataset/"
files = {"jamp-dev": "train_custom-template-hard.tsv", "jamp": "test_2.tsv"}
manifest = json.loads(Path(__file__).with_name("temporal-manifest.json").read_text())
n = 0
with (a.out / "temporal.jsonl").open("w") as f:
    for split, filename in files.items():
        with retrieve(a, filename, base + filename).open() as src:
            texts = {
                hashlib.sha256(
                    (r["premise"] + "\n" + r["hypothesis"]).encode()
                ).hexdigest(): (r["premise"], r["hypothesis"])
                for r in csv.DictReader(src, delimiter="\t")
            }
        for i, digest in enumerate(manifest[split]):
            pair = texts[digest]
            if n % a.shards == a.shard:
                f.write(
                    json.dumps(
                        {"id": [split, i], "prompt": nli_prompt(*pair)},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            n += 1
