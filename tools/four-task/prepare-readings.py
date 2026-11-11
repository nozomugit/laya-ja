"""Write two label-free prompts from the same hashed semantic input manifest."""

import argparse
import hashlib
import json
from pathlib import Path

from prepare import inputs

from semantic_readings import prompt

p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, required=True)
p.add_argument("--cache", type=Path, default=Path(".cache/four-public"))
p.add_argument("--shards", type=int, default=8)
p.add_argument("--shard", type=int, required=True)
a = p.parse_args()
a.cache.mkdir(exist_ok=True, parents=True)
a.out.mkdir(exist_ok=True, parents=True)
manifest = {}
for name in ("manifest", "semantic-manifest"):
    manifest.update(json.loads(Path(__file__).with_name(name + ".json").read_text()))
source = inputs(a, manifest)
n = 0
with (
    (a.out / "reading0.jsonl").open("w") as left,
    (a.out / "reading1.jsonl").open("w") as right,
):
    for k, rows in manifest.items():
        for i, (src, ident, digest) in enumerate(rows):
            x, y = source[src, ident]
            if hashlib.sha256((x + "\n" + y).encode()).hexdigest() != digest:
                raise ValueError("Manifest mismatch")
            if n % a.shards == a.shard:
                for v, f in enumerate((left, right)):
                    f.write(
                        json.dumps(
                            {"id": [k, i], "prompt": prompt(x, y, v)},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            n += 1
