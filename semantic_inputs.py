"""Reserve additional, premise-disjoint JSeM cases and encode a separate calibration set."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from safetensors.torch import load_file

from context_tasks import Readout, batch, item
from corpus_data import jsem, normalized
from laya_ja import load_base
from semantic_worlds import generate


def run(a):
    root = Path(__file__).parent
    manifest = json.loads((root / "tools/four-task/manifest.json").read_text())
    old = [json.loads(s) for s in (a.inputs / "jsem.jsonl").read_text().splitlines()]
    blocked = {normalized(r["text"]) for r in old}
    original = {q[1] for q in manifest["jsem"]}
    fresh = []
    pool = list(jsem(a.corpora))
    random.Random(341907).shuffle(pool)
    for label in range(3):
        count = 0
        for (text, _, _), y, _, ident in pool:
            p, h = text.removeprefix("前提：").split("\n仮説：", 1)
            if y == label and ident not in original and normalized(p) not in blocked:
                fresh.append((ident, p, h, y))
                blocked.add(normalized(p))
                count += 1
                if count == 32:
                    break
        if count != 32:
            raise ValueError("Insufficient novel premises")
    groups = json.loads((a.inputs / "groups.json").read_text())
    worlds = generate()

    def as_rows(values, dataset, source, agent, head):
        spec = []
        output = []
        for i in range(0, len(values), 12):
            rows = values[i : i + 12]
            items = [item(agent.tok, "nli", p, [h]) for _, p, h, _ in rows]
            with torch.no_grad():
                z = head(batch(agent, items)).softmax(-1).tolist()
            for (ident, p, h, y), prob in zip(rows, z):
                spec.append(
                    [source, ident, hashlib.sha256((p + "\n" + h).encode()).hexdigest()]
                )
                output.append({"text": p, "options": [h], "gold": y, "laya": prob})
        manifest[dataset] = spec
        (a.inputs / (dataset + ".jsonl")).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in output)
        )

    agent = load_base(a.model_path, "cpu")
    head = Readout("nli").eval()
    head.load_state_dict(load_file(str(root / "weights/context/nli.safetensors")))
    as_rows(
        [(k, p, h, y) for k, (p, h, y, _) in worlds.items()],
        "semantic-cal",
        "world",
        agent,
        head,
    )
    as_rows(fresh, "jsem-fresh", "jsem", agent, head)
    groups["semantic-cal"] = [v[3] for v in worlds.values()]
    groups["jsem-fresh"] = [r[0].split(":")[0] for r in fresh]
    (a.inputs / "groups.json").write_text(json.dumps(groups, ensure_ascii=False) + "\n")
    (root / "tools/four-task/semantic-manifest.json").write_text(
        json.dumps(
            {k: manifest[k] for k in ("semantic-cal", "jsem-fresh")},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, default=Path(".cache/four"))
    p.add_argument("--corpora", type=Path, default=Path(".cache/corpora"))
    p.add_argument("--model-path")
    torch.set_num_threads(3)
    run(p.parse_args())
