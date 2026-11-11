"""Prepare aligned inputs and frozen Laya probabilities for the four-task experiment."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from context_experiment import dataset, parse, samples
from context_tasks import DomainCorrection, Readout
from corpus_data import splits


@torch.inference_mode()
def prepare(args):
    args.out.mkdir(parents=True, exist_ok=True)
    groups = {}
    weights = Path(__file__).with_name("weights") / "context"
    for task in ("nli", "jcommonsenseqa"):
        parts = samples(task, args)
        head = Readout(task).eval()
        head.load_state_dict(load_file(str(weights / (task + ".safetensors"))))
        domain = None
        if task == "nli":
            domain = DomainCorrection().eval()
            domain.load_state_dict(load_file(str(weights / "domain-jamp.safetensors")))
        keys = ["jamp", "jsem", "challenge", "tune"] if domain else [task, "tune"]
        for key in keys:
            rows = parts[key]
            x, _ = dataset(None, task, key, rows, args).tensors
            z = head(x)
            selections = [(key, list(range(len(rows))), z)]
            if task == "nli" and key == "jamp":
                selections = [
                    (key, list(range(len(rows))), z + domain(head.representation(x)))
                ]
            if key == "tune" and task == "nli":
                dev = splits(args.corpora, args.data)["jamp-tune"]
                indices = [rows.index(row) for row in dev]
                selections = [
                    ("jamp-dev", indices, z + domain(head.representation(x))),
                    ("nli-dev", list(range(0, 384, 4)), z),
                ]
            if task == "jcommonsenseqa" and key == "tune":
                selections = [("qa-dev", list(range(0, len(rows), 3)), z)]
            for name, ids, logits in selections:
                if name == "jsem":
                    groups[name] = [rows[i][2] for i in ids]
                if name == "challenge":
                    source = json.loads(
                        Path(__file__).with_name("challenge.json").read_text()
                    )
                    groups[name] = [source[i][0] for i in ids]
                with (args.out / (name + ".jsonl")).open("w") as f:
                    for i in ids:
                        a, b = parse(task, rows[i])
                        f.write(
                            json.dumps(
                                {
                                    "text": a,
                                    "options": b,
                                    "laya": logits[i].softmax(-1).tolist(),
                                    "gold": rows[i][1],
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

    (args.out / "groups.json").write_text(json.dumps(groups, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k, v in {
        "data": ".cache/data",
        "corpora": ".cache/corpora",
        "cache": ".cache/context",
        "out": ".cache/four",
    }.items():
        p.add_argument("--" + k, type=Path, default=Path(v))
    torch.set_num_threads(2)
    prepare(p.parse_args())
