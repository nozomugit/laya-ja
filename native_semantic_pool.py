"""Average four Japanese adapters from one shared Laya encoder forward pass."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from scipy.special import softmax

from context_tasks import ContextReader, DomainCorrection
from fit_vote_calibration import fit_one
from four_task_assist import LABELS
from four_task_report import load, score
from native_distillation import encode
from semantic_vote import calibrate


class NativePool:
    def __init__(self, model_path):
        self.reader = ContextReader("nli", model_path)
        self.heads = []
        weights = Path(__file__).with_name("weights") / "context"
        for domain in ("jnli", "jsick", "jamp", "logic"):
            head = DomainCorrection().eval()
            head.load_state_dict(
                load_file(str(weights / f"domain-{domain}.safetensors"))
            )
            self.heads.append(head)

    @torch.inference_mode()
    def probabilities(self, rows):
        x, logits = encode(self.reader, rows)
        x = torch.from_numpy(x).unsqueeze(1)
        probs = [softmax(logits, axis=1)]
        for head in self.heads:
            probs.append(softmax(logits + head(x).numpy(), axis=1))
        return probs[0], np.mean(probs, axis=0)


def experiment(args, pool):
    datasets = {
        k: load(args.inputs / (k + ".jsonl")) for k in ["nli-dev", "semantic-cal"]
    }
    dev = datasets["nli-dev"] + datasets["semantic-cal"]
    groups = json.loads((args.inputs / "groups.json").read_text())["semantic-cal"]
    g = np.array(["jnli"] * 48 + ["jsick"] * 48 + ["world-" + str(i) for i in groups])
    y = np.array([r["gold"] for r in dev])
    _, prob = pool.probabilities(dev)
    theta = fit_one(prob, y, g).tolist()
    args.weights.write_text(json.dumps(theta) + "\n")
    result = {}
    for key in ["jsem", "jsem-fresh", "challenge"]:
        rows = load(args.inputs / (key + ".jsonl"))
        base, avg = pool.probabilities(rows)
        y = np.array([r["gold"] for r in rows])
        result[key] = {
            name: score(y, p)
            for name, p in [
                ("shared", base),
                ("equal_pool", avg),
                ("calibrated", calibrate(avg, theta)),
            ]
        }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path")
    p.add_argument(
        "--weights", type=Path, default=Path("weights/native-semantic-pool.json")
    )
    sub = p.add_subparsers(dest="task", required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("--inputs", type=Path, default=Path(".cache/four"))
    fit.add_argument(
        "--output", type=Path, default=Path("results/native-semantic-pool.json")
    )
    pred = sub.add_parser("predict")
    pred.add_argument("premise")
    pred.add_argument("hypothesis")
    args = p.parse_args()
    torch.set_num_threads(2)
    pool = NativePool(args.model_path)
    if args.task == "fit":
        experiment(args, pool)
    else:
        _, avg = pool.probabilities(
            [{"text": args.premise, "options": [args.hypothesis]}]
        )
        probs = calibrate(avg[0], json.loads(args.weights.read_text()))
        print(
            json.dumps(
                {
                    "choice": LABELS[probs.argmax()],
                    "probabilities": dict(zip(LABELS, probs.tolist())),
                },
                ensure_ascii=False,
            )
        )
