"""Local specialist inference for Japanese commonsense and textual inference."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

import numpy as np
from scipy.special import erf, softmax

from four_task_prompts import nli_prompt

LABELS = ("含意", "矛盾", "判断できない")


def post(url, endpoint, payload):
    req = Request(
        url.rstrip("/") + "/" + endpoint,
        json.dumps(payload, ensure_ascii=False).encode(),
        {"Content-Type": "application/json"},
    )
    with build_opener(ProxyHandler({})).open(req, timeout=240) as f:
        return json.load(f)


class LocalJudges:
    def __init__(self, url, mode, head=None):
        self.url = url
        self.mode = mode
        if mode == "qa":
            with np.load(head) as f:
                self.head = {k: v for k, v in f.items()}
        else:
            self.tokens = []
            for letter in "ABC":
                tokens = post(
                    url, "tokenize", {"content": letter, "add_special": False}
                )["tokens"]
                if len(tokens) != 1:
                    raise ValueError(
                        "Each decision label must encode to a single token"
                    )
                self.tokens.append(tokens[0])

    def commonsense(self, question, choices):
        texts = [f"\n{question}\n{c}" for c in choices]
        raw = post(self.url, "embedding", {"content": texts, "embd_normalize": -1})
        rows = sorted(raw, key=lambda r: r["index"])
        x = np.asarray([np.asarray(r["embedding"]).reshape(-1) for r in rows])
        w = self.head
        h = x @ w["mlp.0.weight"].T + w["mlp.0.bias"]
        h = 0.5 * h * (1 + erf(h / np.sqrt(2)))
        return softmax((h @ w["mlp.2.weight"].T + w["mlp.2.bias"]).reshape(-1))

    def infer(self, premise, hypothesis, contrast=False):
        prompt = nli_prompt(premise, hypothesis, contrast)
        payload = {
            "prompt": prompt,
            "temperature": 1.0,
            "samplers": ["temperature"],
            "top_k": 0,
            "n_predict": 1,
            "n_probs": 3,
            "post_sampling_probs": True,
            "cache_prompt": True,
            "seed": 19,
            "logit_bias": [[t, 80.0] for t in self.tokens],
        }
        res = post(self.url, "completion", payload)["completion_probabilities"][0][
            "top_probs"
        ]
        probs = {r["id"]: r["prob"] for r in res}
        scores = np.array([probs.get(t, 0.0) for t in self.tokens])
        if scores.sum() < 0.99:
            raise ValueError(f"Invalid answer-token probabilities: {res}")
        scores = scores / scores.sum()
        return scores[[1, 2, 0]] if contrast else scores


def blend(a, b, w):
    return (1 - w) * np.asarray(a) + w * np.asarray(b)


def score_file(judge, path, out, jobs, contrast):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    existing = (
        [json.loads(line) for line in out.read_text().splitlines()]
        if out.exists()
        else []
    )
    if any(r["i"] != i for i, r in enumerate(existing)):
        raise ValueError("Prediction order changed")

    def one(index):
        r = rows[index]
        prob = (
            judge.commonsense(r["text"], r["options"])
            if judge.mode == "qa"
            else judge.infer(r["text"], r["options"][0], contrast=contrast)
        )
        return {"i": index, "specialist": prob.tolist()}

    with ThreadPoolExecutor(jobs) as pool, out.open("a") as fp:
        for v in pool.map(one, range(len(existing), len(rows))):
            fp.write(json.dumps(v) + "\n")
            fp.flush()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["qa", "nli"])
    p.add_argument("--head", type=Path)
    p.add_argument("--url", required=True)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--inputs", type=Path, default=Path(".cache/four"))
    p.add_argument("--out", type=Path, default=Path(".cache/four-predictions"))
    p.add_argument("--contrast", action="store_true")
    p.add_argument("--splits", nargs="+", required=True)
    a = p.parse_args()
    a.out.mkdir(exist_ok=True, parents=True)
    judge = LocalJudges(a.url, a.mode, a.head)
    for split in a.splits:
        suffix = "-contrast" if a.contrast else ""
        score_file(
            judge,
            a.inputs / (split + ".jsonl"),
            a.out / (split + suffix + ".jsonl"),
            a.jobs,
            a.contrast,
        )
