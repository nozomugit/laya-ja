"""Export anonymous per-item outputs and verify the published accuracy totals."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax

from four_task_reason import combine
from four_task_report import load, shift
from semantic_verifier import features
from semantic_verifier import predict as verify
from temporal_graph import predict as temporal_graph
from temporal_relations import probabilities


def run(a):
    config = json.loads((a.weights / "four-fusion.json").read_text())
    deep_config = json.loads((a.weights / "four-reason.json").read_text())
    graph_cfg = json.loads((a.weights / "temporal-graph.json").read_text())
    plain = json.loads((a.results / "four-fusion.json").read_text())
    deep = json.loads((a.results / "four-reason.json").read_text())
    temporal = json.loads(
        gzip.decompress((a.weights / "temporal.json.gz").read_bytes())
    )
    temporal_hard = json.loads(
        gzip.decompress((a.weights / "temporal-hard.json.gz").read_bytes())
    )
    ls = {}
    for style in ("direct", "contrast", "reason"):
        ls[style] = {
            tuple(r["id"]): r["logits"] for r in load(a.pred / (style + "-out.jsonl"))
        }
    qa = load(a.pred / "jcommonsenseqa.jsonl")
    verifier_cfg = json.loads((a.weights / "semantic-verifier.json").read_text())
    verifier_report = json.loads((a.results / "four-semantic.json").read_text())
    raw_sem = [
        {
            tuple(r["id"]): np.array(r["logits"])
            for r in load(a.semantic / (k + "-out.jsonl"))
        }
        for k in ("must_true", "must_false", "direct")
    ]
    out = []
    consensus = {}
    for dataset in ("jamp", "jcommonsenseqa", "jsem", "challenge", "jsem-fresh"):
        rs = load(a.inputs / (dataset + ".jsonl"))
        correct = {}
        for i, r in enumerate(rs):
            y = r["gold"]
            prior = np.array(r["laya"])
            probs = {"laya": prior}
            if dataset == "jamp":
                probs["combined"] = probabilities(
                    temporal, r["text"], r["options"][0], prior
                )
                probs["template_hard"] = probabilities(
                    temporal_hard, r["text"], r["options"][0], prior
                )
                probs["temporal_graph"] = temporal_graph(
                    probs["template_hard"],
                    r["text"],
                    r["options"][0],
                    graph_cfg["supported"],
                )
            elif dataset == "jcommonsenseqa":
                assert qa[i]["i"] == i
                probs["jev"] = np.array(qa[i]["specialist"])
                w = config["qa_laya_weight"]
                probs["combined"] = w * prior + (1 - w) * probs["jev"]
            elif dataset != "jsem-fresh":
                key = (dataset, i)
                q = softmax(ls["direct"][key])
                c = softmax(ls["contrast"][key])[[1, 2, 0]]
                d = softmax(ls["reason"][key])
                v = config["contrast_weight"]
                w = config["nli_laya_weight"]
                quick = (1 - v) * q + v * c
                probs.update(
                    direct=q,
                    direct_laya=0.1 * prior + 0.9 * q,
                    fixed_mix=combine(
                        prior, quick, d, {"native_weight": 0.1, "deep_weight": 0.75}
                    ),
                    contrast=c,
                    reason=d,
                    fast=shift(w * prior + (1 - w) * quick, config["label_bias"]),
                    single_judge=0.1 * prior + 0.9 * d,
                    combined=combine(prior, quick, d, deep_config),
                )
            if dataset in ("jsem", "challenge", "jsem-fresh"):
                key = (dataset, i)
                probs["sparse_direct_laya"] = 0.1 * prior + 0.9 * softmax(
                    raw_sem[2][key]
                )
                probs["verifier"] = verify(
                    features(*(d[key] for d in raw_sem), prior),
                    verifier_cfg["coefficients"],
                )
                if "combined" in probs:
                    probs["consensus"] = (
                        0.5 * probs["verifier"] + 0.5 * probs["combined"]
                    )
            pred = {k: int(z.argmax()) for k, z in probs.items()}
            for k, n in pred.items():
                correct[k] = correct.get(k, 0) + int(n == y)
            out.append(
                {
                    "split": dataset,
                    "row": i,
                    "gold": y,
                    "prediction": pred,
                    "probabilities": {k: z.round(7).tolist() for k, z in probs.items()},
                }
            )
        if dataset in verifier_report:
            assert (
                correct["verifier"] == verifier_report[dataset]["combined"]["correct"]
            )
            if "consensus" in correct:
                consensus[dataset] = {"correct": correct["consensus"], "n": len(rs)}
        if dataset == "jsem-fresh":
            continue
        if dataset == "jamp":
            assert (
                correct["temporal_graph"]
                == json.loads((a.results / "four-temporal-graph.json").read_text())[
                    "test"
                ]["correct"]
            )
            expected = json.loads((a.results / "four-temporal.json").read_text())[
                "correct"
            ]
        elif dataset == "jcommonsenseqa":
            expected = plain[dataset]["combined"]["correct"]
        else:
            expected = deep[dataset]["combined"]["correct"]
            assert correct["fast"] == plain[dataset]["combined"]["correct"]
            assert correct["single_judge"] == deep[dataset]["single_judge"]["correct"]
        assert correct["combined"] == expected, (dataset, correct, expected)
    (a.results / "four-consensus.json").write_text(
        json.dumps(consensus, indent=2) + "\n"
    )
    raw = "".join(json.dumps(v, separators=(",", ":")) + "\n" for v in out).encode()
    (a.results / "four-evaluation.jsonl.gz").write_bytes(gzip.compress(raw, mtime=0))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k, v in {
        "inputs": ".cache/four",
        "pred": ".cache/four-predictions",
        "semantic": ".cache/semantic-predictions",
        "weights": "weights",
        "results": "results",
    }.items():
        p.add_argument("--" + k, type=Path, default=Path(v))
    run(p.parse_args())
