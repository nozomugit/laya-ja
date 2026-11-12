"""All-layer LoRA for commonsense choice, similarity score and acceptability, alone and in one model with inference."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import laya.train as laya_train
import numpy as np
import torch
from laya.common import QTYPES, collate_items
from laya.train import (
    TrainConfig,
    encode_item,
    encode_state,
    make_item,
    to_internal,
    train_model,
)
from safetensors.torch import save_file

import lora_nli
from corpus_experiment import measures
from experiment import partition
from laya_ja import BASE, REVISION, nli_request
from lora import add_lora, apply_adapter, merge_lora, trainable_state
from scoring import diagnostic, metrics
from typed import SOURCES, load_split

TASKS = ("jcommonsenseqa", "jsts", "jcola")
RANK, ALPHA, DROPOUT = lora_nli.RANK, lora_nli.ALPHA, lora_nli.DROPOUT


def task_rows(task, data):
    """Same tuning, evaluation and reserved rows as context_experiment.samples; all remaining rows train."""
    if task == "jcommonsenseqa":
        d = partition(task, data, 42, 10**8, 192, 10**8)
        train, tune, tests, old = d["train"], d["tune"], {"dev": d["eval"]}, 4096
    else:
        names = list(SOURCES[task])
        train, *sets = [load_split(task, n, data) for n in names]
        random.Random(42).shuffle(train)
        groups = {r[2] for v in sets for r in v}
        train = [r for r in train if r[2] not in groups]
        tune = train[:192]
        tune_groups = {r[2] for r in tune}
        train = [r for r in train[192:] if r[2] not in tune_groups]
        short = {"jsts-valid": "dev", "in_domain_valid-v1.0": "dev", "out_of_domain_valid-v1.0": "dev-out"}
        tests, old = {short[n]: v for n, v in zip(names[1:], sets)}, 2048
    used = {r[2] for r in train[:old]}
    reserved = [r for r in train[old:] if r[2] not in used][:256]
    reserved_groups = {r[2] for r in reserved}
    train = [r for r in train if r[2] not in reserved_groups]
    return train, tune, {**tests, "reserved": reserved}


def task_items(bench, task, rows, weights=None, train=False):
    """Typed questions as Laya asks them: five-way choice, six-level score, or a yes/no acceptability."""
    out = []
    for (state, instruction, options), y, _, _ in rows:
        if task == "jcommonsenseqa":
            question = {"type": "choice", "instructions": instruction, "criteria": dict.fromkeys(options, "")}
            target = [float(i == y) for i in range(len(options))]
        elif task == "jsts":
            question = {"type": "score", "instructions": instruction, "criteria": list(options)}
            low = min(int(y), 4)
            target = [0.0] * 6
            target[low], target[low + 1] = 1.0 - (y - low), y - low
        else:
            question = {
                "type": "noul",
                "instructions": instruction,
                "criteria": {"false": options[0], "true": options[1]},
            }
            scale = weights[int(y)] if weights else 1.0
            target = [scale * float(i == y) for i in range(2)]
        q = to_internal("decision", question)
        state_ids = encode_state(bench.tok, state, bench.max_len)
        item, reason = make_item(bench.tok, q, target, state_ids, bench.head_max_len, bench.max_len)
        if item is None and not train:
            if reason != "options_collapsed":
                raise ValueError(f"{task} row cannot be encoded: {reason}")
            item = {
                "q": q,
                "state_ids": state_ids,
                "target": target,
                "qtype": QTYPES[q["t"]],
                "k": len(target),
            }
        if item is not None:
            out.append(item)
    return out


def distill_items(bench, path, train=False):
    """Questions labeled by the two teachers (tools/distill/combine_teachers.py); the target is their mean probability.

    Inference questions are re-asked in the wording every other inference source uses, so the model sees one format.
    """
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        question, target, state = r["question"], [float(v) for v in r["target"]], r["state"]
        if question["type"] == "choice" and list(question["criteria"]) == ["含意", "矛盾", "判断できない"]:
            premise, hypothesis = state.removeprefix("前提：").rsplit("\n仮説：", 1)
            out += bench.items([(nli_request(premise, hypothesis), tuple(target), "distill", r["id"])], train=train)
            continue
        q = to_internal("decision", question)
        state_ids = encode_state(bench.tok, state, bench.max_len)
        item, _ = make_item(bench.tok, q, target, state_ids, bench.head_max_len, bench.max_len)
        if item is not None:
            out.append(item)
    return out


@torch.no_grad()
def score(bench, model, items, batch_size=64):
    was_training = model.training
    model.eval()
    order = sorted(range(len(items)), key=lambda i: len(items[i]["state_ids"]))
    out = [None] * len(items)
    keys = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
    for start in range(0, len(items), batch_size):
        idx = order[start : start + batch_size]
        chunk = [encode_item(bench.tok, items[i], bench.max_len, bench.head_max_len) for i in idx]
        b = collate_items([chunk], bench.tok.pad_token_id)
        logits, _ = model(*(b[k].to(bench.device) for k in keys))
        for row, i in zip(logits.float().cpu(), idx):
            out[i] = row[: items[i]["k"]]
    model.train(was_training)
    return torch.stack(out)


def tune_loss(logits, items):
    target = torch.tensor([it["target"] for it in items])
    mask = torch.ones_like(target, dtype=torch.bool)
    return float(lora_nli.mixed_loss(logits, target, mask))


def evaluate(source, name, logits, rows):
    """Headline metrics per task; inference sets use the same measures as lora_nli.py."""
    y = torch.tensor([r[1] for r in rows])
    if source == "jcommonsenseqa":
        m = metrics(logits, y)
        return {"n": m["n"], "correct": m["correct"], "accuracy": m["accuracy"], "nll": m["nll"]}
    if source in ("jsts", "jcola"):
        m = diagnostic(source, logits, y.float() if source == "jsts" else y)
        return {k: (float(v) if isinstance(v, float) else v) for k, v in m.items()}
    m = measures(logits, y, name == "janli")[0]
    return {
        "n": m["n"],
        "correct": m["correct"],
        "accuracy": m["accuracy"],
        "recall": m["recall"],
        "nll": m["nll"],
    }


def headline(source, name, m):
    if source == "jsts":
        return f"pearson={m['pearson']:.4f} mae={m['mae']:.3f}"
    if source == "jcola":
        return f"mcc={m['mcc']:.4f} acc={m['accuracy']:.4f}"
    return f"correct={m['correct']}/{m['n']}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key, default in (
        ("data", ".cache/data"),
        ("corpora", ".cache/corpora"),
        ("out", ".cache/lora-tasks"),
        ("result", "results/lora-tasks.json"),
    ):
        ap.add_argument("--" + key, type=Path, default=Path(default))
    ap.add_argument("--regimes", nargs="+", default=[*TASKS, "unified"], choices=[*TASKS, "unified"])
    ap.add_argument("--nli-regime", default="mix4-logic", choices=list(lora_nli.REGIMES))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--encoder-lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--extra-cap", type=int, default=4000)
    ap.add_argument("--entailment-weight", type=float, default=1.0)
    ap.add_argument("--logic-cap", type=int, default=6000)
    ap.add_argument("--quant-cap", type=int, default=6000)
    ap.add_argument("--contrast-cap", type=int, default=6000)
    ap.add_argument("--teacher", type=Path)
    ap.add_argument("--teacher-cap", type=int, default=8000)
    ap.add_argument("--teacher-all", action="store_true")
    ap.add_argument("--distill", type=Path, help="folder written by tools/distill/combine_teachers.py")
    ap.add_argument("--max-train", type=int)
    ap.add_argument("--summary", type=Path)
    ap.add_argument("--model-path")
    ap.add_argument("--device")
    args = ap.parse_args()

    laya_train.soft_ce_loss = lora_nli.mixed_loss
    bench = lora_nli.Bench(args)
    train_items, tune_items, eval_sets = {}, {}, {}  # eval_sets: (source, name) -> (rows, items)
    for task in TASKS:
        train, tune, tests = task_rows(task, args.data)
        if args.max_train:
            train = train[: args.max_train]
        weights = None
        if task == "jcola":
            n1 = sum(r[1] for r in train)
            weights = [len(train) / (2 * (len(train) - n1)), len(train) / (2 * n1)]
        train_items[task] = task_items(bench, task, train, weights, train=True)
        tune_items[task] = task_items(bench, task, tune)
        for name, rows in tests.items():
            eval_sets[(task, name)] = (rows, task_items(bench, task, rows))
    nli_sources = lora_nli.REGIMES[args.nli_regime]
    if "unified" in args.regimes:
        train, tune, tests = lora_nli.corpora(args)
        for s in nli_sources:
            train_items[s] = bench.items(train[s], train=True)
            if s in tune:
                tune_items[s] = bench.items(tune[s])
        for name, rows in tests.items():
            eval_sets[("nli", name)] = (rows, bench.items(rows))
    sources_of = {t: [t] for t in TASKS}
    sources_of["unified"] = [*TASKS, *nli_sources]
    if args.distill:
        train_items["distill"] = distill_items(bench, args.distill / "train.jsonl", train=True)
        tune_items["distill"] = distill_items(bench, args.distill / "tune.jsonl")
        sources_of["unified"].append("distill")
    run_config = {
        k: getattr(args, k)
        for k in (
            "epochs",
            "encoder_lr",
            "head_lr",
            "micro_batch",
            "grad_accum",
            "max_train",
            "nli_regime",
            "extra_cap",
            "entailment_weight",
            "logic_cap",
            "teacher_cap",
        )
    }
    if args.distill:
        run_config["distill"] = hashlib.sha256((args.distill / "train.jsonl").read_bytes()).hexdigest()

    def score_all(model, regime):
        wanted = set(sources_of[regime]) | ({"nli"} if regime == "unified" else set())
        return {
            f"{source}/{name}": score(bench, model, items)
            for (source, name), (rows, items) in eval_sets.items()
            if source in wanted
        }

    def run(regime, seed):
        folder = args.out / f"{regime}-s{seed}"
        done, adapter = folder / "run.pt", folder / "adapter.safetensors"
        if done.exists() and adapter.exists():
            saved = torch.load(done)
            if saved["config"] == run_config:
                return saved
        folder.mkdir(exist_ok=True, parents=True)
        sources = sources_of[regime]
        items = [it for s in sources for it in train_items[s]]
        tuned_on = [s for s in sources if s in tune_items]
        model = bench.model()
        add_lora(model, RANK, ALPHA, DROPOUT)
        config = TrainConfig(
            epochs=args.epochs,
            micro_batch=args.micro_batch,
            grad_accum=args.grad_accum,
            encoder_lr=args.encoder_lr,
            head_lr=args.head_lr,
            loss="soft-ce",
            shuffle_options=("choice",),
            gradient_checkpointing=False,
            seed=seed,
            log_every=400,
        )
        best = {"loss": float("inf"), "epoch": None, "state": None}
        history = []

        def on_epoch_end(epoch, loss):
            per = {s: tune_loss(score(bench, model, tune_items[s]), tune_items[s]) for s in tuned_on}
            mean = sum(per.values()) / len(per)
            history.append({"epoch": epoch + 1, "train_loss": loss, "tune_loss": mean, "per_source": per})
            print(regime, seed, history[-1], flush=True)
            if mean < best["loss"]:
                best.update(loss=mean, epoch=epoch + 1, state=trainable_state(model))

        train_model(
            model,
            bench.tok,
            items,
            config,
            bench.device,
            bench.max_len,
            bench.head_max_len,
            on_epoch_end=on_epoch_end,
        )
        model.load_state_dict(best["state"], strict=False)
        result = {
            "config": run_config,
            "regime": regime,
            "seed": seed,
            "epoch": best["epoch"],
            "tune_loss": best["loss"],
            "history": history,
            "train_items": len(items),
            "scores": score_all(model, regime),
        }
        meta = {
            "base": BASE,
            "base_revision": REVISION,
            "regime": regime,
            "seed": str(seed),
            "rank": str(RANK),
            "alpha": str(ALPHA),
        }
        save_file({k: v.contiguous() for k, v in best["state"].items()}, str(adapter), metadata=meta)
        torch.save(result, done)
        del model
        return result

    model = bench.model()
    base_scores = {
        f"{source}/{name}": score(bench, model, items) for (source, name), (rows, items) in eval_sets.items()
    }
    del model
    runs = {r: {s: run(r, s) for s in args.seeds} for r in args.regimes}

    def measured(key, logits):
        source, name = key.split("/")
        return evaluate(source, name, logits, eval_sets[(source, name)][0])

    report = {
        "base": BASE,
        "base_revision": REVISION,
        "config": run_config,
        "seeds": args.seeds,
        "regimes": args.regimes,
        "counts": {
            "train": {s: len(v) for s, v in train_items.items()},
            "eval": {f"{s}/{n}": len(v[0]) for (s, n), v in eval_sets.items()},
        },
        "upstream": {k: measured(k, v) for k, v in base_scores.items()},
        "runs": {},
        "mean_of_seeds": {},
    }
    lines = [f"train={report['counts']['train']}"]
    for regime in args.regimes:
        report["runs"][regime] = {
            str(seed): {
                "epoch": r["epoch"],
                "tune_loss": r["tune_loss"],
                "history": r["history"],
                "train_items": r["train_items"],
                "evaluation": {k: measured(k, v) for k, v in r["scores"].items()},
            }
            for seed, r in runs[regime].items()
        }
        keys = list(runs[regime][args.seeds[0]]["scores"])
        report["mean_of_seeds"][regime] = {
            k: measured(k, torch.stack([runs[regime][s]["scores"][k] for s in args.seeds]).mean(0)) for k in keys
        }
        for k in keys:
            source, name = k.split("/")
            per_seed = [headline(source, name, report["runs"][regime][str(s)]["evaluation"][k]) for s in args.seeds]
            lines.append(
                f"{regime} {k} upstream[{headline(source, name, report['upstream'][k])}] seeds{per_seed} "
                f"mean_scores[{headline(source, name, report['mean_of_seeds'][regime][k])}]"
            )
        lines.append(f"{regime} epochs={[runs[regime][s]['epoch'] for s in args.seeds]}")
    # Publish the unified seed (or the last regime's) with the lowest tuning loss as a merged checkpoint.
    main_regime = "unified" if "unified" in args.regimes else args.regimes[-1]
    best_seed = min(args.seeds, key=lambda s: runs[main_regime][s]["tune_loss"])
    model = apply_adapter(bench.model(), args.out / f"{main_regime}-s{best_seed}" / "adapter.safetensors")
    merge_lora(model)
    final = args.out / f"{main_regime}-final"
    bench.save_checkpoint(model, final)
    del model
    model = bench.model(final)
    reloaded = score_all(model, main_regime)
    del model
    report["selected_seed"] = best_seed
    report["saved_checkpoint_check"] = {
        k: {
            "in_memory": headline(*k.split("/"), measured(k, runs[main_regime][best_seed]["scores"][k])),
            "reloaded": headline(*k.split("/"), measured(k, v)),
        }
        for k, v in reloaded.items()
    }
    changed = {k: v for k, v in report["saved_checkpoint_check"].items() if v["in_memory"] != v["reloaded"]}
    lines.append(f"selected_seed={best_seed} saved_checkpoint_differs={changed}")
    args.result.parent.mkdir(exist_ok=True, parents=True)
    args.result.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.summary:
        args.summary.write_text(text)


if __name__ == "__main__":
    main()
