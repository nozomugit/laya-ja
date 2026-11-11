"""Train low-rank updates on every Laya encoder layer for Japanese inference; compare corpora, seeds and the frozen base."""

import argparse
import gzip
import hashlib
import json
import os
import shutil
from pathlib import Path

import laya.train as laya_train
import numpy as np
import torch
from laya.common import QTYPES, collate_items, uses_parallel_layout
from laya.train import (
    TrainConfig,
    encode_item,
    encode_state,
    load_checkpoint,
    make_item,
    to_internal,
    train_model,
)
from safetensors.torch import save_file

from corpus_data import SEED as CORPUS_SEED
from corpus_data import jsem, load_tsv, normalized, pair_key, shuffled, splits, unique
from corpus_experiment import measures
from experiment import bootstrap_difference, partition
from laya_ja import BASE, REVISION, nli_request
from logic_worlds import generate as logic_rows
from lora import add_lora, apply_adapter, merge_lora, trainable_state

BASE_SHA256 = "9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204"
MIX4 = ["jnli", "jsick", "janli", "jamp"]
REGIMES = {"mix4": MIX4, "jnli": ["jnli"], "mix4-logic": [*MIX4, "logic"]}
TRAIN_FILES = {
    "jsick": "JSICK/jsick/train.tsv",
    "janli": "JaNLI/janli.tsv",
    "jamp": "temporalNLI_dataset/dataset/train_custom-template-hard.tsv",
}
RANK, ALPHA, DROPOUT = 16, 32, 0.05


def premise(row):
    return normalized(row[0][0].split("\n仮説：")[0])


def jsem_fresh(folder):
    """The 96 premise-disjoint JSeM problems fixed by semantic_inputs.py, checked against their hashes."""
    root = Path(__file__).parent / "tools/four-task/semantic-manifest.json"
    spec = json.loads(root.read_text())["jsem-fresh"]
    pool = {r[3]: r for r in jsem(folder)}
    rows = []
    for _, ident, digest in spec:
        row = pool[ident]
        p, h = row[0][0].removeprefix("前提：").split("\n仮説：", 1)
        if hashlib.sha256((p + "\n" + h).encode()).hexdigest() != digest:
            raise ValueError(f"JSeM text changed: {ident}")
        rows.append(row)
    return rows


def logic(split, count, seed):
    return [
        (nli_request(p, h), y, "logic", ident)
        for p, h, y, ident in logic_rows(split, count, seed)
    ]


def corpora(args):
    """Evaluation and tuning rows match corpus_data.splits; training rows share no sentence pair with them."""
    multi = splits(args.corpora, args.data)
    d = partition("jnli", args.data, 42, 10**8, 192, 10**8)
    pool = d["train"]
    used = {r[2] for r in pool[:4096]}
    reserved = [r for r in pool[4096:] if r[2] not in used][:256]
    reserved_groups = {r[2] for r in reserved}
    probes = json.loads(Path(__file__).with_name("challenge.json").read_text())
    evalsets = {
        "jnli": d["eval"],
        "reserved": reserved,
        "jsick": multi["jsick-eval"],
        "janli": multi["janli-eval"],
        "jamp": multi["jamp-eval"],
        "jsem": multi["jsem-eval"],
        "jsem-fresh": jsem_fresh(args.corpora),
        "logic-held": logic("held", 300, 3003),
        "challenge": [
            (nli_request(p, h), y, cat, str(i))
            for i, (cat, p, h, y) in enumerate(probes)
        ],
    }
    tune = {"jnli": d["tune"], **{s: multi[s + "-tune"] for s in TRAIN_FILES}}
    tune["logic"] = logic("tune", 192, 2002)
    block = {
        pair_key(r[0][0]) for rows in [*evalsets.values(), *tune.values()] for r in rows
    }
    train = {
        "jnli": [
            r
            for r in pool
            if r[2] not in reserved_groups and pair_key(r[0][0]) not in block
        ]
    }
    for name, rel in TRAIN_FILES.items():
        rows = unique(
            shuffled(load_tsv(args.corpora / rel, name, "train"), CORPUS_SEED + 1),
            block,
        )
        tune_premises = {premise(r) for r in tune[name]}
        rows = [r for r in rows if premise(r) not in tune_premises]
        train[name] = rows[: args.extra_cap] if args.extra_cap else rows
    train["logic"] = [
        r
        for r in logic("train", args.logic_cap, 1001)
        if pair_key(r[0][0]) not in block
    ]
    if args.max_train:
        train = {s: rows[: args.max_train] for s, rows in train.items()}
    return train, tune, evalsets


def set_likelihood_loss(logits, target, mask):
    """weight * -log(total probability of the marked options); one mark is ordinary cross-entropy."""
    logp = torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)
    weight = target.max(-1).values
    marked = logp.masked_fill(target <= 0, float("-inf"))
    return -(weight * torch.logsumexp(marked, -1)).mean()


class Bench:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.base_dir = args.model_path or self.download()
        _, self.tok, self.cfg = load_checkpoint(self.base_dir)
        if uses_parallel_layout(self.cfg):
            raise ValueError("Parallel option layout is not supported here.")
        self.max_len = self.cfg.get("max_len", 512)
        self.head_max_len = self.cfg.get("head_max_len", 192)

    @staticmethod
    def download():
        from huggingface_hub import snapshot_download

        folder = snapshot_download(
            BASE,
            revision=REVISION,
            allow_patterns=[
                "rl_agent_config.json",
                "model.safetensors",
                "tokenizer/*",
                "encoder/*",
            ],
        )
        h = hashlib.sha256()
        with open(os.path.join(folder, "model.safetensors"), "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != BASE_SHA256:
            raise ValueError("Unexpected base checkpoint.")
        return folder

    def model(self, path=None):
        m, _, _ = load_checkpoint(str(path or self.base_dir))
        return m.to(self.device).eval()

    def items(self, rows, train=False):
        # Target = weight * marks. JaNLI non-entailment (label 3) marks contradiction and neutral.
        out, skipped = [], 0
        for (state, instruction, choices), y, _, _ in rows:
            question = {
                "type": "choice",
                "instructions": instruction,
                "criteria": dict.fromkeys(choices, ""),
            }
            q = to_internal("decision", question)
            target = [0.0, 1.0, 1.0] if y == 3 else [float(i == y) for i in range(3)]
            if train and y == 0:
                target = [v * self.args.entailment_weight for v in target]
            state_ids = encode_state(self.tok, state, self.max_len)
            item, reason = make_item(
                self.tok, q, target, state_ids, self.head_max_len, self.max_len
            )
            if item is None and not train:
                if reason != "options_collapsed":
                    raise ValueError(f"Evaluation row cannot be encoded: {reason}")
                item = {
                    "q": q,
                    "state_ids": state_ids,
                    "target": target,
                    "qtype": QTYPES["choice"],
                    "k": 3,
                }
            if item is None:
                skipped += 1
            else:
                out.append(item)
        if skipped:
            print("skipped training rows:", skipped, flush=True)
        return out

    @torch.no_grad()
    def score(self, model, items, batch_size=64):
        was_training = model.training
        model.eval()
        order = sorted(range(len(items)), key=lambda i: len(items[i]["state_ids"]))
        out = [None] * len(items)
        keys = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
        for start in range(0, len(items), batch_size):
            idx = order[start : start + batch_size]
            chunk = [
                encode_item(self.tok, items[i], self.max_len, self.head_max_len)
                for i in idx
            ]
            b = collate_items([chunk], self.tok.pad_token_id)
            logits, _ = model(*(b[k].to(self.device) for k in keys))
            for row, i in zip(logits.float().cpu(), idx):
                out[i] = row[:3]
        model.train(was_training)
        return torch.stack(out)

    def save_checkpoint(self, model, folder):
        """Write the layout laya.load reads (fp16, as the upstream release)."""
        folder.mkdir(exist_ok=True, parents=True)
        save_file(
            {
                k: v.detach().half().cpu().contiguous()
                for k, v in model.state_dict().items()
            },
            str(folder / "model.safetensors"),
        )
        model.encoder.config.save_pretrained(str(folder / "encoder"))
        self.tok.save_pretrained(str(folder / "tokenizer"))
        cfg = dict(self.cfg, max_len=self.max_len, head_max_len=self.head_max_len)
        (folder / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key, default in (
        ("data", ".cache/data"),
        ("corpora", ".cache/corpora"),
        ("out", ".cache/lora-nli"),
        ("result", "results/lora-nli.json"),
    ):
        ap.add_argument("--" + key, type=Path, default=Path(default))
    ap.add_argument("--regimes", nargs="+", choices=list(REGIMES), default=["mix4"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--encoder-lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--extra-cap", type=int, default=4000)
    ap.add_argument("--entailment-weight", type=float, default=1.0)
    ap.add_argument("--logic-cap", type=int, default=6000)
    ap.add_argument("--fusion", type=Path, default=Path("results/four-dense.jsonl.gz"))
    ap.add_argument("--max-train", type=int)
    ap.add_argument("--previous", type=Path, help="Checkpoint to score as a reference")
    ap.add_argument(
        "--reuse",
        type=Path,
        help="Folder of notebook runs (nli-<regime>-lora-s<seed>/adapter.safetensors) to score instead of retraining",
    )
    ap.add_argument("--model-path")
    ap.add_argument("--device")
    args = ap.parse_args()

    laya_train.soft_ce_loss = set_likelihood_loss
    bench = Bench(args)
    train, tune, evalsets = corpora(args)
    eval_items = {n: bench.items(rows) for n, rows in evalsets.items()}
    eval_y = {n: torch.tensor([r[1] for r in rows]) for n, rows in evalsets.items()}
    tune_items = {n: bench.items(rows) for n, rows in tune.items()}
    tune_y = {n: torch.tensor([r[1] for r in rows]) for n, rows in tune.items()}
    train_items = {n: bench.items(rows, train=True) for n, rows in train.items()}
    run_config = {
        k: getattr(args, k)
        for k in (
            "epochs",
            "encoder_lr",
            "head_lr",
            "micro_batch",
            "grad_accum",
            "extra_cap",
            "entailment_weight",
            "max_train",
        )
    }
    run_config["lora"] = [RANK, ALPHA, DROPOUT]

    def measure(name, logits, y=None):
        return measures(logits, eval_y[name] if y is None else y, name == "janli")

    def score_evalsets(model):
        return {n: bench.score(model, items) for n, items in eval_items.items()}

    def reference(path):
        model = bench.model(path)
        scores = score_evalsets(model)
        del model
        return scores

    def run(regime, seed):
        folder = args.out / f"{regime}-s{seed}"
        done, adapter = folder / "run.pt", folder / "adapter.safetensors"
        sources = REGIMES[regime]
        if done.exists() and adapter.exists():
            saved = torch.load(done)
            if saved["config"] == run_config and set(saved["scores"]) == set(evalsets):
                return saved
        earlier = (
            args.reuse / f"nli-{regime}-lora-s{seed}" / "adapter.safetensors"
            if args.reuse
            else None
        )
        if not adapter.exists() and earlier is not None and earlier.exists():
            folder.mkdir(exist_ok=True, parents=True)
            shutil.copyfile(earlier, adapter)
        if adapter.exists():
            # An adapter placed here by hand (for example from an earlier run) is scored, not retrained.
            model = apply_adapter(bench.model(), adapter)
            tuned = [
                measure(s, bench.score(model, tune_items[s]), tune_y[s])[0]["nll"]
                for s in sources
            ]
            result = {
                "config": run_config,
                "regime": regime,
                "seed": seed,
                "epoch": None,
                "tune_nll": sum(tuned) / len(tuned),
                "history": [],
                "train_items": None,
                "reused_adapter": True,
                "scores": score_evalsets(model),
            }
            torch.save(result, done)
            del model
            return result
        folder.mkdir(exist_ok=True, parents=True)
        items = [it for s in sources for it in train_items[s]]
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
            log_every=200,
        )
        best = {"nll": float("inf"), "epoch": None, "state": None}
        history = []

        def on_epoch_end(epoch, loss):
            scores = {
                s: measure(s, bench.score(model, tune_items[s]), tune_y[s])[0]
                for s in sources
            }
            nll = sum(m["nll"] for m in scores.values()) / len(scores)
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": loss,
                    "tune_nll": nll,
                    "tune_accuracy": {s: m["accuracy"] for s, m in scores.items()},
                }
            )
            print(regime, seed, history[-1], flush=True)
            if nll < best["nll"]:
                best.update(nll=nll, epoch=epoch + 1, state=trainable_state(model))

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
            "tune_nll": best["nll"],
            "history": history,
            "train_items": len(items),
            "scores": score_evalsets(model),
        }
        meta = {
            "base": BASE,
            "base_revision": REVISION,
            "regime": regime,
            "seed": str(seed),
            "rank": str(RANK),
            "alpha": str(ALPHA),
        }
        save_file(
            {k: v.contiguous() for k, v in best["state"].items()},
            str(adapter),
            metadata=meta,
        )
        torch.save(result, done)
        del model
        return result

    scores = {"base": reference(None)}
    if args.previous:
        scores["prev"] = reference(args.previous)
    runs = {r: {s: run(r, s) for s in args.seeds} for r in args.regimes}
    for regime in args.regimes:
        scores[regime] = {
            n: torch.stack([runs[regime][s]["scores"][n] for s in args.seeds]).mean(0)
            for n in evalsets
        }
    main_regime = args.regimes[-1]
    report = {
        "base": BASE,
        "base_revision": REVISION,
        "config": run_config,
        "seeds": args.seeds,
        "regimes": args.regimes,
        "counts": {
            "train": {s: len(v) for s, v in train_items.items()},
            "eval": {s: len(v) for s, v in evalsets.items()},
        },
        # "base" is the upstream checkpoint; a regime name is the mean of its seeds' scores.
        "models": {
            key: {n: measure(n, s)[0] for n, s in value.items()}
            for key, value in scores.items()
        },
        "runs": {
            regime: {
                str(seed): {
                    "epoch": r["epoch"],
                    "tune_nll": r["tune_nll"],
                    "history": r["history"],
                    "train_items": r["train_items"],
                    "reused_adapter": r.get("reused_adapter", False),
                    "evaluation": {n: measure(n, s)[0] for n, s in r["scores"].items()},
                }
                for seed, r in runs[regime].items()
            }
            for regime in args.regimes
        },
        "comparisons": {},
    }
    for ref in [k for k in ("prev", "base") if k in scores]:
        block = report["comparisons"][f"{main_regime}_vs_{ref}"] = {}
        for name, rows in evalsets.items():
            diff = measure(name, scores[main_regime][name])[1].astype(int) - measure(
                name, scores[ref][name]
            )[1].astype(int)
            block[name] = {
                "difference": float(diff.mean()),
                "gained": int((diff == 1).sum()),
                "lost": int((diff == -1).sum()),
                "group_bootstrap_95pct": bootstrap_difference(
                    diff, np.array([r[2] for r in rows])
                ),
            }
    # Replace only the 10% Laya share of the fixed mixture in dense_transfer.py (10% Laya, 45% + 45% direct
    # judges). The published rows give the two judges' probabilities; the shares stay as published.
    if args.fusion.exists():
        published = {}
        for line in gzip.open(args.fusion, "rt"):
            r = json.loads(line)
            published.setdefault(r["split"], []).append(r)
        report["fusion"] = {}
        for name in ("jsem", "challenge", "jsem-fresh"):
            rows = sorted(published[name], key=lambda r: r["row"])
            gold = torch.tensor([r["gold"] for r in rows])
            if gold.tolist() != eval_y[name].tolist():
                raise ValueError(f"Published rows do not align with {name}")
            dense, single, dual = (
                torch.tensor([r["probabilities"][k] for r in rows], dtype=torch.float64)
                for k in ("dense_only", "single", "dual")
            )
            native = (single - 0.9 * dense) / 0.1
            sparse = (dual - 0.1 * native - 0.45 * dense) / 0.45
            judges = 0.45 * dense + 0.45 * sparse

            def correct(p):
                return int((p.argmax(-1) == gold).sum())

            block = {
                "n": len(rows),
                "published_laya_alone": correct(native),
                "published_mixture": correct(dual),
                "judges_without_laya": correct(judges),
            }
            for key, value in scores.items():
                p = value[name].double().softmax(-1)
                block[key] = {
                    "laya_alone": correct(p),
                    "mixture": correct(0.1 * p + judges),
                }
            for regime in args.regimes:
                for seed in args.seeds:
                    p = runs[regime][seed]["scores"][name].double().softmax(-1)
                    block[f"{regime}-s{seed}"] = {
                        "laya_alone": correct(p),
                        "mixture": correct(0.1 * p + judges),
                    }
            report["fusion"][name] = block
    y = eval_y["jnli"]
    report["entailment_predictions"] = {
        key: [int((value["jnli"][y == 0].argmax(-1) == i).sum()) for i in range(3)]
        for key, value in scores.items()
    }

    # Publish the seed with the lowest tuning loss; the official sets take no part in the choice.
    best_seed = min(args.seeds, key=lambda s: runs[main_regime][s]["tune_nll"])
    model = apply_adapter(
        bench.model(), args.out / f"{main_regime}-s{best_seed}" / "adapter.safetensors"
    )
    merge_lora(model)
    final = args.out / f"{main_regime}-final"
    bench.save_checkpoint(model, final)
    del model
    reloaded = reference(final)
    report["selected_seed"] = best_seed
    report["saved_checkpoint_check"] = {
        n: {
            "in_memory": measure(n, runs[main_regime][best_seed]["scores"][n])[0][
                "accuracy"
            ],
            "reloaded": measure(n, reloaded[n])[0]["accuracy"],
        }
        for n in evalsets
    }
    args.result.parent.mkdir(exist_ok=True, parents=True)
    args.result.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for name in evalsets:
        per_seed = [
            report["runs"][main_regime][str(s)]["evaluation"][name]["accuracy"]
            for s in args.seeds
        ]
        print(
            f"{name:10s} base {report['models']['base'][name]['accuracy']:.3f} "
            f"{main_regime} {np.mean(per_seed):.3f} +/- {np.std(per_seed):.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
