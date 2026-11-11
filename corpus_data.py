"""Pinned Japanese inference corpora. Keep upstream splits and score JaNLI as binary."""

import csv
import hashlib
import json
import random
import subprocess
import unicodedata
from pathlib import Path
from xml.etree import ElementTree

from experiment import partition
from laya_ja import nli_request

REPOS = {
    "JSICK": ("verypluming/JSICK", "b3034994192fae2f41b5937bcf69544e4282fc39"),
    "JaNLI": ("verypluming/JaNLI", "2bbb63622570f7b20abc118146141b501efe5183"),
    "temporalNLI_dataset": (
        "tomo-ut/temporalNLI_dataset",
        "f578b5daac7bff3855d238c032a60fd047cddd9c",
    ),
    "JSeM": ("DaisukeBekki/JSeM", "c2cd39168e267007b4690dbf625329bac57a8185"),
}
LABEL = {"entailment": 0, "contradiction": 1, "neutral": 2, "non-entailment": 3}
SEED = 6801


def checkout(folder):
    folder.mkdir(exist_ok=True, parents=True)
    for name, (repo, rev) in REPOS.items():
        path = folder / name
        if not path.exists():
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--no-checkout",
                    f"https://github.com/{repo}.git",
                    str(path),
                ],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(path), "checkout", "--detach", rev], check=True
            )
        if (
            subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
            ).strip()
            != rev
        ):
            raise ValueError(f"Expected {name} at {rev}")
        subprocess.run(
            ["git", "-C", str(path), "diff", "--quiet", "HEAD", "--"], check=True
        )


def normalized(text):
    return "".join(unicodedata.normalize("NFKC", text).split())


def pair_key(state):
    p, h = state.removeprefix("前提：").split("\n仮説：", 1)
    return tuple(sorted([normalized(p), normalized(h)]))


def load_tsv(path, source, part):
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if source == "janli" and r["split"] != part:
                continue
            if source == "jamp":
                p, h, label, idx, group = (
                    r[k]
                    for k in (
                        "premise",
                        "hypothesis",
                        "gold_label",
                        "num",
                        "template_num",
                    )
                )
            else:
                p, h, label = (
                    r[k]
                    for k in ("sentence_A_Ja", "sentence_B_Ja", "entailment_label_Ja")
                )
                idx = r[""] if source == "janli" else r["pair_ID"]
                group = r["semtag"] if source == "janli" else (r.get("image_ID") or idx)
            if p.strip() and h.strip():
                yield (nli_request(p, h), LABEL[label], str(group), str(idx))


def jsem(folder):
    labels = {"yes": 0, "no": 1, "unknown": 2}
    for f in sorted((folder / "JSeM/data/v1.0").glob("*.xml")):
        for r in ElementTree.parse(f).getroot().findall("problem"):
            if (
                r.get("language") != "jp"
                or r.get("inference_type") != "entailment"
                or r.get("answer") not in labels
            ):
                continue
            ps = ["".join(p.itertext()).strip() for p in r.findall("p/script")]
            h = r.findtext("h/script")
            if ps and all(ps) and h and h.strip():
                yield (
                    nli_request(" ".join(ps), h.strip()),
                    labels[r.get("answer")],
                    f.stem,
                    f.stem + ":" + r.get("jsem_id", ""),
                )


def shuffled(rows, seed):
    result = list(rows)
    random.Random(seed).shuffle(result)
    return result


def unique(rows, blocked):
    seen = set(blocked)
    for row in rows:
        k = pair_key(row[0][0])
        if k not in seen:
            yield row
            seen.add(k)


def splits(folder, data):
    checkout(folder)
    j = partition("jnli", data, 42, 4096, 192, 10**9)
    first_keys = {pair_key(r[0][0]) for part in j.values() for r in part}
    inputs = {
        "jsick": (folder / "JSICK/jsick/train.tsv", folder / "JSICK/jsick/test.tsv"),
        "janli": (folder / "JaNLI/janli.tsv", folder / "JaNLI/janli.tsv"),
        "jamp": (
            folder / "temporalNLI_dataset/dataset/train_custom-template-hard.tsv",
            folder / "temporalNLI_dataset/dataset/test_2.tsv",
        ),
    }
    # All independent eval inputs are reserved before selecting any training rows.
    blocked = set(first_keys)
    probes = json.loads(Path(__file__).with_name("challenge.json").read_text())
    blocked.update(pair_key(nli_request(p, h)[0]) for _, p, h, _ in probes)
    out = {}
    for name, (_, test) in inputs.items():
        candidates = list(unique(shuffled(load_tsv(test, name, "test"), SEED), blocked))
        out[name + "-eval"] = candidates[:512]
        blocked.update(pair_key(r[0][0]) for r in candidates)
    candidates = list(unique(shuffled(jsem(folder), SEED), blocked))
    out["jsem-eval"] = [
        r for y in range(3) for r in [v for v in candidates if v[1] == y][:128]
    ]
    blocked.update(pair_key(r[0][0]) for r in out["jsem-eval"])
    for name, (train, _) in inputs.items():
        rows = list(unique(shuffled(load_tsv(train, name, "train"), SEED + 1), blocked))
        tune = rows[:192]
        tune_p = {normalized(r[0][0].split("\n仮説：")[0]) for r in tune}
        fit = [
            r
            for r in rows[len(tune) :]
            if normalized(r[0][0].split("\n仮説：")[0]) not in tune_p
        ][:1536]
        out[name + "-tune"] = tune
        out[name + "-train"] = fit
        blocked.update(pair_key(r[0][0]) for r in tune + fit)
    return out


def encode(args):
    import numpy as np
    import torch
    from experiment import extract
    from laya_ja import REVISION, load_base

    torch.set_num_threads(args.threads)
    agent = load_base(args.model_path, "cpu")
    args.cache.mkdir(exist_ok=True, parents=True)
    for name, rows in splits(args.corpora, args.data).items():
        fingerprint = hashlib.sha256(
            json.dumps([REVISION, rows], ensure_ascii=False).encode()
        ).hexdigest()
        path = args.cache / f"{name}.npz"
        if path.exists():
            with np.load(path) as old:
                if old["fingerprint"].item() == fingerprint:
                    continue
        print(name, len(rows), flush=True)
        x = extract(agent, rows, args.batch_size)
        temp = path.with_suffix(".tmp.npz")
        np.savez(
            temp,
            x=x.numpy(),
            y=[r[1] for r in rows],
            group=[r[2] for r in rows],
            ids=[r[3] for r in rows],
            fingerprint=fingerprint,
        )
        temp.replace(path)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    for key, default in (
        ("corpora", ".cache/corpora"),
        ("data", ".cache/data"),
        ("cache", ".cache/multi"),
    ):
        p.add_argument("--" + key, type=Path, default=Path(default))
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--model-path")
    encode(p.parse_args())
