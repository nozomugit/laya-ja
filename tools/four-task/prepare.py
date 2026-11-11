"""Retrieve the fixed public inputs by ID and verify text before inference."""

import argparse
import csv
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from four_task_prompts import nli_prompt

RAW = "https://raw.githubusercontent.com/"
PATHS = {
    "jnli": "yahoojapan/JGLUE/6f071c09316baae89c3d083a90985b4b1cb9968c/datasets/jnli-v1.3/train-v1.3.json",
    "jsick": "verypluming/JSICK/b3034994192fae2f41b5937bcf69544e4282fc39/jsick/train.tsv",
    "jsem": "DaisukeBekki/JSeM/c2cd39168e267007b4690dbf625329bac57a8185/data/v1.0/",
}


def retrieve(args, name, url):
    p = args.cache / name
    if not p.exists():
        with urllib.request.urlopen(RAW + url, timeout=90) as response:
            tmp = p.with_suffix(".download")
            tmp.write_bytes(response.read())
            tmp.replace(p)
    return p


def inputs(a, manifest):
    ix = {}
    with retrieve(a, "jnli-train.json", PATHS["jnli"]).open() as f:
        for r in map(json.loads, f):
            ix["jnli", str(r["sentence_pair_id"])] = (r["sentence1"], r["sentence2"])
    with retrieve(a, "jsick.tsv", PATHS["jsick"]).open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            ix["jsick", r["pair_ID"]] = (r["sentence_A_Ja"], r["sentence_B_Ja"])
    stems = {
        i.split(":")[0]
        for rows in manifest.values()
        for source, i, _ in rows
        if source == "jsem"
    }
    for stem in sorted(stems):
        name = stem + ".xml"
        path = retrieve(a, name, PATHS["jsem"] + urllib.parse.quote(name))
        for r in ElementTree.parse(path).getroot().findall("problem"):
            if r.get("language") != "jp" or r.get("inference_type") != "entailment":
                continue
            p = " ".join("".join(n.itertext()).strip() for n in r.findall("p/script"))
            h = r.findtext("h/script")
            if h:
                ix["jsem", stem + ":" + r.get("jsem_id", "")] = (p, h.strip())
    for i, (_, p, h, _) in enumerate(json.loads((ROOT / "challenge.json").read_text())):
        ix["challenge", str(i)] = (p, h)
    if "semantic-cal" in manifest:
        from semantic_worlds import generate

        ix.update({("world", k): (p, h) for k, (p, h, _, _) in generate().items()})
    return ix


def run(a):
    a.cache.mkdir(parents=True, exist_ok=True)
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(__file__).with_name("manifest.json").read_text())
    if a.semantic:
        manifest.update(
            json.loads(Path(__file__).with_name("semantic-manifest.json").read_text())
        )
    index = inputs(a, manifest)
    n = 0
    with (
        (a.out / "direct.jsonl").open("w") as f,
        (a.out / "contrast.jsonl").open("w") as g,
        (a.out / "gemma.jsonl").open("w") as gemma,
        (a.out / "reason.jsonl").open("w") as reason,
        (a.out / "must_true.jsonl").open("w") as truth,
        (a.out / "must_false.jsonl").open("w") as falsity,
    ):
        for split, rows in manifest.items():
            for i, (source, ident, digest) in enumerate(rows):
                p, h = index[source, ident]
                if hashlib.sha256((p + "\n" + h).encode()).hexdigest() != digest:
                    raise ValueError(f"Input mismatch at {split}/{i}: {ident}")
                if n % a.shards == a.shard:
                    if a.semantic:
                        from semantic_probes import probe

                        for dest, kind in (
                            (truth, "must_true"),
                            (falsity, "must_false"),
                        ):
                            dest.write(
                                json.dumps(
                                    {"id": [split, i], "prompt": probe(p, h, kind)},
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                    from gemma_prompts import prompt as gemma_prompt

                    gemma.write(
                        json.dumps(
                            {"id": [split, i], "prompt": gemma_prompt(p, h)},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    direct = nli_prompt(p, h)
                    suffix = "<think>\n\n</think>\n\n"
                    assert direct.endswith(suffix)
                    prompt = direct[: -len(suffix)] + "<think>\n"
                    reason.write(
                        json.dumps(
                            {"id": [split, i], "prompt": prompt}, ensure_ascii=False
                        )
                        + "\n"
                    )
                    for fp, contrast in ((f, False), (g, True)):
                        fp.write(
                            json.dumps(
                                {
                                    "id": [split, i],
                                    "prompt": nli_prompt(p, h, contrast),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                n += 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=Path(".cache/four-public"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--semantic", action="store_true")
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    a = p.parse_args()
    if not 0 <= a.shard < a.shards:
        raise ValueError("shard outside valid range")
    run(a)
