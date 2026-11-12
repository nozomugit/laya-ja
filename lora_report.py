"""Print a compact per-seed table from one or more lora_nli.py result files."""

import argparse
import json
from pathlib import Path


def summarize(path):
    r = json.loads(Path(path).read_text())
    lines = [f"# {Path(path).name} train={r['counts']['train']}"]
    names = list(r["counts"]["eval"])
    for regime, runs in r["runs"].items():
        for name in names:
            correct = [runs[s]["evaluation"][name]["correct"] for s in runs]
            n = r["counts"]["eval"][name]
            mean = r["models"][regime][name]
            recall = " ".join(f"{v:.3f}" for v in mean["recall"].values())
            lines.append(
                f"{regime} {name} n={n} seeds={correct} "
                f"avg={100 * sum(correct) / len(correct) / n:.1f} "
                f"mean_scores={mean['correct']} recall={recall}"
            )
        tuned = {s: (v["epoch"], round(v["tune_nll"], 4)) for s, v in runs.items()}
        lines.append(f"{regime} epoch_and_tune_nll={tuned}")
    for key in ("base", "prev"):
        if key in r["models"]:
            row = {n: r["models"][key][n]["correct"] for n in names}
            lines.append(f"{key} {row}")
    for name, block in r.get("fusion", {}).items():
        cells = {
            k: (v["laya_alone"], v["mixture"])
            for k, v in block.items()
            if isinstance(v, dict)
        }
        lines.append(
            f"fusion {name} published={block['published_mixture']} "
            f"judges_only={block['judges_without_laya']} {cells}"
        )
    lines.append(f"entailment_predictions={r['entailment_predictions']}")
    lines.append(f"selected_seed={r['selected_seed']}")
    return lines


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", nargs="+")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    text = "\n".join(line for p in args.results for line in summarize(p)) + "\n"
    print(text)
    if args.out:
        args.out.write_text(text)
