"""Time one checkpoint on the inference questions: one question at a time, and in batches, on GPU and CPU."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

import lora_nli


def timed(bench, model, items, batch, device):
    bench.device = torch.device(device)
    model.to(bench.device)
    bench.score(model, items[: batch * 2], batch)  # warm up
    spans = []
    for start in range(0, len(items), batch):
        if bench.device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        bench.score(model, items[start : start + batch], batch)
        if bench.device.type == "cuda":
            torch.cuda.synchronize()
        spans.append(time.perf_counter() - t)
    spans.sort()
    return {
        "batch": batch,
        "questions": len(items),
        "median_ms_per_call": round(1000 * statistics.median(spans), 2),
        "p95_ms_per_call": round(1000 * spans[int(0.95 * (len(spans) - 1))], 2),
        "questions_per_second": round(len(items) / sum(spans), 1),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path(".cache/data"))
    ap.add_argument("--corpora", type=Path, default=Path(".cache/corpora"))
    ap.add_argument("--checkpoint", type=Path, help="folder laya.load reads; the upstream base when omitted")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument("--cpu-threads", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--model-path")
    ap.add_argument("--device")
    args = ap.parse_args()
    args.teacher, args.max_train = None, None
    args.extra_cap, args.logic_cap, args.teacher_cap, args.teacher_all = 4000, 6000, 8000, False
    bench = lora_nli.Bench(args)
    sets = lora_nli.corpora(args)[2]
    rows = (sets["jnli"][: args.count // 2] + sets["jsem"][: args.count // 2])[: args.count]
    items = bench.items(rows)
    result = {"checkpoint": str(args.checkpoint or "upstream"), "runs": []}
    for name, path in (("laya-ja", args.checkpoint), ("laya-upstream", None)):
        if name == "laya-ja" and path is None:
            continue
        model = bench.model(path).float()
        if torch.cuda.is_available():
            for batch in (1, 64):
                result["runs"].append(
                    {
                        "model": name,
                        "device": torch.cuda.get_device_name(0),
                        **timed(bench, model, items, batch, "cuda"),
                    }
                )
        for threads in args.cpu_threads:
            torch.set_num_threads(threads)
            run = timed(bench, model, items[: max(40, args.count // 5)], 1, "cpu")
            result["runs"].append({"model": name, "device": f"cpu x{threads}", **run})
        del model
    for r in result["runs"]:
        print(r, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
