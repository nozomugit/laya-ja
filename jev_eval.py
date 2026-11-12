"""Ask TypeSafe's hosted Jev the evaluation questions exactly as Laya is asked them, and score the answers the same way.

Jev is the closed model whose interface Laya reproduces. The request carries the same state, instructions and
criteria; the returned probabilities are scored with the functions used for the LoRA results. The API key is read
from the environment variable TYPESAFE_API_KEY (or a Colab secret) and is never written anywhere.
"""

import argparse
import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

import lora_nli
import lora_tasks

URL = "https://api.typesafe.ai/v1/systemone"
FLOOR = 1e-6
BACKOFF = float(os.environ.get("JEV_BACKOFF", "1"))


def question(source, instruction, options):
    if source == "jsts":
        return {"type": "score", "instructions": instruction, "criteria": list(options)}
    if source == "jcola":
        return {"type": "noul", "instructions": instruction, "criteria": {"false": options[0], "true": options[1]}}
    return {"type": "choice", "instructions": instruction, "criteria": dict.fromkeys(options)}


def probabilities(source, options, answer):
    if source == "jcola":
        p = float(answer["noul"])
        return [1.0 - p, p]
    given = answer["probabilities"]
    if source == "jsts":
        return [float(given[str(i)]) for i in range(len(options))]
    return [float(given[o]) for o in options]


class Client:
    def __init__(self, url, model, key, attempts=8):
        self.url, self.model, self.key, self.attempts = url, model, key, attempts
        self.lock = threading.Lock()
        self.tokens = 0
        self.resolved = set()

    def ask(self, state, q):
        body = json.dumps({"model": self.model, "state": state, "questions": {"q": q}}).encode()
        for attempt in range(self.attempts):
            request = urllib.request.Request(
                self.url,
                data=body,
                headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    reply = json.loads(response.read())
                with self.lock:
                    self.tokens += int(reply.get("usage", {}).get("input_tokens", 0))
                    self.resolved.add(reply.get("model"))
                return reply["answers"]["q"]
            except urllib.error.HTTPError as error:
                if error.code not in (408, 429, 500, 502, 503, 504, 529):
                    detail = error.read().decode("utf-8", "replace")[:300]
                    raise SystemExit(f"Jev refused the request: HTTP {error.code} {detail}") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                pass
            time.sleep(min(60, 2**attempt) * BACKOFF)
        raise SystemExit("Jev did not answer after repeated attempts; run again to resume.")


def collect(client, source, name, rows, folder, workers):
    """One line per problem, in problem order; finished problems are kept when the run is repeated."""
    path = folder / f"{source}-{name}.jsonl"
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done[r["index"]] = r["probabilities"]
    todo = [i for i in range(len(rows)) if i not in done]
    lock = threading.Lock()

    def one(i):
        (state, instruction, options), _, _, key = rows[i]
        answer = client.ask(state, question(source, instruction, options))
        p = probabilities(source, options, answer)
        with lock:
            done[i] = p
            with path.open("a") as f:
                f.write(json.dumps({"index": i, "id": str(key), "probabilities": p}) + "\n")

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, todo))
    return torch.tensor([[math.log(max(v, FLOOR)) for v in done[i]] for i in range(len(rows))])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path(".cache/data"))
    ap.add_argument("--corpora", type=Path, default=Path(".cache/corpora"))
    ap.add_argument("--out", type=Path, default=Path(".cache/jev"))
    ap.add_argument("--result", type=Path, default=Path("results/jev.json"))
    ap.add_argument("--summary", type=Path)
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("--url", default=URL)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--secret", default="JEV_API_KEY", help="name of the Colab secret used when the variable is unset")
    ap.add_argument("--limit", type=int, help="first N problems of each set, for a trial run")
    ap.add_argument("--only", nargs="+", help="sources to ask: nli, jcommonsenseqa, jsts, jcola")
    args = ap.parse_args()
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        try:  # In Colab, started with %run so that the notebook's secret store is reachable.
            from google.colab import userdata

            key = userdata.get(args.secret)
        except Exception as error:
            raise SystemExit(
                f"Set TYPESAFE_API_KEY, or the Colab secret {args.secret}: {type(error).__name__}"
            ) from None
    args.teacher, args.max_train = None, None
    args.extra_cap, args.logic_cap, args.teacher_cap, args.teacher_all = 4000, 6000, 8000, False
    sets = {}
    if not args.only or "nli" in args.only:
        for name, rows in lora_nli.corpora(args)[2].items():
            sets[("nli", name)] = rows
    for task in lora_tasks.TASKS:
        if not args.only or task in args.only:
            for name, rows in lora_tasks.task_rows(task, args.data)[2].items():
                sets[(task, name)] = rows
    folder = args.out / args.model
    folder.mkdir(parents=True, exist_ok=True)
    client = Client(args.url, args.model, key)
    evaluation, lines = {}, []
    for (source, name), rows in sets.items():
        rows = rows[: args.limit] if args.limit else rows
        logits = collect(client, source, name, rows, folder, args.workers)
        m = lora_tasks.evaluate(source, name, logits, rows)
        evaluation[f"{source}/{name}"] = m
        text = f"{source}/{name} n={len(rows)} {lora_tasks.headline(source, name, m)}"
        if "recall" in m:
            text += " recall=" + " ".join(
                f"{v:.3f}" for v in (m["recall"].values() if isinstance(m["recall"], dict) else m["recall"])
            )
        lines.append(text)
        print(text, flush=True)
    result = {
        "requested_model": args.model,
        "answered_by": sorted(str(v) for v in client.resolved),
        "input_tokens_this_run": client.tokens,
        "limit": args.limit,
        "note": "Same state, instructions and criteria as the Laya requests; one question per call; no examples.",
        "evaluation": evaluation,
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    lines.append(f"answered_by={result['answered_by']} input_tokens_this_run={client.tokens}")
    print(lines[-1])
    if args.summary:
        args.summary.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
