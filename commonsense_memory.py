"""Score five answers using character n-gram retrieval over labeled training questions."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from sklearn.feature_extraction.text import TfidfVectorizer

from context_experiment import parse, samples


class AnswerMemory:
    def __init__(self, rows):
        self.items = [
            (q, choices, row[1])
            for row in rows
            for q, choices in [parse("jcommonsenseqa", row)]
        ]
        self.q = TfidfVectorizer(
            analyzer="char", ngram_range=(1, 3), sublinear_tf=True, dtype=np.float32
        )
        self.bank = self.q.fit_transform([r[0] for r in self.items])
        self.correct = {}
        self.appears = {}
        for _, choices, y in self.items:
            self.correct[choices[y]] = self.correct.get(choices[y], 0) + 1
            for c in choices:
                self.appears[c] = self.appears.get(c, 0) + 1

    def features(self, text, options):
        sim = (self.q.transform([text]) @ self.bank.T).toarray()[0]
        for i in np.flatnonzero(sim > 0.999):
            if self.items[i][0] == text:
                sim[i] = -1
        ids = np.argpartition(sim, -70)[-70:]
        x = []
        for opt in options:
            yes = [sim[i] for i in ids if self.items[i][1][self.items[i][2]] == opt]
            no = [
                sim[i]
                for i in ids
                if opt in self.items[i][1] and self.items[i][1][self.items[i][2]] != opt
            ]
            pos, denom = self.correct.get(opt, 0), self.appears.get(opt, 0)
            x.append(
                [
                    max(yes, default=0),
                    max(no, default=0),
                    sum(v**3 for v in yes),
                    sum(v**3 for v in no),
                    np.log((1 + pos) / (2 + denom - pos)),
                    np.log1p(denom),
                    float(opt in text),
                ]
            )
        return np.array(x)


def run(a):
    import torch
    from safetensors.torch import load_file

    from context_experiment import dataset
    from context_tasks import Readout

    rows = samples("jcommonsenseqa", a)
    # Keep every fitting query outside the retrieval index.
    bank = AnswerMemory(rows["train"][:4500])
    fit = rows["train"][4500:]

    def matrix(rs):
        return np.stack([bank.features(*parse("jcommonsenseqa", r)) for r in rs])

    x = matrix(fit)
    y = np.array([r[1] for r in fit])

    def loss(w):
        z = x @ w
        values = logsumexp(z, axis=-1) - z[np.arange(len(z)), y]
        residual = softmax(z, axis=-1)
        residual[np.arange(len(z)), y] -= 1
        return values.mean() + 0.0003 * (w @ w), np.einsum(
            "nc,ncf->f", residual, x
        ) / len(z) + 0.0006 * w

    result = minimize(loss, np.zeros(x.shape[-1]), jac=True, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(result.message)
    head = Readout("jcommonsenseqa").eval()
    head.load_state_dict(
        load_file(
            str(
                Path(__file__).with_name("weights")
                / "context/jcommonsenseqa.safetensors"
            )
        )
    )
    output = {}
    for key in ("tune", "jcommonsenseqa", "reserved"):
        rs = rows[key]
        xs = matrix(rs) @ result.x
        x_ctx, y = dataset(None, "jcommonsenseqa", key, rs, a).tensors
        with torch.no_grad():
            logits = head(x_ctx).numpy()
        if key == "tune":

            def dev_score(w, logits=logits, xs=xs, y=y):
                p = softmax(logits + w * xs, axis=-1)
                return int((p.argmax(1) == y.numpy()).sum()), np.log(
                    p[np.arange(len(y)), y]
                ).mean()

            weight = max((0, 0.25, 0.5, 1.0, 1.5, 2.0), key=dev_score)
        pred = (logits + weight * xs).argmax(1)
        output[key] = {"n": len(rs), "correct": int((pred == y.numpy()).sum())}
    output.update(
        bank=len(bank.items),
        fit=len(fit),
        weight=weight,
        coefficients=result.x.tolist(),
    )
    a.result.write_text(json.dumps(output, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k, v in {
        "data": ".cache/data",
        "cache": ".cache/context",
        "result": "results/four-memory.json",
    }.items():
        p.add_argument("--" + k, type=Path, default=Path(v))
    run(p.parse_args())
