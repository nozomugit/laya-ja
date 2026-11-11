"""Metrics for Japanese choice, binary acceptability and ordinal similarity."""

import torch


def metrics(scores, labels):
    p = scores.softmax(-1)
    conf, pred = p.max(-1)
    correct = pred == labels
    ece = 0.0
    for i in range(10):
        members = (conf > i / 10) & (conf <= (i + 1) / 10)
        if members.any():
            ece += (
                members.float().mean()
                * (conf[members].mean() - correct[members].float().mean()).abs()
            ).item()
    return {
        "n": len(labels),
        "correct": int(correct.sum()),
        "accuracy": correct.float().mean().item(),
        "nll": torch.nn.functional.cross_entropy(scores, labels).item(),
        "ece_10_bins": ece,
    }


def diagnostic(task, scores, y):
    if task == "jcola":
        result = metrics(scores, y)
        pred = scores.argmax(dim=-1)
        tp = ((pred == 1) & (y == 1)).sum().item()
        fp = ((pred == 1) & (y == 0)).sum().item()
        tn = ((pred == 0) & (y == 0)).sum().item()
        fn = ((pred == 0) & (y == 1)).sum().item()
        denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        result.update(
            mcc=(tp * tn - fp * fn) / denom if denom else 0.0,
            recall_acceptable=tp / (tp + fn),
            recall_unacceptable=tn / (tn + fp),
            majority_accuracy=max(tp + fn, tn + fp) / len(y),
        )
        return result
    pred = scores.softmax(-1) @ torch.arange(6, dtype=scores.dtype)
    return {
        "n": len(y),
        "pearson": float(torch.corrcoef(torch.stack([pred, y]))[0, 1]),
        "mae": float((pred - y).abs().mean()),
        "rmse": float((pred - y).square().mean().sqrt()),
    }


def interpolate(y):
    low, high = y.floor().long(), y.ceil().long()
    a = torch.nn.functional.one_hot(low, num_classes=6)
    b = torch.nn.functional.one_hot(high, num_classes=6)
    fraction = (y - low).unsqueeze(1)
    return a * (1 - fraction) + b * fraction
