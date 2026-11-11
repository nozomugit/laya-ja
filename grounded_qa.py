"""Rank five answer spans inside a Japanese passage using the frozen Laya encoder."""

import re
from pathlib import Path

import torch
from laya_ja import REVISION, load_base
from torch import nn

LIMIT = 4
WIDTH = 768
WH = re.compile(
    r"どのような|どのように|どちら|どんな|どういう|どれくらい|どれほど|どこ|いつ|だれ|誰|なに|何|いくつ|どの|どれ|どう"
)
PARTS = (
    "年",
    "月",
    "日",
    "人",
    "国",
    "市",
    "県",
    "時",
    "個",
    "本",
    "円",
    "％",
    "歳",
    "名",
    "語",
    "大学",
    "会社",
    "地域",
    "場所",
    "理由",
    "誰",
    "いつ",
    "どこ",
    "どのよう",
    "何",
)


def grams(s, n):
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def overlap(a, b, n):
    x, y = grams(a, n), grams(b, n)
    return len(x & y) / max(1, len(x))


def edges(q, c):
    return max((i for i in range(1, min(len(c), 16) + 1) if c[:i] in q), default=0) / 10


def lexical(ctx, q, c, begin, end):
    left, right = ctx[:begin], ctx[end:]
    chunk = re.split("[。！？]", left)[-1] + re.split("[。！？]", right)[0]
    wh = WH.search(q)
    qb, qa = (q[: wh.start()], q[wh.end() :]) if wh else (q, "")
    f = [
        len(c) / 16,
        len(ctx) / 150,
        len(q) / 40,
        begin / max(len(ctx), 1),
        float(c in q),
        float(begin < ctx.find("[SEP]") if "[SEP]" in ctx else False),
        min(5, ctx.count(c)) / 5,
        float(c.isdigit()),
        float(any(t.isdigit() for t in c)),
        float(bool(re.search("[ァ-ヶ]", c))),
        float(bool(re.search("[a-zA-Z]", c))),
        overlap(c, q, 1),
        overlap(c, q, 2),
        overlap(q, chunk, 1),
        overlap(q, chunk, 2),
        edges(q, right),
        edges(q[::-1], left[::-1]),
        edges(qb[::-1], left[::-1]),
        edges(qa, right),
    ]
    for window in (5, 12, 30):
        near = left[-window:] + right[:window]
        f.extend(
            [
                overlap(q, near, 1),
                overlap(q, near, 2),
                overlap(qb, left[-window:], 2),
                overlap(qa, right[:window], 2),
            ]
        )
    for part in PARTS:
        f.extend([float(part in q), float(part in c), float(part in right[:2])])
    return f


FEATURES = len(lexical("例", "何", "例", 0, 1))


def encode_input(tok, ctx, q, choices):
    # Encode the two segments separately to keep context character offsets exact.
    qt = tok(q, add_special_tokens=False)["input_ids"]
    ct = tok(ctx, add_special_tokens=False, return_offsets_mapping=True)
    ids = (
        [tok.cls_token_id]
        + qt
        + [tok.sep_token_id]
        + ct["input_ids"]
        + [tok.sep_token_id]
    )
    if len(ids) > min(tok.model_max_length, 8192):
        raise ValueError("Passage plus question exceeds encoder limit")
    shift = len(qt) + 2
    occurrences = []
    for c in choices:
        matches = []
        for m in re.finditer(re.escape(c), ctx):
            start, end = m.span()
            indices = [
                i + shift
                for i, (a, b) in enumerate(ct["offset_mapping"])
                if a < end and b > start and b > a
            ]
            if indices:
                matches.append(
                    (indices[0], indices[-1], lexical(ctx, q, c, start, end))
                )
            if len(matches) == LIMIT:
                break
        occurrences.append(matches)
    return ids, occurrences


@torch.no_grad()
def extract_batch(encoder, pad, items):
    ids = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(i[0]) for i in items], batch_first=True, padding_value=pad
    )
    dev = next(encoder.parameters()).device
    ids = ids.to(dev)
    h = encoder(input_ids=ids, attention_mask=(ids != pad).long()).last_hidden_state
    # Vectorized boundary lookups and span means, including partial edge tokens.
    dims = (len(items), 5, LIMIT)
    mask = torch.zeros(dims, dtype=torch.bool)
    ix = torch.zeros((*dims, 2), dtype=torch.long)
    feat = torch.zeros((*dims, FEATURES))
    for i, (_, options) in enumerate(items):
        for c, spans in enumerate(options):
            for k, (a, b, v) in enumerate(spans):
                mask[i, c, k] = True
                ix[i, c, k] = torch.tensor([a, b])
                feat[i, c, k] = torch.tensor(v)
    idx = torch.arange(len(items), device=dev)[:, None, None]
    ix = ix.to(dev)
    a, b = ix[..., 0], ix[..., 1]
    cs = torch.cat((torch.zeros_like(h[:, :1]), h.cumsum(1)), 1)
    z = torch.cat(
        (h[idx, a], h[idx, b], (cs[idx, b + 1] - cs[idx, a]) / (b - a + 1)[..., None]),
        -1,
    )
    return z.half().cpu(), feat, mask


class SpanHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(WIDTH * 3, elementwise_affine=False)
        self.vector = nn.Linear(WIDTH * 3, 96)
        self.lex = nn.Linear(FEATURES, 32)
        self.score = nn.Sequential(
            nn.GELU(), nn.Dropout(0.15), nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 1)
        )

    def forward(self, z, f, mask):
        x = torch.cat((self.vector(self.norm(z.float())), self.lex(f.float())), dim=-1)
        return self.score(x).squeeze(-1).masked_fill(~mask, -10000).amax(-1)


def save_head(model, path):
    from safetensors.torch import save_file

    save_file(
        {k: v.detach().half().contiguous() for k, v in model.state_dict().items()},
        str(path),
        metadata={"base_revision": REVISION, "schema": "laya-span-v1"},
    )


def load_head(path, device="cpu"):
    from safetensors import safe_open

    head = SpanHead()
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata() or {}
        if (
            meta.get("base_revision") != REVISION
            or meta.get("schema") != "laya-span-v1"
        ):
            raise ValueError("Incompatible Laya span weights")
        head.load_state_dict({k: f.get_tensor(k) for k in list(f.keys())})
    return head.to(device).eval()


class GroundedReader:
    def __init__(self, model_path=None, device="cpu", weights=None):
        self.agent = load_base(model_path, device)
        path = (
            Path(weights)
            if weights
            else Path(__file__).with_name("weights") / "grounded-qa.safetensors"
        )
        self.head = load_head(path, device)

    @torch.no_grad()
    def predict(self, context, question, choices):
        if (
            not isinstance(context, str)
            or not isinstance(question, str)
            or not context.strip()
            or not question.strip()
        ):
            raise ValueError("Supply a non-empty passage and question")
        if (
            len(choices) != 5
            or any(not isinstance(c, str) or not c.strip() for c in choices)
            or len(set(choices)) != 5
        ):
            raise ValueError("Supply five distinct non-empty strings")
        tok = self.agent.tok
        item = encode_input(tok, context, question, choices)
        z, f, m = extract_batch(self.agent.model.encoder, tok.pad_token_id, [item])
        if not m.any():
            raise ValueError(
                "This extractive reader requires at least one choice in the passage"
            )
        d = next(self.head.parameters()).device
        s = self.head(z.to(d), f.to(d), m.to(d))[0]
        p = s.softmax(-1).cpu().tolist()
        return {
            "choice": choices[s.argmax().item()],
            "probabilities": dict(zip(choices, p)),
        }
