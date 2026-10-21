"""Arrow experiment, diagnostic: under normal reading (M0), where does the question/answer
attention go — preamble, entities, relations, arrows, or newlines?

For N questions, the suffix tokens (question + "[/INST]") are run on top of the M0 context and
their attention weights over the context are summed per token category, per layer.
Also checks the token categorisation used by the A-* conditions (prints one example).

  python experiments/arrow/attn_diag.py --model rmanluo/RoG --dataset rmanluo/RoG-cwq --n 100
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "graphrag-main"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "common"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "exp2"))
from conditions import build_arrow                 # noqa: E402
from masked_engine import MaskedEngine             # noqa: E402
from run import PATHS, load_model                  # noqa: E402
from table2_worker import load_ds                  # noqa: E402

CATS = ["pre", "ent", "rel", "arrow", "nl"]


@torch.no_grad()
def suffix_attention(eng, ctx, suf, s0):
    """Per layer: attention mass of (a) all suffix tokens averaged, (b) the last suffix token,
    over each context category. Context = M0 (A-all builds the same computation as M0)."""
    Ks, Vs = eng.encode_context(ctx["ids"], ctx["pos_int"], ctx["mask"], ctx["pos_out"],
                                ctx_rows=ctx["ctx_rows"])
    nctx = len(ctx["ids"])
    cat = torch.tensor([CATS.index(c) for c in ctx["categories"]], device=eng.dev)
    n = len(suf)
    c, s = eng._cos_sin(list(range(s0, s0 + n)))
    H = eng.core.embed_tokens(torch.tensor(suf, device=eng.dev))
    per_layer = []
    for li, layer in enumerate(eng.layers):
        q, k, v = eng._qkv(layer, layer.input_layernorm(H))
        K = torch.cat([Ks[li], eng._rope(k, c, s)], dim=1)
        V = torch.cat([Vs[li], v], dim=1)
        qr = eng._rope(q, c, s)
        rep = qr.shape[0] // K.shape[0]
        Kr = K.repeat_interleave(rep, 0)
        sc = (qr @ Kr.transpose(-1, -2)).float() / math.sqrt(eng.hd)     # [H, n, nk]
        nk = K.shape[1]
        m = torch.ones(n, nk, device=eng.dev, dtype=torch.bool)
        m[:, nk - n:] = torch.tril(m[:, nk - n:])
        w = sc.masked_fill(~m[None], float("-inf")).softmax(-1).mean(0)   # [n, nk]
        wc = w[:, :nctx]
        mass = torch.stack([wc[:, cat == j].sum(-1) for j in range(len(CATS))], -1)   # [n, ncat]
        per_layer.append((mass.mean(0).cpu(), mass[-1].cpu()))
        H = H + eng._attn(layer, qr, K, V, m)
        H = H + layer.mlp(layer.post_attention_layernorm(H))
    counts = torch.tensor([(cat == j).sum().item() for j in range(len(CATS))], dtype=torch.float)
    return per_layer, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rmanluo/RoG")
    ap.add_argument("--dataset", default="rmanluo/RoG-cwq")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ds = load_ds(args.dataset).select(range(args.n))
    pdict = json.load(open(os.path.join(ROOT, "graphrag-main/precomputed_paths_v2", PATHS[args.dataset])))
    tok, model, template = load_model(args.model)
    eng = MaskedEngine(model)

    q0 = ds[0]["question"]
    ctx, suf, s0, _ = build_arrow("A-all", tok, q0, [[tuple(t) for t in p] for p in pdict[q0]], template, 8)
    first = [i for i, c in enumerate(ctx["categories"]) if c != "pre"][:16]
    print("categorisation example:", [(tok.convert_ids_to_tokens(ctx["ids"][i]), ctx["categories"][i]) for i in first])

    acc_all, acc_last, acc_cnt = defaultdict(float), defaultdict(float), torch.zeros(len(CATS))
    L = len(eng.layers)
    for i in range(len(ds)):
        q = ds[i]["question"]
        paths = [[tuple(t) for t in p] for p in pdict.get(q, [])]
        ctx, suf, s0, _ = build_arrow("A-all", tok, q, paths, template, 8)
        per_layer, counts = suffix_attention(eng, ctx, suf, s0)
        acc_cnt += counts
        for li, (ma, ml) in enumerate(per_layer):
            for j, cname in enumerate(CATS):
                acc_all[(li, cname)] += ma[j].item()
                acc_last[(li, cname)] += ml[j].item()
    N = len(ds)
    res = {"model": args.model, "dataset": args.dataset, "n": N,
           "token_share": {c: (acc_cnt[j] / acc_cnt.sum()).item() for j, c in enumerate(CATS)},
           "layers": L,
           "all_suffix": {c: [acc_all[(li, c)] / N for li in range(L)] for c in CATS},
           "last_suffix": {c: [acc_last[(li, c)] / N for li in range(L)] for c in CATS}}
    json.dump(res, open(args.out, "w"), indent=1)
    print("token share:", {c: round(v, 3) for c, v in res["token_share"].items()})
    for key in ["all_suffix", "last_suffix"]:
        for band, rng in [("layers 0-7", range(0, 8)), ("8-15", range(8, 16)), ("16-23", range(16, 24)), ("24-31", range(24, 32))]:
            row = {c: round(sum(res[key][c][li] for li in rng) / len(rng), 3) for c in CATS}
            print(f"{key:12s} {band:10s} mass:", row)


if __name__ == "__main__":
    main()
