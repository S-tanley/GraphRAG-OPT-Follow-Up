"""Gates for the A-* conditions (RoG, CWQ, first N questions):
  1. A-all (every path token recomputed in context) reproduces M0 predictions exactly.
  2. A cached piece's K/V is identical wherever it appears (piece = entity or relation),
     i.e. the 'dictionary' KV really is context-free.
"""

import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "graphrag-main"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "common"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "exp2"))
from faithful_pfkv import extract_answer           # noqa: E402
from conditions import build, build_arrow          # noqa: E402
from masked_engine import MaskedEngine             # noqa: E402
from run import load_model                         # noqa: E402
from table2_worker import load_ds                  # noqa: E402

N = int(os.environ.get("N_TEST", 60))
tok, model, template = load_model("rmanluo/RoG")
eng = MaskedEngine(model)
ds = load_ds("rmanluo/RoG-cwq").select(range(N))
pdict = json.load(open(os.path.join(ROOT, "graphrag-main/precomputed_paths_v2/paths_rmanluo-RoG-cwq.json")))


def answer(cond, q, paths):
    ctx, suf, s0, _ = build(cond, tok, q, paths, template, 8)
    gen = eng.generate(ctx, suf, s0, 16, tok.eos_token_id, ctx_rows=ctx.get("ctx_rows"))
    return extract_answer(tok.decode(gen, skip_special_tokens=True))


same = 0
for i in range(N):
    q = ds[i]["question"]
    P = [[tuple(t) for t in p] for p in pdict.get(q, [])]
    same += answer("A-all", q, P) == answer("M0", q, P)
print(f"[gate1] A-all vs M0: identical predictions {same}/{N}")

cache, pairs, worst = {}, 0, 0.0
for i in range(N):
    q = ds[i]["question"]
    P = [[tuple(t) for t in p] for p in pdict.get(q, [])]
    ctx, _, _, _ = build_arrow("A-arrow", tok, q, P, template, 8)
    Ks, Vs = eng.encode_context(ctx["ids"], ctx["pos_int"], ctx["mask"], ctx["pos_out"], ctx_rows=ctx["ctx_rows"])
    ids, cat, rec = ctx["ids"], ctx["categories"], ctx["ctx_rows"]
    # group consecutive cached tokens of one piece; compare raw (un-rotated) V and K rotated
    # back to piece-local positions via pos_int (K stored at pos_out: rotate by -(pos_out-pos_int))
    i0 = 0
    while i0 < len(ids):
        if cat[i0] in ("ent", "rel") and not rec[i0]:
            j = i0
            while j + 1 < len(ids) and cat[j + 1] == cat[i0] and not rec[j + 1] and ctx["pos_int"][j + 1] == ctx["pos_int"][j] + 1:
                j += 1
            key = (cat[i0], tuple(ids[i0:j + 1]))
            vv = [V[:, i0:j + 1].clone() for V in Vs]
            if key in cache:
                d = max((a - b).abs().max().item() for a, b in zip(vv, cache[key]))
                worst = max(worst, d); pairs += 1
            else:
                cache[key] = vv
            i0 = j + 1
        else:
            i0 += 1
print(f"[gate2] cached pieces seen >=2 times: {pairs} comparisons; max |dV| over layers = {worst:.3e}")
