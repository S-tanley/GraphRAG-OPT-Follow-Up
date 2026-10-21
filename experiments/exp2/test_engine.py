"""Correctness gates for the masked engine (RoG on CWQ, fp16).

1. M0 through the engine == HF full_generate (same prompt) on N queries: identical predictions,
   and first-step logits close.
2. M1-hyb through the engine reproduces GraphRAG-Opt reuse-all (ablation 'none' predictions).
3. Trie reuse exactness: a trie node shared by two queries gets the same K/V in both contexts.
"""

import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "graphrag-main"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "common"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "exp2"))
from faithful_pfkv import build_prompt, extract_answer   # noqa: E402
from hf_reuse import full_generate                       # noqa: E402
from table2_worker import load_ds                        # noqa: E402
from conditions import build, build_trie, path_lines, pre_suf  # noqa: E402
from masked_engine import MaskedEngine                   # noqa: E402
from run import load_model                               # noqa: E402

N = int(os.environ.get("N_TEST", 100))
tok, model, template = load_model("rmanluo/RoG")
eng = MaskedEngine(model)
ds = load_ds("rmanluo/RoG-cwq").select(range(500))
pdict = json.load(open(os.path.join(ROOT, "graphrag-main/precomputed_paths/paths_rmanluo-RoG-cwq.json")))
qs = [ds[i]["question"] for i in range(500)]
P = {q: [[tuple(t) for t in p] for p in pdict.get(q, [])] for q in qs}

# ---- 1. M0 == HF
same, maxdiff = 0, 0.0
for q in qs[:N]:
    ctx, suf, s0, _ = build("M0", tok, q, P[q], template, 8)
    full, _, _, _ = build_prompt(tok, q, P[q], 8, template)
    assert ctx["ids"] + suf == full, "token sequence mismatch vs build_prompt"
    a = extract_answer(tok.decode(eng.generate(ctx, suf, s0, 16, tok.eos_token_id), skip_special_tokens=True))
    b = full_generate(model, tok, q, P[q], template, 8, 16, "cuda")
    same += a == b
    with torch.no_grad():
        ref = model(input_ids=torch.tensor([full], device="cuda")).logits[0, -1].float()
        Ks, Vs = eng.encode_context(ctx["ids"], ctx["pos_int"], ctx["mask"], ctx["pos_out"])
        got = eng._step(suf, list(range(s0, s0 + len(suf))), Ks, Vs).float()
    maxdiff = max(maxdiff, (ref.softmax(-1) - got.softmax(-1)).abs().max().item())
print(f"[gate1] M0 engine vs HF: identical predictions {same}/{N}; max |Δprob| first token {maxdiff:.2e}")

# ---- 2. M1-hyb vs ablation 'none'
abl = json.load(open(os.path.join(ROOT, "outputs/exp0/abl_template.json")))
none = abl["preds"]["none"]
agree = 0
for i, q in enumerate(qs[:N]):
    ctx, suf, s0, _ = build("M1-hyb", tok, q, P[q], template, 8)
    a = extract_answer(tok.decode(eng.generate(ctx, suf, s0, 16, tok.eos_token_id), skip_special_tokens=True))
    agree += a == none[i]
print(f"[gate2] M1-hyb engine vs GraphRAG-Opt reuse-all (manual engine): identical {agree}/{N}")

# ---- 3. trie node exactness across queries
def node_kv(q):
    ctx, suf, s0, _ = build("M2-par", tok, q, P[q], template, 8)
    Ks, Vs = eng.encode_context(ctx["ids"], ctx["pos_int"], ctx["mask"], ctx["pos_out"])
    toks, parent, depth = build_trie(path_lines(tok, P[q], 8))
    nP = len(pre_suf(tok, q, template)[0])
    keys = {}
    for i in range(len(toks)):                      # reconstruct each node's token prefix
        pre, a = [], i
        while a >= 0:
            pre.append(toks[a]); a = parent[a]
        keys[tuple(reversed(pre))] = nP + i
    return keys, Ks, Vs

pairs, worst = 0, 0.0
cache = {}
for q in qs[:N]:
    keys, Ks, Vs = node_kv(q)
    for k, idx in keys.items():
        if k in cache:
            K0, V0 = cache[k]
            d = max(max((Ks[l][:, idx] - K0[l]).abs().max().item(), (Vs[l][:, idx] - V0[l]).abs().max().item())
                    for l in range(len(Ks)))
            worst = max(worst, d); pairs += 1
        else:
            cache[k] = ([K[:, idx].clone() for K in Ks], [V[:, idx].clone() for V in Vs])
print(f"[gate3] trie nodes seen in >=2 queries: {pairs} comparisons; max |ΔK|,|ΔV| over all layers = {worst:.3e}")
