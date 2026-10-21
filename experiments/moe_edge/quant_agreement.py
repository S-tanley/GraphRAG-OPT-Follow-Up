"""bf16 (H200, HF transformers, batched) vs MLX 4bit (Mac, batch 1) routing on the same prompts.
  .venv-mac/bin/python experiments/moe_edge/quant_agreement.py <bf16 trace dir> <4bit trace dir>
Reports, over (token, layer): identical top-k set, mean overlap |A∩B|/k, separately for prefill and for decode
positions where both runs generated the same prefix; plus answer agreement and Hit@1 of both runs."""
import glob, json, os, sys
import numpy as np

a_dir, b_dir = sys.argv[1], sys.argv[2]
st = dict(n=0, pe=0, po=0, pn=0, de=0, do=0, dn=0, ans=0)
per_layer_p, per_layer_d = None, None
for fa in sorted(glob.glob(a_dir + "/q*.npz")):
    fb = os.path.join(b_dir, os.path.basename(fa))
    if not os.path.exists(fb):
        continue
    za, zb = np.load(fa), np.load(fb)
    ia, ib = za["prompt_ids"], zb["prompt_ids"]
    off = 0
    if not np.array_equal(ia, ib):
        # MLX Mixtral prompt lacks the BOS that HF prepends; align on the shared suffix
        if len(ia) == len(ib) + 1 and np.array_equal(ia[1:], ib):
            off = 1
        else:
            continue
    st["n"] += 1
    st["bos_off"] = st.get("bos_off", 0) + off
    st["ans"] += str(za["pred"]) == str(zb["pred"])
    pa, pb = np.sort(za["pre_idx"][:, off:], -1), np.sort(zb["pre_idx"], -1)
    k = pa.shape[-1]
    eq = (pa == pb).all(-1)
    ov = np.array([[len(set(x) & set(y)) / k for x, y in zip(pa[l], pb[l])] for l in range(pa.shape[0])])
    st["pe"] += eq.sum(); st["po"] += ov.sum(); st["pn"] += eq.size
    per_layer_p = ov.mean(1) if per_layer_p is None else per_layer_p + ov.mean(1)
    D = min(za["dec_idx"].shape[1], zb["dec_idx"].shape[1])
    ga, gb = za["gen_ids"][:D], zb["gen_ids"][:D]
    same = int(np.argmin(ga == gb)) if not np.array_equal(ga, gb) else D
    if same:
        da, db = np.sort(za["dec_idx"][:, :same], -1), np.sort(zb["dec_idx"][:, :same], -1)
        st["de"] += (da == db).all(-1).sum(); st["dn"] += da.shape[0] * same
        st["do"] += sum(len(set(x) & set(y)) / k for l in range(da.shape[0]) for x, y in zip(da[l], db[l]))
print(f"{st['n']} requests aligned ({st.get('bos_off', 0)} with BOS offset)")
print(f"prefill: identical top-{k} {st['pe'] / st['pn']:.1%}, mean overlap {st['po'] / st['pn']:.1%}")
if st["dn"]:
    print(f"decode (same generated prefix): identical {st['de'] / st['dn']:.1%}, mean overlap {st['do'] / st['dn']:.1%} over {st['dn']} (token, layer)")
L = len(per_layer_p); pl = per_layer_p / st["n"]
print(f"prefill overlap by depth: first third {pl[:L//3].mean():.1%}, middle {pl[L//3:2*L//3].mean():.1%}, last third {pl[2*L//3:].mean():.1%}")
print(f"same final answer string: {st['ans']}/{st['n']} ({st['ans'] / st['n']:.0%})")
for d in (a_dir, b_dir):
    j = d + ".json"
    if os.path.exists(j):
        print(f"  {os.path.basename(d)} hit {json.load(open(j))['hit']:.3f}")
