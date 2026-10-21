"""
Literature-alignment sanity check for the leakage-free Table 2 (MICRO 2026).

For each accuracy dataset it reports, on the fixed-seed N=500 subset:
  - retrieval recall   = fraction of questions whose GOLD answer entity appears
                         in any retrieved RoG-plan path (an upper bound on Hits@1),
  - our GraphRAG-R Hits@1 (from the results TSVs),
  - generation efficiency = Hits@1 / recall (how much of the retrievable signal
                            the reader captures),
  - the published RoG number and SOTA bracket.

Purpose: verify our numbers MAKE SENSE against the literature and surface any
dataset where accuracy is capped by retrieval rather than by the method.
A healthy leakage-free cell sits BELOW SOTA by roughly the recall gap, with
generation efficiency ~0.85-1.0.

    python check_literature_alignment.py
"""
import json
import os

# Published references (RoG, Luo et al. ICLR 2024) + SOTA brackets (Hits@1).
LIT = {
    "CWQ": dict(rog=0.626, sota="0.62-0.76", full=3531,
                ds="rmanluo/RoG-cwq", pf="precomputed_paths/paths_rmanluo-RoG-cwq.json"),
    "WEB": dict(rog=0.857, sota="0.85-0.92", full=1628,
                ds="rmanluo/RoG-webqsp", pf="precomputed_paths/paths_rmanluo-RoG-webqsp.json"),
    "MQA": dict(rog=0.848, sota="0.98-0.99", full=14274,
                ds="metaqa-3hop", pf="precomputed_paths/paths_metaqa-3hop.json"),
}
N = 500


def load_ds(name):
    if name == "metaqa-3hop":
        from metaqa_adapter import load_metaqa
        mdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metaQA")
        return load_metaqa(hops=3, split="test", data_dir=mdir).shuffle(seed=42).select(range(N))
    from datasets import load_dataset
    return load_dataset(name, split="test").shuffle(seed=42).select(range(N))


def recall(ds, pf):
    paths = json.load(open(pf))
    hit = 0
    for i in range(len(ds)):
        q, ans = ds[i]["question"], ds[i]["answer"]
        golds = [g.lower() for g in ([ans] if isinstance(ans, str) else list(ans))]
        nodes = set()
        for p in paths.get(q, []):
            for h, r, t in p:
                nodes.add(str(h).lower()); nodes.add(str(t).lower())
        hit += any(g in nodes for g in golds)
    return hit / len(ds)


def our_R():
    """Read GraphRAG-R Hits@1 per dataset from the result TSVs (RoG backbone)."""
    out = {}
    # table2_mqa.tsv LAST so the guided-retriever MQA sweep (max_paths=20) wins
    # over the stale overnight MQA (max_paths=8); merge is last-write-wins.
    for tsv in ["results/table2_overnight.tsv", "results/table2_subset.tsv",
                "results/table2_mqa.tsv"]:
        if not os.path.exists(tsv):
            continue
        for line in open(tsv).read().splitlines()[1:]:
            p = line.split("\t")
            if len(p) < 7 or p[0] != "rmanluo/RoG" or p[2] != "GraphRAG-R" or p[3] == "FAILED":
                continue
            out[p[1]] = float(p[3])
    return out


def main():
    R = our_R()
    print(f"\n=== Literature alignment (RoG backbone, leakage-free, N={N} recall) ===")
    print(f"{'DS':5}{'recall':>8}{'our R':>8}{'gen.eff':>9}{'RoG pub':>9}{'SOTA':>12}  verdict")
    print("-" * 70)
    for d, m in LIT.items():
        rec = recall(load_ds(m["ds"]), m["pf"])
        r = R.get(d)
        eff = (r / rec) if (r and rec) else float("nan")
        below = (r is not None and r < m["rog"])
        capped = rec < 0.45
        verdict = ("OK below-SOTA" if (below and not capped)
                   else "FLAG: retrieval-capped" if capped else "CHECK")
        rs = f"{r:.3f}" if r is not None else "  -  "
        print(f"{d:5}{rec:>8.3f}{rs:>8}{eff:>9.2f}{m['rog']:>9.3f}{m['sota']:>12}  {verdict}")
    print("\nReading: a healthy leakage-free cell is BELOW SOTA with gen.eff ~0.85-1.0.")
    print("A 'retrieval-capped' cell (recall<0.45) means accuracy is bounded by the")
    print("retriever, not the method -- report recall alongside accuracy and frame as")
    print("out-of-domain transfer (the Freebase-trained RoG planner does not match the")
    print("WikiMovies schema), or restrict accuracy claims to the in-domain datasets.")


if __name__ == "__main__":
    main()
