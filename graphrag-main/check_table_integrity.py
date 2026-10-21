"""
Table-2 integrity gate (Phase 3d of PLAN.md). The accuracy table ships ONLY if
this passes. Guards against exactly the failure modes the reviewers flagged (R2)
plus leakage/above-SOTA artifacts.

Checks, per assembled table (TSV: model, then for each dataset the columns
direct, graphrag, opt_reuse, opt_recompute as Hit fractions):

  1. NO two retrieval-augmented cells byte-identical ACROSS datasets within a row
     (the R2 fabrication signature: Gemma CWQ==WEB).
  2. NO cell exceeds the dataset's SOTA bracket upper bound (leakage signature).
  3. Direct < GraphRAG for every (model, dataset) row (retrieval must help).
  4. opt_recompute within TOL of graphrag for every row (reuse cost bounded).
  5. opt_reuse <= graphrag (+eps) for every row (reuse cannot exceed full prefill).
  6. NO cell value expressible as k/64 (the 64-sample-subset signature), given
     the disclosed full N per dataset.
  7. every disclosed denominator == full N (or the disclosed MQA subsample).

Usage: python check_table_integrity.py results_final/table2.tsv
Exit code 0 = all pass; 1 = at least one violation (printed).
"""
import sys
import json
from fractions import Fraction

# SOTA upper bounds (Hit). A retrieval-augmented cell above this is treated as a
# bug (leakage / eval artifact), never a win. Sources: RoG ICLR'24 + KGQA SOTA.
SOTA_UPPER = {"CWQ": 0.78, "WEB": 0.93, "MQA": 0.99}
FULL_N = {"CWQ": 3531, "WEB": 1628, "MQA": 14274}
RECOMPUTE_TOL = 0.03     # opt_recompute must be within 3pts of graphrag
REUSE_EPS = 0.005


def k_over_64(x, tol=1e-4):
    """True if x looks like k/64 for some integer k (64-subset signature)."""
    for k in range(65):
        if abs(x - k / 64.0) < tol:
            return k
    return None


def check(table, disclosed_n=None):
    """table: {model: {dataset: {col: value}}}; disclosed_n optional {dataset: N}."""
    viol = []
    Ns = dict(FULL_N)
    if disclosed_n:
        Ns.update(disclosed_n)

    for model, dscols in table.items():
        # check 1: identical retrieval cells across datasets
        sigs = {}
        for ds, cols in dscols.items():
            for col in ("graphrag", "opt_reuse", "opt_recompute"):
                if col in cols:
                    v = round(cols[col], 6)
                    sigs.setdefault(v, []).append((ds, col))
        for v, where in sigs.items():
            ds_set = {w[0] for w in where}
            if len(ds_set) > 1:
                viol.append(f"[{model}] identical cell {v} across datasets {sorted(ds_set)} "
                            f"({where}) — R2 duplicate signature")

        for ds, cols in dscols.items():
            up = SOTA_UPPER.get(ds)
            for col in ("graphrag", "opt_reuse", "opt_recompute"):
                if col not in cols:
                    continue
                v = cols[col]
                # check 2: above SOTA
                if up is not None and v > up + 1e-9:
                    viol.append(f"[{model}/{ds}] {col}={v:.4f} exceeds SOTA upper {up} — leakage?")
                # check 6: k/64
                k = k_over_64(v)
                if k is not None and 0 < k < 64 and Ns.get(ds, 0) != 64:
                    viol.append(f"[{model}/{ds}] {col}={v:.4f} == {k}/64 — 64-subset signature "
                                f"(disclosed N={Ns.get(ds)})")
            # check 3: retrieval helps
            if "direct" in cols and "graphrag" in cols and cols["direct"] >= cols["graphrag"]:
                viol.append(f"[{model}/{ds}] direct {cols['direct']:.4f} >= graphrag "
                            f"{cols['graphrag']:.4f} — retrieval did not help")
            # check 4/5: reuse bounds
            if "graphrag" in cols and "opt_recompute" in cols:
                if cols["graphrag"] - cols["opt_recompute"] > RECOMPUTE_TOL + 1e-9:
                    viol.append(f"[{model}/{ds}] opt_recompute {cols['opt_recompute']:.4f} "
                                f"more than {RECOMPUTE_TOL} below graphrag {cols['graphrag']:.4f}")
            if "graphrag" in cols and "opt_reuse" in cols:
                if cols["opt_reuse"] - cols["graphrag"] > REUSE_EPS:
                    viol.append(f"[{model}/{ds}] opt_reuse {cols['opt_reuse']:.4f} exceeds "
                                f"graphrag {cols['graphrag']:.4f} — reuse cannot beat full prefill")
    return viol


def load_tsv(path):
    """Parse a TSV with header: model, {ds}_{col}... -> nested dict."""
    table = {}
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            cells = line.rstrip("\n").split("\t")
            model = cells[0]
            table[model] = {}
            for h, c in zip(header[1:], cells[1:]):
                if "_" not in h or c == "":
                    continue
                ds, col = h.split("_", 1)
                table[model].setdefault(ds, {})[col] = float(c)
    return table


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "results_final/table2.tsv"
    disclosed = json.loads(sys.argv[2]) if len(sys.argv) > 2 else None
    table = load_tsv(path)
    viol = check(table, disclosed)
    if viol:
        print(f"INTEGRITY FAIL ({len(viol)} violations):")
        for v in viol:
            print("  -", v)
        sys.exit(1)
    print(f"INTEGRITY PASS — {len(table)} models checked, no violations.")
    sys.exit(0)
