"""
Coverage check: can every cell of the paper's Table 2 (Table:acc) be filled from
our result TSVs? Paper Table 2 = 11 backbones x {CWQ,WEB,MQA} x {Direct, GraphRAG,
GraphRAG-R, GraphRAG-Opt}. GraphRAG (BFS) and GraphRAG-R (JbiDFS) are accuracy-
identical by construction (JbiDFS is exact -> same path set) so GraphRAG := our
GraphRAG-R value. Writes results/PAPER_TABLE2_coverage.txt (+ filled table).
"""
import glob, os

HERE = os.path.dirname(os.path.abspath(__file__))
TSVS = ["table2_overnight.tsv", "table2_subset.tsv", "table2_mqa.tsv",
        "table2_optfix.tsv", "table2_optreuse.tsv", "table2_gemma_fix.tsv",
        "table2_70b.tsv"]
# paper backbone -> our model string
PAPER = [("RoG","rmanluo/RoG"), ("LLama-2-7b","meta-llama/Llama-2-7b-hf"),
         ("LLama-2-13b","meta-llama/Llama-2-13b-hf"), ("Qwen3-8b","Qwen/Qwen3-8B"),
         ("Qwen3-32B","Qwen/Qwen3-32B"), ("Qwen3-Next-80B","Qwen/Qwen3-Next-80B-A3B-Thinking"),
         ("Opt-6.7b","facebook/opt-6.7b"), ("Opt-13b","facebook/opt-13b"),
         ("Opt-30b","facebook/opt-30b"), ("Gemma-2-9b","google/gemma-2-9b"),
         ("Gemma-2-27b","google/gemma-2-27b")]
DSS = ["CWQ", "WEB", "MQA"]
LABEL2METH = {"Direct LLM":"Direct", "GraphRAG":"GraphRAG-R",
              "GraphRAG-R":"GraphRAG-R", "GraphRAG-Opt":"GraphRAG-Opt"}

# load all (model, ds, method) -> hit% ; last non-FAILED wins, gemma_fix last = authoritative
data = {}
for fn in TSVS:
    p = os.path.join(HERE, "results", fn)
    if not os.path.exists(p):
        continue
    for ln in open(p).read().splitlines()[1:]:
        c = ln.split("\t")
        if len(c) < 7 or c[3] == "FAILED":
            continue
        try:
            v = float(c[3]) * 100
        except ValueError:
            continue
        if v == 0.0:           # fp16-overflow zeros are not valid
            continue
        data[(c[0], c[1], c[2])] = v

out = []
out.append("=== PAPER TABLE 2 COVERAGE (Hit%%, our results) ===")
out.append("cols: Direct | GraphRAG(=R) | GraphRAG-R | GraphRAG-Opt   (per CWQ/WEB/MQA)")
missing = []
for pname, model in PAPER:
    row = [f"{pname:<16}"]
    for ds in DSS:
        cells = []
        for lab in ["Direct LLM", "GraphRAG", "GraphRAG-R", "GraphRAG-Opt"]:
            meth = LABEL2METH[lab]
            v = data.get((model, ds, meth))
            if v is None:
                cells.append("  --  ")
                missing.append(f"{pname} / {ds} / {lab}")
            else:
                cells.append(f"{v:5.1f}")
        row.append(" ".join(cells))
    out.append("  ".join(row))

out.append("")
out.append(f"=== MISSING CELLS: {len(missing)} ===")
for m in missing:
    out.append(f"  MISSING: {m}")
if not missing:
    out.append("  NONE -- every paper Table 2 cell is fillable.")

out.append("")
out.append("NOTE: GraphRAG (BFS) and GraphRAG-R (JbiDFS) are accuracy-IDENTICAL by")
out.append("construction (JbiDFS is exact -> same path set); they differ only in")
out.append("retrieval SPEED (throughput/TTFT figures). So the GraphRAG column == the")
out.append("GraphRAG-R column here. Consider merging them or footnoting the identity.")
txt = "\n".join(out)
print(txt)
open(os.path.join(HERE, "results", "PAPER_TABLE2_coverage.txt"), "w").write(txt + "\n")

# ---- emit paper-format LaTeX (Direct | GraphRAG | GraphRAG-R | GraphRAG-Opt) ----
def cell(model, ds, lab):
    v = data.get((model, ds, LABEL2METH[lab]))
    return f"{v:.2f}\\%" if v is not None else "--"
ltx = ["% Auto-filled Table 2 (Hit%) from our results. GraphRAG:=GraphRAG-R (JbiDFS exact).",
       "% 80B GraphRAG-Opt = -- (reuse N/A on hybrid linear-attention, model-parallel)."]
for pname, model in PAPER:
    vals = []
    for ds in DSS:
        for lab in ["Direct LLM", "GraphRAG", "GraphRAG-R", "GraphRAG-Opt"]:
            vals.append(cell(model, ds, lab))
    ltx.append(f"{pname} & " + " & ".join(vals) + r" \\ \hline")
open(os.path.join(HERE, "results", "PAPER_TABLE2_filled.tex"), "w").write("\n".join(ltx) + "\n")
print("\n[wrote results/PAPER_TABLE2_coverage.txt and results/PAPER_TABLE2_filled.tex]")
