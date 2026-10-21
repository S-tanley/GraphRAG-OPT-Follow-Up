"""
Final paper Table 2 generator (post-rebuttal decisions):
- 12 backbones = paper's 11 (incl Qwen3-80B) + Qwen2.5-72B (new large-model row).
- 5 columns: Direct LLM | GraphRAG | GraphRAG-R | GraphRAG-Opt | Opt+RC.
  GraphRAG := GraphRAG-R (JbiDFS is exact -> identical accuracy; footnote it).
- Qwen3-80B GraphRAG-Opt/Opt+RC = n.a. (linear-attention, no reusable KV).
- Q4 fix: green-highlight the strict MAX per (model,dataset) block; ties both.
Outputs: results/PAPER_TABLE2_final.txt (plain) + results/PAPER_TABLE2_final.tex
(LaTeX rows with \best{} = max-per-block).
"""
import os
TSVS = ["table2_overnight.tsv", "table2_subset.tsv", "table2_mqa.tsv",
        "table2_optfix.tsv", "table2_optreuse.tsv", "table2_gemma_fix.tsv",
        "table2_qwen72.tsv"]
MODELS = [("RoG", "rmanluo/RoG"), ("LLama-2-7b", "meta-llama/Llama-2-7b-hf"),
          ("LLama-2-13b", "meta-llama/Llama-2-13b-hf"), ("Qwen3-8b", "Qwen/Qwen3-8B"),
          ("Qwen3-32B", "Qwen/Qwen3-32B"), ("Qwen2.5-72B", "Qwen/Qwen2.5-72B-Instruct"),
          ("Qwen3-Next-80B", "Qwen/Qwen3-Next-80B-A3B-Thinking"),
          ("Opt-6.7b", "facebook/opt-6.7b"), ("Opt-13b", "facebook/opt-13b"),
          ("Opt-30b", "facebook/opt-30b"), ("Gemma-2-9b", "google/gemma-2-9b"),
          ("Gemma-2-27b", "google/gemma-2-27b")]
DSS = ["CWQ", "WEB", "MQA"]
COLS = [("Direct LLM", "Direct"), ("GraphRAG", "GraphRAG-R"),
        ("GraphRAG-R", "GraphRAG-R"), ("GraphRAG-Opt", "GraphRAG-Opt"),
        ("Opt+RC", "GraphRAG-Opt+RC")]

data = {}
for fn in TSVS:
    p = os.path.join("results", fn)
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
        if v == 0.0:
            continue
        data[(c[0], c[1], c[2])] = v

plain, tex = [], []
plain.append("Model           DS  " + " ".join(f"{lab:>12}" for lab, _ in COLS))
for pn, model in MODELS:
    texrow = [pn]
    for ds in DSS:
        vals = [data.get((model, ds, meth)) for _, meth in COLS]
        present = [v for v in vals if v is not None]
        mx = max(present) if present else None
        prow = []
        for v in vals:
            if v is None:
                prow.append("    n.a.    "); texrow.append("n.a.")
            elif mx is not None and abs(v - mx) < 1e-9:
                prow.append(f"  *{v:5.2f}*  "); texrow.append(rf"\best{{{v:.2f}}}")
            else:
                prow.append(f"   {v:5.2f}   "); texrow.append(f"{v:.2f}")
        plain.append(f"{pn:<15} {ds:<3} " + " ".join(prow)) if ds == DSS[0] else \
            plain.append(f"{'':<15} {ds:<3} " + " ".join(prow))
    tex.append(f"{pn} & " + " & ".join(texrow[1:]) + r" \\ \hline")

txt = "\n".join(plain)
print(txt)
open("results/PAPER_TABLE2_final.txt", "w").write(txt + "\n")
header = (r"% Final Table 2 (Hit%). \best{x}=\textbf{\textcolor{codegreen}{x\%}} = max per (model,dataset)."
          "\n% GraphRAG==GraphRAG-R (JbiDFS exact -> identical accuracy; footnote). "
          "Qwen3-80B Opt/Opt+RC = n.a. (linear-attention, no reusable KV).\n")
open("results/PAPER_TABLE2_final.tex", "w").write(header + "\n".join(tex) + "\n")
print("\n[wrote results/PAPER_TABLE2_final.txt and .tex]")
