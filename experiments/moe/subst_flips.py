"""Paired comparison of subst_replay.py runs on GSM8K: per question, right -> wrong and wrong -> right against the
no-change run, the paired accuracy difference with a 95% interval, and the exact McNemar p-value.

The baseline is the "none" run of the same file (each file was generated in one process); --base_file uses another
file's "none" run instead (e.g. to check that two processes produce the same baseline).

  python experiments/moe/subst_flips.py outputs/moe/subst/p2b_qwen3_think200_r0.05.json ...
"""
import argparse
import json
import math


def golds(split, n, skip=0):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main")[split]
    return [ds[i]["answer"].split("####")[-1].strip().replace(",", "") for i in range(skip, skip + n)]


def right(ans, gold):
    try:
        return ans is not None and abs(float(ans) - float(gold)) < 1e-6
    except ValueError:
        return False


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--base_file", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = []
    for f in a.files:
        d = json.load(open(f))
        assert d["task"] == "gsm8k", f
        res = d["results"]
        base_res = json.load(open(a.base_file))["results"] if a.base_file else res
        base = next(r for r in base_res if r["run"] == "none")
        n = len(base["answers"])
        g = golds(d["args"]["split"], n, d["args"].get("skip", 0))
        rb = [right(x, y) for x, y in zip(base["answers"], g)]
        for r in res:
            ra = [right(x, y) for x, y in zip(r["answers"], g)]
            r2w = sum(p and not q for p, q in zip(rb, ra))
            w2r = sum(q and not p for p, q in zip(rb, ra))
            diff = (w2r - r2w) / n
            se = math.sqrt(max((w2r + r2w) - (w2r - r2w) ** 2 / n, 0.0)) / n
            same = sum(x == y for x, y in zip(base["answers"], r["answers"])) / n
            run = r["run"].split(":")[-1] if r["run"] != "none" else "none"
            if "batch" in r and len({x.get("batch") for x in res}) > 1:   # neutral control: several batch sizes
                run += f"@b{r['batch']}"
            row = dict(file=f, run=run, n=n, acc=sum(ra) / n, base_acc=sum(rb) / n, right_to_wrong=r2w,
                       wrong_to_right=w2r, diff=diff, ci95=(diff - 1.96 * se, diff + 1.96 * se),
                       p_mcnemar=mcnemar(r2w, w2r), same_final=same, change_rate=r.get("change_rate"),
                       gen_tokens=r.get("gen_tokens"), sec=r.get("sec"))
            rows.append(row)
            print(f"{f.split('/')[-1]:38} {run:6} acc {row['acc']:.3f} (base {row['base_acc']:.3f})  "
                  f"right->wrong {r2w:3d}  wrong->right {w2r:3d}  diff {100 * diff:+.1f} pt "
                  f"[{100 * row['ci95'][0]:+.1f}, {100 * row['ci95'][1]:+.1f}]  McNemar p {row['p_mcnemar']:.2f}  "
                  f"same final {same:.0%}  tokens {row['gen_tokens']}", flush=True)
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
