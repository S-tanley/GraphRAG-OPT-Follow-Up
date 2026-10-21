"""
Build RoG-style relation-path PLANNING training data for MetaQA, so we can
fine-tune the RoG planner on ~1-2k MetaQA samples (RoG's exact transfer recipe:
"a further finetuning based on RoG trained for Freebase using 1000 samples").

For each sampled train question we extract the GOLD relation sequence = the
relation labels along a shortest path from the topic entity to a gold answer in
the FULL WikiMovies KG (RoG's get_truth_paths uses shortest paths). We take the
most frequent relation sequence across the question's answers (the template rule).
No test data touched. Output JSONL: {"question","rule"} where rule is a list of
relations, serialized for training as "<PATH>r1<SEP>r2<SEP>r3</PATH>".
"""
import argparse
import json
import random
from collections import deque, defaultdict, Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent

# MetaQA questions explicitly name their relations; use this to disambiguate the
# intended template rule from dense shortcut rules (has_tags/has_genre) that
# coincidentally cover the answer set.
KEYMAP = {
    "directed_by": ["direct", "director"],
    "starred_actors": ["star", "actor", "acted", "act in", "appear", "cast"],
    "written_by": ["wrote", "writer", "written"],
    "has_genre": ["genre"],
    "in_language": ["language"],
    "release_year": ["year", "when", "release"],
    "has_tags": ["tag", "topic"],
    "has_imdb_rating": ["rating", "rated"],
    "has_imdb_votes": ["vote", "popular"],
}


def kw_score(rule, ql):
    """+1 per rule-relation whose keyword appears in the question, -0.5 if the
    relation has clear keywords but none appear (penalize spurious dense rels)."""
    s = 0.0
    for rel in set(rule):
        kws = KEYMAP.get(rel, [])
        if any(k in ql for k in kws):
            s += 1.0
        elif kws:
            s -= 0.5
    return s


def parse_kb(kb_path):
    adj = defaultdict(list)
    for line in open(kb_path, encoding="utf-8"):
        p = line.strip().split("|")
        if len(p) == 3:
            h, r, t = p[0].strip(), p[1].strip(), p[2].strip()
            adj[h].append((r, t)); adj[t].append((r, h))
    return adj


def parse_qa(qa_path):
    import re
    ent = re.compile(r"\[([^\]]+)\]")
    out = []
    for line in open(qa_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 1)
        if len(parts) < 2:
            continue
        q_ents = ent.findall(parts[0])
        clean = ent.sub(lambda m: m.group(1), parts[0])
        answers = [a.strip() for a in parts[1].split("|") if a.strip()]
        out.append({"question": clean, "q_entity": q_ents, "answer": answers})
    return out


def gold_rules(adj, topic, answers, hop):
    """Relation sequences of length EXACTLY `hop` from topic to a gold answer
    (the intended reasoning depth for MetaQA-`hop`; ignores shorter shortcut
    paths that would mislabel the template rule). DFS over simple paths to depth
    `hop`. Returns a Counter of relation-tuples reaching an answer at depth hop."""
    ansset = set(answers)
    # For each length-`hop` relation sequence, track the SET of distinct gold
    # answers it reaches. The intended template rule covers (nearly) the whole
    # answer set; spurious shortcut rules reach only a few -> rank by coverage.
    cover = defaultdict(set)
    stack = [(topic, [], {topic})]
    explored = 0
    while stack and explored < 300000:
        node, rule, seen = stack.pop()
        explored += 1
        if len(rule) == hop:
            if node in ansset:
                cover[tuple(rule)].add(node)
            continue
        for rel, nb in adj.get(node, ()):
            if nb not in seen:
                stack.append((nb, rule + [rel], seen | {nb}))
    return Counter({rule: len(ans) for rule, ans in cover.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hops", default="3", help="comma list of hop variants to mix, e.g. 1,2,3")
    ap.add_argument("--n_per_hop", type=int, default=1200)
    ap.add_argument("--out", default="metaqa_plan_train.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    adj = parse_kb(HERE / "metaQA" / "kb.txt")
    print(f"KB: {len(adj):,} nodes")
    rows = []
    for hop in [int(h) for h in args.hops.split(",")]:
        qa = parse_qa(HERE / "metaQA" / f"{hop}-hop" / "vanilla" / "qa_train.txt")
        random.shuffle(qa)
        kept = 0
        for ex in qa:
            if kept >= args.n_per_hop:
                break
            if not ex["q_entity"]:
                continue
            topic = ex["q_entity"][0]
            if topic not in adj:
                continue
            rules = gold_rules(adj, topic, ex["answer"], hop)
            if not rules:
                continue
            # rank by (keyword match to the question, then answer coverage):
            # picks the template rule over dense shortcut rules.
            ql = ex["question"].lower()
            best = max(rules.items(), key=lambda kv: (kw_score(kv[0], ql), kv[1]))[0]
            if not best:
                continue
            rows.append({"question": ex["question"], "rule": list(best)})
            kept += 1
        print(f"hop {hop}: kept {kept}")
    random.shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} planning examples -> {args.out}")
    # show a few
    for r in rows[:5]:
        print("  ", r["question"][:70], "=>", r["rule"])


if __name__ == "__main__":
    main()
