"""Build the context (ids, positions, mask) for each attention condition.

Tokenization always follows faithful_pfkv.build_prompt: preamble with BOS, each path line
`path_str + "\\n"` tokenized on its own, suffix tokenized on its own. The trie is built on
exactly these per-line tokens, so M0 and M2 see the same token vocabulary.

Conditions (see plan v2 §3.3):
  M0            flat, causal, sequential positions                        (= GraphRAG-R prompt)
  M0-shuf<k>    M0 with path lines shuffled by seed k                      (order noise floor)
  T             trie serialized as text ("A -> r -> {B; C}") + causal       (text-level dedup)
  M1-seq        block-diagonal (path sees P + itself), sequential positions
  M1-par        block-diagonal, every path starts at |P| (parallel windows)
  M1-hyb        block-diagonal computed at parallel positions, read at sequential positions
                (= GraphRAG-Opt reuse-all with template prefix)
  M2-par        trie mask (token sees P + its ancestors), position = |P| + depth - 1
  M2-hyb        trie mask computed at depth positions, read at compact layout positions
Variants with "+top<l>" (Exp3) make the last l layers use a causal mask over the layout,
read positions, i.e. full cross-path attention in the top layers only.
"""

import random
import re

import torch

from faithful_pfkv import _ENCODE_PREFIX, _FEWSHOT, path_to_str_rog


def pre_suf(tok, query, template):
    if template == "fewshot":
        pre, suf = _FEWSHOT, f"\nQ: {query}\nA:"
    else:
        pre, suf = _ENCODE_PREFIX["rog"], f"\nQuestion:\n{query} [/INST]"
    return tok.encode(pre, add_special_tokens=True), tok.encode(suf, add_special_tokens=False)


def path_lines(tok, paths, cap):
    out = []
    for p in paths[:cap]:
        ids = tok.encode(path_to_str_rog(p) + "\n", add_special_tokens=False)
        if ids:
            out.append(ids)
    return out


def trie_text_lines(paths, cap):
    """One line per root entity, shared prefixes written once: A -> r -> {B; C -> r2 -> D}."""
    roots = {}
    for p in paths[:cap]:
        if not p:
            continue
        units = [p[0][0]] + [u for _h, r, t in p for u in (r, t)]
        node = roots.setdefault(units[0], {})
        for u in units[1:]:
            node = node.setdefault(u, {})

    def ser(label, children):
        if not children:
            return label
        parts = [ser(c, sub) for c, sub in children.items()]
        tail = parts[0] if len(parts) == 1 else "{" + "; ".join(parts) + "}"
        return f"{label} -> {tail}"

    return [ser(r, ch) + "\n" for r, ch in roots.items()]


def build_trie(lines):
    """Token-prefix trie over per-line token lists, nodes in first-appearance order.
    Returns tokens, parent index (-1 = attached to preamble), depth (1-based)."""
    index, toks, parent, depth = {}, [], [], []
    for line in lines:
        prev = -1
        for j in range(len(line)):
            key = tuple(line[:j + 1])
            if key not in index:
                index[key] = len(toks)
                toks.append(line[j])
                parent.append(prev)
                depth.append(j + 1)
            prev = index[key]
    return toks, parent, depth


def _causal(n):
    return torch.tril(torch.ones(n, n, dtype=torch.bool))


def line_units(tok, path):
    """Per-line tokens (identical to path_lines) labelled by the unit they belong to.
    Units are, in order: entity, ' -> ', relation, ' -> ', entity, ..., '\\n'.
    Returns ids, category per token ('ent'|'rel'|'arrow'|'nl'), unit index per token."""
    units, cats = [path[0][0]], ["ent"]
    for _h, r, t in path:
        units += [" -> ", r, " -> ", t]
        cats += ["arrow", "rel", "arrow", "ent"]
    units.append("\n")
    cats.append("nl")
    s = "".join(units)
    starts, pos = [], 0
    for u in units:
        starts.append(pos)
        pos += len(u)

    def unit_at(ch):
        u = 0
        while u + 1 < len(starts) and starts[u + 1] <= ch:
            u += 1
        return u

    enc = tok(s, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    assert ids == tok.encode(s, add_special_tokens=False), "offset tokenization mismatch"
    tok_unit = []
    for a, b in enc["offset_mapping"]:
        chars = [i for i in range(a, b) if not s[i].isspace()]
        if chars:
            tok_unit.append(unit_at(chars[-1]))
        elif "\n" in s[a:b]:
            tok_unit.append(len(units) - 1)
        else:                                       # pure-space token: belongs to the next unit
            tok_unit.append(unit_at(min(b, len(s) - 1)))
    return ids, [cats[u] for u in tok_unit], tok_unit


def build_arrow(cond, tok, query, paths, template, cap=8):
    """'Dictionary' KV: every entity / relation / connector is its own cached piece (sees only the
    preamble P and itself, positions |P|.. -> globally reusable), except the RECOMPUTED tokens,
    which are computed in context (causal over everything, sequential positions).
      A-pieces    nothing recomputed
      A-arrow     ' -> ' and '\\n' tokens recomputed
      A-arrowlast + last token of every entity
      A-rand      random path tokens recomputed, same count as A-arrow (control)
      A-all       every path token recomputed (gate: must equal M0)
      A-noarrow   everything recomputed except the arrows (are arrows reusable for free?)"""
    P, S = pre_suf(tok, query, template)
    nP = len(P)
    ids, cat, piece, local = list(P), ["pre"] * nP, [-1] * nP, list(range(nP))
    pid = 0
    for p in paths[:cap]:
        if not p:
            continue
        lid, lcat, lunit = line_units(tok, p)
        seen = {}
        for t, c, u in zip(lid, lcat, lunit):
            if u not in seen:
                seen[u] = 0
            ids.append(t); cat.append(c); piece.append(pid + u); local.append(nP + seen[u])
            seen[u] += 1
        pid += max(lunit) + 1
    n = len(ids)
    rec = [False] * n
    if cond == "A-all":
        rec = [c != "pre" for c in cat]
    elif cond == "A-noarrow":                       # only the arrows are cached; everything else normal
        rec = [c not in ("pre", "arrow") for c in cat]
    elif cond in ("A-arrow", "A-arrowlast"):
        rec = [c in ("arrow", "nl") for c in cat]
        if cond == "A-arrowlast":
            for i in range(nP, n):
                if cat[i] == "ent" and (i + 1 == n or piece[i + 1] != piece[i]):
                    rec[i] = True
    elif cond == "A-rand":
        k = sum(c in ("arrow", "nl") for c in cat)
        pool = [i for i in range(nP, n) if cat[i] not in ("arrow", "nl")]
        for i in random.Random(query).sample(pool, min(k, len(pool))):
            rec[i] = True
    elif cond != "A-pieces":
        raise ValueError(cond)
    pc = torch.tensor(piece)
    mask = _causal(n) & ((pc[:, None] == pc[None, :]) | (pc[None, :] < 0))
    r = torch.tensor(rec)
    mask[r] = _causal(n)[r]                         # recomputed rows: full causal context
    seq = list(range(n))
    pin = [seq[i] if rec[i] else local[i] for i in range(n)]
    ctx = dict(ids=ids, pos_int=pin, pos_out=seq, mask=mask, ctx_rows=rec,
               n_recompute=sum(rec), categories=cat)
    return ctx, S, n, n - nP


def build(cond, tok, query, paths, template, cap=8):
    """Returns ctx dict(ids, pos_int, pos_out, mask, [full_mask, full_layers_top]),
    suffix ids, suffix start position, and #path-region tokens."""
    if cond.startswith("A-"):
        return build_arrow(cond, tok, query, paths, template, cap)
    P, S = pre_suf(tok, query, template)
    nP = len(P)
    base, _, top = cond.partition("+top")
    top = int(top) if top else 0

    if base.startswith("M0") or base == "T":
        if base == "T":
            lines = [tok.encode(s, add_special_tokens=False) for s in trie_text_lines(paths, cap)]
        else:
            lines = path_lines(tok, paths, cap)
            if base.startswith("M0-shuf"):
                random.Random(int(base[len("M0-shuf"):])).shuffle(lines)
        ids = P + [t for l in lines for t in l]
        n = len(ids)
        pos = list(range(n))
        ctx = dict(ids=ids, pos_int=pos, pos_out=pos, mask=_causal(n))
        return ctx, S, n, n - nP

    if base.startswith("M1"):
        lines = path_lines(tok, paths, cap)
        ids, seq, par, owner = list(P), list(range(nP)), list(range(nP)), [-1] * nP
        for li, l in enumerate(lines):
            for j, t in enumerate(l):
                ids.append(t); seq.append(len(seq)); par.append(nP + j); owner.append(li)
        n = len(ids)
        own = torch.tensor(owner)
        same = (own[:, None] == own[None, :]) & (own[:, None] >= 0)
        mask = _causal(n) & (same | (own[None, :] < 0))
        maxlen = max((len(l) for l in lines), default=0)
        if base == "M1-seq":
            pin, pout, s0 = seq, seq, n
        elif base == "M1-par":
            pin, pout, s0 = par, par, nP + maxlen
        elif base == "M1-hyb":
            pin, pout, s0 = par, seq, n
        else:
            raise ValueError(cond)
        ctx = dict(ids=ids, pos_int=pin, pos_out=pout, mask=mask)
        if top:
            ctx.update(full_mask=_causal(n), full_layers_top=top)
        return ctx, S, s0, n - nP

    if base.startswith("M2"):
        toks, parent, depth = build_trie(path_lines(tok, paths, cap))
        m = len(toks)
        n = nP + m
        mask = torch.zeros(n, n, dtype=torch.bool)
        mask[:nP, :nP] = _causal(nP)
        mask[nP:, :nP] = True
        anc = []
        for i in range(m):                       # ancestors (and self)
            chain, a = [], i
            while a >= 0:
                chain.append(a)
                mask[nP + i, nP + a] = True
                a = parent[a]
            anc.append(set(chain))
        if base.endswith("S"):
            # sibling groups: token i also sees every earlier token in the subtree of its
            # deepest strict ancestor that branches (>= 2 children)
            nchild = [0] * m
            for p in parent:
                if p >= 0:
                    nchild[p] += 1
            for i in range(m):
                b, a = None, parent[i]
                while a >= 0:
                    if nchild[a] >= 2:
                        b = a
                        break
                    a = parent[a]
                if b is None:
                    continue
                for j in range(b + 1, i):
                    if b in anc[j]:
                        mask[nP + i, nP + j] = True
            base = base[:-1]
        par = list(range(nP)) + [nP + d - 1 for d in depth]
        seq = list(range(n))
        maxd = max(depth, default=0)
        if base == "M2-par":
            pin, pout, s0 = par, par, nP + maxd
        elif base == "M2-hyb":
            pin, pout, s0 = par, seq, n
        else:
            raise ValueError(cond)
        ctx = dict(ids=P + toks, pos_int=pin, pos_out=pout, mask=mask)
        if top:
            ctx.update(full_mask=_causal(n), full_layers_top=top)
        return ctx, S, s0, m

    raise ValueError(cond)
