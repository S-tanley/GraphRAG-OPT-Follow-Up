"""MoE expert-offloading engine for MLX on Apple silicon (edge plan §6.5; also used to collect B-tier traces).

Tiers (plan §1.1): non-expert weights resident in unified memory; routed experts in experts.bin on the
internal SSD (written by export_experts.py), fetched with preadv + F_NOCACHE straight into slots of a
per-layer cache pool (zero copy: numpy views of the MLX pool arrays are writable and shared).

Each MoE block's `switch_mlp` is replaced by OffloadSwitch:
  decode  (few distinct experts): make them resident in the layer's pool (wait for in-flight prefetches,
          load the rest in parallel), remap expert ids -> slots, run the usual SwitchGLU on the pool.
  prefill (more distinct experts than slots): resident experts are used from the pool; missing ones are read
          into a full-layer staging buffer; the SwitchGLU runs on both and the output of each (token, k)
          assignment is taken from where its expert is. Afterwards the policy decides which staged experts
          to keep in the pool (the E4 retention decision).
Policies: lru | lfu (global use count) | moeinf (within-request use count) | hyb (moeinf + lam*prior);
`sw` = lfu during prefill. Prefetch: la1 (next layer's gate on this layer's input, decode only) and a
request-level prior loaded in the background (set_prior).

  eng = OffloadEngine.attach(model, "~/mlx_offload/Qwen3-30B-A3B-4bit", slots=64, policy="moeinf")
"""

import fcntl
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models import switch_layers

F_NOCACHE = 48
DT = {"uint32": mx.uint32, "bfloat16": mx.bfloat16, "float16": mx.float16, "float32": mx.float32,
      "uint8": mx.uint8}


def _views(arrs, slot):
    return [np.array(a, copy=False)[slot] for a in arrs]


class OffloadSwitch(nn.Module):
    def __init__(self, eng, j):
        super().__init__()
        self._eng, self._j = eng, j
        self.on_call = None                      # trace hook (collect_traces_mlx)

    def __call__(self, x, indices):
        if self.on_call is not None:
            self.on_call(x, indices)
        return self._eng.forward(self._j, x, indices)


class OffloadEngine:
    def __init__(self, store_dir, slots, policy="lru", threads=8, la1=False, lam=1.0, mode_cfg=None):
        meta = json.load(open(os.path.join(os.path.expanduser(store_dir), "meta.json")))
        self.meta, self.E, self.L = meta, meta["E"], len(meta["layer_ids"])
        self.rec = meta["record_bytes"]
        self.tensors = meta["tensors"]
        self.fd = os.open(os.path.join(os.path.expanduser(store_dir), "experts.bin"), os.O_RDONLY)
        fcntl.fcntl(self.fd, F_NOCACHE, 1)
        self.S = int(slots)
        self._sw = policy.startswith("sw-")
        self.policy, self.la1, self.lam = (policy[3:] if self._sw else policy), la1, lam
        self.phase = "prefill"
        # bf16 tensors are stored as uint16 (numpy cannot view bf16) and bit-cast back when computing
        self.cdt = [DT[t["dtype"]] for t in self.tensors]
        sdt = [mx.uint16 if d == mx.bfloat16 else d for d in self.cdt]
        mk = lambda n: [mx.zeros([n] + t["shape"], dtype=d) for t, d in zip(self.tensors, sdt)]
        self.pool = [mk(self.S) for _ in range(self.L)]
        self.stage = mk(self.E)
        mx.eval(self.pool, self.stage)
        self.pool_np = [[np.array(a, copy=False) for a in p] for p in self.pool]
        self.stage_np = [np.array(a, copy=False) for a in self.stage]
        # iovec gaps between tensors inside a record
        self.gaps = []
        pos = 0
        for t in self.tensors:
            self.gaps.append(bytearray(t["offset"] - pos) if t["offset"] > pos else None)
            pos = t["offset"] + t["nbytes"]
        self.slot_of = np.full((self.L, self.E), -1, np.int32)
        self.exp_in = np.full((self.L, self.S), -1, np.int32)
        self.last = np.zeros((self.L, self.E))
        self.freq = np.zeros((self.L, self.E))
        self.ruse = np.zeros((self.L, self.E))
        self.prior = np.zeros((self.L, self.E))
        self.pending = {}                         # (j, e) -> future (load into its slot)
        self.lock = threading.Lock()
        self.pool_exec = ThreadPoolExecutor(threads)
        self.clock = 0
        self.mode_cfg = mode_cfg
        self.shell = self._shell()
        self.gates = None
        self.reset_stats()

    # ------------------------------------------------------------------ setup
    def _shell(self):
        g = self.mode_cfg
        sh = switch_layers.SwitchGLU(g["input_dims"], g["hidden_dims"], 1)
        for proj in ("gate_proj", "up_proj", "down_proj"):
            q = switch_layers.QuantizedSwitchLinear(g["group_size"], 32, 1, bias=False, group_size=g["group_size"],
                                                     bits=g["bits"], mode=g["mode"])
            setattr(sh, proj, q)
        return sh

    def _bind(self, arrs):
        """Point the shell SwitchGLU at a set of stacked expert tensors (pool of a layer, or the stage)."""
        names = [t["name"] for t in self.tensors]
        for proj in ("gate_proj", "up_proj", "down_proj"):
            q = getattr(self.shell, proj)
            for part in ("weight", "scales", "biases"):
                n = f"{proj}.{part}"
                if n in names:
                    i = names.index(n)
                    a = arrs[i]
                    setattr(q, part, a.view(self.cdt[i]) if a.dtype != self.cdt[i] else a)
                elif part == "biases":
                    q.biases = None
        return self.shell

    @classmethod
    def attach(cls, model, store_dir, slots, **kw):
        """Replace every MoE block's switch_mlp of a lazily loaded mlx-lm model (load(..., lazy=True)) by
        OffloadSwitch before any expert weight is materialised; then materialise the rest."""
        blocks = []
        for i, layer in enumerate(model.layers):
            blk = getattr(layer, "mlp", None) or getattr(layer, "block_sparse_moe", None)
            if blk is not None and hasattr(blk, "switch_mlp"):
                blocks.append(blk)
        gp = blocks[0].switch_mlp.gate_proj
        cfg = dict(group_size=gp.group_size, bits=gp.bits, mode=getattr(gp, "mode", "affine"),
                   input_dims=gp.scales.shape[2] * gp.group_size, hidden_dims=gp.weight.shape[1])
        eng = cls(store_dir, slots, mode_cfg=cfg, **kw)
        assert len(blocks) == eng.L, (len(blocks), eng.L)
        eng.blocks = blocks
        eng.mt = model.args.model_type
        eng.k = getattr(model.args, "num_experts_per_tok", None)
        for j, blk in enumerate(blocks):
            blk.switch_mlp = OffloadSwitch(eng, j)
        mx.eval(model.parameters())
        return eng

    def reset_stats(self):
        self.stats = dict(loads=0, bytes=0, stall_s=0.0, prefetch_loads=0, prefetch_used=0, la1_issued=0,
                          retained=0, staged=0)

    # ------------------------------------------------------------------ I/O
    def _read_into(self, j, e, dst_views):
        iov = []
        for gap, v in zip(self.gaps, dst_views):
            if gap is not None:
                iov.append(gap)
            iov.append(memoryview(v).cast("B"))
        n = os.preadv(self.fd, iov, (j * self.E + e) * self.rec)
        return n

    def _score(self, j, es):
        if self.policy == "lru" and not (self.policy_sw and self.phase == "prefill"):
            return self.last[j, es]
        if self.policy == "lfu" or (self.policy_sw and self.phase == "prefill"):
            return self.freq[j, es] + 1e9 * self.prior[j, es]
        if self.policy == "moeinf":
            return self.ruse[j, es] * 1e6 + self.freq[j, es]
        return self.ruse[j, es] + self.lam * self.prior[j, es] + 1e-6 * self.freq[j, es]   # hyb

    @property
    def policy_sw(self):
        return getattr(self, "_sw", False)

    def _victim(self, j, protect, min_score=None):
        occ = self.exp_in[j]
        free = np.flatnonzero(occ < 0)
        if len(free):
            return int(free[0])
        cand = [s for s in range(self.S) if occ[s] not in protect and (j, int(occ[s])) not in self.pending]
        if not cand:
            return None
        sc = self._score(j, occ[cand])
        i = int(np.argmin(sc))
        if min_score is not None and sc[i] >= min_score:
            return None
        return cand[i]

    def _assign(self, j, e, s):
        old = self.exp_in[j, s]
        if old >= 0:
            self.slot_of[j, old] = -1
        self.exp_in[j, s] = e
        self.slot_of[j, e] = s

    def _load_async(self, j, e, protect, background=False):
        if self.slot_of[j, e] >= 0 or (j, e) in self.pending:
            return
        ms = self._score(j, np.array([e]))[0] if background else None
        s = self._victim(j, protect, ms)
        if s is None:
            return
        self._assign(j, e, s)
        views = [v[s] for v in self.pool_np[j]]
        self.pending[(j, e)] = self.pool_exec.submit(self._read_into, j, e, views)
        self.stats["loads"] += 1
        self.stats["bytes"] += self.rec
        if background:
            self.stats["prefetch_loads"] += 1

    def _wait(self, j, es):
        t = time.perf_counter()
        for e in es:
            f = self.pending.pop((j, int(e)), None)
            if f is not None:
                f.result()
        self.stats["stall_s"] += time.perf_counter() - t

    # ------------------------------------------------------------------ prefetch API
    def new_request(self, prior=None):
        """prior: [L, E] expected decode uses of each expert for this request (or None)."""
        self.ruse[:] = 0
        self.prior[:] = 0 if prior is None else prior
        if prior is not None:                     # background: most valuable predicted experts first
            order = np.argsort(-prior, axis=None)
            for flat in order[: self.S * self.L]:
                j, e = divmod(int(flat), self.E)
                if prior[j, e] <= 0:
                    break
                self._load_async(j, e, protect=set(), background=True)

    def _gate_select(self, j, x):
        g, k = self.blocks[j].gate, self.k
        if isinstance(g, (nn.Linear, nn.QuantizedLinear)):
            logits = g(x)
            if self.mt == "olmoe":
                w = mx.softmax(logits, axis=-1, precise=True)
                return mx.argpartition(-w, kth=k - 1, axis=-1)[..., :k]
            if self.mt == "qwen3_moe":
                w = mx.softmax(logits, axis=-1, precise=True)
                return mx.argpartition(w, kth=-k, axis=-1)[..., -k:]
            return mx.argpartition(-logits, kth=k - 1, axis=-1)[..., :k]
        out = g(x)
        return out[0] if isinstance(out, tuple) else out

    # ------------------------------------------------------------------ forward
    def forward(self, j, x, indices):
        idx = np.array(indices)                   # forces the graph up to this layer's gate
        uniq = np.unique(idx)
        self.clock += 1
        ntok = int(np.prod(idx.shape[:-1]))
        if self.la1 and ntok == 1 and j + 1 < self.L:
            nxt = np.unique(np.array(self._gate_select(j + 1, x)))
            self.stats["la1_issued"] += len(nxt)
            for e in nxt:
                self._load_async(j + 1, int(e), protect=set())
        if len(uniq) <= self.S:
            y = self._forward_pool(j, x, idx, uniq)
        else:
            y = self._forward_stage(j, x, idx, uniq)
        self.last[j, uniq] = self.clock
        self.freq[j, uniq] += 1
        self.ruse[j, uniq] += 1
        return y

    def _forward_pool(self, j, x, idx, uniq):
        protect = set(int(e) for e in uniq)
        for e in uniq:
            if self.slot_of[j, e] < 0 and (j, int(e)) not in self.pending:
                self._load_async(j, int(e), protect)
        self._wait(j, uniq)
        remap = self.slot_of[j][idx].astype(np.uint32)
        return switch_layers.SwitchGLU.__call__(self._bind(self.pool[j]), x, mx.array(remap))

    def _forward_stage(self, j, x, idx, uniq):
        res = np.array([self.slot_of[j, e] >= 0 or (j, int(e)) in self.pending for e in uniq])
        miss = uniq[~res]
        self._wait(j, uniq[res])
        futs = [self.pool_exec.submit(self._read_into, j, int(e), [v[int(e)] for v in self.stage_np]) for e in miss]
        t = time.perf_counter()
        for f in futs:
            f.result()
        self.stats["stall_s"] += time.perf_counter() - t
        self.stats["loads"] += len(miss)
        self.stats["bytes"] += len(miss) * self.rec
        self.stats["staged"] += len(miss)
        in_pool = self.slot_of[j][idx] >= 0
        pool_idx = np.where(in_pool, self.slot_of[j][idx], 0).astype(np.uint32)
        y_pool = switch_layers.SwitchGLU.__call__(self._bind(self.pool[j]), x, mx.array(pool_idx))
        if len(miss):
            y_stage = switch_layers.SwitchGLU.__call__(self._bind(self.stage), x, mx.array(idx.astype(np.uint32)))
            y = mx.where(mx.array(in_pool)[..., None], y_pool, y_stage)
        else:
            y = y_pool
        mx.eval(y)                                # the staging buffer is reused by the next layer
        # retention (E4): keep the highest-scoring experts among resident + staged
        self.ruse[j, uniq] += 1                   # count this use before deciding
        cand = np.concatenate([self.exp_in[j][self.exp_in[j] >= 0], miss]).astype(np.int64)
        keep = set(cand[np.argsort(-self._score(j, cand))[: self.S]].tolist())
        self.ruse[j, uniq] -= 1
        for e in miss:
            if int(e) in keep:
                s = self._victim(j, protect=keep)
                if s is None:
                    continue
                self._assign(j, int(e), s)
                for dst, src in zip(self.pool_np[j], self.stage_np):
                    dst[s] = src[int(e)]
                self.stats["retained"] += 1
        return y
