"""Multi-user plan 4.1.2: expert offloading for transformers 5.2 MoE models on one GPU (H200 over PCIe), with
use-time substitution -- the CUDA counterpart of experiments/moe_edge/offload_engine.py + subst_offload.py.

Experts (each MoE layer's `experts` module: gate_up_proj [E, 2I, D], down_proj [E, D, I]; GLM-4.5 and Qwen3-MoE)
live in pinned host memory; every layer has a GPU pool of S_j slots (sizes from calibration counts,
tsim_mu.layer_caps) and the experts module is pointed at its pool with expert ids remapped to slots, so the model's own
experts implementation (eager / batched_mm / grouped_mm) runs unchanged.
  decode   (distinct experts <= S_j): missing experts are copied into slots (LFU victims, the layer's own experts
           protected) on a side stream; the compute stream waits for them.
  prefill  (more distinct experts than slots): resident experts are copied GPU->GPU and missing ones host->GPU into a
           full-layer staging buffer, the layer runs on it with the original ids, then the highest-scoring of
           resident + staged are kept (as the Mac engine).
Substitution (decode only, set eng.subst): the MoE block's routing is patched (GLM: Glm4MoeMoE.route_tokens_to_experts;
Qwen3: Qwen3MoeTopKRouter.forward); a late expert (not in the pool, or its copy not finished) is replaced, least
routing weight first and within the budget, by the pool expert with the highest selection score the token does not
use; weights are recomputed from the router's scores of the new set; a replaced expert is not copied.

  eng = HFOffload.attach(model, caps, device="cuda:0"); eng.warm(counts); eng.subst = Subst(...)
  (eng.phase follows the input: a forward with one token per sequence is decode)

Fast decode path (plan 4.2.6, eng.fast = True, needs eng.subst; Subst("none") for no substitution): the slot table
lives on the GPU too; the routing patch finds, on the GPU, each row's late positions (ordered by routing weight) and
its candidate substitutes (ready, not selected, highest selection score), brings them to the CPU in ONE transfer,
runs the budget row by row exactly as Subst.apply (Subst.decide), copies the new ids back, issues the reads of what is
still missing and makes the compute stream wait for them; the experts module then remaps on the GPU (no further sync).
"""
import time
import types

import numpy as np
import torch


class Subst:
    """The substitution rule and budget (same as tsim_mu.Subst / subst_offload.Subst)."""

    def __init__(self, spec, L, E, seed=0):
        self.spec, self.b, self.rate = spec, None, None
        parts = [] if spec == "none" else spec.split(":")
        for i in range(0, len(parts), 2):
            if parts[i] == "tok":
                self.b = int(parts[i + 1])
            elif parts[i] == "rate":
                self.rate = float(parts[i + 1])
            else:
                raise ValueError(f"subst spec {spec}")
        self.rng = np.random.default_rng(seed)
        self.uses = self.changed = self.late = 0
        self.layer_uses, self.layer_changed = np.zeros(L, np.int64), np.zeros(L, np.int64)

    def decide(self, j, n_late, order, cand, k):
        """The budget of apply() on precomputed arrays (fast path): n_late [N] late positions per row, order [N, k]
        positions ordered late-first by routing weight, cand [N, k] substitutes (best first). -> m [N] changes."""
        m = np.zeros(len(n_late), np.int64)
        for n in range(len(n_late)):
            self.uses += k
            self.layer_uses[j] += k
            nl = int(n_late[n])
            if nl == 0:
                continue
            self.late += nl
            mm = nl if self.spec != "none" else 0
            if self.b is not None:
                mm = min(mm, self.b)
            if self.rate is not None:
                mm = min(mm, max(0, int(self.rate * self.uses + self.rng.random()) - self.changed))
            if mm <= 0:
                continue
            mm = min(mm, int((cand[n] >= 0).sum()))
            m[n] = mm
            self.changed += mm
            self.layer_changed[j] += mm
        return m

    def apply(self, ready, j, inds, w, sel):
        N, k = inds.shape
        out = inds.copy()
        for n in range(N):
            self.uses += k
            self.layer_uses[j] += k
            late = ~ready[inds[n]]
            if not late.any():
                continue
            idx = np.flatnonzero(late)
            self.late += len(idx)
            idx = idx[np.argsort(w[n, idx], kind="stable")]
            m = len(idx) if self.spec != "none" else 0
            if self.b is not None:
                m = min(m, self.b)
            if self.rate is not None:
                m = min(m, max(0, int(self.rate * self.uses + self.rng.random()) - self.changed))
            if m <= 0:
                continue
            ok = ready.copy()
            ok[inds[n]] = False
            cand = [int(c) for c in np.argsort(-sel[n], kind="stable") if ok[c]]
            for jj, s in zip(idx[:m], cand):
                out[n, jj] = s
                self.changed += 1
                self.layer_changed[j] += 1
        return out

    def summary(self):
        return dict(change_rate=self.changed / max(self.uses, 1), late_rate=self.late / max(self.uses, 1),
                    layer_rate=(self.layer_changed / np.maximum(self.layer_uses, 1)).round(5).tolist())


def moe_blocks(model):
    out = []
    for layer in model.model.layers:
        m = getattr(layer, "mlp", None)
        ex = getattr(m, "experts", None)
        if ex is not None and getattr(ex, "gate_up_proj", None) is not None and ex.gate_up_proj.ndim == 3:
            out.append(m)
    return out


class HFOffload:
    def __init__(self, blocks, caps, device, policy="lfu"):
        self.blocks, self.device, self.policy = blocks, torch.device(device), policy
        self.L = len(blocks)
        ex0 = blocks[0].experts
        self.E = ex0.gate_up_proj.shape[0]
        self.S_j = np.asarray(caps, np.int64)
        self.S = int(self.S_j.max())
        self.copy_stream = torch.cuda.Stream(self.device)
        self.host_gu, self.host_dn, self.pool_gu, self.pool_dn = [], [], [], []
        for j, b in enumerate(blocks):
            ex = b.experts
            gu, dn = ex.gate_up_proj.data, ex.down_proj.data
            hg = torch.empty(gu.shape, dtype=gu.dtype, pin_memory=True)
            hd = torch.empty(dn.shape, dtype=dn.dtype, pin_memory=True)
            hg.copy_(gu)
            hd.copy_(dn)
            self.host_gu.append(hg)
            self.host_dn.append(hd)
            s = int(self.S_j[j])
            pg = torch.empty((s,) + tuple(gu.shape[1:]), dtype=gu.dtype, device=self.device)
            pd = torch.empty((s,) + tuple(dn.shape[1:]), dtype=dn.dtype, device=self.device)
            self.pool_gu.append(pg)
            self.pool_dn.append(pd)
            ex.gate_up_proj = torch.nn.Parameter(pg, requires_grad=False)
            ex.down_proj = torch.nn.Parameter(pd, requires_grad=False)
            ex.num_experts = s
            ex._eng, ex._j = self, j
            ex._class_forward = type(ex).forward
            ex.forward = types.MethodType(_offload_forward, ex)
            b._eng, b._j = self, j
        sh_g, sh_d = self.host_gu[0].shape[1:], self.host_dn[0].shape[1:]
        self.stage_gu = torch.empty((self.E,) + tuple(sh_g), dtype=self.host_gu[0].dtype, device=self.device)
        self.stage_dn = torch.empty((self.E,) + tuple(sh_d), dtype=self.host_dn[0].dtype, device=self.device)
        self.slot_of = np.full((self.L, self.E), -1, np.int64)
        self.exp_in = np.full((self.L, self.S), -1, np.int64)
        for j, s in enumerate(self.S_j):
            self.exp_in[j, s:] = -2
        self.freq = np.zeros((self.L, self.E))
        self.last = np.zeros((self.L, self.E))
        self.clock = 0
        self.pending = {}                                  # (j, e) -> cuda event of its copy
        self.slot_gpu = torch.full((self.L, self.E), -1, dtype=torch.long, device=self.device)
        self.fast = False
        self.decided = {}                                  # j -> True when the fast routing handled layer j
        self.pin_buf = {}                                  # (j, shape) -> pinned host buffer for new ids
        self.subst = None
        self.phase = "prefill"
        self.route_log = None
        self.reset_stats()

    @classmethod
    def attach(cls, model, caps, device="cuda:0", policy="lfu"):
        """model: loaded on the CPU. Moves everything except the routed experts to `device`, then builds the pools."""
        blocks = moe_blocks(model)
        keep = {id(b.experts) for b in blocks}
        for name, mod in model.named_modules():
            if id(mod) in keep:
                continue
            for pn, p in list(mod.named_parameters(recurse=False)):
                setattr(mod, pn, torch.nn.Parameter(p.data.to(device), requires_grad=False))
            for bn, buf in list(mod.named_buffers(recurse=False)):
                mod._buffers[bn] = buf.to(device)
        eng = cls(blocks, caps, device, policy)
        install()
        link_qwen_routers(eng)

        def phase_hook(_mod, args, kwargs):                 # decode = one new token per sequence
            ids = kwargs.get("input_ids", args[0] if args else None)
            if ids is not None:
                eng.phase = "decode" if ids.shape[1] == 1 else "prefill"
        model.register_forward_pre_hook(phase_hook, with_kwargs=True)
        return eng

    def reset_stats(self):
        self.stats = dict(loads=0, bytes=0, stall_s=0.0, staged=0, retained=0)

    # ---------------------------------------------------------------- cache
    def _score(self, j, es):
        if self.policy == "lru":
            return self.last[j, es]
        return self.freq[j, es]

    def _victim(self, j, protect):
        occ = self.exp_in[j][: int(self.S_j[j])]
        free = np.flatnonzero(occ == -1)
        if len(free):
            return int(free[0])
        ok = np.ones(len(occ), bool)
        if protect:
            ok &= ~np.isin(occ, np.fromiter(protect, np.int64))
        if self.pending:
            ok &= ~np.isin(occ, [e for (jj, e) in self.pending if jj == j])
        cand = np.flatnonzero(ok)
        if not len(cand):
            return None
        return int(cand[int(np.argmin(self._score(j, occ[cand])))])

    def _assign(self, j, e, s):
        old = self.exp_in[j, s]
        if old >= 0:
            self.slot_of[j, old] = -1
            if self.fast:
                self.slot_gpu[j, old] = -1
        self.exp_in[j, s] = e
        self.slot_of[j, e] = s
        if self.fast:
            self.slot_gpu[j, e] = s

    def _copy_in(self, j, e, s, dst_gu=None, dst_dn=None):
        dst_gu = self.pool_gu[j][s] if dst_gu is None else dst_gu
        dst_dn = self.pool_dn[j][s] if dst_dn is None else dst_dn
        with torch.cuda.stream(self.copy_stream):
            dst_gu.copy_(self.host_gu[j][e], non_blocking=True)
            dst_dn.copy_(self.host_dn[j][e], non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(self.copy_stream)
        self.stats["loads"] += 1
        self.stats["bytes"] += dst_gu.numel() * dst_gu.element_size() + dst_dn.numel() * dst_dn.element_size()
        return ev

    def ready_mask(self, j):
        m = self.slot_of[j] >= 0
        for (jj, e), ev in self.pending.items():
            if jj == j and not ev.query():
                m[e] = False
        return m

    def warm(self, counts):
        """Fill each layer's pool with its most used experts under counts [L, E]; lfu counts start from them."""
        torch.cuda.synchronize(self.device)
        self.pending.clear()
        self.slot_of[:] = -1
        self.exp_in[self.exp_in >= 0] = -1
        self.last[:] = 0
        self.clock = 0
        self.freq[:] = counts
        for j in range(self.L):
            for e in np.argsort(-counts[j], kind="stable")[: int(self.S_j[j])]:
                s = self._victim(j, set())
                self._assign(j, int(e), s)
                self._copy_in(j, int(e), s)
        self.slot_gpu.copy_(torch.as_tensor(self.slot_of, device=self.device))
        torch.cuda.synchronize(self.device)
        self.reset_stats()

    # ---------------------------------------------------------------- forward of one layer's experts
    def route_fast(self, j, idx, w, sel):
        """Fast decode routing for layer j: idx / w [N, k] the router's choice and weights, sel [N, E] the selection
        scores. One device->host transfer; returns the ids to run [N, k] (a GPU tensor; idx itself if unchanged)."""
        N, k = idx.shape
        ready = self.slot_gpu[j] >= 0                                       # [E] (no reads pending at routing time)
        late = ~ready[idx]                                                  # [N, k]
        key = torch.where(late, w.float(), torch.full_like(w, float("inf"), dtype=torch.float32))
        order = torch.argsort(key, dim=1, stable=True)                      # late positions first, lightest first
        ok = ready.unsqueeze(0).expand(N, -1).clone()
        ok.scatter_(1, idx, False)
        cand = torch.topk(sel.float().masked_fill(~ok, float("-inf")), k, dim=1)
        cid = torch.where(torch.isfinite(cand.values), cand.indices, torch.full_like(cand.indices, -1))
        pack = torch.cat([idx.long(), order, late.sum(1, keepdim=True), cid], 1).cpu().numpy()   # the one sync
        I, O, nl, C = pack[:, :k], pack[:, k:2 * k], pack[:, 2 * k], pack[:, 2 * k + 1:]
        m = self.subst.decide(j, nl, O, C, k)
        new = I.copy()
        rows = np.flatnonzero(m)
        for n in rows:
            new[n, O[n, :m[n]]] = C[n, :m[n]]
        if self.route_log is not None:
            self.route_log[j] = (I, new)
        # what is still missing: late positions beyond the changed ones
        miss = np.unique(np.concatenate([I[n, O[n, m[n]:nl[n]]] for n in range(N)] + [np.zeros(0, np.int64)]))
        uniq = np.unique(new)
        fits = len(uniq) <= self.S_j[j]                    # more distinct experts than slots: run() stages them
        evs = []
        if fits and len(miss):
            protect = set(uniq.tolist())
            for e in miss.tolist():
                s = self._victim(j, protect)
                if s is None:                                # cannot happen when it fits; be safe
                    fits = False
                    break
                self._assign(j, e, s)
                evs.append(self._copy_in(j, e, s))
        for ev in evs:                                                     # the GPU waits, the CPU does not
            torch.cuda.current_stream(self.device).wait_event(ev)
        if fits:
            self.clock += 1
            self.last[j, uniq] = self.clock
            self.freq[j, uniq] += 1
            self.decided[j] = True
        else:
            self.stats["fast_fallback"] = self.stats.get("fast_fallback", 0) + 1
        if not len(rows):
            return idx
        buf = self.pin_buf.get((j, new.shape))            # pinned, one per layer: a non-blocking copy, no sync;
        if buf is None:                                   # reused only at the next step (synced by then)
            buf = self.pin_buf[(j, new.shape)] = torch.empty(new.shape, dtype=idx.dtype, pin_memory=True)
        buf.numpy()[:] = new
        return buf.to(idx.device, non_blocking=True)

    def run(self, ex, j, hidden, idx, w):
        if self.decided.pop(j, False):                                      # fast path: routing already handled
            return ex._class_forward(ex, hidden, self.slot_gpu[j][idx], w)
        self.clock += 1
        ids = idx.detach().cpu().numpy()                    # syncs: the layer's routes are known here
        uniq = np.unique(ids)
        if len(uniq) <= self.S_j[j]:
            miss = uniq[self.slot_of[j, uniq] < 0]
            evs = []
            if len(miss):
                protect = set(uniq.tolist())
                for e in miss.tolist():
                    s = self._victim(j, protect)
                    self._assign(j, e, s)
                    self.pending[(j, e)] = self._copy_in(j, e, s)
            if self.pending:
                for e in uniq.tolist():
                    if (j, e) in self.pending:
                        evs.append(self.pending.pop((j, e)))
            if evs:
                t = time.perf_counter()
                for ev in evs:
                    ev.synchronize()
                self.stats["stall_s"] += time.perf_counter() - t
            remap = torch.as_tensor(self.slot_of[j][ids], device=idx.device)
            y = ex._class_forward(ex, hidden, remap, w)
        else:
            y = self._run_stage(ex, j, hidden, idx, w, uniq)
        self.last[j, uniq] = self.clock
        self.freq[j, uniq] += 1
        return y

    def _run_stage(self, ex, j, hidden, idx, w, uniq):
        res = uniq[self.slot_of[j, uniq] >= 0]
        miss = uniq[self.slot_of[j, uniq] < 0]
        for e in res:                                       # GPU -> GPU
            s = self.slot_of[j, e]
            self.stage_gu[e].copy_(self.pool_gu[j][s])
            self.stage_dn[e].copy_(self.pool_dn[j][s])
        evs = [self._copy_in(j, int(e), None, self.stage_gu[e], self.stage_dn[e]) for e in miss]
        t = time.perf_counter()
        for ev in evs:
            ev.synchronize()
        self.stats["stall_s"] += time.perf_counter() - t
        self.stats["staged"] += len(miss)
        gu, dn, ne = ex.gate_up_proj, ex.down_proj, ex.num_experts
        ex.gate_up_proj = torch.nn.Parameter(self.stage_gu, requires_grad=False)
        ex.down_proj = torch.nn.Parameter(self.stage_dn, requires_grad=False)
        ex.num_experts = self.E
        y = ex._class_forward(ex, hidden, idx, w)
        ex.gate_up_proj, ex.down_proj, ex.num_experts = gu, dn, ne
        # retention: keep the highest-scoring of resident + staged (this use counted)
        self.freq[j, uniq] += 1
        cand = np.concatenate([self.exp_in[j][self.exp_in[j] >= 0], miss]).astype(np.int64)
        keep = set(cand[np.argsort(-self._score(j, cand), kind="stable")[: int(self.S_j[j])]].tolist())
        self.freq[j, uniq] -= 1
        for e in miss:
            if int(e) in keep:
                s = self._victim(j, keep)
                if s is None:
                    continue
                self._assign(j, int(e), s)
                self.pool_gu[j][s].copy_(self.stage_gu[e])
                self.pool_dn[j][s].copy_(self.stage_dn[e])
                self.stats["retained"] += 1
        return y


def _offload_forward(self, hidden_states, top_k_index, top_k_weights):
    return self._eng.run(self, self._j, hidden_states, top_k_index, top_k_weights)


# -------------------------------------------------------------------- routing patches (substitution in decode)
def _decide(blk, inds, weights, sel):
    eng, j = blk._eng, blk._j
    I = inds.detach().cpu().numpy().astype(np.int64)
    W = weights.detach().float().cpu().numpy()
    S = sel.detach().float().cpu().numpy()
    new = eng.subst.apply(eng.ready_mask(j), j, I, W, S)
    if eng.route_log is not None:
        eng.route_log[j] = (I, new, S, W)
    if (new == I).all():
        return None
    return torch.as_tensor(new, device=inds.device, dtype=inds.dtype)


def _glm_route(self, router_logits):
    idx, w = _ORIG["glm"](self, router_logits)
    eng = getattr(self, "_eng", None)
    if eng is None or eng.subst is None or eng.phase != "decode":
        return idx, w
    probs = router_logits.sigmoid()
    if eng.fast:
        new = eng.route_fast(self._j, idx, w, probs + self.gate.e_score_correction_bias)
        if new is idx:
            return idx, w
        w = probs.gather(1, new)
        if self.norm_topk_prob:
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
        return new, w * self.routed_scaling_factor
    new = _decide(self, idx, w, probs + self.gate.e_score_correction_bias)
    if new is None:
        return idx, w
    w = probs.gather(1, new)
    if self.norm_topk_prob:
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
    return new, w * self.routed_scaling_factor


def _qwen_router(self, hidden_states):
    probs, w, idx = _ORIG["qwen"](self, hidden_states)          # transformers 5.2: the softmax, not the logits
    blk = getattr(self, "_blk", None)
    if blk is None or blk._eng.subst is None or blk._eng.phase != "decode":
        return probs, w, idx
    if blk._eng.fast:
        new = blk._eng.route_fast(blk._j, idx, w, probs)
        if new is idx:
            return probs, w, idx
        w = probs.gather(1, new)
        if self.norm_topk_prob:
            w = w / w.sum(dim=-1, keepdim=True)
        return probs, w.to(probs.dtype), new
    new = _decide(blk, idx, w, probs)
    if new is None:
        return probs, w, idx
    w = probs.gather(1, new)
    if self.norm_topk_prob:
        w = w / w.sum(dim=-1, keepdim=True)
    return probs, w.to(probs.dtype), new


_ORIG = {}


def install():
    try:
        from transformers.models.glm4_moe import modeling_glm4_moe as g
        if "glm" not in _ORIG:
            _ORIG["glm"] = g.Glm4MoeMoE.route_tokens_to_experts
            g.Glm4MoeMoE.route_tokens_to_experts = _glm_route
    except ImportError:
        pass
    try:
        from transformers.models.qwen3_moe import modeling_qwen3_moe as q
        if "qwen" not in _ORIG:
            _ORIG["qwen"] = q.Qwen3MoeTopKRouter.forward
            q.Qwen3MoeTopKRouter.forward = _qwen_router
    except ImportError:
        pass


def link_qwen_routers(eng):
    """Qwen3: the router module needs its block (engine, layer) for the substitution."""
    for b in eng.blocks:
        if hasattr(b, "gate") and not hasattr(b, "route_tokens_to_experts"):
            b.gate._blk = b
