"""Multi-user MoE decode simulator (plan 2026-10-21-moe-multiuser-scheduling, item 0.2).

Merges tsim_ahead.py (single user, multi-layer-ahead prefetch) and the synchronous part of tsim_async.py (N decode
streams, B advance per step); both files stay unchanged so their results remain reproducible.

Modes
  seq   requests one after another, each with prefill (P-layer prefetch window) and then decode: the single-user
        case (N = B = 1), as in P3. The cache carries over between requests.
  pool  N streams in a closed loop (each replays a trace request from a random offset and wraps around), decode
        only; every step B of them advance (fcfs | prev | oracle selection, as in tsim_async), as in Q1 / A0.

Per decode step (synchronous batching), for every MoE layer l:
  t_gate = t + attn(B) + tick                    attention + routing of the whole batch
  prefetch: for h = 1..H, the batch's predicted experts of layer l+h (top-M, optionally only scores >= p) replace
            the previous prediction for that target layer: queued experts the new one drops are cancelled
  need: the batch's distinct experts of layer l; missing ones are loaded (demand loads go before prefetches)
  t = t_ready + gmm(m)                           grouped expert GEMM over the m distinct experts
One loader moves one expert at a time; the lookahead queue is ordered by deadline (absolute layer index).

Cache: C = f x L x E experts = resident region (rho x C, a fixed set, never evicted; filled by --fill) + flowing
region (lru | lfu | moeinf). Protection (P3): an expert prefetched for a target layer that has not run yet is
evicted last, and a prefetch whose only victims are such experts (needed no later than it) is dropped.
--protect_resident (C2) also protects predicted experts that were already cached.

Waste breakdown of decode prefetch loads, by the lookahead distance h of the prediction that first queued them:
  hit (used at its target layer) | late (not at the target, but at a later token) | early (evicted before its
  target layer ran) | corrected (a later prediction for the target dropped it) | final_wrong (still predicted,
  the target did not use it) | end (unused, still cached at the end).

Hardware (times in us): h200_pcie, mac_ssd (queue_microbench.py / tsim_async.py); h200_resident (all experts in
HBM: no loads); p3_glm4bit, p3_h200 (the bandwidth model of tsim.py / tsim_ahead.py, for the P3 regression only).

Policies "<evict>/<prefetch>": prefetch = none | la1 | ah<H>:<pred>[:m<M>][:p<thr>][:t<tau>] as in tsim_ahead
(pred: anchor_dump.py predictors GATE TOKGATE AV AVG ORACLE MIX, and with v4 traces GATEs AVGs (calibrated GATE
probability); without --preds, GATE comes from the trace's dec_gate_h and ORACLE from the true route).

  python experiments/moe_edge/tsim_mu.py --mode seq --trace outputs/moe/traces_v3/GLM-4.5-Air_cwq_M0_cot_chat_cap8 \
      --preds outputs/moe/anchor_preds/GLM-4.5-Air_cwq_M0_cot_chat_cap8 --n 50 --hw p3_glm4bit --cache 0.5 \
      --policy lru/none,lru/la1,lru/ah8:AVG:m8:p0.9 --out out.json
  python experiments/moe_edge/tsim_mu.py --mode pool --trace outputs/moe/traces_v3/GLM-4.5-Air_cwq_M0_cot_chat_cap8 \
      --N 64 --hw h200_pcie --cache 0.5 0.75 0.9 --policy lfu/none --out out.json
"""
import argparse
import glob
import heapq
import itertools
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def _legacy(eb, nonexp, mem_bw, ssd, act, c_pre):
    """tsim.py's bandwidth model in this simulator's terms (us): attention = non-expert bytes / BW, each expert
    = its bytes / BW, prefill = T x active params x c_pre per layer."""
    e = eb / mem_bw * 1e6
    return dict(attn=[(1, nonexp / mem_bw * 1e6)], tick=0.0, g0=e, g1=e, load=eb / ssd * 1e6, pre_tok=act * c_pre * 1e6)


HW = {
    # queue_microbench.py on H200 (GLM-4.5-Air bf16 shapes): non-fused attention + projections + router per layer vs
    # batch (context 2048), grouped GEMM 50 us + 12.5 us per extra expert, one 34.6 MB expert over PCIe 1.0 ms.
    # Prefill per token and layer from tsim_ahead (0.6e9 active params x 2e-6 s / 1e9).
    "h200_pcie": dict(attn=[(1, 138), (4, 146), (8, 155), (16, 172), (32, 246), (64, 363), (128, 581), (256, 1052)],
                      tick=32.0, g0=50.0, g1=12.5, load=1000.0, pre_tok=1.2),
    # GLM-4.5-Air 4bit on the M5 Max: non-expert layer 0.39 ms, 43 us per expert, 1.15 ms per expert from the SSD.
    "mac_ssd": dict(attn=[(1, 390), (8, 420), (32, 520), (64, 650), (128, 900), (256, 1400)],
                    tick=60.0, g0=43.0, g1=43.0, load=1150.0, pre_tok=22.95),
    # mac_ssd calibrated on the Mac prototype (plan 4.1.3, subst_offload.py, GLM-4.5-Air 4bit, single user, the two
    # no-substitution runs at cache 50% / 75%): 1.64 ms wait per expert read + 0.66 ms host work per demand read, and a
    # token without loads costs 23.7 ms (tick shifted to -208 us). Checked at B = 1 only (larger B: sensitivity check).
    "mac_ssd_cal": dict(attn=[(1, 390), (8, 420), (32, 520), (64, 650), (128, 900), (256, 1400)],
                        tick=-207.9, g0=43.0, g1=43.0, load=1637.5, load_cpu=660.5, pre_tok=22.95),
    # the same, but prefetch reads pay the host work too (where it comes from is not known: Python per read, or reads
    # competing with compute); mac_ssd_cal charges demand reads only. The two bracket the prefetch benefit.
    "mac_ssd_cal_all": dict(attn=[(1, 390), (8, 420), (32, 520), (64, 650), (128, 900), (256, 1400)],
                            tick=-207.9, g0=43.0, g1=43.0, load=1637.5, load_cpu=660.5, load_cpu_all=True, pre_tok=22.95),
    # the whole model in HBM: same compute as h200_pcie, nothing to load (the cache holds every expert)
    "h200_resident": dict(attn=[(1, 138), (4, 146), (8, 155), (16, 172), (32, 246), (64, 363), (128, 581), (256, 1052)],
                          tick=32.0, g0=50.0, g1=12.5, load=None, pre_tok=1.2),
    # P3 (tsim.HW["glm4bit"], tsim_ahead.HW["h200_glm_bf16"]): regression only
    "p3_glm4bit": _legacy(3 * 4096 * 1408 * 0.5625, 4.3e9 / 46, 242e9, 9e9, 0.27e9, 85e-6 / 1e9),
    "p3_h200": _legacy(3 * 4096 * 1408 * 2, 6.4e9 * 2 / 46, 3.5e12, 50e9, 0.6e9, 2e-6 / 1e9),
}
HMAX = 17                                          # histogram size over the lookahead distance h (0 = demand)
LOAD_BINS = np.array([0, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 1e9])  # queued prefetches / experts loadable per layer
WASTE = ("hit", "late", "early", "corrected", "final_wrong", "end")


def attn_cost(hw, B):
    xs, ys = zip(*hw["attn"])
    return float(np.interp(B, xs, ys))


def gmm_cost(hw, m):
    return 0.0 if m == 0 else hw["g0"] + hw["g1"] * (m - 1)


class Engine:
    """Expert cache (resident + flowing region) and one loader with three queues: 0 demand, 1 lookahead (ordered by
    deadline), 2 background. Keys are layer * E + expert."""

    def __init__(self, L, E, C, hw, evict="lru", rmask=None, protect_resident=False, cap_l=None):
        """cap_l: per-layer capacities (as the Mac prototype's per-layer pools; sum = C, eviction within the layer);
        default one cache for all layers (global eviction)."""
        n = L * E
        self.L, self.E, self.C, self.hw, self.evict = L, E, C, hw, evict
        self.protect_resident = protect_resident
        self.res = np.zeros(n, bool)                 # resident (or being loaded)
        self.queued = np.full(n, -1, np.int8)        # queue priority if waiting to be loaded, else -1
        self.ready = np.zeros(n)                     # time a resident expert becomes usable
        self.last = np.zeros(n)                      # last use (lru)
        self.freq = np.zeros(n)                      # use count (lfu; moeinf tie-break)
        self.ruse = np.zeros(n)                      # uses within the current request (moeinf, seq mode)
        self.pin = np.zeros(n, bool)                 # needed by the running layer
        self.used = np.zeros(n, bool)                # used since it was loaded
        self.want = np.full(n, -1.0)                 # deadline of a pending prediction (protection)
        self.now = 0.0                               # absolute index of the layer now running
        self.tq = {}                                 # deadline -> keys of the latest prediction for it
        self.acc = {}                                # deadline -> [weighted score sum over experts, weight sum]
        self.k = 8                                   # experts per token (for the per-layer load capacity)
        self.load_hist = np.zeros(len(LOAD_BINS) - 1, np.int64)   # queue load seen by the adaptive threshold
        self.rmask = np.zeros(n, bool) if rmask is None else rmask.copy()
        self.res[self.rmask] = True
        self.used[self.rmask] = True
        self.C_flow = C - int(self.rmask.sum())
        self.n_flow = 0
        self.cap_l = None if cap_l is None else np.asarray(cap_l, np.int64)
        if self.cap_l is not None:
            assert not self.rmask.any() and self.cap_l.sum() == C, "per-layer capacities: no resident region"
        self.n_l = np.zeros(L, np.int64)             # cached keys per layer (cap_l only)
        self.t_loader = 0.0
        self.qs = [[], [], []]                       # heaps of (order, seq, key, t_issue)
        self.seq = 0
        self.dur = hw["load"]
        self.phase = "dec"
        # waste bookkeeping: why the cached copy was loaded
        self.src = np.full(n, -1, np.int8)           # 0 demand, 1 decode prefetch, 2 prefill prefetch, 3 warm / bg
        self.src_h = np.zeros(n, np.int16)
        self.src_dl = np.full(n, -1.0)
        self.stale = np.zeros(n, bool)               # no longer in the latest prediction for its target
        self.q_h = np.zeros(n, np.int16)             # h / deadline of the prediction that queued it
        self.q_dl = np.full(n, -1.0)
        self.needed = np.zeros(n, bool)              # keys of the need() in progress
        self.stats = dict(loaded=0, demand=0, la=0, bg=0, bg_dropped=0, wasted=0, la_dropped=0, la_cancelled=0,
                          overflow=0, res_loads=0, res_swaps=0)
        self.pf = {w: np.zeros(HMAX, np.int64) for w in ("loads",) + WASTE}

    # ---- eviction
    def base_score(self, keys):
        if self.evict == "lru":
            return self.last[keys]
        if self.evict == "lfu":
            return self.freq[keys]
        if self.evict == "moeinf":
            return self.ruse[keys] * 1e6 + self.freq[keys]
        raise ValueError(self.evict)

    def victim(self, cand):
        """Lowest score among unprotected candidates; protected ones only if nothing else is left (tsim_ahead adds
        1e12 to protected scores: the same order, without its float resolution limit)."""
        prot = self.want[cand] >= self.now
        pool = cand[~prot] if not prot.all() else cand
        return int(pool[int(np.argmin(self.base_score(pool)))])

    def _evict(self, v):
        self.res[v] = False
        self.n_flow -= 1
        self.n_l[v // self.E] -= 1
        if not self.used[v]:
            self.stats["wasted"] += 1
            if self.src[v] == 1:
                h = self.src_h[v]
                if self.src_dl[v] > self.now or (self.src_dl[v] == self.now and self.needed[v]):
                    self.pf["early"][h] += 1
                elif self.stale[v]:
                    self.pf["corrected"][h] += 1
                else:
                    self.pf["final_wrong"][h] += 1
        self.src[v] = -1

    # ---- queues
    def issue(self, keys, prio, t, deadline=None, h=0):
        keys = np.asarray(keys).ravel()
        keys = keys[~self.res[keys]]
        for key in keys:
            key = int(key)
            q = self.queued[key]
            if q != -1 and q <= prio:
                continue                             # already queued at the same or a higher priority
            self.queued[key] = prio
            self.seq += 1
            heapq.heappush(self.qs[prio], (deadline if prio == 1 and deadline is not None else t, self.seq, key, t))
            if prio == 1:
                self.q_h[key], self.q_dl[key] = h, (deadline if deadline is not None else -1.0)
        if prio == 1 and deadline is not None:
            self.want[keys] = np.maximum(self.want[keys], deadline)

    def lookahead(self, keys, t, deadline, h):
        """A prediction for one target layer; it replaces the previous prediction for that layer."""
        keys = np.asarray(keys).ravel()
        old = self.tq.get(deadline)
        if old is not None:
            drop = np.setdiff1d(old, keys)
            if len(drop):
                dq = drop[self.queued[drop] == 1]
                if len(dq):
                    self.queued[dq] = -1
                    self.want[dq] = -1
                    self.stats["la_cancelled"] += len(dq)
                dr = drop[self.res[drop] & (self.src[drop] == 1) & (self.src_dl[drop] == deadline)]
                self.stale[dr] = True
        self.tq[deadline] = keys
        kr = keys[self.res[keys]]
        if len(kr):
            self.stale[kr[(self.src[kr] == 1) & (self.src_dl[kr] == deadline)]] = False
            if self.protect_resident:                # C2: cached experts that are predicted are protected too
                self.want[kr] = np.maximum(self.want[kr], deadline)
        self.issue(keys, 1, t, deadline, h)

    def clear_background(self):
        for item in self.qs[2]:
            if self.queued[item[2]] == 2:
                self.queued[item[2]] = -1
        self.qs[2] = []

    def _pop(self):
        while True:
            best = None
            for p in range(3):
                hq = self.qs[p]
                while hq and (self.queued[hq[0][2]] != p or self.res[hq[0][2]]):   # stale entries
                    heapq.heappop(hq)
                if hq and hq[0][3] <= self.t_loader:
                    best = p
                    break
            if best is None:
                heads = [(hq[0][3], p) for p, hq in enumerate(self.qs) if hq]
                if not heads:
                    return None
                return ("wait", min(heads)[0])
            _o, _s, key, _t = heapq.heappop(self.qs[best])
            return ("load", best, key)

    def _load_resident(self, prio, key):
        """Load a key of the resident region (it entered at a refresh): the slot it takes was handed to the flowing
        region when its predecessor left, so the flowing region gives one back."""
        self.queued[key] = -1
        if self.res[key]:
            return
        while self.n_flow > self.C_flow:
            cand = np.flatnonzero(self.res & ~self.pin & ~self.rmask)
            if len(cand) == 0:
                break
            self._evict(self.victim(cand))
        self.t_loader += self.dur
        self.res[key] = True
        self.ready[key] = self.t_loader
        self.used[key] = False
        self.src[key] = 0 if prio == 0 else 3
        self.src_h[key], self.src_dl[key] = 0, self.now
        self.stats["loaded"] += 1
        self.stats[("demand", "la", "bg")[prio]] += 1
        self.stats["res_loads"] += 1

    def set_resident(self, new, t):
        """Refresh the resident region to `new` (same size): leaving experts stay cached as flowing ones, entering
        ones that are not cached are loaded in the background (lowest priority)."""
        leave = self.rmask & ~new
        enter = new & ~self.rmask
        self.rmask[leave] = False
        self.n_flow += int((self.res & leave).sum())
        self.rmask[enter] = True
        self.n_flow -= int((self.res & enter).sum())
        self.C_flow = self.C - int(self.rmask.sum())
        self.stats["res_swaps"] += int(enter.sum())
        self.issue(np.flatnonzero(enter & ~self.res), 2, t)

    def _full(self, key):
        if self.cap_l is None:
            return self.n_flow >= self.C_flow
        return self.n_l[key // self.E] >= self.cap_l[key // self.E]

    def _cand(self, key):
        """Eviction candidates for loading key: any cached flowing key, or with cap_l those of key's layer."""
        if self.cap_l is None:
            return np.flatnonzero(self.res & ~self.pin & ~self.rmask)
        lo = key // self.E * self.E
        return lo + np.flatnonzero(self.res[lo:lo + self.E] & ~self.pin[lo:lo + self.E])

    def _load(self, prio, key):
        if self.rmask[key]:
            return self._load_resident(prio, key)
        if prio == 1 and not self.res[key] and self._full(key):
            cand = self._cand(key)
            if len(cand):
                v = self.victim(cand)
                if self.want[v] >= self.now and self.want[v] <= self.want[key]:
                    self.queued[key] = -1            # only experts needed no later than this one could go
                    self.want[key] = -1
                    self.stats["la_dropped"] += 1
                    return
        self.queued[key] = -1
        if self.res[key]:
            return
        while self._full(key):
            cand = self._cand(key)
            if len(cand) == 0:
                if prio != 0:
                    return
                self.stats["overflow"] += 1          # everything cached is pinned: exceed the capacity for now
                break
            v = self.victim(cand)
            if prio == 2 and self.base_score(np.array([v]))[0] >= self.base_score(np.array([key]))[0]:
                self.stats["bg_dropped"] += 1
                return
            self._evict(v)
        self.t_loader += self.dur
        self.res[key] = True
        self.n_flow += 1
        self.n_l[key // self.E] += 1
        self.ready[key] = self.t_loader
        self.used[key] = False
        self.stats["loaded"] += 1
        self.stats[("demand", "la", "bg")[prio]] += 1
        self.stale[key] = False
        if prio == 1:
            self.src[key] = 1 if self.phase == "dec" else 2
            self.src_h[key], self.src_dl[key] = self.q_h[key], self.q_dl[key]
            if self.phase == "dec":
                self.pf["loads"][self.q_h[key]] += 1
        else:
            self.src[key] = 0 if prio == 0 else 3
            self.src_h[key], self.src_dl[key] = 0, self.now

    def advance(self, until, stop_keys=None):
        """Run the loader until time `until`; with stop_keys, until all of them are resident."""
        while True:
            if stop_keys is not None and self.res[stop_keys].all():
                return
            if stop_keys is None and self.t_loader >= until:
                return
            nxt = self._pop()
            if nxt is None:
                if stop_keys is not None:            # a needed expert is missing and nothing is queued
                    self.issue(stop_keys[~self.res[stop_keys]], 0, self.t_loader)
                    continue
                self.t_loader = max(self.t_loader, until)
                return
            if nxt[0] == "wait":
                if stop_keys is None and nxt[1] >= until:
                    self.t_loader = max(self.t_loader, until)
                    return
                self.t_loader = max(self.t_loader, nxt[1])
                continue
            self._load(nxt[1], nxt[2])

    def need(self, keys, t):
        """The layer running at time t needs `keys`; returns when they are all usable (>= t)."""
        keys = np.asarray(keys).ravel()
        self.needed[keys] = True
        self.advance(t)
        self.pin[:] = False
        self.pin[keys] = True
        self.issue(keys, 0, t)
        self.advance(t, stop_keys=keys)
        done = max(t, float(self.ready[keys].max()))
        fresh = keys[(self.src[keys] == 1) & ~self.used[keys]]   # decode prefetches used for the first time
        if len(fresh):
            at = self.src_dl[fresh] == self.now
            np.add.at(self.pf["hit"], self.src_h[fresh[at]], 1)
            np.add.at(self.pf["late"], self.src_h[fresh[~at]], 1)
        self.last[keys] = done
        self.used[keys] = True
        self.want[keys] = -1
        self.needed[keys] = False
        if self.cap_l is not None:
            self._trim(int(keys[0]) // self.E)
        return done

    def _trim(self, l):
        """cap_l: a layer that needed more experts than it holds (prefill) went over its capacity; as the Mac engine
        (staging buffer, then keep the highest-scoring of cached + staged), drop the lowest-scoring after the use."""
        over = self.n_l[l] - self.cap_l[l]
        if over <= 0:
            return
        lo = l * self.E
        inl = lo + np.flatnonzero(self.res[lo:lo + self.E])
        for v in inl[np.argsort(self.base_score(inl), kind="stable")[:over]]:
            self._evict(int(v))

    def warm(self, f):
        """Fill the flowing region with the most used keys under counts f (tsim_async.warm: steady state, no cold
        start); their use counts start from f, every other key from 0."""
        if self.cap_l is not None:                   # each layer: its own most used keys, cap_l of them
            fl = np.asarray(f).reshape(self.L, self.E)
            order = [l * self.E + int(e) for l in range(self.L)
                     for e in np.argsort(-fl[l], kind="stable")[:self.cap_l[l]]]
        else:
            order = [int(x) for x in np.argsort(f) if not self.rmask[x]][-self.C_flow:] if self.C_flow > 0 else []
        for key in order:
            self.freq[key] = f[key]
            self.res[key] = True
            self.used[key] = True
            self.src[key] = 3
            self.n_l[key // self.E] += 1
        self.n_flow = len(order)

    def waste_summary(self):
        end = self.res & (self.src == 1) & ~self.used
        lh = self.load_hist / max(self.load_hist.sum(), 1)
        loadq = {f"load_p{q}": float(LOAD_BINS[min(np.searchsorted(np.cumsum(lh), q / 100), len(lh) - 1) + 1])
                 for q in (50, 90)} if self.load_hist.sum() else {}
        pf = {w: self.pf[w].copy() for w in self.pf}
        pf["end"] = np.bincount(self.src_h[end], minlength=HMAX)[:HMAX]
        n = max(int(pf["loads"].sum()), 1)
        out = dict(pf_loads=int(pf["loads"].sum()))
        for w in WASTE:
            out[f"pf_{w}"] = int(pf[w].sum())
            out[f"pf_{w}_frac"] = float(pf[w].sum()) / n
        out["pf_waste_frac"] = sum(out[f"pf_{w}"] for w in ("early", "corrected", "final_wrong", "end")) / n
        out["pf_by_h"] = {w: pf[w].tolist() for w in ("loads",) + WASTE}
        out["load_hist"] = self.load_hist.tolist()
        out.update(loadq)
        return out


# ---------------------------------------------------------------- policies and predictions
def parse_pref(pref):
    """none | la1 | ah<H>:<pred>[:m<M>][:p<thr>][:t<tau>] (tsim_ahead) plus the plan-1.2 mechanisms:
      :g<gamma>   confidence accumulation: a target layer's predictions are averaged with weights gamma^(h-1)
                  (without it the newest prediction replaces the older ones, as in P3)
      :d<d>       freeze window: when the target is <= d layers ahead, queued experts are not cancelled and new
                  ones need confidence >= :x<tau_high> (default: the threshold)
      :k<kappa>   adaptive threshold: tau = thr + kappa x queue load (queued prefetches / experts loadable per layer)
      :kl<kappa>  the same with log2(1 + queue load) (the load is heavy-tailed)
      :hl<delta>  h-layered threshold: predictions one layer ahead need tau + delta"""
    d = dict(H=0, pred="GATE", M=None, tau=None, thr=None, gamma=None, freeze=0, tau_high=None, kappa=0.0, hl=0.0,
             kappa_log=0.0)
    if pref in ("none", "la1"):
        d["H"] = 1 if pref == "la1" else 0
        return d
    parts = pref.split(":")
    d.update(H=int(parts[0][2:]), pred=parts[1])
    for x in parts[2:]:
        if x.startswith("hl"):
            d["hl"] = float(x[2:])
        elif x.startswith("kl"):
            d["kappa_log"] = float(x[2:])
        elif x[0] == "m":
            d["M"] = int(x[1:])
        elif x[0] == "t":
            d["tau"] = float(x[1:])
        elif x[0] == "p":
            d["thr"] = float(x[1:])
        elif x[0] == "g":
            d["gamma"] = float(x[1:])
        elif x[0] == "d":
            d["freeze"] = int(x[1:])
        elif x[0] == "x":
            d["tau_high"] = float(x[1:])
        elif x[0] == "k":
            d["kappa"] = float(x[1:])
        else:
            raise ValueError(f"prefetch option {x}")
    d["adv"] = d["gamma"] is not None or d["freeze"] > 0 or d["kappa"] != 0 or d["hl"] != 0 or d["kappa_log"] != 0
    return d


def pred_names(pp):
    """anchor_dump arrays a prefetch policy reads."""
    if not pp["H"]:
        return set()
    names = {"GATE", "AVG", "conf"} if pp["pred"] == "MIX" else {pp["pred"]}
    if pp["thr"] is not None or pp.get("adv"):
        names |= {nm + "_s" for nm in names if nm not in ("conf",)}
    return names


def predict(r, d, l, h, pp, M):
    """Expert ids (layer-local) predicted at layer l of token d of request r for layer l+h."""
    P = r.get("P")
    if pp["pred"] == "GATE" and pp["H"] == 1 and P is None:
        return r["la_dec"][l, d][:M]
    name = pp["pred"]
    if name == "MIX":
        name = "AVG" if h >= 2 and float(P["conf"][l, d]) >= pp["tau"] else "GATE"
    if P is not None and name in P:
        ids = P[name][l, d, h - 1, :M]
        if pp["thr"] is not None and name + "_s" in P:      # only the experts it is sure about
            ids = ids[P[name + "_s"][l, d, h - 1, :M] >= pp["thr"]]
        return ids
    if name == "GATE":
        return r["gate_h"][h - 1, l, d][:M]
    if name == "ORACLE":
        return r["dec"][l + h, d][:M]
    raise KeyError(f"predictor {name} not available")


def predict_scores(r, d, l, h, pp, M):
    """Top-M predicted experts of layer l+h with their predictor scores (1 where the predictor has none)."""
    P = r.get("P")
    name = pp["pred"]
    if P is not None and name in P:
        ids = P[name][l, d, h - 1, :M].astype(np.int64)
        sc = P[name + "_s"][l, d, h - 1, :M].astype(np.float64) if name + "_s" in P else np.ones(len(ids))
        ok = ids >= 0
        return ids[ok], sc[ok]
    ids = predict(r, d, l, h, dict(pp, thr=None), M).astype(np.int64)
    return ids, np.ones(len(ids))


def prefetch_adv(eng, toks, l, h, t_gate, pos, pp, M, E, tau):
    """Plan-1.2 issue rule for target layer l+h: (accumulated) confidence summed over the batch >= tau, ordered by
    confidence; inside the freeze window nothing queued is cancelled and additions need tau_high."""
    T = pos + l + h
    sv = np.zeros(E)
    for r, d in toks:
        ids, sc = predict_scores(r, d, l, h, pp, M)
        np.add.at(sv, ids, sc)
    if pp["gamma"] is not None:
        acc = eng.acc.get(T)
        if acc is None:
            acc = eng.acc[T] = [np.zeros(E), 0.0]
        w = pp["gamma"] ** (h - 1)
        acc[0] += w * sv
        acc[1] += w
        conf = acc[0] / acc[1]
    else:
        conf = sv
    sel = np.flatnonzero(conf >= tau)
    sel = sel[np.argsort(-conf[sel], kind="stable")]
    if h <= pp["freeze"]:
        old = eng.tq.get(T)
        if old is not None:
            th = pp["tau_high"] if pp["tau_high"] is not None else tau
            oldl = old - (l + h) * E
            add = sel[conf[sel] >= th]
            sel = np.concatenate([oldl, add[~np.isin(add, oldl)]])
    eng.lookahead((l + h) * E + sel, t_gate, T, h)


class Subst:
    """Use-time substitution (plan 2.4 / 1.3): at layer l an expert of a token's top-k that is late -- not cached, or
    still being loaded and not ready by the time the layer needs it (+ slack us) -- is replaced, least routing
    weight first and within the budget, by the cached-and-ready expert with the highest router score that the token
    does not already use (same number of experts computed). A queued load of the replaced expert is cancelled
    (one in flight continues). Spec: none | tok:<b> (<= b per token and layer) | rate:<rho> (running share of changed
    uses <= rho) | tok:<b>:rate:<rho>; prefix skip: drops late experts instead (the Q1 approximation); :slack<us>."""

    def __init__(self, spec, L, E, log=False):
        parts = spec.split(":")
        self.skip = parts[0] == "skip"
        if self.skip:
            parts = parts[1:]
        self.b, self.rate, self.slack = None, None, 0.0
        i = 0
        while i < len(parts):
            if parts[i] == "tok":
                self.b, i = int(parts[i + 1]), i + 2
            elif parts[i] == "rate":
                self.rate, i = float(parts[i + 1]), i + 2
            elif parts[i].startswith("slack"):
                self.slack, i = float(parts[i][5:]), i + 1
            else:
                raise ValueError(f"subst spec {spec}")
        self.E = E
        self.rng = np.random.default_rng(0)          # rate budget: random rounding, see apply()
        self.uses = self.changed = self.late = 0
        self.layer_uses, self.layer_changed = np.zeros(L, np.int64), np.zeros(L, np.int64)
        self.records = [] if log else None

    def apply(self, eng, r, d, l, keys, t_gate, rid=0):
        E, k = self.E, len(keys)
        self.uses += k
        self.layer_uses[l] += k
        late = ~eng.res[keys] | (eng.ready[keys] > t_gate + self.slack)
        if not late.any():
            return keys
        idx = np.flatnonzero(late)
        self.late += len(idx)
        idx = idx[np.argsort(r["w"][l, d][idx], kind="stable")]       # least important first
        n = len(idx)
        if self.b is not None:
            n = min(n, self.b)
        if self.rate is not None:
            # running budget rate x uses, rounded with a random offset: plain int() rounding hands the budget to the
            # same layers of every token (45 layers x 8 uses x 0.05 = exactly 18 per token)
            n = min(n, max(0, int(self.rate * self.uses + self.rng.random()) - self.changed))
        if n <= 0:
            return keys
        lay = l * E
        if self.skip:
            drop = idx[:n]
            self.changed += n
            self.layer_changed[l] += n
            for j in drop:
                if eng.queued[keys[j]] >= 1:
                    eng.queued[keys[j]] = -1
                    eng.want[keys[j]] = -1
            return np.delete(keys, drop)
        ok = eng.res[lay:lay + E] & (eng.ready[lay:lay + E] <= t_gate + self.slack)
        ok[keys - lay] = False
        cand = [int(c) for c in np.argsort(-r["score"][l, d].astype(np.float32), kind="stable") if ok[c]]
        out = keys.copy()
        for j, s in zip(idx[:n], cand):
            e = int(keys[j])
            out[j] = lay + s
            if eng.queued[e] >= 1:
                eng.queued[e] = -1
                eng.want[e] = -1
            self.changed += 1
            self.layer_changed[l] += 1
            if self.records is not None:
                self.records.append((rid, d, l, e - lay, s))
        return out

    def summary(self):
        return dict(subst_change_rate=self.changed / max(self.uses, 1), subst_late_rate=self.late / max(self.uses, 1),
                    subst_layer_rate=(self.layer_changed / np.maximum(self.layer_uses, 1)).round(5).tolist())


# ---------------------------------------------------------------- one synchronous decode step
def step(eng, toks, t, pos, pp, M, hw, E, sub_n=0, acc=None, subst=None):
    """All tokens in `toks` (list of (request, token index)) go through every layer together. Returns the new time,
    the stall (time waiting for experts) and the GPU busy time."""
    L = eng.L
    B = len(toks)
    stall = busy = 0.0
    a = attn_cost(hw, B) + hw["tick"]
    cap = (a + gmm_cost(hw, eng.k)) / hw["load"] if pp.get("adv") and hw["load"] else 1.0   # experts loadable / layer
    cnt = "loaded" if hw.get("load_cpu_all") else "demand"       # which reads pay the host work (load_cpu)
    for l in range(L):
        eng.now = pos + l
        d0 = eng.stats[cnt]                                   # every read during this layer (prefetches included)
        t_gate = t + a
        busy += a
        if pp.get("adv"):
            load = float((eng.queued == 1).sum()) / cap
            eng.load_hist[np.searchsorted(LOAD_BINS, load, side="right") - 1] += 1
            tau_l = (pp["thr"] or 0.0) + pp["kappa"] * load + pp["kappa_log"] * np.log2(1 + load)
        for h in range(1, pp["H"] + 1):
            if l + h >= L:
                break
            if pp.get("adv"):
                prefetch_adv(eng, toks, l, h, t_gate, pos, pp, M, E, tau_l + (pp["hl"] if h == 1 else 0.0))
                continue
            if B == 1:
                ids = predict(toks[0][0], toks[0][1], l, h, pp, M)
            else:
                ids = np.unique(np.concatenate([predict(r, d, l, h, pp, M) for r, d in toks]))
            eng.lookahead((l + h) * E + ids.astype(np.int64), t_gate, pos + l + h, h)
        if subst is not None:
            eng.advance(t_gate)                               # loader state at the time the layer needs its experts
            final = [subst.apply(eng, r, d, l, l * E + r["dec"][l, d], t_gate, r.get("rid", 0)) for r, d in toks]
            keys = final[0] if B == 1 else np.unique(np.concatenate(final))
        elif B == 1 and not sub_n:
            r, d = toks[0]
            keys = l * E + r["dec"][l, d]
        else:
            users = {}                                        # expert -> routing weights of the batch's uses
            for r, d in toks:
                w = r["w"][l, d] if sub_n else None
                for j, e in enumerate(r["dec"][l, d]):
                    users.setdefault(int(e), []).append(1.0 if w is None else float(w[j]))
            if acc is not None:
                acc["uses"] += sum(len(v) for v in users.values())
            if sub_n:                                         # missing and rarely needed: skipped (lossy)
                for e in list(users):
                    if not eng.res[l * E + e] and len(users[e]) < sub_n:
                        acc["drop_uses"] += len(users[e])
                        acc["drop_w"] += sum(users[e])
                        del users[e]
            keys = np.array([l * E + e for e in users], np.int64)
        if len(keys):
            t_ready = eng.need(keys, t_gate)
        else:
            t_ready = t_gate
        stall += t_ready - t_gate
        t_ready += hw.get("load_cpu", 0.0) * (eng.stats[cnt] - d0)        # host work per read (serial)
        g = gmm_cost(hw, len(keys))
        t = t_ready + g
        busy += g
        eng.ruse[keys] += 1
        eng.freq[keys] += 1
    for l in range(L):
        eng.tq.pop(pos + l, None)
        eng.acc.pop(pos + l, None)
    return t, stall, busy


# ---------------------------------------------------------------- data
def load_reqs(trace, n=None, min_dec=0, preds=None, names=(), need_w=False, max_reqs=None, need_score=False, skip=0):
    """Trace requests (files skip .. skip+n, or with min_dec the first max_reqs whose decode has >= min_dec tokens)."""
    files = sorted(glob.glob(trace + "/q*.npz"))[skip:][:n]
    pfiles = {os.path.basename(f): f for f in glob.glob(preds + "/q*.npz")} if preds else {}
    reqs = []
    for f in files:
        z = np.load(f)
        if z["dec_idx"].shape[1] < min_dec:
            continue
        r = dict(stem=os.path.basename(f), pre=z["pre_idx"].astype(np.int64), dec=z["dec_idx"].astype(np.int64))
        r["D"] = r["dec"].shape[1]
        r["la_dec"] = z["la_dec_idx"].astype(np.int64) if z["la_dec_idx"].size else None
        if "dec_gate_h" in z.files:
            r["gate_h"] = z["dec_gate_h"].astype(np.int64)
        if (need_w or need_score) and r["D"] and "dec_wt" in z.files:     # the routing weights (Mac prototype)
            r["w"] = z["dec_wt"].astype(np.float32)
        elif (need_w or need_score) and r["D"]:               # GLM: weight = score / sum of the top-k scores
            sc = np.take_along_axis(z["dec_score"].astype(np.float32), r["dec"], -1)
            r["w"] = sc / sc.sum(-1, keepdims=True)
        if need_score and r["D"]:
            r["score"] = z["dec_score"]                       # [L, D, E] router scores (substitution candidates)
        if pfiles and names and r["stem"] in pfiles:
            pz = np.load(pfiles[r["stem"]])
            r["P"] = {nm: pz[nm] for nm in names if nm in pz.files}
            assert r["P"][next(iter(r["P"]))].shape[1] == r["D"], "preds / trace mismatch"
        reqs.append(r)
        if max_reqs and len(reqs) >= max_reqs:
            break
    return reqs


def use_counts(reqs, L, E):
    """Token-level (layer, expert) use counts over the whole decode of these requests (tsim_async.warm)."""
    f = np.zeros(L * E)
    for r in reqs:
        for l in range(L):
            f[l * E:(l + 1) * E] += np.bincount(r["dec"][l].ravel(), minlength=E)
    return f


def layer_caps(f, C, L, E, floor=0):
    """Per-layer capacities of a cache of C experts: each layer's `floor` most used experts under counts f, then the
    globally most used of the rest (the Mac prototype's per-layer pools, sized from the calibration counts)."""
    f = np.asarray(f, np.float64).reshape(L, E)
    assert L * floor <= C
    forced = np.zeros((L, E), bool)
    forced[np.arange(L)[:, None], np.argsort(-f, axis=1, kind="stable")[:, :floor]] = True
    rest = np.flatnonzero(~forced.ravel())
    pick = rest[np.argsort(-f.ravel()[rest], kind="stable")[:C - L * floor]]
    m = forced.ravel().copy()
    m[pick] = True
    return m.reshape(L, E).sum(1)


def top_mask(score, nr):
    m = np.zeros(len(score), bool)
    m[np.argsort(-score, kind="stable")[:nr]] = True
    return m


def resident_mask(kind, rho, C, L, E, fd):
    """Initial resident region (rho x C keys). fd holds the calibration statistics (never the test requests):
      glob   most used (layer, expert) pairs of a generic workload (fd["glob"], Wtrain);
      layer  the same counts, an equal quota in every layer;
      work   most used pairs of other requests of the same workload type (fd["work"]);
      ema:<period>:<half-life>:<swap>   starts as glob, then follows an online moving average (EmaFill);
      pred:<k>   per request, the decode use of its k nearest calibration requests by prefill experts (PredFill)."""
    nr = int(round(rho * C))
    if nr <= 0:
        return np.zeros(L * E, bool)
    name = kind.split(":")[0]
    if name in ("glob", "ema", "pred"):
        return top_mask(fd["glob"], nr)
    if name == "work":
        return top_mask(fd["work"], nr)
    if name == "layer":
        m = np.zeros(L * E, bool)
        q, extra = divmod(nr, L)
        for l in range(L):
            c = fd["glob"][l * E:(l + 1) * E]
            m[l * E + np.argsort(-c, kind="stable")[:q + (1 if l < extra else 0)]] = True
        return m
    raise ValueError(f"fill {kind}")


class EmaFill:
    """R-ema: exponential moving average of decode uses (half-life in tokens); every `period` decode tokens the
    resident region swaps at most swap x its size, entering experts whose average beats the leaving ones."""

    def __init__(self, spec, nr, init_counts, L, k):
        _, period, half, swap = spec.split(":")
        self.period, self.swap, self.nr = int(period), float(swap), nr
        self.decay = 0.5 ** (1.0 / float(half))
        self.e = init_counts / max(init_counts.sum(), 1) * L * k / (1 - self.decay)   # steady-state scale
        self.n = 0

    def step(self, eng, keys, t):
        self.e *= self.decay
        self.e[keys] += 1
        self.n += 1
        if self.n % self.period:
            return
        res = np.flatnonzero(eng.rmask)
        out = res[np.argsort(self.e[res], kind="stable")]
        cand = np.flatnonzero(~eng.rmask)
        inn = cand[np.argsort(-self.e[cand], kind="stable")]
        m = int(np.sum(self.e[inn[:len(out)]] > self.e[out[:len(inn)]]))      # pairs worth swapping
        m = min(m, int(self.swap * self.nr))
        if m:
            new = eng.rmask.copy()
            new[out[:m]] = False
            new[inn[:m]] = True
            eng.set_resident(new, t)


def pred_masks(test, calib, nr, L, E, kq):
    """R-pred: per test request, the k nearest calibration requests by cosine of per-layer-normalised prefill
    expert histograms; predicted decode use = their similarity-weighted decode distribution; top nr keys."""
    def hist(idx):
        h = np.stack([np.bincount(idx[l].ravel(), minlength=E) for l in range(L)]).astype(np.float64)
        h /= np.maximum(h.sum(1, keepdims=True), 1)
        return h.ravel()
    Hc = np.stack([hist(c["pre"]) for c in calib])
    Hc /= np.linalg.norm(Hc, axis=1, keepdims=True) + 1e-12
    Dc = np.stack([hist(c["dec"]) for c in calib])
    out = {}
    for r in test:
        h = hist(r["pre"])
        sim = Hc @ (h / (np.linalg.norm(h) + 1e-12))
        nb = np.argsort(-sim)[:kq]
        w = np.maximum(sim[nb], 1e-6)
        out[r["stem"]] = top_mask((w[:, None] * Dc[nb]).sum(0) / w.sum(), nr)
    return out


# ---------------------------------------------------------------- runners
def run_seq(reqs, policy, hw, C, E, rmask=None, protect=False, W=0.0, P_pre=2, ema=None, pmasks=None, subst=None,
            subst_log=None, warm_f=None, caps=None):
    """Single user: requests back to back, prefill (P_pre-layer prefetch window) then decode (tsim_ahead.run).
    ema: an EmaFill refreshing the resident region during decode; pmasks: per-request resident regions (R-pred),
    installed when the request arrives."""
    evict, pref = policy.split("/")
    pp = parse_pref(pref)
    L, k = reqs[0]["pre"].shape[0], reqs[0]["pre"].shape[-1]
    M = pp["M"] or k
    eng = Engine(L, E, C, hw, evict, rmask, protect, cap_l=caps)
    eng.k = k
    if warm_f is not None:                           # default: cold start (P3)
        eng.warm(warm_f)
    t, pos = 0.0, 0.0
    out = dict(stall_dec=[], tpot=[], ttft=[], toks=0, dec_loads=0, dec_wasted=0)
    lay = np.arange(L)[:, None] * E
    sb = Subst(subst, L, E, log=subst_log is not None) if subst and subst != "none" else None
    for ri, r in enumerate(reqs):
        r["rid"] = ri
        eng.ruse[:] = 0
        eng.clear_background()
        if pmasks is not None:
            eng.set_resident(pmasks[r["stem"]], t)
        t_arr = t
        t += W
        eng.advance(t)
        eng.phase = "pre"
        pre_idx = r["pre"]
        T = pre_idx.shape[1]
        for l in range(L):
            for ll in range(l + 1, min(L, l + 1 + P_pre)):
                eng.issue(ll * E + np.unique(pre_idx[ll]), 1, t, deadline=pos + ll, h=ll - l)
            eng.now = pos + l
            keys = l * E + np.unique(pre_idx[l])
            t = eng.need(keys, t) + T * hw["pre_tok"]
            eng.ruse[keys] += 1
            eng.freq[keys] += 1
        pos += L
        eng.tq.clear()
        out["ttft"].append(t - t_arr)
        eng.phase = "dec"
        D = r["D"]
        t_dec0, stall = t, 0.0
        l0, w0 = eng.stats["loaded"], eng.stats["wasted"]
        for s in range(D):
            t, st, _ = step(eng, [(r, s)], t, pos, pp, M, hw, E, subst=sb)
            stall += st
            pos += L
            if ema is not None:
                ema.step(eng, (lay + r["dec"][:, s]).ravel(), t)
        out["dec_loads"] += eng.stats["loaded"] - l0
        out["dec_wasted"] += eng.stats["wasted"] - w0
        if D:
            out["tpot"].append((t - t_dec0) / D)
            out["toks"] += D
        out["stall_dec"].append(stall)
    toks = max(out["toks"], 1)
    res = dict(policy=policy, C=C, ttft_ms=1e-3 * float(np.mean(out["ttft"])), tpot_ms=1e-3 * float(np.mean(out["tpot"])),
               stall_ms_per_tok=1e-3 * float(np.sum(out["stall_dec"])) / toks,
               loads_per_tok=out["dec_loads"] / toks, wasted_frac=out["dec_wasted"] / max(1, out["dec_loads"]),
               toks=out["toks"], loads=dict(eng.stats), res_loads_per_tok=eng.stats["res_loads"] / toks)
    res.update(eng.waste_summary())
    if sb is not None:
        res.update(sb.summary())
        if subst_log is not None:
            os.makedirs(os.path.dirname(subst_log) or ".", exist_ok=True)
            np.savez_compressed(subst_log, records=np.array(sb.records, np.int32).reshape(-1, 5),
                                stems=np.array([r["stem"] for r in reqs]),
                                columns=np.array(["request", "token", "layer", "expert", "substitute"]))
            res["subst_log"] = subst_log
    return res


def select_batch(N, B, wait, select, klim, route_of, L, E, var_b=0.0):
    """tsim_async.run_sync's selection: fcfs = longest waiting; prev / oracle / anchor / mix = greedy fewest new
    (layer, expert) pairs under that prediction, streams forced in once they waited ceil(N/B)-1+klim steps.
    var_b > 0: variable B -- stop adding (beyond the forced ones and at least B/2) once the best candidate would add
    more than var_b new experts per layer."""
    if B >= N:
        return list(range(N))
    if select == "fcfs":
        return [int(i) for i in np.lexsort((np.arange(N), -wait))[:B]]
    lim = int(np.ceil(N / B)) - 1 + klim
    act = [int(i) for i in np.argsort(-wait) if wait[i] >= lim][:B]
    U = np.zeros((L, E), bool)
    for i in act:
        U[np.arange(L)[:, None], route_of(i)] = True
    cand = [i for i in range(N) if i not in set(act)]
    while len(act) < B and cand:
        best, bi, bm = None, None, None
        for i in cand:
            m = np.zeros((L, E), bool)
            m[np.arange(L)[:, None], route_of(i)] = True
            inc = (m & ~U).sum() - 1e-6 * wait[i]
            if best is None or inc < best:
                best, bi, bm = inc, i, m
        if var_b > 0 and len(act) >= max(1, B // 2) and best > var_b * L:
            break
        act.append(bi)
        U |= bm
        cand.remove(bi)
    return act


def run_pool(reqs, offs, policy, hw, C, E, horizon, rmask=None, protect=False, B=None, select="fcfs", klim=10 ** 9,
             sub_n=0, warm=True, subst=None, var_b=0.0, ema=None, warm_f=None, caps=None):
    """N decode streams in a closed loop, B advance per step (tsim_async.run_sync without its la1 variant).
    The flowing region starts filled by use counts warm_f (default: the streams' own whole traces, as Q1)."""
    evict, pref = policy.split("/")
    pp = parse_pref(pref)
    L, k = reqs[0]["dec"].shape[0], reqs[0]["dec"].shape[-1]
    M = pp["M"] or k
    N = len(reqs)
    B = B or N
    eng = Engine(L, E, C, hw, evict, rmask, protect, cap_l=caps)
    eng.k = k
    if warm:
        eng.warm(use_counts(reqs, L, E) if warm_f is None else warm_f)
    t, pos_abs = 0.0, 0.0
    pos = list(offs)
    wait = np.zeros(N, int)
    emit = [[] for _ in range(N)]
    gpu_busy = stall = 0.0
    acc = dict(uses=0, drop_uses=0, drop_w=0.0)
    sb = Subst(subst, L, E) if subst and subst != "none" else None
    for ri, r in enumerate(reqs):
        r["rid"] = ri
    nsteps = 0
    union = []
    route_of = lambda i, dd=0: reqs[i]["dec"][:, (pos[i] + dd) % reqs[i]["D"]]      # [L, k]
    while t < horizon:
        if select == "oracle":
            pred = lambda i: route_of(i, 0)
        elif select in ("anchor", "mix"):                     # next_token_pred.py: prediction for token pos from pos-1
            pred = lambda i: reqs[i]["next_" + select][:, pos[i] % reqs[i]["D"]]
        else:
            pred = lambda i: route_of(i, -1)
        act = select_batch(N, B, wait, select, klim, pred, L, E, var_b)
        toks = [(reqs[i], pos[i] % reqs[i]["D"]) for i in act]
        u0 = acc["uses"]
        t, st, busy = step(eng, toks, t, pos_abs, pp, M, hw, E, sub_n, acc, sb)
        if ema is not None:                                   # one EMA step per decode step, all the batch's uses
            ema.step(eng, np.unique(np.concatenate([(np.arange(L)[:, None] * E + r["dec"][:, d]).ravel()
                                                    for r, d in toks])), t)
        union.append(len(toks) and sum(len(np.unique(np.concatenate([r["dec"][l, d] for r, d in toks])))
                                       for l in range(L)) / L)
        stall += st
        gpu_busy += busy
        pos_abs += L
        nsteps += 1
        sel = np.zeros(N, bool)
        sel[act] = True
        wait[~sel] += 1
        wait[sel] = 0
        for i in act:
            emit[i].append(t)
            pos[i] += 1
    toks = sum(len(e) for e in emit)
    gaps = np.concatenate([np.diff(e) for e in emit if len(e) > 1]) if toks > len(emit) else np.zeros(1)
    st = eng.stats
    res = dict(policy=policy, C=C, tokens=toks, steps=nsteps, throughput_tok_s=toks / (t / 1e6),
               tpot_p50_ms=float(np.percentile(gaps, 50)) / 1e3, tpot_p99_ms=float(np.percentile(gaps, 99)) / 1e3,
               loads_per_token=st["loaded"] / max(toks, 1), stall_ms_per_step=1e-3 * stall / max(nsteps, 1),
               gpu_busy=gpu_busy / t, loader_busy=min(1.0, st["loaded"] * (hw["load"] or 0) / t),
               distinct_per_layer=float(np.mean(union)), wasted_frac=st["wasted"] / max(1, st["loaded"]),
               dropped_use_frac=acc["drop_uses"] / max(acc["uses"], 1), loads=dict(st))
    res.update(eng.waste_summary())
    if sb is not None:
        res.update(sb.summary())
    return res


# ---------------------------------------------------------------- CLI
_DATA = {}


def _job(job):
    mode, pol, hwname, frac, rho, fill, protect, N, B, select, klim, sub_n, subst, var_b = job
    hw = HW[hwname]
    reqs = _DATA["reqs"]
    a = _DATA["args"]
    L, k = reqs[0]["dec"].shape[0], reqs[0]["dec"].shape[-1]
    C = L * a.E if hw["load"] is None else int(frac * L * a.E)
    nr = int(round(rho * C))
    fd = _DATA["fill"]
    rmask = resident_mask(fill, rho, C, L, a.E, fd) if rho > 0 else None
    ema = EmaFill(fill, nr, fd["glob"], L, k) if rho > 0 and fill.startswith("ema") else None
    caps = layer_caps(fd["glob"], C, L, a.E, a.layer_caps) if a.layer_caps >= 0 and hw["load"] is not None else None
    pm = (pred_masks(reqs, _DATA["work_reqs"], nr, L, a.E, int(fill.split(":")[1]))
          if rho > 0 and fill.startswith("pred") else None)
    if mode == "seq":
        log = None
        if a.subst_log and subst != "none":
            tag = "_".join(x.replace("/", "-").replace(":", "") for x in (hwname, f"f{frac:g}", pol, subst))
            log = os.path.join(a.subst_log, tag + ".npz")
        res = run_seq(reqs, pol, hw, C, a.E, rmask, protect, a.W, a.P, ema, pm, subst, log,
                      warm_f=fd["glob"] if a.warm == "glob" else None, caps=caps)
    else:
        if pm is not None:
            raise ValueError("pred (per-request) resident fill is single-user (seq) only")
        streams = reqs[:N]
        offs = _DATA["offs"][:N]
        res = run_pool(streams, offs, pol, hw, C, a.E, a.horizon_ms * 1e3, rmask, protect, B, select, klim, sub_n,
                       warm=a.warm != "none", subst=subst, var_b=var_b, ema=ema,
                       warm_f=fd["glob"] if a.warm == "glob" else None, caps=caps)
    res.update(mode=mode, hw=hwname, cache_frac=frac, rho=rho, fill=fill, protect=protect,
               n=len(reqs) if mode == "seq" else N, B=B, select=select, klim=klim, sub_n=sub_n, subst=subst, var_b=var_b,
               warm=a.warm, layer_caps=a.layer_caps)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["seq", "pool"], required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--preds", default=None, help="anchor_dump.py output dir (per-request npz)")
    ap.add_argument("--n", type=int, default=None, help="seq: first n requests")
    ap.add_argument("--skip", type=int, default=0, help="seq: skip the first requests (e.g. 50: the CoT calibration split)")
    ap.add_argument("--hw", nargs="+", default=["p3_glm4bit"])
    ap.add_argument("--cache", type=float, nargs="+", default=[0.5], help="cache size f (fraction of all experts)")
    ap.add_argument("--rho", type=float, nargs="+", default=[0.0], help="resident share of the cache")
    ap.add_argument("--fill", nargs="+", default=["glob"],
                    help="resident region: glob | layer | work | ema:<period>:<half-life>:<swap> | pred:<k>")
    ap.add_argument("--glob_trace", "--fill_trace", default=None,
                    help="generic calibration workload for glob / layer / ema (default: --trace itself, i.e. leaky)")
    ap.add_argument("--work_trace", default=None, help="same-type calibration workload for work / pred")
    ap.add_argument("--work_skip", type=int, default=0, help="skip the first n requests of --work_trace (the test ones)")
    ap.add_argument("--protect_resident", type=int, nargs="+", default=[0], help="C2 protection off / on")
    ap.add_argument("--policy", required=True, help="comma-separated <evict>/<prefetch>")
    ap.add_argument("--N", type=int, nargs="+", default=[64], help="pool: number of streams")
    ap.add_argument("--B", type=int, nargs="+", default=[0], help="pool: streams per step (0 = all)")
    ap.add_argument("--select", nargs="+", default=["fcfs"], help="pool: fcfs | prev | oracle")
    ap.add_argument("--klim", type=int, nargs="+", default=[10 ** 9], help="pool: max extra wait (steps)")
    ap.add_argument("--sub_n", type=int, nargs="+", default=[0], help="pool: skip missing experts used by < n")
    ap.add_argument("--min_dec", type=int, default=64, help="pool: streams need this many decode tokens")
    ap.add_argument("--warm", choices=["self", "glob", "none"], default=None,
                    help="initial cache: the streams' own traces (pool default: Q1, leaky) | --glob_trace | empty "
                         "(seq default: cold, as P3; seq takes none or glob)")
    ap.add_argument("--layer_caps", type=int, default=-1,
                    help=">= 0: per-layer capacities from the --glob_trace counts with this floor per layer (the Mac "
                         "prototype's pools); eviction within the layer. -1: one cache for all layers")
    ap.add_argument("--subst", nargs="+", default=["none"],
                    help="use-time substitution: none | tok:<b> | rate:<rho> | tok:<b>:rate:<rho> | skip:... [:slack<us>]")
    ap.add_argument("--subst_log", default=None, help="seq: write each run's substitution records here (npz)")
    ap.add_argument("--next_pred", default=None, help="pool: next_token_pred.py output (select anchor / mix)")
    ap.add_argument("--var_b", type=float, nargs="+", default=[0.0], help="pool: variable-B threshold (new experts / layer)")
    ap.add_argument("--horizon_ms", type=float, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--E", type=int, default=128)
    ap.add_argument("--W", type=float, default=0.0, help="seq: idle time before each prefill (us)")
    ap.add_argument("--P", type=int, default=2, help="seq: prefill prefetch window (layers)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pols = a.policy.split(",")
    names = set().union(*[pred_names(parse_pref(p.split("/")[1])) for p in pols])
    need_score = any(x != "none" for x in a.subst)
    if a.mode == "seq":
        reqs = load_reqs(a.trace, a.n, preds=a.preds, names=names, need_score=need_score, skip=a.skip)
        offs = None
    else:
        reqs = load_reqs(a.trace, None, a.min_dec, a.preds, names, need_w=max(a.sub_n) > 0, max_reqs=max(a.N),
                         need_score=need_score)
        rng = np.random.default_rng(a.seed)
        offs = [int(rng.integers(0, r["D"])) for r in reqs]
        if a.next_pred:
            for r in reqs:
                p = np.load(os.path.join(a.next_pred, r["stem"]))
                for nm in ("anchor", "mix"):
                    r["next_" + nm] = p["next_" + nm].transpose(1, 0, 2).astype(np.int64)    # [L, D, k]
    L = reqs[0]["dec"].shape[0]
    if a.glob_trace is None and max(a.rho) > 0:
        print("WARNING: resident region filled from the test requests themselves (no --glob_trace)", flush=True)
    cal = load_reqs(a.glob_trace) if a.glob_trace else reqs
    if a.glob_trace and os.path.realpath(a.glob_trace) == os.path.realpath(a.trace):
        used = {r["stem"] for r in reqs}                      # same workload: only the requests that are not tested
        cal = [r for r in cal if r["stem"] not in used]
        assert cal, "--glob_trace = --trace leaves no calibration requests"
        print(f"calibration: the {len(cal)} requests of {a.trace} that are not tested", flush=True)
    if a.warm is None:
        a.warm = "self" if a.mode == "pool" else "none"
    if a.mode == "seq" and a.warm == "self":
        raise ValueError("seq mode: --warm none or glob")
    if (a.warm == "glob" or a.layer_caps >= 0) and not a.glob_trace:
        raise ValueError("--warm glob / --layer_caps need --glob_trace")
    work = load_reqs(a.work_trace)[a.work_skip:] if a.work_trace else []
    if work and a.mode == "seq":
        overlap = {r["stem"] for r in work} & {r["stem"] for r in reqs} if a.work_trace.rstrip("/").split("/")[-1] == \
            a.trace.rstrip("/").split("/")[-1] else set()
        assert not overlap, f"calibration and test requests overlap: {sorted(overlap)[:5]}"
    fd = dict(glob=use_counts(cal, L, a.E), work=use_counts(work, L, a.E) if work else None)
    _DATA.update(reqs=reqs, offs=offs, args=a, fill=fd, work_reqs=work)
    jobs = list(itertools.product([a.mode], pols, a.hw, a.cache, a.rho, a.fill, [bool(x) for x in a.protect_resident],
                                  a.N if a.mode == "pool" else [1], a.B if a.mode == "pool" else [1],
                                  a.select if a.mode == "pool" else ["-"], a.klim if a.mode == "pool" else [0],
                                  a.sub_n if a.mode == "pool" else [0], a.subst, a.var_b if a.mode == "pool" else [0.0]))
    jobs = [j for j in jobs if j[4] > 0 or j[5] == a.fill[0]]          # rho = 0: the fill does not matter
    print(f"{a.trace}: {len(reqs)} requests, L={L}; {len(jobs)} jobs", flush=True)
    with ProcessPoolExecutor(min(a.workers, len(jobs)), mp_context=mp.get_context("fork")) as ex:
        rows = list(ex.map(_job, jobs))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=1)
    for r in rows:
        tag = (f"{r['hw']:13} f={r['cache_frac']:.2f} rho={r['rho']:.2f}{'/' + r['fill'] if r['rho'] else ''}"
               f"{' C2' if r['protect'] else ''} {r['policy']:26}")
        if r["subst"] != "none":
            tag += f" sub={r['subst']} (changed {r['subst_change_rate']:.1%})"
        if r["mode"] == "seq":
            print(f"{tag} stall {r['stall_ms_per_tok']:7.2f} ms/tok  TPOT {r['tpot_ms']:7.2f} ms  loads/tok "
                  f"{r['loads_per_tok']:5.1f}  wasted {r['wasted_frac']:.0%} (pf waste {r['pf_waste_frac']:.0%})", flush=True)
        else:
            print(f"{tag} N={r['n']} B={r['B'] or r['n']} {r['select']}{' vb' + str(r['var_b']) if r['var_b'] else ''} k{r['klim']}  {r['throughput_tok_s']:7.1f} tok/s  "
                  f"TPOT p50 {r['tpot_p50_ms']:7.1f} p99 {r['tpot_p99_ms']:7.1f} ms  loads/tok {r['loads_per_token']:6.2f}"
                  f"  GPU {r['gpu_busy']:.0%} loader {r['loader_busy']:.0%}  ({r['tokens']} tok)", flush=True)


if __name__ == "__main__":
    main()
