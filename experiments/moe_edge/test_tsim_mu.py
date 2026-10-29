"""Unit tests for tsim_mu.py: cache eviction (resident region, lru / lfu), protection (P3 and C2), cancellation,
waste breakdown, demand-before-prefetch, overflow; and exact agreement with tsim_ahead.run (single user) and
tsim_async.run_sync (multi user) on synthetic traces.

  .venv-mac/bin/python -m pytest experiments/moe_edge/test_tsim_mu.py -q      (or run this file directly)
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tsim_mu as mu  # noqa: E402

HW1 = dict(attn=[(1, 0.0)], tick=0.0, g0=1.0, g1=1.0, load=10.0, pre_tok=0.0)


def eng_(L=2, E=4, C=3, evict="lru", rmask=None, protect=False):
    return mu.Engine(L, E, C, HW1, evict, rmask, protect)


def test_lru_evicts_least_recent_flowing_never_resident():
    rm = np.zeros(8, bool)
    rm[0] = True                                     # key 0 in the resident region
    e = eng_(C=3, rmask=rm)                          # flowing capacity 2
    t = e.need([1], 0.0) + 1
    t = e.need([2], t) + 1
    assert e.res[[0, 1, 2]].all() and e.n_flow == 2
    t = e.need([1], t) + 1                           # 1 more recent than 2
    t = e.need([3], t) + 1                           # evicts 2 (lru), never 0
    assert e.res[[0, 1, 3]].all() and not e.res[2]
    t = e.need([5], t) + 1                           # evicts 1
    assert e.res[[0, 3, 5]].all() and not e.res[1]


def test_lfu_tie_breaks_on_smallest_key():
    e = eng_(C=2, evict="lfu")
    t = e.need([3], 0.0)
    t = e.need([2], t)
    e.freq[[2, 3]] = 5
    t = e.need([1], t)                               # tie: evict key 2 (smallest)
    assert e.res[3] and e.res[1] and not e.res[2]


def test_demand_goes_before_prefetch():
    e = eng_(L=4, E=4, C=8)
    e.now = 0
    e.lookahead([3 * 4 + 1, 3 * 4 + 2], 0.0, 3, 3)   # two prefetches queued at t=0
    done = e.need([0 * 4 + 3], 0.0)                   # demand at t=0 is loaded first
    assert done == 10.0
    assert e.stats["demand"] == 1 and e.stats["la"] == 0


def test_prefetch_protection_and_drop():
    e = eng_(L=4, E=4, C=2)
    e.now = 0
    e.lookahead([1 * 4 + 1], 0.0, 1, 1)               # needed at layer 1
    e.lookahead([2 * 4 + 1], 0.0, 2, 2)               # needed at layer 2
    e.advance(25.0)
    assert e.res[5] and e.res[9]
    e.lookahead([3 * 4 + 1], 25.0, 3, 3)              # only protected victims needed earlier: dropped
    e.advance(40.0)
    assert not e.res[13] and e.stats["la_dropped"] == 1
    e.now = 1
    e.need([5], 40.0)                                 # used: no longer protected
    e.now = 2
    e.need([0 * 4 + 2], 41.0)                         # demand evicts 5 (unprotected), not 9 (want=2)
    assert e.res[9] and not e.res[5]


def test_c2_protects_cached_predicted_experts():
    for protect in (False, True):
        e = eng_(L=4, E=4, C=2, protect=protect)
        t = e.need([2 * 4 + 0], 0.0) + 1              # key 8, older
        t = e.need([2 * 4 + 1], t) + 1                # key 9
        e.now = 0
        e.lookahead([8], t, 2, 2)                     # 8 predicted for layer 2, already cached
        e.need([0 * 4 + 3], t)                        # demand needs a victim: lru would take 8
        assert e.res[8] == protect and e.res[9] != protect


def test_cancel_and_waste_breakdown():
    e = eng_(L=4, E=4, C=8)
    e.now = 0
    e.lookahead([3 * 4 + 1, 3 * 4 + 2], 0.0, 3, 3)    # h=3 prediction for layer 3
    e.advance(10.0)                                   # 13 loaded, 14 still queued
    assert e.res[13] and not e.res[14]
    e.now = 1
    e.lookahead([3 * 4 + 3], 10.0, 3, 2)              # newer prediction drops both: 14 cancelled, 13 stale
    assert e.stats["la_cancelled"] == 1 and e.stale[13]
    e.advance(30.0)
    e.now = 3
    e.need([3 * 4 + 3], 30.0)                         # 15 used at its target: hit (h=2)
    ws = e.waste_summary()
    assert ws["pf_hit"] == 1 and ws["pf_by_h"]["hit"][2] == 1
    assert ws["pf_end"] == 1 and ws["pf_by_h"]["end"][3] == 1  # 13 never used
    e.C_flow = 1                                      # force evictions: 13 goes first (lru)
    e.now = 4
    e.need([0 * 4 + 0], 40.0)
    ws = e.waste_summary()
    assert ws["pf_corrected"] == 1


def test_final_wrong_and_early():
    e = eng_(L=4, E=4, C=8)
    e.now = 0
    e.lookahead([2 * 4 + 1], 0.0, 2, 2)
    e.advance(20.0)
    e.now = 2
    e.need([2 * 4 + 0], 20.0)                         # target ran without 9
    e.C_flow = 1
    e.now = 3
    e.need([3 * 4 + 0], 40.0)
    assert e.waste_summary()["pf_final_wrong"] == 1
    e2 = eng_(L=4, E=4, C=8)
    e2.now = 0
    e2.lookahead([3 * 4 + 1], 0.0, 3, 3)
    e2.advance(20.0)
    e2.C_flow = 1
    e2.now = 1
    e2.need([1 * 4 + 0], 20.0)                        # demand at layer 1 must evict 13 before layer 3 runs
    assert e2.waste_summary()["pf_early"] == 1


def test_overflow_when_everything_is_pinned():
    e = eng_(L=2, E=4, C=2)
    done = e.need([0, 1, 2], 0.0)                     # 3 distinct experts, capacity 2
    assert done == 30.0 and e.stats["overflow"] >= 1 and e.res[[0, 1, 2]].all()


def test_step_time_without_misses():
    hw = dict(HW1, attn=[(1, 5.0), (8, 5.0)], tick=1.0, g0=2.0, g1=0.5)
    L, E = 3, 8
    r = dict(dec=np.array([[[0, 1]], [[2, 3]], [[4, 5]]]), D=1)
    e = mu.Engine(L, E, L * E, hw, "lru")
    e.warm(np.ones(L * E))
    t, stall, busy = mu.step(e, [(r, 0)], 0.0, 0, mu.parse_pref("none"), 2, hw, E)
    assert stall == 0.0 and t == L * (5.0 + 1.0 + 2.0 + 0.5) and busy == t


def test_resident_refresh_keeps_capacity():
    rm = np.zeros(8, bool)
    rm[[0, 1]] = True
    e = eng_(L=2, E=4, C=4, rmask=rm)                # resident {0, 1}, flowing capacity 2
    t = e.need([2], 0.0) + 1
    t = e.need([3], t) + 1
    assert e.n_flow == 2 and e.res[[0, 1, 2, 3]].all()
    new = np.zeros(8, bool)
    new[[1, 3]] = True                               # 0 leaves (cached -> flowing), 3 enters (already cached)
    e.set_resident(new, t)
    assert e.rmask[[1, 3]].all() and not e.rmask[0] and e.n_flow == 2 and e.stats["res_loads"] == 0
    new2 = np.zeros(8, bool)
    new2[[3, 5]] = True                              # 1 leaves, 5 enters and is not cached: background load
    e.set_resident(new2, t)
    assert e.n_flow == 3 and e.queued[5] == 2
    e.advance(t + 100)
    assert e.res[5] and e.rmask[5] and e.n_flow == 2 and e.stats["res_loads"] == 1
    assert e.res.sum() == 4


def test_ema_swaps_capped_and_only_when_better():
    L, E, k = 2, 4, 1
    rm = np.zeros(8, bool)
    rm[[0, 1]] = True
    e = eng_(L=L, E=E, C=4, rmask=rm)
    ema = mu.EmaFill("ema:1:4:0.5", 2, np.array([5, 5, 1, 1, 1, 1, 1, 1.0]), L, k)
    swaps = []
    for _ in range(6):
        ema.step(e, np.array([2, 6]), 0.0)          # 2 and 6 used every token; 0, 1 never
        swaps.append(e.stats["res_swaps"])
    assert all(b - a <= 1 for a, b in zip([0] + swaps, swaps))   # at most 0.5 x 2 = 1 swap per refresh
    assert swaps[0] == 0                              # not yet better than the initial (glob) residents
    assert e.rmask[[2, 6]].all() and e.rmask.sum() == 2


def test_pred_masks_follow_nearest_calibration_request():
    L, E = 1, 4
    calib = [dict(pre=np.array([[[0], [0]]]), dec=np.array([[[1], [1]]])),
             dict(pre=np.array([[[3], [3]]]), dec=np.array([[[2], [2]]]))]
    test = [dict(stem="a", pre=np.array([[[0], [0], [3]]])), dict(stem="b", pre=np.array([[[3]]]))]
    m = mu.pred_masks(test, calib, 1, L, E, 1)
    assert m["a"][1] and m["a"].sum() == 1 and m["b"][2]


def test_accumulation_with_tiny_gamma_is_replace():
    E = 16
    reqs = synth_trace(E=E, seed=4)
    L = reqs[0]["dec"].shape[0]
    for pol in ("lru/ah4:AVG:m8:p0.55", "moeinf/ah3:GATE:m8:p0.55"):
        for frac in (0.3, 0.6):
            C = int(frac * L * E)
            a = mu.run_seq(reqs, pol, mu.HW["mac_ssd"], C, E)
            b = mu.run_seq(reqs, pol + ":g0.000001", mu.HW["mac_ssd"], C, E)
            assert abs(a["stall_ms_per_tok"] - b["stall_ms_per_tok"]) < 1e-9, (pol, frac, a["stall_ms_per_tok"], b["stall_ms_per_tok"])
            assert a["loads"] == b["loads"]


def test_freeze_window_keeps_queued_and_adaptive_threshold_issues_less():
    E = 16
    reqs = synth_trace(E=E, seed=5)
    L = reqs[0]["dec"].shape[0]
    C = int(0.3 * L * E)
    base = mu.run_seq(reqs, "lru/ah4:AVG:m8:p0.1:g1", mu.HW["mac_ssd"], C, E)      # low threshold: unstable sets
    assert base["loads"]["la_cancelled"] < mu.run_seq(reqs, "lru/ah4:AVG:m8:p0.1", mu.HW["mac_ssd"], C, E)["loads"]["la_cancelled"]
    frz = mu.run_seq(reqs, "lru/ah4:AVG:m8:p0.1:g1:d4:x2", mu.HW["mac_ssd"], C, E)
    assert frz["loads"]["la_cancelled"] < base["loads"]["la_cancelled"]
    assert frz["loads"]["la_cancelled"] == 0          # whole horizon frozen: nothing is ever cancelled
    ada = mu.run_seq(reqs, "lru/ah4:AVG:m8:p0.1:g1:k0.5", mu.HW["mac_ssd"], C, E)
    assert ada["loads"]["la"] < base["loads"]["la"]   # a long queue raises the threshold: fewer prefetches loaded
    hl = mu.run_seq(reqs, "lru/ah4:AVG:m8:p0.1:g1:hl5", mu.HW["mac_ssd"], C, E)
    assert hl["pf_by_h"]["loads"][1] == 0             # nothing first queued one layer ahead


def test_substitute_is_best_scored_cached_and_ready():
    L, E = 2, 4
    e = eng_(L=L, E=E, C=8)
    t = e.need([1], 0.0) + 1                          # layer 0: experts 1 and 3 cached and ready
    t = e.need([3], t) + 1
    e.lookahead([2], t, 0, 1)                         # expert 2 queued (not started yet)
    r = dict(score=np.array([[[0.1, 0.5, 0.9, 0.7]]], np.float16), w=np.array([[[0.6, 0.4]]], np.float32))
    sb = mu.Subst("tok:1", L, E, log=True)
    out = sb.apply(e, r, 0, 0, np.array([0, 2]), t)  # experts 0 (cold) and 2 (queued): both late
    assert sb.late == 2 and sb.changed == 1           # budget 1: the lower-weight one (expert 2, w 0.4) goes
    assert list(out) == [0, 3]                        # replaced by 3 (cached, ready, best score after 2)
    assert e.queued[2] == -1                          # its queued load is cancelled
    assert sb.records == [(0, 0, 0, 2, 3)]
    out2 = mu.Subst("tok:2", L, E).apply(e, r, 0, 0, np.array([0, 2]), t)
    assert sorted(out2) == [1, 3]                     # both replaced, best-scored candidates first
    out3 = mu.Subst("skip:tok:1", L, E).apply(e, r, 0, 0, np.array([0, 2]), t)
    assert list(out3) == [0]


def test_substitution_cuts_stall_within_rate_budget():
    E = 16
    reqs = synth_trace(E=E, seed=6)
    rng = np.random.default_rng(1)
    for r in reqs:
        sc = rng.random((r["dec"].shape[0], r["D"], E)).astype(np.float16)
        np.put_along_axis(sc, r["dec"], 2.0, -1)
        r["score"] = sc
        w = np.take_along_axis(sc.astype(np.float32), r["dec"], -1)
        r["w"] = w / w.sum(-1, keepdims=True)
    L = reqs[0]["dec"].shape[0]
    C = int(0.4 * L * E)
    base = mu.run_seq(reqs, "lru/none", mu.HW["mac_ssd"], C, E)
    for spec, cap in (("rate:0.05", 0.05), ("rate:0.2", 0.2), ("tok:1", None)):
        r = mu.run_seq(reqs, "lru/none", mu.HW["mac_ssd"], C, E, subst=spec)
        assert r["stall_ms_per_tok"] < base["stall_ms_per_tok"]
        if cap is not None:
            assert r["subst_change_rate"] <= cap + 1e-9
    big = mu.run_seq(reqs, "lru/none", mu.HW["mac_ssd"], C, E, subst="rate:0.2")
    small = mu.run_seq(reqs, "lru/none", mu.HW["mac_ssd"], C, E, subst="rate:0.05")
    assert big["stall_ms_per_tok"] < small["stall_ms_per_tok"]


# ---------------------------------------------------------------- agreement with the original simulators
def synth_trace(n_req=4, L=6, E=16, k=4, T=12, D=40, H=4, M=8, seed=0):
    """Random routes with some temporal locality, plus anchor_dump-style predictions."""
    rng = np.random.default_rng(seed)
    hot = rng.random((L, E)) ** 3
    reqs = []
    for _ in range(n_req):
        def route(n):
            out = np.zeros((L, n, k), np.int64)
            for l in range(L):
                p = hot[l] / hot[l].sum()
                prev = rng.choice(E, k, replace=False, p=p)
                for i in range(n):
                    cur = prev.copy() if rng.random() < 0.4 else rng.choice(E, k, replace=False, p=p)
                    out[l, i] = cur
                    prev = cur
            return out
        pre, dec = route(T), route(D)
        la = np.stack([np.where(rng.random((D, k)) < 0.7, dec[l + 1], rng.integers(0, E, (D, k))) for l in range(L - 1)])
        P = {}
        for name, noise in (("GATE", 0.35), ("AVG", 0.25), ("ORACLE", 0.0)):
            ids = np.full((L, D, H, M), -1, np.int16)
            sc = np.zeros((L, D, H, M), np.float16)
            for l in range(L):
                for h in range(1, H + 1):
                    if l + h >= L:
                        break
                    s = rng.random((D, E)) * noise + 1e-3 * hot[l + h]
                    s[np.arange(D)[:, None], dec[l + h]] += 1.0 - noise
                    o = np.argsort(-s, -1)[:, :M]
                    ids[l, :, h - 1] = o
                    sc[l, :, h - 1] = np.take_along_axis(s, o, -1)
            P[name], P[name + "_s"] = ids, sc
        P["conf"] = rng.random((L, D)).astype(np.float16)
        reqs.append(dict(pre=pre, dec=dec, la_dec=la, P=P, D=D))
    return reqs


def test_seq_matches_tsim_ahead():
    import tsim_ahead as ta
    E = 16
    slow = ta.HW["glm4bit"]
    fast = dict(slow, ssd=90e9)          # loads 10x faster: prefetches complete, protected experts fill the cache
    fast_mu = mu._legacy(fast["eb"], fast["nonexp_layer"], fast["mem_bw"], fast["ssd"], fast["act_params_layer"],
                         fast["c_pre"])
    cases = [(synth_trace(E=E), slow, mu.HW["p3_glm4bit"], (0.3, 0.6)),
             (synth_trace(E=E, T=2), fast, fast_mu, (9 / 96, 0.2))]          # C = 9: thousands of dropped prefetches
    pols = ("lru/none", "moeinf/la1", "lru/la1", "lfu/ah3:GATE", "lru/ah4:AVG:m8:p0.6", "moeinf/ah4:ORACLE:m8",
            "lru/ah4:MIX:m8:t0.5")
    dropped = 0
    for (reqs, hw_old, hw_new, fracs), pol in itertools.product(cases, pols):
        L = reqs[0]["dec"].shape[0]
        for frac in fracs:
            C = int(round(frac * L * E))
            old = ta.run([dict(pre=r["pre"], dec=r["dec"], la_dec=r["la_dec"]) for r in reqs],
                         [r["P"] for r in reqs], pol, hw_old, C, E, 0.0, 2)
            new = mu.run_seq(reqs, pol, hw_new, C, E)
            dropped += new["loads"]["la_dropped"]
            for key in ("stall_ms_per_tok", "tpot_ms", "ttft_ms", "loads_per_tok", "wasted_frac"):
                assert abs(old[key] - new[key]) <= 1e-6 * max(1.0, abs(old[key])), (pol, frac, key, old[key], new[key])
            for key in ("loaded", "demand", "la", "wasted", "la_dropped", "la_cancelled"):
                assert old["loads"][key] == new["loads"][key], (pol, frac, key, old["loads"], new["loads"])
    assert dropped > 1000                                # the drop path was exercised


def test_pool_matches_tsim_async():
    import tsim_async as tq
    E = 16
    reqs = synth_trace(n_req=12, E=E, D=80, seed=1)
    L = reqs[0]["dec"].shape[0]
    rng = np.random.default_rng(3)
    for r in reqs:
        w = rng.random(r["dec"].shape).astype(np.float32)
        r["w"] = w / w.sum(-1, keepdims=True)
    streams = [(r["dec"].transpose(1, 0, 2), r["la_dec"].transpose(1, 0, 2), r["w"].transpose(1, 0, 2)) for r in reqs]
    offs = [int(rng.integers(0, r["D"])) for r in reqs]
    horizon = 3e5
    for frac in (0.4, 0.7):
        C = int(frac * L * E)
        for kw in (dict(), dict(sub_n=2), dict(B=4, select="prev", klim=2), dict(B=4, select="fcfs"),
                   dict(B=4, select="oracle", klim=1)):
            old = tq.run_sync(streams, offs, tq.HW["h200_pcie"], C, E, horizon, False, **kw)
            new = mu.run_pool(reqs, offs, "lfu/none", mu.HW["h200_pcie"], C, E, horizon, **kw)
            for key in ("tokens", "throughput_tok_s", "tpot_p50_ms", "tpot_p99_ms", "loads_per_token", "gpu_busy"):
                assert abs(old[key] - new[key]) <= 1e-9 * max(1.0, abs(old[key])), (kw, frac, key, old[key], new[key])
            if kw.get("sub_n"):
                assert abs(old["dropped_use_frac"] - new["dropped_use_frac"]) < 1e-12


def test_pool_warm_counts():
    """warm_f = the streams' own counts is the default; other counts (or no warm start) change the run."""
    E = 16
    reqs = synth_trace(n_req=8, E=E, D=60, seed=2)
    L = reqs[0]["dec"].shape[0]
    offs = [0] * len(reqs)
    C = int(0.5 * L * E)
    ref = mu.run_pool(reqs, offs, "lfu/none", mu.HW["h200_pcie"], C, E, 2e5, B=4)
    same = mu.run_pool(reqs, offs, "lfu/none", mu.HW["h200_pcie"], C, E, 2e5, B=4, warm_f=mu.use_counts(reqs, L, E))
    assert ref["tokens"] == same["tokens"] and ref["loads_per_token"] == same["loads_per_token"]
    other = mu.use_counts(synth_trace(n_req=8, E=E, D=60, seed=9), L, E)
    alt = mu.run_pool(reqs, offs, "lfu/none", mu.HW["h200_pcie"], C, E, 2e5, B=4, warm_f=other)
    cold = mu.run_pool(reqs, offs, "lfu/none", mu.HW["h200_pcie"], C, E, 2e5, B=4, warm=False)
    assert alt["loads_per_token"] > ref["loads_per_token"] and cold["loads_per_token"] > ref["loads_per_token"]


def test_layer_caps_and_per_layer_eviction():
    L, E = 3, 4
    f = np.array([[9, 8, 1, 0], [5, 4, 3, 2], [7, 0, 0, 0]], float)
    caps = mu.layer_caps(f, 6, L, E, floor=1)
    assert caps.tolist() == [2, 3, 1]          # forced 9 | 5 | 7, then the largest rest 8, 4, 3
    assert mu.layer_caps(f, 6, L, E, floor=0).tolist() == [2, 3, 1]           # 9 8 7 5 4 3
    assert mu.layer_caps(f, 5, L, E, floor=0).tolist() == [2, 2, 1]
    assert mu.layer_caps(f, 6, L, E, floor=2).tolist() == [2, 2, 2]
    eng = mu.Engine(L, E, 6, mu.HW["h200_pcie"], "lfu", cap_l=[3, 2, 1])
    eng.warm(f.ravel())
    assert eng.res.reshape(L, E).sum(1).tolist() == [3, 2, 1]
    assert eng.res.reshape(L, E)[0, :3].all() and eng.res.reshape(L, E)[2, 0]
    eng._load(0, 2 * E + 3)                    # layer 2 is full: evicts its own key, never another layer's
    assert eng.res.reshape(L, E).sum(1).tolist() == [3, 2, 1]
    assert eng.res[2 * E + 3] and not eng.res[2 * E + 0]
    eng._load(0, 1 * E + 3)                    # layer 1 holds experts 0 (count 5), 1 (4): lfu evicts 1
    assert eng.res.reshape(L, E)[1].tolist() == [True, False, False, True]
    assert eng.res.reshape(L, E).sum(1).tolist() == [3, 2, 1]
    eng.need(np.arange(E), 0.0)              # layer 0 needs all 4 (> capacity 3): over for the use, then trimmed
    assert eng.stats["overflow"] == 1 and eng.res.reshape(L, E).sum(1).tolist() == [3, 2, 1]
    assert eng.res.reshape(L, E)[0].tolist() == [True, True, True, False]      # the lowest count (0) left


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
