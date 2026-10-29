"""Tests of prefetch.Prefetcher with a fake engine (no model, no I/O): queue order, replacement of a target's
prediction, expiry, protection of experts wanted sooner, pinned experts, demand take-over, cleanup after a read.

  .venv-mac/bin/python experiments/moe_edge/test_prefetch.py
"""
import os
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prefetch  # noqa: E402


class FakeEng:
    def __init__(self, L=3, E=8, S=2):
        self.L, self.E, self.S = L, E, S
        self.lock = threading.Lock()
        self.exp_in = np.full((L, S), -1, np.int64)
        self.slot_of = np.full((L, E), -1, np.int64)
        self.pending = {}
        self.want = np.full((L, E), -1.0)
        self.now = 0.0
        self.pinned, self.pin_layer = set(), -1
        self.freq = np.zeros((L, E))
        self.pool_np = [[np.zeros((S, 4))] for _ in range(L)]
        self.stats = dict(loads=0, bytes=0, prefetch_loads=0)
        self.rec = 4
        self.reads = []

    def _score(self, j, es):
        return self.freq[j, es]

    def _assign(self, j, e, s):
        old = self.exp_in[j, s]
        if old >= 0:
            self.slot_of[j, old] = -1
        self.exp_in[j, s] = e
        self.slot_of[j, e] = s

    def _read_into(self, j, e, views):
        views[0][:] = e
        self.reads.append((j, e))


def drain(pf):
    while True:
        with pf.cv:
            if not pf.heap:
                return
            task = pf._next()
        if task is not None:
            pf._do(task)


def test_order_and_cleanup():
    eng = FakeEng()
    pf = prefetch.Prefetcher(eng, workers=0)
    pf.lookahead(2, [5, 6], target=2)
    pf.lookahead(1, [3], target=1)
    drain(pf)
    assert eng.reads == [(1, 3), (2, 5), (2, 6)]                 # earliest target first
    assert eng.slot_of[2, 5] >= 0 and eng.slot_of[2, 6] >= 0 and not eng.pending   # pending cleaned after the read
    assert eng.want[2, 5] == 2 and pf.stats["loaded"] == 3


def test_replace_prediction_cancels():
    eng = FakeEng()
    pf = prefetch.Prefetcher(eng, workers=0)
    pf.lookahead(2, [5, 6], target=2)
    pf.lookahead(2, [6, 7], target=2)                           # 5 is no longer predicted for this target
    drain(pf)
    assert sorted(eng.reads) == [(2, 6), (2, 7)] and pf.stats["cancelled"] == 1 and eng.want[2, 5] == -1


def test_expired_and_taken():
    eng = FakeEng()
    pf = prefetch.Prefetcher(eng, workers=0)
    pf.lookahead(1, [3], target=1)
    pf.lookahead(2, [4], target=2)
    eng.now = 1.5                                               # layer 1 of this step has passed
    with eng.lock:
        pf.take(2, 4)                                           # a demand read takes (2, 4) over
    drain(pf)
    assert eng.reads == [] and pf.stats["expired"] == 1 and pf.stats["taken"] == 1


def test_protection_and_pin():
    eng = FakeEng(S=2)
    for e in (0, 1):                                            # layer 2 full: experts 0 (freq 5) and 1 (freq 1)
        eng._assign(2, e, e)
    eng.freq[2, 0], eng.freq[2, 1] = 5, 1
    eng.want[2, 1] = 2                                          # expert 1 is wanted at target 2
    pf = prefetch.Prefetcher(eng, workers=0)
    pf.lookahead(2, [4], target=3)                              # victim: 0 (unwanted first) even if higher freq
    drain(pf)
    assert eng.slot_of[2, 4] >= 0 and eng.slot_of[2, 0] < 0 and eng.slot_of[2, 1] >= 0
    eng.want[2, 4] = 2                                          # now both cached experts are wanted at target 2
    pf.lookahead(2, [6], target=5)                              # wanted no later than 5: dropped
    drain(pf)
    assert eng.slot_of[2, 6] < 0 and pf.stats["dropped"] == 1
    eng.want[2, [1, 4]] = -1
    eng.pinned, eng.pin_layer = {1, 4}, 2                       # the running layer's experts are never evicted
    pf.lookahead(2, [7], target=6)
    drain(pf)
    assert eng.slot_of[2, 7] < 0 and pf.stats["dropped"] == 2


def test_thread_runs():
    eng = FakeEng()
    pf = prefetch.Prefetcher(eng, workers=1)
    pf.lookahead(1, [2, 3], target=1)
    for _ in range(200):
        if len(eng.reads) == 2:
            break
        threading.Event().wait(0.01)
    pf.close()
    assert sorted(eng.reads) == [(1, 2), (1, 3)] and not eng.pending


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
