"""Multi-user plan 4.2.1: background multi-layer prefetch for the Mac offload engine (offload_engine.OffloadEngine),
mirroring tsim_mu.Engine's lookahead queue:

  - a prediction for one target position (step x L + layer) replaces the previous one for that target: experts no
    longer predicted are taken off the queue (cancelled); new ones are queued unless cached, being read or queued;
    want[layer, expert] = the latest target that wants it (protection);
  - one reader thread takes the queued expert with the earliest target; a target already passed is dropped (expired);
    the victim is the lowest-scoring cached expert of that layer that is not pinned (the running layer's experts),
    not being read, and preferably not wanted; if the victim is wanted no later than this target the read is dropped;
  - demand reads (the engine's own pool) never wait behind queued prefetches: a needed expert still queued is taken
    off the queue and read as a demand read; one already being read is waited for;
  - a finished read removes its own pending entry, so an unused prefetched expert can be evicted later.
All shared state is guarded by the engine's lock.

  pf = Prefetcher(eng); eng.prefetcher = pf; ... pf.lookahead(layer, ids, target) ...; pf.close()
"""
import heapq
import threading
from concurrent.futures import Future

import numpy as np


def victim(eng, j, protect=(), deadline=None):
    """Slot to (re)use in layer j's pool, or None. Free slots first; otherwise the lowest-scoring cached expert that is
    not protected / being read, preferring ones nobody wants (tsim_mu.Engine.victim). With a deadline (a prefetch for
    that target): None if the victim is wanted no later than the deadline (tsim_mu.Engine._load, priority 1)."""
    occ = eng.exp_in[j]
    free = np.flatnonzero(occ == -1)
    if len(free):
        return int(free[0])
    cand = np.array([s for s in range(eng.S) if occ[s] >= 0 and int(occ[s]) not in protect
                     and (j, int(occ[s])) not in eng.pending], np.int64)
    if not len(cand):
        return None
    es = occ[cand]
    wanted = eng.want[j, es] >= eng.now
    if not wanted.all():
        cand, es = cand[~wanted], es[~wanted]
    i = int(np.argmin(eng._score(j, es)))
    if deadline is not None and eng.want[j, es[i]] >= eng.now and eng.want[j, es[i]] <= deadline:
        return None
    return int(cand[i])


class Prefetcher:
    def __init__(self, eng, workers=1):
        self.eng = eng
        self.cv = threading.Condition(eng.lock)
        self.heap = []                              # (target, seq, layer, expert)
        self.queued = {}                            # (layer, expert) -> target it is queued for
        self.tq = {}                                # target -> experts of the latest prediction for it
        self.seq = 0
        self.stats = dict(issued=0, loaded=0, cancelled=0, expired=0, dropped=0, taken=0)
        self.alive = True
        self.threads = [threading.Thread(target=self._run, daemon=True) for _ in range(workers)]
        for t in self.threads:
            t.start()

    # ---------------------------------------------------------------- main thread
    def lookahead(self, j, ids, target):
        """The latest prediction for (layer j at position `target`): experts `ids`, most confident first."""
        eng = self.eng
        ids = [int(e) for e in ids]
        with self.cv:
            new = set(ids)
            for e in self.tq.get(target, set()) - new:
                if self.queued.get((j, e)) == target:
                    del self.queued[(j, e)]
                    eng.want[j, e] = -1
                    self.stats["cancelled"] += 1
            self.tq[target] = new
            for e in ids:
                if eng.slot_of[j, e] >= 0 or (j, e) in eng.pending or (j, e) in self.queued:
                    continue
                self.queued[(j, e)] = target
                self.seq += 1
                heapq.heappush(self.heap, (target, self.seq, j, e))
                self.stats["issued"] += 1
            if ids:
                eng.want[j, ids] = np.maximum(eng.want[j, ids], target)
            for t in [t for t in self.tq if t < eng.now]:
                del self.tq[t]
            self.cv.notify()

    def take(self, j, e):
        """A demand read of (j, e) is about to be issued (caller holds the lock): drop it from the queue."""
        if (j, e) in self.queued:
            del self.queued[(j, e)]
            self.stats["taken"] += 1

    def cancel(self, j, e):
        """Substitution replaced (j, e): a queued read of it is cancelled (one in progress continues)."""
        with self.cv:
            if (j, e) in self.queued:
                del self.queued[(j, e)]
                self.eng.want[j, e] = -1
                self.stats["cancelled"] += 1

    def clear(self):
        """Forget every queued read (a new configuration starts from warm pools)."""
        with self.cv:
            self.heap, self.queued, self.tq = [], {}, {}

    def n_queued(self):
        return len(self.queued)

    def close(self):
        with self.cv:
            self.alive = False
            self.cv.notify_all()
        for t in self.threads:
            t.join(timeout=5)

    # ---------------------------------------------------------------- reader thread
    def _next(self):
        """(caller holds the lock) Pop the next valid queued read and claim its slot -> task, or None."""
        eng = self.eng
        target, _s, j, e = heapq.heappop(self.heap)
        if self.queued.get((j, e)) != target:          # cancelled, taken by a demand read, re-queued
            return None
        del self.queued[(j, e)]
        if target < eng.now:
            self.stats["expired"] += 1
            return None
        if eng.slot_of[j, e] >= 0 or (j, e) in eng.pending:
            return None
        s = victim(eng, j, eng.pinned if eng.pin_layer == j else (), target)
        if s is None:
            self.stats["dropped"] += 1
            return None
        eng._assign(j, e, s)
        fut = Future()
        eng.pending[(j, e)] = fut
        eng.stats["loads"] += 1
        eng.stats["bytes"] += eng.rec
        eng.stats["prefetch_loads"] += 1
        self.stats["loaded"] += 1
        return j, e, fut, [v[s] for v in eng.pool_np[j]]

    def _do(self, task):
        j, e, fut, views = task
        self.eng._read_into(j, e, views)
        with self.cv:
            fut.set_result(True)
            if self.eng.pending.get((j, e)) is fut:
                del self.eng.pending[(j, e)]

    def _run(self):
        while True:
            with self.cv:
                while self.alive and not self.heap:
                    self.cv.wait()
                if not self.alive:
                    return
                task = self._next()
            if task is not None:
                self._do(task)
