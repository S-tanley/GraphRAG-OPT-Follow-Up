"""A small GPU queue for the shared node (only nvidia-smi is used; other users' processes are never touched).

Two kinds of jobs (user's rule, 2026-10-26):
  share: true    results do not depend on sharing the GPU (accuracy runs, trace collection: the numbers are the same,
                 only slower). A GPU qualifies if it has >= mem_gb + headroom_gb free, even with other users' processes.
  share: false   timing-sensitive (benchmarks): a GPU qualifies once no other user's process has been on it for
                 >= idle_min minutes. The wait is done once (user, 2026-10-26): the GPU then stays cleared while only
                 my own processes use it (e.g. back-to-back jobs of this queue) and is re-checked only after another
                 user's process shows up. --cleared marks GPUs already waited for (e.g. when the queue is restarted).
A GPU running one of this queue's jobs is never given a second one. Jobs run in file order, but a later job may start
first if it fits (e.g. a one-GPU job while a two-GPU job waits).

jobs json: [{"name": "...", "gpus": 1, "share": true, "mem_gb": 110, "cmd": "... $CUDA ..."}, ...]

  python experiments/moe/gpu_queue.py --jobs logs/gpu_jobs.json --allowed 0 1 2 --status logs/gpu_queue.status
"""
import argparse
import getpass
import json
import subprocess
import time


def gpu_state(allowed, me):
    """{gpu: (free GB, another user is on it)} for the allowed GPUs. Another user = a compute process owned by someone
    else, or >= 1 GB in use with no visible process at all."""
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout
    rows = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip().splitlines()
    apps = [tuple(x.strip() for x in r.split(",")) for r in rows if r.strip()]
    owner = {}
    if apps:
        ps = subprocess.run(["ps", "-o", "pid=,user=", "-p", ",".join(p for _u, p in apps)],
                            capture_output=True, text=True).stdout.split("\n")
        owner = {x.split()[0]: x.split()[1] for x in ps if len(x.split()) == 2}
    out = {}
    for line in q.strip().splitlines():
        i, uuid, used, total = [x.strip() for x in line.split(",")]
        if int(i) in allowed:
            here = [p for u, p in apps if u == uuid]
            other = any(owner.get(p, "?") != me for p in here) or (not here and float(used) >= 1000)
            out[int(i)] = ((float(total) - float(used)) / 1024, other)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--allowed", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--status", required=True)
    ap.add_argument("--poll", type=int, default=120)
    ap.add_argument("--headroom_gb", type=float, default=15.0)
    ap.add_argument("--idle_min", type=float, default=30.0, help="share=false jobs: no other user for this long, once")
    ap.add_argument("--cleared", type=int, nargs="*", default=[], help="GPUs already waited for")
    a = ap.parse_args()
    me = getpass.getuser()
    pending = json.load(open(a.jobs))
    running = {}                                      # name -> (Popen, gpus)
    idle_since = {}                                   # gpu -> since when no other user has been seen on it
    cleared = set(a.cleared)                          # GPUs waited for; kept until another user shows up

    def log(msg):
        with open(a.status, "a") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")
    log(f"queue start: {[j['name'] for j in pending]}; cleared {sorted(cleared)}")
    while pending or running:
        for name, (p, g) in list(running.items()):
            if p.poll() is not None:
                log(f"done {name} exit {p.returncode} (GPUs {g})")
                del running[name]
        mine = {i for _, g in running.values() for i in g}
        state = gpu_state(a.allowed, me)
        now = time.time()
        for i, (_free, other) in state.items():
            if other:
                idle_since.pop(i, None)
                if i in cleared:
                    cleared.discard(i)
                    log(f"GPU {i}: another user's process; wait {a.idle_min:g} min again before timing jobs")
            else:
                idle_since.setdefault(i, now)
                if i not in cleared and now - idle_since[i] >= 60 * a.idle_min:
                    cleared.add(i)
                    log(f"GPU {i}: no other user for {a.idle_min:g} min, cleared for timing jobs")
        for j in list(pending):
            if j.get("share", False):
                ok = [i for i, (free, _b) in state.items() if i not in mine and free >= j["mem_gb"] + a.headroom_gb]
            else:
                ok = [i for i, (_f, other) in state.items() if i not in mine and not other and i in cleared]
            if len(ok) >= j["gpus"]:
                g = sorted(ok, key=lambda i: -state[i][0])[:j["gpus"]]       # the emptiest GPUs first
                cmd = j["cmd"].replace("$CUDA", ",".join(map(str, g)))
                p = subprocess.Popen(["bash", "-c", cmd])
                running[j["name"]] = (p, g)
                pending.remove(j)
                mine |= set(g)
                log(f"start {j['name']} on GPUs {g} (free GB {[round(state[i][0]) for i in g]})")
        time.sleep(a.poll if pending else 30)
    log("all done")


if __name__ == "__main__":
    main()
