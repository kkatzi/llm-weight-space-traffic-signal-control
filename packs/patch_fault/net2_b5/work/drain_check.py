"""Exhaustive drain check (forward passes on frozen synthetic observations; no traffic model).

Configurations: every set of 1-3 lanes (12 stop-line lanes incl. the hidden W2 pocket, plus 'spill' = hidden full
W2 with 2 left-turners halted on W1u) holding 1 or 3 waiting vehicles. From each start phase, follow the
controller's switching rule; when a phase turns green, the lanes it serves are cleared (one green of >= 10 s
clears 1-3 queued vehicles); continue until everything is served. Reports configurations never cleared within
900 s and the slowest clearing time."""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, tools
from drain_table import obs, SERVE

LANES = list(SERVE) + ["spill"]


def serve_of(l):
    return 3 if l == "spill" else SERVE[l]


def to_counts(cfg):
    c = {}
    for l, n in cfg.items():
        if l == "spill":
            c["W1u"] = c.get("W1u", 0) + 2          # visible part; the full W2 pocket is invisible
        elif l != "W2":
            c[l] = c.get(l, 0) + n
    return c


def clear_time(w, cfg, p0, horizon=900, cache=None):
    cfg = dict(cfg); p, age, t = p0, 12, 0
    while cfg and t < horizon:
        served = [l for l in cfg if serve_of(l) == p]
        if served and age <= 12:                     # the green that just started clears its lanes
            for l in served:
                del cfg[l]
            if not cfg:
                return t
        key = (tuple(sorted(to_counts(cfg).items())), p, min(age, 92))
        if cache is not None and key in cache:
            pick = cache[key]
        else:
            pick = int(tools.greedy(w, obs(to_counts(cfg), p, age))[0])
            if cache is not None:
                cache[key] = pick
        if pick != p and age >= 10:
            p, age, t = pick, 2, t + 5
            age_after = 2
        else:
            age, t = age + 5, t + 5
        if age > 12 and any(serve_of(l) == p for l in cfg):
            for l in [l for l in cfg if serve_of(l) == p]:
                del cfg[l]
    return t if not cfg else np.inf


def check(w, name, sizes=(1, 2, 3), counts=(1, 3)):
    cache, fails, worst, n = {}, [], (0, None), 0
    for k in sizes:
        for lanes in itertools.combinations(LANES, k):
            for nv in counts:
                cfg = {l: nv for l in lanes}
                for p in range(4):
                    if all(serve_of(l) == p for l in lanes):
                        continue
                    n += 1
                    tt = clear_time(w, cfg, p, cache=cache)
                    if tt == np.inf:
                        fails.append((lanes, nv, p))
                    elif tt > worst[0]:
                        worst = (tt, (lanes, nv, p))
    print(f"{name}: {n} drain cases, {len(fails)} never cleared; slowest {worst}")
    if fails:
        from collections import Counter
        print("   lanes most often involved in failures:", Counter(l for f in fails for l in f[0]).most_common(8))
        print("   examples:", fails[:6])
    return fails


if __name__ == "__main__":
    import build
    from impute import W0, fold
    check(W0, "start network (dead W2)")
    check(fold({17: 1.0}, {17: 1.0}), "mirror")
    check(build.imputation(), "imputation only")
    check(build.build(1.0), "imputation + detector (alpha 1)")
