"""Decision trace on a STATIC synthetic observation (forward passes only; no traffic model).

The observation is frozen (one or a few waiting vehicles, everything else empty). Starting from each phase p at a
green age of 12 s, apply the controller's switching rule (decision every 5 s; a different pick with green >= 10 s
-> 3 s yellow, the new green is 2 s old at the next decision) and report how long it takes until the phase that
serves the waiting vehicle turns green (inf = never within 900 s: a deadlock that lasts until the drain cap).
A hidden W2 vehicle is an EMPTY observation whose serving phase is 3.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, tools
from drain_table import obs, SERVE

LANES = [l for l in SERVE if l != "W2"]


def trace(w, counts, serve, p0, horizon=900, log=None):
    serve = {serve} if isinstance(serve, int) else set(serve)
    p, age, t = p0, 12, 0
    while t < horizon:
        if log is not None and age == 12:
            log.append(p)
        if p in serve:
            return t
        pick = int(tools.greedy(w, obs(counts, p, age))[0])
        if pick != p and age >= 10:
            p, age, t = pick, 2, t + 5
        else:
            age, t = age + 5, t + 5
    return np.inf


def configs():
    out = [("hiddenW2", {}, 3)]
    for l in LANES:
        for n in (1, 2):
            out.append((f"{l}x{n}", {l: n}, SERVE[l]))
    for n in (1, 2, 5):                     # hidden full W2 pocket with left-turners halted behind it on W1u
        out.append((f"spillW1u{n}", {"W1u": n}, 3))
    for a, b in [("N0", "W0"), ("S1", "W0"), ("N0", "E0"), ("E0", "W0"), ("N2", "E1"), ("S2", "W1"), ("E2", "W0")]:
        out.append((f"{a}+{b}", {a: 1, b: 1}, None))    # two lanes, both must get served
    return out


def report(w, name, verbose=True):
    bad, worst = [], 0
    for lab, counts, serve in configs():
        s = {serve} if serve is not None else {SERVE[l] for l in counts}   # reaching any one is enough
        ts = [trace(w, counts, s, p) for p in range(4) if p not in s]
        m = max(ts)
        if m == np.inf:
            loops = []
            for p in range(4):
                if p in s: continue
                lg = []; trace(w, counts, s, p, log=lg); loops.append("".join(map(str, lg[-6:])))
            bad.append(f"{lab}->{''.join(map(str, sorted(s)))}" + (f"[{'/'.join(loops)}]" if verbose else ""))
        else:
            worst = max(worst, m)
    print(f"{name}: deadlocks {len(bad)}: {bad}  worst finite wait {worst} s")
    return bad


if __name__ == "__main__":
    from compare import cands
    from impute import fold, W0
    report(W0, "start (dead W2)")
    for k, (aq, ad, cq, cd) in cands.items():
        report(fold(aq, ad, cq, cd), k)
