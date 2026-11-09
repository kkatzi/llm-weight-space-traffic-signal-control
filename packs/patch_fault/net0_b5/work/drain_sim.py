"""Toy drain replay (forward passes on synthetic observations only; not a traffic simulator).

End-of-episode situation: no new arrivals, a few vehicles left on random stop-line lanes (W2 hidden). The signal
logic follows TASK.md (5 s decisions, 10 s min green, 3 s yellow). A lane whose phase is green discharges one
vehicle every 2 s after 2 s of green. Report the share of situations not cleared within 1800 s and the mean
vehicle-seconds spent until clearing (cap 1800 s).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import tools  # noqa: E402

I = {k: i for i, k in enumerate(tools.names())}
LANES = ["N0", "N1", "N2", "E0", "E1", "E2", "S0", "S1", "S2", "W0", "W1", "W2"]
SERVE = {"N0": 0, "N1": 0, "N2": 1, "S0": 0, "S1": 0, "S2": 1, "E0": 2, "E1": 2, "E2": 3, "W0": 2, "W1": 2,
         "W2": 3}
CAP = 22.4 / 7.5


def obs(cnt, phase, g):
    x = np.zeros(46)
    for ln, k in cnt.items():
        if ln == "W2" or k <= 0:
            continue
        v = min(1.0, k / CAP)
        x[I["q_" + ln]] = v
        x[I["d_" + ln]] = v
    x[40 + phase] = 1
    x[44] = 1.0 if g >= 10 else 0.0
    x[45] = min(1.0, g / 60)
    return x


def run(w, cnt, phase, g, cap=1800):
    cnt = dict(cnt)
    t, vs = 0, 0.0
    yellow = 0
    while t < cap and sum(cnt.values()) > 0:
        if t % 5 == 0:
            a = int(tools.greedy(w, obs(cnt, phase, g))[0])
            if a != phase and g >= 10:
                phase, g, yellow = a, -3, 3
        # one second
        vs += sum(cnt.values())
        if yellow > 0:
            yellow -= 1
            g += 1
            if yellow == 0:
                g = 0
        else:
            g += 1
            if g >= 2 and g % 2 == 0:
                for ln in LANES:
                    if SERVE[ln] == phase and cnt.get(ln, 0) > 0:
                        cnt[ln] -= 1
        t += 1
    return sum(cnt.values()) == 0, vs + (sum(cnt.values()) * (cap - t) if t < cap else 0)


def situations(n=600, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        k = rng.integers(1, 4)
        cnt = {}
        for ln in rng.choice(LANES, size=k, replace=True):
            cnt[ln] = min(3, cnt.get(ln, 0) + 1)
        out.append((cnt, int(rng.integers(0, 4)), int(rng.choice([12, 17, 22, 40, 60]))))
    return out


def score(w, sits=None):
    sits = situations() if sits is None else sits
    res = [run(w, c, p, g) for c, p, g in sits]
    stuck = [s for s, r in zip(sits, res) if not r[0]]
    return {"stuck_share": len(stuck) / len(sits), "mean_veh_s": float(np.mean([r[1] for r in res]))}, stuck


if __name__ == "__main__":
    for f in sys.argv[1:]:
        sc, stuck = score(tools.load(f))
        print(f, sc)
        from collections import Counter
        print("   stuck lanes:", Counter(ln for c, p, g in stuck for ln in c).most_common())
