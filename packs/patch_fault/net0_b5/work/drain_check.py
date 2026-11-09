"""Drain-phase check with synthetic observations (forward passes only, no traffic simulator).

At the end of an episode only a few vehicles are left. If the network never gives green to the phase that serves
them, they stay until the 30-min cap. For each 'leftover' pattern (one or a few vehicles on a lane, frozen until
their phase is green for >= 5 s), replay the signal logic of TASK.md (decision every 5 s, 10 s minimum green,
3 s yellow, new green is 2 s old at the next decision) and report whether and when the serving phase is reached.
The W2 pocket is invisible (inputs 11, 23 are always 0), so a W2 leftover only shows through the W1u queue.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import tools  # noqa: E402

N = tools.names()
I = {k: i for i, k in enumerate(N)}
CAP_STOP, CAP_UP = 22.4 / 7.5, 256 / 7.5
SERVE = {"N0": 0, "N1": 0, "N2": 1, "S0": 0, "S1": 0, "S2": 1, "E0": 2, "E1": 2, "E2": 3, "W0": 2, "W1": 2, "W2": 3}


def obs(lanes, phase, g):
    """lanes: dict lane-> number of halted vehicles. Upstream names end with 'u'."""
    x = np.zeros(46)
    for ln, k in lanes.items():
        if ln == "W2":
            continue  # failed detector
        if ln.endswith("u"):
            v = min(1.0, k / CAP_UP)
        else:
            v = min(1.0, k / CAP_STOP)
        x[I["q_" + ln]] = v
        x[I["d_" + ln]] = v
    x[40 + phase] = 1
    x[44] = 1.0 if g >= 10 else 0.0
    x[45] = min(1.0, g / 60)
    return x


def replay(w, lanes, phase, g, horizon=900):
    """Return the first time (s) a phase serving the leftover is green for >= 5 s, or None."""
    need = {SERVE[ln[:2]] for ln in lanes}
    t = 0
    while t < horizon:
        if phase in need and g >= 5:
            return t
        a = int(tools.greedy(w, obs(lanes, phase, g))[0])
        if a != phase and g >= 10:
            phase, g = a, 2  # 3 s yellow, then 2 s of the new green by the next decision
        else:
            g += 5
        t += 5
    return None


PATTERNS = [{ln: 1} for ln in SERVE if ln != "W2"] + [{ln: 3} for ln in SERVE] + [
    {"W2": 1}, {"W2": 3}, {"W2": 3, "W1u": 1}, {"W2": 3, "W1u": 3}, {"W2": 3, "W1u": 6},
    {"E2": 3, "E1u": 1}, {"N2": 3, "N1u": 2}, {"S2": 3, "S1u": 2},
]


def check(w, verbose=True):
    bad = []
    worst = 0
    for pat in PATTERNS:
        for p0 in range(4):
            for g0 in (12, 60):
                r = replay(w, pat, p0, g0)
                if r is None:
                    bad.append((pat, p0, g0))
                else:
                    worst = max(worst, r)
    if verbose:
        print("never served:", bad if bad else "none", "| worst wait (s):", worst)
    return bad, worst


if __name__ == "__main__":
    for f in sys.argv[1:]:
        print(f)
        check(tools.load(f))
