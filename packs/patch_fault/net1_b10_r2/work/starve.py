"""starve.py: closed-loop check of the signal logic on frozen synthetic observations (forward passes only).

A few vehicles wait on some lanes and nothing else moves (the end-of-episode drain situation). We play the
junction's switching rules (5 s decisions, 10 s minimum green, 3 s yellow) and report how long it takes until every
waiting lane has had its phase green. A lane that is never served within 600 s is 'starved'.

    python work/starve.py work/<file>.npz
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools  # noqa: E402

N = tools.names()
I = {n: i for i, n in enumerate(N)}
SERVE = {"N0": 0, "N1": 0, "S0": 0, "S1": 0, "N2": 1, "S2": 1, "E0": 2, "E1": 2, "W0": 2, "W1": 2, "E2": 3,
         "W2": 3}
LANES = list(SERVE)


def run(w, waiting, start_phase, start_green=60, horizon=600):
    """waiting: dict lane -> cars. W2 cars are invisible (inputs stay 0). Returns time until all served or None."""
    left = dict(waiting)
    phase, green = start_phase, start_green
    t = 0
    while t < horizon:
        # lanes of the current phase get served once green >= 2 s (the car starts moving)
        for ln in list(left):
            if SERVE[ln] == phase and green >= 2:
                del left[ln]
        if not left:
            return t
        x = np.zeros(46)
        for ln, k in left.items():
            if ln == "W2":
                continue
            v = min(k / 2.99, 1.0)
            x[I["q_" + ln]] = v
            x[I["d_" + ln]] = v
        x[40 + phase] = 1
        x[I["ok"]] = 1.0 if green >= 10 else 0.0
        x[I["t_green"]] = min(green / 60.0, 1.0)
        a = int(tools.greedy(w, x)[0])
        if a != phase and green >= 10:
            phase, green = a, 2
            t += 5
        else:
            green += 5
            t += 5
    return None


def main(path):
    w = tools.load(path)
    worst, starved, total = 0, [], 0
    cases = [{ln: k} for ln in LANES for k in (1, 2, 3)]
    cases += [{a: 1, b: 1} for a, b in itertools.combinations(LANES, 2)]
    for case in cases:
        for p in range(4):
            for g in (12, 60):
                total += 1
                r = run(w, case, p, g)
                if r is None:
                    starved.append((case, p, g))
                else:
                    worst = max(worst, r)
    print(f"{path}: {total} cases, starved {len(starved)}, worst time-to-serve {worst} s")
    for s in starved[:15]:
        print("   starved:", s)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
