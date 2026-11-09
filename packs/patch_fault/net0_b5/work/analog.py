"""Analog test of imputation schemes.

The recorded states have real values for the N, S and E pockets. Hide one of them (N2, S2 or E2),
rebuild it from other inputs with a hand-written scheme, and measure how often the start network's
decision with the rebuilt value matches its decision with the real value (honoured states only).
This is only used to compare a handful of hand-designed schemes; no weights are fitted.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools  # noqa: E402

S = tools.states()
X = S["obs"].astype(float)
n = tools.names()
I = {k: i for i, k in enumerate(n)}
w = tools.load("start_weights.npz")
ok = X[:, 44] > 0.5

# target pocket, opposite pocket, own upstream lane, phase that serves the pocket
CASES = [("N2", "S2", "N1u", 1), ("S2", "N2", "S1u", 1), ("E2", None, "E1u", 3)]


def scheme(name, Y, T, O, U, ph, rate):
    qO = Y[:, I["q_" + O]] if O else None
    dO = Y[:, I["d_" + O]] if O else None
    qU = Y[:, I["q_" + U]]
    tg = Y[:, 45]
    p = Y[:, 40 + ph]
    if name == "zero":
        return 0 * tg, 0 * tg
    if name == "const":
        return 0.3 + 0 * tg, 0.35 + 0 * tg
    if name == "copy":
        return qO, dO
    if name == "half":
        return 0.15 + 0.5 * qO, 0.2 + 0.5 * dO
    if name == "copy+spill":
        return qO + 5 * qU, dO + 5 * qU
    if name == "half+spill":
        return 0.15 + 0.5 * qO + 5 * qU, 0.2 + 0.5 * dO + 5 * qU
    if name == "half+spill+tg":
        return 0.05 + 0.5 * qO + 5 * qU + 0.5 * rate * tg, 0.1 + 0.5 * dO + 5 * qU + 0.5 * rate * tg
    if name == "half+spill+tg-ph":
        return (0.05 + 0.5 * qO + 5 * qU + 0.5 * rate * tg - 0.1 * p,
                0.1 + 0.5 * dO + 5 * qU + 0.5 * rate * tg - 0.1 * p)
    if name == "spill+tg":
        return 0.15 + 5 * qU + rate * tg, 0.2 + 5 * qU + rate * tg
    raise KeyError(name)


def run(names, clip):
    for T, O, U, ph in CASES:
        if O is None:
            continue
        rate = 0.6  # N/S left-turn arrival rate relative to E/W
        for c in [0, 1, 2, None]:
            m = ok if c is None else ok & (S["controller"] == c)
            Y = X[m]
            ref = tools.greedy(w, Y)
            row = []
            for nm in names:
                q, d = scheme(nm, Y, T, O, U, ph, rate)
                if clip:
                    q, d = np.clip(q, 0, 1), np.clip(d, 0, 1)
                Z = Y.copy()
                Z[:, I["q_" + T]] = q
                Z[:, I["d_" + T]] = d
                row.append((tools.greedy(w, Z) == ref).mean())
            print(T, "ctrl", c, " ".join(f"{nm}={v:.3f}" for nm, v in zip(names, row)))


if __name__ == "__main__":
    names = ["zero", "const", "copy", "half", "copy+spill", "half+spill", "half+spill+tg", "half+spill+tg-ph",
             "spill+tg"]
    print("unclipped")
    run(names, False)
    print("clipped")
    run(names, True)
