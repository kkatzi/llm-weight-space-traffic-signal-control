"""Frozen-queue decision chains on sparse synthetic states (no traffic dynamics: the waiting vehicles are held fixed
while the phase changes) to detect permanent holds: does the net ever reach a phase that serves the waiting lanes?"""
from common import *
import itertools

SERVE = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2, 5: 3, 6: 0, 7: 0, 8: 1, 9: 2, 10: 2, 11: 3}


def state(phase, tg, ok, cells):
    x = np.zeros(46)
    x[40 + phase] = 1; x[44] = ok; x[45] = tg
    for l, k in cells.items():
        x[l] = k / 3; x[12 + l] = k / 3
    return x


def chain(w, start_phase, cells, mirror=True, steps=40):
    """Follow decisions with frozen queues. Returns ('served', t) or ('stuck', phase)."""
    need = {SERVE[l] for l in cells}
    phase, tg = start_phase, 12.0  # green already 12 s old
    for t in range(steps):
        x = state(phase, min(tg / 60, 1), 1.0 if tg >= 10 else 0.0, cells)
        if mirror:
            x = impute(x)
        a = int(tools.forward(w, x)[0].argmax())
        if a != phase and tg >= 10:
            phase, tg = a, 2.0
            if phase in need:
                return ("served", t, phase)
        else:
            tg += 5
    return ("stuck", phase, None)


def run(w, mirror=True, verbose=False):
    lanes = [l for l in range(11)]  # W2 (11) is invisible
    bad = []
    n = 0
    for k in [1, 2, 3]:
        for combo in list(itertools.combinations(lanes, 1)) + list(itertools.combinations(lanes, 2)):
            cells = {l: k for l in combo}
            need = {SERVE[l] for l in combo}
            for p in range(4):
                if p in need:
                    continue
                n += 1
                r = chain(w, p, cells, mirror)
                if r[0] == "stuck":
                    bad.append((p, {NAMES[l]: v for l, v in cells.items()}, r[1]))
    if verbose:
        for b in bad:
            print(b)
    print("chains", n, "stuck", len(bad))
    return bad


if __name__ == "__main__":
    print("start net, no imputation:"); run(W0, mirror=False)
    print("start net, mirror imputation:"); run(W0, mirror=True, verbose=True)
