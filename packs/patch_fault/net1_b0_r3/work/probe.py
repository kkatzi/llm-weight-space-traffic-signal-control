"""Synthetic probes: one approach lane holds k vehicles, everything else empty; which phase does the net pick?"""
from common import *

# lane -> serving phase (stop-line lanes 0..11)
SERVE = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2, 5: 3, 6: 0, 7: 0, 8: 1, 9: 2, 10: 2, 11: 3}


def make(phase, tg, lanes, level, ok=1.0):
    x = np.zeros(46)
    x[40 + phase] = 1
    x[44] = ok
    x[45] = tg
    for l in lanes:
        x[l] = level          # queue
        x[12 + l] = level     # density
    return x


def run(w, mirror=False, verbose=True):
    bad = 0; tot = 0; rows = []
    for phase in range(4):
        for tg in [0.2, 0.5, 1.0]:
            for lane in range(12):
                if lane == 11:
                    continue
                for level in [1 / 3, 2 / 3, 1.0]:
                    x = make(phase, tg, [lane], level)
                    if mirror:
                        x = impute(x)
                    lg = tools.forward(w, x)[0]
                    pick = int(lg.argmax())
                    want = SERVE[lane]
                    tot += 1
                    if pick != want:
                        bad += 1
                        rows.append((phase, tg, NAMES[lane], round(level, 2), pick, want, np.round(lg, 2)))
    if verbose:
        for r in rows:
            print(r)
    print("probe mismatches", bad, "of", tot)
    return bad


if __name__ == "__main__":
    run(W0)
