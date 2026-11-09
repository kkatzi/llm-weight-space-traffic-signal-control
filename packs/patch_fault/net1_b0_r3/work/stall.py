"""Count 'wasted holds': ok=1, the net keeps the current phase although that phase's stop-line lanes are empty
and some other phase has halting vehicles."""
from common import *

SERVE_D = {0: [12, 13, 18, 19], 1: [14, 20], 2: [15, 16, 21, 22], 3: [17, 23]}
SERVE_Q = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}


def wasted(w, Xs, mirror=False):
    Xe = impute(Xs) if mirror else Xs
    ph = Xs[:, 40:44].argmax(1)
    g = tools.greedy(w, Xe)
    ok = Xs[:, 44] > 0.5
    own_empty = np.array([Xe[i, SERVE_D[ph[i]]].sum() == 0 for i in range(len(Xs))])
    other_q = np.array([sum(Xe[i, SERVE_Q[p]].sum() for p in range(4) if p != ph[i]) for i in range(len(Xs))])
    m = ok & own_empty & (other_q > 0)
    return m, (g == ph) & m


if __name__ == "__main__":
    for mirror in [False, True]:
        m, wh = wasted(W0, X, mirror)
        print("mirror" if mirror else "orig  ", "eligible", m.sum(), "wasted holds", wh.sum(),
              "by ctrl", [int((wh & (C == c)).sum()) for c in range(3)],
              "by phase", np.bincount(X[wh, 40:44].argmax(1), minlength=4))
