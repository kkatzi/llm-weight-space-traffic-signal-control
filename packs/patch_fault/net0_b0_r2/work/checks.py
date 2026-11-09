"""Checks for candidate repairs. Forward passes only (recorded + synthetic states); no fitting."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import tools  # noqa: E402
from build import build  # noqa: E402

np.set_printoptions(linewidth=200, precision=3, suppress=True)
w0 = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"].astype(float)
ctrl, ep, t = S["controller"], S["episode"], S["time_s"]
ok = X[:, 44] > 0.5
CAP_U = 256 / 7.5


def impute(Xin, A, G, C, mirror=1.0):
    Z = Xin.copy()
    extra = A * Z[:, 45] - G * Z[:, 43] + C * Z[:, 31]
    Z[:, 11] = mirror * Z[:, 5] + extra
    Z[:, 23] = mirror * Z[:, 17] + extra
    return Z


def empty(ph, tg, quW1=0.0, e2=0.0):
    x = np.zeros(46)
    x[40 + ph] = 1
    x[44] = 1.0 if tg >= 10 / 60 else 0.0
    x[45] = tg
    x[31] = x[39] = quW1
    x[5] = x[17] = e2
    return x


def report(A, C=0.0, mirror=1.0, verbose=True):
    G = 0.2 * A
    w = build(A=A, C=C, mirror=mirror)
    es = tools.edit_size(w, w0)
    # 1. equivalence of the weight edit with input imputation
    eq = np.abs(tools.forward(w, X) - tools.forward(w0, impute(X, A, G, C, mirror))).max()
    a0, a1 = tools.greedy(w0, X), tools.greedy(w, X)
    # 2. deadlock tails of start_network episodes 0 and 5 (t > 3700)
    dl = t > 3700  # post-demand tails (start_network eps 0,5 and actuated ep 5 are stuck there)
    dead_pick = np.bincount(a1[dl], minlength=4)
    # 3. synthetic empty junction: switch time out of phase 2 (first t_green at which it picks 3)
    def first_switch(ph, quW1=0.0):
        for k in range(2, 61):
            tg = min(1.0, (2 + 5 * (k - 2)) / 60)
            x = empty(ph, tg, quW1)
            if x[44] > 0 and tools.greedy(w, x)[0] != ph:
                return round(tg * 60), int(tools.greedy(w, x)[0])
        return None
    sw = {ph: first_switch(ph) for ph in range(4)}
    sw_q = first_switch(2, 6 / CAP_U)
    # 4. phase-3 lock: phase 3, honoured, any t_green, E2 empty -> must leave
    lock = []
    for tg in [0.2, 0.3, 0.5, 0.8, 1.0]:
        for e2 in [0.0, 1 / 3]:
            lock.append(int(tools.greedy(w, empty(3, tg, 0.0, e2))[0]))
    # 5. recorded states: decisions changed at honoured states, by controller; pick distribution
    rows = []
    for ci in range(3):
        m = (ctrl == ci) & ok & ~dl
        rows.append((S["controller_names"][ci], round(float((a0[m] != a1[m]).mean()), 3),
                     np.bincount(a0[m], minlength=4), np.bincount(a1[m], minlength=4)))
    tr = tools.trainability(w)
    if verbose:
        print(f"=== A={A} G={G:.2f} C={C} mirror={mirror}  changed={es['changed']}  equiv_err={eq:.2e}")
        print("  deadlock-tail picks (0..3):", dead_pick)
        print("  empty junction: first honoured switch (green s, new phase) from each phase:", sw,
              " | phase 2 with 6 halting on W1u:", sw_q)
        print("  phase-3 lock check (picks; 3 = stays):", lock)
        for r in rows:
            print(f"  {r[0]:14s} changed {r[1]:.3f}  picks before {r[2]}  after {r[3]}")
        print("  trainability:", {k: round(v, 3) for k, v in tr.items()})
    return w


def surrogate(A, mirror=1.0):
    """N/S analogue: hide S2 (served by phase 1, like W2 by phase 3) and impute it from N2 with the
    same construction; agreement with the full-information decisions at honoured states."""
    G = 0.2 * A
    full = tools.greedy(w0, X)
    Z = X.copy()
    extra = A * Z[:, 45] - G * Z[:, 41]
    Z[:, 8] = mirror * Z[:, 2] + extra
    Z[:, 20] = mirror * Z[:, 14] + extra
    Z0 = X.copy(); Z0[:, 8] = 0; Z0[:, 20] = 0
    imp, zer = tools.greedy(w0, Z), tools.greedy(w0, Z0)
    out = []
    for ci in range(3):
        m = (ctrl == ci) & ok & (t <= 3700)  # leave out the stuck post-demand tails
        out.append((S["controller_names"][ci], round(float((zer[m] == full[m]).mean()), 3),
                    round(float((imp[m] == full[m]).mean()), 3)))
    return out


if __name__ == "__main__":
    print("faulty start network on recorded data, deadlock tails picks:",
          np.bincount(tools.greedy(w0, X)[(ctrl == 0) & np.isin(ep, [0, 5]) & (t > 3700)], minlength=4))
    for A, C in [(0.0, 0.0), (0.4, 0.0), (0.5, 0.0), (0.6, 0.0), (1.0, 0.0), (0.6, 2.0)]:
        report(A, C)
    print("\nN/S surrogate (agreement with full-info decisions: zero-imputed vs constructed)")
    for A in [0.0, 0.5, 1.0]:
        print(" A =", A, surrogate(A))
