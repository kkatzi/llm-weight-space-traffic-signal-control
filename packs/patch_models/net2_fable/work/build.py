"""Build the edited network: add a 'waiting cars per phase' contrast term to the logits.

logit_q += c * (u_q - mean_q u_q),   u_q = approx. vehicles waiting for phase q (stop-line + upstream lanes)

Implementation (by construction, no fitting):
  * 3 first-layer units (H1_UNITS) are rewritten as linear Hadamard contrasts of u: A=u0+u1-u2-u3, B=u0-u1+u2-u3,
    C=u0-u1-u2+u3, squashed by tanh with slope S per car.
  * 3 second-layer units (H2_UNITS) are rewritten as pass-throughs tanh(K * h1_i).
  * their W3 columns carry +-c/(4*S*K) with the Hadamard signs, so that in the linear regime the logits receive
    c*(u_q - mean u).  b3 absorbs the mean contribution of the 3 removed h2 units.
usage: python work/build.py <c> <out.npz> [--no-b2]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import tools

C = float(sys.argv[1]) if len(sys.argv) > 1 else 0.7
OUT = sys.argv[2] if len(sys.argv) > 2 else "work/weights.npz"
USE_B2 = "--no-b2" not in sys.argv
S_SLOPE = 0.1      # tanh slope per car in layer 1
K = 1.0            # pass-through gain in layer 2
H1_UNITS = [50, 42, 38]
H2_UNITS = [1, 27, 14]
CAP_SL, CAP_UP = 22.4 / 7.5, 256 / 7.5
def _opt(name, default):
    for a in sys.argv:
        if a.startswith(f"--{name}="):
            return float(a.split("=")[1])
    return default
W_UPQ, W_UPD = _opt("upq", 1.0), _opt("upd", 0.5)   # weight of halting upstream vehicles and of all upstream vehicles

LANES = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}   # stop-line lanes per phase
# upstream lanes (0..7 = N0u N1u E0u E1u S0u S1u W0u W1u) -> (phase, share)
UPSHARE = {0: [(0, 1.0)], 1: [(0, 0.65), (1, 0.35)], 2: [(2, 1.0)], 3: [(2, 0.65), (3, 0.35)],
           4: [(0, 1.0)], 5: [(0, 0.65), (1, 0.35)], 6: [(2, 1.0)], 7: [(2, 0.65), (3, 0.35)]}


def demand_matrix():
    """U (4, 46): u = U @ x gives approximate waiting vehicles per phase."""
    U = np.zeros((4, 46))
    for p, lanes in LANES.items():
        for l in lanes:
            U[p, 12 + l] += CAP_SL                     # stop-line density
    for l, shares in UPSHARE.items():
        for p, sh in shares:
            U[p, 24 + l] += W_UPQ * sh * CAP_UP        # upstream halting
            U[p, 32 + l] += W_UPD * sh * CAP_UP        # upstream density
    return U


HAD = np.array([[1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]], float)   # rows: A, B, C ; columns: phases


def build(c=C, use_b2=USE_B2, verbose=True):
    w0 = tools.load("start_weights.npz")
    w = {k: v.copy() for k, v in w0.items()}
    S = tools.states()
    X = S["obs"]
    h1o, h2o = tools.hidden(w0, X)
    U = demand_matrix()
    # layer 1: contrast units
    for i, row in zip(H1_UNITS, HAD):
        w["W1"][i, :] = S_SLOPE * (row @ U)
        w["b1"][i] = 0.0
    # layer 2: pass-through units
    for j, i in zip(H2_UNITS, H1_UNITS):
        w["W2"][j, :] = 0.0
        w["W2"][j, i] = K
        w["b2"][j] = 0.0
    # output: Hadamard synthesis, b3 absorbs the removed units' mean contribution
    gain = c / (4 * S_SLOPE * K)
    for j in H2_UNITS:
        w["b3"] += w0["W3"][:, j] * h2o[:, j].mean()
    for j, row in zip(H2_UNITS, HAD):
        w["W3"][:, j] = gain * row
    # optional: absorb the mean input shift of the rewritten h1 units into b2 of the most affected other units
    if use_b2:
        h1n = np.tanh(X @ w["W1"].T + w["b1"])
        shift = (h1n[:, H1_UNITS] - h1o[:, H1_UNITS]).mean(0)           # (3,)
        others = [k for k in range(64) if k not in H2_UNITS]
        dz = w0["W2"][others][:, H1_UNITS] @ shift                        # mean pre-activation shift per h2 unit
        order = np.argsort(-np.abs(dz))
        budget = 371 - tools.edit_size(w, w0)["changed"]
        for idx in order[:budget]:
            w["b2"][others[idx]] -= dz[idx]
    if verbose:
        print("c =", c, " edit size:", tools.edit_size(w, w0))
        check(w, w0, X, U)
    return w


def check(w, w0, X, U):
    ok = X[:, 44] > 0.5
    phase = X[:, 40:44].argmax(1)
    g0, g = tools.greedy(w0, X), tools.greedy(w, X)
    u = X @ U.T
    print("  decision flips vs original (honoured states):", round((g[ok] != g0[ok]).mean(), 4))
    sw0 = ok & (g0 != phase)
    sw = ok & (g != phase)
    print("  switch rate honoured: original", round(sw0.mean() / ok.mean(), 3), " new", round(sw.mean() / ok.mean(), 3))
    idx = np.arange(len(X))
    for name, gg, m in [("original", g0, sw0), ("new", g, sw)]:
        best = u[m].argmax(1)
        forgone = u[m].max(1) - u[m][np.arange(m.sum()), gg[m]]
        print(f"  {name}: chosen==argmax cars {np.mean(gg[m] == best):.3f}; cars forgone mean {forgone.mean():.2f}, "
              f"share >=2: {(forgone >= 2).mean():.3f}")
    # stays where original switched: how much demand is left on current lanes
    st = ok & (g == phase) & (g0 != phase)
    print("  new stays where original switched:", int(st.sum()), " mean cars on current phase there:",
          round(u[st][np.arange(st.sum()), phase[st]].mean(), 2) if st.any() else None)
    print("  trainability:", {k: round(v, 3) if isinstance(v, float) else v for k, v in tools.trainability(w).items()})


if __name__ == "__main__":
    w = build()
    tools.save(w, OUT)
    print("saved", OUT)
