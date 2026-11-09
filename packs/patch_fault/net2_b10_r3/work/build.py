"""build.py: construct repaired weights by hand (no optimiser, no regression).

Idea
----
The W2 detector (x[11]=q_W2, x[23]=d_W2) reads 0. We replace its contribution to every layer-1
pre-activation by a *proxy* of what it would have read:

  1. Shrunk-symmetry proxy (linear, folded into existing columns):
        W2_hat = rho * E2 + c_red            (phases 0,1,2)
        W2_hat = rho * E2 + c_srv            (phase 3, EW_left, which empties both pockets)
     E2 is the east left-turn pocket: same demand, same geometry, emptied by the same phase.
     rho / c_* are taken from statistics of the analogous N2/S2 pair (both detectors working)
     and from the mean E2 occupancy per phase.
     Fold:  W1[:,5]  += rho*W1[:,11]            (q_E2 carries the q_W2 proxy)
            W1[:,17] += rho*W1[:,23]            (d_E2 carries the d_W2 proxy)
            b1       += c_red_q*W1[:,11] + c_red_d*W1[:,23]
            W1[:,43] += (c_srv_q-c_red_q)*W1[:,11] + (c_srv_d-c_red_d)*W1[:,23]

  2. Spillback unit: a halted car on upstream lane W1u outside phase 3 means the W2 pocket is
     full and a left-turner blocks W1u (P ~ 0.96-0.99 for the E-side analogue). A least-important
     layer-1 unit is rewired into the indicator s = tanh(K*q_W1u - K3*phase3 + beta); its old
     contribution is folded into b2 (at its mean); its new outgoing column injects into layer 2
     the mean change of layer-2 pre-activation between "W2 full" (q=d=1) and the proxy, over
     recorded spill states.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
sys.path.insert(0, PACK)
import tools  # noqa: E402

W0 = tools.load(os.path.join(PACK, "start_weights.npz"))
S = tools.states()
X = S["obs"].astype(np.float64)
CTRL = S["controller"]
PH = X[:, 40:44].argmax(1)


def copyw(w):
    return {k: v.copy() for k, v in w.items()}


def fold_proxy(w, rho_q, rho_d, c_red_q, c_red_d, c_srv_q, c_srv_d, merge=False):
    w = copyw(w)
    a11, a23 = W0["W1"][:, 11].copy(), W0["W1"][:, 23].copy()
    if merge:   # both slope terms on the d_E2 column (saves 64 parameters)
        w["W1"][:, 17] += rho_q * a11 + rho_d * a23
    else:
        if rho_q:
            w["W1"][:, 5] += rho_q * a11
        if rho_d:
            w["W1"][:, 17] += rho_d * a23
    if c_red_q or c_red_d:
        w["b1"] += c_red_q * a11 + c_red_d * a23
    dq, dd = c_srv_q - c_red_q, c_srv_d - c_red_d
    if dq or dd:
        w["W1"][:, 43] += dq * a11 + dd * a23
    return w


def spill_states(mask_ctrl=None):
    m = (X[:, 31] > 0) & (PH != 3)
    if mask_ctrl is not None:
        m &= mask_ctrl
    return m


def a2_of(w, Xs):
    h1 = np.tanh(Xs @ w["W1"].T + w["b1"])
    return h1 @ w["W2"].T + w["b2"]


def add_spill(w, j, K=120.0, K3=80.0, beta=-1.75, zero_row=True, ref_mask=None, Kt=0.0):
    """Rewire layer-1 unit j into the W2-spillback indicator.
    s = tanh(K*q_W1u + Kt*t_green - K3*phase3 + beta): on (+1) when a car halts on W1u, or (Kt>0)
    when a green other than phase 3 has lasted ~57-60 s; off (-1) otherwise and always in phase 3."""
    w = copyw(w)
    # (a) fold unit j's old contribution into b2 at its mean over all recorded states
    h1_old = np.tanh(X @ w["W1"].T + w["b1"])[:, j]
    w["b2"] = w["b2"] + w["W2"][:, j] * h1_old.mean()
    # (b) desired injection: mean change in layer-2 pre-activation when W2 is truly full
    m = spill_states(ref_mask)
    Xs = X[m]
    Xfull = Xs.copy()
    Xfull[:, 11] = 1.0                 # what a working detector reads when the pocket spills back
    Xfull[:, 23] = 1.0
    # target: ORIGINAL layer 1 on the true reading; current: PROXY layer 1 on the faulty reading.
    # unit j is excluded from both (its outgoing column is zeroed while measuring).
    wcur = copyw(w)
    wcur["W2"][:, j] = 0.0
    wtgt = copyw(wcur)
    wtgt["W1"], wtgt["b1"] = W0["W1"].copy(), W0["b1"].copy()
    D = (a2_of(wtgt, Xfull) - a2_of(wcur, Xs)).mean(0)
    # (c) new row: indicator of q_W1u > 0 outside phase 3
    if zero_row:
        w["W1"][j, :] = 0.0
        w["b1"][j] = beta
    else:              # keep the other weights; shift the bias so the typical off-state is beta
        pre = X @ w["W1"][j] + w["b1"][j]
        rest = pre - w["W1"][j, 31] * X[:, 31] - w["W1"][j, 43] * X[:, 43] - w["W1"][j, 45] * X[:, 45]
        w["b1"][j] = beta - (rest.mean() - w["b1"][j])
    w["W1"][j, 31] = K
    w["W1"][j, 43] = -K3
    if Kt:
        w["W1"][j, 45] = Kt
    s_off, s_on = np.tanh(beta), np.tanh(K * (1 / 34.13) + beta)
    w["W2"][:, j] = D / (s_on - s_off)
    w["b2"] = w["b2"] - w["W2"][:, j] * s_off
    return w, D


def add_maxgreen3(w, G=50, U=1, Kt=60.0, K3=100.0, beta=-157.0, gain=20.0, v=(-3.0, 0.0, 3.0, -3.0)):
    """Phase-3 max-green: when phase 3 (EW_left) has been green ~57-60 s, vote for phase 2.
    Layer-1 unit G becomes g = tanh(K3*phase3 + Kt*t_green + beta + rest) (rest = its other, small,
    weights: kept to save parameters); layer-2 unit U becomes u = tanh(gain*g + centred rest);
    logits get v*u (offset in b3 so the off-state adds nothing). Old contributions of G and U are
    folded into b2 / b3 at their means over the recorded states."""
    w = copyw(w)
    h1 = np.tanh(X @ w["W1"].T + w["b1"])
    # --- layer-1 unit G
    g_mean_old = h1[:, G].mean()
    pre = X @ w["W1"][G] + w["b1"][G]
    rest = pre - w["W1"][G, 43] * X[:, 43] - w["W1"][G, 45] * X[:, 45]
    w["b1"][G] = beta - (rest.mean() - w["b1"][G])
    w["W1"][G, 43] = K3
    w["W1"][G, 45] = Kt
    # G's old outgoing column is kept: off-state g=-1 contributes -W2[:,G]; fold the difference into b2
    w["b2"] = w["b2"] + w["W2"][:, G] * (g_mean_old + 1.0)
    # --- layer-2 unit U
    h1n = np.tanh(X @ w["W1"].T + w["b1"])
    h2 = np.tanh(h1n @ w["W2"].T + w["b2"])
    u_mean_old = h2[:, U].mean()
    w["b3"] = w["b3"] + w["W3"][:, U] * u_mean_old
    w["W2"][U, G] = 0.0
    restU = h1n @ w["W2"][U] + w["b2"][U]
    w["b2"][U] = w["b2"][U] - restU.mean()
    w["W2"][U, G] = gain
    v = np.asarray(v, dtype=np.float64)
    w["W3"][:, U] = v
    w["b3"] = w["b3"] + v           # off-state u=-1 adds -v; cancel it
    return w


def least_important_units(w, n=5):
    h1 = np.tanh(X @ w["W1"].T + w["b1"])
    g = tools.greedy(w, X)
    ok = X[:, 44] > 0.5
    imp = []
    for j in range(64):
        w2 = copyw(w)
        w2["b2"] = w2["b2"] + w2["W2"][:, j] * h1[:, j].mean()
        w2["W2"][:, j] = 0
        imp.append((tools.greedy(w2, X)[ok] != g[ok]).mean())
    return np.argsort(imp)[:n], np.sort(imp)[:n]


# ----------------------------------------------------------------------------------------------
# candidate definitions (numbers come from work/NOTES.md statistics)
RHO = 0.45
C_RED_Q, C_RED_D = 0.22, 0.25      # (1-rho) * mean E2 occupancy outside phase 3 (q ~0.40, d ~0.46)
C_SRV_Q, C_SRV_D = 0.03, 0.10      # analog N2/S2 'served' intercepts; E2 mean in phase 3 q 0.06, d 0.22


def cand(name):
    if name == "A":     # full symmetric substitution W2 := E2
        return fold_proxy(W0, 1.0, 1.0, 0, 0, 0, 0)
    if name == "B":     # shrunk proxy with phase-dependent intercept
        return fold_proxy(W0, RHO, RHO, C_RED_Q, C_RED_D, C_SRV_Q, C_SRV_D)
    if name == "C":     # B (merged slope column) + spillback unit
        w = fold_proxy(W0, RHO, RHO, C_RED_Q, C_RED_D, C_SRV_Q, C_SRV_D, merge=True)
        return add_spill(w, 42)[0]
    if name == "C2":    # C + long-green trigger on the same unit (anti-starvation), same parameter count
        w = fold_proxy(W0, RHO, RHO, C_RED_Q, C_RED_D, C_SRV_Q, C_SRV_D, merge=True)
        return add_spill(w, 42, K=1400.0, K3=1500.0, beta=-37.5, Kt=40.0)[0]
    if name == "C3":    # C2 with the spill row kept (saves 43) + phase-3 max-green gadget (+10)
        w = fold_proxy(W0, RHO, RHO, C_RED_Q, C_RED_D, C_SRV_Q, C_SRV_D, merge=True)
        w = add_spill(w, 42, K=2400.0, K3=2600.0, beta=-66.0, Kt=80.0, zero_row=False)[0]
        return add_maxgreen3(w, Kt=80.0, K3=100.0, beta=-166.0)
    if name.startswith("V"):   # C3 structure with other proxy numbers: V<rho*100>
        rho = int(name[1:]) / 100.0
        mq, md = 0.34, 0.39          # E2 mean outside phase 3 when E1u has no queue (start network)
        sq, sd = 0.056, 0.219        # E2 mean during phase 3 (start network)
        w = fold_proxy(W0, rho, rho, (1 - rho) * mq, (1 - rho) * md, (1 - rho) * sq, (1 - rho) * sd, merge=True)
        w = add_spill(w, 42, K=2400.0, K3=2600.0, beta=-66.0, Kt=80.0, zero_row=False)[0]
        return add_maxgreen3(w, Kt=80.0, K3=100.0, beta=-166.0)
    if name == "D":     # A + spillback unit
        w = fold_proxy(W0, 1.0, 1.0, 0, 0, 0, 0)
        return add_spill(w, 42)[0]
    raise KeyError(name)


if __name__ == "__main__":
    for nm in sys.argv[1:]:
        w = cand(nm)
        es = tools.edit_size(w, W0)
        print(nm, es)
        assert es["changed"] <= 371
        tools.save(w, os.path.join(HERE, f"cand_{nm}.npz"))
