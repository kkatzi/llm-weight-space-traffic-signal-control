"""Checks of the transferred network (forward passes only; no simulator, no fitting)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
sys.path.insert(0, PACK)
sys.path.insert(0, HERE)
import tools  # noqa: E402
from build import build, L2_NAMES, NS_SCALE_MINIMAX, CAP_L2, CAP_L2W_NS  # noqa: E402

np.set_printoptions(precision=3, suppress=True, linewidth=200)
src = tools.load(os.path.join(PACK, "source_weights_L2.npz"))
L2 = tools.states("states_L2.npz")
XL2 = L2["obs"].astype(np.float64)
S = tools.states()
X = S["obs"].astype(np.float64)
ctrl = S["controller"]
W = tools.names()
wi = {n: i for i, n in enumerate(W)}
li = {n: i for i, n in enumerate(L2_NAMES)}


def lift(XL):
    """An L2 observation re-expressed in L2W's layout under the lane-role reading (used for plumbing checks)."""
    out = np.zeros((len(XL), 54))
    for kind in ("q", "d"):
        for ap in ("N", "S"):
            for k in range(3):
                n = np.round(XL[:, li[f"{kind}_{ap}{k}"]] * CAP_L2)          # vehicle count (<=3 after clip)
                out[:, wi[f"{kind}_{ap}{k}"]] = np.minimum(1.0, n / CAP_L2W_NS)
            for k in range(2):
                out[:, wi[f"{kind}_{ap}{k}u"]] = XL[:, li[f"{kind}_{ap}{k}u"]]
        for ap in ("E", "W"):
            out[:, wi[f"{kind}_{ap}0"]] = XL[:, li[f"{kind}_{ap}0"]]
            out[:, wi[f"{kind}_{ap}1"]] = XL[:, li[f"{kind}_{ap}0"]]
            out[:, wi[f"{kind}_{ap}2"]] = XL[:, li[f"{kind}_{ap}1"]]
            out[:, wi[f"{kind}_{ap}3"]] = XL[:, li[f"{kind}_{ap}2"]]
            out[:, wi[f"{kind}_{ap}0u"]] = XL[:, li[f"{kind}_{ap}0u"]]
            out[:, wi[f"{kind}_{ap}1u"]] = XL[:, li[f"{kind}_{ap}0u"]]
            out[:, wi[f"{kind}_{ap}2u"]] = XL[:, li[f"{kind}_{ap}1u"]]
    for n in ("phase0", "phase1", "phase2", "phase3", "ok", "t_green"):
        out[:, wi[n]] = XL[:, li[n]]
    return out


print("=== 0. frozen layers identical, NS scale", round(NS_SCALE_MINIMAX, 4))
w, M, _ = build("role")
for k in ("W2", "b2", "W3", "b3"):
    assert np.array_equal(w[k], src[k]), k
print("ok")

print("\n=== 1. plumbing: L2 states lifted to L2W layout and run through the L2W net vs the L2 net")
a_src = tools.greedy(src, XL2)
XL2W = lift(XL2)
for s in (CAP_L2W_NS / CAP_L2, NS_SCALE_MINIMAX, 1.0):
    wv, _, _ = build("role", ns_scale=s)
    a = tools.greedy(wv, XL2W)
    ok = XL2[:, 44] > 0.5
    print(f"ns_scale {s:.3f}: agreement all {np.mean(a == a_src):.4f}  honoured(ok=1) {np.mean(a[ok] == a_src[ok]):.4f}")
wexact, _, _ = build("role", ns_scale=1.0)
# with no N/S rescaling AND an exact (unclipped-count-preserving) lift, the mapping must reproduce L2 exactly
XL2W_id = XL2W.copy()
for kind in ("q", "d"):
    for ap in ("N", "S"):
        for k in range(3):
            XL2W_id[:, wi[f"{kind}_{ap}{k}"]] = XL2[:, li[f"{kind}_{ap}{k}"]]
print("exact plumbing (identity N/S, identity lift): max |logit diff| =",
      float(np.abs(tools.forward(wexact, XL2W_id) - tools.forward(src, XL2)).max()))

print("\n=== 2. virtual-L2 feature statistics: L2W recorded states mapped by M vs L2 recorded states")
V = X @ M.T
for n in ["q_E0", "q_E1", "q_E2", "d_E0", "d_E1", "d_E2", "q_N0", "q_N2", "d_N0", "d_N2",
          "q_E0u", "q_E1u", "d_E0u", "d_E1u", "d_N0u", "d_N1u"]:
    i = li[n]
    print(f"{n:6s} L2 mean {XL2[:, i].mean():.3f} q90 {np.quantile(XL2[:, i], .9):.3f} | "
          f"L2W->L2 mean {V[:, i].mean():.3f} q90 {np.quantile(V[:, i], .9):.3f}  (fixed {V[ctrl == 0, i].mean():.3f}, actuated {V[ctrl == 1, i].mean():.3f})")

print("\n=== 3. variants on the L2W recorded states")
ok = X[:, wi["ok"]] > 0.5
res = {}
for v in ("role", "interp", "drop", "sum"):
    wv, _, _ = build(v)
    a = tools.greedy(wv, X)
    res[v] = a
    cur = X[:, 48:52].argmax(1)
    sw = (a != cur) & ok
    t = tools.trainability(wv)
    print(f"{v:6s} picks {np.bincount(a, minlength=4) / len(a)}  ok=1 picks {np.bincount(a[ok], minlength=4) / ok.sum()}"
          f"  switch|ok {sw.sum() / ok.sum():.3f}  agree actuated {np.mean(a[ctrl == 1] == S['act'][ctrl == 1]):.3f}"
          f"  median_max_prob {t['median_max_prob']:.3f} sat_l1 {t['sat_l1']:.4f} sat_l2 {t['sat_l2']:.4f}")
for v in ("interp", "drop", "sum"):
    print(f"agreement role vs {v}: all {np.mean(res['role'] == res[v]):.3f}, ok=1 {np.mean(res['role'][ok] == res[v][ok]):.3f}")
cur = XL2[:, 40:44].argmax(1)
okL = XL2[:, 44] > 0.5
print("source on L2 own states: picks", np.bincount(a_src, minlength=4) / len(a_src),
      " ok=1 picks", np.bincount(a_src[okL], minlength=4) / okL.sum(),
      " switch|ok %.3f" % (((a_src != cur) & okL).sum() / okL.sum()))

print("\n=== 4. trainability (delivered variant 'role')")
print("L2W net on L2W states:", tools.trainability(w))
print("source net on L2 states:", tools.trainability(src, XL2))
h1, h2 = tools.hidden(w, X)
g1, g2 = tools.hidden(src, XL2)
print("h1 |act| mean L2W %.3f vs source %.3f; h2 %.3f vs %.3f" % (np.abs(h1).mean(), np.abs(g1).mean(),
                                                                    np.abs(h2).mean(), np.abs(g2).mean()))
pre1 = X @ w["W1"].T + w["b1"]
pre1s = XL2 @ src["W1"].T + src["b1"]
print("layer-1 pre-activation std L2W %.3f vs source %.3f; max |pre| %.2f vs %.2f" % (
    pre1.std(), pre1s.std(), np.abs(pre1).max(), np.abs(pre1s).max()))
print("|W1| max L2W %.3f source %.3f; Frobenius %.2f vs %.2f" % (np.abs(w["W1"]).max(), np.abs(src["W1"]).max(),
                                                               np.linalg.norm(w["W1"]), np.linalg.norm(src["W1"])))
