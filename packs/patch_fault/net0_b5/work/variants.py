"""Offline comparison of a few hand-built variants: toy drain replay + how much normal-operation decisions move.

Every variant = W2 estimate folded into layer 1 (design.build) plus optional extra terms on columns that the
W2 estimate already touches or on phase one-hot columns. Nothing is fitted; the numbers are written by hand.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "work"))
import tools  # noqa: E402
import design  # noqa: E402
import drain_sim  # noqa: E402

I = design.I
w0 = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"].astype(float)
dead = np.zeros(len(X), bool)
for c, e in [(0, 0), (0, 5), (2, 5)]:
    dead |= (S["controller"] == c) & (S["episode"] == e) & (S["time_s"] > 3700)
OK = (X[:, 44] > 0.5) & ~dead
NS_Q = ["q_N0", "q_N1", "q_S0", "q_S1"]
NS_D = ["d_N0", "d_N1", "d_S0", "d_S1"]


def make(a=0.65, c_q=0.04, c_d=0.08, k=3.0, tau=0.4, mu3=0.0, use_d=True, use_spill=True,
         ns_tau=0.0, ns_ph0=0.0, ns_d=True):
    aq = {"q_E2": a, "t_green": tau}
    ad = {("d_E2" if use_d else "q_E2"): a, "t_green": tau}
    if not use_d:
        c_d = c_d + 0.04  # d_E2 runs about 0.08 above q_E2 on average
    if use_spill:
        aq["q_W1u"] = k
        ad["q_W1u"] = k
    if mu3:
        aq["phase3"] = -mu3
        ad["phase3"] = -mu3
    w = design.build(c_q, c_d, aq, ad, w0)
    # optional phantom N/S through demand that grows with green time (lone-car timer), cancelled in phase 0
    for nm in NS_Q + (NS_D if ns_d else []):
        w["W1"][:, I["t_green"]] += ns_tau * w0["W1"][:, I[nm]]
        w["W1"][:, I["phase0"]] += -ns_ph0 * w0["W1"][:, I[nm]]
    return w


def report(name, w, ref):
    a = tools.greedy(w, X[OK])
    sc, _ = drain_sim.score(w)
    print(f"{name:28s} edit={tools.edit_size(w, w0)['changed']:3d} drain_stuck={sc['stuck_share']:.3f} "
          f"drain_veh_s={sc['mean_veh_s']:.0f} normal_change_vs_ref={(a != ref).mean():.3f} "
          f"share={np.round(np.bincount(a, minlength=4) / OK.sum(), 3)}")


if __name__ == "__main__":
    base = make()
    ref = tools.greedy(base, X[OK])
    report("start", w0, ref)
    report("A", tools.load(os.path.join(HERE, "work", "w_A.npz")), ref)
    report("C", tools.load(os.path.join(HERE, "work", "w_C.npz")), ref)
    report("base a=.65", base, ref)
    report("base mu3=.2", make(mu3=0.2), ref)
    report("noD mu3=.2", make(mu3=0.2, use_d=False), ref)
    for nt in [0.2, 0.3, 0.4]:
        for p0 in [0.0, nt * 0.6, nt]:
            report(f"noD mu3=.2 ns={nt} p0={p0:.2f}", make(mu3=0.2, use_d=False, ns_tau=nt, ns_ph0=p0), ref)


def make_centered(ns=0.4, center=0.2, ns_d=True, **kw):
    """Same, but the N/S lone-car timer is ns * (t_green - center): zero at the usual 12 s decision."""
    w = make(**kw)
    for nm in NS_Q + (NS_D if ns_d else []):
        w["W1"][:, I["t_green"]] += ns * w0["W1"][:, I[nm]]
        w["b1"] = w["b1"] - ns * center * w0["W1"][:, I[nm]]
    return w
