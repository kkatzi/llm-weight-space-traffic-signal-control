"""Checks of work/supervisor.py (no simulator): replay on the recorded states (open loop) + synthetic edge cases."""
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import tools

spec = importlib.util.spec_from_file_location("sup", os.path.join(HERE, "supervisor.py"))
sup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sup)

S = tools.states()
w = tools.load("start_weights.npz")
X = S["obs"].astype(np.float64)
R = S["rule_state"].astype(np.float64)
L = tools.forward(w, X)          # running the network to get its logits, as the harness will
net = L.argmax(1)
RULES = {"min_green_through": 15, "min_green_left": 10, "max_green": 45, "max_wait": 90}


def mkstate(r):
    return {"green": int(r[17]), "t_green": float(r[16]), "since_served": r[0:4].copy(),
            "legal": r[4:8] > 0.5, "overdue": r[8:12] > 0.5, "phase_halting": r[12:16].copy(), "rules": RULES}


def shield(req, st):
    lg = st["legal"]
    if lg[req]:
        return req
    cand = [p for p in range(4) if lg[p]]
    return max(cand, key=lambda p: st["phase_halting"][p])


out = np.array([sup.decide(X[i], L[i], mkstate(R[i])) for i in range(len(X))])
assert set(np.unique(out)) <= {0, 1, 2, 3}
eff_sup = np.array([shield(out[i], mkstate(R[i])) for i in range(len(X))])
eff_net = np.array([shield(net[i], mkstate(R[i])) for i in range(len(X))])
print("decisions:", len(X))
print("supervisor output != network argmax:", int((out != net).sum()), f"({(out != net).mean():.4f})")
print("applied phase differs from naive shield:", int((eff_sup != eff_net).sum()))
legal = R[:, 4:8] > 0.5
print("supervisor outputs illegal (left to shield, only when 1 legal phase):", int((~legal[np.arange(len(X)), out]).sum()),
      " of which >1 legal:", int(((~legal[np.arange(len(X)), out]) & (legal.sum(1) > 1)).sum()))
for i in np.where(out != net)[0]:
    st = mkstate(R[i])
    print(f"  override ep{S['episode'][i]} t={S['time_s'][i]} green={st['green']} tg={st['t_green']:.0f} net={net[i]} "
          f"-> {out[i]}  halt={st['phase_halting']} ss={st['since_served']} L={np.round(L[i], 2)}")

# ---------------- synthetic edge cases ----------------
def base(**kw):
    st = {"green": 2, "t_green": 17.0, "since_served": np.array([40., 30., 0., 20.]),
          "legal": np.array([True, True, True, True]), "overdue": np.zeros(4, bool),
          "phase_halting": np.array([3., 2., 1., 2.]), "rules": RULES}
    st.update(kw)
    return st

x0 = X[100].copy()
# 1) legal network choice is kept
assert sup.decide(x0, np.array([0., 1., 0., 3.]), base()) == 3
# 2) min green: only current legal, network wants other -> returns network choice (shield maps to current)
assert sup.decide(x0, np.array([5., 0., 0., 0.]), base(t_green=12.0, legal=np.array([0, 0, 1, 0], bool))) == 0
# 3) max green: current illegal, network wants current -> best-logit legal phase
st = base(t_green=47.0, legal=np.array([1, 1, 0, 1], bool), phase_halting=np.array([1., 5., 9., 0.]))
assert sup.decide(x0, np.array([0.5, 0.2, 3.0, 1.0]), st) == 3
# 4) two overdue phases legal, network wants a third -> best-logit among the overdue
st = base(legal=np.array([1, 1, 0, 0], bool), overdue=np.array([1, 1, 0, 0], bool),
          since_served=np.array([95., 92., 0., 30.]))
assert sup.decide(x0, np.array([0.1, 0.4, 2.0, 3.0]), st) == 1
# 5) single overdue phase legal -> network choice returned (shield forces the overdue phase)
st = base(legal=np.array([0, 1, 0, 0], bool), overdue=np.array([0, 1, 0, 0], bool),
          since_served=np.array([40., 95., 0., 30.]))
assert sup.decide(x0, np.array([0.1, 0.4, 2.0, 3.0]), st) == 3
# 6) anticipation: phase 0 has ss 85 and 6 halting, network picks left phase 3 with an empty pocket
xa = np.zeros(46); xa[42] = 1.0; xa[44] = 1.0; xa[45] = 17 / 60
st = base(since_served=np.array([85., 30., 0., 20.]), phase_halting=np.array([6., 1., 0., 0.]))
assert sup.decide(xa, np.array([1.0, -1.0, 0.0, 2.0]), st) == 0
# 7) same but phase 3 loaded (2 vehicles on each pocket) -> keep the network
xb = xa.copy(); xb[17] = 2 / sup.LANE_CAP; xb[23] = 2 / sup.LANE_CAP
assert sup.decide(xb, np.array([1.0, -1.0, 0.0, 2.0]), st) == 3
# 8) no anticipation if p would not be overdue by the end of the pick's minimum green
st = base(since_served=np.array([60., 30., 0., 20.]), phase_halting=np.array([6., 1., 0., 0.]))
assert sup.decide(xa, np.array([1.0, -1.0, 0.0, 2.0]), st) == 3
# 9) robustness: lists instead of arrays, rules missing
st = {"green": 0, "t_green": 17, "since_served": [0, 10, 20, 30], "legal": [True] * 4, "overdue": [False] * 4,
      "phase_halting": [0, 1, 2, 3]}
assert sup.decide(list(x0), [0, 0, 1, 0], st) == 2
print("edge cases: all passed")
