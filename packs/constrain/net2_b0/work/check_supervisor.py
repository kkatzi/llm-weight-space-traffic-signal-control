"""Checks of work/supervisor.py on the recorded states and on hand-made edge cases.

Single-step replay only: after a different decision the real trajectory would differ, so this checks legality and
counts where/why the supervisor departs from the network (and from the shield's outcome); it does not estimate TTS.
"""
import ast
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import tools  # noqa: E402
import supervisor  # noqa: E402

S = tools.states()
w = tools.load("start_weights.npz")
X = S["obs"].astype(np.float64)
LG = tools.forward(w, X)
rs = S["rule_state"].astype(np.float64)
N = len(X)
assert (LG.argmax(1) == S["act"]).all()


def mk_state(i):
    return {"green": int(rs[i, 17]), "t_green": float(rs[i, 16]), "since_served": rs[i, 0:4].copy(),
            "legal": rs[i, 4:8] > 0.5, "overdue": rs[i, 8:12] > 0.5, "phase_halting": rs[i, 12:16].copy(),
            "rules": {}}


def shield(req, st):
    if st["legal"][req]:
        return req
    lp = [p for p in range(4) if st["legal"][p]]
    return max(lp, key=lambda p: st["phase_halting"][p])


out = np.zeros(N, int); applied_naive = np.zeros(N, int); reason = []
for i in range(N):
    st = mk_state(i)
    a = int(supervisor.decide(X[i], LG[i], st))
    assert a in (0, 1, 2, 3)
    out[i] = a
    applied_naive[i] = shield(int(LG[i].argmax()), st)
legal = rs[:, 4:8] > 0.5
net = LG.argmax(1)
print("output legal in all states:", bool(legal[np.arange(N), out].all()))
print("differs from network argmax: %d / %d (%.3f)" % ((out != net).sum(), N, (out != net).mean()))
multi = legal.sum(1) >= 2
print("  of which with >=2 legal phases (consequential): %d (%.3f of all)" % (((out != net) & multi).sum(),
                                                                            ((out != net) & multi).mean()))
print("differs from naive (network + shield) applied phase: %d (%.4f), per episode %.2f"
      % ((out != applied_naive).sum(), (out != applied_naive).mean(), (out != applied_naive).sum() / 6))
green = rs[:, 17].astype(int)
thr = np.isin(net, [0, 2])
for i in np.where(out != applied_naive)[0]:
    kind = "max-wait early service" if out[i] not in (net[i],) and (rs[i, 0 + out[i]] >= 60) else "other"
    if out[i] in (1, 3) and net[i] in (0, 2) and kind == "other":
        kind = "through->left near-tie"
    reason.append(kind)
print("  reasons:", Counter(reason))
for i in np.where(out != applied_naive)[0][:12]:
    print("   i=%d green=%d tg=%d net=%d out=%d gap=%.2f since=%s halt=%s" %
          (i, green[i], rs[i, 16], net[i], out[i], LG[i, net[i]] - LG[i, out[i]], rs[i, 0:4].astype(int),
           rs[i, 12:16].astype(int)))

# --- hand-made edge cases -------------------------------------------------------------------------------------
def st(green, tg, since, legal, overdue, halt):
    return {"green": green, "t_green": tg, "since_served": list(since), "legal": list(legal),
            "overdue": list(overdue), "phase_halting": list(halt), "rules": {}}

x0 = np.zeros(46)
# max green: current (2) illegal, others legal; network wants to stay -> its best other phase, not most halting
s = st(2, 47, [40, 30, 0, 20], [1, 1, 0, 1], [0, 0, 0, 0], [9, 1, 12, 2])
r = supervisor.decide(x0, np.array([1.0, 0.5, 3.0, 2.0]), s)
print("max green, net prefers 3 among legal:", r, "(shield would give 0)"); assert r == 3
# minimum green: only current legal
s = st(0, 7, [0, 20, 30, 40], [1, 0, 0, 0], [0, 0, 0, 0], [3, 2, 8, 4])
r = supervisor.decide(x0, np.array([0.0, 0.0, 5.0, 0.0]), s); print("min green:", r); assert r == 0
# single overdue phase
s = st(3, 12, [60, 95, 20, 0], [0, 1, 0, 0], [0, 1, 0, 0], [5, 2, 6, 1])
r = supervisor.decide(x0, np.array([0.0, -2.0, 5.0, 0.0]), s); print("overdue:", r); assert r == 1
# two overdue phases legal -> network's pick among them
s = st(3, 12, [92, 95, 20, 0], [1, 1, 0, 0], [1, 1, 0, 0], [5, 2, 6, 1])
r = supervisor.decide(x0, np.array([1.0, 2.0, 5.0, 0.0]), s); print("two overdue:", r); assert r == 1
# nothing legal (should not happen) -> network
s = st(3, 12, [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0])
r = supervisor.decide(x0, np.array([1.0, 2.0, 5.0, 0.0]), s); print("none legal:", r); assert r == 2
# max-wait anticipation: q=1 at 75 s, net wants through 2 (H=20) with 1 logit... gap 0.5, halt q 4 vs r 4
s = st(3, 12, [30, 75, 40, 0], [1, 1, 1, 1], [0, 0, 0, 0], [3, 4, 4, 1])
r = supervisor.decide(x0, np.array([0.0, 2.5, 3.0, -1.0]), s); print("anticipate max wait:", r); assert r == 1
# same but large gap -> network
r = supervisor.decide(x0, np.array([0.0, 0.5, 3.0, -1.0]), s); print("  large gap:", r); assert r == 2
# same but q not at risk (since 50) -> through near-tie rule may apply instead (left 1 within 0.5, halt>0)
s = st(3, 12, [30, 50, 40, 0], [1, 1, 1, 1], [0, 0, 0, 0], [3, 4, 4, 1])
r = supervisor.decide(x0, np.array([0.0, 2.6, 3.0, -1.0]), s); print("through near-tie:", r); assert r == 1
r = supervisor.decide(x0, np.array([0.0, 2.0, 3.0, -1.0]), s); print("  not a tie:", r); assert r == 2
# through near-tie but through phase itself close to max wait -> keep through
s = st(3, 12, [30, 50, 80, 0], [1, 1, 1, 1], [0, 0, 0, 0], [3, 4, 4, 1])
r = supervisor.decide(x0, np.array([0.0, 2.8, 3.0, -1.0]), s); print("  through at risk:", r); assert r == 2

# --- imports used by supervisor.py -----------------------------------------------------------------------------
tree = ast.parse(open(os.path.join(HERE, "supervisor.py")).read())
mods = set()
for n in ast.walk(tree):
    if isinstance(n, ast.Import):
        mods |= {a.name.split(".")[0] for a in n.names}
    elif isinstance(n, ast.ImportFrom):
        mods.add(n.module.split(".")[0])
print("supervisor imports:", sorted(mods)); assert mods <= {"numpy", "math"}
print("ALL CHECKS PASSED")
