"""Offline checks of work/supervisor.py on the recorded states (no simulator, no search).

1. every output is a legal phase;
2. where the network's own choice is legal, the supervisor agrees with it except for the documented near-tie rules;
3. counts of each kind of override;
4. robustness to input types / missing keys, and a few hand-made edge cases.
"""
import importlib.util
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import tools

spec = importlib.util.spec_from_file_location("supervisor", os.path.join(HERE, "supervisor.py"))
sup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sup)

S = tools.states()
w = tools.load("start_weights.npz")
X = S["obs"].astype(np.float64)
R = S["rule_state"].astype(np.float64)
lg = tools.forward(w, X)
net = lg.argmax(1)
N = len(X)
RULES = {"min_green_through": 15, "min_green_left": 10, "max_green": 45, "max_wait": 90}


def make_state(i, kind="list"):
    r = R[i]
    conv = (lambda a: [float(v) for v in a]) if kind == "list" else (lambda a: np.array(a))
    return {"green": int(r[17]), "t_green": float(r[16]), "since_served": conv(r[0:4]),
            "legal": [bool(v > 0.5) for v in r[4:8]] if kind == "list" else (r[4:8] > 0.5),
            "overdue": [bool(v > 0.5) for v in r[8:12]] if kind == "list" else (r[8:12] > 0.5),
            "phase_halting": conv(r[12:16]), "rules": RULES}


out = np.array([sup.decide(X[i], lg[i], make_state(i)) for i in range(N)])
out_np = np.array([sup.decide(X[i], lg[i].astype(np.float32), make_state(i, "np")) for i in range(N)])
assert (out == out_np).all(), "list vs numpy state inputs disagree"

legal = R[:, 4:8] > 0.5
green = R[:, 17].astype(int)
print("1. all outputs legal:", bool(legal[np.arange(N), out].all()))

net_legal = legal[np.arange(N), net]
print("2. network choice legal at", int(net_legal.sum()), "decisions; supervisor differs at",
      int((out[net_legal] != net[net_legal]).sum()))

# what the naive shield would do (most halting legal phase when the request is illegal)
ph = R[:, 12:16]
shield = net.copy()
for i in np.where(~net_legal)[0]:
    cand = np.where(legal[i])[0]
    shield[i] = cand[np.argmax(ph[i, cand])]

kinds = Counter()
for i in range(N):
    if out[i] == net[i]:
        kinds["same as network"] += 1
    elif legal[i].sum() == 1:
        kinds["forced (one legal phase)"] += 1
    elif not net_legal[i]:
        kinds["network choice illegal, best legal by logits"] += 1
    else:
        sec = "through->left" if (net[i] % 2 == 0 and out[i] % 2 == 1) else "other"
        od = R[i, 0 + out[i]] + (20 if net[i] % 2 == 0 else 15) >= 90
        kinds[f"near-tie override ({'soon-overdue' if od and sec != 'through->left' else sec})"] += 1
print("3. decision kinds:")
for k, v in kinds.most_common():
    print(f"   {k}: {v} ({v / N:.1%})")
print("   differs from network overall:", f"{(out != net).mean():.1%}",
      " | differs at decisions with >1 legal phase:", f"{(out != net)[legal.sum(1) > 1].mean():.1%}")
print("   supervisor vs naive shield executed choice differs at", int((out != shield).sum()), "decisions")
ok = X[:, 44] > 0.5
print("   differs from network where a switch would be honoured (ok=1):", f"{(out != net)[ok].mean():.1%}")

# list the near-tie overrides
print("near-tie overrides:")
for i in np.where((out != net) & net_legal)[0][:30]:
    print(f"   ep{S['episode'][i]} t{S['time_s'][i]} g{green[i]} tg{R[i,16]:.0f} net{net[i]} -> {out[i]} "
          f"lg{np.round(lg[i], 2)} since{R[i,0:4].astype(int)} halt{ph[i].astype(int)}")

# 4. robustness
x0, l0 = X[100], lg[100]
assert sup.decide(x0, l0, {}) == int(np.argmax(l0))
assert sup.decide(x0, l0, None) == int(np.argmax(l0))
assert sup.decide(x0, list(l0), make_state(100)) == out[100]
st = make_state(100); st["rules"] = None
assert sup.decide(x0, l0, st) == out[100]
# max green: current illegal, others legal -> best legal by logits
st = {"green": 2, "t_green": 47, "since_served": [40, 50, 0, 30], "phase_halting": [3, 1, 0, 2],
      "legal": [True, True, False, True], "overdue": [False] * 4, "rules": RULES}
assert sup.decide(x0, np.array([0.0, -3.0, 5.0, 1.0]), st) == 3
# two overdue phases legal -> network's preference among them
st = {"green": 0, "t_green": 17, "since_served": [0, 95, 92, 60], "phase_halting": [0, 1, 6, 2],
      "legal": [False, True, True, False], "overdue": [False, True, True, False], "rules": RULES}
assert sup.decide(x0, np.array([5.0, 2.0, 1.0, 9.0]), st) == 1
# min green: only current legal
st = {"green": 0, "t_green": 12, "since_served": [0, 30, 20, 10], "phase_halting": [0, 3, 6, 2],
      "legal": [True, False, False, False], "overdue": [False] * 4, "rules": RULES}
assert sup.decide(x0, np.array([-5.0, 2.0, 7.0, 1.0]), st) == 0
# near tie: through top, left within DELTA with queue -> left
st = {"green": 1, "t_green": 12, "since_served": [30, 0, 40, 20], "phase_halting": [4, 0, 6, 3],
      "legal": [True] * 4, "overdue": [False] * 4, "rules": RULES}
assert sup.decide(x0, np.array([1.0, -4.0, 3.0, 2.5]), st) == 3
# ... but not if the through phase would go overdue meanwhile
st["since_served"] = [30, 0, 80, 20]
assert sup.decide(x0, np.array([1.0, -4.0, 3.0, 2.5]), st) == 2
# soon-overdue alternative within DELTA is served now
st = {"green": 3, "t_green": 12, "since_served": [40, 75, 20, 0], "phase_halting": [5, 2, 4, 0],
      "legal": [True] * 4, "overdue": [False] * 4, "rules": RULES}
assert sup.decide(x0, np.array([3.0, 2.2, 1.0, -2.0]), st) == 1
print("4. robustness and edge cases: ok")
