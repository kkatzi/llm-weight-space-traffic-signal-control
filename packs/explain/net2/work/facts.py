"""Facts about the network's behaviour quoted in EXPLANATION.md."""
from common import *
m0 = OK & (CTRL == 0)
print("network's own runs: honoured decisions", m0.sum(), " extensions (picks current phase):", (m0 & (A == PH)).sum(),
      " by phase", [int((m0 & (A == PH) & (PH == p)).sum()) for p in range(4)])
print("all recorded honoured states: extensions", (OK & (A == PH)).sum(), "of", OK.sum())
tg = np.round(X[m0, 45]*60)
print("green age at honoured decisions in network runs:", dict(zip(*np.unique(tg, return_counts=True))))
T = np.zeros((4, 4), int)
for p, q in zip(PH[m0], A[m0]): T[p, q] += 1
print("transition counts current->next (network runs)"); print(T)
print("row shares"); print((T / T.sum(1, keepdims=True)).round(2))
# synthetic: network with all stop-line lanes holding k halting cars per lane on one group, others empty
GR = {0: [0,1,6,7], 1: [2,8], 2: [3,4,9,10], 3: [5,11]}
def make(p, lv):
    x = np.zeros(46); x[40+p] = 1; x[44] = 1; x[45] = 12/60; x[32:40] = 0.05
    for g, l in lv.items():
        for i in GR[g]: x[i] = l/3; x[12+i] = l/3
    return x
print("main-road priority check: after EW-left (phase 3), NS-through lanes with 3 halting cars each, EW-through k cars each")
for k in range(4):
    print("  EW-through cars per lane", k, "-> network picks", tools.greedy(w, make(3, {0: 3, 2: k}))[0])
print("after NS-left (phase 1), NS-through 3 cars/lane, EW-through k cars/lane")
for k in range(4):
    print("  k", k, "->", tools.greedy(w, make(1, {0: 3, 2: k}))[0])
print("after EW-through (phase 2), NS-through k cars/lane, NS-left 2 cars/lane, EW-left 2 cars/lane")
for k in range(4):
    print("  k", k, "->", tools.greedy(w, make(2, {0: k, 1: 2, 3: 2}))[0])
import importlib.util
def load(path):
    spec = importlib.util.spec_from_file_location("r", path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m.decide
small, full = load("rules_small.py"), load("rules_full.py")
print("same synthetic probes, network / small / full:")
for p, lvset in [(3, [{0: 3, 2: k} for k in range(4)]), (1, [{0: 3, 2: k} for k in range(4)]), (2, [{0: k, 1: 2, 3: 2} for k in range(4)])]:
    print(" phase", p, [(tools.greedy(w, make(p, lv))[0], small(make(p, lv)), full(make(p, lv))) for lv in lvset])
# synthetic grid agreement (uniform levels per group, as in explore20)
agree = {"small": 0, "full": 0}; n = 0
for p in range(4):
    others = [j for j in range(4) if j != p]
    for l1 in range(4):
        for l2 in range(4):
            for l3 in range(4):
                x = make(p, {others[0]: l1, others[1]: l2, others[2]: l3}); a = tools.greedy(w, x)[0]
                agree["small"] += small(x) == a; agree["full"] += full(x) == a; n += 1
print("agreement on the 256 uniform-level synthetic grid states:", {k: round(v/n, 3) for k, v in agree.items()})
