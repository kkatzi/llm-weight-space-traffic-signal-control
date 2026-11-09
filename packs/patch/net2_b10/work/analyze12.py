import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, tools
from collections import Counter
np.set_printoptions(precision=2, suppress=True, linewidth=200)
w = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"].astype(float); ctrl = S["controller"]; ep = S["episode"]; t = S["time_s"]
cap = 22.4/7.5; capu = 256/7.5
q = np.round(X[:, 0:12]*cap).astype(int); qu = np.round(X[:, 24:32]*capu).astype(int)
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
lg = tools.forward(w, X); g = lg.argmax(1)
appr = "NESW"
# upstream lane index: 2*a + j ; j=0 feeds lane 0, j=1 feeds lanes 1,2
for c in range(3):
    print("=== ctrl", S["controller_names"][c])
    for j in [0, 1]:
        cnt = Counter()
        for a in range(4):
            ui = 2*a + j
            m = (ctrl == c) & (qu[:, ui] > 0)
            if j == 0:
                key = [f"L0full={int(v==3)}" for v in q[m, 3*a]]
            else:
                key = [f"L1={l1},L2={l2}" for l1, l2 in zip(q[m, 3*a+1], q[m, 3*a+2])]
            cnt.update(key)
        print(f" upstream lane j={j} halted>0 states: ", cnt.most_common(8))
# net decisions when an upstream-1 lane is halted with full pocket
leftphase = {0: 1, 2: 1, 1: 3, 3: 3}   # approach -> left phase
thrphase = {0: 0, 2: 0, 1: 2, 3: 2}
for c in range(3):
    res = Counter(); n = 0
    for a in range(4):
        ui = 2*a + 1
        m = (ctrl == c) & ok & (qu[:, ui] >= 2) & (q[:, 3*a+2] == 3)
        n += m.sum()
        for i in np.where(m)[0]:
            res["picks left phase" if g[i] == leftphase[a] else ("stays" if g[i] == ph[i] else ("cur is left phase" if ph[i] == leftphase[a] else "picks other"))] += 1
    print("ctrl", c, "honoured states with up1 halted>=2 & pocket full:", n, res)
for c in range(3):
    res = Counter(); n = 0
    for a in range(4):
        ui = 2*a
        m = (ctrl == c) & ok & (qu[:, ui] >= 2)
        n += m.sum()
        for i in np.where(m)[0]:
            res["picks thr phase" if g[i] == thrphase[a] else ("cur is thr" if ph[i] == thrphase[a] else "picks other")] += 1
    print("ctrl", c, "honoured states with up0 halted>=2:", n, res)
