import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, tools
np.set_printoptions(precision=2, suppress=True, linewidth=200)
w = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"].astype(float); ctrl = S["controller"]
names = tools.names()
ok = X[:, 44] > 0.5; ph = X[:, 40:44].argmax(1)
lg = tools.forward(w, X); g = lg.argmax(1)
m3 = ok & (ph == 3)
st = m3 & (g == 3); sw = m3 & (g != 3)
print("phase-3 honoured: stays", st.sum(), "switches", sw.sum())
diff = X[st].mean(0) - X[sw].mean(0)
o = np.argsort(-np.abs(diff))
for i in o[:14]:
    print(f"  {names[i]:7s} stay {X[st, i].mean():.3f} switch {X[sw, i].mean():.3f}")
print("t_green in stays:", np.unique(np.round(X[st, 45]*60), return_counts=True))
print("t_green in switches:", np.unique(np.round(X[sw, 45]*60), return_counts=True))
# through-blocked vs pocket-blocked upstream lane1 and whether X0u halted too
cap = 22.4/7.5; capu = 256/7.5
q = np.round(X[:, 0:12]*cap).astype(int); qu = np.round(X[:, 24:32]*capu).astype(int)
for c in range(3):
    tb = 0; tb0 = 0; pb = 0
    for a in range(4):
        m = (ctrl == c) & (qu[:, 2*a+1] > 0)
        tbm = m & (q[:, 3*a+1] == 3) & (q[:, 3*a+2] < 3)
        tb += tbm.sum(); tb0 += (tbm & (qu[:, 2*a] > 0)).sum()
        pb += (m & (q[:, 3*a+2] == 3)).sum()
    print("ctrl", c, "X1u halted: pocket-full", pb, " lane1-full&pocket-not", tb, " of which X0u halted too", tb0)
