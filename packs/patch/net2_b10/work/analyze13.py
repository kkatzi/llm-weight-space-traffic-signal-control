import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, tools
np.set_printoptions(precision=1, suppress=True, linewidth=200)
w = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"].astype(float); ctrl = S["controller"]; ep = S["episode"]; t = S["time_s"]
cap = 22.4/7.5; capu = 256/7.5
q = np.round(X[:, 0:12]*cap).astype(int); qu = np.round(X[:, 24:32]*capu).astype(int); du = np.round(X[:, 32:40]*capu).astype(int)
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
lg = tools.forward(w, X); g = lg.argmax(1)
leftphase = {0: 1, 2: 1, 1: 3, 3: 3}
thrphase = {0: 0, 2: 0, 1: 2, 3: 2}
print("LEFT spillback, picks other (ctrl 0 and 2):")
for c in [0, 2]:
    k = 0
    for a in range(4):
        m = (ctrl == c) & ok & (qu[:, 2*a+1] >= 2) & (q[:, 3*a+2] == 3) & (g != leftphase[a])
        for i in np.where(m)[0]:
            if k < 14:
                print(f" c{c} appr {'NESW'[a]} ph {ph[i]} -> {g[i]} (left={leftphase[a]}) q {q[i]} qu {qu[i]} lg {lg[i]}")
            k += 1
print("THROUGH spillback (up0), picks other:")
for c in [0, 2]:
    k = 0
    for a in range(4):
        m = (ctrl == c) & ok & (qu[:, 2*a] >= 2) & (g != thrphase[a]) & (ph != thrphase[a])
        for i in np.where(m)[0]:
            if k < 10:
                print(f" c{c} appr {'NESW'[a]} ph {ph[i]} -> {g[i]} (thr={thrphase[a]}) q {q[i]} qu {qu[i]} lg {lg[i]}")
            k += 1
