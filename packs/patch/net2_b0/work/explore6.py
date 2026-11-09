import numpy as np, sys
sys.path.insert(0, '.')
import tools
from collections import Counter
np.set_printoptions(linewidth=220, precision=3, suppress=True)
w = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"].astype(np.float64); act = S["act"]; ctl = S["controller"]; ep = S["episode"]; t = S["time_s"]
cap=22.4/7.5; capu=256/7.5
cur = X[:,40:44].argmax(1); ok = X[:,44]>0.5
a = tools.greedy(w, X)
lg = tools.forward(w, X)
def show(i):
    q = (X[i,0:12]*cap).round().astype(int); d=(X[i,12:24]*cap).round().astype(int)
    uq=(X[i,24:32]*capu).round().astype(int); ud=(X[i,32:40]*capu).round().astype(int)
    s = f"t={t[i]:4d} ep{ep[i]} cur{cur[i]} tg{round(X[i,45]*60):2d} pick{a[i]} | "
    for k,ap in enumerate("NESW"):
        s += f"{ap}: q{q[3*k]}{q[3*k+1]}{q[3*k+2]} d{d[3*k]}{d[3*k+1]}{d[3*k+2]} u{uq[2*k]}/{ud[2*k]},{uq[2*k+1]}/{ud[2*k+1]} | "
    s += " lg " + np.array2string(lg[i], precision=1)
    print(s)
m = (ctl==0)&ok&(a==cur)
print("STAYS")
for i in np.where(m)[0]: show(i)
