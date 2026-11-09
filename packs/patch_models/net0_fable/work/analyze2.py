import numpy as np, tools
np.set_printoptions(precision=2, suppress=True, linewidth=160)
w = tools.load("start_weights.npz")
S = tools.states()
X, A, ep, t, c = S["obs"], S["act"], S["episode"], S["time_s"], S["controller"]
names = tools.names()
m = (c==0)&(ep==2)
Xe, Ae, te = X[m], A[m], t[m]
ph = Xe[:,40:44].argmax(1)
# print samples over time
for i in range(0, len(te), 40):
    x = Xe[i]
    print(f"t={te[i]:5.0f} ph={ph[i]} a={Ae[i]} tg={x[45]*60:4.0f} q={x[:12]} d={x[12:24]} uq={x[24:32]}")
print("---- last 60 decisions")
for i in range(len(te)-60, len(te)):
    x = Xe[i]
    print(f"t={te[i]:5.0f} ph={ph[i]} a={Ae[i]} ok={x[44]:.0f} tg={x[45]*60:4.0f} q={x[:12]} d={x[12:24]}")
