import sys; sys.path.insert(0, ".")
import numpy as np, tools
S = tools.states()
X = S["obs"].astype(float); act = S["act"].astype(int); rs = S["rule_state"]
ep = S["episode"]; ts = S["time_s"]
w = tools.load("start_weights.npz")
L = tools.forward(w, X)
ss = rs[:, 0:4]; legal = rs[:, 4:8] > 0.5; overdue = rs[:, 8:12] > 0.5; ph = rs[:, 12:16]; tg = rs[:, 16]; g = rs[:, 17].astype(int)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
m = ep == 0
idx = np.where(m)[0][:120]
for i in idx:
    print(ts[i], "g", g[i], "tg", tg[i], "act", act[i], "lg", L[i], "ss", ss[i], "ph", ph[i], "leg", legal[i].astype(int), "od", overdue[i].astype(int))
