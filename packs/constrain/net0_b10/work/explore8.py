import sys; sys.path.insert(0, ".")
import numpy as np, tools
S = tools.states()
X = S["obs"].astype(float); act = S["act"].astype(int); rs = S["rule_state"]; ep = S["episode"]; ts = S["time_s"]
w = tools.load("start_weights.npz")
L = tools.forward(w, X)
ss = rs[:, 0:4]; legal = rs[:, 4:8] > 0.5; od = rs[:, 8:12] > 0.5; ph = rs[:, 12:16]; tg = rs[:, 16]; g = rs[:, 17].astype(int)
N = len(X)
ok = X[:, 44] > 0.5
# consistency between request at tg=12 and decision at tg=17 for through greens
agree = []; 
for i in range(N - 1):
    if ok[i] and tg[i] == 12 and g[i] in (0, 2) and ep[i + 1] == ep[i] and tg[i + 1] == 17:
        agree.append(act[i] == act[i + 1])
print("through: request at 12 == request at 17:", np.mean(agree), len(agree))
# forced services due to overdue: the forced phase, its halting, the network's wanted phase and its halting
F = np.where(ok & (legal.sum(1) == 1) & od.any(1) & (act != g) & ~legal[np.arange(N), act])[0]
fp = legal[F].argmax(1)
print("forced overdue services:", len(F), "forced phase dist", np.bincount(fp, minlength=4))
print("  halting on forced phase", np.quantile(ph[F, fp], [.1, .5, .9]), " on wanted phase", np.quantile(ph[F, act[F]], [.1, .5, .9]))
print("  per episode", np.bincount(ep[F]))
# demand level per episode (number of decisions)
print("decisions per episode", np.bincount(ep))
# halting per phase at real decisions, mean
dec = ok & legal.all(1)
print("mean halting per phase at real decisions", ph[dec].mean(0))
print("mean since_served at real decisions", ss[dec].mean(0), "max", ss[dec].max(0))
