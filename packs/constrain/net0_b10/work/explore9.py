import sys; sys.path.insert(0, ".")
import numpy as np, tools
S = tools.states()
X = S["obs"].astype(float); act = S["act"].astype(int); rs = S["rule_state"]; ep = S["episode"]; ts = S["time_s"]
w = tools.load("start_weights.npz")
L = tools.forward(w, X)
ss = rs[:, 0:4]; legal = rs[:, 4:8] > 0.5; od = rs[:, 8:12] > 0.5; ph = rs[:, 12:16]; tg = rs[:, 16]; g = rs[:, 17].astype(int)
N = len(X)
np.set_printoptions(precision=1, suppress=True, linewidth=220)
ok = X[:, 44] > 0.5
# real forced overdue services: decision where current green may end (ok & tg >= min), only one legal which is not current
minend = np.where(np.isin(g, [0, 2]), 15, 10)
F = np.where(ok & (tg >= minend) & (legal.sum(1) == 1) & ~legal[np.arange(N), g])[0]
fp = legal[F].argmax(1)
print("forced switches", len(F), "forced phase", np.bincount(fp, minlength=4), "network wanted == forced", (act[F] == fp).mean())
nw = F[act[F] != fp]
print("forced against network:", len(nw), "per ep", np.bincount(ep[nw], minlength=6))
print("halting forced", np.quantile(ph[nw, legal[nw].argmax(1)], [.1, .25, .5, .75, .9]))
print("halting wanted", np.quantile(ph[nw, act[nw]], [.1, .25, .5, .75, .9]))
# logit rank of forced phase
rk = np.array([(L[i] > L[i, legal[i].argmax()]).sum() for i in nw])
print("rank of forced phase in network logits (0=top)", np.bincount(rk, minlength=4))
gp = np.array([L[i].max() - L[i, legal[i].argmax()] for i in nw])
print("logit gap top - forced", np.quantile(gp, [.1, .25, .5, .75, .9]))
# look back: for each forced (against network) service, the earlier real decisions within the last 30 s - min gap between top and p
for i in nw[:25]:
    p = legal[i].argmax()
    prev = [j for j in range(i - 8, i) if j >= 0 and ep[j] == ep[i] and legal[j].sum() > 1]
    gaps = [(ts[j], round(float(L[j].max() - L[j, p]), 1), int(ss[j, p]), int(ph[j, p]), int(act[j])) for j in prev]
    print(f"ep{ep[i]} t{ts[i]} g{g[i]} forced {p} (ss {ss[i,p]:.0f}, halt {ph[i,p]:.0f}) wanted {act[i]} (halt {ph[i,act[i]]:.0f}) | earlier real decisions (t, gap, ss_p, halt_p, act): {gaps}")
