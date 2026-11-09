import sys; sys.path.insert(0, ".")
import numpy as np, tools
S = tools.states()
X = S["obs"].astype(float); act = S["act"].astype(int); rs = S["rule_state"]; ep = S["episode"]; ts = S["time_s"]
w = tools.load("start_weights.npz")
L = tools.forward(w, X)
legal = rs[:, 4:8] > 0.5; ph = rs[:, 12:16]; tg = rs[:, 16]; g = rs[:, 17].astype(int); ss = rs[:, 0:4]
N = len(X); ok = X[:, 44] > 0.5
nm = tools.names(); I = {n: i for i, n in enumerate(nm)}
# spillback indicator: pocket full and upstream lane 1 halting
for appr, thr_ph, left_ph in [("E", 2, 3), ("W", 2, 3), ("N", 0, 1), ("S", 0, 1)]:
    full = X[:, I[f"q_{appr}2"]] > 0.99
    upq = X[:, I[f"q_{appr}1u"]]
    sb = full & (upq > 0.05)
    dur = sb & (g == thr_ph) & (tg >= 2)
    print(f"{appr}: pocket full {full.mean():.3f}, full & up1 halting {sb.mean():.3f}, during its through green {dur.sum()} states; up1 halting q mean when so {upq[dur].mean() if dur.sum() else 0:.3f}")
# upstream halting overall
for k in ["q_N0u","q_N1u","q_E0u","q_E1u","q_S0u","q_S1u","q_W0u","q_W1u"]:
    v = X[:, I[k]]
    print(k, "mean", v.mean().round(3), "share>0.1", (v > 0.1).mean().round(3), "max", v.max().round(2))
# time profile of total halting in each episode (rough demand)
for e in range(6):
    m = ep == e
    print("ep", e, "mean total phase_halting", ph[m].sum(1).mean().round(1), "max", ph[m].sum(1).max())
