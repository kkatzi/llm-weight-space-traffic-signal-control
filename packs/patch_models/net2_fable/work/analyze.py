import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import tools

w = tools.load("start_weights.npz")
S = tools.states()
X, act, ep, t, ctl = S["obs"], S["act"], S["episode"], S["time_s"], S["controller"]
names = tools.names()
print("controllers:", S.get("controller_names"))
print("X shape", X.shape, "ctl counts", np.bincount(ctl))

lg = tools.forward(w, X)
g = lg.argmax(1)
print("agree greedy vs recorded act (start net states):", (g[ctl == 0] == act[ctl == 0]).mean())

ok = X[:, 44] > 0.5
phase = X[:, 40:44].argmax(1)
print("\n--- network's greedy choices on all states, by current phase (honoured only) ---")
for p in range(4):
    m = ok & (phase == p)
    print(p, "n", m.sum(), "choice dist", np.bincount(g[m], minlength=4) / max(m.sum(), 1))

# Switching behaviour on network-recorded episodes: green durations
print("\n--- green durations (start-net episodes) ---")
for c in range(3):
    m = ctl == c
    tg = X[m, 45] * 60
    ph = phase[m]
    # count switch events: t_green small after being larger
    durs = []
    prev = None
    for i in range(m.sum()):
        if prev is not None and tg[i] < prev:
            durs.append(prev)
        prev = tg[i]
    durs = np.array(durs)
    print("ctl", c, "mean green", durs.mean(), "median", np.median(durs), "q90", np.quantile(durs, .9), "max", durs.max(),
          "n switches", len(durs), "share at cap 60", (durs >= 59).mean())
    print("   phase share:", np.bincount(ph, minlength=4) / m.sum())

# Per-phase choice vs. queues
print("\n--- mean queue features when network picks each phase (honoured, start-net) ---")
m0 = (ctl == 0) & ok
for p in range(4):
    m = m0 & (g == p)
    if m.sum() == 0:
        continue
    mq = X[m, :12].mean(0)
    print(p, "n", m.sum(), " ".join(f"{names[i]}={mq[i]:.2f}" for i in range(12)))

# Hidden unit stats
h1, h2 = tools.hidden(w, X)
print("\nh1 std per unit:", np.round(h1.std(0), 2))
print("h2 std per unit:", np.round(h2.std(0), 2))
print("W3 abs max per unit:", np.round(np.abs(w["W3"]).max(0), 2))
print("b3", w["b3"])
print(tools.trainability(w))
