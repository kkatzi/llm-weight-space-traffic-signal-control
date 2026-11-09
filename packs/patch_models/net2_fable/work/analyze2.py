import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import tools

w = tools.load("start_weights.npz")
S = tools.states()
X, act, ep, t, ctl = S["obs"], S["act"], S["episode"], S["time_s"], S["controller"]
names = tools.names()
g = tools.greedy(w, X)
ok = X[:, 44] > 0.5
phase = X[:, 40:44].argmax(1)
tg = X[:, 45] * 60

# lanes per phase (stop-line indices into queue/density blocks)
LANES = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}
UP = {0: [0, 4], 1: [1, 5], 2: [2, 6], 3: [3, 7]}  # upstream lane idx (0..7) feeding each phase's lanes (lane1u feeds both 1 and 2)

def demand(X, p):
    d = X[:, 12 + np.array(LANES[p])].sum(1)
    return d

m0 = ctl == 0
print("t_green values seen:", np.unique(tg[m0]))
print("decision times within green (start-net):", np.bincount(tg[m0].astype(int))[:40])

# When honoured and switching, demand on the current lanes
print("\n--- start-net, honoured states: switch or stay vs. density on current green lanes ---")
for p in range(4):
    m = m0 & ok & (phase == p)
    cur = demand(X, p)[m] / len(LANES[p])
    stay = g[m] == p
    print(f"phase {p}: n={m.sum()}, stay rate={stay.mean():.2f}; mean cur-lane density when switching={cur[~stay].mean():.2f}, when staying={cur[stay].mean() if stay.any() else float('nan'):.2f}")
    # histogram of current density at switch
    hist, edges = np.histogram(cur[~stay], bins=[0, .17, .34, .5, .67, .84, 1.01])
    print("   switch density hist (per-lane avg, bins of ~0.5 car):", hist)
    # demand on chosen phase vs. max alternative
    ch = g[m][~stay]
    Xs = X[m][~stay]
    dem = np.stack([demand(Xs, q) / len(LANES[q]) for q in range(4)], 1)
    qd = np.stack([Xs[:, np.array(LANES[q])].sum(1) / len(LANES[q]) for q in range(4)], 1)
    best = dem.argmax(1)
    print("   chosen==argmax density among phases:", (ch == best).mean(), " chosen==argmax queue:", (ch == qd.argmax(1)).mean())
    print("   mean density of chosen phase lanes:", dem[np.arange(len(ch)), ch].mean(), " of current:", dem[:, p].mean())

# same for actuated controller states: how long does it hold?
print("\n--- actuated: stay rate by phase, honoured ---")
m2 = ctl == 2
for p in range(4):
    m = m2 & ok & (phase == p)
    print(p, m.sum(), (act[m] == p).mean())
print("\n--- fixed: ---")
m1 = ctl == 1
for p in range(4):
    m = m1 & ok & (phase == p)
    print(p, m.sum(), (act[m] == p).mean(), "tg at switch", np.unique(tg[m][act[m] != p]))

# What does network do with upstream densities: correlate logit differences
print("\nmean upstream density by controller:", [X[ctl == c, 32:40].mean() for c in range(3)])
print("mean stop-line density by controller:", [X[ctl == c, 12:24].mean() for c in range(3)])
# Episode-level time to empty
for c in range(3):
    for e in np.unique(ep[ctl == c]):
        m = (ctl == c) & (ep == e)
        print("ctl", c, "ep", e, "n decisions", m.sum(), "last t", t[m].max(), "mean sum density", X[m, 12:40].sum(1).mean().round(2))
