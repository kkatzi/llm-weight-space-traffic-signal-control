import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import tools

w = tools.load("start_weights.npz")
S = tools.states()
X, act, ep, t, ctl = S["obs"], S["act"], S["episode"], S["time_s"], S["controller"]
names = tools.names()
g = tools.greedy(w, X)
lg = tools.forward(w, X)
ok = X[:, 44] > 0.5
phase = X[:, 40:44].argmax(1)
tg = X[:, 45] * 60

LANES = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}
UPL = {0: [0, 4], 1: [1, 5], 2: [2, 6], 3: [3, 7]}   # upstream lanes (index 0..7) whose traffic mostly goes to the phase
CAP_SL, CAP_UP = 22.4 / 7.5, 256 / 7.5           # 2.99, 34.1 cars

def cars(X, p):
    """approx vehicles present on phase p's stop-line lanes (+ upstream lanes feeding them; up lane 1 shared)."""
    sl = X[:, 12 + np.array(LANES[p])].sum(1) * CAP_SL
    up = X[:, 32 + np.array(UPL[p])].sum(1) * CAP_UP
    return sl, up

def upq(X, p):
    return X[:, 24 + np.array(UPL[p])].sum(1) * CAP_UP

m = (ctl == 0) & ok & (g != phase)
Xs = X[m]; ch = g[m]; cur = phase[m]
sl = np.stack([cars(Xs, q)[0] for q in range(4)], 1)
up = np.stack([cars(Xs, q)[1] for q in range(4)], 1)
uq = np.stack([upq(Xs, q) for q in range(4)], 1)
tot = sl + up
n = len(ch)
idx = np.arange(n)
print("switches:", n)
print("chosen phase stop-line cars == 0:", (sl[idx, ch] < 0.5).mean(), " and upstream cars==0 too:", ((sl[idx, ch] < 0.5) & (up[idx, ch] < 0.5)).mean())
print("chosen == argmax stop-line cars:", (ch == sl.argmax(1)).mean())
print("chosen == argmax total cars:", (ch == tot.argmax(1)).mean())
# how much worse is the chosen vs best, in cars
loss_sl = sl.max(1) - sl[idx, ch]
print("stop-line cars forgone (best - chosen): mean", loss_sl.mean(), " quantiles", np.quantile(loss_sl, [.5, .75, .9, .95, .99]))
loss_tot = tot.max(1) - tot[idx, ch]
print("total cars forgone: mean", loss_tot.mean(), " quantiles", np.quantile(loss_tot, [.5, .75, .9, .95, .99]))

# Look at the bad cases
bad = loss_sl >= 2
print("\nshare of switches with >=2 stop-line cars forgone:", bad.mean(), "count", bad.sum())
print("in those, chosen phase, current phase pairs:")
pairs = {}
for c, p in zip(ch[bad], cur[bad]):
    pairs[(p, c)] = pairs.get((p, c), 0) + 1
print(sorted(pairs.items(), key=lambda kv: -kv[1]))
print("in bad cases, sl of chosen vs best:", np.c_[sl[bad][:, :], ch[bad], cur[bad]][:15].round(1))

# Is the net following a fixed order?  transitions matrix
T = np.zeros((4, 4), int)
for p, c in zip(cur, ch):
    T[p, c] += 1
print("\ntransition matrix (rows current, cols chosen):\n", T)

# Transition matrix for actuated controller
m2 = (ctl == 2) & ok & (act != phase)
T2 = np.zeros((4, 4), int)
for p, c in zip(phase[m2], act[m2]):
    T2[p, c] += 1
print("actuated transitions:\n", T2)
Xa = X[m2]; cha = act[m2]
sla = np.stack([cars(Xa, q)[0] for q in range(4)], 1)
print("actuated chosen == argmax stop-line cars:", (cha == sla.argmax(1)).mean())

# Logit margin stats at switches: how confident
srt = np.sort(lg[m], 1)
print("\nlogit gap at switches: quantiles", np.quantile(srt[:, -1] - srt[:, -2], [.1, .25, .5, .75]))
