"""Exploration of the recorded states: rule state, network requests, shield interventions."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import tools

S = tools.states()
w = tools.load("start_weights.npz")
X = S["obs"].astype(np.float64)
R = S["rule_state"]
act = S["act"].astype(int)
ep = S["episode"]
ts = S["time_s"]
lg = tools.forward(w, X)
net = lg.argmax(1)
print("recorded act == net argmax:", (act == net).mean())

since = R[:, 0:4]; legal = R[:, 4:8] > 0.5; overdue = R[:, 8:12] > 0.5
ph = R[:, 12:16]; tg = R[:, 16]; green = R[:, 17].astype(int)
print("green == obs phase one-hot:", (green == X[:, 40:44].argmax(1)).mean())
print("t_green obs*60 vs tg:", np.abs(np.minimum(tg, 60) / 60 - X[:, 45]).max())

# legal pattern counts
from collections import Counter
c = Counter()
for i in range(len(X)):
    c[(green[i], tuple(legal[i].astype(int)), tuple(overdue[i].astype(int)))] += 1
for k, v in sorted(c.items(), key=lambda kv: -kv[1])[:40]:
    print(k, v)

# t_green distribution
print("t_green values:", Counter(tg.astype(int).tolist()).most_common(30))

# how often the net's request is illegal
ill = ~legal[np.arange(len(X)), net]
print("net request illegal share:", ill.mean(), ill.sum())
# split by reason
only_cur = legal.sum(1) == 1
print("states with exactly one legal phase:", only_cur.sum())
print("overdue any:", overdue.any(1).sum())

# Within the illegal ones: is it min-green, max-green, or overdue?
cur_legal = legal[np.arange(len(X)), green]
for i in np.where(ill)[0][:60]:
    print(ep[i], ts[i], "g", green[i], "tg", tg[i], "net", net[i], "legal", legal[i].astype(int),
          "od", overdue[i].astype(int), "since", since[i].astype(int), "ph", ph[i].astype(int))
