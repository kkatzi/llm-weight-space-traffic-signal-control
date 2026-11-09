"""How since_served / overdue are counted, and how often a phase will be overdue at the next switching opportunity."""
import os, sys
import numpy as np
from collections import Counter
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import tools

S = tools.states()
w = tools.load("start_weights.npz")
X = S["obs"].astype(np.float64)
rs = S["rule_state"]
since = rs[:, 0:4]; legal = rs[:, 4:8] > 0.5; overdue = rs[:, 8:12] > 0.5; ph = rs[:, 12:16]
tg = rs[:, 16].astype(int); green = rs[:, 17].astype(int)
ep = S["episode"]; t = S["time_s"]
lg = tools.forward(w, X)
net = lg.argmax(1)
N = len(net)
# a stretch of the record
for i in range(100, 130):
    print(t[i], green[i], tg[i], since[i].astype(int), ph[i].astype(int), overdue[i].astype(int), legal[i].astype(int), net[i])
# boundary of overdue
odv = since[overdue]
print("min since among overdue:", odv.min())
nod = (~overdue) & (ph > 0)
nod[np.arange(N), green] = False
print("max since among not-overdue with halting>0 (non-current):", since[nod].max(), Counter(since[nod][since[nod] >= 85]).most_common())
# overdue with halting 0?
print("overdue with ph==0:", (overdue & (ph == 0)).sum())
