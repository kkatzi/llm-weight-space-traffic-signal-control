"""Sensitivity check (analysis only, not used by the supervisor): at the through-phase decisions at t_green=17,
does the network's choice depend on the unusual t_green value? And: does the request at 12 s agree with 17 s?"""
import os, sys
from collections import Counter
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import tools

S = tools.states()
w = tools.load("start_weights.npz")
X = S["obs"].astype(np.float64)
R = S["rule_state"]
ep = S["episode"]; ts = S["time_s"]
lg = tools.forward(w, X)
net = lg.argmax(1)
ph = R[:, 12:16]; tg = R[:, 16]; green = R[:, 17].astype(int)
N = len(X)

m17 = np.where((green % 2 == 0) & (tg == 17))[0]
Xm = X[m17].copy(); Xm[:, 45] = 12 / 60
alt = tools.greedy(w, Xm)
print("t=17 through decisions:", len(m17), " argmax unchanged if t_green were 12:", (alt == net[m17]).mean())
agree = [net[i - 1] == net[i] for i in m17 if i > 0 and ep[i - 1] == ep[i] and tg[i - 1] == 12]
print("request at 12 s equals choice at 17 s:", np.mean(agree))
# t_green sweep for left phases at 12
m12 = np.where((green % 2 == 1) & (tg == 12))[0]
for v in (7, 17, 22):
    Xm = X[m12].copy(); Xm[:, 45] = v / 60
    print(f"left t=12 decisions with t_green set to {v}: argmax unchanged {(tools.greedy(w, Xm) == net[m12]).mean():.3f}")
# what does the network do at 17 s for through phases: stay share
print("stay share at through t=17:", (net[m17] == green[m17]).mean())
