"""Trace one episode: phase sequence, network requests, rule state."""
import os, sys
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
since = R[:, 0:4]; legal = R[:, 4:8] > 0.5; overdue = R[:, 8:12] > 0.5
ph = R[:, 12:16]; tg = R[:, 16]; green = R[:, 17].astype(int)

E = int(sys.argv[1]) if len(sys.argv) > 1 else 1
a, b = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 120)
idx = np.where(ep == E)[0][a:b]
np.set_printoptions(precision=2, suppress=True, linewidth=200)
for i in idx:
    q = X[i, 0:12]
    print(f"t{ts[i]:5d} g{green[i]} tg{tg[i]:3.0f} net{net[i]} lg{np.round(lg[i],1)} L{legal[i].astype(int)} "
          f"OD{overdue[i].astype(int)} ss{since[i].astype(int)} ph{ph[i].astype(int)} "
          f"qN{np.round(q[0:3]*3).astype(int)} qE{np.round(q[3:6]*3).astype(int)} qS{np.round(q[6:9]*3).astype(int)} qW{np.round(q[9:12]*3).astype(int)}")
