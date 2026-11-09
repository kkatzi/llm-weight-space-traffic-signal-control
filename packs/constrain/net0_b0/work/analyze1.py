"""Exploratory statistics of the recorded run (network under the rules, shield only)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools

S = tools.states()
w = tools.load("start_weights.npz")
X = S["obs"].astype(np.float64)
R = S["rule_state"]
ss, legal, over, ph, tg, gr = R[:, 0:4], R[:, 4:8] > 0.5, R[:, 8:12] > 0.5, R[:, 12:16], R[:, 16], R[:, 17].astype(int)
act = S["act"].astype(int)
ep, t = S["episode"], S["time_s"]
L = tools.forward(w, X)
print("act == argmax:", (L.argmax(1) == act).mean())

np.set_printoptions(linewidth=200, precision=2, suppress=True)
e = int(sys.argv[1]) if len(sys.argv) > 1 else 0
a0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
n = int(sys.argv[3]) if len(sys.argv) > 3 else 120
idx = np.where(ep == e)[0][a0:a0 + n]
print(" t   gr tg  act legal     over      since_served          halting       logits")
for i in idx:
    print(f"{t[i]:4d} {gr[i]} {tg[i]:4.0f}  {act[i]}  {''.join('L' if l else '.' for l in legal[i])}  "
          f"{''.join('O' if o else '.' for o in over[i])}  {ss[i]}  {ph[i]}  {L[i]}")
