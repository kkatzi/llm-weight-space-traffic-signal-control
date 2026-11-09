import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, tools
np.set_printoptions(precision=3, suppress=True, linewidth=200)
S = tools.states()
for k, v in S.items():
    print(k, v.shape, v.dtype)
print(S["controller_names"])
print(np.unique(S["controller"], return_counts=True))
print(np.unique(S["episode"], return_counts=True))
w = tools.load("start_weights.npz")
lg = tools.forward(w, S["obs"])
g = lg.argmax(1)
print("greedy==act", (g == S["act"]).mean())
rs = S["rule_state"]
print("rule_state shape", rs.shape)
print(rs[:20])
print(S["time_s"][:20])
