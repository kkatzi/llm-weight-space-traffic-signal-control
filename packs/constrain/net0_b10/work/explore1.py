import sys; sys.path.insert(0, ".")
import numpy as np, tools
S = tools.states()
for k,v in S.items(): print(k, v.shape, v.dtype)
print(S["controller_names"])
print(np.unique(S["controller"], return_counts=True))
print(np.unique(S["episode"], return_counts=True))
rs = S["rule_state"]
print(rs[:5])
w = tools.load("start_weights.npz")
a = tools.greedy(w, S["obs"])
print("greedy==act", (a==S["act"]).mean())
print(tools.trainability(w))
