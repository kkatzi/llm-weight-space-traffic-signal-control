import numpy as np, sys
sys.path.insert(0, '.')
import tools
w = tools.load("start_weights.npz")
S = tools.states()
for k,v in S.items(): print(k, v.shape, v.dtype, v[:5] if v.ndim==1 else '')
X = S["obs"]; act = S["act"]; ctl = S["controller"]; ep = S["episode"]; t = S["time_s"]
print("controller names", S.get("controller_names"))
a = tools.greedy(w, X)
for c in np.unique(ctl):
    m = ctl==c
    print("ctl", c, "n", m.sum(), "agree net vs recorded act", (a[m]==act[m]).mean())
    print("  episodes", np.unique(ep[m]))
print(tools.trainability(w))
for k in w: print(k, w[k].shape, np.abs(w[k]).mean(), np.abs(w[k]).max())
print("b3", w["b3"])
