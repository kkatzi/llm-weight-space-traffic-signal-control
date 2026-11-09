import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0")
import numpy as np, tools
np.set_printoptions(precision=2, suppress=True, linewidth=220)
w = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"].astype(float); ctrl = S["controller"]; T=S["time_s"]; ep=S["episode"]
net = tools.greedy(w, X)
ok = X[:,44] > 0.5
cur = X[:,40:44].argmax(1)
m = ok&(cur==2)&(ctrl==0)&(X[:,45]>0.99)
print("time of long phase2 holds:", np.percentile(T[m],[0,10,50,90,100]))
print("total vehicles density sum in those:", np.percentile(X[m,12:24].sum(1)+X[m,32:40].sum(1),[0,50,90,100]))
# per episode time range
for e in np.unique(ep[ctrl==0]):
    mm=(ctrl==0)&(ep==e)
    print(e, T[mm].min(), T[mm].max(), (m&(ep==e)).sum())
