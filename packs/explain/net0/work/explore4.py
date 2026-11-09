import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0")
import numpy as np, tools
np.set_printoptions(precision=2, suppress=True, linewidth=220)
w = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"].astype(float); ctrl = S["controller"]; T=S["time_s"]; ep=S["episode"]
net = tools.greedy(w, X)
names=tools.names()
m = np.where((ctrl==0)&(ep==2)&(T>=3600))[0]
for i in m[:3].tolist()+m[-2:].tolist():
    b = tools.block(X[i]); print(T[i], net[i], {k:v for k,v in b.items() if v!=0}, np.round(tools.forward(w,X[i])[0],2))
# also look before 3675
m2 = np.where((ctrl==0)&(ep==2)&(T>=3550)&(T<3700))[0]
for i in m2:
    b = tools.block(X[i]); print(T[i], net[i], {k:v for k,v in b.items() if v!=0})
