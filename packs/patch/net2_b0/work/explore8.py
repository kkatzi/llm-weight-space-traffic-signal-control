import numpy as np, sys
sys.path.insert(0, '.')
import tools
from collections import Counter
np.set_printoptions(linewidth=220, precision=3, suppress=True)
w = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"].astype(np.float64); act = S["act"]; ctl = S["controller"]; ep = S["episode"]; t = S["time_s"]
cap=22.4/7.5; capu=256/7.5
cur = X[:,40:44].argmax(1); ok = X[:,44]>0.5
a = tools.greedy(w, X)
lg = tools.forward(w, X)
QL = {0:[0,1,6,7],1:[2,8],2:[3,4,9,10],3:[5,11]}
d = np.stack([ (X[:,[i+12 for i in QL[p]]]*cap).round().sum(1) for p in range(4)],1)
q = np.stack([ (X[:,QL[p]]*cap).round().sum(1) for p in range(4)],1)
uq = (X[:,24:32]*capu).round(); ud=(X[:,32:40]*capu).round()
# upstream per phase: through phases get lane0u + lane1u; left phases get lane1u
up_q = np.stack([uq[:,0]+uq[:,4]+uq[:,1]+uq[:,5], uq[:,1]+uq[:,5], uq[:,2]+uq[:,6]+uq[:,3]+uq[:,7], uq[:,3]+uq[:,7]],1)
m = (ctl==0)&ok
idx=np.where(m)[0]
print("upstream halted totals at honoured decisions:", Counter(uq[idx].sum(1)).most_common(15))
print("time-avg total vehicles (stop+up density) per controller:")
for c in range(3):
    mm=ctl==c
    tot = (X[mm,12:24]*cap).sum(1)+(X[mm,32:40]*capu).sum(1)
    hal = (X[mm,0:12]*cap).sum(1)+(X[mm,24:32]*capu).sum(1)
    print(S["controller_names"][c], "mean veh in obs", tot.mean().round(2), "mean halted", hal.mean().round(2), "decisions", mm.sum(), "veh*5s/3600 =", (tot.sum()*5/3600/6).round(2), "vehh per ep")
