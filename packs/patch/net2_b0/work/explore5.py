import numpy as np, sys
sys.path.insert(0, '.')
import tools
from collections import Counter
np.set_printoptions(linewidth=220, precision=3, suppress=True)
w = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"].astype(np.float64); act = S["act"]; ctl = S["controller"]; ep = S["episode"]; t = S["time_s"]
QL = {0:[0,1,6,7],1:[2,8],2:[3,4,9,10],3:[5,11]}
cap=22.4/7.5; capu=256/7.5
def nveh(X):
    # counts of vehicles: stop-line per phase, upstream lane0 for through, upstream lane1 shared
    sl = np.stack([X[:,[i+12 for i in QL[p]]].sum(1)*cap for p in range(4)],1)
    slq = np.stack([X[:,QL[p]].sum(1)*cap for p in range(4)],1)
    up0 = np.stack([X[:,[32,36]].sum(1), np.zeros(len(X)), X[:,[34,38]].sum(1), np.zeros(len(X))],1)*capu
    up1 = np.stack([X[:,[33,37]].sum(1), X[:,[33,37]].sum(1), X[:,[35,39]].sum(1), X[:,[35,39]].sum(1)],1)*capu
    return sl.round(), slq.round(), up0.round(), up1.round()
sl, slq, up0, up1 = nveh(X)
cur = X[:,40:44].argmax(1); ok = X[:,44]>0.5
a = tools.greedy(w, X)
m = (ctl==0)&ok
idx=np.where(m)[0]
# picks where chosen phase stop-line empty
empty_pick = (sl[idx, a[idx]]==0)
print("honoured decisions", len(idx), "picked phase with empty stop-line lanes:", empty_pick.sum())
# and others nonempty
others_max = np.array([max(sl[i,p] for p in range(4) if p!=a[i]) for i in idx])
print("  of which some other phase had vehicles at stop line:", (empty_pick & (others_max>0)).sum())
# picks of chosen phase with least sl count
best = sl[idx].argmax(1)
print("picked phase = max stop-line count phase:", (a[idx]==best).mean())
# by time
late = t[idx]>=3600
print("decisions after 3600:", late.sum())
# show some examples of empty picks
for k in np.where(empty_pick & (others_max>0))[0][:25]:
    i=idx[k]
    print(t[i], "cur",cur[i],"pick",a[i], "sl",sl[i], "q",slq[i], "up0",up0[i], "up1",up1[i], "tg",round(X[i,45]*60))
