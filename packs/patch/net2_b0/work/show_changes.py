import numpy as np, sys, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import tools
from collections import Counter
w0 = tools.load("start_weights.npz"); w = tools.load("work/weights.npz")
S = tools.states(); X = S["obs"].astype(np.float64); ctl = S["controller"]; t = S["time_s"]; ep=S["episode"]
cap=22.4/7.5; capu=256/7.5
ok = X[:,44]>0.5; cur = X[:,40:44].argmax(1)
a0 = tools.greedy(w0,X); a = tools.greedy(w,X)
QL = {0:[0,1,6,7],1:[2,8],2:[3,4,9,10],3:[5,11]}
dd = np.stack([ (X[:,[i+12 for i in QL[p]]]*cap).round().sum(1) for p in range(4)],1)
qq = np.stack([ (X[:,QL[p]]*cap).round().sum(1) for p in range(4)],1)
m = ok&(ctl==0)&(a!=a0)
print("transitions old->new:", Counter(zip(a0[m].tolist(), a[m].tolist())))
print("new pick has vehicles at its stop line:", (dd[m, a[m]]>0).sum(), "of", m.sum())
print("halted at new pick's stop line minus at old pick's (mean):", (qq[m,a[m]]-qq[m,a0[m]]).mean().round(2))
for i in np.where(m)[0]:
    print(f"ep{ep[i]} t={t[i]:4d} cur{cur[i]} old{a0[i]} new{a[i]}  veh per phase {dd[i].astype(int)} halted {qq[i].astype(int)}")
