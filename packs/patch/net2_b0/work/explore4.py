import numpy as np, sys
sys.path.insert(0, '.')
import tools
from collections import Counter
np.set_printoptions(linewidth=220, precision=3, suppress=True)
w = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"].astype(np.float64); act = S["act"]; ctl = S["controller"]; ep = S["episode"]; t = S["time_s"]
QL = {0:[0,1,6,7],1:[2,8],2:[3,4,9,10],3:[5,11]}
DL = {p:[i+12 for i in v] for p,v in QL.items()}
UQ = {0:[24,28],1:[25,29],2:[26,30],3:[27,31]}  # crude: lane1u feeds through+left; assign to left for now
def phase_q(X):
    return np.stack([X[:,QL[p]].sum(1) for p in range(4)],1)
def phase_d(X):
    return np.stack([X[:,DL[p]].sum(1) for p in range(4)],1)
cur = X[:,40:44].argmax(1); ok = X[:,44]>0.5
a = tools.greedy(w, X)
m = (ctl==0)&ok
print("network honoured decisions:", m.sum())
print("stay share", (a[m]==cur[m]).mean())
# transitions
print(Counter(zip(cur[m], a[m])))
# when it stays, what tgreen
print("stay by tgreen", Counter((X[m & (a==cur),45]*60).round()))
# per-controller run lengths
for c in range(3):
    mm = (ctl==c)
    L=[]
    for e in range(6):
        idx=np.where(mm&(ep==e))[0]
        ph=cur[idx]; tg=X[idx,45]*60
        # green run ends when next decision has different phase
        for i in range(len(idx)-1):
            if ph[i+1]!=ph[i]: L.append((ph[i], tg[i]))
    L=np.array(L)
    print(S["controller_names"][c], "num greens/ep", len(L)/6)
    for p in range(4):
        g=L[L[:,0]==p,1]
        print("   phase",p,"n",len(g),"mean last tg",g.mean().round(1), Counter(g.round()).most_common(5))
