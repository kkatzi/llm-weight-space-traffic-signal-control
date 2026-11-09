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
m = (ctl==0)&ok
idx = np.where(m)[0]
# For switch decisions: rank of chosen phase by stop-line density among non-current phases
sw = idx[a[idx]!=cur[idx]]
print("switches", len(sw))
rank=[]
for i in sw:
    cand=[p for p in range(4) if p!=cur[i]]
    # normalise per-lane? use raw counts
    vals = {p: d[i,p] for p in cand}
    r = sorted(cand, key=lambda p:-vals[p]).index(a[i])
    rank.append((r, vals[a[i]], max(vals.values())))
rank=np.array(rank)
print("rank of chosen among candidates by stopline count:", Counter(rank[:,0]))
print("chosen with 0 while max>=3:", ((rank[:,1]==0)&(rank[:,2]>=3)).sum())
print("chosen count - max count distribution:", Counter(rank[:,1]-rank[:,2]))
# per-phase: how long since each phase was last served? measure max red time per phase in network episodes
for c in range(3):
    reds=[]
    for e in range(6):
        ii = np.where((ctl==c)&(ep==e))[0]
        last = {p:0 for p in range(4)}
        for i in ii:
            for p in range(4):
                if cur[i]==p: last[p]=t[i]
                else: reds.append((p, t[i]-last[p]))
    reds=np.array(reds)
    print(S["controller_names"][c], [ (p, np.percentile(reds[reds[:,0]==p,1],[50,90,99,100])) for p in range(4)])
