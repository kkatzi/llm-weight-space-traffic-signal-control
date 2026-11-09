import numpy as np, sys
sys.path.insert(0, '.')
import tools
from collections import Counter
w = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"].astype(np.float64); ctl = S["controller"]; t=S["time_s"]
cap=22.4/7.5
cur = X[:,40:44].argmax(1); ok = X[:,44]>0.5
a = tools.greedy(w, X); lg = tools.forward(w,X)
nsE = (X[:,14]==0)&(X[:,20]==0)
ewE = (X[:,17]==0)&(X[:,23]==0)
for c in range(3):
    m = ok&(ctl==c)
    print(S["controller_names"][c], "honoured", m.sum(),
          "| EW pockets empty:", (m&ewE).sum(), "picks3:", (m&ewE&(a==3)).sum(), "(stay", (m&ewE&(a==3)&(cur==3)).sum(), ")",
          "| NS pockets empty:", (m&nsE).sum(), "picks1:", (m&nsE&(a==1)).sum(), "(stay", (m&nsE&(a==1)&(cur==1)).sum(), ")")
srt=np.sort(lg,1); print("max logit spread (max-min):", (srt[:,-1]-srt[:,0]).max())
# for EW-empty picks of 3: what's the second choice
m = ok&(ctl==0)&ewE&(a==3)
lg2 = lg.copy(); lg2[:,3] = -1e9
print("second choice", Counter(lg2[m].argmax(1)), "time", t[m])
