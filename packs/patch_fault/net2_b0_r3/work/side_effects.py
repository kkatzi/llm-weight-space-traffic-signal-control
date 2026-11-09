import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import tools, numpy as np
from build import impute_w2
import build_full as B
w0 = tools.load("start_weights.npz")
S = tools.states(); X = S["obs"].astype(np.float64); t = S["time_s"]
ok = X[:, 44] > 0.5; ph = X[:, 40:44].argmax(1)
wI = impute_w2(w0)
lgI = tools.forward(wI, X); aI = lgI.argmax(1)
srt = np.sort(lgI, 1); gap = srt[:, -1] - srt[:, -2]
condA = (ph == 2) & (X[:, 45] >= B.THETA_A)
condB = (ph == 3) & (X[:, 45] >= B.THETA_B) & (X[:, 5] == 0) & (X[:, 17] == 0)
other = ~(condA | condB) & ok
print("imputed-net logit gap quantiles (honoured):", np.percentile(gap[ok], [10, 25, 50, 75]).round(3))
# scan relay/detector choices for side effects
cands_u = [(50, 1.0), (38, -1.0), (42, 1.0), (5, 1.0), (62, -1.0), (48, -1.0), (4, 1.0)]
cands_j = [1, 63, 52, 27, 54, 4]
res = []
for (ua, sa) in cands_u:
    for (ub, sb) in cands_u:
        if ub == ua: continue
        for ja in cands_j:
            for jb in cands_j:
                if jb == ja: continue
                B.U_A, B.J_A, B.U_B, B.J_B, B.U_B_NORMAL = ua, ja, ub, jb, sb
                if sa != 1.0: continue
                wF = B.build(w0, X)
                aF = tools.greedy(wF, X)
                res.append((np.mean(aF[other] != aI[other]), ua, ja, ub, sb, jb))
res.sort()
for r in res[:8]: print("  side effect %.4f  A: u=%d j=%d | B: u=%d normal=%+.0f j=%d" % r)
B.U_A, B.J_A, B.U_B, B.J_B, B.U_B_NORMAL = 50, 1, 38, 63, -1.0
wF = B.build(w0, X); aF = tools.greedy(wF, X)
ch = other & (aF != aI)
print("current choice: changed states' imputed-net gap quantiles:", np.percentile(gap[ch], [25, 50, 75, 100]).round(3), " n", ch.sum())
