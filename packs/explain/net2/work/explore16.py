from common import *
np.set_printoptions(linewidth=220, precision=2, suppress=True)
# For current phase 2 (EW through) ok states, tabulate the network's choice by
# NS-through halting count and NS-left / EW-left counts
def grp(X):
    q = np.round(X[:, 0:12]*3).astype(int); d = np.round(X[:, 12:24]*3).astype(int)
    return q, d
q, d = grp(X)
for p in range(4):
    m = OK & (PH == p)
    print(f"=== current {p}: network choice frequencies by total halting in each group")
    gq = np.stack([q[:, [0,1,6,7]].sum(1), q[:, [2,8]].sum(1), q[:, [3,4,9,10]].sum(1), q[:, [5,11]].sum(1)], 1)
    gd = np.stack([d[:, [0,1,6,7]].sum(1), d[:, [2,8]].sum(1), d[:, [3,4,9,10]].sum(1), d[:, [5,11]].sum(1)], 1)
    # among candidate groups, which has largest halting count? compare network choice
    cands = [j for j in range(4) if j != p]
    for j in cands:
        mm = m & (A == j)
        print(f" chose {j}: n={mm.sum()}  mean halting per group {gq[mm].mean(0).round(2)}  mean veh per group {gd[mm].mean(0).round(2)}")
