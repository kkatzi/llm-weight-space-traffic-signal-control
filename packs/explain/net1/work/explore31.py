from try9 import *
B = np.array([[-99, 0.4, 0.0, -0.3],[0.45, 0.5, 0.0, 0.2],[0.0, -0.4, 1.1, -0.12],[0.0, -0.5, -0.75, 0.75]])
S_all = np.array([scores9(x) for x in X]) + B[cur]
# residual of each candidate logit vs score, relative to the mean of the other candidates, near boundaries
def resid(p):
    # for states where p is in the top-2 and gap<2 and p != cur
    m = ok & (cur!=p) & ((top2[:,0]==p)|(top2[:,1]==p)) & (gap<2)
    other = np.where(top2[m,0]==p, top2[m,1], top2[m,0])
    idx = np.where(m)[0]
    r = (L[idx,p]-L[idx,other])/K - (S_all[idx,p]-S_all[idx,other])
    return idx, r
feat = {1: ['q_N2','q_S2'], 3: ['q_E2','q_W2'], 0: ['q_N0','q_S0','q_S1'], 2: ['q_E0','q_E1','q_W1']}
for p in range(4):
    idx, r = resid(p)
    print(f"phase {p}: n={len(idx)} mean resid {r.mean():+.2f}")
    for f in feat[p]:
        full = X[idx, ix[f]] > 0.99
        print(f"   {f} full: resid {r[full].mean() if full.any() else float('nan'):+.2f} (n={full.sum()})  not full: {r[~full].mean():+.2f}")
    # by number of empty own lanes
    tot = sum(X[idx, ix['d_'+f[2:]]] for f in feat[p])
    for lo,hi in [(0,0.34),(0.34,1.01),(1.01,2.01),(2.01,9)]:
        mm = (tot>=lo)&(tot<hi)
        if mm.sum(): print(f"   own density sum in [{lo},{hi}): resid {r[mm].mean():+.2f} n={mm.sum()}")
