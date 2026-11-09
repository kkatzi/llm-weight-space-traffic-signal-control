"""Derive per-current-phase handicaps from the network's own near-tie states (median score gap where the
network's top two logits are within 0.5), for a given unbiased score function. Statistic, not a fit."""
import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work")
from evalh import *
Lg = tools.forward(w, X)
top2 = np.argsort(-Lg,1)[:,:2]
gap = Lg[np.arange(len(X)),top2[:,0]] - Lg[np.arange(len(X)),top2[:,1]]
def tie_biases(RS, exclude_dead=True, tol=0.5, verbose=False):
    B = {}
    for c in range(4):
        cand = [p for p in range(4) if p!=c or c==2]
        eqs = []
        for i,p in enumerate(cand):
            for q in cand[i+1:]:
                m = ok&(cur==c)&(gap<tol)&(((top2[:,0]==p)&(top2[:,1]==q))|((top2[:,0]==q)&(top2[:,1]==p)))
                if exclude_dead: m &= ~dead
                if m.sum()>=5:
                    eqs.append((p,q,-np.median(RS[m,p]-RS[m,q]), m.sum()))
        # least-squares for b with b2 = 0 (weights by sqrt n) -- solving the over-determined tie equations
        idx = [p for p in cand if p!=2]
        A=[]; y=[]; wt=[]
        for p,q,v,n in eqs:
            row = np.zeros(len(idx))
            if p in idx: row[idx.index(p)] += 1
            if q in idx: row[idx.index(q)] -= 1
            A.append(row); y.append(v); wt.append(np.sqrt(n))
        A=np.array(A)*np.array(wt)[:,None]; y=np.array(y)*np.array(wt)
        sol = np.linalg.lstsq(A,y,rcond=None)[0]
        b = np.zeros(4)
        for p,v in zip(idx,sol): b[p]=v
        B[c]=b
        if verbose: print("cur",c,"ties",[(p,q,round(v,2),n) for p,q,v,n in eqs],"->",np.round(b,2))
    return B
