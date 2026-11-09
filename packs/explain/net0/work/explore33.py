import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work")
from evalh import *
np.set_printoptions(precision=2, suppress=True, linewidth=250)
def L(x):
    q = x[0:12]*2.985; d = x[12:24]*2.985
    return dict(zip("N0 N1 N2 E0 E1 E2 S0 S1 S2 W0 W1 W2".split(), (q+d)/2))
def upEW(x): return (x[[26,27,30,31]].sum()+x[[34,35,38,39]].sum())*34.13/2
def raw_scores(x):
    l = L(x)
    nsT = 1.2*(l["N0"]+l["S0"]) + 0.8*(l["N1"]+l["S1"]); ewT = 1.33*(l["E0"]+l["E1"]+l["W1"])
    nsL = l["N2"]+l["S2"]; ewL = l["E2"]+l["W2"]
    return np.array([nsT - 0.6*ewL, 1.1*nsL, 0.6*ewT - 0.5*nsL + 0.1*upEW(x), ewL])
RS = np.array([raw_scores(x) for x in X])
Lg = tools.forward(w, X)
top2 = np.argsort(-Lg,1)[:,:2]
gap = Lg[np.arange(len(X)),top2[:,0]] - Lg[np.arange(len(X)),top2[:,1]]
for c in range(4):
    print("cur",c)
    for p in range(4):
        for q in range(p+1,4):
            if c!=2 and (p==c or q==c): continue
            m = ok&(cur==c)&(gap<0.5)&(((top2[:,0]==p)&(top2[:,1]==q))|((top2[:,0]==q)&(top2[:,1]==p)))
            if m.sum()<5: 
                print(f"   {p} vs {q}: n={m.sum()}"); continue
            dsc = RS[m,p]-RS[m,q]
            print(f"   {p} vs {q}: n={m.sum():3d}  unbiased score diff S{p}-S{q} at tie: median {np.median(dsc):+.2f}  mean {dsc.mean():+.2f}  IQR {np.percentile(dsc,25):+.2f}..{np.percentile(dsc,75):+.2f}")
