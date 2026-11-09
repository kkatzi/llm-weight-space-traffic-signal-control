import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'work')
import tools, numpy as np
from build3 import build
S = tools.states(); X = S['obs'].astype(np.float64); T=S['time_s']
ok = X[:,44]>0.5; mh = ok&(T<3600)
wp,_ = build(); ap = tools.greedy(wp, X)
def obs(phase, g, extra={}):
    x = np.zeros(46); x[40+phase]=1; x[44]=1.0 if g>=10 else 0.0; x[45]=min(g/60,1)
    for k,v in extra.items(): x[k]=v
    return x
def first(w, p0, extra, target, steps=60):
    p=p0; g=2
    for s in range(steps):
        a=tools.forward(w,obs(p,g,extra))[0].argmax()
        if a!=p and g>=10: p=a; g=2
        else: g+=5
        if p==target: return s*5
    return None
cases = [("empty->3",{},3),("W1u6->3",{31:0.18,39:0.18},3),("W1u3->3",{31:0.09,39:0.09},3)]
for lane in [0,1,6,7]:
    cases.append((f"lone {tools.names()[lane]}->0",{lane:1/3,lane+12:1/3},0))
cases.append(("lone N0 moving d only ->0",{12:1/3},0))
for c3 in [0.5,0.75]:
    for cN in [0.0,0.25,0.5,0.75,1.0,1.5]:
        w,w0 = build(c3,cN)
        a = tools.greedy(w,X)
        res = []
        for name,extra,tgt in cases:
            res.append(f"{name}:{min([t for t in [first(w,p0,extra,tgt) for p0 in [2,3]] if t is not None], default=None) if all(first(w,p0,extra,tgt) is not None for p0 in [2,3]) else 'NEVER'}")
        print(f"c3={c3} cN={cN} changed-in-hour vs proxy {(a[mh]!=ap[mh]).mean():.4f} | " + " ".join(res))
