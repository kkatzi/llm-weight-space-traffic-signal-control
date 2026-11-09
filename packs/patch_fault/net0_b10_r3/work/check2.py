import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'work')
import tools, numpy as np
from build2 import build
S = tools.states(); X = S['obs'].astype(np.float64); C=S['controller']; T=S['time_s']
ok = X[:,44]>0.5
w1,_ = build(0.0)
a1 = tools.greedy(w1, X)
def obs(phase, g, extra={}):
    x = np.zeros(46); x[40+phase]=1; x[44]=1.0 if g>=10 else 0.0; x[45]=min(g/60,1)
    for k,v in extra.items(): x[k]=v
    return x
def run(w, p0, extra, steps=40):
    p=p0; g=2; seq=[]
    for s in range(steps):
        a=tools.forward(w,obs(p,g,extra))[0].argmax()
        if a!=p and g>=10: p=a; g=2
        else: g+=5
        seq.append(int(p))
    return seq
for c in [0.5,0.75,1.0,1.25]:
    w,w0 = build(c)
    a = tools.greedy(w, X)
    mh = ok&(T<3600)
    print(f"c={c}: changed honoured decisions in hour vs proxy: {(a[mh]!=a1[mh]).mean():.4f}  dist {np.bincount(a[mh],minlength=4)} vs {np.bincount(a1[mh],minlength=4)}")
    tg = X[:,45]
    for lo,hi in [(0.19,0.21),(0.21,0.5),(0.5,1.01)]:
        mm = mh&(tg>lo)&(tg<hi)
        if mm.sum(): print(f"    tg in ({lo},{hi}): n={mm.sum()} changed {(a[mm]!=a1[mm]).mean():.3f}")
    for name,extra in [("empty",{}),("W1u 6 halted",{31:0.18,39:0.18}),("W1u 3 halted",{31:0.09,39:0.09})]:
        seq = run(w,2,extra,40)
        t3 = next((i for i,p in enumerate(seq) if p==3), None)
        print(f"    {name}: first phase3 after {None if t3 is None else t3*5} s; seq {seq[:30]}")
    print("   ", tools.edit_size(w,w0))
