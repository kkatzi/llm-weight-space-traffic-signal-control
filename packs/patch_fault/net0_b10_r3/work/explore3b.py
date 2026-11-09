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
def seq(w, p0, extra, steps=60):
    p=p0; g=2; out=[]
    for s in range(steps):
        a=tools.forward(w,obs(p,g,extra))[0].argmax()
        if a!=p and g>=10: p=a; g=2
        else: g+=5
        out.append(int(p))
    # compress runs: phase x duration
    comp=[]; 
    for q in out:
        if comp and comp[-1][0]==q: comp[-1][1]+=5
        else: comp.append([q,5])
    return " ".join(f"{a}:{b}" for a,b in comp[:8])
cases = [("empty",{}),("W1u6",{31:0.18,39:0.18}),("N0",{0:1/3,12:1/3}),("N1",{1:1/3,13:1/3}),("S1",{7:1/3,19:1/3})]
import itertools
for c3,cN in [(0.5,0.0),(0.75,0.0),(0.75,0.5),(0.75,0.4),(1.0,0.5),(1.0,0.6),(1.25,0.6)]:
    w,w0 = build(c3,cN)
    a = tools.greedy(w,X)
    print(f"c3={c3} cN={cN} changed-in-hour {(a[mh]!=ap[mh]).mean():.4f}")
    for name,extra in cases:
        print(f"     {name:6s} from2: {seq(w,2,extra)}")
