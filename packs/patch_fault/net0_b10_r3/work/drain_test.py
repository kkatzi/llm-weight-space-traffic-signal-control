"""Toy drain test: lone halted cars at stop lines; a car leaves once its phase has been green for >= 7 s.
Measures whether every car is eventually served and the total wait (car-seconds) from each start phase."""
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'work')
import tools, numpy as np, json, itertools
from build5 import build
n = tools.names()
lane_phase = {0:0,1:0,2:1,3:2,4:2,5:3,6:0,7:0,8:1,9:2,10:2,11:3}
def run(w, p0, lanes, steps=360):
    p=p0; g=2; left=dict((l,1) for l in lanes); wait=0
    for s in range(steps):
        if not left: return wait, True
        x=np.zeros(46); x[40+p]=1; x[44]=1.0 if g>=10 else 0.0; x[45]=min(g/60,1)
        for l,c in left.items(): x[l]=c/2.987; x[l+12]=c/2.987
        x[11]=0; x[23]=0
        a=tools.forward(w,x)[0].argmax()
        if a!=p and g>=10: p=a; g=2
        else: g+=5
        if g>=7:
            for l in [l for l in left if lane_phase[l]==p]: del left[l]
        wait += 5*len(left)
    return wait, not left
def score(w, combos):
    fails=0; tot=0
    for lanes in combos:
        for p0 in range(4):
            wt, okk = run(w,p0,lanes)
            tot+=wt; fails+= (not okk)
    return fails, tot
singles=[(l,) for l in range(12)]
pairs=list(itertools.combinations(range(12),2))
rng=np.random.default_rng(0)
triples=[tuple(sorted(rng.choice(12,3,replace=False))) for _ in range(150)]
for cfg in [dict()]:
    w,_=build(**cfg)
    print(cfg, "singles", score(w,singles), "pairs", score(w,pairs), "triples", score(w,triples))
w0=tools.load("start_weights.npz")
print("start", "singles", score(w0,singles), "pairs", score(w0,pairs), "triples", score(w0,triples))
