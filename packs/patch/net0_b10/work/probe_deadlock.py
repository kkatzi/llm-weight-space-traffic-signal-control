"""Probe: does the network hold a phase whose lanes are empty while vehicles wait elsewhere?"""
import sys, itertools; sys.path.insert(0, '.')
import numpy as np, tools
w = tools.load(sys.argv[1] if len(sys.argv)>1 else "start_weights.npz")
N = tools.names(); ix = {n:i for i,n in enumerate(N)}
SL = {0:['N0','N1','S0','S1'], 1:['N2','S2'], 2:['E0','E1','W0','W1'], 3:['E2','W2']}
UP = {0:['N0u','N1u','S0u','S1u'], 1:['N1u','S1u'], 2:['E0u','E1u','W0u','W1u'], 3:['E1u','W1u']}
lanes = ['N0','N1','N2','E0','E1','E2','S0','S1','S2','W0','W1','W2']
v1 = 1/(22.4/7.5)
def make(p, tg, waiting, nveh=1):
    x = np.zeros(46); x[40+p]=1; x[44]=1; x[45]=tg
    for l in waiting:
        x[ix['q_'+l]] = min(1, nveh*v1); x[ix['d_'+l]] = min(1, nveh*v1)
    return x
tot=0; held=0; res={}
for p in range(4):
    others = [l for l in lanes if l not in SL[p]]
    cnt=0; h=0
    for k in (1,2,3):
        for comb in itertools.combinations(others, k):
            for tg in (12/60, 17/60, 0.5, 1.0):
                for nv in (1,2,3):
                    x = make(p, tg, comb, nv)
                    cnt+=1
                    if tools.greedy(w, x)[0]==p: h+=1
    res[p]=(h,cnt); tot+=cnt; held+=h
print('held empty phase (per current phase: held/total):', res, 'overall %.3f'%(held/tot))
