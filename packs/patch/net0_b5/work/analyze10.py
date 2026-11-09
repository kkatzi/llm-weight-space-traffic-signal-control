import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
from build import build, demand, SL, UP
np.set_printoptions(linewidth=220, precision=2, suppress=True)
S=tools.states(); w0=tools.load('start_weights.npz')
X=S['obs'].astype(np.float64); C=S['controller']; ep=S['episode']; T=S['time_s']
ok=X[:,44]>0.5
d=X[:,12:24]; ud=X[:,32:40]; q=X[:,0:12]; uq=X[:,24:32]
cap_sl=22.4/7.5; cap_up=256/7.5
# vehicle counts
NV=np.stack([d[:,SL[p]].sum(1)*cap_sl + ud[:,UP[p]].sum(1)*cap_up*(1.0 if p in (0,2) else 0.5) for p in range(4)],1)
NQ=np.stack([q[:,SL[p]].sum(1)*cap_sl + uq[:,UP[p]].sum(1)*cap_up*(1.0 if p in (0,2) else 0.5) for p in range(4)],1)
g=tools.greedy(w0,X); ph=X[:,40:44].argmax(1)
m=ok&(C==0)&(T<3600)&(T>60)
i=np.where(m)[0]
cur_nv=NV[i,ph[i]]; cur_nq=NQ[i,ph[i]]; new_nq=NQ[i,g[i]]; new_nv=NV[i,g[i]]
tot=NV[i].sum(1)
print('network switches at ok (T 60..3600):',len(i))
for lo,hi in [(0,10),(10,20),(20,35),(35,60),(60,200)]:
    mm=(tot>=lo)&(tot<hi)
    if mm.sum()==0: continue
    print(f'total veh {lo}-{hi}: n={mm.sum()}, cur phase veh mean {cur_nv[mm].mean():.1f}, cur halting {cur_nq[mm].mean():.1f}, chosen halting {new_nq[mm].mean():.1f}, chosen veh {new_nv[mm].mean():.1f}, frac chosen max-halting(excl cur) {(g[i][mm]==np.where(np.arange(4)[None,:]==ph[i][mm][:,None],-1,NQ[i][mm]).argmax(1)).mean():.2f}')
# per-episode total demand level proxy
for e in range(6):
    mm=(C==0)&(ep==e)&(T<3600)
    print('ep',e,'mean veh in system proxy',NV[mm].sum(1).mean().round(1))
