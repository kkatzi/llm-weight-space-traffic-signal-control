import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
from build import build, demand, SL, UP
np.set_printoptions(linewidth=220, precision=2, suppress=True)
S=tools.states(); w0=tools.load('start_weights.npz'); w=build()
X=S['obs'].astype(np.float64); C=S['controller']; ep=S['episode']; T=S['time_s']
ok=X[:,44]>0.5
d=X[:,12:24]; ud=X[:,32:40]; q=X[:,0:12]
D=np.stack([d[:,SL[p]].sum(1) for p in range(4)],1)
UD=np.stack([ud[:,UP[p]].sum(1) for p in range(4)],1)
g=tools.greedy(w,X); ph=X[:,40:44].argmax(1)
for c in range(3):
    m=ok&(C==c)
    sel_sl_empty=(D[np.arange(len(X)),g]==0)
    other_sl=(D.sum(1)-D[np.arange(len(X)),g])>0
    mm=m&sel_sl_empty&other_sl
    print(S['controller_names'][c],'ok',m.sum(),'picks phase w/ empty stop-line while others queued:',mm.sum(),
          ' of which stays',(mm&(g==ph)).sum(), ' upstream dens of picked: ', np.percentile(UD[mm,][np.arange(mm.sum()),g[mm]],[10,50,90]).round(3) if mm.sum() else '')
# also time-of-day of these for network
m=ok&(C==0)&(D[np.arange(len(X)),g]==0)&((D.sum(1)-D[np.arange(len(X)),g])>0)
print('times',np.sort(T[m])[:60])
