import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
from build import SL, UP
np.set_printoptions(linewidth=220, precision=2, suppress=True)
S=tools.states(); w0=tools.load('start_weights.npz')
X=S['obs'].astype(np.float64); C=S['controller']; ep=S['episode']; T=S['time_s']
ok=X[:,44]>0.5
q=X[:,0:12]; uq=X[:,24:32]; d=X[:,12:24]; ud=X[:,32:40]
cs, cu = 22.4/7.5, 256/7.5
# halting vehicle estimate per phase: stop-line halting + upstream halting (lane 1u split half for through/left)
fr={0:[1,.5,1,.5],1:[.5,.5],2:[1,.5,1,.5],3:[.5,.5]}
H=np.stack([q[:,SL[p]].sum(1)*cs + (uq[:,UP[p]]*np.array(fr[p])).sum(1)*cu for p in range(4)],1)
g=tools.greedy(w0,X); ph=X[:,40:44].argmax(1); lg=tools.forward(w0,X)
for c in range(3):
    m=ok&(C==c)&(T<3600)
    i=np.where(m)[0]
    Hx=H[i].copy(); Hx[np.arange(len(i)),ph[i]]=-1
    mx=Hx.max(1); ch=H[i,g[i]]
    diff=mx-ch
    print(S['controller_names'][c],'n',len(i),'halting deficit (max other - chosen) quantiles',np.percentile(diff,[50,75,90,95,99]).round(1),'share >=5:',(diff>=5).mean().round(3))
# example of large deficits for network
m=ok&(C==0)&(T<3600); i=np.where(m)[0]
Hx=H[i].copy(); Hx[np.arange(len(i)),ph[i]]=-1
diff=Hx.max(1)-H[i,g[i]]
for k in i[np.argsort(-diff)[:15]]:
    print(ep[k],T[k],'ph',ph[k],'pick',g[k],'H',H[k].round(1),'lg',lg[k].round(1))
