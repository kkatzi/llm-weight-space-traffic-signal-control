import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
np.set_printoptions(linewidth=200, precision=3, suppress=True)
w=tools.load('start_weights.npz')
S=tools.states()
X=S['obs'].astype(np.float64); A=S['act']; ep=S['episode']; T=S['time_s']; C=S['controller']
SL={0:[0,1,6,7],1:[2,8],2:[3,4,9,10],3:[5,11]}
UP={0:[0,1,4,5],1:[1,5],2:[2,3,6,7],3:[3,7]}  # upstream lane idx (0..7)
def dem(X):
    q=X[:,0:12]; d=X[:,12:24]; uq=X[:,24:32]; ud=X[:,32:40]
    D=np.stack([d[:,SL[p]].sum(1) for p in range(4)],1)
    Q=np.stack([q[:,SL[p]].sum(1) for p in range(4)],1)
    UD=np.stack([ud[:,UP[p]].sum(1) for p in range(4)],1)
    return Q,D,UD
g=tools.greedy(w,X)
Q,D,UD=dem(X)
ok=X[:,44]>0.5
ph=X[:,40:44].argmax(1)
for c in range(3):
    m=(C==c)&ok
    chosenD=D[m,g[m]]
    chosenUD=UD[m,g[m]]
    othersD=D[m].sum(1)-chosenD
    waste=(chosenD==0)&(chosenUD==0)&(othersD>0)
    print(S['controller_names'][c],'ok states',m.sum(),'net picks empty phase while others nonempty:',waste.sum())
    stay=(g[m]==ph[m])
    stayempty=stay&(D[m,ph[m]]==0)&(othersD>0)
    print('   stays',stay.sum(),'stays on empty stopline while others nonempty',stayempty.sum())
    # does the network choose the max-demand phase (excluding current)?
    Dx=D[m].copy(); Dx[np.arange(m.sum()),ph[m]]=-1
    print('   pick == argmax D excl current', (g[m]==Dx.argmax(1)).mean(), ' pick==argmax Q excl', (g[m]==np.where(np.arange(4)[None,:]==ph[m][:,None],-1,Q[m]).argmax(1)).mean())
    print('   pick D rank stats: chosen D mean', chosenD.mean(), 'max other D mean', Dx.max(1).mean())
