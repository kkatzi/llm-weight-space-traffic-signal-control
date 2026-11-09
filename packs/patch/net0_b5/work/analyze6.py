import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
np.set_printoptions(linewidth=220, precision=2, suppress=True)
w=tools.load('start_weights.npz')
S=tools.states()
X=S['obs'].astype(np.float64); A=S['act']; ep=S['episode']; T=S['time_s']; C=S['controller']
SL={0:[0,1,6,7],1:[2,8],2:[3,4,9,10],3:[5,11]}
UP={0:[0,1,4,5],1:[1,5],2:[2,3,6,7],3:[3,7]}
d=X[:,12:24]; ud=X[:,32:40]
D=np.stack([d[:,SL[p]].sum(1) for p in range(4)],1)
UD=np.stack([ud[:,UP[p]].sum(1) for p in range(4)],1)
g=tools.greedy(w,X); lg=tools.forward(w,X)
ok=X[:,44]>0.5; ph=X[:,40:44].argmax(1)
dead=(C==0)&(ep==2)&(T>3620)
m=(C==0)&ok&(g==ph)&~dead
print('non-deadlock stays on network states', m.sum())
for i in np.where(m)[0]:
    print(ep[i],T[i],'ph',ph[i],'D',D[i],'UD',UD[i].round(2),'lg',lg[i].round(1),'tg',X[i,45].round(2))
print('--- end of each network episode')
for e in range(6):
    mm=np.where((C==0)&(ep==e))[0]
    for i in mm[-12:]:
        if T[i]>=3600 and T[i]<3700:
            print(e,T[i],'ph',ph[i],'ok',X[i,44],'pick',g[i],'D',D[i],'UD',UD[i].round(2),'lg',lg[i].round(1))
