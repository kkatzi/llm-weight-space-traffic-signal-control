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
m=ok&~dead
emptyc=(D[np.arange(len(X)),g]==0)&(UD[np.arange(len(X)),g]==0)
others=(D.sum(1)+UD.sum(1))>0
bad=m&emptyc&others
print('non-deadlock ok states',m.sum(),'bad picks',bad.sum(), 'by controller',[ (bad&(C==c)).sum() for c in range(3)])
idx=np.where(bad)[0]
for i in idx[:40]:
    print(C[i],ep[i],T[i],'ph',ph[i],'pick',g[i],'D',D[i],'UD',UD[i].round(2),'lg',lg[i].round(1),'tg',X[i,45].round(2))
# deadlock state logits
i=np.where(dead)[0][5]
print('deadlock logits', lg[i], tools.block(X[i]))
