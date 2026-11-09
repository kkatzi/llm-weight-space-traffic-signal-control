import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
np.set_printoptions(linewidth=220, precision=3, suppress=True)
w=tools.load('start_weights.npz')
S=tools.states()
X=S['obs'].astype(np.float64); A=S['act']; ep=S['episode']; T=S['time_s']; C=S['controller']
SL={0:[0,1,6,7],1:[2,8],2:[3,4,9,10],3:[5,11]}
UP={0:[0,1,4,5],1:[1,5],2:[2,3,6,7],3:[3,7]}
d=X[:,12:24]; ud=X[:,32:40]
D=np.stack([d[:,SL[p]].sum(1) for p in range(4)],1)
UD=np.stack([ud[:,UP[p]].sum(1) for p in range(4)],1)
ph=X[:,40:44].argmax(1)
# max red time per phase while it has stop-line demand
for c in range(3):
    res=[]
    for e in range(6):
        idx=np.where((C==c)&(ep==e)&(T<3600))[0]
        last=np.zeros(4)
        mx=np.zeros(4)
        for i in idx:
            for p in range(4):
                if ph[i]==p: last[p]=T[i]
                elif D[i,p]>0: mx[p]=max(mx[p],T[i]-last[p])
        res.append(mx)
    print(S['controller_names'][c],'max red-with-demand per phase per episode\n',np.array(res))
# how often each phase is fully empty (stop-line + upstream)
E=(D+UD)==0
for c in range(3):
    m=C==c
    print(S['controller_names'][c],'empty-phase frequency',E[m].mean(0).round(3), 'any empty', E[m].any(1).mean().round(3),'all empty',E[m].all(1).mean().round(3))
# min nonzero density values
print('min nonzero ud', ud[ud>0].min(), 'min nonzero d', d[d>0].min())
