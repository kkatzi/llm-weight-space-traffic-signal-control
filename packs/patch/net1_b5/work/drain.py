from common import *
g=tools.greedy(W0,X)
# per-phase vehicles on lanes (stop + upstream)
stop_l={0:[0,1,6,7],1:[2,8],2:[3,4,9,10],3:[5,11]}
up_l={0:[0,1,4,5],1:[1,5],2:[2,3,6,7],3:[3,7]}
V=np.zeros((len(X),4))
for p in range(4):
    V[:,p]=d[:,stop_l[p]].sum(1)+du[:,up_l[p]].sum(1)
np.save('/home/traffic-ntua/wsp_v3_agent_runs/patch/net1_b5/work/V.npy',V)
emptycur=V[np.arange(len(X)),cur]<0.5
others=V.sum(1)-V[np.arange(len(X)),cur]
for c in range(3):
    m=(ctl==c)&ok
    st=(g==cur)
    print(S['controller_names'][c],'ok states',m.sum(),' cur empty & others>0:',(m&emptycur&(others>0.5)).sum(),
          ' of which net stays:',(m&emptycur&(others>0.5)&st).sum(),
          ' net picks an empty phase while some phase non-empty:', (m&(V[np.arange(len(X)),g]<0.5)&(V.max(1)>0.5)).sum())
# show episodes ends for start network
for e in range(6):
    idx=np.where((ctl==0)&(ep==e)&(t>=3540))[0]
    print('ep',e,[(int(t[i]),int(cur[i]),int(g[i]),int(ok[i]),tuple(np.round(V[i]).astype(int))) for i in idx])
