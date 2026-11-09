from common import *
from build_gate import counts
wG=tools.load('/home/traffic-ntua/wsp_v3_agent_runs/patch/net1_b5/work/wG.npz')
V=counts(X)
g=tools.greedy(wG,X)
m=np.where(ok&(g==cur)&(ctl==0))[0]
print('stays',len(m))
lanes={0:[0,1,6,7],1:[2,8],2:[3,4,9,10],3:[5,11]}
for i in m[:60]:
    p=cur[i]
    print('t',t[i],'p',p,'tg',round(X[i,45]*60),'V',np.round(V[i]).astype(int),'Lq',np.round(Lq[i],1),'halting on p stop lanes',round(q[i,lanes[p]].sum(),1))
