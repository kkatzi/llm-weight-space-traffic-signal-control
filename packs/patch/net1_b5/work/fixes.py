from common import *
from build_gate import counts
wG=tools.load('/home/traffic-ntua/wsp_v3_agent_runs/patch/net1_b5/work/wG.npz')
V=counts(X)
g0=tools.greedy(W0,X); g1=tools.greedy(wG,X)
m=np.where(ok&(g0!=g1)&(V[np.arange(len(X)),g0]<0.5)&(V.max(1)>0.5))[0]
for i in m:
    print(S['controller_names'][ctl[i]],'ep',ep[i],'t',t[i],'cur',cur[i],'old',g0[i],'new',g1[i],'V',np.round(V[i],1),'tg',round(X[i,45]*60))
