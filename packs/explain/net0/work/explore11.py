import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work")
from evalh import *
np.set_printoptions(precision=2, suppress=True, linewidth=250)
H = np.round(X[:,0:12]*2.985).astype(int)
for e in range(6):
    m=(ctrl==0)&(ep==e)
    print("ep",e,"n",m.sum(),"ok",(m&ok).sum(),"T>=3600 ok",(m&ok&(T>=3600)).sum(), "end", T[m].max())
# drain-phase states, own: show sequence for ep 0
m=np.where((ctrl==0)&(ep==0)&(T>=3600))[0]
for i in m:
    print(T[i], cur[i], int(X[i,44]), net[i], H[i], np.round(X[i,12:24]*2.985).astype(int), np.round(tools.forward(w,X[i])[0],1))
