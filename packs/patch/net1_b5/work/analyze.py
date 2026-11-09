import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/patch/net1_b5")
import tools, numpy as np
w=tools.load('start_weights.npz')
S=tools.states()
X=S['obs'].astype(np.float64); act=S['act']; ep=S['episode']; ctl=S['controller']; t=S['time_s']
names=tools.names()
g=tools.greedy(w,X)
print('greedy==act for start_network:', (g[ctl==0]==act[ctl==0]).mean())
# phase durations under start network
cur=X[:,40:44].argmax(1)
for c in range(3):
    m=ctl==c
    print('ctrl',S['controller_names'][c],'n',m.sum(),'episodes',np.unique(ep[m]))
# greens durations for start_network episodes
m0=np.where(ctl==0)[0]
for e in np.unique(ep[m0]):
    idx=m0[ep[m0]==e]
    ph=cur[idx]; tg=X[idx,45]*60
    # segments
    segs=[]; s=0
    for i in range(1,len(idx)):
        if ph[i]!=ph[i-1]:
            segs.append((ph[s], (i-s)*5)); s=i
    segs.append((ph[s],(len(idx)-s)*5))
    d={p:[l for (q,l) in segs if q==p] for p in range(4)}
    print('ep',e,'len',len(idx)*5,'nseg',len(segs), {p:(len(v),np.mean(v) if v else 0,max(v) if v else 0) for p,v in d.items()})
    print('  seq', ''.join(str(p) for p,_ in segs[:60]))
