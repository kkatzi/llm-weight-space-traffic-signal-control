import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/patch/net1_b5")
import tools, numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
w=tools.load('start_weights.npz')
S=tools.states()
X=S['obs'].astype(np.float64); act=S['act']; ep=S['episode']; ctl=S['controller']; t=S['time_s']
cur=X[:,40:44].argmax(1); ok=X[:,44]>0.5
for c in [1,2]:
    allsegs={p:[] for p in range(4)}
    for e in range(6):
        idx=np.where((ctl==c)&(ep==e))[0]
        ph=cur[idx]; s=0
        for i in range(1,len(idx)):
            if ph[i]!=ph[i-1]:
                allsegs[ph[s]].append((i-s)*5); s=i
    print(S['controller_names'][c], {p:(len(v), np.mean(v), np.percentile(v,[10,50,90])) for p,v in allsegs.items()})
