import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/patch/net1_b5")
import tools, numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=200)
W0=tools.load('start_weights.npz')
S=tools.states()
X=S['obs'].astype(np.float64); act=S['act']; ep=S['episode']; ctl=S['controller']; t=S['time_s']
N=tools.names(); ix={n:i for i,n in enumerate(N)}
cur=X[:,40:44].argmax(1); ok=X[:,44]>0.5
cs=22.4/7.5; cu=256/7.5
d=X[:,12:24]*cs; q=X[:,0:12]*cs; du=X[:,32:40]*cu; qu=X[:,24:32]*cu
def phase_load(D,DU,share=0.6):
    L=np.zeros((len(D),4))
    for a in range(4):
        thr = 0 if a in (0,2) else 2
        lef = 1 if a in (0,2) else 3
        L[:,thr]+= D[:,3*a]+D[:,3*a+1]+DU[:,2*a]+share*DU[:,2*a+1]
        L[:,lef]+= D[:,3*a+2]+(1-share)*DU[:,2*a+1]
    return L
Lq=phase_load(q,qu); Ld=phase_load(d,du)
