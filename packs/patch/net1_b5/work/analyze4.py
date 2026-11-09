import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/patch/net1_b5")
import tools, numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=200)
w=tools.load('start_weights.npz')
S=tools.states()
X=S['obs'].astype(np.float64); act=S['act']; ep=S['episode']; ctl=S['controller']; t=S['time_s']
N=tools.names(); ix={n:i for i,n in enumerate(N)}
cur=X[:,40:44].argmax(1); ok=X[:,44]>0.5
g=tools.greedy(w,X)
cs=22.4/7.5; cu=256/7.5
def veh(prefix, lanes, up):
    pass
# vehicles (density) per lane in counts
d=np.zeros((len(X),12)); q=np.zeros((len(X),12))
d=X[:,12:24]*cs; q=X[:,0:12]*cs
du=X[:,32:40]*cu; qu=X[:,24:32]*cu
# approach index a: N=0,E=1,S=2,W=3 ; stop lanes 3a+0..2 ; up lanes 2a+0..1
def phase_load(D,DU):
    L=np.zeros((len(X),4))
    for a in range(4):
        thr = 0 if a in (0,2) else 2
        lef = 1 if a in (0,2) else 3
        L[:,thr]+= D[:,3*a]+D[:,3*a+1]+DU[:,2*a]+0.6*DU[:,2*a+1]
        L[:,lef]+= D[:,3*a+2]+0.4*DU[:,2*a+1]
    return L
Lq=phase_load(q,qu); Ld=phase_load(d,du)
m=(ctl==0)&ok
print('mean load (halting) per phase at ok:', Lq[m].mean(0), ' density:', Ld[m].mean(0))
# how often net picks phase with max halting
print('net pick == argmax halting load', (g[m]==Lq[m].argmax(1)).mean())
print('net pick == argmax density load (excl current)', )
L2=Ld[m].copy(); L2[np.arange(m.sum()),cur[m]]=-1
print((g[m]==L2.argmax(1)).mean())
# load of chosen phase vs max
ch=Ld[m][np.arange(m.sum()),g[m]]
mx=L2.max(1)
print('chosen load quantiles', np.percentile(ch,[5,25,50,75,95]), ' max other', np.percentile(mx,[5,25,50,75,95]))
print('cases chosen load < 1 veh:', (ch<1).mean(), ' while max other >5:', ((ch<1)&(mx>5)).mean())
# by time of episode
tt=t[m]
for lo,hi in [(0,900),(900,1800),(1800,2700),(2700,3600),(3600,9999)]:
    mm=(tt>=lo)&(tt<hi)
    print(lo,hi,'n',mm.sum(),'mean total veh', (d[m][mm].sum(1)+du[m][mm].sum(1)).mean(), 'chosen<1', (ch[mm]<1).mean(), 'stay',(g[m][mm]==cur[m][mm]).mean())
