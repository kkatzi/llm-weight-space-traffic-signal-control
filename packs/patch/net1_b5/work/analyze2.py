import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/patch/net1_b5")
import tools, numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
w=tools.load('start_weights.npz')
S=tools.states()
X=S['obs'].astype(np.float64); act=S['act']; ep=S['episode']; ctl=S['controller']; t=S['time_s']
cur=X[:,40:44].argmax(1); ok=X[:,44]>0.5
g=tools.greedy(w,X); lg=tools.forward(w,X)
for c in range(3):
    m=(ctl==c)&ok
    print(S['controller_names'][c], 'ok decisions',m.sum(),'net stays',(g[m]==cur[m]).mean())
    # transition matrix cur->g
    T=np.zeros((4,4),int)
    for a,b in zip(cur[m],g[m]): T[a,b]+=1
    print(T)
# for not-ok decisions, what does the net pick?
m=(ctl==0)&~ok
T=np.zeros((4,4),int)
for a,b in zip(cur[m],g[m]): T[a,b]+=1
print('not ok, start network:\n',T)
# t_green at decisions for start net
print('t_green values start net', np.unique(np.round(X[ctl==0,45]*60),return_counts=True))
# episode lengths and times
for c in range(3):
    m=ctl==c
    print(S['controller_names'][c], [int(t[m&(ep==e)].max()) for e in range(6)])
