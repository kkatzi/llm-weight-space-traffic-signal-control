import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0")
import numpy as np, tools
np.set_printoptions(precision=2, suppress=True, linewidth=250)
w = tools.load("start_weights.npz")
names=tools.names()
# probe: moving-only vehicle (d only, q=0) vs halted, for each lane, cur=2 and cur=0
for p in [0,2,3]:
    x0 = np.zeros(46); x0[40+p]=1; x0[44]=1; x0[45]=12/60
    l0 = tools.forward(w,x0)[0]
    print("=== cur",p)
    for i in range(12):
        x=x0.copy(); x[12+i]=0.335
        l=tools.forward(w,x)[0]
        x2=x0.copy(); x2[i]=0.335; x2[12+i]=0.335
        l2=tools.forward(w,x2)[0]
        print(f"{names[i][2:]:4s} moving d={l-l0}  halted d={l2-l0}")
# t_green effect on empty and some states
for p in range(4):
    for tg in [12,17,22,32,60]:
        x0 = np.zeros(46); x0[40+p]=1; x0[44]=1; x0[45]=min(tg/60,1)
        print(p,tg,tools.forward(w,x0)[0])
# ok=0 vs ok=1
for p in range(4):
    x0 = np.zeros(46); x0[40+p]=1; x0[44]=0; x0[45]=7/60
    print("ok0",p,tools.forward(w,x0)[0])
