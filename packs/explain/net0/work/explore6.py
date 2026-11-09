import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0")
import numpy as np, tools
np.set_printoptions(precision=2, suppress=True, linewidth=250)
w = tools.load("start_weights.npz")
names=tools.names()
# synthetic: empty junction, current phase p, ok=1, t_green=12/60
for p in range(4):
    x0 = np.zeros(46); x0[40+p]=1; x0[44]=1; x0[45]=12/60
    l0 = tools.forward(w,x0)[0]
    print("=== cur",p,"empty logits",l0, "pick",l0.argmax())
    # add one stopped vehicle on each stop-line lane (q=d=1/3) and see logits change
    for i in range(12):
        x=x0.copy(); x[i]=0.335; x[12+i]=0.335
        l=tools.forward(w,x)[0]
        x2=x0.copy(); x2[i]=1; x2[12+i]=1
        l2=tools.forward(w,x2)[0]
        print(f"{names[i][2:]:4s} 1veh d={l-l0} pick {l.argmax()} | full d={l2-l0} pick {l2.argmax()}")
    for j in range(8):
        x=x0.copy(); x[24+j]=0.3; x[32+j]=0.3   # ~10 queued vehicles upstream
        l=tools.forward(w,x)[0]
        x2=x0.copy(); x2[32+j]=0.3   # 10 moving
        l2=tools.forward(w,x2)[0]
        print(f"{names[24+j][2:]:4s} 10q d={l-l0} pick {l.argmax()} | 10mov d={l2-l0} pick {l2.argmax()}")
