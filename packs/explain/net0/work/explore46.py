import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0")
import numpy as np, tools
np.set_printoptions(precision=2, suppress=True, linewidth=250)
w = tools.load("start_weights.npz"); names = tools.names()
LN = "N0 N1 N2 E0 E1 E2 S0 S1 S2 W0 W1 W2".split()
for c in range(4):
    x0=np.zeros(46); x0[40+c]=1; x0[44]=1; x0[45]=0.2
    l0=tools.forward(w,x0)[0]; b = l0-l0[2]
    print(f"cur {c} empty (rel EW_through, /3.5): {np.round(b/3.5,2)}")
    rows=[]
    for i,ln in enumerate(LN):
        x=x0.copy(); x[i]=1/2.985; x[12+i]=1/2.985
        l=tools.forward(w,x)[0]; e=((l-l[2])-b)/3.5
        rows.append(e)
    R=np.array(rows)
    for i,ln in enumerate(LN): print(f"   {ln}: {R[i]}")
