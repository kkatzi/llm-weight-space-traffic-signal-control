import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net1")
import numpy as np, tools
np.set_printoptions(precision=2, suppress=True, linewidth=200)
w = tools.load("start_weights.npz")
names = tools.names(); ix = {n:i for i,n in enumerate(names)}
def base(p, tg=12):
    x = np.zeros(46); x[40+p]=1; x[44]=1; x[45]=tg/60; return x
for p in range(4):
    print("cur",p,"empty network logits:", tools.forward(w, base(p))[0])
# single-lane perturbation: add one lane full (density 1, queue 1)
lanes = ['N0','N1','N2','E0','E1','E2','S0','S1','S2','W0','W1','W2']
for p in range(4):
    print(f"=== cur {p}: logits change when one stop-line lane has 3 halted cars (q=d=1)")
    b = tools.forward(w, base(p))[0]
    for ln in lanes:
        x = base(p); x[ix['q_'+ln]]=1; x[ix['d_'+ln]]=1
        print(f"  {ln}: {tools.forward(w,x)[0]-b}")
