import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work")
from sparsecheck import *
# realistic "end of own green" sparse states: own lanes have moving vehicles (d only); others halted
def mk(cur, halted=(), moving=(), tg=0.2):
    x=np.zeros(46); x[40+cur]=1; x[44]=1; x[45]=tg
    for ln in halted: x[LANES[ln]]+=1/2.985; x[12+LANES[ln]]+=1/2.985
    for ln in moving: x[12+LANES[ln]]+=1/2.985
    return x
cases = [(3,(),("E2",)),(3,(),("E2","W2")),(3,("E2",),()),(3,("N0",),("E2",)),(3,("N2",),("E2",)),
         (0,(),("N0","N1")),(0,("N2",),("N0",)),(0,("N0",),()),(0,("N0","N1"),()),
         (1,(),("N2",)),(1,("N2",),()),(1,("N0",),("N2",)),(1,("N0","N1"),()),(1,("E2",),()),
         (2,(),("E0","W1")),(2,("N0",),("E0",))]
for c,h,mv in cases:
    x = mk(c,h,mv); l = tools.forward(w,x)[0]
    print(f"cur {c} halted {h} moving {mv}: pick {l.argmax()} logits {np.round(l,1)}")
