import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work")
from sparsecheck import *
np.set_printoptions(precision=2, suppress=True, linewidth=250)
tgs = [12,17,22,27,32,37,42,47,52,57,60]
configs = [(), ("N2",),("S2",),("E2",),("W2",),("N0",),("N0","S0"),("N0","N1"),("N1","S1"),("N0","W2"),("N1","W2"),("N0","N1","W2"),("N0","S0","W2"),("N2","S1"),("S2","N1"),("N2","W2"),("S2","W2"),("N0","N1","S1"),("N2","S2")]
print("config".ljust(18), " ".join(f"{t:3d}" for t in tgs))
for cfg in configs:
    row=[]
    for t in tgs:
        x=np.zeros(46); x[42]=1; x[44]=1; x[45]=min(t/60,1)
        for ln in cfg: x[LANES[ln]]+=1/2.985; x[12+LANES[ln]]+=1/2.985
        row.append(tools.greedy(w,x)[0])
    print(str(cfg).ljust(18), " ".join(f"{v:3d}" for v in row))
# also moving-only single vehicles (approaching, not yet stopped) at tg 0.2 and 1
print("moving-only single vehicles (d only)")
for ln in SIDE:
    out=[]
    for t in [12,60]:
        x=np.zeros(46); x[42]=1; x[44]=1; x[45]=min(t/60,1); x[12+LANES[ln]]=1/2.985
        out.append(tools.greedy(w,x)[0])
    print("  ",ln,out)
