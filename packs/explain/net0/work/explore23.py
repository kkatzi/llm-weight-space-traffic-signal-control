import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work")
from sparsecheck import *
np.set_printoptions(precision=2, suppress=True, linewidth=250)
for cur in [2]:
  for tg in [0.2,1.0]:
    x0=np.zeros(46); x0[40+cur]=1; x0[44]=1; x0[45]=tg
    l0=tools.forward(w,x0)[0]; print("cur",cur,"tg",tg,"empty rel2", np.round(l0-l0[2],2))
    eff={}
    for ln in SIDE+["E0","E1","W0","W1"]:
        x=x0.copy(); x[LANES[ln]]=1/2.985; x[12+LANES[ln]]=1/2.985
        l=tools.forward(w,x)[0]; e=(l-l[2])-(l0-l0[2]); eff[ln]=e
        print("  ",ln, np.round(e,2))
    # additivity check over 2- and 3- combos
    errs=[]; flips=0; n=0
    for k in [2,3]:
        for combo in itertools.combinations(SIDE,k):
            x=x0.copy()
            for ln in combo: x[LANES[ln]]=1/2.985; x[12+LANES[ln]]=1/2.985
            l=tools.forward(w,x)[0]; actual=l-l[2]
            pred = (l0-l0[2]) + sum(eff[ln] for ln in combo)
            errs.append(actual-pred); n+=1; flips += int(actual.argmax()!=pred.argmax())
    errs=np.array(errs); print("  additivity: mean abs err", np.round(np.abs(errs).mean(0),2), "argmax flips", flips, "/", n)
