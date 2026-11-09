import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work")
from evalh import *
from sparsecheck import check
np.set_printoptions(precision=2, suppress=True, linewidth=250)
def L(x):
    q = x[0:12]*2.985; d = x[12:24]*2.985
    return dict(zip("N0 N1 N2 E0 E1 E2 S0 S1 S2 W0 W1 W2".split(), (q+d)/2))
def upEW(x): return (x[[26,27,30,31]].sum()+x[[34,35,38,39]].sum())*34.13/2
BIAS = {0:[-1.2,-2.3,0,-2.1], 1:[-2.4,-0.25,0,-1.5], 2:[-2.8,-1.9,0,-0.7], 3:[-1.9,-0.56,0,-0.05]}
def make(nsL_w=(1,1), ewL_w=(1,1), tg_w=0.0, nsT_vs_3=0.0, bias=BIAS):
    def dec(x):
        c = int(np.argmax(x[40:44])); l = L(x)
        nsT = 1.2*(l["N0"]+l["S0"]) + 0.8*(l["N1"]+l["S1"])
        nsL = nsL_w[0]*l["N2"] + nsL_w[1]*l["S2"]
        ewT = 1.33*(l["E0"]+l["E1"]+l["W1"])
        ewL = ewL_w[0]*l["E2"] + ewL_w[1]*l["W2"]
        tg = x[45]-0.2
        s = np.array([nsT - 0.6*ewL - tg_w*tg, 1.1*nsL - tg_w*tg, 0.6*ewT - 0.5*nsL + 0.1*upEW(x),
                      ewL + tg_w*tg - nsT_vs_3*(l["N1"]+l["S1"])]) + bias[c]
        if c!=2: s[c]=-99
        return int(np.argmax(s))
    return dec
variants = [("base", make()),
            ("nsL N2 heavier", make(nsL_w=(1.2,0.8))),
            ("ewL E2 heavier", make(ewL_w=(1.2,0.8))),
            ("t_green 0.4", make(tg_w=0.4)),
            ("N1/S1 against 3", make(nsT_vs_3=0.2)),
            ("all four", make((1.2,0.8),(1.2,0.8),0.4,0.2))]
for nm,d in variants:
    report(d, nm)
    print("     sparse cur2:", check(d,cur=2))
