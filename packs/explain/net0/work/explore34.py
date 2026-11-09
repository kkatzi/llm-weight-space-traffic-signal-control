import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work")
src = open("/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work/explore33.py").read().split("Lg = tools")[0]
exec(src)
from sparsecheck import check
def hold(x):
    l = L(x)
    nsT = l["N0"]+l["N1"]+l["S0"]+l["S1"]; nsL = l["N2"]+l["S2"]
    return (l["E2"] < 0.25) and (nsL < 0.25) and (l["W2"] <= 1) and (nsT <= 2 + l["W2"])
def mk(bias_fn, use_hold=True):
    def dec(x):
        c = int(np.argmax(x[40:44]))
        if use_hold and c==2 and hold(x): return 2
        s = raw_scores(x) + bias_fn(c)
        if c!=2: s[c] = -99
        return int(np.argmax(s))
    return dec
T1 = {0:[0,-1.6,0,-1.3], 1:[-2.1,0,0,-1.2], 2:[-2.9,-1.9,0,-0.9], 3:[-2.4,-1.05,0,0]}
for nm,fn in [("threshold table", lambda c: np.array(T1[c])),
              ("shared [-2.5,-1.5,0,-1.1]", lambda c: np.array([-2.5,-1.5,0,-1.1]))]:
    d = mk(fn); r = report(d, nm)
    m = ok&(ctrl==0)&~dead
    print("   per-cur own:", [round(float((r[m&(cur==a)]==net[m&(cur==a)]).mean()),3) for a in range(4)])
    print("   sparse cur2:", check(d,cur=2))
