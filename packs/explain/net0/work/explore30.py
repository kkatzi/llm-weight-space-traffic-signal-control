import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work")
src = open("/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work/explore24.py").read()
src = src.split("variants = [")[0]; exec(src)
from sparsecheck import check
NEXT = {0:2, 2:3, 3:1}
def mk(e2=1.0, w2=1.0, bonus=0.8, base=(-2.4,-1.6,0,-1.4)):
    def dec(x):
        c = int(np.argmax(x[40:44])); l = L(x)
        nsT = 1.2*(l["N0"]+l["S0"]) + 0.8*(l["N1"]+l["S1"]); ewT = 1.33*(l["E0"]+l["E1"]+l["W1"])
        nsL = l["N2"]+l["S2"]; ewL = e2*l["E2"]+w2*l["W2"]
        b = np.array(base, dtype=float)
        if c in NEXT: b[NEXT[c]] += bonus
        s = np.array([nsT - 0.6*ewL, 1.1*nsL, 0.6*ewT - 0.5*nsL + 0.1*upEW(x), ewL]) + b
        if c!=2: s[c]=-99
        return int(np.argmax(s))
    return dec
for nm,d in [("succ 0.8", mk()), ("E2 1.25 W2 0.75", mk(1.25,0.75)), ("E2 1.25 W2 0.75, base3 -1.6", mk(1.25,0.75,base=(-2.4,-1.6,0,-1.6)))]:
    report(d, nm); print("   sparse:", check(d, cur=2))
