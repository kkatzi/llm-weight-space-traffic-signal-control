import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work")
src = open("/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work/explore24.py").read()
src = src.split("variants = [")[0]; exec(src)
def mk(bias_fn, lane_weighted=True):
    def dec(x):
        c = int(np.argmax(x[40:44])); l = L(x)
        if lane_weighted:
            nsT = 1.2*(l["N0"]+l["S0"]) + 0.8*(l["N1"]+l["S1"]); ewT = 1.33*(l["E0"]+l["E1"]+l["W1"])
        else:
            nsT = l["N0"]+l["S0"]+l["N1"]+l["S1"]; ewT = l["E0"]+l["E1"]+l["W0"]+l["W1"]
        nsL = l["N2"]+l["S2"]; ewL = l["E2"]+l["W2"]
        s = np.array([nsT - 0.6*ewL, 1.1*nsL, 0.6*ewT - 0.5*nsL + 0.1*upEW(x), ewL]) + bias_fn(c)
        if c!=2: s[c]=-99
        return int(np.argmax(s))
    return dec
shared = lambda c: np.array([-2.4,-1.6,0,-1.4])
NEXT = {0:2, 2:3, 3:1}
def succ(c):
    b = np.array([-2.4,-1.6,0,-1.4])
    if c in NEXT: b[NEXT[c]] += 0.8
    return b
report(mk(lambda c: np.array(BIAS[c])), "per-cur bias table")
report(mk(shared), "shared bias")
report(mk(succ), "shared + successor bonus 0.8")
report(mk(succ, False), "shared + successor bonus, plain sums")
report(mk(lambda c: np.array(BIAS[c]), False), "per-cur table, plain sums")
