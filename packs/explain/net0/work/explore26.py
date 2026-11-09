import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work")
src = open("/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work/explore24.py").read()
src = src.split("variants = [")[0]; exec(src)
def make2(k=0.15, t0=10, bias=BIAS, only2=False):
    def dec(x):
        c = int(np.argmax(x[40:44])); l = L(x)
        nsT = 1.2*(l["N0"]+l["S0"]) + 0.8*(l["N1"]+l["S1"])
        nsL = l["N2"]+l["S2"]; ewT = 1.33*(l["E0"]+l["E1"]+l["W1"]); ewL = l["E2"]+l["W2"]
        tot = nsT+nsL+ewT+ewL
        s = np.array([nsT - 0.6*ewL, 1.1*nsL, 0.6*ewT - 0.5*nsL + 0.1*upEW(x) + k*(tot-t0), ewL]) + bias[c]
        if c!=2: s[c]=-99
        return int(np.argmax(s))
    return dec
for k in [0.0, 0.1, 0.15]:
    report(make2(k), f"S2 bonus {k}*(tot-10)")
