import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work")
src = open("/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work/explore24.py").read()
src = src.split("variants = [")[0]; exec(src)
def feats(x):
    l = L(x)
    nsT = 1.2*(l["N0"]+l["S0"]) + 0.8*(l["N1"]+l["S1"]); nsL = l["N2"]+l["S2"]
    ewT = 1.33*(l["E0"]+l["E1"]+l["W1"]); ewL = l["E2"]+l["W2"]
    return np.array([nsT - 0.6*ewL, 1.1*nsL, 0.6*ewT - 0.5*nsL + 0.1*upEW(x), ewL]), nsT+nsL+ewT+ewL
SC = []; TOT=[]
for x in X:
    s,t = feats(x); SC.append(s); TOT.append(t)
SC=np.array(SC); TOT=np.array(TOT)
L_ = tools.forward(w,X)
# For each cur and load bin: what bias b (relative to phase 2) makes argmax(SC + b) best match? -> not allowed (fitting).
# Instead: the network's logit gap vs score gap: conditional means of (logit_p - logit_2) and (S_p - S_2) by load bin.
m = ok&~dead
for c in range(4):
    print("cur",c)
    for lo,hi in [(0,4),(4,8),(8,12),(12,16),(16,40)]:
        mm = m&(cur==c)&(TOT>=lo)&(TOT<hi)
        if mm.sum()<15: continue
        dl = L_[mm]-L_[mm][:,[2]]; ds = SC[mm]-SC[mm][:,[2]]
        # per phase slope via ratio of std (scale) and intercept: mean(dl) - k*mean(ds), k = cov/var (descriptive only)
        out=[]
        for p in [0,1,3]:
            if p==c: continue
            k = np.cov(dl[:,p],ds[:,p])[0,1]/np.var(ds[:,p])
            b = dl[:,p].mean()/k - ds[:,p].mean()
            out.append(f"p{p}: slope {k:.2f} implied bias {b:+.2f}")
        print(f"  load[{lo},{hi}) n={mm.sum():4d} ", " | ".join(out))
