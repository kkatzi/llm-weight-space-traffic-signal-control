import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work")
src = open("/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work/explore24.py").read()
src = src.split("variants = [")[0]; exec(src)
d = make()
r = np.array([d(x) for x in X])
Lg = tools.forward(w,X)
m = ok&(ctrl==0)&~dead
C = np.zeros((4,4,4),int)
for a,b,c in zip(cur[m],net[m],r[m]): C[a,b,c]+=1
for a in range(4): print("cur",a,"rows=net cols=rule\n",C[a])
def sc(x):
    l = L(x)
    nsT = 1.2*(l["N0"]+l["S0"]) + 0.8*(l["N1"]+l["S1"]); nsL = l["N2"]+l["S2"]
    ewT = 1.33*(l["E0"]+l["E1"]+l["W1"]); ewL = l["E2"]+l["W2"]
    c = int(np.argmax(x[40:44]))
    return np.array([nsT - 0.6*ewL, 1.1*nsL, 0.6*ewT - 0.5*nsL + 0.1*upEW(x), ewL]) + BIAS[c], (nsT,nsL,ewT,ewL,upEW(x))
idx = np.where(m&(r!=net)&(cur==0))[0]
for i in idx[:25]:
    s,f = sc(X[i])
    H = np.round(X[i,0:12]*2.985).astype(int); D = np.round(X[i,12:24]*2.985).astype(int)
    print(f"net {net[i]} rule {r[i]} logits {np.round(Lg[i],1)} scores {np.round(s,1)} loads {np.round(f,1)} H {H} D {D}")
