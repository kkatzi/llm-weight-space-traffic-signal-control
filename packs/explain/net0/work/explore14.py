import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work")
from evalh import *
np.set_printoptions(precision=2, suppress=True, linewidth=250)
def jac(x):
    a1 = w["W1"]@x+w["b1"]; h1=np.tanh(a1)
    a2 = w["W2"]@h1+w["b2"]; h2=np.tanh(a2)
    return (w["W3"]*(1-h2**2)) @ (w["W2"]*(1-h1**2)) @ w["W1"]
# groups in halted-vehicle units: use q (halted) and m = d - q (moving) per stop-line lane
G = {"nsT":[0,1,6,7],"nsL":[2,8],"ewT":[3,4,9,10],"ewL":[5,11]}
UPEW = [26,27,30,31]; UPEWd=[34,35,38,39]; UPNS=[24,25,28,29]; UPNSd=[32,33,36,37]
def feats(x):
    q = x[0:12]*2.985; d = x[12:24]*2.985; m = np.maximum(d-q,0)
    f = {}
    for g,ix in G.items():
        f[g+"_h"] = q[ix].sum(); f[g+"_m"] = m[ix].sum()
    f["upEW"] = (x[UPEWd].sum()+x[UPEW].sum())*34.13/2   # ~ vehicles (halted count double)
    f["upNS"] = (x[UPNSd].sum()+x[UPNS].sum())*34.13/2
    return f
FN = list(feats(X[0]).keys())
F = np.array([[feats(x)[k] for k in FN] for x in X])
# group weights from jacobian: d logit / d feature (chain: feature -> raw inputs)
def group_weights(J):
    Wg = {}
    for g,ix in G.items():
        # one halted vehicle on lane i: q_i += 1/2.985, d_i += 1/2.985
        Wg[g+"_h"] = np.mean([ (J[:,i]+J[:,12+i])/2.985 for i in ix],axis=0)
        Wg[g+"_m"] = np.mean([ J[:,12+i]/2.985 for i in ix],axis=0)
    # upstream: one halted vehicle = q+d each 1/34.13; feature counts (q+d)/2 *34.13 = 1 per halted vehicle
    Wg["upEW"] = np.mean([(J[:,i]+J[:,i+8])/34.13 for i in UPEW],axis=0)
    Wg["upNS"] = np.mean([(J[:,i]+J[:,i+8])/34.13 for i in UPNS],axis=0)
    return np.array([Wg[k] for k in FN]).T  # 4 x nf
lin = {}
for c in range(4):
    msk = ok&(ctrl==0)&~dead&(cur==c)
    xb = X[msk].mean(0); xb[40:44]=0; xb[40+c]=1; xb[44]=1
    J = jac(xb); l0 = tools.forward(w,xb)[0]
    Wg = group_weights(J); fb = np.array([feats(xb)[k] for k in FN])
    b = l0 - Wg@fb
    lin[c] = (Wg,b)
    Wc = Wg - Wg.min(0,keepdims=True)
    print("cur",c,"bias",np.round(b - b.max(),2))
    for k,col in zip(FN, Wg.T): print(f"   {k:6s}", np.round(col - col.mean(),2))
def dec_g(x, i=None):
    c = int(np.argmax(x[40:44])); Wg,b = lin[c]
    f = np.array([feats(x)[k] for k in FN])
    return int(np.argmax(b + Wg@f))
report(dec_g, "grouped Taylor per cur")
