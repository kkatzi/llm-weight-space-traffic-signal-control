import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work")
exec(open("/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work/explore14.py").read().split("lin = {}")[0])
# load features: halted + 0.5 moving
def loads(x):
    q = x[0:12]*2.985; d = x[12:24]*2.985
    L = (q + d)/2     # = halted + 0.5*moving
    nsT = L[[0,1,6,7]].sum(); nsL = L[[2,8]].sum(); ewT = L[[3,4,9,10]].sum(); ewL = L[[5,11]].sum()
    upEW = (x[[26,27,30,31]].sum()+x[[34,35,38,39]].sum())*34.13/2
    return np.array([nsT,nsL,ewT,ewL,upEW])
LN = ["nsT","nsL","ewT","ewL","upEW"]
FL = np.array([loads(x) for x in X])
# weights for the loads: from jacobian, one 'load unit' = one halted vehicle
Ws = {}; info={}
for c in range(4):
    msk = ok&(ctrl==0)&~dead&(cur==c)
    xb = X[msk].mean(0); xb[40:44]=0; xb[40+c]=1; xb[44]=1
    J = jac(xb); l0 = tools.forward(w,xb)[0]
    cols=[]
    for g,ix in G.items():
        cols.append(np.mean([(J[:,i]+J[:,12+i])/2.985 for i in ix],axis=0))
    cols.append(np.mean([(J[:,i]+J[:,i+8])/34.13 for i in UPEW],axis=0))
    Wc = np.array(cols).T
    Ws[c]=Wc; info[c]=(xb,l0,msk.sum())
Wsh = sum(Ws[c]*info[c][2] for c in range(4))/sum(info[c][2] for c in range(4))
Wsh_c = Wsh - Wsh.mean(0,keepdims=True)
print("shared weights centered (rows phases, cols", LN,")\n", np.round(Wsh_c,2))
B={}
for c in range(4):
    xb,l0,n = info[c]
    B[c] = l0 - Wsh@loads(xb)
    print("cur",c,"bias", np.round(B[c]-B[c].max(),2))
def dec_sh(x):
    c = int(np.argmax(x[40:44]))
    return int(np.argmax(B[c] + Wsh@loads(x)))
report(dec_sh, "shared load weights, per-cur bias")
def dec_sh2(x):
    c = int(np.argmax(x[40:44]))
    s = B[c] + Wsh@loads(x)
    if c!=2: s[c]=-99
    return int(np.argmax(s))
report(dec_sh2, "  + never keep 0/1/3")
