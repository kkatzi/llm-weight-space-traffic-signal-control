import numpy as np, tools
w0 = tools.load("start_weights.npz"); S = tools.states()
X, A, ep, t, c = S["obs"], S["act"], S["episode"], S["time_s"], S["controller"]
names = tools.names(); idx={n:i for i,n in enumerate(names)}
ph = X[:,40:44].argmax(1); ok = X[:,44]>0.5; tg = np.round(X[:,45]*60).astype(int)
qsum = X[:,:12].sum(1) + X[:,24:32].sum(1)
dsum_sl = X[:,12:24].sum(1); dsum_up = X[:,32:40].sum(1)*34/3   # in units of ~cars/3
veh = X[:,12:24].sum(1)*3 + X[:,32:40].sum(1)*34
m0 = (c==0)&ok
for lab, sel in [("no halting anywhere", qsum==0), ("<=1 halting-car-equiv", qsum<=0.34), ("veh<=3", veh<=3), ("veh<=6", veh<=6), ("veh>6", veh>6)]:
    m = m0&sel
    if m.sum()==0: continue
    print(f"{lab}: n={m.sum()} ({m.sum()/m0.sum():.2%}) stay rate {(A[m]==ph[m]).mean():.2f}; chosen dist {np.round(np.bincount(A[m],minlength=4)/m.sum(),2)}; current dist {np.round(np.bincount(ph[m],minlength=4)/m.sum(),2)}")
# by episode time (demand level): idle share per quarter
for lo,hi in [(0,900),(900,1800),(1800,2700),(2700,3600),(3600,6000)]:
    m = m0&(t>=lo)&(t<hi)
    print(f"t {lo}-{hi}: idle (veh<=3) share {(veh[m]<=3).mean():.3f}, mean veh {veh[m].mean():.1f}")
# when the junction is near-idle and the net switches, where does the next arriving demand appear? Count, for idle states, which phase's lanes have the most density
L = {0: ["N0","N1","S0","S1","N0u","N1u","S0u","S1u"], 1:["N2","S2"], 2: ["E0","E1","W0","W1","E0u","E1u","W0u","W1u"], 3:["E2","W2"]}
m = m0&(veh<=3)
print("idle states: chosen phase vs phase with max density (incl upstream):")
D = np.stack([sum(X[m, idx["d_"+l]]*(34 if l.endswith("u") else 3) for l in L[p]) for p in range(4)],1)
print(" chosen==argmax density:", (A[m]==D.argmax(1)).mean(), " all-empty share:", (D.sum(1)==0).mean())
# NS-bias magnitude proxy
Xs = X[m0].copy(); ns = [i for i,n in enumerate(names) if n[2]=="N" or n[2]=="S"]; ew=[i for i,n in enumerate(names) if n[2]=="E" or n[2]=="W"]
Xs[:,ns] = np.minimum(1, Xs[:,ns]*np.exp(0.05)); Xs[:,ew] = Xs[:,ew]*np.exp(-0.05)
lg0 = tools.forward(w0, X[m0]); lg1 = tools.forward(w0, Xs)
d = (lg1-lg0); print("mean logit change when NS features x1.05, EW x0.95:", np.round(d.mean(0),3), " => NS-vs-EW shift", round(((d[:,0]+d[:,1])-(d[:,2]+d[:,3])).mean()/2,3))
for delta in (0.25,0.5,1.0):
    w=dict(w0); w["b3"]=w0["b3"]+np.array([delta,delta,0,0])
    print(f"b3 NS +{delta}: flipped honoured decisions {(tools.greedy(w,X[m0])!=tools.greedy(w0,X[m0])).mean():.3f}")
    w=dict(w0); w["b3"]=w0["b3"]+np.array([0,delta,0,delta])
    print(f"b3 left +{delta}: flipped honoured decisions {(tools.greedy(w,X[m0])!=tools.greedy(w0,X[m0])).mean():.3f}")
