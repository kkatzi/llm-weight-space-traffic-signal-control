import numpy as np, tools
w0 = tools.load("start_weights.npz"); S = tools.states()
X, A, ep, t, c = S["obs"], S["act"], S["episode"], S["time_s"], S["controller"]
names = tools.names(); idx={n:i for i,n in enumerate(names)}
ph = X[:,40:44].argmax(1)
# consecutive pairs within same controller/episode, 5 s apart, same phase both times and phase age grows (green continuing)
same = (c[1:]==c[:-1])&(ep[1:]==ep[:-1])&(t[1:]-t[:-1]==5)
i0 = np.where(same)[0]; i1 = i0+1
cont = (ph[i1]==ph[i0]) & (X[i1,45]>X[i0,45])
lanes = ["N0","N1","N2","E0","E1","E2","S0","S1","S2","W0","W1","W2"]
print("mean change of stop-line density during a continuing green, by phase (x100):")
for p in range(4):
    s = cont&(ph[i0]==p)
    d = (X[i1][s][:,12:24]-X[i0][s][:,12:24]).mean(0)*100
    print(p, dict(zip(lanes, np.round(d,1))))
# per controller: how often each phase chosen at switches and the mean served queue
for ci,name in enumerate(S["controller_names"]):
    m=(c==ci)&(X[:,44]>0.5)
    sw = m & (A!=ph)
    print(name, "switch-target dist:", np.round(np.bincount(A[sw],minlength=4)/sw.sum(),3), "n switches", sw.sum(), "stay rate", 1-sw.sum()/m.sum())
# pocket occupancy and upstream lane-1 density by controller
for ci,name in enumerate(S["controller_names"]):
    m=c==ci
    print(name, "mean d:", {l: round(X[m, idx["d_"+l]].mean(),3) for l in ["N2","S2","E2","W2","N1u","S1u","E1u","W1u","N0u","E0u"]})
    print(name, "mean total density (sum over all 20 lanes, proxy for vehicles):", round((X[m,12:24].sum(1)*3 + X[m,32:40].sum(1)*34).mean(),1))
