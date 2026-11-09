import numpy as np, tools
w0 = tools.load("start_weights.npz"); S = tools.states()
X, A, ep, t, c = S["obs"], S["act"], S["episode"], S["time_s"], S["controller"]
names = tools.names()
ok = X[:,44]>0.5; m0=(c==0)&ok; ph = X[:,40:44].argmax(1)
ns = [i for i,n in enumerate(names) if n.startswith(("q_","d_")) and n[2] in "NS"]
ew = [i for i,n in enumerate(names) if n.startswith(("q_","d_")) and n[2] in "EW"]
print(len(ns), len(ew))
Xs = X[m0].copy(); Xs[:,ns] = np.minimum(1, Xs[:,ns]*np.exp(0.05)); Xs[:,ew] = Xs[:,ew]*np.exp(-0.05)
lg0 = tools.forward(w0, X[m0]); lg1 = tools.forward(w0, Xs); d = lg1-lg0
print("mean logit change when NS features x1.05, EW x0.95:", np.round(d.mean(0),3), " => NS-vs-EW shift", round(((d[:,0]+d[:,1])-(d[:,2]+d[:,3])).mean()/2,3))
g0 = tools.greedy(w0, X[m0])
for delta in (0.25,0.5,1.0,1.5):
    for lab,vec in [("NS",[1,1,0,0]),("left",[0,1,0,1]),("EW",[0,0,1,1]),("through",[1,0,1,0])]:
        w=dict(w0); w["b3"]=w0["b3"]+delta*np.array(vec)
        g=tools.greedy(w,X[m0])
        print(f"b3 {lab:8s} +{delta}: flipped honoured {(g!=g0).mean():.3f}; chosen dist {np.round(np.bincount(g,minlength=4)/len(g),3)}")
print("original chosen dist", np.round(np.bincount(g0,minlength=4)/len(g0),3))
# how does the choice depend on scenario? states from the 6 net episodes: estimate demand level per episode by mean veh
veh = X[:,12:24].sum(1)*3 + X[:,32:40].sum(1)*34
for e in range(6):
    m=(c==0)&(ep==e)
    print("ep",e,"mean veh",round(veh[m].mean(),1), "chosen dist at honoured", np.round(np.bincount(A[m&ok],minlength=4)/(m&ok).sum(),2))
