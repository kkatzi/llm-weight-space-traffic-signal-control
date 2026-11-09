from harness import *
def grad(x):
    a1 = w["W1"]@x + w["b1"]; h1 = np.tanh(a1)
    a2 = w["W2"]@h1 + w["b2"]; h2 = np.tanh(a2)
    return w["W3"] @ ((1-h2**2)[:,None] * w["W2"]) @ ((1-h1**2)[:,None] * w["W1"])
# Taylor expansion of the network around the mean ok state of each current phase
lin = {}
for p in range(4):
    m = ok & (cur==p)
    xm = X[m].mean(0); xm[40:44]=0; xm[40+p]=1; xm[44]=1
    lin[p] = (tools.forward(w, xm)[0], grad(xm), xm)
def dec_lin(x, allow_stay=True):
    p = int(np.argmax(x[40:44]))
    l0, G, xm = lin[p]
    l = l0 + G[:, :40] @ (x[:40]-xm[:40])
    if not allow_stay: l[p] = -1e9
    return int(np.argmax(l))
report(dec_lin, "taylor(mean,per cur)")
report(lambda x: dec_lin(x, False), "taylor no-stay")
# also: network but forbid staying
def net_nostay(x):
    l = tools.forward(w, x)[0]; p = int(np.argmax(x[40:44])); l[p]=-1e9; return int(np.argmax(l))
report(net_nostay, "net with stay forbidden")
