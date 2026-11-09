import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work")
from evalh import *
np.set_printoptions(precision=2, suppress=True, linewidth=250)
def jac(x):
    a1 = w["W1"]@x+w["b1"]; h1=np.tanh(a1)
    a2 = w["W2"]@h1+w["b2"]; h2=np.tanh(a2)
    return (w["W3"]*(1-h2**2)) @ (w["W2"]*(1-h1**2)) @ w["W1"]
# Taylor expansion per current phase at mean of own-run ok states (excluding deadlock)
lin = {}
for c in range(4):
    m = ok&(ctrl==0)&~dead&(cur==c)
    xb = X[m].mean(0)
    xb[40:44]=0; xb[40+c]=1; xb[44]=1
    J = jac(xb); l0 = tools.forward(w,xb)[0]
    lin[c]=(xb,J,l0)
def dec_lin(x):
    c = int(np.argmax(x[40:44])); xb,J,l0 = lin[c]
    return int(np.argmax(l0 + J@(x-xb)))
report(dec_lin, "Taylor at own mean per cur")
# Taylor using only lanes 0-39 (ignore t_green)
def dec_lin2(x):
    c = int(np.argmax(x[40:44])); xb,J,l0 = lin[c]
    d = x-xb; d[44:]=0
    return int(np.argmax(l0 + J@d))
report(dec_lin2, "Taylor ignoring t_green")
def dec_lin3(x):
    c = int(np.argmax(x[40:44])); xb,J,l0 = lin[c]
    d = x-xb; d[44:]=0; d[24:40]=0
    return int(np.argmax(l0 + J@d))
report(dec_lin3, "Taylor ignoring t_green+upstream")
