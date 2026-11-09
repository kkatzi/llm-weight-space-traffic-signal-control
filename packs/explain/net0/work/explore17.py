import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work")
from evalh import *
np.set_printoptions(precision=2, suppress=True, linewidth=250)
names=tools.names()
def jac(x):
    a1 = w["W1"]@x+w["b1"]; h1=np.tanh(a1)
    a2 = w["W2"]@h1+w["b2"]; h2=np.tanh(a2)
    return (w["W3"]*(1-h2**2)) @ (w["W2"]*(1-h1**2)) @ w["W1"]
# average jacobian over own ok non-dead states (exact per-state, then mean)
m = ok&(ctrl==0)&~dead
Js = np.array([jac(x) for x in X[m]])
J = Js.mean(0)
print("per halted vehicle (q+d)/2.985 effect, centered so that min over phases... shown as diff vs phase-2 logit")
for i in range(12):
    v = (J[:,i]+J[:,12+i])/2.985
    mv = J[:,12+i]/2.985
    print(f"{names[i][2:]:4s} halted: logit_p - logit_2 = {np.round(v - v[2],2)}  moving: {np.round(mv-mv[2],2)}")
for j in range(8):
    v = (J[:,24+j]+J[:,32+j])/34.13
    print(f"{names[24+j][2:]:4s} up halted per veh: {np.round(v-v[2],3)}")
v=J[:,45]; print("t_green full-range", np.round(v-v[2],2))
