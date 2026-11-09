import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work")
from evalh import *
np.set_printoptions(precision=2, suppress=True, linewidth=250)
def scores_q(x):
    return np.array([(x[0]+x[1]+x[6]+x[7])/2, x[2]+x[8], (x[3]+x[4]+x[9]+x[10])/2, x[5]+x[11]])
def scores_d(x):
    return np.array([(x[12]+x[13]+x[18]+x[19])/2, x[14]+x[20], (x[15]+x[16]+x[21]+x[22])/2, x[17]+x[23]])
def mk(scf, order_bias):
    def dec(x):
        s = scf(x) + order_bias
        c = int(np.argmax(x[40:44]))
        s[c] = -9
        return int(np.argmax(s))
    return dec
# tie-break biases: tiny, favouring 2 then 3 then 0 then 1 ?
tb = np.array([0.002,0.001,0.004,0.003])
r = report(mk(scores_q, tb), "q, exclude current")
r = report(mk(scores_d, tb), "d, exclude current")
r = report(mk(lambda x: scores_q(x)+scores_d(x), tb), "q+d, exclude current")
r = mk(scores_q, tb); rr = np.array([r(x) for x in X])
m = ok&(ctrl==0)&~dead
C = np.zeros((4,4,4),int)
for a,b,c in zip(cur[m],net[m],rr[m]): C[a,b,c]+=1
for a in range(4): print("cur",a,"\n rows=net, cols=rule\n",C[a])
