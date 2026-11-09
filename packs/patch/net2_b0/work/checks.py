import numpy as np, sys, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import tools
from collections import Counter
np.set_printoptions(linewidth=200, precision=3, suppress=True)
w0 = tools.load("start_weights.npz"); w = tools.load("work/weights.npz")
S = tools.states(); X = S["obs"].astype(np.float64); ctl = S["controller"]; t = S["time_s"]
ok = X[:, 44] > 0.5
ewE = (X[:, 17] == 0) & (X[:, 23] == 0); nsE = (X[:, 14] == 0) & (X[:, 20] == 0)
print("edit size:", tools.edit_size(w, w0))
for k in tools.KEYS:
    print(f"  {k}: {int((w[k]!=w0[k]).sum())} changed")
K3, J3, K1, J1 = 50, 1, 38, 56
h1, h2 = tools.hidden(w, X)
print("detector EW (h1[50]) values:", np.unique(h1[:,K3].round(6)), " on-rate:", (h1[:,K3]>0).mean().round(3))
print("detector NS (h1[38]) values:", np.unique(h1[:,K1].round(6)), " on-rate:", (h1[:,K1]>0).mean().round(3))
print("detector EW on == (ok & EW pockets empty):", np.array_equal(h1[:,K3]>0, ok&ewE))
print("detector NS on == (ok & NS pockets empty):", np.array_equal(h1[:,K1]>0, ok&nsE))
a1 = np.tanh(h1 @ w["W2"].T + w["b2"])
pre = h1 @ w["W2"].T + w["b2"]
print("relay EW pre-act range off/on:", pre[h1[:,K3]<0, J3].min().round(2), pre[h1[:,K3]<0, J3].max().round(2), "/", pre[h1[:,K3]>0, J3].min().round(2), pre[h1[:,K3]>0, J3].max().round(2))
print("relay NS pre-act range off/on:", pre[h1[:,K1]<0, J1].min().round(2), pre[h1[:,K1]<0, J1].max().round(2), "/", pre[h1[:,K1]>0, J1].min().round(2), pre[h1[:,K1]>0, J1].max().round(2))
# worst-case relay rest over ANY input: |sum W2[j,i]| bound with h1 in [-1,1]
for j,k in [(J3,K3),(J1,K1)]:
    rest_bound = np.abs(np.delete(w["W2"][j], k)).sum()
    print(f"relay {j}: worst-case |rest| <= {rest_bound:.2f} (+|b2|={abs(w['b2'][j]):.2f}) vs G=20")
a0 = tools.greedy(w0, X); a = tools.greedy(w, X)
print("picks 3 with EW pockets empty (ok):", int((ok&ewE&(a==3)).sum()), " was", int((ok&ewE&(a0==3)).sum()))
print("picks 1 with NS pockets empty (ok):", int((ok&nsE&(a==1)).sum()), " was", int((ok&nsE&(a0==1)).sum()))
for c in range(3):
    m = ok & (ctl==c)
    ch = m & (a!=a0)
    intended = ch & ((ewE&(a0==3)) | (nsE&(a0==1)))
    print(f"{S['controller_names'][c]}: honoured {m.sum()}, changed {ch.sum()} (intended {intended.sum()}, side {(ch&~intended).sum()}), non-honoured changed {int(((a!=a0)&~ok&(ctl==c)).sum())}")
# side effects in network episodes
L0 = tools.forward(w0, X); L = tools.forward(w, X)
m = ok&(ctl==0)&(a!=a0)&~((ewE&(a0==3))|(nsE&(a0==1)))
for i in np.where(m)[0]:
    print("  side effect t=",t[i],"cur",X[i,40:44].argmax(),"old",a0[i],"new",a[i],"old lg",L0[i],"new lg",L[i])
# logit drift on states where gates are off
off = ~(ok&(ewE|nsE))
print("max |logit change| where gates off:", np.abs(L-L0)[off].max().round(3), " mean:", np.abs(L-L0)[off].mean().round(3))
print("trainability original:", tools.trainability(w0))
print("trainability edited:  ", tools.trainability(w))
# synthetic stress test: random observations
rng = np.random.default_rng(0)
Z = rng.random((200000, 46))
Z[:, 40:44] = np.eye(4)[rng.integers(0,4,len(Z))]; Z[:,44] = rng.integers(0,2,len(Z))
for cols in [[2,8,14,20],[5,11,17,23]]:
    pass
# discretise pocket densities to vehicle counts sometimes zero
for c in [14,20,17,23]:
    Z[:, c] = rng.integers(0,4,len(Z))/2.987; Z[:,c]=np.minimum(Z[:,c],1)
Zok = Z[:,44]>0.5
zew = (Z[:,17]==0)&(Z[:,23]==0); zns=(Z[:,14]==0)&(Z[:,20]==0)
az = tools.greedy(w, Z); az0 = tools.greedy(w0, Z)
print("synthetic: picks 3 w/ EW empty & ok:", int((Zok&zew&(az==3)).sum()), "| picks 1 w/ NS empty & ok:", int((Zok&zns&(az==1)).sum()))
hz1,_ = tools.hidden(w, Z)
print("synthetic detector exactness:", np.array_equal(hz1[:,K3]>0, Zok&zew), np.array_equal(hz1[:,K1]>0, Zok&zns), "min |h| det:", np.abs(hz1[:,[K3,K1]]).min().round(6))
