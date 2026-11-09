from try9 import *
B = np.array([[-99, 0.4, 0.0, -0.3],[0.45, 0.5, 0.0, 0.2],[0.0, -0.65, 1.22, -0.12],[0.0, -0.5, -0.75, 0.75]])
S_all = np.array([scores9(x) for x in X]) + B[cur]
r = S_all.argmax(1)
def show(mask, label, k=8):
    idx = np.where(mask)[0]
    print(f"--- {label}: {len(idx)} states")
    cols = ['q_N2','q_S2','d_N2','d_S2','q_E2','q_W2','d_E2','d_W2','q_N0','q_S0','q_S1','q_E1','q_W1']
    print("   " + " ".join(f"{c:>5s}" for c in cols) + "  ewUp nsUp | S(rule) | L(net)")
    for i in idx[:k]:
        x = X[i]
        ewUp = 34.1*sum(x[ix['d_'+l]] for l in ['E0u','E1u','W0u','W1u']); nsUp = 34.1*sum(x[ix['d_'+l]] for l in ['N0u','N1u','S0u','S1u'])
        print("   " + " ".join(f"{x[ix[c]]:5.2f}" for c in cols) + f"  {ewUp:4.1f} {nsUp:4.1f} | {np.round(S_all[i],1)} | {np.round(L[i],1)}")
    # mean feature differences vs correctly-classified 3s with similar margin
show(ok & (cur==2) & (net==1) & (r==3), "cur2 net=1 rule=3")
show(ok & (cur==2) & (net==0) & (r==3), "cur2 net=0 rule=3")
show(ok & (cur==2) & (net==3) & (r==0), "cur2 net=3 rule=0")
