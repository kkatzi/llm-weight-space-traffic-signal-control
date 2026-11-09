from harness import *
np.set_printoptions(precision=1, suppress=True, linewidth=220)
def grad(x):
    a1 = w["W1"]@x + w["b1"]; h1 = np.tanh(a1)
    a2 = w["W2"]@h1 + w["b2"]; h2 = np.tanh(a2)
    return w["W3"] @ ((1-h2**2)[:,None] * w["W2"]) @ ((1-h1**2)[:,None] * w["W1"])
lanes = ['N0','N1','N2','E0','E1','E2','S0','S1','S2','W0','W1','W2']
ups = ['N0u','N1u','E0u','E1u','S0u','S1u','W0u','W1u']
pairs = {0:[(1,2),(2,3),(1,3)], 1:[(2,3),(0,2),(0,3)], 2:[(0,3),(1,3),(0,1),(2,3)], 3:[(0,2),(0,1),(1,2),(3,0)]}
print("columns: stop-line lanes (q+d effect) | upstream (d effect, per 0.1)")
print("        "+" ".join(f"{l:>5s}" for l in lanes)+" | "+" ".join(f"{l:>5s}" for l in ups))
for p in range(4):
    m = ok & (cur==p)
    G = np.mean([grad(x) for x in X[m]], axis=0)
    for a,b in pairs[p]:
        g = G[a]-G[b]
        sl = [g[ix['q_'+l]]+g[ix['d_'+l]] for l in lanes]
        up = [0.1*g[ix['d_'+l]] + 0.1*g[ix['q_'+l]]*0 for l in ups]
        print(f"c{p} {a}-{b}: "+" ".join(f"{v:5.1f}" for v in sl)+" | "+" ".join(f"{v:5.2f}" for v in up))
