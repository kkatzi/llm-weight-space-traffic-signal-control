from harness import *
def grad(x):
    a1 = w["W1"]@x + w["b1"]; h1 = np.tanh(a1)
    a2 = w["W2"]@h1 + w["b2"]; h2 = np.tanh(a2)
    return w["W3"] @ ((1-h2**2)[:,None] * w["W2"]) @ ((1-h1**2)[:,None] * w["W1"])
lanes = ['N0','N1','N2','E0','E1','E2','S0','S1','S2','W0','W1','W2']
ups = ['N0u','N1u','E0u','E1u','S0u','S1u','W0u','W1u']
for p in [2,3]:
    m = np.where(ok & (cur==p))[0]
    other = [q for q in range(4) if q!=p]
    stay_margin = L[m,p] - L[m][:,other].max(1)
    sel = m[np.abs(stay_margin) < 2.5]
    print(f"cur {p}: {len(sel)} states near stay boundary")
    gs = []
    for i in sel:
        G = grad(X[i]); b = other[int(np.argmax(L[i,other]))]
        gs.append(G[p]-G[b])
    g = np.mean(gs,0)
    print("   d(stay margin) per halted vehicle (q+d)/3:", " ".join(f"{l}:{(g[ix['q_'+l]]+g[ix['d_'+l]])/3:+.1f}" for l in lanes))
    print("   per moving vehicle d/3:                   ", " ".join(f"{l}:{g[ix['d_'+l]]/3:+.1f}" for l in lanes))
    print("   upstream per vehicle:", " ".join(f"{l}:{g[ix['d_'+l]]/34.1:+.2f}" for l in ups))
    print("   t_green per 5 s:", g[45]*5/60)
