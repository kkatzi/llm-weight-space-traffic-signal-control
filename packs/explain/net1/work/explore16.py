from harness import *
def grad(x):
    a1 = w["W1"]@x + w["b1"]; h1 = np.tanh(a1)
    a2 = w["W2"]@h1 + w["b2"]; h2 = np.tanh(a2)
    return w["W3"] @ ((1-h2**2)[:,None] * w["W2"]) @ ((1-h1**2)[:,None] * w["W1"])
srt = np.sort(L,1); gap = srt[:,-1]-srt[:,-2]
lanes = ['N0','N1','N2','E0','E1','E2','S0','S1','S2','W0','W1','W2']
ups = ['N0u','N1u','E0u','E1u','S0u','S1u','W0u','W1u']
GG = {}
for p in range(4):
    m = ok & (cur==p) & (gap < 2.0)
    G = np.mean([grad(x) for x in X[m]], axis=0)
    GG[p] = G
    cand = [q for q in range(4) if q!=p]
    Gc = G - G[cand].mean(0)   # centred over the 3 switch candidates
    print(f"=== cur {p} (n={m.sum()}), centred over candidates {cand}; per vehicle (=1/3 of lane) for stop lanes: q-part | d-part")
    print("      "+" ".join(f"{l:>9s}" for l in lanes))
    for q in range(4):
        print(f" L{q}: "+" ".join(f"{Gc[q,ix['q_'+l]]/3:4.1f}|{Gc[q,ix['d_'+l]]/3:4.1f}" for l in lanes))
    print("  upstream d per vehicle (1/34):   "+" ".join(f"{l}" for l in ups))
    for q in range(4):
        print(f" L{q}: "+" ".join(f"{Gc[q,ix['d_'+l]]/34.1:5.2f}" for l in ups) + "   q-up: " + " ".join(f"{Gc[q,ix['q_'+l]]/34.1:5.2f}" for l in ups))
np.save("GG.npy", np.stack([GG[p] for p in range(4)]))
