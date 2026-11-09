from harness import *
GG = np.load("GG.npy")
Gavg = np.mean([GG[p]-GG[p].mean(0) for p in range(4)],0)
np.save("Gavg.npy", Gavg)
lanes = ['N0','N1','N2','E0','E1','E2','S0','S1','S2','W0','W1','W2']
ups = ['N0u','N1u','E0u','E1u','S0u','S1u','W0u','W1u']
print("shared centred gradient, stop-line lanes, per full lane: q | d")
print("     "+" ".join(f"{l:>11s}" for l in lanes))
for q in range(4):
    print(f"L{q}: "+" ".join(f"{Gavg[q,ix['q_'+l]]:5.1f}|{Gavg[q,ix['d_'+l]]:5.1f}" for l in lanes))
print("upstream, per full lane: q | d")
print("     "+" ".join(f"{l:>11s}" for l in ups))
for q in range(4):
    print(f"L{q}: "+" ".join(f"{Gavg[q,ix['q_'+l]]:5.1f}|{Gavg[q,ix['d_'+l]]:5.1f}" for l in ups))
