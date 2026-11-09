from harness import *
Gavg = np.load("Gavg.npy")
lanes = ['N0','N1','N2','E0','E1','E2','S0','S1','S2','W0','W1','W2']
ups = ['N0u','N1u','E0u','E1u','S0u','S1u','W0u','W1u']
Gq = np.array([[Gavg[r,ix['q_'+l]]+Gavg[r,ix['d_'+l]] for l in lanes] for r in range(4)])  # halted lane
Gd = np.array([[Gavg[r,ix['d_'+l]] for l in lanes] for r in range(4)])  # moving
Gu = np.array([[Gavg[r,ix['d_'+l]]+Gavg[r,ix['q_'+l]] for l in ups] for r in range(4)])
print("pairwise push (La - Lb) per HALTED vehicle (=1/3 lane), shared gradient")
print("        "+" ".join(f"{l:>5s}" for l in lanes))
for a,b in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]:
    print(f" L{a}-L{b}: "+" ".join(f"{(Gq[a,j]-Gq[b,j])/3:5.1f}" for j in range(12)))
print("pairwise push per MOVING vehicle")
for a,b in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]:
    print(f" L{a}-L{b}: "+" ".join(f"{(Gd[a,j]-Gd[b,j])/3:5.1f}" for j in range(12)))
print("pairwise push per upstream vehicle (density, 1/34.1)")
print("        "+" ".join(f"{l:>5s}" for l in ups))
for a,b in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]:
    print(f" L{a}-L{b}: "+" ".join(f"{(Gavg[a,ix['d_'+l]]-Gavg[b,ix['d_'+l]])/34.1:5.2f}" for l in ups))
