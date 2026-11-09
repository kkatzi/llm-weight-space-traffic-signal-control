from harness import *
GG = np.load("GG.npy")
own = {'N0':0,'N1':0,'S0':0,'S1':0,'N2':1,'S2':1,'E0':2,'E1':2,'W0':2,'W1':2,'E2':3,'W2':3}
print("per-vehicle push of each lane toward its own phase vs each other candidate phase (q+d, i.e. a halted vehicle; d-only in brackets)")
for l, a in own.items():
    out = []
    for p in range(4):
        if p == a: continue
        G = GG[p]
        for b in range(4):
            if b in (a, p): continue
            vqd = (G[a,ix['q_'+l]]+G[a,ix['d_'+l]] - G[b,ix['q_'+l]]-G[b,ix['d_'+l]])/3
            vd = (G[a,ix['d_'+l]] - G[b,ix['d_'+l]])/3
            out.append(f"c{p} vs{b}:{vqd:4.1f}({vd:4.1f})")
    print(f"{l} (phase {a}): " + "  ".join(out))
