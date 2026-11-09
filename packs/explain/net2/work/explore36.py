from common import *
exec(open('explore25.py').read().split("evalmodel(lambda p: GP[p], \"per-phase full\")")[0])
np.set_printoptions(linewidth=250, precision=3, suppress=True)
Gs = np.mean([GP[p] for p in range(4)], 0)
Gc = Gs - Gs.mean(0, keepdims=True)
ups = ['N0u','N1u','E0u','E1u','S0u','S1u','W0u','W1u']
print("upstream: per vehicle (d/34.13) and per halting vehicle ((q+d)/34.13), centered")
print("        " + " ".join(f"{u:>6s}" for u in ups))
for o in range(4): print(f"L{o} mov " + " ".join(f"{Gc[o, 32+k]/34.13:6.3f}" for k in range(8)))
for o in range(4): print(f"L{o} hlt " + " ".join(f"{(Gc[o, 24+k]+Gc[o, 32+k])/34.13:6.3f}" for k in range(8)))
print("t_green per second:", (Gc[:, 45]/60).round(3), " ok:", Gc[:, 44].round(2))
for p in range(4):
    G = GP[p]; Gc2 = G - G.mean(0, keepdims=True)
    print("phase", p, "upstream moving per vehicle", " ".join(f"{Gc2[o, 32+k]/34.13:6.3f}" for o in range(4) for k in [1, 3, 5, 7]))
