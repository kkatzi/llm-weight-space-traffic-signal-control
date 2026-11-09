from harness import *
print("mean halting q per lane at ok decisions, by current green (served lanes should be ~0)")
lanes = ['N0','N1','N2','E0','E1','E2','S0','S1','S2','W0','W1','W2']
print("cur  "+" ".join(f"{l:>5s}" for l in lanes))
for p in range(4):
    m = ok & (cur==p)
    print(f" {p}   "+" ".join(f"{X[m,ix['q_'+l]].mean():5.2f}" for l in lanes))
print("mean density d per lane")
for p in range(4):
    m = ok & (cur==p)
    print(f" {p}   "+" ".join(f"{X[m,ix['d_'+l]].mean():5.2f}" for l in lanes))
