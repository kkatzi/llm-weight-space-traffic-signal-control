from try11 import *
B = np.array([[-99, 0.5, 0.0, -0.3],
              [0.55, 0.6, 0.0, 0.25],
              [0.0, -0.5, 0.95, -0.26],
              [0.0, -0.6, -0.88, 0.7]])
def dec(x):
    c = int(np.argmax(x[40:44])); return int(np.argmax(scores11(x) + B[c]))
r = report(dec, "upstream 0.12/0.05 + offsets")
tot = X[:,12:24].sum(1)*3 + X[:,32:40].sum(1)*34.1
for lo,hi in [(0,10),(10,20),(20,30),(30,40),(40,200)]:
    m = ok & (tot>=lo)&(tot<hi); print(f"  total in [{lo},{hi}): fidelity {(r[m]==net[m]).mean():.3f}")
for p in range(4): confusion(r,p)
