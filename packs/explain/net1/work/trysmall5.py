from trysmall import *
BA2 = np.array([[-99, 0.95, 0.0, 0.2],
                [0.3, 0.9, 0.56, 0.76],
                [0.0, -1.0, 1.4, -0.05],
                [0.0, -0.9, -0.6, 1.2]])
def dec(x):
    c = int(np.argmax(x[40:44])); return int(np.argmax(sc_small(x, up=0.1) + BA2[c]))
r = report(dec, "small, crossing-adjusted offsets")
for p in range(4): confusion(r,p)
