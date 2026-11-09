from try7 import *
B = np.array([[-99, 0.55, 0.0, -0.35],
              [0.4, 0.45, 0.0, 0.2],
              [0.0, -0.7, 1.3, -0.1],
              [0.0, -0.4, -0.7, 0.8]])
def dec(x):
    c = int(np.argmax(x[40:44])); return int(np.argmax(scores(x) + B[c]))
r = report(dec, "pairwise-boundary offsets")
for p in range(4): confusion(r, p)
