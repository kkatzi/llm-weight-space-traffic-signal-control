from try9 import *
B = np.array([[-99, 0.4, 0.0, -0.3],
              [0.45, 0.44, 0.0, 0.2],
              [0.0, -0.65, 1.22, -0.12],
              [0.0, -0.5, -0.75, 0.75]])
def dec(x):
    c = int(np.argmax(x[40:44])); return int(np.argmax(scores9(x) + B[c]))
r = report(dec, "lane-level scores + offsets")
for p in range(4): confusion(r, p)
