from try9 import *
B = np.array([[-99, 0.5, 0.0, -0.45],
              [0.5, 0.55, 0.1, 0.1],
              [0.0, -0.4, 1.1, -0.12],
              [0.0, -0.7, -0.55, 0.75]])
def dec(x):
    c = int(np.argmax(x[40:44])); return int(np.argmax(scores9(x) + B[c]))
r = report(dec, "full, second crossing adjustment")
