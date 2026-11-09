from trysmall import *
BA = np.array([[-99, 0.75, 0, 0.7],[0, 0.9, 0.26, 0.76],[0, -0.8, 2.0, 0.28],[0, -0.7, -0.5, 1.2]])
def decA(x):
    c = int(np.argmax(x[40:44])); return int(np.argmax(sc_small(x, up=0.0) + BA[c]))
def decA_up(x):
    c = int(np.argmax(x[40:44])); return int(np.argmax(sc_small(x, up=0.1) + BA[c]))
r = report(decA, "S-A no upstream (offsets from up=0.1 table)")
r = report(decA_up, "S-A with ew upstream 0.1")
for p in range(4): confusion(r,p)
