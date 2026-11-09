import try10
from try10 import *
B2row = [0.0, -0.4, 1.1, -0.12]; B1row = [0.45, 0.5, 0.0, 0.2]
variants = {
 "V1 LW0=1.2 LW3=0.85": ({0:1.2,1:1,2:1,3:0.85}, np.array([[-99,-0.42,0,-0.17], B1row, B2row, [0,0.35,-0.7,0.75]])),
 "V2 LW0=1.2 LW3=1.0":  ({0:1.2,1:1,2:1,3:1.0},  np.array([[-99,-0.42,0,-0.17], B1row, B2row, [0,-0.5,-0.75,0.75]])),
}
for name,(lw,B) in variants.items():
    try10.LW.update(lw)
    def dec(x, B=B):
        c = int(np.argmax(x[40:44])); return int(np.argmax(scores10(x) + B[c]))
    report(dec, name)
