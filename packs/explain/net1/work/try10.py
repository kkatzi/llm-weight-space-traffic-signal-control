from try9 import *
LW = {0: 1.2, 1: 1.0, 2: 1.0, 3: 0.85}   # multiplier on the NS-left score, by current phase
def scores10(x):
    c = int(np.argmax(x[40:44])); s = scores9(x); s[1] *= LW[c]; return s
if __name__ == "__main__":
    pair_table(scores10)
