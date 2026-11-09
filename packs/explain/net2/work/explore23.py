from common import *
np.set_printoptions(linewidth=250, precision=2, suppress=True)
names = tools.names()
W1, b1, W2, b2, W3, b3 = [w[k] for k in ["W1","b1","W2","b2","W3","b3"]]
h1, h2 = tools.hidden(w, X[OK])
print("b3", b3)
# importance of h2 units for logits: W3 column norms times h2 std
imp2 = np.abs(W3).sum(0) * h2.std(0)
order2 = np.argsort(-imp2)
print("top layer-2 units: W3 column (effect on logits 0..3), std of h2, mean of h2")
for k in order2[:20]:
    print(k, W3[:, k].round(2), round(h2[:, k].std(), 2), round(h2[:, k].mean(), 2))
print()
# layer1 units: top inputs (scaled by typical range: stop-line 1, upstream 0.2)
scale = np.ones(46); scale[24:40] = 0.2; scale[45] = 0.3
print("layer-1 units: bias, top inputs by |w*scale|; std of h1")
for k in range(64):
    ws = W1[k] * scale
    top = np.argsort(-np.abs(ws))[:8]
    print(f"{k:2d} b={b1[k]:5.2f} sd={h1[:, k].std():.2f} | " + ", ".join(f"{names[i]}:{W1[k, i]:+.1f}" for i in top))
