from common import *
exec(open('explore38.py').read().split("S_small = ")[0])
S_full = np.load('S_tmp.npy'); Bf = bonuses(S_full)
m3 = OK & (PH == 3)
L = S_full + Bf[3]; Lc = L.copy(); Lc[:, 3] = -1e9
best = Lc.max(1)
stay = LG.argmax(1) == 3
print("P(stay | phase 3) by best candidate score (rows) and ew_left load D (cols)")
bb = [-10, 0, 2, 4, 6, 8, 10, 14, 50]; dd = [0, 1, 2, 3, 4, 13]
for i in range(len(bb)-1):
    row = []
    for j in range(len(dd)-1):
        mm = m3 & (best >= bb[i]) & (best < bb[i+1]) & (ew_left >= dd[j]) & (ew_left < dd[j+1])
        row.append(f"{(stay & mm).sum():3d}/{mm.sum():<4d}" if mm.sum() else "   .    ")
    print(f"  {bb[i]:>4}..{bb[i+1]:<4}: " + " ".join(row))
print("same restricted to network's own runs (c0)")
for i in range(len(bb)-1):
    row = []
    for j in range(len(dd)-1):
        mm = m3 & (CTRL == 0) & (best >= bb[i]) & (best < bb[i+1]) & (ew_left >= dd[j]) & (ew_left < dd[j+1])
        row.append(f"{(stay & mm).sum():3d}/{mm.sum():<4d}" if mm.sum() else "   .    ")
    print(f"  {bb[i]:>4}..{bb[i+1]:<4}: " + " ".join(row))
# also: which candidate wins, when stay
print("rule's best candidate on stay states:", np.bincount(Lc[m3 & stay].argmax(1), minlength=4))
# ns demand
nsd = ns_thru + ns_left
print("P(stay) by ns demand (ns_thru+ns_left) and ew_thru")
nb = [0, 2, 4, 6, 8, 10, 15, 60]; eb = [0, 2, 4, 6, 8, 10, 15, 60]
for i in range(len(nb)-1):
    row = []
    for j in range(len(eb)-1):
        mm = m3 & (nsd >= nb[i]) & (nsd < nb[i+1]) & (ew_thru >= eb[j]) & (ew_thru < eb[j+1])
        row.append(f"{(stay & mm).sum():3d}/{mm.sum():<4d}" if mm.sum() else "   .    ")
    print(f"  {nb[i]:>3}..{nb[i+1]:<3}: " + " ".join(row))
