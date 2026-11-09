from common import *
import itertools
w = tools.load("start_weights.npz")
def scan(w, tgs=(12,17,22,37,60), verbose=True):
    bad = []; n = 0
    for mask in range(1, 4096):
        lanes = [l for l in range(12) if mask >> l & 1]
        need = set(lane_phase(l) for l in lanes)
        for p in range(4):
            if p in need: continue   # own lanes non-empty -> holding is service
            for tg in tgs:
                x = make_state(lanes, p, tg)
                a = int(tools.greedy(w, x)[0]); n += 1
                if a not in need:
                    bad.append((tuple(lanes), p, tg, a))
    return bad, n
if __name__ == "__main__":
    bad, n = scan(w)
    print('checked', n, 'bad', len(bad))
    from collections import Counter
    print('by current phase', Counter(b[1] for b in bad))
    print('by action', Counter(b[3] for b in bad))
    print('by (phase,action)', Counter((b[1],b[3]) for b in bad))
    print('by tg', Counter(b[2] for b in bad))
    # small configs
    sm = [b for b in bad if len(b[0])<=2]
    for b in sm[:60]: print(b)
