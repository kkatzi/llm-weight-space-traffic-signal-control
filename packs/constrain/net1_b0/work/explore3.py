"""Statistics of the recorded run under the rules (no fitting): where do the rules bind?"""
import os, sys
from collections import Counter
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import tools

S = tools.states()
w = tools.load("start_weights.npz")
X = S["obs"].astype(np.float64)
R = S["rule_state"]
ep = S["episode"]; ts = S["time_s"]
lg = tools.forward(w, X)
net = lg.argmax(1)
since = R[:, 0:4]; legal = R[:, 4:8] > 0.5; overdue = R[:, 8:12] > 0.5
ph = R[:, 12:16]; tg = R[:, 16]; green = R[:, 17].astype(int)
N = len(X)

# executed phase = the phase that is green at the next decision (same episode)
nxt = np.full(N, -1)
for i in range(N - 1):
    if ep[i + 1] == ep[i]:
        nxt[i] = green[i + 1]

free = legal.all(1)                    # every phase legal -> network fully in charge
switch_pt = (tg >= 12) & ((green % 2 == 1) | (tg >= 17))
print("decisions:", N, " all-legal decisions:", free.sum())
print("net stays at all-legal decisions:", (net[free] == green[free]).mean())

# Episode lengths
for e in np.unique(ep):
    m = ep == e
    print(f"ep {e}: decisions {m.sum()}, last t {ts[m].max()}")

# green durations per phase (t_green at the decision where the switch is decided)
dur = Counter()
for i in range(N - 1):
    if nxt[i] >= 0 and nxt[i] != green[i]:
        dur[(green[i], int(tg[i]))] += 1
print("switch decided at (phase, t_green):", sorted(dur.items()))

# transitions
T = np.zeros((4, 4), int)
for i in range(N - 1):
    if nxt[i] >= 0 and nxt[i] != green[i]:
        T[green[i], nxt[i]] += 1
print("transition counts from->to:\n", T)
print("greens per phase:", T.sum(0))

# shield overrides that changed the executed phase
ill = ~legal[np.arange(N), net]
ok = X[:, 44] > 0.5
eff = ill & ok          # network request would have been honoured by the env, but the rules forbid it
print("rule-blocked honoured requests:", eff.sum())
kinds = Counter()
for i in np.where(eff)[0]:
    if overdue[i].any() and not legal[i, green[i]]:
        kinds["forced overdue (switch to overdue)"] += 1
    elif overdue[i].any():
        kinds["overdue present, still min green"] += 1
    elif legal[i].sum() == 1 and legal[i, green[i]]:
        kinds["min green hold"] += 1
    elif not legal[i, green[i]]:
        kinds["max green"] += 1
    else:
        kinds["other"] += 1
print(kinds)

# min-green holds of through phases at tg=12: how much demand is left on the held phase?
m = eff & (green % 2 == 0) & (tg == 12)
print("through holds at 12 s:", m.sum())
print("  halting on held phase:", Counter(ph[m, :][np.arange(m.sum()), green[m]].astype(int).tolist()))
dens_held = []
for i in np.where(m)[0]:
    g = green[i]
    lanes = [0, 1, 6, 7] if g == 0 else [3, 4, 9, 10]
    dens_held.append(X[i, 12 + np.array(lanes)].sum() * 3)
print("  vehicles (density) on held phase stop-line lanes: mean", np.mean(dens_held),
      Counter(np.round(dens_held).astype(int).tolist()))

# overdue forced switches
f = eff & overdue.any(1) & ~legal[np.arange(N), green]
print("forced overdue switches:", f.sum())
for i in np.where(f)[0][:40]:
    print(f"  ep{ep[i]} t{ts[i]} g{green[i]} tg{tg[i]:.0f} net{net[i]} lg{np.round(lg[i],1)} od{overdue[i].astype(int)} "
          f"ph{ph[i].astype(int)} ss{since[i].astype(int)}")
print("forced overdue phase:", Counter(np.where(overdue[f])[1].tolist()))
print("halting on forced phase:", Counter([int(ph[i][overdue[i]].max()) for i in np.where(f)[0]]))

# since_served distribution at service start, per phase
starts = [(green[i + 1], since[i, green[i + 1]]) for i in range(N - 1) if nxt[i] >= 0 and nxt[i] != green[i]]
for p in range(4):
    v = np.array([s for q, s in starts if q == p])
    print(f"phase {p}: services {len(v)}, since at service start: mean {v.mean():.1f}, pct>=90 {np.mean(v>=90):.2f}, quantiles {np.percentile(v,[10,50,90]).round()}")
