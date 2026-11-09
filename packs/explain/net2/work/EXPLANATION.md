# What the trained controller does (start_weights.npz)

Short version: **every phase gets only the minimum green, about 12 s.** At each switch the network picks
the next phase from the queues at the stop line. It strongly prefers the main road's (EW) through phase, and it
has fixed habits about which phase follows which. The only real exception is that it sometimes holds the EW
left-turn phase in light traffic.

## 1. When it switches

* The controller decides every 5 s, and a request to change is honoured only after 10 s of green. A new green is
  2 s old at its first decision, so the first honoured decision comes at 12 s of green.
* At that decision the network switches in 97 % of cases. In its own six recorded runs it made 1503 honoured
  decisions. It extended a green only 42 times: 39 times in EW-left and 3 times in EW-through. It never extended
  NS-through or NS-left. 1455 of the 1503 decisions came at 12 s of green.
* So in practice the network runs no cycle and no green extension. It gives **12 s of green and 3 s of yellow,
  then the next phase**, with the order chosen by demand. Elapsed green time (`t_green`) plays no role.
* **The exception (holding EW-left):** the network keeps EW-left for another 5 s when both conditions hold:
  * a left-turn pocket on E or W still holds a car, or cars are still arriving into it;
  * demand everywhere else is light.

  In the rules: `ew_left ≥ 1` and the best alternative's score is below `5 + 3·ew_left` (`3.5 + 2.5·ew_left`
  in the full rules). This happens in about 7–9 % of EW-left decisions, almost always in light traffic.

## 2. Which phase it switches to

The network gives each of the three other phases a demand score and picks the highest one. The current phase
always counts as "just served" and is excluded.

**Lane load.** Each stop-line lane (3 cars long) is measured as *vehicles on the lane + halting vehicles on it*.
A stopped car counts 2 and a moving car counts 1, so a lane load runs from 0 to 6.

**Demand per movement group** (the weights come from the network's sensitivities):

| group | lanes and weights | notes |
|---|---|---|
| NS-through | N1 + 0.75·(S0 + S1) + 0.25·N0 | the north right+straight lane N0 barely counts |
| NS-left | N2 + S2 | |
| EW-through | E1 + 1.5·W1 + 0.75·(E0 + W0) | the west straight lane W1 counts most |
| EW-left | E2 + W2 | |

Per vehicle, a car in a left pocket counts about 1.3 times a car in a typical through lane (W1 is the exception).
A left pocket is one lane per approach, while the through movement has two lanes per approach. Per lane, the
network therefore treats a full left pocket and full through lanes as roughly equally urgent.

**Phase scores** (rules_small.py; rules_full.py is similar with a few more terms):

* NS-through = NS-through demand (+ 0.2·EW-left)
* NS-left = 1.35·NS-left demand
* EW-through = 1.25·EW-through demand + 0.5·NS-through demand − 0.55·NS-left demand
* EW-left = 1.55·EW-left demand + 0.45·EW-through demand − 0.25·NS-through − 0.15·NS-left

Cars on the upstream lanes (still approaching) add a little. A car upstream adds 0.1–0.2 to a phase score,
against 1.35–3 for a car waiting at the stop line. Cars on N1u and S1u, the lanes that feed the NS left pockets,
add to both NS phases, most of all to NS-left. Cars on the EW upstream lanes add mostly to EW-left.

**Phase-transition bonus.** A bonus or penalty that depends on which phase has just ended is added to the score.
Units are lane loads:

| just ended | NS-through | NS-left | EW-through | EW-left |
|---|---|---|---|---|
| NS-through | – | **+3.0** | −1.05 | −1.95 |
| NS-left | +1.1 | – | −0.7 | −0.45 |
| EW-through | +0.65 | 0 | – | −0.6 |
| EW-left | **+2.8** | −1.1 | −1.7 | – |

**How it plays out.** These are the transitions in the network's own runs, as shares of decisions:

| just ended | → NS-through | → NS-left | → EW-through | → EW-left | hold |
|---|---|---|---|---|---|
| NS-through | – | 39 % | 19 % | 42 % | 0 |
| NS-left | 9 % | – | 56 % | 35 % | 0 |
| EW-through | 46 % | 24 % | – | 29 % | 1 % |
| EW-left | 23 % | 10 % | 58 % | – | 9 % |

* **Main-road priority.** When it chooses between the two through phases, EW-through demand counts about
  2.5 times as much as NS-through demand. A score unit of NS-through costs EW-through only 0.5 instead of 1.
  * After EW-left: NS-through lanes holding 3 halting cars each still lose to EW-through lanes holding one car each.
  * After NS-left the effect is even stronger. EW-through wins as soon as each of its lanes has a car, and the
    network goes back to NS-through only if the main road's through lanes are empty.
* **Left turn after its own through phase.** After NS-through, NS-left gets a large bonus (+3), so NS-through is
  usually followed by NS-left or by EW-left. EW-through rarely follows, because the main-road through lanes were
  usually served one or two phases earlier.
* **After EW-through it is close to "largest demand wins"** (bonuses within ±0.65). It then goes to NS-through
  (46 %), EW-left (29 %) or NS-left (24 %).
* Typical sequences in the recorded runs look like 2 → 0 → 3 → 2 → 1 → 3 → 2 → 0 → 1 → 2 … EW-through returns
  every second or third phase. Each left-turn phase is used every few phases, which is enough because a pocket
  holds only 3 cars and a 12 s green clears it.

## 3. What it ignores

* **Green time already elapsed** (`t_green`). It switches at the first chance whatever the age of the green. The
  rare EW-left holds depend on demand, not on time.
* **Upstream queues** (halting cars on the 256 m upstream lanes) carry almost no weight. Upstream *densities*
  have a weak effect: about a tenth of a stop-line car each, mostly through lanes N1u, S1u, E1u and W1u. In
  practice the network sees only the last 22 m before the stop line, about 3 cars per lane.
* **The north right+straight lane N0** counts only a quarter as much as N1. S0 and S1 count three quarters, and
  W1 counts 1.5 times E1. These asymmetries have no basis in the junction or the demand, which are symmetric. They
  are habits learned in training.
* Decisions before the minimum green (they are not executed anyway).

## 4. Where it may perform poorly

* **No green extension under heavy load.** Every green is 12 s followed by 3 s of yellow, so about 20 % of the
  time is yellow, plus a start-up loss at every switch. At the tested demand this still beats the tuned actuated
  controller (55.0 vs 56.6 veh-h in the high scenario). As demand rises further, the fixed 15 s switching overhead
  takes capacity that longer greens on the main road would recover.
* **Short sight.** The stop-line lanes saturate at 3 cars, and the network largely ignores the upstream queue. It
  cannot tell a 3-car queue from a 30-car queue. When everything is congested, the choice comes from the
  transition habits rather than from queue length. In synthetic fully saturated states the network
  alternates NS-through ↔ EW-through (empty upstream) or EW-through ↔ EW-left (busy upstream), which would starve
  the left turns or the side road. These states are outside what it met in training. Treat them as a warning,
  not a measured result.
* **Side-road through traffic waits.** Because of the main-road priority, NS-through vehicles are often passed over
  for an EW-through phase with only a few cars. This is fine for total delay at the trained demand split
  (750 vs 450 veh/h). It would be poor if side-road demand grew or the split reversed.
* **Lane asymmetries** (N0 under-weighted, W1 over-weighted). These would hurt if, for example, the northbound
  right-turn flow grew.
* **Holding EW-left in light traffic.** It sometimes holds EW-left for an extra 5 s, serving one late left-turner
  while a few cars wait elsewhere. This is a small loss, but it is the only time the network extends a green.

## 5. Rule files

* `rules_small.py` (6 comparisons): the four demand groups, upstream counts, the bonus table, argmax, and the
  EW-left hold. It agrees with the network on 90.2 % of the recorded honoured decisions, and 90.4 % on the
  network's own runs.
* `rules_full.py` (6 comparisons, longer score formulas): the same logic, with EW split into straight lanes
  (E1, W1) and right+straight lanes (E0, W0). The network uses these two sets differently: E0 and W0 favour
  EW-left over NS-left, while E1 and W1 count mainly for EW-through. Agreement is 91.2 % (91.5 % on the
  network's own runs).

Most remaining disagreements are close calls for the network. On those states the median logit margin between
its choice and the rule's choice is about 1.1, and 70–74 % of them are within 2 logits. On average the network's
own (stochastic) policy gives the rule's pick a probability of about 0.24.
