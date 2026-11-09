# What the trained controller does

Phases: **0 NS_through**, **1 NS_left**, **2 EW_through** (main road), **3 EW_left**. A decision is made every
5 s. A switch is only honoured once the current green has run 10 s, which in practice means the 3rd decision
of a green (12 s).

## In one paragraph

The network is a **"minimum green, then go to the strongest call"** controller with a built-in
preference for the main-road through phase. Once the minimum green has elapsed it scores every phase from
the vehicles waiting at the stop lines and switches to the best one. It almost never extends the current
green: in its own runs it re-selected NS_through, NS_left or EW_left 1 time out of 987 decisions, because
their short stop-line lanes are empty after 12 s. So nearly every green lasts the minimum 12 s. The one
exception is **EW_through**, the default phase: it is held whenever the side streets are nearly empty. In
light traffic that hold can go on indefinitely while a few side-street vehicles wait.

## How the next phase is chosen (normal traffic)

Count vehicles on the 22 m stop-line lanes: a halted vehicle counts 1, a moving one ½ (each lane holds
about 3).

| phase | score |
|---|---|
| NS_through | N+S through/right vehicles − 0.6 × E/W left-turners |
| NS_left | 1.1 × N+S left-turners |
| EW_through | 0.6 × E+W through/right vehicles − 0.5 × N/S left-turners + 0.1 × vehicles queued on the E/W upstream lanes |
| EW_left | E+W left-turners |

Add the handicap for the current green (in vehicles; EW_through is the reference, 0) and take the highest:

| current green | → NS_through | → NS_left | → EW_through | → EW_left |
|---|---|---|---|---|
| NS_through | (keep −1.5) | −1.6 | 0 | −1.5 |
| NS_left | −1.8 | (keep −1.0) | 0 | −0.6 |
| EW_through | −2.6 | −1.9 | 0 (hold) | −0.8 |
| EW_left | −2.1 | −0.8 | 0 | (keep −0.6) |

What this means in practice:
- **EW_through is the default.** Its own queue counts only about 0.6 per vehicle, so it is not extended
  for its own traffic. It is chosen when no side movement has built up enough demand.
- **NS_through** needs about 2–2.6 more vehicles (in the units above) than its rivals. E/W left-turners
  waiting count against it.
- **NS_left** needs about 1.5–2 left-turners (its pockets hold 3 per approach). When it is called, it also
  pulls priority away from EW_through.
- **EW_left** needs only about 1 left-turner, and the least of all right after EW_through.
- Usual order, as shares of the decisions taken in each green (network's own runs, stall excluded):
  - EW_through → EW_left 55 %, NS_through 26 %, NS_left 13 %, held 6 %;
  - EW_left → NS_through 43 %, NS_left 30 %, EW_through 27 %;
  - NS_through → EW_through 52 %, NS_left 27 %, EW_left 20 %;
  - NS_left → EW_through 73 %.

  NS_left is inserted when its pockets fill. There is no fixed cycle.

## Light traffic (fewer than about 4 vehicles at the stop lines)

At an almost empty junction the network leans much harder on EW_through:
- With **no vehicles** it always goes to (or stays on) EW_through.
- While EW_through is green it **holds** if the only side demand is:
  - **up to 2 N/S through vehicles** (3 if a west left-turner is also waiting), except one halted car on
    each of N0 and S0, which is served while EW_through is younger than about 35 s;
  - **one west left-turner (W2)**, alone or with N/S through vehicles;
  - **one south left-turner (S2)** once EW_through has run 17 s or more. One north left-turner (N2) is
    served unless EW_through has run about a minute.
- **One east left-turner (E2)**, two N/S left-turners or three N/S through vehicles always end the hold.
- After NS_left in light traffic it returns to EW_through unless about 3 N/S through vehicles, or one
  left-turner on each of E2 and W2, are waiting.

## What it ignores

- **Queues on the N/S upstream lanes** (the 256 m sections). Only the 3-car stop-line lanes count for the
  side street. Queues on the E/W upstream lanes do count, in favour of EW_through.
- **How long anyone has waited.** There is no memory and no maximum red. The green time so far only
  matters in light traffic while EW_through is held.
- The **W0** lane (west through/right) barely counts. W2 counts about 40 % less than E2, and S2 about 30 %
  less than N2, for no traffic reason. These asymmetries are training artefacts.

## Where it may perform poorly

1. **Stalls at the end of the demand period.** Once arrivals stop, any leftover vehicles in the hold region
   above (for example 1–2 N/S through cars, or a lone W2 or S2 left-turner) are never served. EW_through
   stays green until the simulation cap. In one of the six recorded network episodes, 3 vehicles (N0, N1,
   W2) waited from 3625 s to the 5400 s cap, about 1.5 veh-h of extra TTS. The same effect delays
   side-street vehicles in light traffic until more arrive. A maximum red, or "serve any vehicle waiting
   more than X s", would remove it.
2. **Heavy side-street demand.** N/S queues longer than the 3-car stop-line lanes are invisible. E/W
   upstream queues push towards EW_through. Under heavy N/S demand the side street can be under-served.
3. **Minimum-green cycling in heavy traffic.** Almost every green is the 12 s minimum, followed by 3 s of
   yellow. That is a lot of lost time per cycle, although it still beat the tuned fixed-time and actuated
   plans on the dev episodes.
4. **West and south left-turners** are served a little later than east and north ones (see the
   asymmetries above).

The rule files (`rules_small.py`, `rules_full.py`) reproduce this behaviour, flaws included. They do not
fix it.
