# v3 results (30 held-out test episodes; % = change in mean TTS, 95 % CI)

**L2**: fixed 48.38, actuated 46.38, PPO 44.91 (sd over seeds 0.24); PPO vs actuated -3.2 % [-5.0, -1.3], vs fixed -7.2 %

**L2W**: fixed 62.76, actuated 64.02, PPO 58.69 (sd over seeds 0.19); PPO vs actuated -8.3 % [-11.6, -5.1], vs fixed -6.5 %

**L2_rules**: fixed 49.85, actuated 48.74, PPO 46.97 (sd over seeds 0.07); PPO vs actuated -3.6 % [-4.8, -2.5], vs fixed -5.8 %

Runs failing the convergence criterion: ['L2W/ppo_s203']; L2W/ppo_s203 is superseded by L2W/ppo_s203_r1 (converged: True, 4.8M steps), which the analysis uses


## Initialise

- zero-shot 1: TTS 45.80; vs fixed -5.3 %, vs actuated -1.3 %
- zero-shot 2: TTS 48.12; vs fixed -0.6 %, vs actuated +3.7 %
- zero-shot 3: TTS 46.22; vs fixed -4.5 %, vs actuated -0.4 %
- warm vs scratch, first50_mean_delay: 20.4 vs 49.4 (-58.8 % [-85.5, -32.0])
- warm vs scratch, first1M_total_delay: 26314.2 vs 28093.8 (-6.3 % [-10.4, -2.3])
- warm vs scratch, final_tts: 45.4 vs 44.9 (+1.0 % [-0.0, +2.0])

## Transfer

- zero-shot 0: TTS 58.55; vs fixed -6.7 %, vs actuated -8.6 %, vs padding -98.3 %
- zero-shot 1: TTS 59.51; vs fixed -5.2 %, vs actuated -7.0 %, vs padding -97.2 %
- zero-shot 2: TTS 60.06; vs fixed -4.3 %, vs actuated -6.2 %, vs padding -95.3 %
- warm vs scratch, first50_mean_delay: 34.3 vs 99.8 (-65.6 % [-100.7, -30.6])
- warm vs scratch, first1M_total_delay: 39848.6 vs 44514.2 (-10.5 % [-15.6, -5.3])
- warm vs scratch, final_tts: 58.2 vs 58.7 (-0.8 % [-1.3, -0.2])

## Patch (vs each target; pooled over 3 targets)

- budget 0: -0.13 % [-0.50, +0.23], wins 19/30
- budget 5: -0.07 % [-0.58, +0.44], wins 18/30
- budget 10: -0.43 % [-0.92, +0.06], wins 19/30
- es_seed0: +0.27 % [-0.45, +0.99]
- es_seed1: -0.09 % [-0.48, +0.30]
- wrapper: +0.02 % [-0.02, +0.05]
- finetune_300k: +0.23 % [-0.36, +0.82]
- finetune_1M: -0.05 % [-0.57, +0.46]
- writer sonnet: +0.41 % [-0.12, +0.94]
- writer haiku: +1.39 % [+0.55, +2.23]
- writer fable: -0.17 % [-0.69, +0.34]

## Patch under a failed detector (W2; recovered = share of the fault's cost)

- net0: healthy 45.2, faulty 48.7
- net1: healthy 44.8, faulty 376.3
- net2: healthy 44.6, faulty 47.5
- agents_b0: pooled vs faulty -71.1 %; recovered 96 % / 100 % / 57 %
- agents_b5: pooled vs faulty -71.4 %; recovered 100 % / 100 % / 88 %
- agents_b10: pooled vs faulty -71.3 %; recovered 100 % / 100 % / 81 %
- mean_imputation: pooled vs faulty -54.2 %; recovered 56 % / 76 % / 54 %
- es_seed0: pooled vs faulty -69.8 %; recovered 14 % / 99 % / 19 %
- es_seed1: pooled vs faulty -70.1 %; recovered 25 % / 99 % / 30 %
- finetune_300k: pooled vs faulty -71.0 %; recovered 85 % / 100 % / 56 %
- finetune_1M: pooled vs faulty -71.3 %; recovered 101 % / 100 % / 74 %
- agents b0 vs mean_imputation: pooled -37.0 % [-78.3, +4.3]; per target -3.0 % [-4.3, -1.6], -63.5 % [-135.7, +8.6], -0.2 % [-1.6, +1.2]
- agents b0 vs es_seed0: pooled -4.5 % [-6.3, -2.7]; per target -5.9 % [-8.8, -3.0], -5.2 % [-7.5, -2.9], -2.4 % [-3.9, -0.9]
- agents b0 vs finetune_1M: pooled +0.5 % [-0.1, +1.1]; per target +0.4 % [-0.6, +1.4], +0.1 % [-0.9, +1.0], +1.1 % [-0.2, +2.4]
- agents b5 vs mean_imputation: pooled -37.6 % [-78.9, +3.7]; per target -3.2 % [-4.3, -2.2], -63.6 % [-135.8, +8.5], -2.1 % [-3.6, -0.7]
- agents b5 vs es_seed0: pooled -5.3 % [-7.2, -3.5]; per target -6.2 % [-8.7, -3.6], -5.5 % [-7.5, -3.5], -4.3 % [-5.9, -2.7]
- agents b5 vs finetune_1M: pooled -0.4 % [-0.9, +0.2]; per target +0.1 % [-0.7, +0.9], -0.3 % [-1.1, +0.5], -0.9 % [-1.9, +0.1]
- agents b10 vs mean_imputation: pooled -37.4 % [-78.7, +3.8]; per target -3.3 % [-4.3, -2.2], -63.6 % [-135.8, +8.5], -1.7 % [-2.9, -0.4]
- agents b10 vs es_seed0: pooled -5.2 % [-7.0, -3.3]; per target -6.2 % [-8.9, -3.5], -5.4 % [-7.7, -3.2], -3.9 % [-5.4, -2.4]
- agents b10 vs finetune_1M: pooled -0.2 % [-0.6, +0.3]; per target +0.1 % [-0.5, +0.7], -0.2 % [-0.8, +0.5], -0.4 % [-1.4, +0.5]

## Failed detector: three sessions per network (first run + 2 replicates)

- b0 net0: recovered 96 % / 101 % / 82 % (mean 93 %); session-mean vs faulty -6.6 %
- b0 net1: recovered 100 % / 100 % / 100 % (mean 100 %); session-mean vs faulty -88.0 %
- b0 net2: recovered 57 % / 67 % / 80 % (mean 68 %); session-mean vs faulty -4.2 %
- b0 session-mean vs mean_imputation: pooled -37.1 % [-78.3, +4.2]; per target -2.8*, -63.4, -0.9
- b0 session-mean vs es_seed0: pooled -4.6 % [-6.4, -2.8]; per target -5.7*, -5.0*, -3.1*
- b0 session-mean vs es_seed1: pooled -3.6 % [-4.9, -2.2]; per target -4.9*, -3.3*, -2.4*
- b0 session-mean vs finetune_1M: pooled +0.4 % [-0.2, +1.1]; per target +0.6, +0.3, +0.4
- b10 net0: recovered 100 % / -1006 % / 107 % (mean -266 %); session-mean vs faulty +19.0 %
- b10 net1: recovered 100 % / 100 % / 100 % (mean 100 %); session-mean vs faulty -88.0 %
- b10 net2: recovered 81 % / 74 % / 82 % (mean 79 %); session-mean vs faulty -4.9 %
- b10 session-mean vs mean_imputation: pooled -31.5 % [-75.3, +12.3]; per target +24.0, -63.6, -1.6*
- b10 session-mean vs es_seed0: pooled +3.8 % [-14.6, +22.2]; per target +20.2, -5.3*, -3.8*
- b10 session-mean vs es_seed1: pooled +4.9 % [-13.5, +23.4]; per target +21.2, -3.7*, -3.1*
- b10 session-mean vs finetune_1M: pooled +9.3 % [-10.1, +28.7]; per target +28.3, -0.1, -0.3

## Explain (fidelity on held-out honoured states; closed-loop TTS vs the network)

- agent net0 small: 8 comparisons, 80 constants, fidelity 91.1 %, TTS +1.5 %
- agent net0 full: 1 comparisons, 132 constants, fidelity 91.1 %, TTS +0.1 %
- agent net1 small: 3 comparisons, 43 constants, fidelity 91.2 %, TTS +1.2 %
- agent net1 full: 3 comparisons, 53 constants, fidelity 92.8 %, TTS +0.4 %
- agent net2 small: 6 comparisons, 65 constants, fidelity 67.4 %, TTS +0.5 %
- agent net2 full: 6 comparisons, 69 constants, fidelity 68.1 %, TTS +0.5 %
- cart_net0_small: 7 comparisons, fidelity 71.3 %, TTS +23.1 %, unserved 142
- cart_net0_mid: 31 comparisons, fidelity 56.7 %, TTS +14.3 %, unserved 107
- cart_net0_full: 135 comparisons, fidelity 64.2 %, TTS +5.3 %, unserved 23
- dagger_net0_small: 7 comparisons, fidelity 71.3 %, TTS +23.1 %, unserved 142
- dagger_net0_mid: 31 comparisons, fidelity 56.2 %, TTS +6.8 %, unserved 31
- dagger_net0_full: 153 comparisons, fidelity 84.5 %, TTS +3.4 %, unserved 26
- cart_net1_small: 7 comparisons, fidelity 40.0 %, TTS +22.5 %, unserved 107
- cart_net1_mid: 31 comparisons, fidelity 62.2 %, TTS +9.9 %, unserved 43
- cart_net1_full: 159 comparisons, fidelity 50.7 %, TTS +6.4 %, unserved 34
- dagger_net1_small: 7 comparisons, fidelity 40.1 %, TTS +17.8 %, unserved 111
- dagger_net1_mid: 31 comparisons, fidelity 45.7 %, TTS +8.3 %, unserved 30
- dagger_net1_full: 208 comparisons, fidelity 84.5 %, TTS +3.4 %, unserved 6
- cart_net2_small: 7 comparisons, fidelity 45.1 %, TTS +697.7 %, unserved 6990
- cart_net2_mid: 31 comparisons, fidelity 58.2 %, TTS +10.6 %, unserved 82
- cart_net2_full: 148 comparisons, fidelity 55.3 %, TTS +5.6 %, unserved 41
- dagger_net2_small: 7 comparisons, fidelity 44.3 %, TTS +15.8 %, unserved 76
- dagger_net2_mid: 31 comparisons, fidelity 51.2 %, TTS +5.9 %, unserved 13
- dagger_net2_full: 192 comparisons, fidelity 64.3 %, TTS +3.4 %, unserved 12

## Constrain (cost of the rules under the naive shield: +5.80 %)

- supervisors, budget 0: vs naive -0.10 % [-0.43, +0.23]
- supervisors, budget 10: vs naive -0.31 % [-0.63, +0.02]
- masked: +0.00 % [-0.00, +0.00]
- finetune_300k: -1.14 % [-1.65, -0.63]
- finetune_1M: -0.53 % [-1.05, -0.00]
- scratch_under_rules_vs_naive: -1.03 % [-1.61, -0.44]
