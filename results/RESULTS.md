# v3 results (30 held-out test episodes; % = change in mean TTS, 95 % CI)

**L2**: fixed 48.38, actuated 46.38, PPO 44.84 (sd over seeds 0.18); PPO vs actuated -3.3 % [-5.2, -1.5], vs fixed -7.3 %

**L2W**: fixed 62.76, actuated 64.02, PPO 58.44 (sd over seeds 0.23); PPO vs actuated -8.7 % [-12.0, -5.5], vs fixed -6.9 %

**L2_rules**: fixed 49.85, actuated 48.74, PPO 46.97 (sd over seeds 0.07); PPO vs actuated -3.6 % [-4.8, -2.5], vs fixed -5.8 %

Runs failing the convergence criterion: ['L2W/ppo_s203']; L2W/ppo_s203 is superseded by L2W/ppo_s203_r1 (converged: True, 4.8M steps), which the analysis uses


## Initialise

- zero-shot 1: TTS 45.80; vs fixed -5.3 %, vs actuated -1.3 %
- zero-shot 2: TTS 47.68; vs fixed -1.4 %, vs actuated +2.8 %
- zero-shot 3: TTS 46.22; vs fixed -4.5 %, vs actuated -0.4 %
- warm vs scratch, first50_mean_delay: 20.4 vs 49.4 (-58.8 % [-85.5, -32.0])
- warm vs scratch, first1M_total_delay: 26314.2 vs 28093.8 (-6.3 % [-10.4, -2.3])
- warm vs scratch, final_tts: 45.1 vs 44.8 (+0.5 % [-0.2, +1.1])

## Transfer

- zero-shot 0: TTS 58.55; vs fixed -6.7 %, vs actuated -8.6 %, vs padding -97.4 %
- zero-shot 1: TTS 59.50; vs fixed -5.2 %, vs actuated -7.1 %, vs padding -95.7 %
- zero-shot 2: TTS 59.88; vs fixed -4.6 %, vs actuated -6.5 %, vs padding -93.0 %
- warm vs scratch, first50_mean_delay: 34.3 vs 99.8 (-65.6 % [-100.7, -30.6])
- warm vs scratch, first1M_total_delay: 39848.6 vs 44514.2 (-10.5 % [-15.6, -5.3])
- warm vs scratch, final_tts: 58.1 vs 58.4 (-0.5 % [-1.2, +0.1])

## Patch (vs each target; pooled over 3 targets)

- budget 0: +0.02 % [-0.28, +0.32], wins 17/30
- budget 5: +0.09 % [-0.38, +0.56], wins 18/30
- budget 10: -0.35 % [-0.74, +0.05], wins 19/30
- es_seed0: +0.26 % [-0.37, +0.88]
- es_seed1: -0.08 % [-0.36, +0.20]
- wrapper: +0.02 % [-0.02, +0.05]
- finetune_300k: +0.08 % [-0.44, +0.60]
- finetune_1M: -0.13 % [-0.55, +0.29]
- writer sonnet: +0.28 % [-0.17, +0.72]
- writer haiku: +1.50 % [+0.66, +2.34]
- writer fable: -0.01 % [-0.48, +0.46]

## Patch under a failed detector (W2; recovered = share of the fault's cost)

- net0: healthy 45.0, faulty 48.3
- net1: healthy 44.8, faulty 242.2
- net2: healthy 44.6, faulty 47.3
- agents_b0: pooled vs faulty -59.8 %; recovered 97 % / 100 % / 62 %
- agents_b5: pooled vs faulty -60.0 %; recovered 94 % / 100 % / 86 %
- agents_b10: pooled vs faulty -59.9 %; recovered 94 % / 100 % / 78 %
- mean_imputation: pooled vs faulty -45.7 %; recovered 59 % / 76 % / 59 %
- es_seed0: pooled vs faulty -58.0 %; recovered 14 % / 99 % / 22 %
- es_seed1: pooled vs faulty -58.5 %; recovered 27 % / 99 % / 35 %
- finetune_300k: pooled vs faulty -59.7 %; recovered 87 % / 100 % / 66 %
- finetune_1M: pooled vs faulty -59.9 %; recovered 98 % / 100 % / 78 %
- agents b0 vs mean_imputation: pooled -25.8 % [-54.3, +2.6]; per target -2.6 % [-3.7, -1.6], -50.5 % [-107.5, +6.5], -0.2 % [-1.0, +0.6]
- agents b0 vs es_seed0: pooled -4.2 % [-5.9, -2.5]; per target -5.6 % [-8.1, -3.2], -4.6 % [-6.6, -2.6], -2.3 % [-3.5, -1.1]
- agents b0 vs finetune_1M: pooled +0.5 % [+0.1, +0.9]; per target +0.1 % [-0.5, +0.7], +0.4 % [-0.4, +1.1], +1.0 % [+0.1, +1.8]
- agents b5 vs mean_imputation: pooled -26.2 % [-54.6, +2.2]; per target -2.4 % [-3.3, -1.5], -50.7 % [-107.6, +6.3], -1.6 % [-2.6, -0.6]
- agents b5 vs es_seed0: pooled -4.7 % [-6.3, -3.1]; per target -5.4 % [-7.6, -3.2], -5.0 % [-6.7, -3.2], -3.7 % [-5.1, -2.2]
- agents b5 vs finetune_1M: pooled -0.1 % [-0.5, +0.4]; per target +0.3 % [-0.4, +1.1], -0.0 % [-0.7, +0.6], -0.5 % [-1.2, +0.3]
- agents b10 vs mean_imputation: pooled -26.1 % [-54.5, +2.3]; per target -2.4 % [-3.3, -1.6], -50.6 % [-107.6, +6.4], -1.1 % [-1.9, -0.4]
- agents b10 vs es_seed0: pooled -4.5 % [-6.2, -2.9]; per target -5.4 % [-7.8, -3.1], -4.8 % [-6.8, -2.9], -3.2 % [-4.5, -2.0]
- agents b10 vs finetune_1M: pooled +0.1 % [-0.2, +0.5]; per target +0.3 % [-0.3, +0.8], +0.1 % [-0.4, +0.6], -0.0 % [-0.7, +0.7]

## Failed detector: three sessions per network (first run + 2 replicates)

- b0 net0: recovered 97 % / 98 % / 89 % (mean 94 %); session-mean vs faulty -6.4 %
- b0 net1: recovered 100 % / 100 % / 100 % (mean 100 %); session-mean vs faulty -81.3 %
- b0 net2: recovered 62 % / 71 % / 78 % (mean 70 %); session-mean vs faulty -4.0 %
- b0 session-mean vs mean_imputation: pooled -25.9 % [-54.2, +2.5]; per target -2.5*, -50.4, -0.7
- b0 session-mean vs es_seed0: pooled -4.2 % [-5.9, -2.6]; per target -5.5*, -4.4*, -2.8*
- b0 session-mean vs es_seed1: pooled -3.2 % [-4.4, -2.0]; per target -4.7*, -2.8*, -2.1*
- b0 session-mean vs finetune_1M: pooled +0.4 % [-0.1, +0.9]; per target +0.2, +0.5, +0.5
- b10 net0: recovered 94 % / -579 % / 101 % (mean -128 %); session-mean vs faulty +8.7 %
- b10 net1: recovered 100 % / 100 % / 100 % (mean 100 %); session-mean vs faulty -81.4 %
- b10 net2: recovered 78 % / 80 % / 80 % (mean 79 %); session-mean vs faulty -4.5 %
- b10 session-mean vs mean_imputation: pooled -22.1 % [-52.2, +8.0]; per target +13.2, -50.5, -1.2*
- b10 session-mean vs es_seed0: pooled +0.6 % [-10.0, +11.2]; per target +9.7, -4.7*, -3.3*
- b10 session-mean vs es_seed1: pooled +1.7 % [-8.9, +12.3]; per target +10.7, -3.1*, -2.6*
- b10 session-mean vs finetune_1M: pooled +5.5 % [-5.7, +16.7]; per target +16.4, +0.2, -0.1

## Explain (fidelity on held-out honoured states; closed-loop TTS vs the network)

- agent net0 small: 8 comparisons, 80 constants, fidelity 91.1 %, TTS +1.1 %
- agent net0 full: 1 comparisons, 132 constants, fidelity 91.1 %, TTS +0.2 %
- agent net1 small: 3 comparisons, 43 constants, fidelity 91.2 %, TTS +0.6 %
- agent net1 full: 3 comparisons, 53 constants, fidelity 92.8 %, TTS +0.1 %
- agent net2 small: 6 comparisons, 65 constants, fidelity 67.4 %, TTS +0.5 %
- agent net2 full: 6 comparisons, 69 constants, fidelity 68.1 %, TTS +0.5 %
- cart_net0_small: 7 comparisons, fidelity 71.3 %, TTS +18.4 %, unserved 142
- cart_net0_mid: 31 comparisons, fidelity 56.7 %, TTS +10.9 %, unserved 107
- cart_net0_full: 135 comparisons, fidelity 64.2 %, TTS +4.9 %, unserved 23
- dagger_net0_small: 7 comparisons, fidelity 71.3 %, TTS +18.4 %, unserved 142
- dagger_net0_mid: 31 comparisons, fidelity 56.2 %, TTS +6.1 %, unserved 31
- dagger_net0_full: 153 comparisons, fidelity 84.5 %, TTS +2.9 %, unserved 26
- cart_net1_small: 7 comparisons, fidelity 40.0 %, TTS +18.5 %, unserved 107
- cart_net1_mid: 31 comparisons, fidelity 62.2 %, TTS +8.3 %, unserved 43
- cart_net1_full: 159 comparisons, fidelity 50.7 %, TTS +5.1 %, unserved 34
- dagger_net1_small: 7 comparisons, fidelity 40.1 %, TTS +13.6 %, unserved 111
- dagger_net1_mid: 31 comparisons, fidelity 45.7 %, TTS +7.2 %, unserved 30
- dagger_net1_full: 208 comparisons, fidelity 84.5 %, TTS +3.2 %, unserved 6
- cart_net2_small: 7 comparisons, fidelity 45.1 %, TTS +436.6 %, unserved 6990
- cart_net2_mid: 31 comparisons, fidelity 58.2 %, TTS +7.6 %, unserved 82
- cart_net2_full: 148 comparisons, fidelity 55.3 %, TTS +4.2 %, unserved 41
- dagger_net2_small: 7 comparisons, fidelity 44.3 %, TTS +13.0 %, unserved 76
- dagger_net2_mid: 31 comparisons, fidelity 51.2 %, TTS +5.4 %, unserved 13
- dagger_net2_full: 192 comparisons, fidelity 64.3 %, TTS +3.0 %, unserved 12

## Constrain (cost of the rules under the naive shield: +5.97 %)

- supervisors, budget 0: vs naive -0.10 % [-0.43, +0.23]
- supervisors, budget 10: vs naive -0.31 % [-0.63, +0.02]
- masked: +0.00 % [-0.00, +0.00]
- finetune_300k: -1.14 % [-1.65, -0.63]
- finetune_1M: -0.53 % [-1.05, -0.00]
- scratch_under_rules_vs_naive: -1.03 % [-1.61, -0.44]
