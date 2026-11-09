#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Konstantinos Katzilieris, Emmanouil Kampitakis, Eleni I. Vlahogianni (NTUA)
# Score finished agent packs and screen their notes for outside-context references (project notes, memory).
cd "$(dirname "$0")/.."
# run inside the environment of environment.yml
export OMP_NUM_THREADS=1
for p in "$@"; do
  python -m wsp.agents_eval "$p" 2>&1 | grep -v "^Loading\|^Simulation\|done (\|^ \|^Reason\|^Performance\|^Vehicles\|^Statistics\|^Dijkstra"
  grep -n -i "session memory\|my memory\|memory index\|project note\|earlier project\|previous session" "$p"/work/*.md 2>/dev/null || echo "  (no outside-context references)"
done
python -m wsp.brief "$@"
