#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Konstantinos Katzilieris, Emmanouil Kampitakis, Eleni I. Vlahogianni (NTUA)
# Start the queue runner detached (survives the terminal). Log: queue/runner.log.
# Run it inside the environment of environment.yml (conda activate wsp-traffic).
cd "$(dirname "$0")/.."
mkdir -p queue
if [ -f queue/runner.pid ] && kill -0 "$(cat queue/runner.pid)" 2>/dev/null; then echo "runner already running"; exit 0; fi
rm -f queue/stop
nohup setsid python -m wsp.queue run >> queue/runner.log 2>&1 < /dev/null &
echo "runner started"
