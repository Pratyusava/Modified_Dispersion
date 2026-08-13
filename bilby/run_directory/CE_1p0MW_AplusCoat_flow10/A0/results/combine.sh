#!/bin/bash
# Combine the per-event A0 posteriors for this campaign (CE40 1.0 MW + A+, f_low = 10 Hz).
# Collapsed-sampler runs (run 47) are auto-skipped by objective criteria in
# combine_posterior.py; no hand-picked exclusions.
cd "$(dirname "$0")"
python ../../../../scripts/combine_posterior.py .
