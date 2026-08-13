#!/bin/bash
# Combine the per-event A0 posteriors for this campaign (CE40 1.5 MW + CE20 1.5 MW + A+, f_low = 5 Hz).
# Collapsed-sampler runs are auto-skipped by combine_posterior.py.
cd "$(dirname "$0")"
python ../../../../scripts/combine_posterior.py .
