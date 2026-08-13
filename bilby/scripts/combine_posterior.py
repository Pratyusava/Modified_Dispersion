#!/usr/bin/env python
"""Combine the A (dispersion amplitude) posteriors from a set of injection runs.

Reads <run_dir>/<n>/*_result.json for every numbered run directory, builds a
boundary-reflected Gaussian KDE of the A samples for each run, and multiplies
them on a common grid (sum of log-KDEs). By default (--prior uniform) each
event is first reweighted by the Jacobian |A| to undo the SymmetricLogUniform
sampling prior, so the product is the joint likelihood -- the valid
multi-event combination. Runs with a collapsed-sampler signature are skipped
automatically.

Example:
    python combine_posterior.py \
        ../run_directory/CE_1p0MW_AplusCoat_flow10/A0/results \
        --exclude 6 63 75 76 80 --outdir ../run_directory/CE_1p0MW_AplusCoat_flow10/A0

Outputs (in --outdir, default: run_dir):
  combined_A_posterior.dat  -- two columns: A, normalized combined pdf
  stacked_A_posteriors.png  -- individual KDEs + combined posterior
"""

import argparse
import glob
import json
import os

import numpy as np
from scipy.stats import gaussian_kde
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

trapz = getattr(np, "trapezoid", np.trapz)  # np.trapz removed in NumPy 2.x


def load_A_samples(run_dir, exclude):
    samples = {}
    prior_maxima = set()
    for name in sorted(os.listdir(run_dir)):
        if not name.isdigit() or int(name) in exclude:
            continue
        matches = sorted(glob.glob(os.path.join(run_dir, name, "*_result.json")))
        if not matches:
            print(f"run {name}: no result file, skipping")
            continue
        if len(matches) > 1:
            print(f"run {name}: {len(matches)} result files, using "
                  f"{os.path.basename(matches[0])}")
        path = matches[0]
        with open(path) as f:
            result = json.load(f)
        A = np.asarray(result["posterior"]["content"]["A"])
        # collapsed-sampler signature (cf. runs 47/54/68/81): huge evidence
        # error and/or near-zero posterior spread; healthy runs have
        # logzerr ~ 0.3 and A_sd ~ 2.7e-3
        logzerr = result.get("log_evidence_err") or 0.0
        if logzerr > 1.0 or A.std() < 3e-4:
            print(f"run {name}: PATHOLOGICAL posterior "
                  f"(logzerr={logzerr:.3g}, A_sd={A.std():.3g}) -- auto-skipped, "
                  f"inspect this run")
            continue
        try:
            prior_maxima.add(result["priors"]["A"]["kwargs"]["maximum"])
        except (KeyError, TypeError):
            pass
        samples[int(name)] = A
    if len(prior_maxima) > 1:
        raise SystemExit(f"runs have inconsistent A priors: maxima {sorted(prior_maxima)}")
    return samples, (prior_maxima.pop() if prior_maxima else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir",
                        help="results directory containing the numbered run folders")
    parser.add_argument("--exclude", type=int, nargs="*", default=[],
                        metavar="RUN", help="run numbers to leave out")
    parser.add_argument("--outdir", default=None,
                        help="where to write outputs (default: run_dir)")
    parser.add_argument("--prior-max", type=float, default=None,
                        help="half-width of the symmetric A prior; default: "
                             "read from the result files' stored priors")
    parser.add_argument("--prior", choices=["log-uniform", "uniform"],
                        default="uniform",
                        help="target prior on A. The runs sampled with a "
                             "SymmetricLogUniform prior (density ~ 1/|A|); "
                             "'uniform' (default) reweights each event's "
                             "samples by the Jacobian |A|, giving the valid "
                             "joint-likelihood combination. 'log-uniform' "
                             "multiplies raw posteriors, which stacks the "
                             "1/|A| prior N times -- diagnostic use only")
    parser.add_argument("--ngrid", type=int, default=2001,
                        help="number of grid points (default 2001)")
    parser.add_argument("--bw", type=float, default=1.0,
                        help="multiplier on the Scott's-rule KDE bandwidth "
                             "(default 1.0). Rerun with 0.5 and 2 to check "
                             "the result is bandwidth-stable")
    args = parser.parse_args()

    outdir = args.outdir or args.run_dir
    os.makedirs(outdir, exist_ok=True)

    if args.prior == "log-uniform":
        print("WARNING: --prior log-uniform multiplies raw posteriors, so the "
              "combined curve carries the 1/|A| prior to the Nth power. Its "
              "width is set by the prior and KDE bandwidth, NOT the data. "
              "Do not quote it as a measurement; use --prior uniform.")

    samples, file_prior_max = load_A_samples(args.run_dir, set(args.exclude))
    if not samples:
        raise SystemExit(f"no result files found in {args.run_dir}")
    prior_max = args.prior_max or file_prior_max
    if prior_max is None:
        raise SystemExit("could not read the A prior from the result files; "
                         "pass --prior-max explicitly")
    print(f"A prior half-width: {prior_max:g}"
          + (" (from result files)" if args.prior_max is None else " (override)"))
    print(f"loaded {len(samples)} runs "
          f"({min(len(s) for s in samples.values())}-"
          f"{max(len(s) for s in samples.values())} samples each), "
          f"excluded {sorted(set(args.exclude))}")

    if args.prior == "uniform":
        ess = np.array([np.abs(A).sum() ** 2 / (np.abs(A) ** 2).sum()
                        for A in samples.values()])
        print(f"|A|-reweighting effective sample size per run: "
              f"median {np.median(ess):.0f}, min {ess.min():.0f}")
        if ess.min() < 300:
            print("WARNING: some runs have ESS < 300; their KDEs are noisy "
                  "and can distort the combined product")

    grid = np.linspace(-prior_max, prior_max, args.ngrid)
    log_combined = np.zeros_like(grid)
    kdes = {}
    for run, A in sorted(samples.items()):
        # reweight by the Jacobian |A| to undo the 1/|A| sampling prior
        w = np.abs(A) if args.prior == "uniform" else None
        # reflect samples about the prior edges to correct KDE boundary bias;
        # keep the bandwidth chosen from the unreflected samples
        base = gaussian_kde(A, weights=w)
        aug = np.concatenate([A, -2 * prior_max - A, 2 * prior_max - A])
        waug = None if w is None else np.tile(w, 3)
        kde = gaussian_kde(aug, weights=waug, bw_method=args.bw * base.factor)
        kdes[run] = 3 * kde.evaluate(grid)
        lp = np.log(3) + kde.logpdf(grid)
        nzero = np.isneginf(lp).sum()
        if nzero:
            print(f"run {run}: KDE support is zero on {nzero}/{len(grid)} grid "
                  f"points; those regions are zeroed in the combined posterior")
        log_combined += lp

    combined = np.exp(log_combined - log_combined.max())
    combined /= trapz(combined, grid)

    # 90% credible interval (equal-tailed) of the combined posterior
    cdf = np.concatenate([[0.0], np.cumsum(
        0.5 * (combined[1:] + combined[:-1]) * np.diff(grid))])
    lo, hi = np.interp([0.05, 0.95], cdf, grid)
    print(f"combined 90% CI: [{lo:.3e}, {hi:.3e}]")

    # 90% upper limit on |A| (the natural statement for a flat-in-A prior)
    absgrid = np.linspace(0, prior_max, (args.ngrid + 1) // 2)
    abspdf = np.interp(absgrid, grid, combined) + np.interp(-absgrid, grid, combined)
    abscdf = np.concatenate([[0.0], np.cumsum(
        0.5 * (abspdf[1:] + abspdf[:-1]) * np.diff(absgrid))])
    ul90 = np.interp(0.9 * abscdf[-1], abscdf, absgrid)
    print(f"90% upper limit on |A|: {ul90:.3e}")

    tag = "" if args.prior == "log-uniform" else "_uniform_prior"
    header = (f"combined A posterior from {len(samples)} runs in {args.run_dir}, "
              f"prior={args.prior}, excluded {sorted(set(args.exclude))}; "
              f"90% CI [{lo:.3e}, {hi:.3e}]; 90% UL on |A| {ul90:.3e}")
    np.savetxt(os.path.join(outdir, f"combined_A_posterior{tag}.dat"),
               np.column_stack([grid, combined]), header=header)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for run, pdf in kdes.items():
        ax.plot(grid, pdf, color="0.6", lw=0.8, alpha=0.4,
                label="individual events" if run == min(kdes) else None)
    ax.plot(grid, combined, color="#3161cd", lw=2.0, label="combined")
    ax.axvline(0.0, color="0.25", lw=1.0, ls="--", label="injected $A=0$")

    ax.set_xlabel(r"$\mathbb{A}_0$")
    ax.set_ylabel("probability density")
    ax.set_xlim(-1.05 * prior_max, 1.05 * prior_max)
    ax.set_title(f"{args.prior} prior on $A$", fontsize=11)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = os.path.join(outdir, f"stacked_A_posteriors{tag}.png")
    fig.savefig(out, dpi=200)
    print(f"wrote {out} and combined_A_posterior{tag}.dat")


if __name__ == "__main__":
    main()
