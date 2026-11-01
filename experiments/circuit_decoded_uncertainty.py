"""Decoded uncertainty predicts recall error (van Bergen et al. 2015).

Van Bergen et al. (Nat Neurosci 2015) decoded a full probability distribution
over the remembered feature from human visual cortex on each trial, and showed
that the *width* of that decoded distribution (the decoded uncertainty) predicts
the magnitude of the behavioral error on the same trial: more uncertain trials
are less accurate.

The Mexican-hat circuit that implements MemGP maintenance makes this exact
prediction: on each trial the posterior over the colour slice at the probed
location has a width; the report is a single sample of that posterior. So the
width of the decoded distribution should predict the single-trial recall error.

This script runs the *explicit* network (fixed W; Adam on m and S), not the
full plastic GP, so the panel matches the circuit claim of Figure 5.

Width here is the circular SD of the recalled colour over independent samples
of the same posterior, drawn separately from the sample that produced the
report. The shift Fisher information 1/sqrt(J) is kept for comparison.

Outputs (results/circuit/):
    decoded_uncertainty.npz            Fig. 5e
    decoded_uncertainty_preview.png    diagnostic plot

Run:  python -m experiments.circuit_decoded_uncertainty --trials 100 --set-size 4
"""
import os
import argparse
import copy
import numpy as np
import torch

from memgp import generator
from memgp.generator import circular_error
from memgp.paths import results_dir
from memgp.simulation import (
    setup_trial, encoding_phase, load_config, device, normalized_reports,
)
from memgp.circuit import (
    seed_all, clone_state, explicit_maintenance,
)

N_COLOR = 360
COLOR_AXIS = np.linspace(-180.0, 180.0, N_COLOR, endpoint=False)

NPZ = str(results_dir("circuit") / "decoded_uncertainty.npz")
PREVIEW = str(results_dir("circuit") / "decoded_uncertainty_preview.png")


def circular_sd_deg(errs):
    e = np.deg2rad(np.asarray(errs, float))
    R = np.abs(np.mean(np.exp(1j * e)))
    R = min(max(R, 1e-12), 1.0 - 1e-12)
    return float(np.rad2deg(np.sqrt(-2.0 * np.log(R))))


def item_readout(model, item_locs, j, true_color, norm, n_mc=200, jitter=1e-6):
    """Return (signed report error, posterior width, Fisher uncertainty) for item j.

    The report and the width use the normalized readout: the report is one
    draw, and the width is the circular SD of the reported color over
    ``n_mc`` further draws of the same posterior.
    """
    model.eval()
    reports = normalized_reports(model, item_locs, norm, n_draws=n_mc + 1)[:, j]
    signed = circular_error(float(reports[0]), true_color)
    width = circular_sd_deg(reports[1:])

    colors = torch.tensor(COLOR_AXIS, dtype=torch.float32, device=device)
    grid = torch.stack([torch.full_like(colors, float(item_locs[j])), colors], dim=-1)
    with torch.no_grad():
        pred = model(grid)
        mu = pred.mean.cpu().numpy().astype(np.float64)
        Sigma = pred.covariance_matrix.cpu().numpy().astype(np.float64)

    dc = 360.0 / N_COLOR
    mup = (np.roll(mu, -1) - np.roll(mu, 1)) / (2.0 * dc)
    S = Sigma + jitter * np.eye(N_COLOR)
    J = float(mup @ np.linalg.solve(S, mup))
    sd_ideal = 1.0 / np.sqrt(J) if J > 0 else np.nan
    return signed, width, sd_ideal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--set-size", type=int, default=4)
    ap.add_argument("--config", default="config_set_size.yaml")
    ap.add_argument("--n-mc", type=int, default=200,
                    help="posterior draws used for the width of each item")
    args = ap.parse_args()

    config = load_config(filename=args.config)
    base = int(config["experiment"]["random_seed"])

    errs, width, fisher = [], [], []
    for t in range(args.trials):
        seed = base + t * 7919 + args.set_size * 101
        seed_all(seed)
        items = generator.generate_items(args.set_size, seed=seed)
        samples, weights, model, lik, opt, mll = setup_trial(items, config)
        encoding_phase(model, lik, opt, mll, samples, weights,
                       config["training"]["encoding_epochs"])
        enc_m, enc_l = clone_state(model), clone_state(lik)
        enc_o = copy.deepcopy(opt.state_dict())
        rnn_model, _, _ = explicit_maintenance(
            enc_m, enc_l, items, config, cue_idx=None, enc_opt_state=enc_o)

        locs = [float(l) for l, _ in items]
        for j, (l, c) in enumerate(items):
            e, w, f = item_readout(rnn_model, locs, j, c, config["normalization"],
                                   n_mc=args.n_mc)
            if np.isfinite(w):
                errs.append(e)
                width.append(w)
                fisher.append(f)
        if (t + 1) % 20 == 0:
            print(f"  trial {t+1}/{args.trials} done", flush=True)

    errs = np.asarray(errs, float)
    width = np.asarray(width, float)
    fisher = np.asarray(fisher, float)

    from scipy.stats import spearmanr
    rho, p = spearmanr(width, np.abs(errs))
    ok = np.isfinite(fisher)
    rho_f, p_f = spearmanr(fisher[ok], np.abs(errs[ok]))

    os.makedirs(os.path.dirname(NPZ), exist_ok=True)
    np.savez(NPZ, errs=errs, uncert=width, uncert_fisher=fisher,
             rho=rho, p=p, rho_fisher=rho_f, p_fisher=p_f,
             n_mc=args.n_mc, set_size=args.set_size, n_trials=args.trials,
             backend="explicit_rnn")
    print(f"\nsaved -> {NPZ}")
    print(f"Spearman rho(posterior width, |error|) = {rho:.3f}  (p={p:.2e}); "
          f"n={len(errs)} items  [explicit network]")
    print(f"  for comparison, Fisher 1/sqrt(J): rho = {rho_f:.3f}  (p={p_f:.2e})")

    _preview(errs, width, fisher, rho, rho_f, args)


def _preview(errs, width, fisher, rho, rho_f, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.6))

    ax[0].scatter(width, np.abs(errs), s=10, color="#B86B1E",
                  alpha=0.45, edgecolors="none")
    ax[0].set_xlabel("posterior width (deg)")
    ax[0].set_ylabel("|recall error| (deg)")
    ax[0].set_title(f"Posterior width (Spearman \u03c1={rho:.2f})")

    ax[1].scatter(fisher, np.abs(errs), s=10, color="#7F7F7F",
                  alpha=0.45, edgecolors="none")
    ax[1].set_xlabel("Fisher uncertainty  1/\u221aJ  (deg)")
    ax[1].set_ylabel("|recall error| (deg)")
    ax[1].set_title(f"Fisher, for comparison (\u03c1={rho_f:.2f})")

    fig.suptitle(
        "Explicit-network decoded uncertainty predicts recall error "
        f"(van Bergen et al. 2015)   N={args.set_size}, {args.trials} trials",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(PREVIEW, dpi=150)
    plt.close(fig)
    print(f"saved preview -> {PREVIEW}")


if __name__ == "__main__":
    main()
