"""Bump-diffusion signatures of the explicit MemGP circuit (Wimmer et al. 2014).

Wimmer, Nykamp, Constantinidis & Compte (Nat Neurosci 2014) showed, in monkey
dlPFC during spatial working memory, that the mnemonic activity bump diffuses
over the delay: its position variance grows with time.

The Mexican-hat circuit that implements MemGP maintenance makes the same
claim. KL pressure relaxes the posterior toward the prior during the delay, so
the peak of the posterior mean drifts further the longer the delay runs. The
report is one sample from that posterior. Its circular SD is the behavioral
scale; the peak's displacement over the delay is about a degree.

This script runs the *explicit* network (fixed W; Adam on m and S), not the
full plastic GP, so the panel matches the circuit claim of Figure 5.

Faithful to the single-item spatial task, we use N = 1 (one clean bump) with
neutral maintenance.

Outputs (results/circuit/):
    bump_diffusion.npz            Fig. 5d
    bump_diffusion_preview.png    diagnostic plot

Run:  python -m experiments.circuit_bump_drift --trials 100
"""
import os
import argparse
import copy
import numpy as np
import torch

from memgp import generator
from memgp.paths import results_dir
from memgp.simulation import (
    setup_trial, encoding_phase, load_config, device, normalized_reports,
)
from memgp.circuit import (
    seed_all, clone_state, explicit_maintenance, set_mean, set_covar,
)

N_COLOR = 360
COLOR_AXIS = np.linspace(-180.0, 180.0, N_COLOR, endpoint=False)

# The report keeps the paper's 360-point readout grid, but decoding the bump
# centre on that grid quantises drift at 1 deg, which is the same order as the
# drift itself.  The centre is therefore decoded on a finer axis, and by a
# population vector as well as by the tracked peak (``track_peak``).
N_DECODE = 3600
DECODE_AXIS = np.linspace(-180.0, 180.0, N_DECODE, endpoint=False)

NPZ = str(results_dir("circuit") / "bump_diffusion.npz")
PREVIEW = str(results_dir("circuit") / "bump_diffusion_preview.png")


def _circ_disp(a, b):
    """Signed circular displacement a - b wrapped to [-180, 180)."""
    return (a - b + 180.0) % 360.0 - 180.0


def decode_slice(model, likelihood, query_loc, axis=DECODE_AXIS):
    """Posterior color slice at ``query_loc``: return (mean, var) over ``axis``."""
    pts = torch.stack(
        [torch.full((len(axis),), float(query_loc)),
         torch.tensor(axis, dtype=torch.float32)],
        dim=-1,
    ).to(device)
    model.eval()
    with torch.no_grad():
        pred = likelihood(model(pts))
        mean = pred.mean.cpu().numpy()
        var = pred.variance.cpu().numpy()
    return mean, var


TRACK_WINDOW = 45.0


def track_peak(axis_deg, profile, prev_deg):
    """Argmax of ``profile`` within ``TRACK_WINDOW`` deg of ``prev_deg``.

    The color profile can hold a second peak of nearly equal height far from
    the item; the global argmax then hops between the two. Restricting the
    argmax to a window around the previous position keeps the readout on the
    item's own bump. Nearest-local-maximum tracking would instead lock onto
    numerical ripples on the flat top of the bump.
    """
    near = np.abs(_circ_disp(axis_deg, prev_deg)) <= TRACK_WINDOW
    idx = np.flatnonzero(near)
    return float(axis_deg[idx[int(np.argmax(profile[idx]))]])


def centre_pv(axis_deg, profile):
    """Population-vector centre: circular mean of the rectified profile."""
    w = np.clip(profile, 0.0, None)
    total = w.sum()
    if total <= 0:
        return float(axis_deg[int(np.argmax(profile))])
    th = np.deg2rad(axis_deg)
    return float(np.rad2deg(np.arctan2((w * np.sin(th)).sum(),
                                       (w * np.cos(th)).sum())))


def run_trial(items, config, probed, record_every=1):
    """Encode, run the explicit-network delay, record bump centre and width."""
    samples, weights, model, likelihood, opt, mll = setup_trial(items, config)
    encoding_phase(model, likelihood, opt, mll, samples, weights,
                   config["training"]["encoding_epochs"])
    enc_m, enc_l = clone_state(model), clone_state(likelihood)
    enc_o = copy.deepcopy(opt.state_dict())

    query_loc, true_color = items[probed]
    T = config["training"]["maintenance_epochs"]

    rnn_model, rnn_lik, info = explicit_maintenance(
        enc_m, enc_l, items, config, cue_idx=None, enc_opt_state=enc_o,
        record=True)
    m_traj = info["m_traj"]
    C_traj = info["C_traj"]

    centres, centres_pv, widths, epochs = [], [], [], []

    def record(ep, mvec=None, C=None):
        if mvec is not None:
            set_mean(rnn_model, mvec)
            set_covar(rnn_model, C)
        mean, var = decode_slice(rnn_model, rnn_lik, query_loc)
        prev = centres[-1] if centres else float(true_color)
        centres.append(track_peak(DECODE_AXIS, mean, prev))
        centres_pv.append(centre_pv(DECODE_AXIS, mean))
        widths.append(float(np.mean(var)))
        epochs.append(ep)

    # Delay onset = encoding end (m0 / C0 before any maintenance step).
    C0 = enc_m["variational_strategy._variational_distribution."
               "chol_variational_covar"].double().clone()
    m0 = enc_m["variational_strategy._variational_distribution."
               "variational_mean"].double().clone()
    record(0, m0, C0)

    for epoch, (mvec, C) in enumerate(zip(m_traj, C_traj), start=1):
        if epoch % record_every == 0 or epoch == T:
            record(epoch, mvec, C)

    # Normalized report at the probed location from the delay-end network.
    set_mean(rnn_model, m_traj[-1])
    set_covar(rnn_model, C_traj[-1])
    report_color = float(normalized_reports(
        rnn_model, [float(query_loc)], config["normalization"], n_draws=1)[0, 0])

    return (np.asarray(epochs), np.asarray(centres), np.asarray(centres_pv),
            np.asarray(widths), report_color, float(true_color))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--set-size", type=int, default=1)
    ap.add_argument("--record-every", type=int, default=2)
    args = ap.parse_args()

    config = load_config(filename="config_retrocue.yaml")
    base = config["experiment"]["random_seed"]

    disp_by_epoch = None
    disp_pv_by_epoch = None
    epochs_ref = None
    widths = []
    end_drift = []
    end_drift_pv = []
    report_err = []

    for t in range(args.trials):
        s = base + t
        seed_all(s)
        items = generator.generate_items(args.set_size, seed=s)
        probed = t % args.set_size
        ep, centres, centres_pv, wid, report, true_c = run_trial(
            items, config, probed, record_every=args.record_every)

        if disp_by_epoch is None:
            epochs_ref = ep
            disp_by_epoch = [[] for _ in ep]
            disp_pv_by_epoch = [[] for _ in ep]
        for i, d in enumerate(_circ_disp(centres, centres[0])):
            disp_by_epoch[i].append(d)
        for i, d in enumerate(_circ_disp(centres_pv, centres_pv[0])):
            disp_pv_by_epoch[i].append(d)
        widths.append(wid)

        end_drift.append(_circ_disp(centres[-1], true_c))
        end_drift_pv.append(_circ_disp(centres_pv[-1], true_c))
        report_err.append(_circ_disp(report, true_c))
        if (t + 1) % 20 == 0:
            print(f"  trial {t+1}/{args.trials} done", flush=True)

    disp_by_epoch = [np.asarray(v, float) for v in disp_by_epoch]
    disp_pv_by_epoch = [np.asarray(v, float) for v in disp_pv_by_epoch]
    disp_matrix = np.vstack(disp_by_epoch).T
    disp_matrix_pv = np.vstack(disp_pv_by_epoch).T
    diff_var = np.asarray([np.var(v) for v in disp_by_epoch])
    diff_var_pv = np.asarray([np.var(v) for v in disp_pv_by_epoch])
    end_drift_pv = np.asarray(end_drift_pv, float)
    widths = np.asarray(widths, float)
    mean_width = widths.mean(axis=0)
    end_drift = np.asarray(end_drift, float)
    report_err = np.asarray(report_err, float)

    A = np.vstack([epochs_ref, np.ones_like(epochs_ref, dtype=float)]).T
    slope, intercept = np.linalg.lstsq(A, diff_var, rcond=None)[0]
    D = slope / 2.0

    slope_pv, intercept_pv = np.linalg.lstsq(A, diff_var_pv, rcond=None)[0]
    r_readout = float(np.corrcoef(end_drift, report_err)[0, 1])
    r_readout_pv = float(np.corrcoef(end_drift_pv, report_err)[0, 1])

    os.makedirs(os.path.dirname(NPZ), exist_ok=True)
    np.savez(NPZ, epochs=epochs_ref, diff_var=diff_var, mean_width=mean_width,
             disp_matrix=disp_matrix, end_drift=end_drift, report_err=report_err,
             slope=slope, intercept=intercept, D=D, r_readout=r_readout,
             disp_matrix_pv=disp_matrix_pv, diff_var_pv=diff_var_pv,
             end_drift_pv=end_drift_pv, slope_pv=slope_pv,
             intercept_pv=intercept_pv, D_pv=slope_pv / 2.0,
             r_readout_pv=r_readout_pv, n_decode=N_DECODE,
             set_size=args.set_size, n_trials=args.trials,
             backend="explicit_rnn")
    print(f"\nsaved -> {NPZ}")

    def line(lab, dm, sl):
        from scipy.stats import spearmanr
        ep = epochs_ref.astype(float)
        k = ep > 0
        a = np.abs(dm[:, k])
        rho = spearmanr(np.tile(ep[k], (a.shape[0], 1)).ravel(), a.ravel()).statistic
        print(f"  {lab:26s} rho={rho:+.3f}  drift {a[:, 0].mean():.3f} -> "
              f"{a[:, -1].mean():.3f} deg  D={sl/2:.5f}")

    print(f"decoded on a {360/N_DECODE:.2f} deg grid (explicit network):")
    line("argmax", disp_matrix, slope)
    line("population vector", disp_matrix_pv, slope_pv)

    _preview(epochs_ref, diff_var, slope, intercept, mean_width,
             end_drift, report_err, args)


def _preview(epochs, diff_var, slope, intercept, mean_width,
             end_drift, report_err, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.5))

    ax[0].plot(epochs, diff_var, "o", ms=4, color="#2166AC", label="explicit RNN")
    ax[0].plot(epochs, slope * epochs + intercept, "-", color="#B2182B", lw=1.6,
               label=f"linear fit (slope {slope:.3f})")
    ax[0].set_xlabel("delay (maintenance epoch)")
    ax[0].set_ylabel(r"bump-centre variance (deg$^2$)")
    ax[0].set_title("Diffusing bump (Wimmer 2014)")
    ax[0].legend(frameon=False, fontsize=8)

    ax[1].plot(epochs, mean_width / mean_width[0], "-", color="#2A887E", lw=1.8)
    ax[1].set_xlabel("delay (maintenance epoch)")
    ax[1].set_ylabel("relative posterior width")
    ax[1].set_title("KL forgetting widens the posterior")

    lim = 1.05 * max(np.abs(end_drift).max(), np.abs(report_err).max(), 1.0)
    ax[2].plot([-lim, lim], [-lim, lim], "--", color="grey", lw=0.9)
    ax[2].scatter(end_drift, report_err, s=16, color="#B86B1E",
                  edgecolors="white", linewidths=0.35)
    ax[2].set_xlim(-lim, lim)
    ax[2].set_ylim(-lim, lim)
    ax[2].set_aspect("equal", adjustable="box")
    ax[2].set_xlabel("decoded delay-end drift (deg)")
    ax[2].set_ylabel("sampled report error (deg)")
    ax[2].set_title("Peak position and sampled report")

    fig.suptitle(
        f"Explicit-network bump-diffusion  (N={args.set_size}, "
        f"{args.trials} trials)", fontsize=12)
    fig.tight_layout()
    fig.savefig(PREVIEW, dpi=150)
    plt.close(fig)
    print(f"saved preview -> {PREVIEW}")


if __name__ == "__main__":
    main()
