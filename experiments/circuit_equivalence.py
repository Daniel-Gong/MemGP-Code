"""GP vs explicit recurrent network on matched trials (Figure 5).

Writes results/circuit/rnn_equiv.npz with

  * a demo trial: epoch-by-epoch cued-item memory trace for the GP and the
    explicit network, their relative divergence, the connectivity W and its
    eigenvalues, and posterior-variance trajectories at cued vs uncued items;
  * an example GP and network posterior-mean surface;
  * per display (N = 4) and condition (neutral, cued), at the probed item:
    posterior mean and variance under the GP and under the network, the
    absolute recall error of one normalized report, and the mean absolute
    error over 100 normalized reports (Fig. 5c);
  * the relative difference between the GP's and the network's variational
    mean and covariance, and a check of the network gradient against the GP.

During maintenance the GP updates only m and S (kernel and inducing grid fixed
at encoding end), which is the computation the network performs.

Run:  python -m experiments.circuit_equivalence --trials 100
"""
import copy, argparse
import numpy as np
import torch

from memgp import generator
from memgp.paths import results_dir
from memgp.simulation import setup_trial, encoding_phase, load_config, normalized_reports
from memgp.circuit import (
    seed_all, fresh_model, clone_state, var_mean, posterior_surface,
    record_gp_maintenance, build_linear_rnn, adam_integrate_selfsustain,
    whitened_covar, warm_adam_state, explicit_maintenance, moments_at,
    LinearRNN, verify_against_gp, set_mean, activation_at,
    run_branch, RC_N, CONDITIONS,
)

OUT = str(results_dir("circuit") / "rnn_equiv.npz")
N_DRAWS = 100


def demo_trial(config):
    """Single cued N=4 trial: memory trace, divergence, W, and precision channel."""
    s = config["experiment"]["random_seed"]
    seed_all(s)
    items = generator.generate_items(RC_N, seed=s)
    probed = 0
    uncued = 1  # a neighboring item for variance contrast
    loc_c, col_c = items[probed]
    loc_u, col_u = items[uncued]

    samples, weights, model, lik, opt, mll = setup_trial(items, config)
    encoding_phase(model, lik, opt, mll, samples, weights,
                   config["training"]["encoding_epochs"])
    enc_m, enc_l = clone_state(model), clone_state(lik)
    enc_o = copy.deepcopy(opt.state_dict())
    warm_m = warm_adam_state(model, enc_o, "variational_mean")

    traj_full, (var_cued, var_uncued) = record_gp_maintenance(
        enc_m, enc_l, items, config, probed, frozen=False,
        track_points=[(loc_c, col_c), (loc_u, col_u)], enc_opt_state=enc_o)
    traj_froz, (var_cued_froz, var_uncued_froz) = record_gp_maintenance(
        enc_m, enc_l, items, config, probed, frozen=True,
        track_points=[(loc_c, col_c), (loc_u, col_u)], enc_opt_state=enc_o)

    Wc, bc, m0, bc_fn, refresh_t, t0, _ = build_linear_rnn(
        enc_m, enc_l, items, config, probed)
    Wn, _, _, bn_fn, _, _, _ = build_linear_rnn(enc_m, enc_l, items, config, None)
    T = config["training"]["maintenance_epochs"]
    cs = config["training"]["cue_start_epoch"]
    traj_exp = adam_integrate_selfsustain(
        m0, Wn, Wc, bn_fn, bc_fn, cs, T,
        lr=config["training"]["maintenance_lr"],
        t0=t0, refresh_t=refresh_t, record=True, state0=warm_m)

    rmodel, _ = fresh_model(config); rmodel.load_state_dict(enc_m)

    def act_series(traj, loc, col):
        return [activation_at(set_mean(rmodel, m), loc, col) for m in traj]

    a_full = np.asarray(act_series(traj_full, loc_c, col_c))
    a_froz = np.asarray(act_series(traj_froz, loc_c, col_c))
    a_exp = np.asarray(act_series(traj_exp, loc_c, col_c))
    div_froz = np.asarray([float((e - f).norm() / (f.norm() + 1e-12))
                           for e, f in zip(traj_exp, traj_froz)])
    div_full = np.asarray([float((e - f).norm() / (f.norm() + 1e-12))
                           for e, f in zip(traj_exp, traj_full)])

    rnn = LinearRNN(Wc, bc)
    return dict(a_full=a_full, a_froz=a_froz, a_exp=a_exp,
                states_full=np.stack([m.cpu().numpy() for m in traj_full]),
                states_froz=np.stack([m.cpu().numpy() for m in traj_froz]),
                states_exp=np.stack([m.cpu().numpy() for m in traj_exp]),
                div_froz=div_froz, div_full=div_full, cue_start=cs,
                W=rnn.W.cpu().numpy(), W_eig=rnn.eig.cpu().numpy(),
                var_cued=var_cued, var_uncued=var_uncued,
                var_cued_froz=var_cued_froz, var_uncued_froz=var_uncued_froz)


def _recall(model, items, probed, norm, seed):
    """|error| of the first normalized report and the mean over N_DRAWS."""
    torch.manual_seed(seed)
    locs = [float(l) for l, _ in items]
    reports = normalized_reports(model, locs, norm, n_draws=N_DRAWS)[:, probed]
    err = np.abs((reports - items[probed][1] + 180.0) % 360.0 - 180.0)
    return float(err[0]), float(err.mean())


def aggregate(config, n_trials):
    keys = ("mean_gp", "mean_net", "var_gp", "var_net", "err1_gp", "err1_net",
            "mae_gp", "mae_net", "reldiff", "covdiff")
    agg = {c: {k: [] for k in keys} for c in CONDITIONS}
    example, verify = {}, {}
    norm = config["normalization"]

    for t in range(n_trials):
        s = config["experiment"]["random_seed"] + t
        seed_all(s)
        items = generator.generate_items(RC_N, seed=s)
        probed = t % RC_N
        loc, col = float(items[probed][0]), float(items[probed][1])

        samples, weights, model, lik, opt, mll = setup_trial(items, config)
        encoding_phase(model, lik, opt, mll, samples, weights,
                       config["training"]["encoding_epochs"])
        enc_m, enc_l = clone_state(model), clone_state(lik)
        # The delay inherits the encoding optimizer, as in run_simulation_trial.
        enc_o = copy.deepcopy(opt.state_dict())

        for cond in CONDITIONS:
            cue = probed if cond == "cued" else None
            m_gp, _ = run_branch(enc_m, enc_l, items, config, cue, frozen=False,
                                 enc_opt_state=enc_o)
            m_net, _, _ = explicit_maintenance(enc_m, enc_l, items, config, cue,
                                               enc_opt_state=enc_o)

            if not verify and cond == "cued":
                Wc, bc, m0, _, _, _, ctx = build_linear_rnn(enc_m, enc_l, items, config, cue)
                mism, gscale = verify_against_gp(LinearRNN(Wc, bc), ctx, m0)
                verify = dict(grad_mismatch=mism, grad_scale=gscale)

            a = agg[cond]
            for tag, mdl in (("gp", m_gp), ("net", m_net)):
                mu, var = moments_at(mdl, loc, col)
                a[f"mean_{tag}"].append(mu)
                a[f"var_{tag}"].append(var)
                e1, mae = _recall(mdl, items, probed, norm, 10_000 * (t + 1))
                a[f"err1_{tag}"].append(e1)
                a[f"mae_{tag}"].append(mae)
            vg, vn = var_mean(m_gp).double(), var_mean(m_net).double()
            a["reldiff"].append(float((vn - vg).norm() / (vg.norm() + 1e-9)))
            sg, sn = whitened_covar(m_gp), whitened_covar(m_net)
            a["covdiff"].append(float((sn - sg).norm() / (sg.norm() + 1e-9)))

            if t == 0 and cond == "cued":
                example = dict(surf_full=posterior_surface(m_gp),
                               surf_exp=posterior_surface(m_net),
                               items=np.asarray(items, float), probed=probed)
        print(f"  trial {t+1}/{n_trials} done", flush=True)
    return agg, example, verify


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    args = ap.parse_args()

    config = load_config(filename="config_retrocue.yaml")
    config["experiment"]["set_sizes"] = [RC_N]

    print(">>> demo trajectory (one cued trial) ...", flush=True)
    demo = demo_trial(config)

    print(">>> GP vs network over trials ...", flush=True)
    agg, example, verify = aggregate(config, args.trials)

    blob = dict(**{f"demo_{k}": v for k, v in demo.items()},
                **{f"ex_{k}": v for k, v in example.items()},
                **{f"verify_{k}": v for k, v in verify.items()})
    for cond in CONDITIONS:
        for k, v in agg[cond].items():
            blob[f"{cond}_{k}"] = np.asarray(v, float)
    np.savez(OUT, **blob)
    print(f"\nsaved -> {OUT}", flush=True)

    print(f"[verify] max|grad L_GP - (Wm-b)| = {verify['grad_mismatch']:.2e} "
          f"(scale {verify['grad_scale']:.2e})")
    for cond in CONDITIONS:
        a = agg[cond]
        r_mae = np.corrcoef(a["mae_gp"], a["mae_net"])[0, 1]
        r_mean = np.corrcoef(a["mean_gp"], a["mean_net"])[0, 1]
        r_var = np.corrcoef(a["var_gp"], a["var_net"])[0, 1]
        print(f"  {cond}: r(mean)={r_mean:.4f} r(var)={r_var:.4f} "
              f"r(MAE over {N_DRAWS} draws)={r_mae:.4f}  "
              f"rel. diff m {np.mean(a['reldiff']):.1e}, S {np.mean(a['covdiff']):.1e}")


if __name__ == "__main__":
    main()
