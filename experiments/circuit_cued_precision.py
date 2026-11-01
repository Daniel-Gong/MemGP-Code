"""Paired cued vs neutral maintenance on the explicit circuit (Figure 5f--g).

A retro-cue stiffens W locally, so the width flow contracts S toward a
narrower attractor at the attended item. Within-trial cued-vs-uncued
contrasts confound that effect with item position, so this script pairs the
two delays on one encoding.

Each trial encodes an N=4 array, then runs the explicit network twice from
the same encoding-end state (cued and neutral). Posterior mean and variance
at the probed item are recorded at delay onset and after every maintenance
epoch.

Output: results/circuit/cued_precision.npz (Fig. 5f-g)

Run:  python -m experiments.circuit_cued_precision --trials 100 --workers 8
"""
import argparse
import copy
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from memgp import generator
from memgp.paths import results_dir
from memgp.simulation import setup_trial, encoding_phase, load_config
from memgp.circuit import (
    seed_all, clone_state, explicit_maintenance, RC_N,
)

NPZ = str(results_dir("circuit") / "cued_precision.npz")


def _probed_traj(enc_m, enc_l, items, config, cue_idx, enc_o, probed):
    """Mean and variance at the probed item after each maintenance epoch."""
    loc, col = float(items[probed][0]), float(items[probed][1])
    _, _, info = explicit_maintenance(
        enc_m, enc_l, items, config, cue_idx, enc_opt_state=enc_o,
        track_points=[(loc, col)])
    # Drop delay onset; keep the post-epoch samples.
    return info["mean_trajs"][0][1:], info["var_trajs"][0][1:]


def run_trial(items, config, probed):
    """Shared encoding, then paired cued / neutral delays at the probed item."""
    samples, weights, model, lik, opt, mll = setup_trial(items, config)
    encoding_phase(model, lik, opt, mll, samples, weights,
                   config["training"]["encoding_epochs"])
    enc_m, enc_l = clone_state(model), clone_state(lik)
    enc_o = copy.deepcopy(opt.state_dict())

    mean_cued, var_cued = _probed_traj(
        enc_m, enc_l, items, config, probed, enc_o, probed)
    mean_neutral, var_neutral = _probed_traj(
        enc_m, enc_l, items, config, None, enc_o, probed)
    return mean_cued, var_cued, mean_neutral, var_neutral


def _trial_job(payload):
    """One paired trial. Payload is (index, base seed, set size, config)."""
    import torch
    torch.set_num_threads(1)
    t, base, set_size, config = payload
    seed_all(base + t)
    items = generator.generate_items(set_size, seed=base + t)
    probed = t % set_size
    mean_c, var_c, mean_n, var_n = run_trial(items, config, probed)
    return t, mean_c, var_c, mean_n, var_n, probed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--set-size", type=int, default=RC_N)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    config = load_config(filename="config_retrocue.yaml")
    config["experiment"]["set_sizes"] = [args.set_size]
    base = config["experiment"]["random_seed"] + 1000
    jobs = [(t, base, args.set_size, config) for t in range(args.trials)]

    mean_c = [None] * args.trials
    var_c = [None] * args.trials
    mean_n = [None] * args.trials
    var_n = [None] * args.trials
    probed_idx = [None] * args.trials

    def _store(result):
        t, mc, vc, mn, vn, probed = result
        mean_c[t] = mc
        var_c[t] = vc
        mean_n[t] = mn
        var_n[t] = vn
        probed_idx[t] = probed

    if args.workers > 1:
        import torch
        torch.set_num_threads(1)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            done = 0
            for result in pool.map(_trial_job, jobs, chunksize=1):
                _store(result)
                done += 1
                if done % 10 == 0 or done == args.trials:
                    print(f"  trial {done}/{args.trials} done", flush=True)
    else:
        for job in jobs:
            _store(_trial_job(job))
            t = job[0]
            if (t + 1) % 10 == 0:
                print(f"  trial {t + 1}/{args.trials} done", flush=True)

    mean_cued = np.stack(mean_c)
    mean_neutral = np.stack(mean_n)
    var_cued = np.stack(var_c)
    var_neutral = np.stack(var_n)
    probed_idx = np.asarray(probed_idx, int)
    cs = int(config["training"]["cue_start_epoch"])

    os.makedirs(os.path.dirname(NPZ), exist_ok=True)
    np.savez(NPZ, mean_cued=mean_cued, mean_neutral=mean_neutral,
             var_cued=var_cued, var_neutral=var_neutral,
             probed=probed_idx, cue_start=cs,
             set_size=args.set_size, n_trials=args.trials,
             backend="explicit_rnn_paired")
    print(f"\nsaved -> {NPZ}")

    def rel_end(var):
        # Refer to the last pre-cue sample so the paired branches share a
        # baseline of 1 at cue onset, then diverge.
        logp = np.log(1.0 / np.maximum(var, 1e-12))
        logp = logp - logp[:, cs - 1:cs]
        return float(np.exp(logp[:, -1].mean()))

    rc, rn = rel_end(var_cued), rel_end(var_neutral)
    dmc = mean_cued - mean_cued[:, cs - 1:cs]
    dmn = mean_neutral - mean_neutral[:, cs - 1:cs]
    print(f"  relative precision at delay end (vs cue onset): cued {rc:.3f}x  "
          f"neutral {rn:.3f}x  ratio {rc/rn:.3f}x "
          f"({args.trials} trials, N={args.set_size}, paired explicit network)")
    print(f"  delta mean at delay end (vs cue onset): "
          f"cued {dmc[:, -1].mean():.6e}  neutral {dmn[:, -1].mean():.6e}  "
          f"max |cued-neutral| {np.abs(mean_cued - mean_neutral).max():.6e}")


if __name__ == "__main__":
    main()
