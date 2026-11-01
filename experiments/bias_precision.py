"""
Memory precision vs repulsion across simulated subjects (Fig. 4g).

Each of 24 simulated subjects (the size of Chunharas et al. 2022 Exp. 4) gets
its own encoding noise, sigma = loc_std = color_std ~ Uniform[10, 40] deg,
drawn from RandomState(1000 + subject). Each subject then runs N=2 displays
(target at 0 deg, distractor at 90 deg) at color distances
{20, 40, ..., 140} deg, 50 trials per distance, with the normalized readout.

Output: results/bias/bias_precision_trials.csv, one row per trial with the
signed error and the oriented error (positive = away from the distractor).
The figure computes, per subject and averaged over distances, the circular SD
and the bias in percent (100 * P(away) - 50) or in degrees.

All subjects:              python -m experiments.bias_precision
One subject, then merge:   python -m experiments.bias_precision --subject_idx 5
                           python -m experiments.bias_precision --merge
"""
import argparse
import copy
import glob

import numpy as np
import pandas as pd
import torch

from memgp.paths import results_dir
from memgp.simulation import (
    run_simulation_trial, retrieve_color, load_config, configure_threads,
)

OUT_DIR = results_dir("bias")
N_SUBJECTS = 24
N_TRIALS = 50
DISTANCES = (20, 40, 60, 80, 100, 120, 140)
NOISE_RANGE = (10.0, 40.0)
T_LOC, NT_LOC = 0.0, 90.0


def subject_noise(subject):
    return float(np.random.RandomState(1000 + subject).uniform(*NOISE_RANGE))


def run_subject(subject, n_trials=N_TRIALS):
    cfg = copy.deepcopy(load_config(filename="config_bias.yaml"))
    sigma = subject_noise(subject)
    cfg["data"]["color_std"] = sigma
    cfg["data"]["loc_std"] = sigma
    norm = cfg["normalization"]
    base = 42 + 100003 * int(subject)
    rows = []
    for di, dist in enumerate(DISTANCES):
        for t in range(n_trials):
            seed = base + 1000 * di + t
            np.random.seed(seed)
            torch.manual_seed(seed)
            rng = np.random.RandomState(seed)
            col = float(rng.uniform(-180, 180))
            direction = int(rng.choice([-1, 1]))
            ncol = (col + direction * dist + 180.0) % 360.0 - 180.0
            model, _, _ = run_simulation_trial([(T_LOC, col), (NT_LOC, ncol)], cfg)
            err = float(retrieve_color(model, T_LOC, col, normalization=norm,
                                       item_locs=[T_LOC, NT_LOC], probabilistic=True))
            rows.append(dict(subject=subject, sigma=sigma, distance=dist, trial=t,
                             direction=direction, error=err,
                             oriented_error=-direction * err))
        print(f"  [bias_precision] subject {subject} distance {dist} done", flush=True)
    return pd.DataFrame(rows)


def _shard_dir():
    path = OUT_DIR / "shards" / "bias_precision"
    path.mkdir(parents=True, exist_ok=True)
    return path


def merge():
    files = sorted(glob.glob(str(_shard_dir() / "subject_*.csv")))
    if not files:
        raise SystemExit("no bias_precision shards")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    out = OUT_DIR / "bias_precision_trials.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out} ({df['subject'].nunique()} subjects, {len(df)} trials)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject_idx", type=int, default=None)
    ap.add_argument("--n_subjects", type=int, default=N_SUBJECTS)
    ap.add_argument("--n_trials", type=int, default=N_TRIALS)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    configure_threads(default=1)
    if args.merge:
        merge()
        return
    subjects = [args.subject_idx] if args.subject_idx is not None else range(args.n_subjects)
    for s in subjects:
        run_subject(s, args.n_trials).to_csv(
            _shard_dir() / f"subject_{s:03d}.csv", index=False)
    if args.subject_idx is None:
        merge()


if __name__ == "__main__":
    main()
