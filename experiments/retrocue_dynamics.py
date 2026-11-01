"""
Maintenance dynamics under a retrocue (Fig. 3k-l).

One pool of 100 trials. Each trial is an N=4 display, encoded once and
maintained once with the retrocue on one item. Every maintenance epoch
records the posterior mean and variance at the cued item and the mean of
the other items.

The trial seeds are the pre-October 2026 pool, ``1000 + 7*t``.

Output: results/retrocue/retrocue_dynamics.npz with arrays of shape
(trials, maintenance epochs): cued_act, uncued_act, cued_var, uncued_var;
plus cue_epoch and n_trials.

    python -m experiments.retrocue_dynamics --n_trials 100 --workers 8
"""
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from memgp.circuit import seed_all
from memgp.generator import generate_items
from memgp.paths import results_dir
from memgp.simulation import load_config, configure_threads
from memgp.tracking import run_trial_with_tracking

OUT_DIR = results_dir("retrocue")
N_TRIALS = 100
SET_SIZE = 4


def _trial_job(payload):
    """One trial. Payload is (index, config)."""
    import torch
    torch.set_num_threads(1)
    t, cfg = payload
    seed = 1000 + 7 * t
    seed_all(seed)
    items = generate_items(SET_SIZE, seed=seed)
    cued = int(np.random.RandomState(seed).randint(0, SET_SIZE))
    (_pts, _vals, acts, variances, _phase, items, cued, _cue) = \
        run_trial_with_tracking(items, cfg, cued_item_idx=cued)
    enc = cfg["training"]["encoding_epochs"]
    acts = np.asarray(acts)[enc:]
    variances = np.asarray(variances)[enc:]
    others = [i for i in range(SET_SIZE) if i != cued]
    return (t, acts[:, cued], acts[:, others].mean(axis=1),
            variances[:, cued], variances[:, others].mean(axis=1))


def run_trials(n_trials, workers):
    cfg = load_config(filename="config_retrocue.yaml")
    jobs = [(t, cfg) for t in range(n_trials)]
    cued_act = [None] * n_trials
    uncued_act = [None] * n_trials
    cued_var = [None] * n_trials
    uncued_var = [None] * n_trials

    def _store(result):
        t, ca, ua, cv, uv = result
        cued_act[t] = ca
        uncued_act[t] = ua
        cued_var[t] = cv
        uncued_var[t] = uv

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            done = 0
            for result in pool.map(_trial_job, jobs, chunksize=1):
                _store(result)
                done += 1
                if done % 10 == 0 or done == n_trials:
                    print(f"  trial {done}/{n_trials} done", flush=True)
    else:
        for job in jobs:
            _store(_trial_job(job))
            t = job[0]
            if (t + 1) % 10 == 0 or t + 1 == n_trials:
                print(f"  trial {t + 1}/{n_trials} done", flush=True)

    path = OUT_DIR / "retrocue_dynamics.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        cued_act=np.stack(cued_act),
        uncued_act=np.stack(uncued_act),
        cued_var=np.stack(cued_var),
        uncued_var=np.stack(uncued_var),
        cue_epoch=int(cfg["training"]["cue_start_epoch"]),
        n_trials=n_trials,
    )
    print(f"wrote {path}: {np.stack(cued_act).shape} (trials, epochs)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_trials", type=int, default=N_TRIALS)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    configure_threads(default=1)
    run_trials(args.n_trials, args.workers)


if __name__ == "__main__":
    main()
