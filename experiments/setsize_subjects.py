"""
Subject-level set-size experiments (Figure 2).

  design       trials per subject           output (results/setsize/)       figure
  ----------   --------------------------   ------------------------------   -------
  main         100 per N, N in {1,2,4,6,8}  setsize_main_trials.csv          2f-i, S2
  M            50 per N per grid size       setsize_M_trials.csv             2j
  beta         50 per N per beta            setsize_beta_trials.csv          2k
  lambda       50 per N per lengthscale     setsize_lambda_trials.csv        2l
  misbinding   30 per N, N in {2,4,6,8}     setsize_misbinding_trials.csv    2m

Each display probes one item, as in human continuous-report tasks, so every
set size gives the same number of errors per subject. Every item of the
display is written out (for non-target analyses); only the probed row has a
report, the argmax of one joint posterior sample after divisive
normalization (``retrieve_color``). Misbinding probes every item and draws
100 normalized reports per probe.

Subject s uses seed base 42 + 100003*s; trial t at set size N uses
seed = base + 10007*t + N (misbinding adds 500000). Displays are therefore
shared across the levels of a sweep, and every row can be regenerated alone.

Run everything for one design in one process:
    python -m experiments.setsize_subjects main
One shard (subject, and sweep level for M / beta / lambda), then merge:
    python -m experiments.setsize_subjects M --subject_idx 3 --level 2
    python -m experiments.setsize_subjects M --merge
"""
import argparse
import copy
import glob

import numpy as np
import pandas as pd

from experiments.registry import DESIGN_LEVELS
from memgp import generator
from memgp.paths import results_dir
from memgp.simulation import (
    run_simulation_trial, retrieve_color, normalized_reports, load_config,
    configure_threads,
)

OUT_DIR = results_dir("setsize")
N_SUBJECTS = 20

DESIGNS = {
    "main": dict(set_sizes=(1, 2, 4, 6, 8), trials=100, levels=[None]),
    "M": dict(set_sizes=(1, 2, 4, 6, 8), trials=50, levels=[5, 7, 10, 14]),
    "beta": dict(set_sizes=(1, 2, 4, 6, 8), trials=50, levels=[0.0, 1.0, 5.0, 20.0]),
    "lambda": dict(set_sizes=(1, 2, 4, 6, 8), trials=50, levels=[5.0, 10.0, 20.0, 40.0]),
    "misbinding": dict(set_sizes=(2, 4, 6, 8), trials=30, levels=[None]),
}
assert {d: len(s["levels"]) for d, s in DESIGNS.items()} == DESIGN_LEVELS
# Displays are randomly rotated, so item 0 sits at a random location.
PROBED_ITEM = 0
MISBIND_DRAWS = 100
MISBIND_WINDOW = 30.0


def subject_base(subject):
    return 42 + 100003 * int(subject)


def design_config(design, level):
    cfg = copy.deepcopy(load_config(filename="config_set_size.yaml"))
    if design == "M":
        cfg["model"]["inducing_grid_size"] = int(level)
    elif design == "beta":
        cfg["training"]["beta"] = float(level)
    elif design == "lambda":
        cfg["model"]["loc_lengthscale"] = float(level)
        cfg["model"]["color_lengthscale"] = float(level)
    return cfg


def _cdist(a, b):
    x = np.abs(np.asarray(a, float) - np.asarray(b, float)) % 360.0
    return np.minimum(x, 360.0 - x)


def run_cell(design, subject, level, n_trials=None):
    """All trials of one (subject, level) cell. Returns a DataFrame."""
    spec = DESIGNS[design]
    cfg = design_config(design, level)
    norm = cfg["normalization"]
    base = subject_base(subject)
    n_trials = spec["trials"] if n_trials is None else int(n_trials)
    rows = []
    for N in spec["set_sizes"]:
        for t in range(n_trials):
            seed = base + 10007 * t + N + (500000 if design == "misbinding" else 0)
            items = generator.generate_items(N, seed=seed)
            model, _, _ = run_simulation_trial(items, cfg)
            locs = [float(l) for l, _ in items]
            if design == "misbinding":
                reports = normalized_reports(model, locs, norm, n_draws=MISBIND_DRAWS)
                cols = np.array([c for _, c in items], float)
                for p in range(N):
                    near_t = _cdist(reports[:, p], cols[p]) < MISBIND_WINDOW
                    for j in range(N):
                        if j == p:
                            continue
                        d = int(round(float(_cdist(locs[p], locs[j])) / 45.0) * 45)
                        near_j = _cdist(reports[:, p], cols[j]) < MISBIND_WINDOW
                        rows.append(dict(subject=subject, N=N, trial=t, probe=p,
                                         other=j, distance=min(max(d, 45), 180),
                                         swap=float(np.mean(near_j & ~near_t))))
                continue
            for i, (l, c) in enumerate(items):
                err = np.nan
                if i == PROBED_ITEM:
                    err = float(retrieve_color(model, l, c, normalization=norm,
                                               item_locs=locs, probabilistic=True))
                rows.append(dict(subject=subject, level=level, N=N, trial=t,
                                 item=i, probed=i == PROBED_ITEM,
                                 loc=float(l), color=float(c),
                                 report=(float(c) + err + 180.0) % 360.0 - 180.0,
                                 error=err))
        print(f"  [{design}] subject={subject} level={level} N={N} done", flush=True)
    return pd.DataFrame(rows)


def _shard_dir(design):
    path = OUT_DIR / "shards" / design
    path.mkdir(parents=True, exist_ok=True)
    return path


def shard_path(design, subject, level_idx):
    return _shard_dir(design) / f"subject_{subject:03d}_level_{level_idx}.csv"


def write_shard(design, subject, level_idx, n_trials=None):
    level = DESIGNS[design]["levels"][level_idx]
    df = run_cell(design, subject, level, n_trials)
    df.insert(0, "design", design)
    path = shard_path(design, subject, level_idx)
    df.to_csv(path, index=False)
    print(f"wrote {path} ({len(df)} rows)", flush=True)


def merge(design):
    files = sorted(glob.glob(str(_shard_dir(design) / "subject_*.csv")))
    if not files:
        raise SystemExit(f"no shards for {design}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    out = OUT_DIR / f"setsize_{design}_trials.csv"
    df.to_csv(out, index=False)
    print(f"merged {len(files)} shards -> {out} ({len(df)} rows, "
          f"{df['subject'].nunique()} subjects)")


def tasks(design, n_subjects=N_SUBJECTS):
    """(subject, level index) pairs that make up one design."""
    return [(s, li) for s in range(n_subjects)
            for li in range(len(DESIGNS[design]["levels"]))]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("design", choices=sorted(DESIGNS))
    ap.add_argument("--subject_idx", type=int, default=None)
    ap.add_argument("--level", type=int, default=None,
                    help="Level index for M / beta / lambda (default: all levels).")
    ap.add_argument("--n_subjects", type=int, default=N_SUBJECTS)
    ap.add_argument("--n_trials", type=int, default=None,
                    help="Override trials per cell (for quick tests).")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    configure_threads(default=1)

    if args.merge:
        merge(args.design)
        return
    subjects = ([args.subject_idx] if args.subject_idx is not None
                else range(args.n_subjects))
    levels = ([args.level] if args.level is not None
              else range(len(DESIGNS[args.design]["levels"])))
    for s in subjects:
        for li in levels:
            write_shard(args.design, s, li, args.n_trials)
    if args.subject_idx is None and args.level is None:
        merge(args.design)


if __name__ == "__main__":
    main()
