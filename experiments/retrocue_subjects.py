"""
Subject-level retrocue experiments (Fig. 3f-j).

Each simulated subject runs ``n_trials`` independent trials with its own seed
base, and the group statistics treat the subject as the unit.

  retrocue     neutral vs cued recall error at one set size (Fig. 3f-g)
  flexibility  cue redirect: item A cued at epochs 50-99, item B at 100-199
               of a 200-epoch delay, N=4 (Fig. 3h)
  aggregate    N=2 vs N=4 summary and the one-tailed load comparison

Outputs (results/retrocue/):
  retrocue_subjects_N{2,4}.csv       per-subject neutral / cued means
  retrocue_trials_N{2,4}.csv         every probed-item signed error (Fig. 3i-j)
  retrocue_setsize_results.csv       per-set-size summary and paired t-tests
  retrocue_setsize_comparison.csv    benefit(N=4) > benefit(N=2), one-tailed
  cue_flexibility_subjects.csv       per-subject redirect errors
  cue_flexibility_summary.csv        group means and the trade-off t-test

All subjects in one process:
    python -m experiments.retrocue_subjects retrocue --set_size 2
    python -m experiments.retrocue_subjects retrocue --set_size 4
    python -m experiments.retrocue_subjects flexibility
    python -m experiments.retrocue_subjects aggregate

One subject per Slurm array task, then merge:
    python -m experiments.retrocue_subjects retrocue --set_size 4 --subject_idx 7
    python -m experiments.retrocue_subjects retrocue --set_size 4 --merge
"""

import argparse
import copy
import glob
import json

import numpy as np
import pandas as pd
import torch
import gpytorch
from scipy import stats
from tqdm import trange

from memgp import generator
from memgp.attention import SpatialProximityAttention
from memgp.generator import sample_training_data
from memgp.gp_model import WorkingMemoryGP
from memgp.paths import results_dir
from memgp.simulation import (
    run_simulation_trial, retrieve_color, load_config, device,
)

OUT_DIR = results_dir("retrocue")

# Cue-redirect design (Fig. 3h)
FLEX_SET_SIZE = 4
FLEX_ENCODING_EPOCHS = 100
FLEX_MAINTENANCE_EPOCHS = 200
CUE_A_START = 50
CUE_B_START = 100


def _patch_config(base_config: dict, n_trials: int, subject_idx: int) -> dict:
    """Return a deep-copied config with n_trials and a subject-unique random seed."""
    cfg = copy.deepcopy(base_config)
    cfg["experiment"]["n_trials"] = n_trials
    # shift the base seed by subject index * large prime so seeds don't overlap
    cfg["experiment"]["random_seed"] = (
        base_config["experiment"]["random_seed"] + 1_000_000 + subject_idx * 100003
    )
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Per-subject runners
# ──────────────────────────────────────────────────────────────────────────────

def _subject_retrocue(config: dict, target_set_size: int = 4):
    """One subject's neutral vs cued retrocue run.

    Returns the mean absolute error per condition and every signed error of
    the probed item (Fig. 3i-j).
    """
    norm = config.get("normalization", None)
    cued_signed = []
    neutral_signed = []

    for t in trange(config["experiment"]["n_trials"],
                    desc="  Retrocue", leave=False):
        seed = config["experiment"]["random_seed"] + t
        items = generator.generate_items(target_set_size, seed=seed)
        probed_idx = np.random.randint(0, target_set_size)
        item_locs = [float(l) for l, _ in items]

        # Neutral (no cue)
        model, _, _ = run_simulation_trial(items, config, cued_item_idx=None)
        neutral_signed.append(float(retrieve_color(
            model, items[probed_idx][0], items[probed_idx][1],
            normalization=norm, item_locs=item_locs, probabilistic=True,
        )))

        # Cued
        model, _, _ = run_simulation_trial(items, config, cued_item_idx=probed_idx)
        cued_signed.append(float(retrieve_color(
            model, items[probed_idx][0], items[probed_idx][1],
            normalization=norm, item_locs=item_locs, probabilistic=True,
        )))

    return {
        "mean_neutral": float(np.mean(np.abs(neutral_signed))),
        "mean_cued": float(np.mean(np.abs(cued_signed))),
        "neutral_signed": neutral_signed,
        "cued_signed": cued_signed,
    }


def _abs_errors(model, items, config):
    """Absolute recall error for every item, with the config's DN readout."""
    norm = config.get("normalization", None)
    item_locs = [float(loc) for loc, _ in items]
    return {
        i: abs(retrieve_color(
            model, loc, col,
            normalization=norm, item_locs=item_locs, probabilistic=True,
        ))
        for i, (loc, col) in enumerate(items)
    }


def run_flexibility_trial(items, config, cued_idx_A: int, cued_idx_B: int):
    """One trial with two sequential retrocues.

    Returns (snapshot_post_A, snapshot_post_B), each mapping item index to the
    absolute recall error at the end of that cue window.
    """
    samples, weights, _ = sample_training_data(
        items,
        n_samples_per_item=config["data"]["n_samples_per_item"],
        loc_std=config["data"]["loc_std"],
        color_std=config["data"]["color_std"],
        loc_encoding_noise_std=config["data"]["loc_encoding_noise_std"],
        color_encoding_noise_std=config["data"]["color_encoding_noise_std"],
    )

    model = WorkingMemoryGP(
        inducing_grid_size=config["model"]["inducing_grid_size"],
        loc_lengthscale=config["model"]["loc_lengthscale"],
        color_lengthscale=config["model"]["color_lengthscale"],
        learn_inducing_locations=config["model"]["learn_inducing_locations"],
    ).to(device)

    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
    ).to(device)
    likelihood.noise = config["likelihood"]["noise_variance"]
    likelihood.noise_covar.raw_noise.requires_grad_(False)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["encoding_lr"])
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(samples))

    # ── Encoding ──
    model.train()
    likelihood.train()
    for _ in range(FLEX_ENCODING_EPOCHS):
        optimizer.zero_grad()
        loss = -mll(model(samples), weights)
        loss.backward()
        optimizer.step()

    # ── Maintenance with two sequential cues (previous-step rehearsal) ──
    maint_lr = config["training"]["maintenance_lr"]
    for pg in optimizer.param_groups:
        pg["lr"] = maint_lr

    attn_module = SpatialProximityAttention(
        spatial_std=config["attention"]["spatial_std"],
        attended_gain=config["attention"]["attended_gain"],
    ).to(device)

    beta = config["training"]["beta"]
    snapshot_post_A = None

    with torch.no_grad():
        Z0 = model.variational_strategy.inducing_points
        prev_target = likelihood(model(Z0)).mean.detach()

    for epoch in range(FLEX_MAINTENANCE_EPOCHS):
        optimizer.zero_grad()
        Z = model.variational_strategy.inducing_points
        output = model(Z)

        var_dist = model.variational_strategy.variational_distribution
        prior_dist = model.variational_strategy.prior_distribution
        kl_div = torch.distributions.kl.kl_divergence(var_dist, prior_dist)
        exp_ll = likelihood.expected_log_prob(prev_target, output)

        M = Z.shape[0]
        if epoch < CUE_A_START:
            attn_w = torch.ones(M, device=device)
        elif epoch < CUE_B_START:
            attn_w = attn_module(Z[:, 0].detach(), items[cued_idx_A][0])
        else:
            attn_w = attn_module(Z[:, 0].detach(), items[cued_idx_B][0])

        weighted_ll = (exp_ll * attn_w).sum() / M
        loss = -weighted_ll + kl_div * beta
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            Z = model.variational_strategy.inducing_points
            prev_target = likelihood(model(Z)).mean.detach()
            model.train()

        # Snapshot at end of cue-A phase (just before cue-B starts)
        if epoch == CUE_B_START - 1:
            model.eval()
            snapshot_post_A = _abs_errors(model, items, config)
            model.train()

    # Snapshot at end of cue-B phase (final epoch)
    model.eval()
    snapshot_post_B = _abs_errors(model, items, config)

    return snapshot_post_A, snapshot_post_B


def _subject_flexibility(config: dict):
    """One subject's cue-redirect run: mean absolute error for A, B and the
    other items at the end of each cue window."""
    cfg = copy.deepcopy(config)
    cfg["training"]["maintenance_epochs"] = FLEX_MAINTENANCE_EPOCHS
    cfg["training"]["encoding_epochs"] = FLEX_ENCODING_EPOCHS
    cfg["training"]["cue_start_epoch"] = CUE_A_START

    keys = (
        "A_post_cueA", "B_post_cueA", "other_post_cueA",
        "A_post_cueB", "B_post_cueB", "other_post_cueB",
    )
    acc = {k: [] for k in keys}
    n_trials = cfg["experiment"]["n_trials"]
    for t in trange(n_trials, desc="  Flexibility", leave=False):
        seed = cfg["experiment"]["random_seed"] + t
        items = generator.generate_items(FLEX_SET_SIZE, seed=seed)
        rng = np.random.RandomState(seed)
        idx_A, idx_B = rng.choice(FLEX_SET_SIZE, size=2, replace=False)
        snap_A, snap_B = run_flexibility_trial(
            items, cfg, cued_idx_A=int(idx_A), cued_idx_B=int(idx_B),
        )
        others = [i for i in range(FLEX_SET_SIZE) if i not in (idx_A, idx_B)]
        acc["A_post_cueA"].append(snap_A[idx_A])
        acc["B_post_cueA"].append(snap_A[idx_B])
        acc["other_post_cueA"].append(float(np.mean([snap_A[i] for i in others])))
        acc["A_post_cueB"].append(snap_B[idx_A])
        acc["B_post_cueB"].append(snap_B[idx_B])
        acc["other_post_cueB"].append(float(np.mean([snap_B[i] for i in others])))

    return {k: float(np.mean(v)) for k, v in acc.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Group summaries
# ──────────────────────────────────────────────────────────────────────────────

def _retrocue_csv(set_size):
    return OUT_DIR / f"retrocue_subjects_N{int(set_size)}.csv"


def finalize_retrocue(subject_records, target_set_size):
    trial_rows = [
        dict(subject=s, trial=t, condition=cond, signed_error=e)
        for s, rec in enumerate(subject_records)
        for cond in ("neutral", "cued")
        for t, e in enumerate(rec[f"{cond}_signed"])
    ]
    trials_path = OUT_DIR / f"retrocue_trials_N{int(target_set_size)}.csv"
    pd.DataFrame(trial_rows).to_csv(trials_path, index=False)
    print(f"  Saved → {trials_path}")

    df = pd.DataFrame([{k: rec[k] for k in ("mean_neutral", "mean_cued")}
                       for rec in subject_records])
    df.insert(0, "Subject", range(len(subject_records)))
    df["Benefit"] = df["mean_neutral"] - df["mean_cued"]
    path = _retrocue_csv(target_set_size)
    df.to_csv(path, index=False)
    t_stat, p_val = stats.ttest_rel(df["mean_neutral"], df["mean_cued"])
    print(f"\nN={target_set_size}: neutral {df.mean_neutral.mean():.2f}  "
          f"cued {df.mean_cued.mean():.2f}  benefit {df.Benefit.mean():.2f}  "
          f"paired t({len(df) - 1})={t_stat:.2f}, p={p_val:.2e}")
    print(f"  Saved → {path}")
    return df


def finalize_flexibility(subject_records):
    """Subject means for the cue redirect, plus a paired test of the trade-off."""
    df = pd.DataFrame(subject_records)
    df.insert(0, "Subject", range(len(subject_records)))
    # Error change across the redirect: abandoned item A, newly cued item B.
    df["delta_A"] = df["A_post_cueB"] - df["A_post_cueA"]
    df["delta_B"] = df["B_post_cueB"] - df["B_post_cueA"]
    df["tradeoff"] = df["delta_A"] - df["delta_B"]
    csv_path = OUT_DIR / "cue_flexibility_subjects.csv"
    df.to_csv(csv_path, index=False)

    t_trade, p_trade = stats.ttest_rel(
        df["delta_A"].to_numpy(), df["delta_B"].to_numpy()
    )
    summary = {
        "n_subjects": int(len(df)),
        "A_post_cueA": float(df["A_post_cueA"].mean()),
        "A_post_cueA_sem": float(stats.sem(df["A_post_cueA"])),
        "B_post_cueA": float(df["B_post_cueA"].mean()),
        "B_post_cueA_sem": float(stats.sem(df["B_post_cueA"])),
        "A_post_cueB": float(df["A_post_cueB"].mean()),
        "A_post_cueB_sem": float(stats.sem(df["A_post_cueB"])),
        "B_post_cueB": float(df["B_post_cueB"].mean()),
        "B_post_cueB_sem": float(stats.sem(df["B_post_cueB"])),
        "delta_A": float(df["delta_A"].mean()),
        "delta_B": float(df["delta_B"].mean()),
        "tradeoff": float(df["tradeoff"].mean()),
        "tradeoff_sem": float(stats.sem(df["tradeoff"])),
        "t_tradeoff": float(t_trade),
        "p_tradeoff": float(p_trade),
        "df": int(len(df) - 1),
    }
    summary_path = OUT_DIR / "cue_flexibility_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print("\nCue-redirect group summary (means across subjects):")
    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"  Saved → {csv_path}")
    print(f"  Saved → {summary_path}")
    return df


def aggregate_set_sizes():
    """Per-set-size summary and the one-tailed test benefit(N=4) > benefit(N=2)."""
    frames = {2: pd.read_csv(_retrocue_csv(2)), 4: pd.read_csv(_retrocue_csv(4))}
    rows = []
    ben = {}
    for ss, df in frames.items():
        b = (df["mean_neutral"] - df["mean_cued"]).to_numpy()
        ben[ss] = b
        t, p = stats.ttest_1samp(b, 0.0)
        rows.append(dict(
            set_size=float(ss),
            neutral_mae=float(df["mean_neutral"].mean()),
            neutral_sem=float(stats.sem(df["mean_neutral"])),
            neutral_sd=float(df["mean_neutral"].std(ddof=1)),
            cued_mae=float(df["mean_cued"].mean()),
            cued_sem=float(stats.sem(df["mean_cued"])),
            cued_sd=float(df["mean_cued"].std(ddof=1)),
            benefit_mean=float(b.mean()),
            benefit_sem=float(stats.sem(b)),
            benefit_sd=float(b.std(ddof=1)),
            t_stat=float(t),
            p_val=float(p),
        ))
        r = rows[-1]
        print(f"N={ss}: neutral={r['neutral_mae']:.2f}  cued={r['cued_mae']:.2f}  "
              f"benefit={r['benefit_mean']:.2f}±{r['benefit_sem']:.2f}  "
              f"t={r['t_stat']:.2f} p={r['p_val']:.2e}  (n={len(b)})")

    results_path = OUT_DIR / "retrocue_setsize_results.csv"
    pd.DataFrame(rows).to_csv(results_path, index=False)

    t_cross, p_cross = stats.ttest_ind(ben[4], ben[2], alternative="greater")
    comp = pd.DataFrame([{
        "comparison": "SS4_benefit > SS2_benefit",
        "benefit_N2": float(ben[2].mean()),
        "benefit_sem_N2": float(stats.sem(ben[2])),
        "benefit_N4": float(ben[4].mean()),
        "benefit_sem_N4": float(stats.sem(ben[4])),
        "t_stat": float(t_cross),
        "p_val": float(p_cross),
        "n_N2": int(len(ben[2])),
        "n_N4": int(len(ben[4])),
    }])
    comp_path = OUT_DIR / "retrocue_setsize_comparison.csv"
    comp.to_csv(comp_path, index=False)
    print(f"Load dependence (N4 > N2): t={t_cross:.2f}, p={p_cross:.4g} (one-tailed)")
    print(f"wrote {results_path}")
    print(f"wrote {comp_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Sharding (one subject per Slurm array task)
# ──────────────────────────────────────────────────────────────────────────────

def _shards_dir(experiment, set_size):
    tag = f"retrocue_N{set_size}" if experiment == "retrocue" else experiment
    path = OUT_DIR / "shards" / tag
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_subject(experiment, base_config, subject_idx, n_trials, set_size):
    cfg = _patch_config(base_config, n_trials, subject_idx)
    if experiment == "retrocue":
        return _subject_retrocue(cfg, target_set_size=set_size)
    return _subject_flexibility(cfg)


def write_shard(experiment, base_config, subject_idx, n_trials, set_size):
    record = run_subject(experiment, base_config, subject_idx, n_trials, set_size)
    payload = {"experiment": experiment, "subject": int(subject_idx),
               "n_trials": int(n_trials), "set_size": int(set_size),
               "record": record}
    path = _shards_dir(experiment, set_size) / f"subject_{subject_idx:04d}.json"
    path.write_text(json.dumps(payload))
    print(f"  Saved shard → {path}", flush=True)


def merge_shards(experiment, set_size):
    files = sorted(glob.glob(str(_shards_dir(experiment, set_size) / "subject_*.json")))
    if not files:
        raise SystemExit(f"No shards under {_shards_dir(experiment, set_size)}")
    payloads = sorted((json.loads(open(f).read()) for f in files),
                      key=lambda p: p["subject"])
    records = [p["record"] for p in payloads]
    print(f"Merging {len(records)} subject shards for '{experiment}'", flush=True)
    if experiment == "retrocue":
        finalize_retrocue(records, set_size)
    else:
        finalize_flexibility(records)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=["retrocue", "flexibility", "aggregate"])
    ap.add_argument("--set_size", type=int, default=4,
                    help="Set size for 'retrocue' (2 or 4; default 4).")
    ap.add_argument("--n_subjects", type=int, default=20)
    ap.add_argument("--n_trials", type=int, default=100,
                    help="Trials per condition per subject (default 100).")
    ap.add_argument("--subject_idx", type=int, default=None,
                    help="Run only this subject and write a JSON shard.")
    ap.add_argument("--merge", action="store_true",
                    help="Combine JSON shards into the group CSVs.")
    args = ap.parse_args()

    if args.experiment == "aggregate":
        aggregate_set_sizes()
        return

    set_size = args.set_size if args.experiment == "retrocue" else FLEX_SET_SIZE
    base_config = load_config(filename="config_retrocue.yaml")

    if args.merge:
        merge_shards(args.experiment, set_size)
        return
    if args.subject_idx is not None:
        write_shard(args.experiment, base_config, args.subject_idx,
                    args.n_trials, set_size)
        return

    records = []
    for subj in range(args.n_subjects):
        print(f"\n── Subject {subj + 1}/{args.n_subjects} ──", flush=True)
        records.append(run_subject(args.experiment, base_config, subj,
                                   args.n_trials, set_size))
    if args.experiment == "retrocue":
        finalize_retrocue(records, set_size)
    else:
        finalize_flexibility(records)


if __name__ == "__main__":
    main()
