"""Unified adaptive attraction/repulsion experiments for manuscript Figure 4.

The primary outcome follows Chunharas et al. (2022):

    bias_pct = 100 * n(responses away from distractor) / n(total responses) - 50

Positive values indicate repulsion and negative values indicate attraction.

One trial is one fresh display, one full encoding+maintenance fit, and one
sampled report, matching the trial semantics of the set-size and retrocue
experiments in ``validation_subjects.py``. Each subject contributes
``TRIALS_PER_SUBJECT`` trials per condition.

Every sampled profile is read out both plain and with the fixed
divisive-normalization rule. The figure uses the normalized readout throughout;
the plain column costs no extra fit and keeps the normalization ablation
available as a paired within-trial comparison.

All files live in ``results/bias/``. The manuscript figure uses only the
confirmatory run (``adaptive_bias_confirm_trials.csv`` and its summary), whose
conditions are fixed by ``adaptive_bias_locked_design.json``. The pilot files
and the lock step that chose those levels are kept for the record.

Reproduce Figure 4 b-e from the shipped locked design:
    python -m experiments.bias_adaptive confirm --figure_only --workers 8

Same run as one subject per Slurm array task, then merge:
    python -m experiments.bias_adaptive confirm_shard --figure_only --subject_idx 0
    python -m experiments.bias_adaptive confirm_merge

How the locked design was chosen (pilot data are independent of the confirm run):
    python -m experiments.bias_adaptive pilot --workers 4
    python -m experiments.bias_adaptive lock
    python -m experiments.bias_adaptive encoding_pilot --workers 4
    python -m experiments.bias_adaptive encoding_lock
The precise-encoding level was then set by hand to color_std = 10 (see
``encoding_selection_note`` in the JSON).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from memgp.generator import circular_error
from memgp.paths import results_dir
from memgp.simulation import device, load_config, normalize_slices, run_simulation_trial


OUT_DIR = results_dir("bias")
DATASET_STEM = "adaptive_bias"
PILOT_CSV = OUT_DIR / f"{DATASET_STEM}_pilot_trials.csv"
CONFIRM_CSV = OUT_DIR / f"{DATASET_STEM}_confirm_trials.csv"
ENCODING_PILOT_CSV = OUT_DIR / f"{DATASET_STEM}_encoding_pilot_trials.csv"
SHARD_DIR = OUT_DIR / f"{DATASET_STEM}_confirm_shards"
LOCK_JSON = OUT_DIR / f"{DATASET_STEM}_locked_design.json"
N_COLOR = 360
TARGET_LOC = 0.0
# One report per fitted display, so a trial here means the same thing it means
# in the set-size and retrocue experiments.
N_DRAWS = 1
TRIALS_PER_SUBJECT = 100
PILOT_TRIALS_PER_SUBJECT = 24
# Chunharas Exp. 4 grid: encoding precision × feature distance (N=2).
# Encoding duration is operationalized as observation color noise (color_std):
# lower std = more precise / longer encoding; epochs stay high enough to
# individuate two peaks (cutting epochs alone blends items into one bump).
ENCODING_DISTANCES = (20.0, 45.0, 75.0, 105.0, 135.0)
ENCODING_COLOR_STDS = (10.0, 20.0, 40.0, 60.0)
ENCODING_FIT_EPOCHS = 100
# Figure 4 panels b–e families (full-scale cluster confirm).
FIGURE_FAMILIES = ("load_distance", "retrocue", "encoding_distance")


@dataclass(frozen=True)
class Condition:
    family: str
    level: str
    set_size: int = 2
    color_distance: float = 75.0
    encoding_epochs: int = 100
    maintenance_epochs: int = 100
    beta: float = 2.0
    cued: bool = False
    color_std: float = 20.0


def _wrap(x: float | np.ndarray) -> float | np.ndarray:
    return (np.asarray(x) + 180.0) % 360.0 - 180.0


def _circ_dist(a: np.ndarray, b: float) -> np.ndarray:
    d = np.abs(np.asarray(a) - b)
    return np.minimum(d, 360.0 - d)


def _make_items(
    condition: Condition, seed: int
) -> tuple[list[tuple[float, float]], float, int]:
    """Create a target-first display and return items, target color, direction."""
    rng = np.random.RandomState(seed)
    target_color = float(rng.uniform(-180.0, 180.0))
    direction = int(rng.choice([-1, 1]))

    if condition.set_size == 2:
        distractor_color = float(
            _wrap(target_color + direction * condition.color_distance)
        )
        return (
            [(TARGET_LOC, target_color), (90.0, distractor_color)],
            target_color,
            direction,
        )

    if condition.set_size != 6:
        raise ValueError(f"Unsupported set size: {condition.set_size}")

    # All distractors lie on the same side of the target in color space. A
    # narrow deterministic spread avoids five exact color duplicates while
    # preserving a well-defined ensemble direction and center distance.
    spread = min(8.0, max(2.0, condition.color_distance * 0.2))
    offsets = condition.color_distance + np.linspace(-spread, spread, 5)
    offsets = np.maximum(offsets, 1.0)
    locs = [60.0, 120.0, -180.0, -120.0, -60.0]
    distractors = [
        (loc, float(_wrap(target_color + direction * offset)))
        for loc, offset in zip(locs, offsets)
    ]
    return [(TARGET_LOC, target_color), *distractors], target_color, direction


def _base_config() -> dict:
    cfg = load_config(filename="config_bias.yaml")
    retro = load_config(filename="config_retrocue.yaml")
    cfg["attention"] = copy.deepcopy(retro["attention"])
    return cfg


def _sample_item_profiles(
    model, item_locs: list[float], seed: int, n_draws: int = N_DRAWS
) -> np.ndarray:
    colors = torch.linspace(-180.0, 180.0, N_COLOR, device=device)
    rows = [
        torch.stack([torch.full_like(colors, float(loc)), colors], dim=-1)
        for loc in item_locs
    ]
    grid = torch.cat(rows, dim=0)
    model.eval()
    torch.manual_seed(seed)
    with torch.no_grad():
        pred = model(grid)
        fields = pred.sample(torch.Size([n_draws]))
    return fields.view(n_draws, len(item_locs), N_COLOR).cpu().numpy()


def _circular_smooth_batch(y: np.ndarray, width_deg: float) -> np.ndarray:
    spacing = 360.0 / y.shape[-1]
    half = min(int(3.0 * width_deg / spacing), y.shape[-1] // 2)
    if half < 1:
        return y.copy()
    x = np.arange(-half, half + 1, dtype=float) * spacing
    kernel = np.exp(-0.5 * (x / width_deg) ** 2)
    kernel /= kernel.sum()
    padded = np.concatenate([y[..., -half:], y, y[..., :half]], axis=-1)
    return np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="valid"), -1, padded
    )


def _read_profiles(
    profiles: np.ndarray,
    item_locs: list[float],
    norm: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Return plain and normalized recalled colors from identical samples."""
    colors = np.linspace(-180.0, 180.0, N_COLOR)
    target = profiles[:, 0, :]
    plain = colors[np.argmax(target, axis=1)]
    response = normalize_slices(
        torch.as_tensor(profiles), [float(l) for l in item_locs], TARGET_LOC,
        float(norm["loc_pool_width"]), float(norm.get("baseline_frac", 0.5)),
    )
    normalized = colors[torch.argmax(response, dim=-1).numpy()]
    return plain, normalized


def _trial_seed(phase: str, family: str, subject: int, trial: int) -> int:
    """Deterministic seed for one trial.

    The manipulated level must not enter the seed, so every level of a family
    shares the display, the model initialization, and the sampling stream on a
    given (subject, trial). Additive index slots would collide once the trial
    count reaches three digits, so the indices go through a stable digest
    instead of a hand-tuned stride.
    """
    key = f"{phase}|{family}|{int(subject)}|{int(trial)}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big") % (
        2**31 - 1
    )


def _run_fit(task: tuple[Condition, int, int, str]) -> list[dict]:
    condition, subject, trial, phase = task
    torch.set_num_threads(1)
    seed = _trial_seed(phase, condition.family, subject, trial)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    cfg = _base_config()
    cfg["training"]["encoding_epochs"] = int(condition.encoding_epochs)
    cfg["training"]["maintenance_epochs"] = int(condition.maintenance_epochs)
    cfg["training"]["beta"] = float(condition.beta)
    cfg["data"]["color_std"] = float(condition.color_std)
    items, target_color, direction = _make_items(condition, seed)
    cued_idx = 0 if condition.cued else None
    model, _, _ = run_simulation_trial(
        items,
        cfg,
        encoding_epochs=condition.encoding_epochs,
        maintenance_epochs=condition.maintenance_epochs,
        cued_item_idx=cued_idx,
        track_visuals=False,
    )
    item_locs = [loc for loc, _ in items]
    profiles = _sample_item_profiles(model, item_locs, seed + 71_113)
    plain, normalized = _read_profiles(profiles, item_locs, cfg["normalization"])

    rows: list[dict] = []
    for readout, reports in (("plain", plain), ("normalized", normalized)):
        for draw, report in enumerate(reports):
            raw_error = float(circular_error(float(report), target_color))
            oriented_error = float(-direction * raw_error)
            rows.append(
                {
                    "phase": phase,
                    "family": condition.family,
                    "level": condition.level,
                    "set_size": condition.set_size,
                    "color_distance": condition.color_distance,
                    "encoding_epochs": condition.encoding_epochs,
                    "maintenance_epochs": condition.maintenance_epochs,
                    "beta": condition.beta,
                    "color_std": condition.color_std,
                    "cue_condition": "target" if condition.cued else "neutral",
                    "subject": subject,
                    "trial": trial,
                    "draw": draw,
                    "seed": seed,
                    "readout": readout,
                    "target_color": target_color,
                    "direction": direction,
                    "reported_color": float(report),
                    "raw_error_deg": raw_error,
                    "oriented_error_deg": oriented_error,
                    "away": int(oriented_error > 0),
                    "exact_zero": int(oriented_error == 0),
                }
            )
    return rows


def _pilot_conditions() -> list[Condition]:
    out: list[Condition] = []
    for n in (2, 6):
        for distance in (20.0, 45.0, 75.0, 105.0, 135.0):
            out.append(
                Condition(
                    "load_distance",
                    f"N{n}_D{int(distance)}",
                    set_size=n,
                    color_distance=distance,
                )
            )
    for epochs in (25, 100, 200):
        out.append(
            Condition(
                "maintenance",
                f"M{epochs}",
                maintenance_epochs=epochs,
            )
        )
    # Fixed-distance encoding probes remain for legacy diagnostics; the
    # Exp. 4 interaction lives in the encoding_distance family.
    for epochs in (40, 100, 200):
        out.append(
            Condition("encoding", f"E{epochs}", encoding_epochs=epochs)
        )
    out.extend(_encoding_distance_conditions())
    out.extend(
        [
            Condition("retrocue", "neutral", cued=False),
            Condition("retrocue", "target", cued=True),
        ]
    )
    for beta in (0.5, 2.0, 8.0):
        out.append(Condition("beta", f"B{beta:g}", beta=beta))
    return out


def _encoding_distance_conditions(
    distances: Iterable[float] = ENCODING_DISTANCES,
    color_stds: Iterable[float] = ENCODING_COLOR_STDS,
) -> list[Condition]:
    """N=2 color-distance × encoding-precision grid (Chunharas Exp. 4).

    ``color_std`` is the encoding-duration proxy: larger values yield noisier,
    less distinctive item peaks while fit epochs remain fixed so items stay
    individuated rather than collapsing into a single under-fit bump.
    """
    out: list[Condition] = []
    for std in color_stds:
        for distance in distances:
            out.append(
                Condition(
                    "encoding_distance",
                    f"C{int(std)}_D{int(distance)}",
                    set_size=2,
                    color_distance=float(distance),
                    encoding_epochs=ENCODING_FIT_EPOCHS,
                    color_std=float(std),
                )
            )
    return out


def _peak_distance(series: pd.Series) -> float:
    """Color distance of maximal repulsion (ties → smaller distance)."""
    best = float(series.max())
    candidates = [float(d) for d, v in series.items() if float(v) == best]
    return min(candidates)


def _bias_pct(values: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=float)
    return float(100.0 * np.mean(x > 0) - 50.0)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "color_std" not in df.columns:
        df["color_std"] = 20.0
    else:
        df["color_std"] = df["color_std"].fillna(20.0)
    keys = [
        "phase",
        "family",
        "level",
        "set_size",
        "color_distance",
        "encoding_epochs",
        "maintenance_epochs",
        "beta",
        "color_std",
        "cue_condition",
        "readout",
        "subject",
    ]
    subject = (
        df.groupby(keys, observed=True)["oriented_error_deg"]
        .agg(
            bias_pct=_bias_pct,
            mean_bias_deg="mean",
            median_bias_deg="median",
            exact_zero_pct=lambda x: 100.0 * np.mean(np.asarray(x) == 0),
            n_responses="size",
        )
        .reset_index()
    )
    group_keys = [k for k in keys if k != "subject"]
    summary = (
        subject.groupby(group_keys, observed=True)
        .agg(
            bias_pct=("bias_pct", "mean"),
            bias_sem=("bias_pct", lambda x: float(pd.Series(x).sem())),
            bias_sd=("bias_pct", lambda x: float(pd.Series(x).std(ddof=1))),
            mean_bias_deg=("mean_bias_deg", "mean"),
            exact_zero_pct=("exact_zero_pct", "mean"),
            n_subjects=("subject", "nunique"),
            n_responses=("n_responses", "sum"),
        )
        .reset_index()
    )
    return summary


def _run_conditions(
    conditions: list[Condition],
    phase: str,
    n_subjects: int,
    trials: int,
    workers: int,
    subjects: list[int] | None = None,
) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    subject_ids = list(subjects) if subjects is not None else list(range(n_subjects))
    tasks = [
        (condition, subject, trial, phase)
        for condition in conditions
        for subject in subject_ids
        for trial in range(trials)
    ]
    rows: list[dict] = []
    if workers <= 1:
        for i, task in enumerate(tasks, 1):
            rows.extend(_run_fit(task))
            print(f"[{phase}] fit {i}/{len(tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_fit, task): task for task in tasks}
            for i, future in enumerate(as_completed(futures), 1):
                rows.extend(future.result())
                print(f"[{phase}] fit {i}/{len(tasks)}", flush=True)
    return pd.DataFrame(rows)


def _confirm_trials(design: dict, override: int | None) -> int:
    return int(override or design.get("confirm_trials_per_subject", TRIALS_PER_SUBJECT))


def _figure_confirm_conditions(design: dict) -> list[Condition]:
    """Conditions for Figure 4 panels b–e only."""
    return [
        c
        for c in _confirm_conditions(design)
        if c.family in FIGURE_FAMILIES
    ]


def _best_ordered_pair(
    values: pd.Series, favor_lower: bool
) -> tuple[float, float]:
    distances = sorted(float(x) for x in values.index)
    scored: list[tuple[float, float, float]] = []
    for low in distances:
        for high in distances:
            if low >= high:
                continue
            contrast = float(values.loc[low] - values.loc[high])
            score = contrast if favor_lower else -contrast
            if score > 0:
                scored.append((score, low, high))
    if not scored:
        for low in distances:
            for high in distances:
                if low < high:
                    contrast = float(values.loc[low] - values.loc[high])
                    score = contrast if favor_lower else -contrast
                    scored.append((score, low, high))
    _, low, high = max(scored)
    return low, high


def _choose_locked_design(pilot: pd.DataFrame) -> dict:
    summary = summarize(pilot)
    norm = summary[
        (summary["family"] == "load_distance")
        & (summary["readout"] == "normalized")
    ]
    pivot = norm.pivot(index="color_distance", columns="set_size", values="bias_pct")
    low_load_distance_low, low_load_distance_high = _best_ordered_pair(
        pivot[2], favor_lower=True
    )
    high_load_distance_low, high_load_distance_high = _best_ordered_pair(
        pivot[6], favor_lower=False
    )

    low_load = norm[norm["set_size"] == 2]
    fixed_distance = float(
        low_load.loc[low_load["bias_pct"].idxmax(), "color_distance"]
    )
    design = {
        "selection_source": PILOT_CSV.name,
        "selection_readout": "normalized",
        "low_load_distance_levels": [
            low_load_distance_low,
            low_load_distance_high,
        ],
        "high_load_distance_levels": [
            high_load_distance_low,
            high_load_distance_high,
        ],
        "fixed_distance": fixed_distance,
        "maintenance_levels": [25, 200],
        "encoding_levels": [40, 200],
        "beta_levels": [0.5, 8.0],
        "confirm_subjects": 20,
        "confirm_trials_per_subject": TRIALS_PER_SUBJECT,
        "posterior_draws_per_trial": N_DRAWS,
        "primary_metric": "100 * responses_away / total_responses - 50",
        "primary_readout": "normalized",
        "ablation_readout": "plain",
    }
    design.update(_choose_encoding_interaction(pilot))
    return design


def _choose_encoding_interaction(pilot: pd.DataFrame) -> dict:
    """Lock Exp. 4-style encoding × distance levels from a pilot grid.

    Target signature (Chunharas et al., 2022, Exp. 4): longer / more precise
    encoding shifts the repulsion peak toward smaller color distances. Encoding
    precision is the observation ``color_std`` (larger = noisier / shorter).
    """
    summary = summarize(pilot)
    enc = summary[
        (summary["family"] == "encoding_distance")
        & (summary["readout"] == "normalized")
    ]
    if enc.empty:
        return {
            "encoding_distance_color_stds": [60.0, 10.0],
            "encoding_distance_levels": [75.0, 135.0],
            "encoding_peak_by_color_std": {},
            "encoding_interaction_ok": False,
            "encoding_selection_note": "no encoding_distance pilot rows",
        }

    pivot = enc.pivot(
        index="color_distance", columns="color_std", values="bias_pct"
    )
    peaks = {float(s): _peak_distance(pivot[s]) for s in pivot.columns}
    std_list = sorted(float(s) for s in pivot.columns)
    # short = noisier (higher std), long = more precise (lower std)
    scored: list[tuple[float, float, float]] = []
    for i, precise in enumerate(std_list):
        for noisy in std_list[i + 1 :]:
            shift = peaks[noisy] - peaks[precise]  # >0 if noisier peaks farther
            scored.append((shift, noisy, precise))
    scored.sort(reverse=True)
    best_shift, short_std, long_std = scored[0]
    near_d = peaks[long_std]
    far_d = peaks[short_std]
    if near_d == far_d:
        diffs = (pivot[short_std] - pivot[long_std]).dropna()
        if not diffs.empty:
            far_d = float(diffs.idxmax())
            near_d = float(diffs.idxmin())
            if near_d > far_d:
                near_d, far_d = far_d, near_d
        if near_d == far_d:
            dists = sorted(float(d) for d in pivot.index)
            near_d, far_d = dists[0], dists[-1]

    interaction_ok = bool(best_shift > 0)
    note = (
        f"peak_shift_noisy_minus_precise={best_shift:g}; "
        f"peaks={peaks}; short_std={short_std:g}, long_std={long_std:g}; "
        f"near={near_d:g}, far={far_d:g}"
    )
    return {
        "encoding_distance_color_stds": [short_std, long_std],
        "encoding_distance_levels": [near_d, far_d],
        "encoding_peak_by_color_std": {str(k): v for k, v in peaks.items()},
        "encoding_interaction_ok": interaction_ok,
        "encoding_selection_note": note,
        "encoding_operationalization": (
            "color_std observation noise with fixed fit epochs; "
            "higher color_std = shorter/noisier encoding"
        ),
    }


def _confirm_conditions(design: dict) -> list[Condition]:
    low2, high2 = (float(x) for x in design["low_load_distance_levels"])
    low6, high6 = (float(x) for x in design["high_load_distance_levels"])
    fixed = float(design["fixed_distance"])
    # Full 5-level distance curves for both loads (not just locked pairs),
    # so panels show the complete bias–distance profile.
    distance_grid = (20.0, 45.0, 75.0, 105.0, 135.0)
    out = [
        Condition("load_distance", f"N2_D{int(d)}", 2, d)
        for d in distance_grid
    ]
    out.extend(
        Condition("load_distance", f"N6_D{int(d)}", 6, d)
        for d in distance_grid
    )
    # Keep locked extremes recorded for highlighting / hist overlays.
    design.setdefault("low_load_distance_levels", [low2, high2])
    design.setdefault("high_load_distance_levels", [low6, high6])
    out.extend(
        Condition("maintenance", f"M{m}", color_distance=fixed, maintenance_epochs=int(m))
        for m in design["maintenance_levels"]
    )
    # Legacy single-distance epoch probes (diagnostics only).
    out.extend(
        Condition("encoding", f"E{e}", color_distance=fixed, encoding_epochs=int(e))
        for e in design.get("encoding_levels", [40, 200])
    )
    # Exp. 4 confirmatory cells: locked color_std × locked distances.
    enc_stds = [
        float(s)
        for s in design.get("encoding_distance_color_stds", [60.0, 10.0])
    ]
    enc_dists = [float(d) for d in design.get("encoding_distance_levels", [fixed])]
    out.extend(_encoding_distance_conditions(distances=enc_dists, color_stds=enc_stds))
    out.extend(
        [
            Condition("retrocue", "neutral", color_distance=fixed, cued=False),
            Condition("retrocue", "target", color_distance=fixed, cued=True),
        ]
    )
    out.extend(
        Condition("beta", f"B{b:g}", color_distance=fixed, beta=float(b))
        for b in design["beta_levels"]
    )
    return out


def _write_outputs(df: pd.DataFrame, path: Path) -> None:
    df = df.sort_values(
        ["family", "level", "subject", "trial", "draw", "readout"]
    ).reset_index(drop=True)
    df.to_csv(path, index=False)
    summary_path = path.with_name(path.stem.replace("_trials", "_summary") + ".csv")
    summarize(df).to_csv(summary_path, index=False)
    npz_path = path.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        oriented_error_deg=df["oriented_error_deg"].to_numpy(np.float32),
        away=df["away"].to_numpy(np.int8),
        exact_zero=df["exact_zero"].to_numpy(np.int8),
        subject=df["subject"].to_numpy(np.int16),
        trial=df["trial"].to_numpy(np.int16),
        draw=df["draw"].to_numpy(np.int16),
        family=df["family"].to_numpy(str),
        level=df["level"].to_numpy(str),
        readout=df["readout"].to_numpy(str),
    )
    print(f"wrote {path}", flush=True)
    print(f"wrote {summary_path}", flush=True)
    print(f"wrote {npz_path}", flush=True)


def _normative_check(pilot: pd.DataFrame, confirm: pd.DataFrame) -> pd.DataFrame:
    """Cross-validate a constant signed shift; negative result is informative."""
    rows = []
    for readout in ("plain", "normalized"):
        train = pilot[pilot["readout"] == readout]["oriented_error_deg"].to_numpy()
        test = confirm[confirm["readout"] == readout]["oriented_error_deg"].to_numpy()
        shift = -float(np.median(train))
        mae_base = float(np.mean(np.abs(test)))
        mae_shift = float(np.mean(np.abs(test + shift)))
        rows.append(
            {
                "readout": readout,
                "pilot_optimal_shift_deg": shift,
                "confirm_mae_unshifted": mae_base,
                "confirm_mae_shifted": mae_shift,
                "mae_change_shifted_minus_unshifted": mae_shift - mae_base,
                "supports_error_reduction": bool(mae_shift < mae_base),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / f"{DATASET_STEM}_normative_check.csv", index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=(
            "pilot",
            "lock",
            "confirm",
            "extend",
            "summarize",
            "encoding_pilot",
            "encoding_lock",
            "confirm_shard",
            "confirm_merge",
        ),
    )
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--subjects", type=int)
    parser.add_argument(
        "--n_trials",
        type=int,
        help="trials (fresh display + fit + report) per condition per subject",
    )
    parser.add_argument("--subject_idx", type=int, default=None)
    parser.add_argument(
        "--figure_only",
        action="store_true",
        help="Restrict confirm to Figure 4 panel b–e families",
    )
    args = parser.parse_args()

    n_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "0") or 0)
    if n_threads > 0:
        torch.set_num_threads(n_threads)

    if args.mode == "pilot":
        df = _run_conditions(
            _pilot_conditions(),
            "pilot",
            n_subjects=args.subjects or 2,
            trials=args.n_trials or PILOT_TRIALS_PER_SUBJECT,
            workers=args.workers,
        )
        _write_outputs(df, PILOT_CSV)
        return

    if args.mode == "encoding_pilot":
        df = _run_conditions(
            _encoding_distance_conditions(),
            "pilot",
            n_subjects=args.subjects or 2,
            trials=args.n_trials or PILOT_TRIALS_PER_SUBJECT,
            workers=args.workers,
        )
        _write_outputs(df, ENCODING_PILOT_CSV)
        # Merge into the main pilot table so lock/summarize see the grid.
        if PILOT_CSV.exists():
            base = pd.read_csv(PILOT_CSV)
            base = base[base["family"] != "encoding_distance"]
            merged = pd.concat([base, df], ignore_index=True)
            _write_outputs(merged, PILOT_CSV)
        return

    if args.mode == "lock":
        pilot = pd.read_csv(PILOT_CSV)
        design = _choose_locked_design(pilot)
        LOCK_JSON.write_text(json.dumps(design, indent=2) + "\n")
        print(json.dumps(design, indent=2))
        return

    if args.mode == "encoding_lock":
        if not LOCK_JSON.exists():
            raise SystemExit("locked design missing; run lock first")
        design = json.loads(LOCK_JSON.read_text())
        source = ENCODING_PILOT_CSV if ENCODING_PILOT_CSV.exists() else PILOT_CSV
        pilot = pd.read_csv(source)
        design.update(_choose_encoding_interaction(pilot))
        design["encoding_selection_source"] = source.name
        LOCK_JSON.write_text(json.dumps(design, indent=2) + "\n")
        print(json.dumps(design, indent=2))
        return

    if args.mode == "confirm_shard":
        if args.subject_idx is None:
            raise SystemExit("confirm_shard requires --subject_idx")
        design = json.loads(LOCK_JSON.read_text())
        design["confirm_subjects"] = int(
            args.subjects or design.get("confirm_subjects", 20)
        )
        conditions = (
            _figure_confirm_conditions(design)
            if args.figure_only
            else _confirm_conditions(design)
        )
        df = _run_conditions(
            conditions,
            "confirm",
            n_subjects=1,
            trials=_confirm_trials(design, args.n_trials),
            workers=args.workers,
            subjects=[int(args.subject_idx)],
        )
        SHARD_DIR.mkdir(parents=True, exist_ok=True)
        shard_path = SHARD_DIR / f"subject_{int(args.subject_idx):03d}.csv"
        df.to_csv(shard_path, index=False)
        print(f"wrote {shard_path} rows={len(df)}", flush=True)
        return

    if args.mode == "confirm_merge":
        shards = sorted(SHARD_DIR.glob("subject_*.csv"))
        if not shards:
            raise SystemExit(f"no shards in {SHARD_DIR}")
        frames = [pd.read_csv(p) for p in shards]
        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(
            ["family", "level", "subject", "trial", "draw", "readout"]
        )
        _write_outputs(df, CONFIRM_CSV)
        if PILOT_CSV.exists():
            print(_normative_check(pd.read_csv(PILOT_CSV), df).to_string(index=False))
        print(
            f"merged {len(shards)} shards -> {CONFIRM_CSV} "
            f"({df['subject'].nunique()} subjects)",
            flush=True,
        )
        return

    if args.mode == "confirm":
        design = json.loads(LOCK_JSON.read_text())
        conditions = (
            _figure_confirm_conditions(design)
            if args.figure_only
            else _confirm_conditions(design)
        )
        df = _run_conditions(
            conditions,
            "confirm",
            n_subjects=args.subjects or int(design["confirm_subjects"]),
            trials=_confirm_trials(design, args.n_trials),
            workers=args.workers,
        )
        _write_outputs(df, CONFIRM_CSV)
        if PILOT_CSV.exists():
            print(_normative_check(pd.read_csv(PILOT_CSV), df).to_string(index=False))
        return

    if args.mode == "extend":
        design = json.loads(LOCK_JSON.read_text())
        desired = _confirm_conditions(design)
        existing = pd.read_csv(CONFIRM_CSV)
        # Drop superseded encoding_distance cells so renamed levels do not linger.
        desired_enc = {
            (c.family, c.level) for c in desired if c.family == "encoding_distance"
        }
        if desired_enc:
            keep = ~(
                (existing["family"] == "encoding_distance")
                & ~existing.apply(
                    lambda r: (r["family"], r["level"]) in desired_enc, axis=1
                )
            )
            existing = existing.loc[keep].copy()
        present = set(zip(existing["family"], existing["level"]))
        missing = [c for c in desired if (c.family, c.level) not in present]
        if missing:
            addition = _run_conditions(
                missing,
                "confirm",
                n_subjects=args.subjects or int(design["confirm_subjects"]),
                trials=_confirm_trials(design, args.n_trials),
                workers=args.workers,
            )
            existing = pd.concat([existing, addition], ignore_index=True)
            existing = existing.drop_duplicates(
                ["family", "level", "subject", "trial", "draw", "readout"]
            )
            _write_outputs(existing, CONFIRM_CSV)
        else:
            _write_outputs(existing, CONFIRM_CSV)
        pilot = pd.read_csv(PILOT_CSV)
        print(_normative_check(pilot, existing).to_string(index=False))
        return

    for path in (PILOT_CSV, CONFIRM_CSV, ENCODING_PILOT_CSV):
        if path.exists():
            summary = summarize(pd.read_csv(path))
            print(f"\n{path.name}\n{summary.to_string(index=False)}")


if __name__ == "__main__":
    main()
