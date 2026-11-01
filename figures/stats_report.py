"""Recompute the statistics quoted in the Results text from results/.

Prints one block per figure and writes the same text to
figures/output/stats_report.txt. Values for Supplementary Table S2 and the
model-comparison RMSEs are printed by ``figures/figS_model_comparison.py``.

Run:  python -m figures.stats_report
"""
import io
import contextlib
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr

from figures import nature_style as ns
from figures.fig2_setsize import model_subject_stats
from figures.fig3_retrocue import model_dynamics
from figures.fig4_bias import (
    _human_precision_repulsion, _memgp_precision_repulsion, memgp_bias_measure,
)
from figures.fig5_circuit import diffusion_stats


def half_up(x, places=1, sign=False):
    """Round half away from zero. Bias percentages are counts out of 2,000
    responses, so they land exactly on .x5 and float formatting would round
    half of them down."""
    q = Decimal(str(round(float(x), 9))).quantize(Decimal(1).scaleb(-places),
                                                  rounding=ROUND_HALF_UP)
    return f"{q:+}" if sign else f"{q}"


def pearson_ci(x, y):
    """Pearson r with a Fisher-z 95% interval."""
    r = float(np.corrcoef(x, y)[0, 1])
    z, se = np.arctanh(r), 1.0 / np.sqrt(len(x) - 3)
    lo, hi = np.tanh([z - 1.96 * se, z + 1.96 * se])
    return r, float(lo), float(hi)


def fig3():
    res = pd.read_csv(ns.result_path("retrocue", "retrocue_setsize_results.csv"))
    comp = pd.read_csv(ns.result_path("retrocue", "retrocue_setsize_comparison.csv")).iloc[0]
    print(f"Figure 3 - retrocue ({int(comp.n_N4)} simulated subjects)")
    for _, r in res.iterrows():
        n = int(comp["n_N2"] if r.set_size == 2 else comp["n_N4"])
        print(f"  N={int(r.set_size)}: cued {r.cued_mae:.1f} deg, neutral {r.neutral_mae:.1f} deg, "
              f"benefit {r.benefit_mean:.1f} +/- {r.benefit_sem:.1f}; "
              f"paired t({n - 1}) = {r.t_stat:.2f}, p = {r.p_val:.1e}")
    dof = int(comp.n_N2 + comp.n_N4 - 2)
    print(f"  load: benefit difference {comp.benefit_N4 - comp.benefit_N2:.1f} deg; "
          f"t({dof}) = {comp.t_stat:.2f}, p = {comp.p_val:.1e} (one-tailed)")

    flex = pd.read_csv(ns.result_path("retrocue", "cue_flexibility_subjects.csv"))
    d = flex["tradeoff"].to_numpy()
    t, p = stats.ttest_rel(flex["delta_A"], flex["delta_B"])
    lo, hi = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    print(f"  redirect: delta A {flex.delta_A.mean():+.1f} deg, delta B {flex.delta_B.mean():+.1f} deg; "
          f"trade-off {d.mean():+.1f} deg, 95% CI [{lo:.1f}, {hi:.1f}], "
          f"paired t({len(d) - 1}) = {t:.2f}, p = {p:.1e}")

    dyn = model_dynamics()
    cue = int(dyn["cue_epoch"])
    for key, label in (("act", "strength (k)"), ("var", "uncertainty (l)")):
        c = np.asarray(dyn[f"cued_{key}"], float)   # subjects x epochs
        other = "neutral" if f"neutral_{key}" in dyn else "uncued"
        u = np.asarray(dyn[f"{other}_{key}"], float)
        dc, du = c[:, -1] - c[:, cue], u[:, -1] - u[:, cue]
        t, p = stats.ttest_rel(dc, du)
        print(f"  dynamics {label}: end-of-delay change from cue onset, cued vs {other}: "
              f"paired t({len(dc) - 1}) = {t:.2f}, p = {p:.2g}")


def fig4():
    s = pd.read_csv(ns.result_path("bias", "adaptive_bias_confirm_summary.csv"))
    s = s[s.readout == "normalized"].set_index(["family", "level"])
    print(f"Figure 4 - bias (% away - 50; {int(s.n_subjects.iloc[0])} simulated subjects)")
    for fam, lvl, label in (
        ("load_distance", "N2_D20", "N=2, 20 deg"),
        ("load_distance", "N2_D105", "N=2, 105 deg"),
        ("load_distance", "N6_D45", "N=6, 45 deg"),
        ("load_distance", "N6_D135", "N=6, 135 deg"),
        ("retrocue", "neutral", "N=2, 135 deg, neutral"),
        ("retrocue", "target", "N=2, 135 deg, target cue"),
        ("encoding_distance", "C10_D135", "sigma_c=10, 135 deg"),
        ("encoding_distance", "C60_D135", "sigma_c=60, 135 deg"),
    ):
        r = s.loc[(fam, lvl)]
        print(f"  {label:26s} {half_up(r.bias_pct, sign=True)} +/- {half_up(r.bias_sem)}")
    memgp = _memgp_precision_repulsion()
    measure, _ = memgp_bias_measure(memgp)
    for name, df, y in (("human (f), %", _human_precision_repulsion(), "bias_pct"),
                        ("MemGP (g), %", memgp, "bias_pct"),
                        ("MemGP (g), deg", memgp, "bias_deg")):
        r, lo, hi = pearson_ci(df["circ_sd"].to_numpy(), df[y].to_numpy())
        p = stats.pearsonr(df["circ_sd"], df[y]).pvalue
        print(f"  precision-repulsion {name}: Pearson r = {r:.2f}, 95% CI [{lo:.2f}, {hi:.2f}], "
              f"p = {p:.1e}, n = {len(df)}")
    print(f"  Fig. 4g plots: {measure}")


def fig2():
    df = pd.read_csv(ns.result_path("setsize", "setsize_main_trials.csv"))
    print(f"Figure 2 - set size ({df['subject'].nunique()} simulated subjects)")
    Ns, sd, sd_sem, prec, prec_sem = model_subject_stats()
    for N, a, b, c, d in zip(Ns, sd, sd_sem, prec, prec_sem):
        print(f"  N={N}: recall SD {a:.1f} +/- {b:.1f} deg; precision {c:.2f} +/- {d:.2f} rad^-1")
    slope, _ = np.polyfit(np.log(Ns), np.log(prec), 1)
    print(f"  power law on group-mean precision: exponent {slope:.2f}")


def fig5():
    data = ns.load_npz(ns.result_path("circuit", "rnn_equiv.npz"))
    print("Figure 5 - recurrent circuit (GP vs network on matched trials)")
    for key, label in (("mae", "mean |error| over 100 reports (Fig. 5c)"),
                       ("err1", "|error| of one report"),
                       ("mean", "posterior mean at the probed item"),
                       ("var", "posterior variance at the probed item")):
        x = np.concatenate([data[f"{c}_{key}_gp"] for c in ("neutral", "cued")])
        y = np.concatenate([data[f"{c}_{key}_net"] for c in ("neutral", "cued")])
        r, lo, hi = pearson_ci(x, y)
        print(f"  {label}: Pearson r = {r:.3f}, 95% CI [{lo:.3f}, {hi:.3f}], n = {len(x)}")
    rel = np.concatenate([data[f"{c}_reldiff"] for c in ("neutral", "cued")])
    cov = np.concatenate([data[f"{c}_covdiff"] for c in ("neutral", "cued")])
    print(f"  relative difference GP vs network: m {rel.mean():.1e}, S {cov.mean():.1e}")

    diff = ns.load_npz(ns.result_path("circuit", "bump_diffusion.npz"))
    st = diffusion_stats(diff)
    report = np.asarray(diff["report_err"], float)
    print(f"  drift (5d): Spearman rho = {st['rho']:.2f}, 95% CI [{st['lo']:.2f}, {st['hi']:.2f}]; "
          f"mean displacement {st['mean'][0]:.2f} -> {st['mean'][-1]:.2f} deg; "
          f"{st['disp'].shape[0]} trials; report circular SD {ns.circ_sd_deg(report):.1f} deg")

    unc = ns.load_npz(ns.result_path("circuit", "decoded_uncertainty.npz"))
    errs = np.abs(np.asarray(unc["errs"], float))
    width = np.asarray(unc["uncert"], float)
    rng = np.random.default_rng(5)
    boot = [spearmanr(errs[i], width[i]).statistic
            for i in (rng.integers(0, errs.size, errs.size) for _ in range(1000))]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  uncertainty (5e): Spearman rho = {float(unc['rho']):.2f}, 95% CI [{lo:.2f}, {hi:.2f}], "
          f"p = {float(unc['p']):.1e}; {int(unc['n_trials'])} trials at N={int(unc['set_size'])}")


def main():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        for block in (fig2, fig3, fig4, fig5):
            block()
            print()
    text = buf.getvalue()
    print(text, end="")
    path = ns.out_path("stats_report") + ".txt"
    with open(path, "w") as f:
        f.write(text)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
