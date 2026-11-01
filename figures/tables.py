"""Supplementary Table S1: configuration parameters, read from memgp/configs/.

Writes figures/output/table_S1_parameters.tex (the body of the manuscript
table) so every value in the table can be traced to the YAML file that the
simulations load. Supplementary Table S2 is written by
``figures/figS_model_comparison.py`` because it needs the model fits.

Run:  python -m figures.tables
"""
import math

from figures import nature_style as ns
from memgp.simulation import load_config

COLUMNS = [("Set-size", "config_set_size.yaml"),
           ("Retrocue", "config_retrocue.yaml"),
           ("Bias", "config_bias.yaml")]

ROWS = [
    (r"$n_{\mathrm{obs}}$ per item", ("data", "n_samples_per_item"), "{:g}"),
    (r"$\sigma_\ell$ (loc sampling noise)", ("data", "loc_std"), "${:g}^\\circ$"),
    (r"$\sigma_c$ (color sampling noise)", ("data", "color_std"), "${:g}^\\circ$"),
    (r"$m$ (inducing grid side)", ("model", "inducing_grid_size"), "{:g}"),
    (r"$\lambda_\ell$ (loc lengthscale)", ("model", "loc_lengthscale"), "{:.1f}"),
    (r"$\lambda_c$ (color lengthscale)", ("model", "color_lengthscale"), "{:.1f}"),
    (r"$\sigma_n^2$ (likelihood noise)", ("likelihood", "noise_variance"), "pow10"),
    (r"$T_{\mathrm{enc}}$ (encoding epochs)", ("training", "encoding_epochs"), "{:g}"),
    (r"$T_{\mathrm{maint}}$ (maintenance epochs)", ("training", "maintenance_epochs"), "{:g}"),
    (r"$\eta_{\mathrm{enc}}$ (encoding LR)", ("training", "encoding_lr"), "{:g}"),
    (r"$\eta_{\mathrm{maint}}$ (maintenance LR)", ("training", "maintenance_lr"), "{:g}"),
    (r"$\beta$ (KL weight)", ("training", "beta"), "{:g}"),
    (r"$\sigma_s$ (attention spatial width)", ("attention", "spatial_std"), "${:g}^\\circ$"),
    (r"$G$ (attended gain)", ("attention", "attended_gain"), "{:.1f}"),
    (r"$\sigma_{\mathrm{pool}}$ (surround reach)", ("normalization", "loc_pool_width"), "${:g}^\\circ$"),
    (r"$\sigma_b$ (baseline fraction)", ("normalization", "baseline_frac"), "{:g}"),
]


def _cell(cfg, keys, fmt):
    section, key = keys
    value = (cfg.get(section) or {}).get(key)
    if value is None:
        return "--"
    if fmt == "pow10":
        return f"$10^{{{round(math.log10(value))}}}$"
    return fmt.format(value)


def main():
    configs = [load_config(filename=f) for _, f in COLUMNS]
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Parameter} & " + " & ".join(rf"\textbf{{{c}}}" for c, _ in COLUMNS) + r" \\",
        r"\midrule",
    ]
    for label, keys, fmt in ROWS:
        cells = [_cell(cfg, keys, fmt) for cfg in configs]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path = ns.out_path("table_S1_parameters") + ".tex"
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
