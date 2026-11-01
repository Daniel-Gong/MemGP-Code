"""Supplementary figure — how the prior kernel shapes the derived connectivity.

The recurrent operator of the derived circuit contains the prior precision
K_ZZ^{-1}; Fig. 5b shows -K_ZZ^{-1} for the paper's periodic kernel. This
figure repeats that computation on the same 10 x 10 inducing grid for kernels
of different smoothness and lengthscale.

Panels
  a-c  Outgoing weights -K_ZZ^{-1} from the unit at the origin for an
       exponential kernel (a), a Matern-3/2 kernel (b) and the periodic kernel
       used in the paper (c). Same product construction, grid, jitter and
       colormap as Fig. 5b.
  d-f  The paper's periodic kernel with jitter 1e-6, 1e-4 and 1e-2 added to
       K_ZZ before inversion (a-c use 1e-3, as Fig. 5b).
  g    Location slice of the outgoing weights (at the source color) for the
       periodic kernel across lengthscales (jitter 1e-3).

Run from paper/code:  ../../.venv/bin/python -m figures.figS_kernel_precision
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from figures import nature_style as ns
from figures.fig5_circuit import HAT_CMAP, _centered_weights


ns.set_style(8)
mpl.rcParams["mathtext.fontset"] = "cm"

FS_NOTE = 7.5
FS_MIN = 7.0
HERE = Path(__file__).resolve().parent
NAME = "figS_kernel_precision"

M_SIDE = 10          # inducing grid side, as in Fig. 5b
JITTER = 1e-3        # added to K_ZZ before inversion, as in Fig. 5b
LAM_PAPER = 10.0     # periodic lengthscale of the paper's kernel
ELL_CMP = 36.0       # lengthscale (deg) of the comparison kernels = one grid step
LAMS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0)
EPS_SWEEP = (1e-6, 1e-4, 1e-2)   # jitter values for d-f; c uses JITTER


# 1-D kernels on circular distance d (deg). Period 360 deg.
def k_periodic(d, lam):
    # GPyTorch PeriodicKernel: exp(-2 sin^2(pi d / P) / lambda), lambda not squared.
    return np.exp(-2.0 * np.sin(np.pi * d / 360.0) ** 2 / lam)


def k_matern32(d, ell):
    r = np.sqrt(3.0) * d / ell
    return (1.0 + r) * np.exp(-r)


def k_exponential(d, ell):
    return np.exp(-d / ell)


KERNELS = (
    ("exponential", lambda d: k_exponential(d, ELL_CMP),
     r"exponential, $\ell=36^\circ$"),
    ("matern32", lambda d: k_matern32(d, ELL_CMP),
     "Matérn-3/2, $\\ell=36^\\circ$"),
    ("periodic", lambda d: k_periodic(d, LAM_PAPER),
     r"periodic, $\lambda=10$ (as Fig. 5b)"),
)


def grid_axis(m=M_SIDE):
    half = 180.0 / m
    return np.linspace(-180.0 + half, 180.0 - half, m)


def outgoing_weights(k1d, m=M_SIDE, jitter=JITTER):
    """-K_ZZ^{-1} on the product kernel k1d(d_loc) k1d(d_col), m x m grid.

    Returns the grid axis, the source coordinates, the weight image
    hat[location, color] from the source, and the smallest eigenvalue of the
    1-D kernel matrix on the grid (conditioning check).
    """
    axis = grid_axis(m)
    loc, color = np.meshgrid(axis, axis, indexing="ij")
    z = np.column_stack([loc.ravel(), color.ravel()])
    d_loc = ns.circ_dist(z[:, 0:1], z[:, 0:1].T)
    d_col = ns.circ_dist(z[:, 1:2], z[:, 1:2].T)
    kzz = k1d(d_loc) * k1d(d_col)
    weights = -np.linalg.inv(kzz + jitter * np.eye(m * m))
    np.fill_diagonal(weights, 0.0)
    weights /= np.max(np.abs(weights))
    src = int(np.argmin(np.abs(z[:, 0]) + np.abs(z[:, 1])))
    hat = weights[src].reshape(m, m)
    k1 = k1d(ns.circ_dist(axis[:, None], axis[None, :]))
    min_eig = float(np.linalg.eigvalsh(k1).min())
    return axis, z[src], hat, min_eig


def location_slice(axis, src_xy, hat):
    """Weights along location at the source color, ordered by signed offset."""
    rel = (axis - src_xy[0] + 180.0) % 360.0 - 180.0
    order = np.argsort(rel, kind="mergesort")
    j = int(np.argmin(np.abs(axis - src_xy[1])))
    return rel[order], hat[order, j]


def panel_map(ax, axis, src_xy, hat, norm, title, ylabel):
    x_edges, y_edges, show = _centered_weights(axis, src_xy, hat)
    ax.pcolormesh(
        x_edges, y_edges, show.T, cmap=HAT_CMAP, norm=norm, shading="flat",
        edgecolors="none", linewidth=0, antialiased=False,
    )
    ax.scatter([0.0], [0.0], s=30, marker="o", facecolors="none",
               edgecolors=ns.INK, linewidths=0.9, zorder=5)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([-180, 0, 180])
    ax.set_yticks([-180, 0, 180])
    ax.set_xlabel("location (deg)")
    if ylabel:
        ax.set_ylabel("color (deg)")
    else:
        ax.set_yticklabels([])
    ax.set_title(title, fontsize=FS_NOTE, pad=4)
    ns.clean(ax)


def panel_slices(ax, norm):
    rows = []
    for lam in LAMS:
        axis, src_xy, hat, _ = outgoing_weights(lambda d: k_periodic(d, lam))
        rel, w = location_slice(axis, src_xy, hat)
        rows.append(w)
    rows = np.array(rows)
    spacing = float(rel[1] - rel[0])
    x_edges = np.concatenate([rel - spacing / 2, [rel[-1] + spacing / 2]])
    y_edges = np.arange(len(LAMS) + 1) - 0.5
    ax.pcolormesh(x_edges, y_edges, rows, cmap=HAT_CMAP, norm=norm,
                  shading="flat", edgecolors="white", linewidth=0.4)
    ax.set_yticks(range(len(LAMS)))
    ax.set_yticklabels([f"{lam:g}" for lam in LAMS])
    ax.set_ylabel(r"$\lambda$")
    ax.set_xticks([-180, -108, -36, 36, 108])
    ax.set_xlabel("location offset from source (deg)")
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ns.clean(ax)
    return rel, rows


def main():
    maps = [outgoing_weights(k) for _, k, _ in KERNELS]
    eps_maps = [outgoing_weights(lambda d: k_periodic(d, LAM_PAPER), jitter=e)
                for e in EPS_SWEEP]
    lo = min(float(h.min()) for _, _, h, _ in maps + eps_maps)
    lo = min(lo, -0.25)
    norm = TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=1.0)

    fig = plt.figure(figsize=(6.5, 7.0))
    outer = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.66], hspace=0.55,
                             left=0.09, right=0.90, top=0.95, bottom=0.07)
    top = outer[0].subgridspec(1, 3, wspace=0.18)
    mid = outer[1].subgridspec(1, 3, wspace=0.18)
    axes_map = [fig.add_subplot(top[0, i]) for i in range(3)]
    axes_eps = [fig.add_subplot(mid[0, i]) for i in range(3)]
    ax_g = fig.add_subplot(outer[2])

    for i, (ax, (axis, src_xy, hat, _), (_, _, title)) in enumerate(
            zip(axes_map, maps, KERNELS)):
        panel_map(ax, axis, src_xy, hat, norm, title, ylabel=(i == 0))
    for i, (ax, (axis, src_xy, hat, _), e) in enumerate(
            zip(axes_eps, eps_maps, EPS_SWEEP)):
        exp10 = int(np.round(np.log10(e)))
        title = r"periodic, $\lambda=10$, $\varepsilon=10^{%d}$" % exp10
        panel_map(ax, axis, src_xy, hat, norm, title, ylabel=(i == 0))

    rel, rows = panel_slices(ax_g, norm)

    fig.canvas.draw()
    p_top = axes_map[-1].get_position()
    p_bot = ax_g.get_position()
    cax = fig.add_axes([0.925, p_bot.y0, 0.016, p_top.y1 - p_bot.y0])
    sm = ScalarMappable(norm=norm, cmap=HAT_CMAP)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, ticks=[lo, 0.0, 1.0])
    cb.ax.set_yticklabels([f"{lo:.2f}", "0", "1"])
    cb.ax.tick_params(labelsize=FS_MIN, length=2.2, pad=1.2)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(ns.GREY_DK)
    cax.set_title(r"$-\mathbf{K}_{ZZ}^{-1}$", fontsize=FS_MIN, pad=6)

    for ax, letter in zip(axes_map, "abc"):
        ns.panel(ax, letter, x=-0.22 if letter == "a" else -0.10, y=1.10)
    for ax, letter in zip(axes_eps, "def"):
        ns.panel(ax, letter, x=-0.22 if letter == "d" else -0.10, y=1.10)
    ns.panel(ax_g, "g", x=-0.055, y=1.10)

    ns.save_fig(fig, NAME)

    # Numbers for the caption and the supplementary text.
    axis0 = grid_axis()
    def _report(tag, axis, src_xy, hat):
        rel_k, w = location_slice(axis, src_xy, hat)
        i0 = int(np.argmin(np.abs(axis - src_xy[0])))
        j0 = int(np.argmin(np.abs(axis - src_xy[1])))
        m = len(axis)
        diag = [hat[(i0 + k) % m, (j0 + k) % m] for k in range(1, 4)]
        ax_w = w[(rel_k > 0) & (rel_k <= 108)]
        print(f"{tag:28s} axis(36,72,108)={np.round(ax_w, 3)} "
              f"diag(36,72,108)={np.round(diag, 3)}")
    for (name, _, _), (axis, src_xy, hat, min_eig) in zip(KERNELS, maps):
        _report(f"{name} eps=1e-3 (min eig {min_eig:.0e})", axis, src_xy, hat)
    for e, (axis, src_xy, hat, _) in zip(EPS_SWEEP, eps_maps):
        _report(f"periodic eps={e:.0e}", axis, src_xy, hat)
    for lam, w in zip(LAMS, rows):
        pairs = ", ".join(f"{int(r):+d}:{v:+.3f}" for r, v in zip(rel, w))
        print(f"periodic lam={lam:<4g} slice {pairs}")


if __name__ == "__main__":
    main()
