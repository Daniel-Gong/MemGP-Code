"""Supplementary figure — GP prior samples of f.

Renders independent draws from GP(0, k) as maps R^2 -> R on the
location x color torus (GPyTorch PeriodicKernel product, lambda = 10,
period = 360 deg):

  a-c  three prior samples as 3-D surfaces
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource, LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from figures import nature_style as ns


ns.set_style(8)
mpl.rcParams["mathtext.fontset"] = "cm"

FS_NOTE = 7.5
FS_MIN = 7.0
HERE = Path(__file__).resolve().parent
NAME = "figS_gp_prior"

LAM = 10.0
PERIOD = 360.0
N_SURF = 72
SEED = 4
SAMPLE_IDX = (0, 1, 2)
VIEW = (26, -58)

# GPyTorch PeriodicKernel: exp(-2 sin^2(pi |x-x'| / P) / lambda)  (lambda not squared)
def k_per(dx, lam=LAM, period=PERIOD):
    return np.exp(-2.0 * np.sin(np.pi * np.abs(dx) / period) ** 2 / lam)


K_CMAP = LinearSegmentedColormap.from_list(
    "kseq", ["#F4F7FB", "#DCE8F5", "#9CC2E6", "#3F7CBF", "#0F4D92", "#08243F"],
)
F_CMAP = LinearSegmentedColormap.from_list(
    "fdiv", ["#2166AC", "#67A9CF", "#F7F7F7", "#EF8A62", "#B2182B"], N=256,
)


def _close_torus(axis, surf):
    """Repeat the first row/column so the plotted square visually wraps."""
    axis_c = np.concatenate([axis, [axis[0] + PERIOD]])
    surf_c = np.pad(np.asarray(surf, float), ((0, 1), (0, 1)), mode="wrap")
    return axis_c, surf_c


def _sample_prior(n, n_draw, seed):
    """Kronecker samples of the product periodic kernel on an n x n torus grid."""
    axis = np.linspace(-180.0, 180.0, n, endpoint=False)
    K1 = k_per(axis[:, None] - axis[None, :])
    evals, evecs = np.linalg.eigh(K1)
    evals = np.clip(evals, 0.0, None)
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, n, n_draw))
    scale = np.sqrt(evals[:, None] * evals[None, :])
    samples = np.einsum("ij,jkn,lk->iln", evecs, scale[..., None] * Z, evecs)
    return axis, samples


def _surface3d(ax, axis, surf, cmap, vmin, vmax, view=VIEW, zlabel=False):
    axis_c, z = _close_torus(axis, surf)
    X, Y = np.meshgrid(axis_c, axis_c, indexing="ij")
    ls = LightSource(azdeg=315, altdeg=48)
    rgb = ls.shade(z, cmap=cmap, blend_mode="soft", vert_exag=1.15,
                   vmin=vmin, vmax=vmax)
    ax.plot_surface(X, Y, z, rcount=z.shape[0], ccount=z.shape[1],
                    facecolors=rgb, linewidth=0, antialiased=True,
                    edgecolor="none", shade=False, rasterized=True)
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1.0, 1.0, 0.50), zoom=0.92)
    for p in (ax.xaxis, ax.yaxis, ax.zaxis):
        p.pane.fill = False
        p.pane.set_facecolor((1, 1, 1, 0))
        p.pane.set_edgecolor((1, 1, 1, 0))
        p._axinfo["axisline"]["color"] = ns.INK
        p._axinfo["axisline"]["linewidth"] = 1.0
        p.line.set_color(ns.INK)
        p.line.set_linewidth(1.0)
        p.line.set_visible(True)
        p.line.set_clip_on(False)
    ax.grid(False)
    ax.set_xticks([-180, 180])
    ax.set_yticks([-180, 180])
    ax.set_zticks([vmin, 0.0, vmax])
    if not zlabel:
        ax.set_zticklabels([])
    ax.tick_params(labelsize=FS_MIN - 0.3, pad=1.5, length=3.5, width=0.8,
                   colors=ns.INK)
    ax.set_xlabel(r"location $\ell$", fontsize=FS_MIN, labelpad=2)
    ax.set_ylabel(r"color $c$", fontsize=FS_MIN, labelpad=2)
    ax.set_zlabel("")
    if zlabel:
        # Above the top of the vertical axis, centered on the line.
        ax.figure.canvas.draw()
        spine = ax.zaxis.line.get_xydata()
        top = spine[int(np.argmax(spine[:, 1]))]
        # Italic f leans right of its box, so the anchor sits slightly left
        # of the spine and the letter is centered on the line.
        ax.text2D(
            top[0] - 0.0022, top[1] + 0.008, r"$f$", transform=ax.transData,
            ha="center", va="bottom", fontsize=FS_MIN, color=ns.INK,
            clip_on=False,
        )
    # Inset the surface from the box so native spines (drawn under the
    # surface) remain visible along the tick-bearing edges.
    xy_pad = 12.0
    z_pad = 0.10 * (vmax - vmin)
    ax.set_xlim(-180.0 - xy_pad, 180.0 + xy_pad)
    ax.set_ylim(-180.0 - xy_pad, 180.0 + xy_pad)
    ax.set_zlim(vmin - z_pad, vmax + z_pad)
    ax.set_facecolor((1, 1, 1, 0))


def panel_a(ax):
    dx = np.linspace(-PERIOD, PERIOD, 721)
    k = k_per(dx)
    ax.axvspan(-180, 180, color=ns.MEM_PALE, lw=0, zorder=0)
    ax.axhline(0, color=ns.GREY_LT, lw=0.6, zorder=1)
    ax.plot(dx, k, color=ns.MEM, lw=1.9, zorder=3, solid_capstyle="round")
    ax.set_xlim(-PERIOD, PERIOD)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xticks([-360, -180, 0, 180, 360])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel(r"displacement $\Delta$ (deg)")
    ax.set_ylabel(r"$k_{\mathrm{per}}(\Delta,\,0)$")
    ax.text(0.0, 0.10, r"one period $P=360^\circ$", ha="center", va="bottom",
            fontsize=FS_MIN, color=ns.MEM, transform=ax.get_xaxis_transform())
    ax.text(270, 0.18, r"$\lambda=10$", ha="center", va="center",
            fontsize=FS_NOTE, color=ns.MEM)
    ns.clean(ax)


def panel_b(ax):
    g = np.linspace(-180.0, 180.0, 241)
    L, C = np.meshgrid(g, g, indexing="ij")
    K = k_per(L) * k_per(C)
    kmin, kmax = float(K.min()), 1.0
    im = ax.imshow(K.T, origin="lower", cmap=K_CMAP, vmin=kmin, vmax=kmax,
                   extent=[-180, 180, -180, 180], interpolation="bilinear",
                   rasterized=True, aspect="equal")
    ax.contour(g, g, K.T, levels=[0.75, 0.85, 0.95], colors="white",
               linewidths=0.55, linestyles="solid", zorder=2)
    ax.plot(0, 0, marker="o", ms=4.0, mfc="white", mec=ns.INK, mew=0.6,
            zorder=3)
    ax.set_xticks([-180, 0, 180])
    ax.set_yticks([-180, 0, 180])
    ax.set_xlabel(r"location $\ell$ (deg)")
    ax.set_ylabel(r"color $c$ (deg)")
    for s in ax.spines.values():
        s.set_linewidth(0.7)
        s.set_color(ns.GREY_DK)
    return im, kmin, kmax


def main():
    axis, samples = _sample_prior(N_SURF, n_draw=max(SAMPLE_IDX) + 1, seed=SEED)
    draws = [samples[:, :, i] for i in SAMPLE_IDX]
    vmax = float(max(np.abs(s).max() for s in draws))
    vmax = 0.05 * np.ceil(vmax / 0.05)

    fig = plt.figure(figsize=(6.5, 2.85))
    gs = fig.add_gridspec(1, 3, wspace=0.04,
                          left=0.08, right=0.97, top=0.92, bottom=0.18)

    ax_a = fig.add_subplot(gs[0, 0], projection="3d")
    ax_b = fig.add_subplot(gs[0, 1], projection="3d")
    ax_c = fig.add_subplot(gs[0, 2], projection="3d")

    for i, (ax, surf) in enumerate(zip((ax_a, ax_b, ax_c), draws)):
        _surface3d(ax, axis, surf, F_CMAP, vmin=-vmax, vmax=vmax,
                   zlabel=(i == 0))

    sm = mpl.cm.ScalarMappable(norm=Normalize(-vmax, vmax), cmap=F_CMAP)
    sm.set_array([])
    fig.canvas.draw()
    pos_a = ax_a.get_position()
    pos_c = ax_c.get_position()
    cax_f = fig.add_axes([pos_a.x0, 0.045, pos_c.x1 - pos_a.x0, 0.022])
    cb_f = fig.colorbar(sm, cax=cax_f, orientation="horizontal",
                        ticks=[-vmax, 0, vmax])
    cb_f.ax.tick_params(labelsize=FS_MIN, length=2.2)
    cb_f.set_label(r"$f(\ell,\,c)$", fontsize=FS_NOTE, labelpad=1)

    def _letter(ax, lab, dx=-0.02, dy=0.02):
        p = ax.get_position()
        fig.text(p.x0 + dx, p.y1 + dy, lab, fontsize=11, fontweight="bold",
                 ha="right", va="bottom", color=ns.INK)

    _letter(ax_a, "a", dx=0.03, dy=-0.01)
    _letter(ax_b, "b", dx=0.03, dy=-0.01)
    _letter(ax_c, "c", dx=0.03, dy=-0.01)

    ns.save_fig(fig, NAME)
    print(f"k_per(180)={k_per(180):.4f}  sample |f| max={vmax:.2f}")


if __name__ == "__main__":
    main()
