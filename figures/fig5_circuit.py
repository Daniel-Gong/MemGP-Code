"""Figure 5 — recurrent implementation and contact with neural data.

The circuit that performs MemGP maintenance is a Mexican-hat recurrent network,
and three of its properties match what neurophysiology has measured.
Panels d, e, f, and g are read from the explicit network (fixed W; Adam on m and S).

Panels
  a  5x5 inducing grid -> 25 recurrent units on a ring -> rate equations
  b  Outgoing weights -K_ZZ^{-1} from one unit on the M=100 grid, source at the origin
  c  GP == explicit RNN: per-item recalled colours agree
  d  Drift of the peak read out from m over the delay (Wimmer et al. 2014)
  e  Circular SD of the peak against recall error (van Bergen et al. 2015)
  f  Change in posterior mean at the probed item from cue onset, cued vs neutral
  g  Change in posterior variance at the probed item from cue onset, cued vs neutral

Sizing: drawn at the printed size (6.5 in = \\textwidth at 1 in margins) so the
LaTeX scale factor is 1.0 and every font here is the font on the page. No text
below 7 pt.
"""

from csv import DictWriter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np
from scipy.stats import spearmanr

from figures import nature_style as ns


ns.set_style(8)
# LaTeX-quality math (Computer Modern) for $...$; labels stay Arial.
mpl.rcParams["mathtext.fontset"] = "cm"
FIG_W, FIG_H = 6.5, 6.0

# Font floor: 7 pt. Ticks and axis labels come from set_style(8) -> 8 / 9 pt.
FS_NOTE = 7.5      # in-panel annotations
FS_MIN = 7.0       # absolute floor (legends, dense corner text)
FS_HEAD = 9.0      # schematic column headings
FS_EQ = 10.0       # the rate equation
FS_PANEL = 11.0    # panel letters

CIRCUIT = Path(ns.result_path("circuit"))
DATA_PATH = CIRCUIT / "rnn_equiv.npz"
DIFF_PATH = CIRCUIT / "bump_diffusion.npz"
UNC_PATH = CIRCUIT / "decoded_uncertainty.npz"
VAR_PATH = CIRCUIT / "cued_precision.npz"
OUT = Path(ns.out_path("fig5_circuit"))
SOURCE = OUT.parent / "fig5_circuit_source_data.csv"

INK = ns.INK
BLUE = ns.MEM
BLUE_MID = ns.MEM_MID
BLUE_PALE = ns.MEM_PALE
AMBER = "#B86B1E"
CUE = ns.CUE
NEUTRAL = ns.NEUTRAL
GREY = ns.GREY
GREY_DK = ns.GREY_DK
GREY_LT = ns.GREY_LT

# Outgoing weights: red inhibition, white zero, green excitation (same as panel a).
HAT_CMAP = LinearSegmentedColormap.from_list(
    "mex_hat",
    [ns.BAD, "#F7F7F7", ns.GOOD],
    N=256,
)


def _arrow(ax, start, end, color=INK, width=1.35, mutation=11, rad=0.0):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation,
        lw=width,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=1,
        shrinkB=1,
        zorder=8,
    )
    ax.add_patch(patch)
    return patch


def _grid_belief(n_side=5, sigma=64.0, noise=0.30):
    """Mean and marginal variance of a two-item GP on an n_side x n_side grid.

    Inducing sites are the centres of a regular tiling of the location x feature
    torus. Both items sit on sites. The mean is the noise-corrupted posterior
    mean; the variance is the matching marginal posterior variance.
    """
    half = 180.0 / n_side
    axis = np.linspace(-180.0 + half, 180.0 - half, n_side)
    loc, col = np.meshgrid(axis, axis, indexing="ij")
    z = np.column_stack([loc.ravel(), col.ravel()])
    items = np.array([[-72.0, 72.0], [72.0, -72.0]])

    def kern(a, b):
        d_loc = ns.circ_dist(a[:, None, 0], b[None, :, 0])
        d_col = ns.circ_dist(a[:, None, 1], b[None, :, 1])
        return np.exp(-0.5 * ((d_loc / sigma) ** 2 + (d_col / sigma) ** 2))

    k_zx = kern(z, items)
    k_xx = kern(items, items) + noise * np.eye(len(items))
    mean = k_zx @ np.linalg.solve(k_xx, np.ones(len(items)))
    mean = np.clip(mean / mean.max(), 0.0, 1.0)
    solved = np.linalg.solve(k_xx, k_zx.T)
    var = np.clip(1.0 - np.sum(k_zx * solved.T, axis=1), 0.0, None)
    return z, items, mean, var


def _ring_coupling(n, lengthscale=1.2):
    """Off-diagonal Mexican hat on a ring: −K^{-1}, local +, surround −."""
    theta = np.linspace(-180.0, 180.0, n, endpoint=False)
    delta = np.abs(theta[:, None] - theta[None, :])
    delta = np.minimum(delta, 360.0 - delta)
    kernel = np.exp(-2.0 * np.sin(np.pi * delta / 360.0) ** 2 / lengthscale ** 2)
    precision = np.linalg.inv(kernel + 1e-3 * np.eye(n))
    coupling = -precision
    np.fill_diagonal(coupling, 0.0)
    coupling /= np.max(np.abs(coupling))
    return coupling


def _inward_rad(start, end):
    """Arc3 bow toward the origin. rad is already scaled by chord length."""
    delta = end - start
    mid = 0.5 * (start + end)
    left = np.array([-delta[1], delta[0]])
    # Arc3 offsets the control point along (dy, -dx) = -left.
    sign = np.sign(np.dot(left, mid)) or 1.0
    return float(sign * 0.16)


def panel_a(ax):
    """GP belief on a 5 x 5 grid, drawn as 25 recurrent units on a ring."""
    ns.blank(ax, equal=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(
        0.5, 0.99, "Recurrent circuit connectivity and dynamics",
        ha="center", va="bottom", fontsize=FS_HEAD, fontweight="bold", color=INK,
    )

    z, items, mean, var = _grid_belief()
    var_n = (var - var.min()) / (var.max() - var.min() + 1e-9)

    # --- left: inducing values, two items ---
    belief = ax.inset_axes([0.012, 0.28, 0.30, 0.56])
    belief.set_xlim(-180, 180)
    belief.set_ylim(-180, 180)
    belief.set_aspect("equal")
    belief.set_facecolor(ns.GREY_FILL)
    belief.set_xticks([-180, 0, 180])
    belief.set_yticks([-180, 0, 180])
    belief.set_xlabel("location", fontsize=FS_NOTE, labelpad=1.2)
    belief.set_ylabel("color", fontsize=FS_NOTE, labelpad=1.2)
    belief.tick_params(labelsize=FS_MIN, length=2.2, pad=1.2)
    for spine in belief.spines.values():
        spine.set_color(ns.GREY_LT)
        spine.set_linewidth(0.7)
    belief.scatter(
        z[:, 0], z[:, 1], c=mean, cmap="viridis", vmin=0.0, vmax=1.0,
        marker="s", s=108, edgecolors="white", lw=0.4, zorder=3,
    )
    for loc, col in items:
        belief.scatter(
            loc, col, marker="*", s=58, color=ns.color_to_rgb(col),
            edgecolors="white", lw=0.5, zorder=6,
        )
    _arrow(ax, (0.325, 0.56), (0.395, 0.56), width=1.6, mutation=12)

    sm = ScalarMappable(norm=mpl.colors.Normalize(0.0, 1.0), cmap="viridis")
    sm.set_array([])
    cax = ax.inset_axes([0.055, 0.028, 0.21, 0.026])
    cb = ax.figure.colorbar(sm, cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(ns.GREY_LT)
    cb.set_ticks([])
    cax.text(-0.02, 0.5, "none", transform=cax.transAxes, fontsize=FS_MIN,
             color=ns.GREY_DK, ha="right", va="center", clip_on=False)
    cax.text(1.02, 0.5, "strong", transform=cax.transAxes, fontsize=FS_MIN,
             color=ns.GREY_DK, ha="left", va="center", clip_on=False)
    cax.text(0.5, -1.15, "predictive mean", transform=cax.transAxes,
             fontsize=FS_MIN, color=ns.GREY_DK, ha="center", va="top",
             clip_on=False)

    # --- middle: the same 25 units on a ring. size = m, spoke = variance ---
    pop = ax.inset_axes([0.395, 0.12, 0.26, 0.68])
    ns.blank(pop, equal=True)
    order = np.lexsort((z[:, 0], z[:, 1]))
    mean_r = mean[order]
    var_r = var_n[order]
    n_units = len(order)
    ang = np.linspace(0.0, 2.0 * np.pi, n_units, endpoint=False)
    radius_ring = 1.0
    xy = np.column_stack([radius_ring * np.cos(ang), radius_ring * np.sin(ang)])
    radii = 0.045 + 0.055 * mean_r
    lengths = 0.06 + 0.38 * var_r ** 2
    coupling = _ring_coupling(n_units)

    # Keep the Mexican-hat lobes: nearest-neighbour excitation and the
    # inhibitory surround. Drop the weak long-range tail.
    pairs = [
        (i, j, coupling[i, j])
        for i in range(n_units)
        for j in range(i + 1, n_units)
        if abs(coupling[i, j]) >= 0.20
    ]
    pairs.sort(key=lambda item: abs(item[2]))
    for i, j, weight in pairs:
        excitatory = weight > 0.0
        mag = abs(weight)
        pop.add_patch(FancyArrowPatch(
            xy[i], xy[j], arrowstyle="-",
            connectionstyle=f"arc3,rad={_inward_rad(xy[i], xy[j])}",
            lw=(0.55 + 1.1 * mag) if excitatory else (0.45 + 0.9 * mag),
            color=ns.GOOD if excitatory else ns.BAD,
            alpha=(0.35 + 0.45 * mag) if excitatory else (0.28 + 0.40 * mag),
            zorder=2 if excitatory else 1,
        ))
    for (x, y), radius, length, theta in zip(xy, radii, lengths, ang):
        pop.plot(
            [x, (radius_ring + radius + length) * np.cos(theta)],
            [y, (radius_ring + radius + length) * np.sin(theta)],
            color=ns.SLOT, lw=1.35, solid_capstyle="butt", zorder=3,
        )
    for (x, y), radius in zip(xy, radii):
        pop.add_patch(Circle(
            (x, y), radius, fc=ns.MEM, ec="white", lw=0.45, zorder=4,
        ))
    pop.set_xlim(-1.62, 1.62)
    pop.set_ylim(-1.62, 1.62)

    cx = 0.525
    y_ms = 0.868
    ax.scatter([cx - 0.070], [y_ms], s=28, color=ns.MEM, zorder=6, clip_on=False)
    ax.text(cx - 0.054, y_ms, r"$\mathbf{m}$", ha="left", va="center",
            fontsize=FS_MIN, color=INK)
    ax.plot([cx + 0.008, cx + 0.034], [y_ms, y_ms], color=ns.SLOT, lw=1.7,
            solid_capstyle="butt", clip_on=False, zorder=6)
    ax.text(cx + 0.040, y_ms, r"$\mathrm{diag}(\mathbf{S})$", ha="left", va="center",
            fontsize=FS_MIN, color=INK)

    y_ei = 0.040
    x_exc = cx - 0.120
    ax.plot([x_exc, x_exc + 0.024], [y_ei, y_ei], color=ns.GOOD, lw=1.7,
            solid_capstyle="butt", clip_on=False, zorder=6)
    ax.text(x_exc + 0.030, y_ei, "excitatory", ha="left", va="center",
            fontsize=FS_MIN, color=INK)
    x_inh = cx + 0.008
    ax.plot([x_inh, x_inh + 0.024], [y_ei, y_ei], color=ns.BAD, lw=1.7,
            solid_capstyle="butt", clip_on=False, zorder=6)
    ax.text(x_inh + 0.030, y_ei, "inhibitory", ha="left", va="center",
            fontsize=FS_MIN, color=INK)

    # --- right: maintenance dynamics ---
    ax.text(0.865, 0.74, r"$\tau\dot{\mathbf{m}}=-\mathbf{W}\mathbf{m}+\mathbf{b}$",
            ha="center", va="center", fontsize=FS_EQ, color=INK)
    ax.text(
        0.865, 0.52,
        r"$\tau\dot{\mathbf{S}}=-(\mathbf{W}-\beta\mathbf{S}^{-1})$",
        ha="center", va="center", fontsize=FS_HEAD, color=INK,
    )
    ax.text(
        0.865, 0.28,
        r"$\mathbf{W}=\dfrac{\mathbf{A}}{\sigma^{2}M}+\beta\mathbf{K}_{ZZ}^{-1}$",
        ha="center", va="center", fontsize=FS_HEAD, color=INK,
    )


def conjunction_weights(m=10, lengthscale=10.0, jitter=1e-3):
    """−K_ZZ^{-1} on the paper's product kernel and m x m inducing grid."""
    half = 180.0 / m
    axis = np.linspace(-180.0 + half, 180.0 - half, m)
    loc, color = np.meshgrid(axis, axis, indexing="ij")
    z = np.column_stack([loc.ravel(), color.ravel()])
    d_loc = ns.circ_dist(z[:, 0:1], z[:, 0:1].T)
    d_col = ns.circ_dist(z[:, 1:2], z[:, 1:2].T)
    kzz = np.exp(-2.0 * np.sin(np.pi * d_loc / 360.0) ** 2 / lengthscale)
    kzz *= np.exp(-2.0 * np.sin(np.pi * d_col / 360.0) ** 2 / lengthscale)
    weights = -np.linalg.inv(kzz + jitter * np.eye(m * m))
    np.fill_diagonal(weights, 0.0)
    weights /= np.max(np.abs(weights))
    src = int(np.argmin(np.abs(z[:, 0]) + np.abs(z[:, 1])))
    hat = weights[src].reshape(m, m)
    return axis, z[src], hat


def _centered_weights(axis, src_xy, hat):
    """Roll the torus so the source unit is at (0, 0).

    Returns location edges, color edges, and the weight image. One extra
    row and column repeat the cell at ±180°, so a window of [-180, 180]
    stays filled after the roll.
    """
    spacing = float(axis[1] - axis[0])
    half = spacing / 2.0

    def _order(coord):
        rel = (axis - coord + 180.0) % 360.0 - 180.0
        order = np.argsort(rel, kind="mergesort")
        return order, rel[order]

    io, loc_c = _order(src_xy[0])
    jo, col_c = _order(src_xy[1])
    rolled = hat[np.ix_(io, jo)]
    m = rolled.shape[0]
    show = np.empty((m + 1, m + 1))
    show[:m, :m] = rolled
    show[:m, m] = rolled[:, 0]
    show[m, :m] = rolled[0, :]
    show[m, m] = rolled[0, 0]

    def _edges(centers):
        edges = np.empty(m + 2)
        edges[:m] = centers - half
        edges[m] = centers[-1] + half
        edges[m + 1] = edges[m] + spacing
        return edges

    return _edges(loc_c), _edges(col_c), show


def panel_b(ax, cax, grid, src_xy, hat):
    """Outgoing −K_ZZ^{-1} from one inducing unit, centered at the origin."""
    x_edges, y_edges, show = _centered_weights(grid, src_xy, hat)
    lo, hi = float(np.min(show)), 1.0
    norm = TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi)
    # show[location, color]; pcolormesh rows are y.
    ax.pcolormesh(
        x_edges, y_edges, show.T, cmap=HAT_CMAP, norm=norm, shading="flat",
        edgecolors="none", linewidth=0, antialiased=False,
    )
    ax.scatter(
        [0.0], [0.0], s=36, marker="o",
        facecolors="none", edgecolors=INK, linewidths=0.9, zorder=5,
    )
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([-180, 0, 180])
    ax.set_yticks([-180, 0, 180])
    ax.set_xlabel("location (deg)")
    ax.set_ylabel("color (deg)")
    ns.clean(ax)

    sm = ScalarMappable(norm=norm, cmap=HAT_CMAP)
    sm.set_array(show)
    cb = ax.figure.colorbar(sm, cax=cax, ticks=[lo, 0.0, hi])
    cb.ax.set_yticklabels([f"{lo:.2f}", "0", f"{hi:.0f}"])
    cb.ax.tick_params(labelsize=FS_MIN, length=2.2, pad=1.2)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(GREY_DK)
    cax.set_title(r"$-\mathbf{K}_{ZZ}^{-1}$", fontsize=FS_MIN, pad=8)


def geometry_analysis(states_gp, states_rnn):
    """Agreement metrics quoted in the text (no longer plotted)."""
    def cosine_rdm(x):
        x = x - x.mean(axis=1, keepdims=True)
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-15)
        return 1.0 - x @ x.T

    rdm_gp = cosine_rdm(states_gp)
    rdm_rnn = cosine_rdm(states_rnn)
    upper = np.triu_indices_from(rdm_gp, k=1)
    rdm_r = float(np.corrcoef(rdm_gp[upper], rdm_rnn[upper])[0, 1])

    velocity_gp = np.diff(states_gp, axis=0)
    velocity_rnn = np.diff(states_rnn, axis=0)
    mismatch = np.linalg.norm(velocity_gp - velocity_rnn, axis=1) / np.maximum(
        np.linalg.norm(velocity_gp, axis=1), 1e-15
    )
    return rdm_r, mismatch


def panel_c(ax, data):
    """GP == explicit RNN: probed-item recall agrees trial by trial.

    One point per trial: the mean absolute error of 100 normalized reports
    from the GP (x) and from the network (y), for the item the cue points at.
    Neutral and cued are coloured apart so both settings of W are visible; the
    correlation is pooled.
    """
    xs, ys = [], []
    for cond in ("neutral", "cued"):
        xs.append(np.asarray(data[f"{cond}_mae_gp"], float))
        ys.append(np.asarray(data[f"{cond}_mae_net"], float))
    lim = float(np.ceil(max(np.max(np.concatenate(xs)),
                            np.max(np.concatenate(ys))) / 10.0) * 10.0)
    ax.plot([0, lim], [0, lim], color=GREY, lw=0.9, ls=(0, (3, 2.5)), zorder=2)
    for x, y, colour, zo in zip(xs, ys, (NEUTRAL, CUE), (3, 4)):
        ax.scatter(x, y, s=13, facecolors=colour, edgecolors="white",
                   linewidths=0.4, alpha=0.85, zorder=zo)
    x_all = np.concatenate(xs)
    y_all = np.concatenate(ys)
    r = float(np.corrcoef(x_all, y_all)[0, 1])
    stats = dict(r=r, n=int(x_all.size))

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xticks([0, lim])
    ax.set_yticks([0, lim])
    ax.set_xlabel("MemGP |error| (deg)")
    ax.set_ylabel("RNN |error| (deg)")

    ax.text(0.04, 0.97, "neutral", transform=ax.transAxes, ha="left", va="top",
            fontsize=FS_MIN, color=NEUTRAL, fontweight="bold")
    ax.text(0.04, 0.865, "cued", transform=ax.transAxes, ha="left", va="top",
            fontsize=FS_MIN, color=CUE, fontweight="bold")
    ax.text(0.96, 0.04, rf"$r={r:.2f}$", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=FS_MIN, color=INK)
    ns.clean(ax)
    return stats


def diffusion_stats(diff, n_boot=1000, seed=3):
    """Single-trial rank correlation of |displacement| with delay, and the
    across-trial mean, both with trial-level bootstrap intervals.

    Resampling whole trials is the point: the 50 epochs within a trial are
    snapshots of one random walk (adjacent snapshots correlate at 0.99), so the
    trial, not the point, is the independent unit.
    """
    ep = np.asarray(diff["epochs"], float)
    # Peak of the posterior mean on the 0.1 deg decode axis, tracked from the
    # item's color through the delay. The population vector of the same
    # profile sits off this peak: the surround enters the vector average.
    disp = np.abs(np.asarray(diff["disp_matrix"], float))
    keep = ep > 0
    ep = ep[keep]
    disp = disp[:, keep]

    n_tr = disp.shape[0]
    xr = np.tile(ep, (n_tr, 1))
    rho = float(spearmanr(xr.ravel(), disp.ravel()).statistic)

    rng = np.random.default_rng(seed)
    boot_rho = np.empty(n_boot)
    boot_mean = np.empty((n_boot, ep.size))
    for i in range(n_boot):
        sample = disp[rng.integers(0, n_tr, n_tr)]
        boot_rho[i] = spearmanr(xr.ravel(), sample.ravel()).statistic
        boot_mean[i] = sample.mean(axis=0)
    lo, hi = np.percentile(boot_rho, [2.5, 97.5])
    band_lo, band_hi = np.percentile(boot_mean, [2.5, 97.5], axis=0)
    return dict(ep=ep, disp=disp, rho=rho, lo=float(lo), hi=float(hi),
                mean=disp.mean(axis=0), band_lo=band_lo, band_hi=band_hi)


def panel_e(ax, st, seed=7):
    """Diffusing bump: single-trial |displacement| grows over the delay."""
    ep, disp = st["ep"], st["disp"]
    rng = np.random.default_rng(seed)
    x = np.tile(ep, (disp.shape[0], 1)).ravel()
    y = disp.ravel()
    # Epochs are sampled every other step, so jitter x to keep the cloud off a
    # lattice; the drift itself is continuous and stays as measured.
    ax.scatter(x + rng.uniform(-0.9, 0.9, x.size), y,
               s=1.6, color=BLUE_MID, alpha=0.13, edgecolors="none",
               rasterized=True, zorder=2)

    ax.fill_between(ep, st["band_lo"], st["band_hi"], color=CUE, alpha=0.30,
                    lw=0, zorder=4)
    ax.plot(ep, st["mean"], color=CUE, lw=1.8, zorder=5)

    ax.set_xlim(0, ep.max() + 1)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("delay (epoch)")
    ax.set_ylabel("|peak displacement| (deg)")
    ax.text(0.04, 0.95,
            rf"$\rho={st['rho']:.2f}$" "\n" rf"[{st['lo']:.2f}, {st['hi']:.2f}]",
            transform=ax.transAxes, ha="left", va="top", fontsize=FS_MIN,
            color=INK, linespacing=1.2)
    ns.clean(ax)


def local_linear(x, y, grid, frac=0.75):
    """LOESS: tricube-weighted local linear fit, no robustness pass.

    Matches statsmodels lowess(..., it=0) to the decimal, so the figure keeps
    its own copy rather than taking on the dependency. Local *linear* rather
    than local constant because the curve is read out to both edges: a kernel
    or k-nearest-neighbour mean takes its whole window from one side there and
    is pulled toward the interior, over-estimating the low-width end by about
    6 deg against the raw slice mean.
    """
    k = max(3, int(frac * x.size))
    out = np.empty(grid.size)
    for i, gi in enumerate(grid):
        dist = np.abs(x - gi)
        h = max(np.partition(dist, k - 1)[k - 1], 1e-12)
        wgt = (1.0 - np.clip(dist / h, 0.0, 1.0) ** 3) ** 3
        sw = wgt.sum()
        mx = (wgt @ x) / sw
        my = (wgt @ y) / sw
        sxx = wgt @ (x - mx) ** 2
        sxy = wgt @ ((x - mx) * (y - my))
        out[i] = my + (sxy / sxx if sxx > 1e-12 else 0.0) * (gi - mx)
    return out


def panel_f(ax, unc, frac=0.75, n_boot=300, seed=5):
    """Circular SD of sampled peaks against recall error (van Bergen et al. 2015).

    Vertical axis is the circular SD of colour peaks over posterior draws,
    taken apart from the draw that produced the report.
    """
    errs = np.abs(np.asarray(unc["errs"], float))
    width = np.asarray(unc["uncert"], float)
    rho = float(unc["rho"])
    # Item bootstrap of the rank correlation. Seed 5 reproduces the interval
    # quoted in the text, [0.28, 0.43].
    rng_rho = np.random.default_rng(5)
    boot_rho = np.empty(1000)
    for i in range(1000):
        idx = rng_rho.integers(0, errs.size, errs.size)
        boot_rho[i] = spearmanr(errs[idx], width[idx]).statistic
    rho_lo, rho_hi = np.percentile(boot_rho, [2.5, 97.5])

    ax.scatter(errs, width, s=5.0, color=AMBER, alpha=0.30, edgecolors="none",
               rasterized=True, zorder=2)

    # Continuous trend instead of bins. Moving the span over 0.3-0.75 shifts
    # the curve by under 3 deg, and the reported rho uses the raw items anyway.
    grid = np.linspace(errs.min(), errs.max(), 140)
    rng = np.random.default_rng(seed)
    boot = np.array([local_linear(errs[i], width[i], grid, frac)
                     for i in (rng.integers(0, errs.size, errs.size)
                               for _ in range(n_boot))])
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    ax.fill_between(grid, lo, hi, color=INK, alpha=0.22, lw=0, zorder=4)
    ax.plot(grid, local_linear(errs, width, grid, frac), color=INK, lw=1.8,
            zorder=5)

    ax.set_xlim(left=-4)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("|recall error| (deg)")
    ax.set_ylabel("circular SD of peak (deg)")
    ax.text(0.04, 0.95,
            rf"$\rho={rho:.2f}$" "\n" rf"[{rho_lo:.2f}, {rho_hi:.2f}]",
            transform=ax.transAxes, ha="left", va="top", fontsize=FS_MIN,
            color=INK, linespacing=1.2)
    ns.clean(ax)


def delta_stats(cued, neutral, cue, n_boot=1000, seed=4, n_trials=None):
    """Change from cue onset, cued and neutral.

    Each trial is subtracted by its own value at cue onset, so the
    per-trial offset drops out, then the across-trial mean is bootstrapped.
    ``cue`` is the 1-indexed epoch at which the cue is applied; the baseline
    is the sample plotted there.
    """
    vc = np.asarray(cued, float)
    vn = np.asarray(neutral, float)
    ep = np.arange(1, vc.shape[1] + 1)
    vc = vc - vc[:, cue - 1:cue]
    vn = vn - vn[:, cue - 1:cue]
    rng = np.random.default_rng(seed)

    def band(trials):
        n = trials.shape[0]
        idx = rng.integers(0, n, size=(n_boot, n))
        means = trials[idx].mean(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5], axis=0)
        return trials.mean(axis=0), lo, hi

    mc, loc, hic = band(vc)
    mn, lon, hin = band(vn)
    return dict(ep=ep, cue=int(cue), n=int(n_trials if n_trials is not None else vc.shape[0]),
                mean_c=mc, lo_c=loc, hi_c=hic, mean_n=mn, lo_n=lon, hi_n=hin)


def variance_stats(vdat, n_boot=1000, seed=4):
    """Change in posterior variance from cue onset. Epoch 1 is the first
    maintenance step; the cue is applied from epoch 50."""
    return delta_stats(vdat["var_cued"], vdat["var_neutral"],
                       int(vdat["cue_start"]), n_boot=n_boot, seed=seed,
                       n_trials=int(vdat["n_trials"]))


def panel_delta(ax, st, ylabel, dash_neutral=False, legend_loc="lower right",
                bottom_pad=0.06):
    """Change from cue onset at the probed item.

    Ticks are in units of 10^{-3}. The stored series stay in raw units.
    When the two traces overlie, the neutral curve is dashed and drawn
    on top so both remain visible.
    """
    ep = st["ep"]
    scale = 1e3
    lo_n, hi_n = st["lo_n"] * scale, st["hi_n"] * scale
    lo_c, hi_c = st["lo_c"] * scale, st["hi_c"] * scale
    ax.axhline(0, color=GREY, lw=0.7, ls=(0, (3, 2)), zorder=0)
    ax.fill_between(ep, lo_n, hi_n, color=NEUTRAL, alpha=0.18, lw=0, zorder=1)
    ax.fill_between(ep, lo_c, hi_c, color=CUE, alpha=0.22, lw=0, zorder=2)
    if dash_neutral:
        ax.plot(ep, st["mean_c"] * scale, color=CUE, lw=1.6, label="cued", zorder=3)
        ax.plot(ep, st["mean_n"] * scale, color=NEUTRAL, lw=1.6, ls=(0, (4, 1.6)),
                label="neutral", zorder=4)
    else:
        ax.plot(ep, st["mean_n"] * scale, color=NEUTRAL, lw=1.6, label="neutral", zorder=3)
        ax.plot(ep, st["mean_c"] * scale, color=CUE, lw=1.6, label="cued", zorder=4)
    ax.axvline(st["cue"], color=GREY, lw=0.8, ls=(0, (3, 2.5)), zorder=2)
    y_lo = min(lo_c.min(), lo_n.min())
    y_hi = max(hi_c.max(), hi_n.max())
    span = y_hi - y_lo
    if span == 0:
        span = 1.0
    ax.set_xlim(0, ep[-1] + 1)
    ax.set_ylim(y_lo - bottom_pad * span, y_hi + 0.16 * span)
    ax.text(st["cue"] + 2, y_hi + 0.12 * span, "cue", ha="left", va="top",
            fontsize=FS_MIN, color=GREY_DK)
    ax.set_xlabel("delay (epoch)")
    ax.set_ylabel(ylabel, fontsize=FS_NOTE)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index("cued"), labels.index("neutral")]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc=legend_loc, fontsize=FS_MIN, borderpad=0.2,
              handlelength=1.6, labelspacing=0.2)
    ns.clean(ax)


def panel_variance(ax, st):
    panel_delta(ax, st, r"$\Delta$ variance ($\times 10^{-3}$)",
                legend_loc="lower left", bottom_pad=0.38)


def _write_delta(w, panel, prefix, st):
    for key, suffix in (("mean_c", "cued_mean"), ("lo_c", "cued_lo"),
                        ("hi_c", "cued_hi"), ("mean_n", "neutral_mean"),
                        ("lo_n", "neutral_lo"), ("hi_n", "neutral_hi")):
        for i, (x, value) in enumerate(zip(st["ep"], st[key])):
            w.writerow(dict(panel=panel, series=f"{prefix}_{suffix}",
                            index=i, x=x, y="", value=value))


def write_source(grid, src_xy, hat, data, err_stats, ep, disp, unc, st_var, st_mean):
    fields = ["panel", "series", "index", "x", "y", "value"]
    with SOURCE.open("w", newline="") as handle:
        w = DictWriter(handle, fieldnames=fields)
        w.writeheader()
        m = grid.size
        for i, loc in enumerate(grid):
            for j, col in enumerate(grid):
                # Same offsets as panel b: the source unit is at the origin.
                x = (loc - src_xy[0] + 180.0) % 360.0 - 180.0
                y = (col - src_xy[1] + 180.0) % 360.0 - 180.0
                w.writerow(dict(panel="b", series="outgoing_weight",
                                index=i * m + j, x=x, y=y, value=hat[i, j]))
        for cond in ("neutral", "cued"):
            xs = np.asarray(data[f"{cond}_mae_gp"], float)
            ys = np.asarray(data[f"{cond}_mae_net"], float)
            for i, (x, y) in enumerate(zip(xs, ys)):
                w.writerow(dict(panel="c", series=f"mean_abs_error_{cond}", index=i,
                                x=x, y=y, value=err_stats["r"]))
        for t in range(disp.shape[0]):
            for i, x in enumerate(ep):
                w.writerow(dict(panel="d", series=f"abs_drift_trial{t}", index=i,
                                x=x, y="", value=disp[t, i]))
        errs = np.abs(np.asarray(unc["errs"], float))
        width = np.asarray(unc["uncert"], float)
        for i, (x, y) in enumerate(zip(errs, width)):
            w.writerow(dict(panel="e", series="width_error", index=i, x=x, y=y,
                            value=float(unc["rho"])))
        _write_delta(w, "f", "delta_mean", st_mean)
        _write_delta(w, "g", "delta_variance", st_var)


def write_html(svg_path, width=FIG_W, height=FIG_H):
    svg = svg_path.read_text()
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Figure 5 - Recurrent implementation and contact with neural data</title>
<style>
@page {{ size: {width}in {height}in; margin: 0; }}
html, body {{ margin: 0; padding: 0; background: white; }}
main {{ width: {width}in; height: {height}in; display: flex; align-items: center;
        justify-content: center; overflow: hidden; }}
svg {{ width: {width}in; height: {height}in; display: block; }}
</style></head><body><main>{svg}</main></body></html>"""
    OUT.with_suffix(".html").write_text(html)


def main():
    data = dict(np.load(DATA_PATH))
    diff = dict(np.load(DIFF_PATH))
    unc = dict(np.load(UNC_PATH))
    vdat = dict(np.load(VAR_PATH))
    states_gp = np.asarray(data["demo_states_froz"], float)
    states_rnn = np.asarray(data["demo_states_exp"], float)

    grid, src_xy, hat = conjunction_weights()
    rdm_r, mismatch = geometry_analysis(states_gp, states_rnn)
    st_e = diffusion_stats(diff)
    st_var = variance_stats(vdat)
    st_mean = delta_stats(vdat["mean_cued"], vdat["mean_neutral"],
                          int(vdat["cue_start"]), seed=5,
                          n_trials=int(vdat["n_trials"]))

    # row a (full width) | row b c d | row e f g
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    outer = fig.add_gridspec(
        3, 1, height_ratios=[1.22, 1.0, 1.05],
        left=0.078, right=0.988, top=0.97, bottom=0.068, hspace=0.48,
    )
    ax_a = fig.add_subplot(outer[0])

    mid = outer[1].subgridspec(1, 3, width_ratios=[1.35, 1.0, 1.05], wspace=0.50)
    cell_b = mid[0, 0].subgridspec(1, 2, width_ratios=[1.0, 0.07], wspace=0.10)
    ax_b = fig.add_subplot(cell_b[0, 0])
    cax_b = fig.add_subplot(cell_b[0, 1])
    ax_c = fig.add_subplot(mid[0, 1])
    ax_d = fig.add_subplot(mid[0, 2])

    bot = outer[2].subgridspec(1, 3, width_ratios=[1.15, 1.0, 1.0], wspace=0.62)
    ax_e = fig.add_subplot(bot[0, 0])
    ax_f = fig.add_subplot(bot[0, 1])
    ax_g = fig.add_subplot(bot[0, 2])

    panel_a(ax_a)
    panel_b(ax_b, cax_b, grid, src_xy, hat)
    err_stats = panel_c(ax_c, data)
    panel_e(ax_d, st_e)
    panel_f(ax_e, unc)
    panel_delta(ax_f, st_mean, r"$\Delta$ mean ($\times 10^{-3}$)",
                dash_neutral=True, legend_loc="lower left")
    panel_variance(ax_g, st_var)

    def _match_spine(src, dst):
        """Put dst's y-axis on the same vertical line as src. Keep dst's size."""
        s, d = src.get_position(), dst.get_position()
        dst.set_position([s.x0, d.y0, d.width, d.height])

    _match_spine(ax_b, ax_e)
    _match_spine(ax_c, ax_f)
    _match_spine(ax_d, ax_g)

    def _letter_x(ax, x):
        pos = ax.get_position()
        return pos.x0 + x * pos.width

    def _place(ax, letter, fig_x, y):
        pos = ax.get_position()
        ns.panel(ax, letter, x=(fig_x - pos.x0) / pos.width, y=y, size=FS_PANEL)

    col_be = min(_letter_x(ax_b, -0.28), _letter_x(ax_e, -0.28))
    col_cf = min(_letter_x(ax_c, -0.20), _letter_x(ax_f, -0.28))
    col_dg = min(_letter_x(ax_d, -0.22), _letter_x(ax_g, -0.28))
    _place(ax_a, "a", col_be, 0.99)
    _place(ax_b, "b", col_be, 1.04)
    _place(ax_e, "e", col_be, 1.08)
    _place(ax_c, "c", col_cf, 1.06)
    _place(ax_f, "f", col_cf, 1.08)
    _place(ax_d, "d", col_dg, 1.06)
    _place(ax_g, "g", col_dg, 1.08)

    write_source(
        grid, src_xy, hat, data, err_stats,
        st_e["ep"], st_e["disp"], unc, st_var, st_mean,
    )
    svg_path = OUT.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches=None, facecolor="white")
    fig.savefig(OUT.with_suffix(".png"), dpi=400, bbox_inches=None, facecolor="white")
    write_html(svg_path)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches=None, facecolor="white")
    plt.close(fig)
    print(
        f"saved fig5_circuit.html / .svg / .pdf / .png\n"
        f"  RDM r={rdm_r:.6f}, median mismatch={np.median(mismatch):.2e}, "
        f"recall r={err_stats['r']:.4f} (n={err_stats['n']} trials)\n"
        f"  diffusion rho={st_e['rho']:.3f} "
        f"[{st_e['lo']:.3f}, {st_e['hi']:.3f}], "
        f"mean drift {st_e['mean'][0]:.2f} -> {st_e['mean'][-1]:.2f} deg\n"
        f"  width rho={float(unc['rho']):.3f} "
        f"(Fisher {float(unc['rho_fisher']):.3f}), n={unc['errs'].size} items\n"
        f"  delta variance at cue {st_var['mean_c'][st_var['cue'] - 1]:.4f}; "
        f"end cued {st_var['mean_c'][-1]:.4f}  "
        f"neutral {st_var['mean_n'][-1]:.4f} (n={st_var['n']})\n"
        f"  delta mean at cue {st_mean['mean_c'][st_mean['cue'] - 1]:.6f}; "
        f"end cued {st_mean['mean_c'][-1]:.6f}  "
        f"neutral {st_mean['mean_n'][-1]:.6f}"
    )


if __name__ == "__main__":
    main()
