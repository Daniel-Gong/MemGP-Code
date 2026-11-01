"""
Figure 1 - Conceptual framework.

Claim: Visual working memory can be formulated as dynamic Bayesian inference
over a continuous function on the location x feature torus, implemented with a
sparse variational Gaussian process; the slot and resource accounts are limiting
projections of this single, mechanistic process.

Schematic-led composite (no data):
  a  reframing : two-item array -> noisy observations -> GP posterior
     mean surface on the location x color torus, colored by posterior s.d.
  b  SVGP machinery: 2D input (location x colour) -> 1D belief f(l,c);
     full belief vs. sparse inducing-point posterior
  c  descriptive -> mechanistic positioning of WM models
  d  the three-stage inference pipeline (encoding / maintenance / retrieval)
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LightSource, Normalize
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from figures import nature_style as ns

ns.set_style(9)
# genuine LaTeX-quality math (Computer Modern) for all $...$ equations, while
# axis/label text stays in the Arial sans family.
mpl.rcParams["mathtext.fontset"] = "cm"

# Same two items as Fig. 5a, so the reader can track one pair through the
# paper: (location, color) in degrees, plus the wheel color of that feature.
def _feature_hex(c_deg):
    rgb = np.clip(ns.color_to_rgb(float(c_deg)), 0.0, 1.0)
    return "#{:02x}{:02x}{:02x}".format(*(int(round(255 * v)) for v in rgb))


ITEMS = [
    (-72.0, 72.0, _feature_hex(72.0)),
    (72.0, -72.0, _feature_hex(-72.0)),
]

# Panel-a posterior. Shorter than the model's lambda = 10 so the two peaks
# stay separate at this panel size. Product of periodic kernels, period 360.
# GPyTorch form: exp(-2 sin^2(pi |d| / P) / lambda).
_LAM = 0.75
_NOISE = 0.04
_N_SURF = 64
_VIEW = (26, -58)
_PERIOD = 360.0
# Posterior standard deviation: deep blue where the observations pin f down
# (low s.d.), pale where the posterior stays near the prior (s.d. = 1).
_UNC_CMAP = LinearSegmentedColormap.from_list(
    "unc_sd",
    ["#08519C", "#2171B5", "#6BAED6", "#C6DBEF", "#F7FBFF"],
    N=256,
)
_UNC_VMIN, _UNC_VMAX = 0.0, 1.0


def _k_per(dx, lam=_LAM):
    return np.exp(-2.0 * np.sin(np.pi * np.abs(dx) / _PERIOD) ** 2 / lam)


def _k2(a, b):
    dloc = ns.circ_dist(a[:, None, 0], b[None, :, 0])
    dcol = ns.circ_dist(a[:, None, 1], b[None, :, 1])
    return _k_per(dloc) * _k_per(dcol)


def _observation_cloud(n_per=160, obs_sd=8.0, seed=2):
    """Noisy location-color samples around each item, with distance-decayed weights."""
    rng = np.random.default_rng(seed)
    groups, weights = [], []
    for loc, col, _ in ITEMS:
        jitter = rng.normal(0.0, obs_sd, size=(n_per, 2))
        pts = np.column_stack([
            (loc + jitter[:, 0] + 180.0) % 360.0 - 180.0,
            (col + jitter[:, 1] + 180.0) % 360.0 - 180.0,
        ])
        dist = np.hypot(ns.circ_dist(pts[:, 0], loc), ns.circ_dist(pts[:, 1], col))
        groups.append(pts)
        weights.append(np.exp(-0.5 * (dist / 12.0) ** 2))
    return groups, np.vstack(groups), np.concatenate(weights)


def _posterior_on_grid(X, y, n=_N_SURF):
    """Posterior mean and standard deviation of f on an n x n torus grid."""
    axis = np.linspace(-180.0, 180.0, n, endpoint=False)
    loc, col = np.meshgrid(axis, axis, indexing="ij")
    Xs = np.column_stack([loc.ravel(), col.ravel()])
    Kxx = _k2(X, X) + _NOISE * np.eye(len(X))
    Ksx = _k2(Xs, X)
    alpha = np.linalg.solve(Kxx, y)
    mean = (Ksx @ alpha).reshape(n, n)
    solved = np.linalg.solve(Kxx, Ksx.T)
    var = np.clip(1.0 - np.sum(Ksx * solved.T, axis=1), 0.0, None)
    return axis, mean, np.sqrt(var).reshape(n, n)


def _close_torus(axis, surf):
    axis_c = np.concatenate([axis, [axis[0] + _PERIOD]])
    surf_c = np.pad(np.asarray(surf, float), ((0, 1), (0, 1)), mode="wrap")
    return axis_c, surf_c


def _shade_uncertainty(z, unc):
    """Color by posterior s.d.; light the surface from the mean's slopes."""
    ls = LightSource(azdeg=315, altdeg=48)
    rgb = _UNC_CMAP(Normalize(_UNC_VMIN, _UNC_VMAX)(np.asarray(unc, float)))
    return ls.shade_rgb(
        rgb[..., :3], np.asarray(z, float), blend_mode="soft",
        vert_exag=1.15, fraction=0.65,
    )


def _dress_posterior(ax, z0, z1):
    """Axis treatment shared with the supplementary GP-prior surfaces."""
    ax.view_init(elev=_VIEW[0], azim=_VIEW[1])
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1.0, 1.0, 0.52), zoom=0.86)
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
    # Same native tick placement as the supplementary GP surfaces: both
    # ends of each axis, including the pair that meets at the near corner.
    ax.set_xticks([-180, 180])
    ax.set_yticks([-180, 180])
    ax.set_zticks([-1, 0, 1])
    ax.tick_params(labelsize=7.0, pad=1.5, length=3.5, width=0.8, colors=ns.INK)
    # Default 3D padding pushes the z labels toward the colorbar. A smaller
    # pad keeps -1, 0, 1 beside the spine without sitting on the line.
    ax.tick_params(axis="z", pad=-3.0)
    ax.set_xlabel(r"location $\ell$", fontsize=7.5, labelpad=1.5)
    ax.set_ylabel(r"color $c$", fontsize=7.5, labelpad=1.5)
    ax.set_zlabel("")
    pad = 14.0
    ax.set_xlim(-180.0 - pad, 180.0 + pad)
    ax.set_ylim(-180.0 - pad, 180.0 + pad)
    ax.set_zlim(z0, z1)
    ax.set_facecolor((1, 1, 1, 0))
    # Centered on the tip of the vertical axis, sitting just above it.
    ax.figure.canvas.draw()
    spine = ax.zaxis.line.get_xydata()
    top = spine[int(np.argmax(spine[:, 1]))]
    ax.text2D(
        top[0] - 0.008, top[1] + 0.002, r"memory strength $f$",
        transform=ax.transData, ha="center", va="bottom", fontsize=7.5,
        color=ns.INK, clip_on=False,
    )


def panel_a(ax):
    """Stimulus array, noisy observations, and the GP posterior over the torus.

    Left: two items on the display. Middle: location-color samples. Right:
    one posterior-mean surface. Height is the mean; color is the posterior
    standard deviation at every (location, color).
    """
    fig = ax.figure
    fig_w, fig_h = fig.get_size_inches()
    bb = ax.get_position()
    pw, ph = bb.width * fig_w, bb.height * fig_h

    ns.blank(ax, equal=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    groups, X, y = _observation_cloud()
    axis, mean, sd = _posterior_on_grid(X, y)

    # Physical layout along the strip. Sizes are inches, then converted to
    # the parent axes so the screen stays square. The Observe arrow stops
    # short of the scatter so the Color label has its own gap.
    pad_l, side = 0.02, min(1.52, ph * 0.84)
    gap_obs, gap_inf = 0.72, 0.50
    sc_w = side
    color_gap = 0.20
    # Middle axes run from -180 to the arrow tips at 215, inside ±245.
    # The display square matches that axis length. task_screen draws a
    # square of side 2.15 inside limits of ±1.15.
    axis_len = side * (215.0 - (-180.0)) / (245.0 - (-245.0))
    disp = axis_len * 2.30 / 2.15
    axis_mid = ((-180.0 + 215.0) / 2.0 - (-245.0)) / (245.0 - (-245.0))
    disp_y0 = 0.22 + side * axis_mid - disp / 2.0

    def ax_x(inches):
        return inches / pw

    def ax_y(inches):
        return inches / ph

    y_bot = ax_y(0.22)
    side_y = ax_y(side)
    sc_h = side_y
    y0 = y_bot

    screen = ax.inset_axes([ax_x(pad_l), ax_y(disp_y0), ax_x(disp), ax_y(disp)])
    ns.blank(screen, equal=True)
    screen.set_xlim(-1.15, 1.15)
    screen.set_ylim(-1.15, 1.15)
    ns.task_screen(screen, 0, 0, s=2.15, lw=1.35, z=2)
    ns.stim_items(
        screen, [(l, c) for l, c, _ in ITEMS], 0, 0,
        ring_r=0.72, sq=0.34, lw=0.7, z=5,
    )

    x_obs0 = pad_l + disp + gap_obs
    mid_y = y0 + 0.50 * side_y
    ns.arrow(ax, (ax_x(pad_l + disp + 0.05), mid_y),
             (ax_x(x_obs0 - color_gap - 0.02), mid_y), lw=1.15, ms=9)
    ax.text(ax_x(pad_l + disp + 0.5 * (gap_obs - color_gap)), mid_y + ax_y(0.11),
            "Observe", ha="center", va="bottom", fontsize=8.0, color=ns.INK)
    ax.text(ax_x(x_obs0 - 0.5 * color_gap), mid_y, "Color",
            rotation=90, ha="center", va="center", fontsize=8.0, color=ns.INK,
            rotation_mode="anchor")

    sc = ax.inset_axes([ax_x(x_obs0), y0, ax_x(sc_w), sc_h])
    sc.set_xlim(-245, 245)
    sc.set_ylim(-245, 245)
    sc.set_xticks([])
    sc.set_yticks([])
    for spine in sc.spines.values():
        spine.set_visible(False)
    sc.annotate(
        "", xy=(215, -180), xytext=(-180, -180),
        arrowprops=dict(arrowstyle="-|>", color=ns.INK, lw=1.05,
                        shrinkA=0, shrinkB=0),
    )
    sc.annotate(
        "", xy=(-180, 215), xytext=(-180, -180),
        arrowprops=dict(arrowstyle="-|>", color=ns.INK, lw=1.05,
                        shrinkA=0, shrinkB=0),
    )
    sc.text(18, -232, "Location", ha="center", va="top", fontsize=8.0,
            color=ns.INK, clip_on=False)
    # Tick marks at the axis ends, same labels as Figure 5b.
    tick = 14
    for xv in (-180, 180):
        sc.plot([xv, xv], [-180, -180 - tick], color=ns.INK, lw=0.8,
                clip_on=False, zorder=4, solid_capstyle="butt")
        sc.text(xv, -180 - tick - 6, f"{xv}", ha="center", va="top",
                fontsize=7.5, color=ns.INK, clip_on=False)
    for yv in (-180, 180):
        sc.plot([-180 - tick, -180], [yv, yv], color=ns.INK, lw=0.8,
                clip_on=False, zorder=4, solid_capstyle="butt")
        sc.text(-180 - tick - 6, yv, f"{yv}", ha="right", va="center",
                fontsize=7.5, color=ns.INK, clip_on=False)
    for pts, (_, _, col) in zip(groups, ITEMS):
        sc.scatter(pts[:, 0], pts[:, 1], s=3.5, color=col, linewidths=0,
                   zorder=3, clip_on=False)

    x_inf0 = x_obs0 + sc_w + gap_inf
    ns.arrow(ax, (ax_x(x_obs0 + sc_w + 0.06), mid_y),
             (ax_x(x_inf0 - 0.06), mid_y), lw=1.15, ms=9)
    ax.text(ax_x(x_obs0 + sc_w + 0.5 * gap_inf), mid_y + ax_y(0.11),
            "Infer", ha="center", va="bottom", fontsize=8.0, color=ns.INK)

    # 3-D posterior sits in figure coordinates so it can be a real 3-D axes.
    # The colorbar stays at the right margin; the surface is shifted left
    # so the z-axis is not tight against that bar.
    x3 = bb.x0 + ax_x(x_inf0) * bb.width
    y3 = bb.y0 + 0.01 * bb.height
    h3 = bb.height * 0.98
    cbar_w, cbar_gap = 0.012, 0.020
    cbar_left = 0.908
    # Same width as before; both edges move left, the colorbar does not.
    shift = 0.022
    x3 -= shift
    ax_right = cbar_left - cbar_gap - shift
    w3 = ax_right - x3
    ax3 = fig.add_axes([x3, y3, w3, h3], projection="3d")

    axis_c, mu = _close_torus(axis, mean)
    _, sd_c = _close_torus(axis, sd)
    Xg, Yg = np.meshgrid(axis_c, axis_c, indexing="ij")
    # Edges match the faces and are a fraction of a point wide so the
    # quad seams do not read as a wireframe on the dark field.
    faces = np.clip(_shade_uncertainty(mu, sd_c), 0.0, 1.0)
    ax3.plot_surface(
        Xg, Yg, mu, facecolors=faces, rcount=mu.shape[0], ccount=mu.shape[1],
        linewidth=0.35, antialiased=True, shade=False, rasterized=True,
    )
    # Item markers on the mean surface, same colors as the display.
    items = np.array([(l, c) for l, c, _ in ITEMS])
    Kxx = _k2(X, X) + _NOISE * np.eye(len(X))
    mu_i = _k2(items, X) @ np.linalg.solve(Kxx, y)
    ax3.computed_zorder = False
    for (loc, col, rgb), z in zip(ITEMS, mu_i):
        ax3.scatter(
            [loc], [col], [z + 0.08], s=42, color=rgb, edgecolors="white",
            linewidths=0.6, depthshade=False, zorder=10,
        )
    _dress_posterior(ax3, -1.80, 1.15)

    sm = mpl.cm.ScalarMappable(
        norm=Normalize(_UNC_VMIN, _UNC_VMAX), cmap=_UNC_CMAP,
    )
    sm.set_array([])
    cax = fig.add_axes([
        cbar_left,
        y3 + 0.22 * h3,
        cbar_w,
        0.52 * h3,
    ])
    cb = fig.colorbar(sm, cax=cax, ticks=[0, 0.5, 1])
    cb.ax.set_yticklabels(["0", "0.5", "1"])
    cb.ax.tick_params(labelsize=7.0, length=2.4, width=0.6, colors=ns.INK)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(ns.GREY_LT)
    cb.set_label(r"posterior s.d.", fontsize=7.5, labelpad=2)
    return ax3


# ---------------------------------------------------------------------------
# Typography shared with fig4a / fig5a schematic panels.
_FS_HEAD = 9.0
_FS_NOTE = 7.5
_FS_MIN = 7.0
_FS_EQ = 10.0


# Cued item in the maintenance / retrieval insets — same two stimuli as
# panel a (ITEMS), so the reader can track one bump through the pipeline.
_CUED = 0

# Feature-space divisive normalization at retrieval (Supplementary S1.5;
# src/config normalization block). Schematic color-bump width is chosen so
# the probed slice stays readable in the inset; the pool weights are the
# model's.
_SIGMA_COL = 24.0
_SIGMA_POOL = 60.0
_BASELINE_FRAC = 0.8


def _retrieval_readout():
    """Probed response, normalization pool, and normalized readout r(c).

    Each occupied location contributes a nonnegative color bump. Spatial
    weights decay with distance from the probe, the pool is their weighted
    sum, and the report is the argmax of the probed bump divided by that pool.
    Display scale: pool and probed response share one scale (peak of the pool
    at 0.72); r(c) is drawn at its own peak of 1 so the report sits above.
    """
    colors = np.linspace(-180.0, 180.0, 360)
    cue_l = ITEMS[_CUED][0]
    amps = [1.0 if i == _CUED else 0.30 for i in range(len(ITEMS))]
    slices = []
    weights = []
    for i, (loc, col, _) in enumerate(ITEMS):
        dcol = ns.circ_dist(colors, col)
        slices.append(amps[i] * np.exp(-0.5 * (dcol / _SIGMA_COL) ** 2))
        dloc = float(ns.circ_dist(loc, cue_l))
        weights.append(float(np.exp(-0.5 * (dloc / _SIGMA_POOL) ** 2)))
    slices = np.stack(slices, axis=0)
    pool = (slices * np.asarray(weights)[:, None]).sum(axis=0)
    target = slices[_CUED]
    sigma_b = _BASELINE_FRAC * float(target.max())
    readout = target / (sigma_b + pool)

    pool_scale = float(pool.max()) + 1e-12
    pool_n = 0.72 * pool / pool_scale
    target_n = 0.72 * target / pool_scale
    read_n = readout / (float(readout.max()) + 1e-12)
    k = int(np.argmax(readout))
    return colors, target_n, pool_n, read_n, k


def _stage_inset(ax, rect, kind):
    """Illustration for one inference stage (open schematic, no card chrome).

    Encoding and maintenance reuse the panel-a item set (locations + colours)
    so the belief maps read as the same two-item array through the figure.
    """
    iax = ax.inset_axes(rect)
    item_xy = [(l, c) for (l, c, _) in ITEMS]
    item_cols = [col for (_, _, col) in ITEMS]
    cue_l, cue_c = item_xy[_CUED]

    if kind in ("encode", "maintain"):
        # Same lengthscale / viridis rendering as panel c; maintenance keeps
        # the cued bump and lets the others decay under the KL term.
        amps = ([1.0] * len(ITEMS) if kind == "encode"
                else [1.0 if i == _CUED else 0.30 for i in range(len(ITEMS))])
        ell = np.degrees(0.44)  # matches panel c / panel a
        n = 140
        g = np.linspace(-180, 180, n)
        GL, GC = np.meshgrid(g, g, indexing="ij")
        grid = np.column_stack([GL.ravel(), GC.ravel()])
        raw = np.zeros(grid.shape[0])
        for (l, c), a in zip(item_xy, amps):
            d = grid - np.array([l, c])
            raw += a * np.exp(-0.5 * np.sum(d * d, axis=-1) / ell ** 2)
        lo, hi = float(raw.min()), float(raw.max())
        surf = np.clip(((raw - lo) / (hi - lo + 1e-12)).reshape(n, n), 0.0, 1.0)
        iax.imshow(
            surf.T, origin="lower", cmap="viridis", vmin=0.0, vmax=1.0,
            aspect="equal", extent=[-180, 180, -180, 180],
            interpolation="bilinear", rasterized=True,
        )
        iax.set_xlim(-180, 180)
        iax.set_ylim(-180, 180)
        iax.set_aspect("equal", adjustable="box")
        iax.set_xticks([-180, 0, 180])
        iax.set_yticks([-180, 0, 180])
        iax.tick_params(labelsize=_FS_MIN, length=2.0, pad=1.0)
        iax.set_xlabel(r"location $\ell$", fontsize=_FS_MIN, labelpad=1.0)
        if kind == "maintain":
            iax.axvline(cue_l, color="#E6E6E6", lw=0.7,
                        ls=(0, (2.2, 1.6)), zorder=4)
            iax.annotate(
                "cue", xy=(cue_l, 180), xytext=(0, 1.5),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=_FS_MIN, color=ns.GREY_DK,
                clip_on=False, annotation_clip=False, zorder=6,
            )
        if kind == "encode":
            iax.set_ylabel(r"color $c$", fontsize=_FS_MIN, labelpad=1.0)
        else:
            # Maintenance sits directly beside Encoding, which already carries
            # the colour-axis scale; dropping its own tick labels here frees
            # the room the "delay" arrow label needs, avoiding overlap.
            iax.set_yticklabels([])
        for sp in iax.spines.values():
            sp.set_linewidth(0.7)
            sp.set_color(ns.GREY_DK)
        for (l, c), col in zip(item_xy, item_cols):
            iax.scatter(l, c, marker="*", s=36, color=col,
                        edgecolors="white", lw=0.55, zorder=5)
    else:  # retrieve — divisive normalization of a joint color sample (S1.5)
        colors, target_n, pool_n, read_n, k = _retrieval_readout()
        iax.fill_between(colors, 0.0, pool_n, color=ns.GREY_LT, linewidth=0,
                         zorder=2)
        iax.plot(colors, target_n, color=ns.INK, lw=1.05,
                 ls=(0, (2.2, 1.5)), zorder=3)
        iax.plot(colors, read_n, color=ns.MEM, lw=1.7, zorder=4)
        iax.axvline(colors[k], color=ns.CUE, lw=1.1, ls=(0, (2.4, 1.8)),
                    zorder=5)
        iax.plot([colors[k]], [read_n[k]], marker="v", ms=5.0,
                 color=ns.CUE, markeredgecolor="white", markeredgewidth=0.55,
                 zorder=6)
        iax.set_xlim(-180, 180)
        iax.set_ylim(0, 1.22)
        iax.set_xticks([-180, 0, 180])
        iax.set_yticks([])
        iax.tick_params(labelsize=_FS_MIN, length=2.0, pad=1.0)
        iax.set_xlabel(r"color $c$", fontsize=_FS_MIN, labelpad=1.0)
        for side in ("top", "right", "left"):
            iax.spines[side].set_visible(False)
        iax.spines["bottom"].set_linewidth(0.7)
        iax.spines["bottom"].set_color(ns.GREY_DK)
        # Keep the "report" tag inside the axes so it cannot collide with the
        # operator equation drawn in the parent panel.
        # Place the tag on the side with more room relative to the report peak.
        tag_x = colors[k] + 14 if colors[k] < 100 else colors[k] - 14
        tag_ha = "left" if colors[k] < 100 else "right"
        iax.text(tag_x, 1.08, "report", ha=tag_ha, va="center",
                 fontsize=_FS_MIN, color=ns.CUE, clip_on=False)
        # Strongest non-target lobe of the pool, so the fill reads as the
        # sum over items rather than a band around the probed bump.
        away = ns.circ_dist(colors, cue_c) > 70.0
        if np.any(away):
            j = int(np.argmax(np.where(away, pool_n, -1.0)))
            if pool_n[j] > 0.05:
                iax.text(colors[j], pool_n[j] + 0.05, "pool",
                         ha="center", va="bottom", fontsize=_FS_MIN,
                         color=ns.GREY_DK)
    return iax


def _torus_axis(a, ylabel=True):
    """Minimal square axis for a (location, colour) belief map."""
    a.set_xlim(-180, 180); a.set_ylim(-180, 180)
    a.set_aspect("equal")
    a.set_xticks([-180, 0, 180]); a.set_yticks([-180, 0, 180])
    a.set_xlabel(r"location $\ell$", fontsize=6.4, labelpad=1.5)
    if ylabel:
        a.set_ylabel(r"color $c$", fontsize=6.4, labelpad=1.5)
    else:
        a.set_yticklabels([])
    a.tick_params(labelsize=5.6, length=2.2, pad=1.5)
    for sp in a.spines.values():
        sp.set_color(ns.GREY_LT); sp.set_linewidth(0.9)


def _bump_surface(pts, centers, ell):
    """Sum of isotropic RBF bumps (Euclidean) at the panel-a item coordinates."""
    diff = pts[:, None, :] - centers[None, :, :]
    sq = np.sum(diff * diff, axis=-1)
    return np.exp(-0.5 * sq / ell ** 2).sum(axis=1)


def panel_b(ax):
    """Why a sparse variational GP? The belief surface, with vs without
    inducing points.

    Both panels show the two-item belief f(l,c) over the location x colour
    domain (same item locations as panel a) -- bright = strong memory, dark = none.

    left  - Gaussian process: the full, continuous belief over every (l, c).
    right - Sparse variational GP: a value lives only at each of the M = 100
            inducing points (Z) -- a bounded budget, the model's analogue of
            limited working-memory resources.
    """
    ns.blank(ax, equal=False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    n = 200
    g = np.linspace(-180, 180, n)
    GL, GC = np.meshgrid(g, g, indexing="ij")
    grid = np.column_stack([GL.ravel(), GC.ravel()])

    centers = np.array([(l, c) for (l, c, _) in ITEMS])
    item_cols = [col for (_, _, col) in ITEMS]
    ell = np.degrees(0.44)  # 25.2 deg

    # two-item surface, normalised to [0, 1]
    raw = _bump_surface(grid, centers, ell)
    lo, hi = float(raw.min()), float(raw.max())
    surf = np.clip(((raw - lo) / (hi - lo)).reshape(n, n), 0.0, 1.0)

    # 10x10 inducing grid: the SVGP carries a value only at each point
    side = np.linspace(-162, 162, 10)
    ZL, ZC = np.meshgrid(side, side, indexing="ij")
    Z = np.column_stack([ZL.ravel(), ZC.ravel()])
    mZ = np.clip((_bump_surface(Z, centers, ell) - lo) / (hi - lo), 0.0, 1.0)

    ext = [-180, 180, -180, 180]
    cmap = "viridis"

    def _stars(a):
        for (l, c), col in zip(centers, item_cols):
            a.scatter(l, c, marker="*", s=58, color=col,
                      edgecolors="white", lw=0.7, zorder=6)

    # ---- left: full GP -- a continuous belief over every (l, c) -------------
    ax_gp = ax.inset_axes([0.085, 0.30, 0.37, 0.62])
    _torus_axis(ax_gp, ylabel=True)
    ax_gp.imshow(surf.T, origin="lower", extent=ext, cmap=cmap, vmin=0.0,
                 vmax=1.0, interpolation="bilinear", zorder=1, rasterized=True)
    _stars(ax_gp)
    ax_gp.text(0.0, 1.04, r"memory strength $f$",
               transform=ax_gp.transAxes, fontsize=5.8, color=ns.INK,
               va="bottom", ha="left", clip_on=False)

    # ---- right: SVGP -- a value lives *only* at each of the M inducing points
    ax_sv = ax.inset_axes([0.575, 0.30, 0.37, 0.62])
    _torus_axis(ax_sv, ylabel=False)
    ax_sv.set_facecolor(ns.GREY_FILL)
    ax_sv.scatter(Z[:, 0], Z[:, 1], c=mZ, cmap=cmap, vmin=0.0, vmax=1.0,
                  marker="s", s=64, edgecolors="white", lw=0.3, zorder=3)
    _stars(ax_sv)
    ax_sv.text(0.0, 1.04, r"$M=100$ inducing points $\mathbf{Z}$",
               transform=ax_sv.transAxes, fontsize=5.8, color=ns.INK,
               va="bottom", ha="left", clip_on=False)

    # ---- connecting arrow, centered on the two maps ----------------------
    mid_y = 0.30 + 0.62 / 2
    ns.arrow(ax, (0.455, mid_y), (0.575, mid_y), lw=1.8, ms=12)
    ax.text(0.515, mid_y + 0.045, "sparsify", ha="center", va="bottom",
            fontsize=6.2, color=ns.GREY_DK, style="italic")

    # ---- shared predictive-mean colorbar ------------------------------------
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0.0, 1.0), cmap=cmap)
    cax = ax.inset_axes([0.405, 0.235, 0.19, 0.026])
    cb = ax.figure.colorbar(sm, cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.6); cb.outline.set_edgecolor(ns.GREY_LT)
    cb.set_ticks([])
    cax.text(-0.04, 0.5, "none", transform=cax.transAxes, fontsize=5.0,
             color=ns.GREY_DK, ha="right", va="center")
    cax.text(1.04, 0.5, "strong", transform=cax.transAxes, fontsize=5.0,
             color=ns.GREY_DK, ha="left", va="center")
    cax.text(0.5, -1.3, "predictive mean",
             transform=cax.transAxes, fontsize=5.2, color=ns.GREY_DK,
             ha="center", va="top")


# ---------------------------------------------------------------------------
def panel_c(ax):
    """MemGP positioned in the descriptive -> mechanistic / scope plane."""
    L = 10.0
    ax.set_xlim(-0.4, L + 0.8); ax.set_ylim(-0.4, L + 0.8)
    for s in ax.spines.values():
        s.set_visible(False)
    # axes drawn as arrows that meet exactly at the origin (shrink=0 so the
    # two tails are not pulled back, leaving no gap at the corner)
    ax.annotate("", xy=(L + 0.7, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=ns.INK, lw=1.2,
                                shrinkA=0, shrinkB=0),
                zorder=3)
    ax.annotate("", xy=(0, L + 0.7), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=ns.INK, lw=1.2,
                                shrinkA=0, shrinkB=0),
                zorder=3)
    ax.set_xticks([1.4, 8.6]); ax.set_xticklabels(["descriptive", "mechanistic"])
    ax.set_yticks([1.4, 8.6]); ax.set_yticklabels(["single\nphenomenon", "unified"],
                                                  rotation=90, va="center")
    ax.tick_params(length=0)
    ax.set_xlabel("Level of explanation")
    ax.set_ylabel("Scope")

    pts = [
        ("Slot",        1.7, 2.0, ns.SLOT),
        ("Resource",    3.0, 3.2, ns.RESOURCE),
        ("Variable\nprecision", 4.2, 4.5, ns.darken(ns.RESOURCE, 0.15)),
        ("Population\ncoding", 7.0, 4.2, ns.GREY_DK),
        ("Sampling",    5.4, 5.4, ns.GREY),
        ("RNN\nmodels", 7.8, 2.0, "#C2185B"),
    ]
    for name, x, y, c in pts:
        ax.scatter([x], [y], s=42, color=c, zorder=4, edgecolors="white", lw=0.8)
        ax.text(x, y - 0.55, name, ha="center", va="top", fontsize=6.8, color=c)
    ax.scatter([8.6], [8.7], s=150, marker="*", color=ns.MEM, zorder=6,
               edgecolors="white", lw=1.0)
    ax.text(8.6, 7.9, "MemGP", ha="center", va="top", fontsize=8.8,
            fontweight="bold", color=ns.MEM)


# ---------------------------------------------------------------------------
def panel_d(ax):
    """Three-stage inference pipeline — open schematic (fig4a / fig5a style).

    Equal visual columns: bold stage title + grey subtitle, illustration,
    operator equation, then a pale objective card. Arrows carry the flow.
    No coloured header bars or full-height stage frames.
    """
    ns.blank(ax, equal=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Pale fills match fig5a's bottom concept strip.
    stages = [
        dict(
            title="Encoding",
            sub="fit posterior to observations",
            col=ns.MEM,
            fill=ns.MEM_PALE,
            cx=0.155,
            kind="encode",
            general=r"$f:\ X \mapsto y$",
            card_title=r"negative ELBO",
            card_body=(
                r"$\mathcal{L}_{\mathrm{enc}}"
                r"=-\mathbb{E}_{q}[\log p(y\,|\,f)]$"
                "\n"
                r"$+\mathrm{KL}(q\,\|\,p)$"
            ),
        ),
        dict(
            title="Maintenance",
            sub="rehearse under forgetting pressure",
            col=ns.RESOURCE,
            fill="#D8EEEA",
            cx=0.500,
            kind="maintain",
            general=r"$f_t\ \mapsto\ f_{t+1}$",
            card_title="attention-weighted ELBO",
            card_body=(
                r"$\mathcal{L}_{\mathrm{mnt}}=-\!\sum_m a(\ell_m)"
                r"\,\mathbb{E}_{q}[\log p(\tilde y_m\,|\,f)]$"
                "\n"
                r"$+\ \beta\,\mathrm{KL}(q\,\|\,p)$"
            ),
            card_fs=_FS_MIN - 0.3,
        ),
        dict(
            title="Retrieval",
            sub="normalize and report the peak",
            col=ns.CUE,
            fill="#F8E8EB",
            cx=0.845,
            kind="retrieve",
            general=r"$\hat{c}=\arg\max_c\, r(c)$",
            card_title="divisive normalization",
            card_body=(
                r"$r(c)=f_t^{+}(c)\,/\,(\sigma_b\max f_t^{+}+\mathrm{pool})$"
                "\n"
                r"$\mathrm{pool}(c)=\sum_i w_i\, f_i^{+}(c)$"
            ),
            card_fs=_FS_MIN - 0.6,
        ),
    ]

    # Belief-map insets must be physically square (panel-c style). Panel d is
    # wide and short, so equal axes-fractions would stretch; convert a target
    # height into a matching width in inches. Retrieval stays wider (1-D plot).
    fig_w, fig_h = ax.figure.get_size_inches()
    bbox = ax.get_position()
    parent_w_in = bbox.width * fig_w
    parent_h_in = bbox.height * fig_h
    inset_h = 0.335
    inset_w_sq = (inset_h * parent_h_in) / parent_w_in
    inset_w_ret = max(inset_w_sq, 0.21)
    inset_y = 0.48
    card_w, card_h = 0.29, 0.165
    card_y = 0.091
    arrow_y = inset_y + 0.5 * inset_h

    for st in stages:
        cx = st["cx"]
        col = st["col"]
        inset_w = inset_w_sq if st["kind"] != "retrieve" else inset_w_ret

        # Title / subtitle — same hierarchy as fig4a mechanism headers.
        ax.text(cx, 0.965, st["title"], ha="center", va="center",
                fontsize=_FS_HEAD, fontweight="bold", color=ns.INK)
        ax.text(cx, 0.900, st["sub"], ha="center", va="center",
                fontsize=_FS_MIN, color=ns.GREY_DK)

        _stage_inset(
            ax,
            [cx - inset_w / 2, inset_y, inset_w, inset_h],
            st["kind"],
        )

        # Operator under the visual, clear of inset tick/axis labels.
        ax.text(cx, 0.31, st["general"], ha="center", va="center",
                fontsize=_FS_EQ, color=col, fontweight="bold")

        # Pale objective card — fig5a bottom strip.
        bx = cx - card_w / 2
        ax.add_patch(
            FancyBboxPatch(
                (bx, card_y), card_w, card_h,
                boxstyle="round,pad=0.008,rounding_size=0.02",
                fc=st["fill"], ec=col, lw=1.0, zorder=3,
            )
        )
        ax.text(cx, card_y + card_h - 0.026, st["card_title"],
                ha="center", va="center", fontsize=_FS_NOTE,
                fontweight="bold", color=col, zorder=4)
        ax.text(cx, card_y + 0.048, st["card_body"],
                ha="center", va="center",
                fontsize=st.get("card_fs", _FS_MIN),
                color=ns.GREY_DK, zorder=4, linespacing=1.25)

    # Flow arrows between columns; endpoints track the square-map edges.
    c0, c1, c2 = stages[0]["cx"], stages[1]["cx"], stages[2]["cx"]
    gap = 0.012
    ns.arrow(ax, (c0 + inset_w_sq / 2 + gap, arrow_y),
             (c1 - inset_w_sq / 2 - gap, arrow_y), lw=1.6, ms=12)
    ax.text(0.5 * (c0 + c1), arrow_y + 0.032, "delay", ha="center",
            va="bottom", fontsize=_FS_MIN, color=ns.GREY_DK)
    x_probe0 = c1 + inset_w_sq / 2 + gap
    x_probe1 = c2 - inset_w_ret / 2 - gap
    ns.arrow(ax, (x_probe0, arrow_y), (x_probe1, arrow_y), lw=1.6, ms=12)
    ax.text(0.5 * (x_probe0 + x_probe1), arrow_y + 0.032, "probe", ha="center",
            va="bottom", fontsize=_FS_MIN, color=ns.GREY_DK)


# ---------------------------------------------------------------------------
def main():
    fig_w, fig_h = 7.2, 8.4
    left, right, top_m, bottom = 0.07, 0.975, 0.965, 0.05
    plot_w = right - left
    # Panel a is a wide strip: screen, observation cloud, 3-D posterior.
    h_a = 2.08 / fig_h
    gap_ab = 0.012  # small breathing room below panel a

    fig = plt.figure(figsize=(fig_w, fig_h))
    y_a = top_m - h_a - 0.02
    ax_a = fig.add_axes([left, y_a, plot_w, h_a])

    bot_top = y_a - gap_ab
    gs = fig.add_gridspec(2, 6, height_ratios=[1.18, 1.18],
                          hspace=0.06, wspace=0.55,
                          left=left, right=right, top=bot_top, bottom=bottom)

    # panel c on the left, panel b on the right (swapped from original)
    ax_c = fig.add_subplot(gs[0, 0:3])
    ax_b = fig.add_subplot(gs[0, 3:6])
    ax_d = fig.add_subplot(gs[1, :])

    # Flatten panel c and align its box with panel b's map area (which only
    # occupies the y-band [0.30, 0.92] of the cell), so it is no longer tall.
    pb, pc = ax_b.get_position(), ax_c.get_position()
    map_lo, map_hi = 0.30, 0.92
    ax_c.set_position([pc.x0, pb.y0 + map_lo * pb.height,
                       pc.width, (map_hi - map_lo) * pb.height])

    # The b/c cell carries ~0.20 of dead band below its maps (colourbar caption
    # / x-label sit around cell fraction 0.19), so the gridspec gap read far
    # larger than the a -> b/c one. Pull panel d's top up to close it; the
    # panel's content is laid out in axes fractions, so it simply scales.
    pd = ax_d.get_position()
    d_top = 0.385
    ax_d.set_position([pd.x0, bottom, pd.width, d_top - bottom])

    panel_a(ax_a)
    panel_b(ax_b)
    panel_c(ax_c)
    panel_d(ax_d)

    def _fig_x(ax, x):
        pos = ax.get_position()
        return pos.x0 + x * pos.width

    def _place(ax, letter, fig_x, y):
        pos = ax.get_position()
        ns.panel(ax, letter, x=(fig_x - pos.x0) / pos.width, y=y)

    # a, b, and d share one vertical line. b is the reference.
    col = _fig_x(ax_c, -0.16)
    _place(ax_a, "a", col, 1.02)
    _place(ax_c, "b", col, 1.04)
    ns.panel(ax_b, "c", x=0.06, y=0.945)
    _place(ax_d, "d", col, 1.0)

    ns.save_fig(fig, "fig1_concept")


if __name__ == "__main__":
    main()
