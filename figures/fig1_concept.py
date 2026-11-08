"""
Figure 1 - Conceptual framework.

Claim: Visual working memory can be formulated as dynamic Bayesian inference
over a continuous function on the location x feature torus, implemented with a
sparse variational Gaussian process; the slot and resource accounts are limiting
projections of this single, mechanistic process.

Schematic-led composite (no data):
  a  two-item display connected to the GP posterior mean surface on the
     location x color torus, colored by posterior s.d.
  b  full belief vs. sparse inducing-point posterior
  c  the three-stage inference pipeline (encoding / maintenance / retrieval)
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LightSource, Normalize
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


def _dress_posterior(ax, z0, z1, zoom=0.86):
    """Axis treatment shared with the supplementary GP-prior surfaces."""
    ax.view_init(elev=_VIEW[0], azim=_VIEW[1])
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1.0, 1.0, 0.52), zoom=zoom)
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
    # Label pads clear the corner tick numbers on this smaller surface.
    ax.set_xlabel(r"location $\ell$", fontsize=7.5, labelpad=6)
    ax.set_ylabel(r"color $c$", fontsize=7.5, labelpad=4)
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
    # Sit the label on the surface side of the vertical axis so it does not
    # run into the inducing-point maps on the right.
    _place_runs(
        ax, top[0] + 0.045, top[1] + 0.010,
        [("memory strength ", ns.INK), (r"$f$", ns.MEM)],
        fontsize=7.5, transform=ax.transData, ha="right", va="baseline",
        dys=[0.0, -0.14],
    )


def panel_a(ax, disp):
    """Stimulus display connected directly to the GP posterior over the torus.

    Left: two items on the display. Right: one posterior-mean surface.
    Height is the mean; color is the posterior standard deviation at every
    (location, color). One arrow, labeled observe and infer, joins them.
    """
    fig = ax.figure
    fig_w, fig_h = fig.get_size_inches()
    bb = ax.get_position()
    pw, ph = bb.width * fig_w, bb.height * fig_h

    ns.blank(ax, equal=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # Same baseline and the same gap after the letter as panel c.
    # Panel c's letter sits 0.015 of its (wider) axes left of the title.
    ax.text(
        0.1011 / pw - 0.015, 0.894,
        r"Encoding as inference, $p(f \mid \mathbf{X},\mathbf{y})$",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=10,
        fontweight="bold", color=ns.INK, clip_on=False,
    )

    _, X, y = _observation_cloud()
    axis, mean, sd = _posterior_on_grid(X, y)

    # Inches within the parent axes. disp matches the panel-c screen.
    # A horizontal colorbar under the surface keeps width for the display,
    # the arrow, and the field itself.
    pad_l = 0.02
    gap = 0.96
    cbar_bottom = 0.32
    cbar_h = 0.070
    cbar_gap = 0.18
    top_pad = 0.02
    y3_in = cbar_bottom + cbar_h + cbar_gap
    h3_in = ph - y3_in - top_pad
    y_disp = y3_in + (h3_in - disp) / 2.0

    def ax_x(inches):
        return inches / pw

    def ax_y(inches):
        return inches / ph

    _display_inset(ax, [ax_x(pad_l), ax_y(y_disp), ax_x(disp), ax_y(disp)])

    x_arrow0 = pad_l + disp + 0.05
    x_arrow1 = pad_l + disp + gap - 0.05
    mid_y = ax_y(y_disp + disp / 2.0)
    ns.arrow(ax, (ax_x(x_arrow0), mid_y), (ax_x(x_arrow1), mid_y),
             lw=1.15, ms=9)
    ax.text(ax_x(0.5 * (x_arrow0 + x_arrow1)), mid_y + ax_y(0.09),
            "observe and infer", ha="center", va="bottom",
            fontsize=7.4, color=ns.INK)

    # 3-D posterior sits in figure coordinates so it can be a real 3-D axes.
    # Right pad keeps the color-axis label inside this panel.
    x3_in = pad_l + disp + gap
    right_pad_in = 0.28
    x3 = bb.x0 + (x3_in / pw) * bb.width
    y3 = bb.y0 + (y3_in / ph) * bb.height
    w3 = bb.x1 - x3 - right_pad_in / fig_w
    h3 = (h3_in / ph) * bb.height
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
    _dress_posterior(ax3, -1.80, 1.15, zoom=1.08)

    sm = mpl.cm.ScalarMappable(
        norm=Normalize(_UNC_VMIN, _UNC_VMAX), cmap=_UNC_CMAP,
    )
    sm.set_array([])
    cax = fig.add_axes([
        x3 + 0.18 * w3,
        bb.y0 + (cbar_bottom / ph) * bb.height,
        0.64 * w3,
        (cbar_h / ph) * bb.height,
    ])
    cb = fig.colorbar(
        sm, cax=cax, orientation="horizontal", ticks=[0, 0.5, 1],
    )
    cb.ax.tick_params(labelsize=7.0, length=2.4, width=0.6, colors=ns.INK)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(ns.GREY_LT)
    cb.set_label(r"memory uncertainty (posterior s.d.)", fontsize=7.5, labelpad=1)
    return cax


# ---------------------------------------------------------------------------
# Typography shared with fig4a / fig5a schematic panels.
_FS_HEAD = 9.0
_FS_MIN = 7.0
_FS_EQ = 10.0


def _place_runs(ax, x, y, runs, fontsize, transform, ha="center", va="center",
                weight="normal", dys=None):
    """Draw colored text runs as one phrase aligned at (x, y).

    dys: per-run vertical shift in em. Negative moves that run down.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = transform.inverted()
    draw = ax.text2D if hasattr(ax, "text2D") else ax.text
    if dys is None:
        dys = [0.0] * len(runs)
    texts, widths, yoffs = [], [], []
    for s, color, dy_em in zip((r[0] for r in runs), (r[1] for r in runs), dys):
        t = draw(
            x, y, s, fontsize=fontsize, color=color, fontweight=weight,
            ha="left", va=va, transform=transform, clip_on=False,
        )
        texts.append(t)
        bb = t.get_window_extent(renderer)
        p0 = inv.transform((0.0, 0.0))
        p1 = inv.transform((bb.width, 0.0))
        widths.append(p1[0] - p0[0])
        dy_px = dy_em * fontsize * fig.dpi / 72.0
        yoffs.append(inv.transform((0.0, dy_px))[1] - p0[1])
    total = sum(widths)
    if ha == "center":
        cursor = x - 0.5 * total
    elif ha == "right":
        cursor = x - total
    else:
        cursor = x
    for t, w, dy in zip(texts, widths, yoffs):
        t.set_position((cursor, y + dy))
        cursor += w


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
        # Same 10x10 inducing grid and color scale as the right-hand map
        # in panel b. Maintenance keeps the cued bump and lets the others decay.
        ell = np.degrees(0.44)  # matches panel b
        centers = np.array(item_xy)
        dense = np.linspace(-180.0, 180.0, 200)
        GL, GC = np.meshgrid(dense, dense, indexing="ij")
        field = _bump_surface(np.column_stack([GL.ravel(), GC.ravel()]), centers, ell)
        lo, hi = float(field.min()), float(field.max())
        grid_axis = np.linspace(-162, 162, 10)
        ZL, ZC = np.meshgrid(grid_axis, grid_axis, indexing="ij")
        Z = np.column_stack([ZL.ravel(), ZC.ravel()])
        if kind == "encode":
            raw = _bump_surface(Z, centers, ell)
        else:
            amps = [1.0 if i == _CUED else 0.30 for i in range(len(ITEMS))]
            raw = np.zeros(Z.shape[0])
            for (l, c), a in zip(item_xy, amps):
                d = Z - np.array([l, c])
                raw += a * np.exp(-0.5 * np.sum(d * d, axis=-1) / ell ** 2)
        mZ = np.clip((raw - lo) / (hi - lo), 0.0, 1.0)
        iax.set_facecolor(ns.GREY_FILL)
        iax.scatter(
            Z[:, 0], Z[:, 1], c=mZ, cmap="viridis", vmin=0.0, vmax=1.0,
            marker="s", s=64, edgecolors="white", lw=0.3, zorder=3,
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
                "(optional) cue", xy=(cue_l, 180), xytext=(0, 1.5),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=_FS_MIN, color=ns.CUE,
                clip_on=False, annotation_clip=False, zorder=6,
            )
        if kind == "encode":
            # Sit above the arrow that joins the display to this map.
            iax.set_ylabel("")
            iax.text(
                -0.36, 0.84, r"color $c$", transform=iax.transAxes,
                rotation=90, ha="center", va="center", fontsize=_FS_MIN,
                color=ns.INK, clip_on=False,
            )
        else:
            # Maintenance sits directly beside Encoding, which already carries
            # the colour-axis scale.
            iax.set_yticklabels([])
        for sp in iax.spines.values():
            sp.set_linewidth(1.35)
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
        iax.plot(colors, read_n, color=ns.CUE, lw=1.7, zorder=4)
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


def panel_b(ax, cbar_a, map_in):
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

    # 10x10 inducing grid: the SVGP carries a value only at each point.
    # Colors use the same scale as the continuous surface.
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

    # Square maps in inches, so the pair stays square in a short top-row cell.
    fig_w, fig_h = ax.figure.get_size_inches()
    bb = ax.get_position()
    pw, ph = bb.width * fig_w, bb.height * fig_h
    # Same baseline and the same gap after the letter as panel c.
    ax.text(
        0.1011 / pw - 0.015, 0.908, "Approximation with a shared resource",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=10,
        fontweight="bold", color=ns.INK, clip_on=False,
    )
    margin_l, gap_maps = 0.36, 0.40
    # Match the vertical center of panel a's display and memory field.
    y_map = 0.58 * ph - map_in / 2.0

    def fx(inches):
        return inches / pw

    def fy(inches):
        return inches / ph

    # ---- left: full GP -- a continuous belief over every (l, c) -------------
    ax_gp = ax.inset_axes([
        fx(margin_l), fy(y_map), fx(map_in), fy(map_in),
    ])
    _torus_axis(ax_gp, ylabel=True)
    ax_gp.imshow(surf.T, origin="lower", extent=ext, cmap=cmap, vmin=0.0,
                 vmax=1.0, interpolation="bilinear", zorder=1, rasterized=True)
    _stars(ax_gp)
    _place_runs(
        ax_gp, 0.0, 1.06,
        [("memory strength ", ns.INK), (r"$f$", ns.MEM)],
        fontsize=5.8, transform=ax_gp.transAxes, ha="left", va="baseline",
        dys=[0.0, -0.14],
    )

    # ---- right: SVGP -- a value lives *only* at each of the M inducing points
    x_sv = margin_l + map_in + gap_maps
    ax_sv = ax.inset_axes([fx(x_sv), fy(y_map), fx(map_in), fy(map_in)])
    _torus_axis(ax_sv, ylabel=False)
    ax_sv.set_facecolor(ns.GREY_FILL)
    ax_sv.scatter(Z[:, 0], Z[:, 1], c=mZ, cmap=cmap, vmin=0.0, vmax=1.0,
                  marker="s", s=64, edgecolors="white", lw=0.3, zorder=3)
    _stars(ax_sv)
    ax_sv.text(0.0, 1.06, "inducing representation",
               transform=ax_sv.transAxes, fontsize=5.8, color=ns.INK,
               va="bottom", ha="left", clip_on=False)

    # ---- connecting arrow, centered on the two maps ----------------------
    mid_y = fy(y_map + map_in / 2.0)
    ns.arrow(ax, (fx(margin_l + map_in + 0.05), mid_y),
             (fx(x_sv - 0.05), mid_y), lw=1.6, ms=11)
    ax.text(fx(margin_l + map_in + gap_maps / 2.0), mid_y + fy(0.07),
            "sparsify", ha="center", va="bottom",
            fontsize=6.2, color=ns.GREY_DK, style="italic")

    # Same bar length, thickness, and vertical position as panel a.
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0.0, 1.0), cmap=cmap)
    ref = cbar_a.get_position()
    maps_cx = bb.x0 + ((margin_l + map_in + gap_maps / 2.0) / pw) * bb.width
    cax = ax.figure.add_axes([
        maps_cx - ref.width / 2.0, ref.y0, ref.width, ref.height,
    ])
    cb = ax.figure.colorbar(sm, cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(ns.GREY_LT)
    cb.set_ticks([])
    cb.ax.tick_params(labelsize=7.0, length=2.4, width=0.6, colors=ns.INK)
    cax.text(-0.04, 0.5, "none", transform=cax.transAxes, fontsize=7.0,
             color=ns.GREY_DK, ha="right", va="center")
    cax.text(1.04, 0.5, "strong", transform=cax.transAxes, fontsize=7.0,
             color=ns.GREY_DK, ha="left", va="center")
    cax.text(0.5, -0.55, "memory strength",
             transform=cax.transAxes, fontsize=7.5, color=ns.GREY_DK,
             ha="center", va="top")


# ---------------------------------------------------------------------------
def _display_inset(ax, rect):
    """Panel-a stimulus screen, scaled to the given axes rectangle."""
    screen = ax.inset_axes(rect)
    ns.blank(screen, equal=True)
    screen.set_xlim(-1.0, 1.0)
    screen.set_ylim(-1.0, 1.0)
    # Frame fills the inset so the screen matches the encoding heatmap box.
    # Item ring and marker keep their size relative to that frame.
    ns.task_screen(screen, 0, 0, s=2.0, lw=1.35, z=2)
    ns.stim_items(
        screen, [(l, c) for l, c, _ in ITEMS], 0, 0,
        ring_r=0.72 / 1.075, sq=0.34 * (2.0 / 2.15), lw=0.7, z=5,
    )
    return screen


def panel_c(ax, side):
    """Display plus the three inference stages.

    The screen and the two belief maps match the panel-b squares. The
    screen sits to the left of encoding. Each stage has a title, the
    illustration, and the operator equation. Short arrows link the display
    to encoding, encoding to maintenance, and maintenance to retrieval.
    """
    ns.blank(ax, equal=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig_w, fig_h = ax.figure.get_size_inches()
    bbox = ax.get_position()
    pw = bbox.width * fig_w
    ph = bbox.height * fig_h

    def fx(inches):
        return inches / pw

    def fy(inches):
        return inches / ph

    # side is the panel-b square, in inches. Stage gaps stay wide enough
    # that the subtitles do not meet.
    ret_w = 1.10
    gap_de = 0.96
    gap_stage = 0.88
    # Same left inset as the screen in panel a, so the two displays line up.
    x_disp = 0.02
    x_enc = x_disp + side + gap_de
    x_mnt = x_enc + side + gap_stage
    x_ret = x_mnt + side + gap_stage

    # Baseline shared with the panel letter. The 0.015 axes gap is the
    # reference the other panel titles copy in inches.
    ax.text(
        0.0, 1.025, "Working memory as operations on the same posterior",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=10,
        fontweight="bold", color=ns.INK, clip_on=False,
    )
    title_y = ph - 0.14
    sub_y = title_y - 0.20
    y_map = sub_y - 0.18 - side
    eq_y = y_map - 0.48
    arrow_y = fy(y_map + side / 2.0)

    _display_inset(ax, [fx(x_disp), fy(y_map), fx(side), fy(side)])

    # f uses the Figure 2 MemGP blue; the rest of each equation is black.
    stages = [
        dict(title="Encoding", sub="fit posterior to observations",
             x=x_enc, w=side, kind="encode",
             runs=[(r"$f$", ns.MEM), (r"$:\ X \mapsto y$", ns.INK)]),
        dict(title="Maintenance", sub="rehearse under forgetting pressure",
             x=x_mnt, w=side, kind="maintain",
             runs=[(r"$f_t$", ns.MEM), (r"$\ \mapsto\ $", ns.INK),
                   (r"$f_{t+1}$", ns.MEM)]),
        dict(title="Retrieval", sub="normalize and report the peak",
             x=x_ret, w=ret_w, kind="retrieve",
             runs=[(r"$\hat{c}=\arg\max_c\, r(c)$", ns.INK)]),
    ]
    for st in stages:
        cx = fx(st["x"] + st["w"] / 2.0)
        ax.text(cx, fy(title_y), st["title"], ha="center", va="center",
                fontsize=_FS_HEAD, fontweight="bold", color=ns.INK)
        ax.text(cx, fy(sub_y), st["sub"], ha="center", va="center",
                fontsize=_FS_MIN, color=ns.INK)
        _stage_inset(
            ax,
            [fx(st["x"]), fy(y_map), fx(st["w"]), fy(side)],
            st["kind"],
        )
        _place_runs(
            ax, cx, fy(eq_y), st["runs"], fontsize=_FS_EQ,
            transform=ax.transAxes, ha="center", va="center", weight="bold",
        )

    def _short_arrow(x_left, x_right, length=0.46, shift=0.0):
        mid = 0.5 * (x_left + x_right) + shift
        ns.arrow(
            ax, (fx(mid - length / 2.0), arrow_y),
            (fx(mid + length / 2.0), arrow_y),
            lw=1.35, ms=9,
        )

    _short_arrow(x_disp + side, x_enc, length=0.30, shift=-0.16)
    _short_arrow(x_enc + side, x_mnt)
    _short_arrow(x_mnt + side, x_ret)


# ---------------------------------------------------------------------------
def main():
    # Top row: display and memory field (a) beside the inducing-point pair (b).
    # Bottom row: encoding, maintenance, and retrieval (c).
    fig_w, fig_h = 7.20, 5.26
    left_in, right_in = 0.40, 0.06
    top_in, bot_in = 0.18, 0.00
    gap_x_in, gap_y_in = 0.18, 0.26
    row1_in = 2.62
    row2_in = fig_h - top_in - row1_in - gap_y_in - bot_in
    inner_w = fig_w - left_in - right_in
    w_b_in = 2.62
    w_a_in = inner_w - gap_x_in - w_b_in

    def frac_x(inches):
        return inches / fig_w

    def frac_y(inches):
        return inches / fig_h

    y1 = bot_in + row2_in + gap_y_in
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax_a = fig.add_axes([
        frac_x(left_in), frac_y(y1), frac_x(w_a_in), frac_y(row1_in),
    ])
    ax_b = fig.add_axes([
        frac_x(left_in + w_a_in + gap_x_in), frac_y(y1),
        frac_x(w_b_in), frac_y(row1_in),
    ])
    ax_c = fig.add_axes([
        frac_x(left_in), frac_y(bot_in), frac_x(inner_w), frac_y(row2_in),
    ])

    # One square size for the panel-b maps and both stimulus screens.
    map_in = min((w_b_in - 0.36 - 0.40 - 0.02) / 2.0, row1_in * 0.46)
    cbar_a = panel_a(ax_a, map_in)
    panel_b(ax_b, cbar_a, map_in)
    panel_c(ax_c, map_in)

    # Letters share a baseline with each panel title.
    ns.panel(ax_a, "a", x=-0.015, y=0.90)
    ns.panel(ax_b, "b", x=-0.015, y=0.90)
    ns.panel(ax_c, "c", x=-0.015, y=1.02)

    ns.save_fig(fig, "fig1_concept")


if __name__ == "__main__":
    main()
