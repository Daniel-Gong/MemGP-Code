"""
Shared Nature/Science-style plotting toolkit for the MemGP figures.

Design rules enforced here (per high-impact-journal conventions):
  * Arial, editable text in SVG/PDF (svg.fonttype='none', pdf.fonttype=42)
  * only left + bottom spines, no panel titles, bold lowercase panel letters
  * one cohesive palette: neutral greys + a MemGP blue family + semantic accents
  * reusable schematic primitives (color wheel, GP posterior surfaces on the
    location x color torus, task "screens", flow-chart boxes/arrows, braces).

All figure scripts import from this module so the four figures share a single
visual language.
"""

import os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap, to_rgb, LightSource
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Wedge, Rectangle
from matplotlib.path import Path
import matplotlib.patches as mpatches

from memgp.paths import RESULTS_DIR, FIGURES_DIR, DATA_DIR

OUT_DIR = str(FIGURES_DIR)
os.makedirs(OUT_DIR, exist_ok=True)


def out_path(name):
    """Base path (no extension) for a rendered figure in figures/output/."""
    return os.path.join(OUT_DIR, name)


# ----------------------------------------------------------------------------
# Palette  (one neutral family + one MemGP-blue family + semantic accents)
# ----------------------------------------------------------------------------
INK        = "#1A1A1A"   # near-black for text / strong lines
GREY_DK    = "#4D4D4D"
GREY       = "#8A8A8A"
GREY_LT    = "#C9C9C9"
GREY_FILL  = "#ECECEC"

MEM        = "#0F4D92"   # MemGP primary deep blue
MEM_MID    = "#3F7CBF"
MEM_LT     = "#9CC2E6"
MEM_PALE   = "#DCE8F5"

SLOT       = "#E08214"   # slot models  (warm amber)
SLOT_LT    = "#F6C68B"
RESOURCE   = "#1B9E77"   # resource models (teal-green)
RESOURCE_LT= "#9BD9C6"

CUE        = "#C8102E"   # attention / cued item (signal red)
CUE_LT     = "#F2A6B0"
NEUTRAL    = "#6E6E6E"   # neutral / uncued

PLUM       = "#5C2E5A"   # human data
PLUM_LT    = "#C4A3C2"   # human neutral / lighter series
PLUM_SEQ   = ["#E6D4E4", "#C4A3C2", "#8E6490", "#5C2E5A", "#351833"]

GOOD       = "#2E9E44"   # directional: improvement / benefit
BAD        = "#E53935"   # directional: degradation

# Bias direction: which way a report is displaced relative to a neighbouring
# item. Held apart from the model-identity palette above so that no colour ever
# means both "MemGP" and "attraction" in the same figure.
ATTRACT    = "#0E7C7B"   # toward the neighbour
ATTRACT_LT = "#8FC9C7"
REPEL      = "#C8102E"   # away from the neighbour
REPEL_LT   = "#F2A6B0"

# ordered sequence for set sizes N = 1,2,4,6,8 (light -> dark blue)
SETSIZE_SEQ = ["#A8CBE8", "#6FA8DC", "#3F7CBF", "#1B5DA0", "#0A335F"]

# activation / "belief mass" colormap for 2-D GP posterior heatmaps (imshow)
ACT_CMAP = LinearSegmentedColormap.from_list(
    "memgp_act",
    ["#FBFCFE", "#E0EAF6", "#9CC2E6", "#3F7CBF", "#0F4D92", "#08243F"],
)
# vivid "deep-ocean -> glowing crest" map for 3-D posterior surfaces: a dark
# cool base makes the bumps read as bright, prominent crests
SURF_CMAP = LinearSegmentedColormap.from_list(
    "memgp_surf",
    ["#0A2452", "#12498F", "#1E7AC4", "#3FB3DC", "#9DE6EC", "#FFFFFF"],
)
# diverging colormap for change/delta surfaces
DELTA_CMAP = "RdBu_r"


# ----------------------------------------------------------------------------
# rcParams
# ----------------------------------------------------------------------------
def set_style(base=9):
    rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": base,
        "axes.labelsize": base + 1,
        "axes.titlesize": base + 1,
        "xtick.labelsize": base,
        "ytick.labelsize": base,
        "legend.fontsize": base - 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "axes.labelcolor": INK,
        "text.color": INK,
        "axes.edgecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "lines.linewidth": 1.8,
        "lines.solid_capstyle": "round",
        "legend.frameon": False,
        "figure.dpi": 130,
        "savefig.dpi": 400,
    })


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def panel(ax, letter, x=-0.13, y=1.04, size=14, weight="bold"):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=size,
            fontweight=weight, va="bottom", ha="right", color=INK)


def clean(ax, left=True, bottom=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    if not left:
        ax.tick_params(left=False, labelleft=False)
    if not bottom:
        ax.tick_params(bottom=False, labelbottom=False)


def blank(ax, equal=True):
    """Turn an axes into a blank schematic canvas.

    equal=True keeps a 1:1 aspect (for circles/screens); equal=False lets the
    0-1 coordinate box fill the whole (possibly wide) axes so flow-chart text
    has room.
    """
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if equal:
        ax.set_aspect("equal")


def lighten(color, f=0.5):
    r, g, b = to_rgb(color)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def darken(color, f=0.3):
    r, g, b = to_rgb(color)
    return (r * (1 - f), g * (1 - f), b * (1 - f))


def save_fig(fig, name, formats=("pdf", "png", "svg"), dpi=400):
    out = []
    for ext in formats:
        p = os.path.join(OUT_DIR, f"{name}.{ext}")
        fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
        out.append(p)
    plt.close(fig)
    print("saved", " / ".join(os.path.basename(p) for p in out))
    return out


# ----------------------------------------------------------------------------
# Circular feature -> colour
# ----------------------------------------------------------------------------
# The continuous-report wheel the experiments use: a circle in CIELAB at fixed
# lightness, so equal steps in degrees are roughly equal perceptual steps.
LAB_L = 70.0
LAB_CHROMA = 38.0


def _lab_to_srgb(L, a, b):
    """CIELAB (D65) to gamma-encoded sRGB, clipped to the displayable gamut."""
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    def finv(t):
        return np.where(t ** 3 > 0.008856, t ** 3, (t - 16.0 / 116.0) / 7.787)

    x, y, z = 0.95047 * finv(fx), 1.0 * finv(fy), 1.08883 * finv(fz)
    lin = np.stack([
        3.2404542 * x - 1.5371385 * y - 0.4985314 * z,
        -0.9692660 * x + 1.8760108 * y + 0.0415560 * z,
        0.0556434 * x - 0.2040259 * y + 1.0572252 * z,
    ], axis=-1)
    lin = np.clip(lin, 0.0, None)
    srgb = np.where(lin <= 0.0031308, 12.92 * lin,
                    1.055 * lin ** (1.0 / 2.4) - 0.055)
    return np.clip(srgb, 0.0, 1.0)


def color_to_rgb(c_deg, L=LAB_L, chroma=LAB_CHROMA):
    """Map a circular feature value in [-180,180) to a CIELAB wheel colour."""
    theta = np.deg2rad(np.asarray(c_deg, float))
    return _lab_to_srgb(np.full(theta.shape, L),
                        chroma * np.cos(theta), chroma * np.sin(theta))


def color_strip(ax, x0, x1, y0, y1, n=512, L=LAB_L, chroma=LAB_CHROMA,
                ec=None, lw=0.6, z=1):
    """Paint a band of an x axis with the colours those x values stand for.

    Lets a colour axis carry the feature dimension itself, so item markers on
    it need no separate colour code.
    """
    xs = np.linspace(x0, x1, n)
    rgb = color_to_rgb(xs, L=L, chroma=chroma)[None, :, :]
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.imshow(rgb, extent=(x0, x1, y0, y1), aspect="auto", origin="lower",
              interpolation="bilinear", zorder=z)
    if ec is not None:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="none", ec=ec,
                               lw=lw, zorder=z + 1))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def circ_dist(a, b, period=360.0):
    d = np.abs(a - b) % period
    return np.minimum(d, period - d)


def color_wheel(ax, center=(0, 0), r_out=1.0, r_in=0.72, n=360, lw=0):
    """Draw the continuous-report colour wheel; ring angle is feature value."""
    cx, cy = center
    for i in range(n):
        a0 = 360.0 * i / n
        a1 = 360.0 * (i + 1) / n
        w = Wedge((cx, cy), r_out, a0, a1, width=(r_out - r_in),
                  facecolor=color_to_rgb(a0), edgecolor="none", lw=lw,
                  antialiased=True)
        ax.add_patch(w)


# ----------------------------------------------------------------------------
# Periodic-kernel GP-like surface on the location x color torus
# ----------------------------------------------------------------------------
def gp_surface(items, n=200, sigma_loc=20.0, sigma_col=20.0, amps=None,
               lo=-180.0, hi=180.0):
    """Sum of circular-Gaussian bumps -> a smooth 'posterior belief' surface.

    items : list of (loc, color) in degrees.  Returns (axis, surface[n,n]) with
    surface indexed [loc, color].
    """
    axis = np.linspace(lo, hi, n)
    L, C = np.meshgrid(axis, axis, indexing="ij")
    surf = np.zeros_like(L)
    if amps is None:
        amps = [1.0] * len(items)
    for (l, c), a in zip(items, amps):
        dl = circ_dist(L, l)
        dc = circ_dist(C, c)
        surf += a * np.exp(-0.5 * (dl / sigma_loc) ** 2
                           - 0.5 * (dc / sigma_col) ** 2)
    return axis, surf


def surface3d(ax, axis, surf, cmap=None, view=(34, -60), zmax=1.05,
              box_aspect=(1, 1, 0.55), vert_exag=2.2, norm_max=None):
    """Render a smooth, light-shaded GP-posterior surface.

    Uses the full mesh resolution (no rstride/cstride down-sampling) plus a
    hill-shaded colormap so the surface looks continuous and the crests pop,
    instead of the faceted wireframe look of a coarse stride + white grid.

    axis : 1-D coordinate array (length n)
    surf : (n, n) heights indexed [x, y]
    norm_max : shared normalizer (e.g. to compare two surfaces); defaults to
        this surface's own maximum.
    """
    cmap = SURF_CMAP if cmap is None else (plt.get_cmap(cmap)
                                           if isinstance(cmap, str) else cmap)
    X, Y = np.meshgrid(axis, axis, indexing="ij")
    denom = (surf.max() if norm_max is None else norm_max) + 1e-9
    z = surf / denom
    n0, n1 = z.shape
    ls = LightSource(azdeg=315, altdeg=52)
    rgb = ls.shade(z, cmap=cmap, blend_mode="soft", vert_exag=vert_exag,
                   vmin=0.0, vmax=1.0)
    # rasterize just the surface (keeps text/axes vector + editable, and avoids
    # multi-MB per-quad polygon output / seam artefacts in SVG/PDF)
    ax.plot_surface(X, Y, z, rcount=n0, ccount=n1, facecolors=rgb,
                    linewidth=0, antialiased=True, edgecolor="none",
                    shade=False, zorder=2, rasterized=True)
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_box_aspect(box_aspect)
    for p in (ax.xaxis, ax.yaxis, ax.zaxis):
        p.pane.set_visible(False)
        p.line.set_color((1, 1, 1, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_zlim(0, zmax)


def color_slice(colors, items, query_loc, sigma_loc=20.0, sigma_col=20.0,
                amps=None):
    """1-D colour tuning curve at a probed location (sum of bumps)."""
    colors = np.asarray(colors, float)
    out = np.zeros_like(colors)
    if amps is None:
        amps = [1.0] * len(items)
    for (l, c), a in zip(items, amps):
        gl = np.exp(-0.5 * (circ_dist(query_loc, l) / sigma_loc) ** 2)
        out += a * gl * np.exp(-0.5 * (circ_dist(colors, c) / sigma_col) ** 2)
    return out


# ----------------------------------------------------------------------------
# Schematic primitives: flow-chart boxes, arrows, braces, task screens
# ----------------------------------------------------------------------------
def box(ax, xy, w, h, text="", fc="white", ec=INK, lw=1.4, fs=9, tc=INK,
        rounding=0.06, weight="normal", z=3, pad=0.02, ha="center", va="center"):
    x, y = xy
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad={pad},rounding_size={rounding}",
                       fc=fc, ec=ec, lw=lw, zorder=z, mutation_aspect=1)
    ax.add_patch(p)
    if text:
        tx = x + w / 2 if ha == "center" else (x + 0.04 if ha == "left" else x + w - 0.04)
        ty = y + h / 2 if va == "center" else (y + h - 0.04 if va == "top" else y + 0.04)
        ax.text(tx, ty, text, ha=ha, va=va, fontsize=fs, color=tc,
                fontweight=weight, zorder=z + 1)
    return p


def arrow(ax, p0, p1, color=INK, lw=1.8, style="-|>", ms=12, rad=0.0,
          z=4, ls="-"):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms,
                        lw=lw, color=color, zorder=z,
                        connectionstyle=f"arc3,rad={rad}", linestyle=ls,
                        shrinkA=2, shrinkB=2, capstyle="round")
    ax.add_patch(a)
    return a


def brace(ax, x, y0, y1, width=0.04, color=GREY_DK, lw=1.4, side="left", z=3):
    """Vertical curly brace between y0 and y1 at position x."""
    ym = (y0 + y1) / 2
    s = -1 if side == "left" else 1
    verts = [
        (x, y0), (x + s * width, y0), (x + s * width, ym - 1e-6),
        (x + 2 * s * width, ym),
        (x + s * width, ym + 1e-6), (x + s * width, y1), (x, y1),
    ]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.CURVE3,
             Path.CURVE3, Path.CURVE3, Path.CURVE3]
    ax.add_patch(mpatches.PathPatch(Path(verts, codes), fill=False,
                                    ec=color, lw=lw, zorder=z))


def task_screen(ax, cx, cy, s=1.0, fc="white", ec=INK, lw=1.5, fixation=True,
                z=2):
    """A square stimulus 'monitor' frame with optional central fixation cross."""
    r = Rectangle((cx - s / 2, cy - s / 2), s, s, fc=fc, ec=ec, lw=lw, zorder=z)
    ax.add_patch(r)
    if fixation:
        f = s * 0.06
        ax.plot([cx - f, cx + f], [cy, cy], color=INK, lw=1.0, zorder=z + 1)
        ax.plot([cx, cx], [cy - f, cy + f], color=INK, lw=1.0, zorder=z + 1)
    return r


def stim_items(ax, items, cx, cy, ring_r=0.30, sq=0.085, edge=INK, lw=1.0,
               z=5, ghost_ring=True, marker="s"):
    """Place coloured item markers on an invisible ring inside a task screen.

    items : list of (loc_deg, color_deg). loc maps to angular position on ring.
    """
    if ghost_ring:
        ax.add_patch(Circle((cx, cy), ring_r, fc="none", ec=GREY_LT, lw=0.8,
                            ls=(0, (3, 3)), zorder=3))
    pts = []
    for (l, c) in items:
        ang = np.deg2rad(l)
        px = cx + ring_r * np.cos(ang)
        py = cy + ring_r * np.sin(ang)
        pts.append((px, py))
        col = color_to_rgb(c)
        if marker == "s":
            ax.add_patch(Rectangle((px - sq / 2, py - sq / 2), sq, sq,
                                   fc=col, ec=edge, lw=lw, zorder=z))
        else:
            ax.add_patch(Circle((px, py), sq / 2, fc=col, ec=edge, lw=lw, zorder=z))
    return pts


def outline_squares(ax, locs, cx, cy, ring_r=0.30, sq=0.085, ec=INK, lw=1.2,
                    z=5):
    """Empty (probe) squares at given locations."""
    for l in locs:
        ang = np.deg2rad(l)
        px = cx + ring_r * np.cos(ang)
        py = cy + ring_r * np.sin(ang)
        ax.add_patch(Rectangle((px - sq / 2, py - sq / 2), sq, sq,
                               fc="white", ec=ec, lw=lw, zorder=z))


# ----------------------------------------------------------------------------
# data loaders
# ----------------------------------------------------------------------------
def result_path(*p):
    """Path under results/ (e.g. ``result_path("setsize", "ls_sweep.npz")``)."""
    return os.path.join(RESULTS_DIR, *p)


def data_path(*p):
    """Path under data/ (published human datasets)."""
    return os.path.join(DATA_DIR, *p)


def load_npz(path):
    return dict(np.load(path, allow_pickle=True))


def circ_sd_deg(errs_deg):
    r = np.deg2rad(np.asarray(errs_deg, float))
    C, S = np.mean(np.cos(r)), np.mean(np.sin(r))
    R = min(max(np.hypot(C, S), 1e-12), 1 - 1e-12)
    return float(np.rad2deg(np.sqrt(-2.0 * np.log(R))))
