# -*- coding: utf-8 -*-
"""
Nearest-neighbor RT analysis: 2×2 quadrants in FF–tau space.

For each quadrant in (tau_char_days, FF_1d):
    Q1 (top-left):  FF >= ff_thresh, tau <= tau_thresh_days
    Q2 (top-right): FF >= ff_thresh, tau >  tau_thresh_days
    Q3 (bot-left):  FF <  ff_thresh, tau <= tau_thresh_days
    Q4 (bot-right): FF <  ff_thresh, tau >  tau_thresh_days

Each panel shows:
    - gray/black contours of all-event RT density (same in all panels)
    - colored pcolormesh of events belonging to that quadrant

Summary CSV columns:
    cluster_id,name,n_events,tau_char_days,chosen_bin_days,FF_1d,maxN_1d,
    clat_med,clon_med
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
import matplotlib.ticker as mticker

class IntegerPowerFormatter(mticker.Formatter):
    """
    Format ticks as 10^n, but only integer powers (no coefficients).
    Using TeX formatting for pretty superscripts: 10^{n}.
    """
    def __call__(self, x, pos=None):
        if x <= 0:
            return ""
        exp = int(np.round(np.log10(x)))
        return f"$10^{{{exp}}}$"

# ======================================================
# USER SETTINGS
# ======================================================
nn_npz      = "output/nn_results_all.npz"
summary_csv = "output/cluster_tau-ff_summary.csv"

# FF–tau thresholds defining quadrants
ff_thresh       = 10.0   # FF_1d threshold
tau_thresh_days = 1.0    # tau_char_days threshold

# RT histogram / smoothing
nbins_T    = 120
nbins_R    = 120
smooth_all = 2.0     # smoothing for all-event RT density
smooth_q   = 1.5     # smoothing for quadrant densities

# global RT axis limits in log10-space (set to None to use percentiles)
T_range = (-7.5, -1.5)   # (min, max) for log10 T   (rescaled time)
R_range = (-4.,  3.)     # (min, max) for log10 R   (rescaled distance)
# If you prefer automatic ranges, set T_range = R_range = None

# contour levels (counts per bin) for all-event RT distribution
contour_levels = [1, 10, 30, 50]

# figure / colormap
figsize   = (11*0.8, 10*0.8)
cmap_main = ["hot_r", "CMRmap_r"][0]

# ------------------------------------------------------
# NEW: publish-style handles for font sizes
# ------------------------------------------------------
FS_LABEL   = 12   # axis labels, colorbar labels
FS_TICK    = 12   # axis tick labels, colorbar tick labels
FS_TITLE   = 14   # subplot titles
FS_CONTOUR = 8    # contour-label fontsize (separate handle)
# ------------------------------------------------------

outdir = "output"
os.makedirs(outdir, exist_ok=True)


# ======================================================
# helpers
# ======================================================
def load_cluster_quadrants(summary_file, ff_th, tau_th_days):
    """
    Read cluster_acf_summary.csv and return a dict:
        cluster_id -> quadrant (1,2,3,4) as defined above.
    """
    cid_to_quad = {}
    with open(summary_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                cid = int(row["cluster_id"])
                ff1 = float(row["FF_1d"])
                tau = float(row["tau_char_days"])
            except Exception:
                continue

            if (ff1 >= ff_th) and (tau <= tau_th_days):
                quad = 1  # high FF, short tau
            elif (ff1 >= ff_th) and (tau > tau_th_days):
                quad = 2  # high FF, long  tau
            elif (ff1 < ff_th) and (tau <= tau_th_days):
                quad = 3  # low FF, short tau
            else:
                quad = 4  # low FF, long  tau

            cid_to_quad[cid] = quad

    print(f"[INFO] Quadrant assignment loaded for {len(cid_to_quad)} clusters.")
    return cid_to_quad


def make_hist2d(logT, logR, xedges, yedges):
    """Simple wrapper to compute 2D histogram on given edges."""
    H, _, _ = np.histogram2d(logT, logR, bins=[xedges, yedges])
    return H


# ======================================================
# MAIN
# ======================================================
# ---- load NN results ----
data  = np.load(nn_npz)
T     = data["T"]
R     = data["R"]
clids = data["cluster"]

valid = np.isfinite(T) & np.isfinite(R) & (T > 0) & (R > 0)
T     = T[valid]
R     = R[valid]
clids = clids[valid]

logT = np.log10(T)
logR = np.log10(R)

# ---- determine axis ranges ----
if (T_range is None) or (R_range is None):
    T_lo, T_hi = np.percentile(logT, [0.5, 99.5])
    R_lo, R_hi = np.percentile(logR, [0.5, 99.5])
    if T_range is None:
        T_range = (T_lo, T_hi)
    if R_range is None:
        R_range = (R_lo, R_hi)
else:
    T_lo, T_hi = T_range
    R_lo, R_hi = R_range

# ---- RT histogram for all events ----
H_all, xedges, yedges = np.histogram2d(
    logT, logR,
    bins=[nbins_T, nbins_R],
    range=[[T_lo, T_hi], [R_lo, R_hi]]
)
H_all_s = gaussian_filter(H_all.T, smooth_all)

# centers for contours
Xc = 0.5 * (xedges[:-1] + xedges[1:])
Yc = 0.5 * (yedges[:-1] + yedges[1:])

# ---- load cluster quadrants ----
cid_to_quad = load_cluster_quadrants(
    summary_csv, ff_thresh, tau_thresh_days
)

# For events whose cluster_id is not in cid_to_quad, we ignore them in quadrant plots
quad_masks = {1: [], 2: [], 3: [], 4: []}
for q in quad_masks:
    quad_masks[q] = np.zeros_like(clids, dtype=bool)

for cid, quad in cid_to_quad.items():
    quad_masks[quad] |= (clids == cid)

# ======================================================
# PLOTTING
# ======================================================
fig, axes = plt.subplots(2, 2, figsize=figsize)  # , sharex=True, sharey=True)

# mapping from quadrant index to subplot position
quad_to_ax = {
    1: axes[0, 0],  # top-left:  FF >= th, tau <= th
    2: axes[0, 1],  # top-right: FF >= th, tau >  th
    3: axes[1, 0],  # bottom-left:  FF < th, tau <= th
    4: axes[1, 1],  # bottom-right: FF < th, tau > th
}

titles = {
    1: r"(a) FF≥%.1f, $\tau_{\mathrm{char}}$≤%.1f d" % (ff_thresh, tau_thresh_days),
    2: r"(b) FF≥%.1f, $\tau_{\mathrm{char}}$>%.1f d" % (ff_thresh, tau_thresh_days),
    3: r"(c) FF<%.1f, $\tau_{\mathrm{char}}$≤%.1f d" % (ff_thresh, tau_thresh_days),
    4: r"(d) FF<%.1f, $\tau_{\mathrm{char}}$>%.1f d" % (ff_thresh, tau_thresh_days),
}

for q in [1, 2, 3, 4]:
    ax = quad_to_ax[q]
    mask_q = quad_masks[q]

    # ---- 2D histogram for this quadrant ----
    if np.any(mask_q):
        H_q = make_hist2d(logT[mask_q], logR[mask_q], xedges, yedges)
        H_q_s = gaussian_filter(H_q.T, smooth_q)

        # avoid vmin >= vmax for very sparse quadrants
        positive = H_q_s[H_q_s > 0]
        if positive.size > 0:
            vmin = max(1.0, positive.min())
            vmax = positive.max()
        else:
            vmin, vmax = 1.0, 1.0  # dummy, panel will look empty

        im = ax.pcolormesh(
            xedges, yedges, H_q_s,
            cmap=cmap_main,
            norm=LogNorm(vmin=vmin, vmax=vmax)
        )

        # vertical colorbar on the right of this axes
        cb = fig.colorbar(
            im, ax=ax,
            orientation="vertical",
            fraction=0.046, pad=0.02
        )
        cb.set_label("Counts per bin", fontsize=FS_LABEL, rotation=-90, va="bottom")
        cb.ax.tick_params(labelsize=FS_TICK)
        cb.ax.yaxis.set_major_locator(
            mticker.LogLocator(base=10, numticks=10)
        )
        # cb.ax.yaxis.set_major_formatter(IntegerPowerFormatter())
    else:
        # no events in this quadrant
        im = None
        print(f"[WARN] No events in quadrant {q} – panel will be empty.")

    # ---- background contours of all events (same in all panels) ----
    for i, lvl in enumerate(contour_levels):
        cs = ax.contour(
            Xc, Yc, H_all_s,
            levels=[lvl],
            colors="k",
            linewidths=1. + 0.75 * (i / max(1, len(contour_levels)-1)),
            alpha=.4 + 0.2 * (i / max(1, len(contour_levels)-1))
        )
        # label contours only in Q1 to avoid clutter; they’re identical elsewhere
        if q == 1:
            ax.clabel(cs, inline=True, fontsize=FS_CONTOUR, fmt="%.0f")

    # axes formatting
    ax.set_xlim(T_range)
    ax.set_ylim(R_range)
    if q>2: ax.set_xlabel("log10 T (rescaled time)", fontsize=FS_LABEL)
    if q in [1,3]: ax.set_ylabel("log10 R (rescaled distance)", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_aspect("equal")
    ax.set_title(titles[q], fontsize=FS_TITLE)

plt.tight_layout()
outfile = os.path.join(outdir, "RT_quadrants_FF-tau.pdf")
# plt.savefig(outfile, dpi=600)
plt.savefig(outfile)
plt.close(fig)
print(f"[OK] Saved quadrant RT figure to {outfile}")
