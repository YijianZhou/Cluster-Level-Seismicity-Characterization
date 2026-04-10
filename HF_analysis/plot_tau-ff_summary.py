# -*- coding: utf-8 -*-
"""
Plot 2x2 summary figure from ACF/FF cluster results:

(1) Map of characteristic timescale tau_char (days).
(2) Map of Fano factor FF(1d).
(3) Map of detection metric:
        sgn * | log(1/tau) * log(FF/10) |, saturated to [-1, 1],
    where sign is positive only if both logs are positive.
(4) tau_char vs FF scatter (log-log), points color-coded by HF-detection score;
    clusters with maxN_1d < 10 are de-emphasized with gray edgecolor.

Also prints:
  - total clusters/events used
  - HF clusters/events (HF-det score > 0)
  - ratios

This script assumes:
  output/cluster_tau-ff_summary.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Rectangle
from obspy import UTCDateTime

# -------------------------
# handles: tune for publication
# -------------------------
FIGSIZE   = (11, 10)
FS_LABEL  = 14
FS_TITLE  = 14
FS_TICK   = 14

MARK_SIZE_MAP = 1.0
MARK_SIZE_SC  = 22

GRID_ALPHA = 0.25
DPI_OUT    = 600

# map tick step (keep ticklabels short: %.1f)
LON_TICK_STEP = 0.4
LAT_TICK_STEP = 0.4

# colorbar anchor in DATA space (fractions of lon/lat range) -> SAME lon/lat anchor in all panels
CBAR_ANCHOR_FRAC_X = 0.10   # from lon_min
CBAR_ANCHOR_FRAC_Y = 0.08   # from lat_min

# colorbar size in DATA space (fractions of lon/lat range)
CBAR_W_FRAC_LON = 0.032
CBAR_H_FRAC_LAT = 0.38

# new: background styles
TOPROW_WHITE_WASH_ALPHA = 0.25   # alpha for white wash rectangle in top row (over scatter, under colorbar)
BOTTOMROW_GRAY_BG = 'darkgray'  # gray background for second row

cmap_list = ['viridis_r', 'cool', 'RdYlBu_r']

# -------------------------
# user params
# -------------------------
fclust       = "input/db-seis_4-manual-cluster_Nmin-100.csv"
summary_file = "output/cluster_tau-ff_summary.csv"

# map settings
map_lon_rng = (-104.8, -102.95)
map_lat_rng = (30.8,  32.15)

# color saturation
tau_sat_days = 10.0   # saturation for tau_char map (days)
ff_sat       = 20.0   # saturation for FF map
det_sat      = 1.0    # saturation for detection metric [-1, 1]

# HF-cluster detection thresholds
tau_thresh_days = 1.0   # tau_char <= 1 d (reference line only)
ff_thresh       = 10.0  # FF >= 10 (reference line only)
maxN_special    = 10    # de-emphasize if maxN_1d < 10

# -------------------------
# helpers
# -------------------------
def read_clusters(fclust, mc=0.0):
    """Read clusters (same structure as previous scripts)."""
    clust_names = []
    clust_list  = []
    dtype = [('ot', 'O'), ('lat', 'f8'), ('lon', 'f8'),
             ('dep', 'f8'), ('mag', 'f8'), ('evid', 'i8')]
    with open(fclust, 'r') as f:
        lines = f.readlines()

    cur = []
    for line in lines:
        if line.startswith('#'):
            if cur:
                clust_list.append(np.array(cur, dtype=dtype))
                cur = []
            clust_names.append(line[2:].strip())
            continue
        parts = line.split(',')
        ot = UTCDateTime(parts[0])
        lat, lon, dep, mag = map(float, parts[1:5])
        dep *= 1e3
        evid = int(parts[-1])
        if mag < mc:
            continue
        cur.append((ot, lat, lon, dep, mag, evid))
    if cur:
        clust_list.append(np.array(cur, dtype=dtype))
    return clust_names, clust_list


def det_metric(tau_char_days, FF_1d, ff_thresh=10.0, det_sat=1.0):
    """
    sign * |log(1/tau) * log(FF/ff_thresh)|, saturated to [-det_sat, det_sat]
    sign positive only if both logs are positive.
    """
    if (tau_char_days is None) or (FF_1d is None):
        return np.nan
    tau_char_days = float(tau_char_days)
    FF_1d = float(FF_1d)
    if (not np.isfinite(tau_char_days)) or (not np.isfinite(FF_1d)) or (tau_char_days <= 0.0) or (FF_1d <= 0.0):
        return 0.0
    a = np.log(1.0 / tau_char_days)
    b = np.log(FF_1d / ff_thresh)
    m0 = abs(a * b)
    sign = 1.0 if (a > 0.0 and b > 0.0) else -1.0
    det_raw = sign * m0
    return max(-det_sat, min(det_sat, det_raw))


def add_cbar_anchored_in_data(fig, ax, mappable, label,
                              lon0, lat0, w_lon, h_lat,
                              fs_label=FS_LABEL, fs_tick=FS_TICK):
    """
    Add a VERTICAL colorbar whose LOWER-LEFT corner is anchored at (lon0, lat0)
    in DATA coordinates, with size (w_lon, h_lat) also in DATA coordinates.

    This guarantees the same lon/lat anchor for all map panels.
    """
    cax = ax.inset_axes([lon0, lat0, w_lon, h_lat], transform=ax.transData)
    cb = fig.colorbar(mappable, cax=cax, orientation="vertical")
    cb.set_label(label, fontsize=fs_label, rotation=-90, va='bottom')
    cb.ax.tick_params(labelsize=fs_tick)
    return cb


def add_white_wash(ax, xlim, ylim, alpha=0.05, zorder=1.5):
    """
    Add a semi-transparent white rectangle OVER the scatter (zorder>scatter),
    but below the colorbar (colorbar is on separate inset axes).
    """
    rect = Rectangle(
        (xlim[0], ylim[0]),
        xlim[1] - xlim[0],
        ylim[1] - ylim[0],
        facecolor='white',
        edgecolor='none',
        alpha=alpha,
        transform=ax.transData,
        zorder=zorder
    )
    ax.add_patch(rect)
    return rect


# -------------------------
# read data
# -------------------------
if not os.path.exists(summary_file):
    raise FileNotFoundError(f"Summary file not found: {summary_file}")

df = pd.read_csv(summary_file)
clust_names, clust_list = read_clusters(fclust, mc=0.3)

summary_by_id = {int(row["cluster_id"]): row for _, row in df.iterrows()}

# event-level arrays for maps and cluster-level arrays for scatter
all_lats_tau, all_lons_tau, all_tau_days = [], [], []
all_lats_ff,  all_lons_ff,  all_ff       = [], [], []
all_lats_det, all_lons_det, all_det      = [], [], []

tau_vals, ff_vals, det_vals, maxN_vals = [], [], [], []

# counts
total_clusters = 0
total_events   = 0
hf_clusters    = 0
hf_events      = 0

for ic, cl in enumerate(clust_list):
    if ic not in summary_by_id:
        continue

    row = summary_by_id[ic]
    tau_char_days = float(row["tau_char_days"])
    FF_1d         = float(row["FF_1d"])
    maxN_1d       = float(row["maxN_1d"])

    if (not np.isfinite(tau_char_days)) or (not np.isfinite(FF_1d)):
        continue

    det = det_metric(tau_char_days, FF_1d, ff_thresh=ff_thresh, det_sat=det_sat)

    # cluster/event totals (only for clusters included in plotting)
    n_ev = len(cl)
    total_clusters += 1
    total_events   += n_ev
    if det > 0.0:
        hf_clusters += 1
        hf_events   += n_ev

    # cluster-level arrays for scatter
    tau_vals.append(tau_char_days)
    ff_vals.append(FF_1d)
    det_vals.append(det)
    maxN_vals.append(maxN_1d)

    # event-level arrays for maps
    lats = cl["lat"]
    lons = cl["lon"]

    all_lats_tau.extend(lats)
    all_lons_tau.extend(lons)
    all_tau_days.extend(np.full_like(lats, tau_char_days, dtype=float))

    all_lats_ff.extend(lats)
    all_lons_ff.extend(lons)
    all_ff.extend(np.full_like(lats, FF_1d, dtype=float))

    all_lats_det.extend(lats)
    all_lons_det.extend(lons)
    all_det.extend(np.full_like(lats, det, dtype=float))

all_lats_tau = np.array(all_lats_tau, dtype=float)
all_lons_tau = np.array(all_lons_tau, dtype=float)
all_tau_days = np.array(all_tau_days, dtype=float)

all_lats_ff = np.array(all_lats_ff, dtype=float)
all_lons_ff = np.array(all_lons_ff, dtype=float)
all_ff      = np.array(all_ff, dtype=float)

all_lats_det = np.array(all_lats_det, dtype=float)
all_lons_det = np.array(all_lons_det, dtype=float)
all_det      = np.array(all_det, dtype=float)

tau_vals  = np.array(tau_vals, dtype=float)
ff_vals   = np.array(ff_vals, dtype=float)
det_vals  = np.array(det_vals, dtype=float)
maxN_vals = np.array(maxN_vals, dtype=float)

# -------------------------
# report counts (screen print)
# -------------------------
clust_ratio = (hf_clusters / total_clusters) if total_clusters > 0 else np.nan
ev_ratio    = (hf_events   / total_events)   if total_events   > 0 else np.nan

print("\n===== HF detection summary (HF-det score > 0) =====")
print(f"Total clusters used: {total_clusters}")
print(f"HF clusters:         {hf_clusters}  (ratio = {clust_ratio:.3f})")
print(f"Total events used:   {total_events}")
print(f"HF events:           {hf_events}    (ratio = {ev_ratio:.3f})")
print("===================================================\n")

# -------------------------
# figure setup
# -------------------------
fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, constrained_layout=True)
ax_tau, ax_ff = axes[0, 0], axes[0, 1]
ax_det, ax_sc = axes[1, 0], axes[1, 1]

# new: gray background for second row
ax_det.set_facecolor(BOTTOMROW_GRAY_BG)
ax_sc.set_facecolor(BOTTOMROW_GRAY_BG)

# central latitude for geographic scaling
if all_lats_tau.size > 0:
    lat0 = 0.5 * (all_lats_tau.min() + all_lats_tau.max())
else:
    lat0 = 0.5 * (map_lat_rng[0] + map_lat_rng[1])
cos_lat = np.cos(np.deg2rad(lat0))

lon_min, lon_max = map_lon_rng
lat_min, lat_max = map_lat_rng

lon_rng = lon_max - lon_min
lat_rng = lat_max - lat_min

# Data-aspect (for maps) and box-aspect (for all panels, including log-log scatter)
data_aspect = 1.0 / cos_lat
box_aspect  = lat_rng / (lon_rng * cos_lat)  # height / width

xticks = np.arange(lon_min, lon_max + 1e-9, LON_TICK_STEP)
yticks = np.arange(lat_min, lat_max + 1e-9, LAT_TICK_STEP)
fmt_1f = FormatStrFormatter('%.1f')

# colorbar anchor in DATA coordinates (same lon/lat for all three map panels)
cbar_lon0 = lon_min + CBAR_ANCHOR_FRAC_X * lon_rng
cbar_lat0 = lat_min + CBAR_ANCHOR_FRAC_Y * lat_rng
cbar_wlon = CBAR_W_FRAC_LON * lon_rng
cbar_hlat = CBAR_H_FRAC_LAT * lat_rng

# -------------------------
# (1) tau_char map
# -------------------------
tau_plot = np.clip(all_tau_days, 0.0, tau_sat_days) if all_tau_days.size else all_tau_days
sc0 = ax_tau.scatter(
    all_lons_tau, all_lats_tau, c=tau_plot, s=MARK_SIZE_MAP,
    cmap=cmap_list[0], vmin=0.0, vmax=tau_sat_days, linewidths=0, alpha=1.
) if all_lats_tau.size else None

ax_tau.set_xlim(map_lon_rng)
ax_tau.set_ylim(map_lat_rng)
ax_tau.set_aspect(data_aspect)
ax_tau.set_box_aspect(box_aspect)
ax_tau.set_xticks(xticks)
ax_tau.set_yticks(yticks)
ax_tau.xaxis.set_major_formatter(fmt_1f)
ax_tau.yaxis.set_major_formatter(fmt_1f)
ax_tau.tick_params(labelsize=FS_TICK)

# new: add white wash overlay (over scatter, under colorbar)
add_white_wash(ax_tau, map_lon_rng, map_lat_rng, alpha=TOPROW_WHITE_WASH_ALPHA, zorder=2.0)

if sc0 is not None:
    add_cbar_anchored_in_data(
        fig, ax_tau, sc0, label=r'$\tau_{\mathrm{char}}$ (day)',
        lon0=cbar_lon0, lat0=cbar_lat0, w_lon=cbar_wlon, h_lat=cbar_hlat
    )

# -------------------------
# (2) FF map
# -------------------------
ff_plot = np.clip(all_ff, 0.0, ff_sat) if all_ff.size else all_ff
sc1 = ax_ff.scatter(
    all_lons_ff, all_lats_ff, c=ff_plot, s=MARK_SIZE_MAP,
    cmap=cmap_list[1], vmin=0.0, vmax=ff_sat, linewidths=0, alpha=1.
) if all_lats_ff.size else None

ax_ff.set_xlim(map_lon_rng)
ax_ff.set_ylim(map_lat_rng)
ax_ff.set_aspect(data_aspect)
ax_ff.set_box_aspect(box_aspect)
ax_ff.set_xticks(xticks)
ax_ff.set_yticks(yticks)
ax_ff.xaxis.set_major_formatter(fmt_1f)
ax_ff.yaxis.set_major_formatter(fmt_1f)
ax_ff.tick_params(labelsize=FS_TICK)

# new: add white wash overlay (over scatter, under colorbar)
add_white_wash(ax_ff, map_lon_rng, map_lat_rng, alpha=TOPROW_WHITE_WASH_ALPHA, zorder=2.0)

if sc1 is not None:
    add_cbar_anchored_in_data(
        fig, ax_ff, sc1, label='FF (1-day)',
        lon0=cbar_lon0, lat0=cbar_lat0, w_lon=cbar_wlon, h_lat=cbar_hlat
    )

# -------------------------
# (3) detection metric map
# -------------------------
det_plot = np.clip(all_det, -det_sat, det_sat) if all_det.size else all_det
sc2 = ax_det.scatter(
    all_lons_det, all_lats_det, c=det_plot, s=MARK_SIZE_MAP,
    cmap=cmap_list[2], vmin=-det_sat, vmax=det_sat, linewidths=0
) if all_lats_det.size else None

ax_det.set_xlim(map_lon_rng)
ax_det.set_ylim(map_lat_rng)
ax_det.set_aspect(data_aspect)
ax_det.set_box_aspect(box_aspect)
ax_det.set_xticks(xticks)
ax_det.set_yticks(yticks)
ax_det.xaxis.set_major_formatter(fmt_1f)
ax_det.yaxis.set_major_formatter(fmt_1f)
ax_det.tick_params(labelsize=FS_TICK)

if sc2 is not None:
    add_cbar_anchored_in_data(
        fig, ax_det, sc2, label='HF-det score',
        lon0=cbar_lon0, lat0=cbar_lat0, w_lon=cbar_wlon, h_lat=cbar_hlat
    )

# -------------------------
# (4) tau vs FF scatter (log-log), color-coded by detection score
# -------------------------
mask_sc = (
    np.isfinite(tau_vals) & np.isfinite(ff_vals) & np.isfinite(det_vals) &
    (tau_vals > 0.0) & (ff_vals > 0.0)
)
tau_sc = tau_vals[mask_sc]
ff_sc  = ff_vals[mask_sc]
det_sc = det_vals[mask_sc]
maxN_sc = maxN_vals[mask_sc]

is_special = maxN_sc < maxN_special

# colormap for detection score (same scale as det map)
cmap_det = plt.get_cmap(cmap_list[2])
norm_det = mpl.colors.Normalize(vmin=-det_sat, vmax=det_sat)
colors_sc = cmap_det(norm_det(det_sc))

# plot regular first, then special on top (special: gray edge)
if np.any(~is_special):
    ax_sc.scatter(
        tau_sc[~is_special], ff_sc[~is_special],
        c=colors_sc[~is_special], s=MARK_SIZE_SC+2, alpha=1.0, zorder=100
        #edgecolors='gray', linewidths=0.4
    )
if np.any(is_special):
    ax_sc.scatter(
        tau_sc[is_special], ff_sc[is_special],
        c=colors_sc[is_special], s=MARK_SIZE_SC, alpha=0.8, zorder=99,
        edgecolors='white', linewidths=0.8
    )

# reference thresholds (both dashed)
ax_sc.axvline(tau_thresh_days, color='k', linestyle='--', alpha=0.8, linewidth=1.0, zorder=1)
ax_sc.axhline(ff_thresh,       color='k', linestyle='--', alpha=0.8, linewidth=1.0, zorder=1)

ax_sc.set_xscale('log')
ax_sc.set_yscale('log')

# log axes ignore data-aspect; enforce matching PANEL (box) aspect instead
ax_sc.set_box_aspect(box_aspect)

ax_sc.set_xlabel(r'Characteristic Timescale ($\tau_{\mathrm{char}}$, day)', fontsize=FS_LABEL)
ax_sc.set_ylabel('Fano factor (FF, 1-day)', fontsize=FS_LABEL)
ax_sc.tick_params(labelsize=FS_TICK)
ax_sc.grid(True, which='both', linestyle='-', color='white', alpha=0.25, zorder=0)

# -------------------------
# save
# -------------------------
outfig = os.path.join(os.path.dirname(summary_file), "tau-ff_summary.jpg")
plt.savefig(outfig, dpi=DPI_OUT)
plt.close(fig)

print(f"✅ Saved 2x2 summary figure to: {outfig}")
