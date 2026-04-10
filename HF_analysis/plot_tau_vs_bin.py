# -*- coding: utf-8 -*-
"""
ACF-based burstiness timescale vs bin size for seismicity clusters.

Bin scan: 0.5–30.0 days, step 0.5 days
Color code by bin size (coolwarm), single fontsize handle (FS).

Top panel:
  - tau_char vs bin size, colored markers
  - reference lines y=x and y=2x, xlim fixed to 0–30 days
  - legend placed in top panel
  - colorbar as inset axes INSIDE the top panel (not outside)

Middle panel:
  - N(t) step curves colored by bin size (right axis)
  - seismicity M-t scatter in black plotted on top

Bottom panel:
  - ACF curves colored by bin size
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as pe
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from obspy import UTCDateTime

# =========================
# user parameters
# =========================
fclust = "input/db-seis_4-manual-cluster_Nmin-100.csv"
mc = 0.3

# bin scan: (0.5, 30.0, 0.5) days
bin_days_min = 0.5
bin_days_max = 30.0
bin_days_dt  = 0.5
bin_days_array = np.arange(bin_days_min, bin_days_max + 1e-9, bin_days_dt, dtype=float)

stride_hours  = 1.0           # stride for sliding windows
acf_thresh    = 1.0 / np.e    # tau_char: first lag where ACF < threshold
zoom_days     = 60.0          # time window for zoomed MT+N(t) panel (days)
acf_max_lag_days = 60.0       # max lag shown for ACF (days)

outdir = "output/acf-bin_spec_clusters_v5"
os.makedirs(outdir, exist_ok=True)

# =========================
# publish-style knobs
# =========================
FS = 14            # single handle for all font sizes (labels, ticks, titles)
DPI = 300          # figure DPI
FIGSIZE = (7, 10)

# colorbar inset config (axes fraction in TOP panel coordinates) :contentReference[oaicite:0]{index=0}
# [left, bottom, width, height] in ax0 coordinates via fig.add_axes in figure coords below.
# (kept explicit for easy tweaking)
cbar_inset = [0.55, 0.35, 0.35, 0.05]  # relative to ax0 bbox (will be converted to fig coords)

# =========================
# helpers
# =========================
def read_clusters(fclust, mc=0.0):
    """Read cluster file into list of structured arrays."""
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


def make_sliding_counts(ots, bin_hours, stride_hours):
    """
    Return bin centers (epoch seconds) and counts for sliding windows.
    """
    ts = np.sort(np.array([float(t) for t in ots], dtype=float))  # epoch seconds
    if ts.size == 0:
        return np.array([]), np.array([])
    tmin, tmax = ts[0], ts[-1]
    bin_sec    = bin_hours * 3600.0
    stride_sec = stride_hours * 3600.0

    centers = []
    counts  = []
    t = tmin
    while t + bin_sec <= tmax:
        c = np.sum((ts >= t) & (ts < t + bin_sec))
        centers.append(t + 0.5 * bin_sec)
        counts.append(c)
        t += stride_sec

    return np.asarray(centers, float), np.asarray(counts, float)


def acf_1d(x):
    """Normalised ACF using FFT, lag >= 0."""
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = len(x)
    if n < 2:
        return np.array([1.0])
    f = np.fft.rfft(x, n=2*n)
    ps = f * np.conjugate(f)
    ac = np.fft.irfft(ps)[:n]
    ac /= ac[0]
    return ac


def apply_axes_fonts(ax, fs):
    """Apply consistent tick/label/title sizes to a matplotlib Axes."""
    ax.tick_params(labelsize=fs)
    ax.xaxis.label.set_size(fs)
    ax.yaxis.label.set_size(fs)
    ax.title.set_size(fs)


def add_cbar_in_ax(fig, ax, sm, inset_rect_ax, label, fs):
    """
    Add a vertical colorbar as an inset INSIDE `ax`.

    inset_rect_ax: [l, b, w, h] in axes fraction coordinates of `ax`.
    """
    bbox = ax.get_position()  # in figure coordinates
    l = bbox.x0 + inset_rect_ax[0] * bbox.width
    b = bbox.y0 + inset_rect_ax[1] * bbox.height
    w = inset_rect_ax[2] * bbox.width
    h = inset_rect_ax[3] * bbox.height

    cax = fig.add_axes([l, b, w, h])
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=sm.cmap, norm=sm.norm, orientation='horizontal')
    cbar.set_label(label, fontsize=fs)
    cbar.ax.tick_params(labelsize=fs)
    # --- move ticks & tick labels to the TOP (for horizontal colorbar) ---
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    return cbar


# =========================
# main
# =========================
clust_names, clust_list = read_clusters(fclust, mc=mc)
print(f"Loaded {len(clust_list)} clusters")

# colormap setup (bin size -> color)
norm = Normalize(vmin=0, vmax=bin_days_max) #TODO
cmap = plt.get_cmap(["coolwarm_r","RdYlBu"][1])
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

for ic, cl in enumerate(clust_list):
    if len(cl) < 20: continue

    ots  = np.sort(cl['ot'])
    mags = cl['mag']

    # ---------- compute N(t), ACF, tau_char for each bin size ----------
    bin_results = {}  # bin_days -> dict(centers, counts, lags_days, acf, tau_char_days)

    for bd in bin_days_array:
        bin_hours = bd * 24.0
        centers, counts = make_sliding_counts(ots, bin_hours, stride_hours)

        if len(counts) < 10 or np.all(counts == 0):
            continue

        ac = acf_1d(counts)
        lags_hours = np.arange(len(ac)) * stride_hours
        lags_days  = lags_hours / 24.0

        m = lags_days <= acf_max_lag_days
        lags_use = lags_days[m]
        ac_use   = ac[m]

        idx = np.where(ac_use < acf_thresh)[0]
        tau_char_d = lags_use[-1] if len(idx) == 0 else lags_use[idx[0]]

        bin_results[float(bd)] = dict(
            centers=centers,
            counts=counts,
            lags_days=lags_use,
            acf=ac_use,
            tau_char_d=float(tau_char_d),
        )

    if not bin_results: continue

    valid_bins = np.array(sorted(bin_results.keys()), dtype=float)
    tau_list   = np.array([bin_results[bd]["tau_char_d"] for bd in valid_bins], dtype=float)

    # ---------- reference peak time based on 1-day bin if possible ----------
    ref_bd = 1.0 if 1.0 in bin_results else float(valid_bins.min())
    centers_ref = bin_results[ref_bd]["centers"]
    counts_ref  = bin_results[ref_bd]["counts"]
    if counts_ref.size == 0: continue
    t_peak = centers_ref[np.argmax(counts_ref)]

    # ---------- figure (3x1) ----------
    cname = clust_names[ic] if ic < len(clust_names) else f"cluster_{ic}"

    fig, (ax0, ax1, ax2) = plt.subplots(
        3, 1, figsize=FIGSIZE, constrained_layout=True
    )

    # ============ (1) tau_char vs bin size ============
    # thin connecting line + colored markers
    ax0.plot(valid_bins, tau_list, '-', color='gray', lw=1.0, alpha=0.6, zorder=2)
    colors_bins = cmap(norm(valid_bins))
    ax0.scatter(valid_bins, tau_list, c=colors_bins, s=28, edgecolor='none', zorder=3)

    # reference lines: y = x and y = 2x; keep xlim fixed 0–30
    xref = np.array([0.0, bin_days_max], dtype=float)
    ax0.plot(xref, xref, ls='-', color='k', lw=1.0, alpha=0.9, label=r"$y=x$", zorder=1)
    ax0.plot(xref, 2.0 * xref, ls='-.', color='k', lw=1.0, alpha=0.9, label=r"$y=2x$", zorder=0)

    ax0.set_xlim(0.0, bin_days_max)
    # y-limit: include data comfortably
    ymax = max(1.0, np.nanmax(tau_list) * 1.05)
    ax0.set_ylim(0.0, ymax)

    ax0.set_xlabel("Bin size (day)")
    ax0.set_ylabel(r"$\tau_{\mathrm{char}}$ (day)")
    ax0.set_title(f"Cluster {ic}: {cname}")
    ax0.grid(True, alpha=0.25, zorder=0)
    apply_axes_fonts(ax0, FS)

    # legend in top panel
    ax0.legend(loc="upper left", fontsize=FS, frameon=False)

    # inset colorbar INSIDE top panel
    _ = add_cbar_in_ax(
        fig, ax0, sm, cbar_inset,
        label="Bin size (day)", fs=FS
    )

    # ============ (2) zoomed MT + N(t) ============
    zoom_half = 0.5 * zoom_days

    # left y: magnitudes (ensure on top)
    ev_t_rel = np.array([(float(t) - t_peak) / 86400.0 for t in ots], dtype=float)
    mask_ev = (ev_t_rel >= -zoom_half) & (ev_t_rel <= zoom_half)

    ax1.set_ylabel("Magnitude")
    ax1.set_xlim(-zoom_half, zoom_half)
    ax1.set_xlabel(f"Time relative to N({ref_bd:.0f}d) peak (day)")
    #ax1.grid(True, alpha=0.25, zorder=0)
    apply_axes_fonts(ax1, FS)

    # right y: N(t) for each bin size
    ax_cnt = ax1.twinx()
    ax_cnt.tick_params(labelsize=FS)
    ax_cnt.yaxis.label.set_size(FS)
    ax_cnt.set_ylabel("Seis. Rate (N/bin)", rotation=-90, va='bottom')

    # draw N(t) first
    for bd in valid_bins:
        res = bin_results[float(bd)]
        csec = res["centers"]
        c_rel = (csec - t_peak) / 86400.0
        m = (c_rel >= -zoom_half) & (c_rel <= zoom_half)
        if not np.any(m):
            continue
        ax_cnt.plot(
            c_rel[m],
            res["counts"][m],
            drawstyle='steps-mid',
            lw=1.05,
            alpha=1.,
            color=cmap(norm(bd)),
            zorder=2
        )

    # then draw seismicity scatter on top layer
    ax1.scatter(
        ev_t_rel[mask_ev],
        mags[mask_ev],
        s=10,
        c='k',
        alpha=0.75,
        linewidths=0,
        zorder=10
    )
    # ensure ax1 artists are drawn over twin axis
    ax1.set_zorder(ax_cnt.get_zorder() + 1)
    ax1.patch.set_visible(False)

    # ============ (3) ACF curves ============
    for bd in valid_bins:
        res = bin_results[float(bd)]
        ax2.plot(
            res["lags_days"],
            res["acf"],
            lw=1.15,
            alpha=1.,
            color=cmap(norm(bd)),
        )

    ax2.axhline(acf_thresh, ls='--', lw=1.0, color='k', alpha=0.8)
    txt = ax2.text(
        0.99, acf_thresh,
        f"ACF threshold = {acf_thresh:.2f}",
        transform=ax2.get_yaxis_transform(),
        ha='right', va='bottom', fontsize=FS
    )
    txt.set_path_effects([pe.Stroke(linewidth=2.5, foreground="white"), pe.Normal()])


    ax2.set_xlim(0.0, acf_max_lag_days)
    ax2.set_xlabel("Time lag (day)")
    ax2.set_ylabel("Auto-Corr. Function")
    #ax2.grid(True, alpha=0.25, zorder=0)
    apply_axes_fonts(ax2, FS)

    ofig = os.path.join(outdir,
        f"acf_binspec_clust_{ic:03d}_{cname.replace(' ', '_')}.jpg")
    ofig_pdf = os.path.join(outdir,
        f"acf_binspec_clust_{ic:03d}_{cname.replace(' ', '_')}.pdf")
    fig.savefig(ofig, dpi=DPI)
    fig.savefig(ofig_pdf)
    plt.close(fig)

    print(f"Saved: {ofig}")
