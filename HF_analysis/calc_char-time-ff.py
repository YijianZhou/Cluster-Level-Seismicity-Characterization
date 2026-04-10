# -*- coding: utf-8 -*-
"""
ACF-based burstiness timescale for seismicity clusters
with bin-size scan and cluster-level summary output.

For each cluster:
- Scan bin sizes: 1–30 days (bin_days = 1..30), stride = 1 h.
- For each bin size, compute sliding-window N(t), N-ACF,
  and τ_char = first lag where ACF < acf_thresh (default 1/e).
- Selection rule:
    * If ALL τ_char(bin) <= 2 * bin_days, use bin_days = 1.
    * Else, pick the smallest bin_days where τ_char > 2 * bin_days.
- Recompute N(t) & ACF with the chosen bin for plotting.
- Compute FF(1 day) using 1-day window (sliding, 1 h stride).
- Save per-cluster summary to CSV for later plotting.

Also outputs per-cluster N(t) + ACF PNGs.

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from obspy import UTCDateTime

# =========================
# user parameters
# =========================
fclust = "input/db-seis_4-manual-cluster_Nmin-100.csv"
mc = 0.3

# bin scan settings
bin_days_min = 0.5
bin_days_max = 30
bin_days_dt = 0.5
stride_hours = 1.0        # slide step (controls ACF resolution)
max_lag_days = 2*bin_days_max       # only use first 30 days of ACF

acf_thresh   = 1 / np.e   # define characteristic timescale where ACF drops below this
outdir       = "output/acf-tau_clusters"
os.makedirs(outdir, exist_ok=True)

# for fixed-1-day FF
ff_bin_hours = 24.0

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
        if mag < mc: continue
        cur.append((ot, lat, lon, dep, mag, evid))
    if cur:
        clust_list.append(np.array(cur, dtype=dtype))

    return clust_names, clust_list

def make_sliding_counts(ots, bin_hours, stride_hours):
    """
    Return bin centers (UTC seconds) and counts for sliding windows.

    ots: array-like of UTCDateTime
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

    centers = np.array(centers, dtype=float)
    counts  = np.array(counts, dtype=float)
    return centers, counts

def acf_1d(x):
    """Normalised ACF using FFT, lag >= 0."""
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = len(x)
    if n < 2:
        return np.array([1.0])
    # FFT-based autocorrelation
    f = np.fft.rfft(x, n=2*n)
    ps = f * np.conjugate(f)
    ac = np.fft.irfft(ps)[:n]
    ac /= ac[0]
    return ac

# =========================
# main
# =========================
clust_names, clust_list = read_clusters(fclust, mc=mc)
print(f"Loaded {len(clust_list)} clusters")

stride_sec     = stride_hours * 3600.0
max_lag_hours  = max_lag_days * 24.0
bin_days_array = np.arange(bin_days_min, bin_days_max + bin_days_dt, bin_days_dt, dtype=float)

summary_rows = []  # to write CSV at the end
for ic, cl in enumerate(clust_list):
    n_ev = len(cl)
    if n_ev < 20: continue

    ots = np.sort(cl['ot'])
    ts = np.array([float(t) for t in ots], dtype=float)
    if ts.size < 2: continue

    # --------------------------
    # Precompute 1-day counts for FF and max N_1d
    # --------------------------
    centers_1d, counts_1d = make_sliding_counts(ots, ff_bin_hours, stride_hours)
    if counts_1d.size < 2:
        # not enough windows even for 1-day bin
        continue

    mean_c = np.mean(counts_1d)
    if mean_c > 0:
        var_c = np.var(counts_1d, ddof=1)
        ff_fixed = var_c / mean_c
    else:
        ff_fixed = np.nan
    maxN_1d = float(np.max(counts_1d))

    # ----------------------------------------
    # Scan bin sizes: 1–30 days, get tau_char
    # ----------------------------------------
    tau_days_list = []
    valid_mask    = []

    for bin_days in bin_days_array:
        bin_hours = 24.0 * bin_days
        centers_b, counts_b = make_sliding_counts(ots, bin_hours, stride_hours)

        # require at least some windows and non-trivial counts
        if counts_b.size < 10 or np.all(counts_b == 0):
            tau_days_list.append(np.nan)
            valid_mask.append(False)
            continue

        ac = acf_1d(counts_b)
        lags_hours = np.arange(len(ac)) * stride_hours

        # restrict to first max_lag_days
        mask = lags_hours <= max_lag_hours
        lags_plot = lags_hours[mask]
        ac_plot   = ac[mask]

        idx = np.where(ac_plot < acf_thresh)[0]
        if len(idx) == 0:
            tau_char_h = lags_plot[-1]
        else:
            tau_char_h = lags_plot[idx[0]]

        tau_days_list.append(tau_char_h / 24.0)
        valid_mask.append(True)

    tau_days_arr = np.array(tau_days_list, dtype=float)
    valid_mask   = np.array(valid_mask, dtype=bool)

    if not np.any(valid_mask): continue

    # For selection logic, treat invalid tau as +inf so they won't be selected spuriously
    tau_for_sel = tau_days_arr.copy()
    tau_for_sel[~valid_mask] = np.inf

    cond = tau_for_sel <= 2.0 * bin_days_array

    if np.all(cond):
        # use 1-day bin result (index 0) if it's valid; else the first valid
        if valid_mask[0]:
            chosen_idx = 0
        else:
            valid_indices = np.where(valid_mask)[0]
            chosen_idx = valid_indices[0]
    else:
        # first bin where tau > 2*bin_days
        idx_candidates = np.where(~cond & valid_mask)[0]
        if len(idx_candidates) == 0:
            # fallback to first valid if none satisfy condition
            valid_indices = np.where(valid_mask)[0]
            chosen_idx = valid_indices[0]
        else:
            chosen_idx = idx_candidates[0]

    chosen_bin_days = float(bin_days_array[chosen_idx])
    tau_char_days   = float(tau_days_arr[chosen_idx])

    # recompute counts & ACF using chosen bin for plotting
    bin_hours_final = 24.0 * chosen_bin_days
    centers_final, counts_final = make_sliding_counts(ots, bin_hours_final, stride_hours)
    ac_final = acf_1d(counts_final)
    lags_hours_final = np.arange(len(ac_final)) * stride_hours

    # restrict to first max_lag_days for plotting
    mask_final = lags_hours_final <= max_lag_hours
    lags_plot   = lags_hours_final[mask_final]
    ac_plot     = ac_final[mask_final]
    tau_char_h  = tau_char_days * 24.0  # back to hours for annotation

    # =========================
    # per-cluster plotting
    # =========================
    cname = clust_names[ic] if ic < len(clust_names) else f"cluster_{ic}"

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 6), sharex=False)

    # (a) N(t) series with chosen bin
    t_rel_days = (centers_final - centers_final[0]) / 86400.0
    ax0.plot(t_rel_days, counts_final, drawstyle='steps-mid')
    ax0.set_ylabel("N per window")
    ax0.set_title(
        f"Cluster {ic}: {cname}\n"
        f"bin={chosen_bin_days:.1f} d, stride={stride_hours:.1f} h, "
        f"τ_char≈{tau_char_days:.2f} d, FF(1 d)≈{ff_fixed:.2f}"
    )
    ax0.grid(True, alpha=0.3)

    # (b) ACF
    ax1.plot(lags_plot, ac_plot, '-o', markersize=3)
    ax1.axhline(acf_thresh, color='r', linestyle='--', alpha=0.6,
                label=f"ACF={acf_thresh:.2f}")
    ax1.axvline(tau_char_h, color='k', linestyle=':', alpha=0.7,
                label=f"τ_char≈{tau_char_days:.2f} d")

    ax1.set_xlim(0.0, max_lag_hours)
    ax1.set_xlabel("Lag (hours)")
    ax1.set_ylabel("ACF")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=8)

    plt.tight_layout()

    ofig = os.path.join(
        outdir,
        f"acf_clust_{ic:03d}_{cname.replace(' ', '_')}.jpg"
    )
    plt.savefig(ofig, dpi=300)
    plt.close(fig)

    print(f"saved: {ofig} (τ_char≈{tau_char_days:.2f} d, FF≈{ff_fixed:.2f}, bin={chosen_bin_days:.1f} d)")

    # =========================
    # accumulate cluster-level summary
    # =========================
    clat_med = float(np.median(cl['lat']))
    clon_med = float(np.median(cl['lon']))

    summary_rows.append({
        "cluster_id": ic,
        "name": cname,
        "n_events": n_ev,
        "tau_char_days": round(tau_char_days,2),
        "chosen_bin_days": chosen_bin_days,
        "FF_1d": round(ff_fixed,2),
        "maxN_1d": maxN_1d,
        "clat_med": round(clat_med,4),
        "clon_med": round(clon_med,4),
    })

# =========================
# write summary CSV
# =========================
if summary_rows:
    df_sum = pd.DataFrame(summary_rows)
    summary_file = os.path.join(outdir, "cluster_tau-ff_summary.csv")
    df_sum.to_csv(summary_file, index=False)
    print(f"\n✅ Saved cluster summary to: {summary_file}")
else:
    print("No clusters produced usable ACF/FF metrics; summary not written.")
