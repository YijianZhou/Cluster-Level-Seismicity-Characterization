# -*- coding: utf-8 -*-
"""
HF – seismicity spatiotemporal correlation (refined)
- per-well selection based on #events within radius
- episode detection + low-volume trimming
- drop unrealistically long HF jobs
- delay applied ONLY to classification, not plotting
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import UTCDateTime

# ============================
# user variables
# ============================
fclust = "input/db-seis_4-manual-cluster_Nmin-100.csv"
f_hf   = "output/cleaned_DisclosureList_well_info.csv"

max_dist_km        = 10.0     # spatial radius for event↔well association
min_evt_per_well   = 10      # keep HF well only if ≥ this many cluster events within radius
hf_delay_days      = 30.0     # ONLY for classification (seismicity can lag)
hf_cover_thresh    = 0.80     # % of cluster events that must fall into HF windows
episode_trim_factor = 0.0    # keep only days in episode with vol >= 0.1 * episode_max
max_job_duration_days = 50    # drop HF jobs longer than this

mc = 0.3
outdir_fig = "output/fig_hf_clusters2"
os.makedirs(outdir_fig, exist_ok=True)

# ============================
# helpers
# ============================
def latlon_to_dist_km_vec(lon_arr, lat_arr, lon0, lat0):
    lon_arr = np.asarray(lon_arr, dtype=float)
    lat_arr = np.asarray(lat_arr, dtype=float)
    cos_lat = np.cos(np.deg2rad(lat0))
    dx = 111.0 * (lon_arr - lon0) * cos_lat
    dy = 111.0 * (lat_arr - lat0)
    return np.sqrt(dx*dx + dy*dy)

def to_utc_safe(s):
    if s is None:
        return None
    if isinstance(s, float) and np.isnan(s):
        return None
    s = str(s).strip()
    if s == "":
        return None
    try:
        return UTCDateTime(s)
    except Exception:
        return None

# ============================
# 0. read clusters
# ============================
clust_names = []
events_clustered = []
dtype = [('ot','O'),('lat','O'),('lon','O'),('dep','O'),('mag','O'),('evid','O')]

with open(fclust, 'r') as f:
    lines = f.readlines()

for line in lines:
    if line[0] == '#':
        events_clustered.append([])
        clust_names.append(line[2:-1])
        continue
    codes = line.split(',')
    ot = UTCDateTime(codes[0])
    lat, lon, dep, mag = [float(code) for code in codes[1:5]]
    evid = int(codes[-1])
    if mag < mc:
        continue
    events_clustered[-1].append((ot, lat, lon, dep, mag, evid))

events_clustered = [np.array(clust, dtype=dtype) for clust in events_clustered]
print(f"Loaded {len(events_clustered)} clusters from {fclust}")

# ============================
# 1. read HF disclosure
# ============================
hf = pd.read_csv(f_hf)
hf = hf.dropna(subset=["job_start", "job_end"]).copy()

hf["lat"] = hf["lat"].astype(float)
hf["lon"] = hf["lon"].astype(float)
hf["fluid_volume_gal"] = pd.to_numeric(hf["fluid_volume_gal"], errors="coerce").fillna(0.0)
hf["job_start_utc"] = hf["job_start"].apply(to_utc_safe)
hf["job_end_utc"]   = hf["job_end"].apply(to_utc_safe)
hf = hf[hf["job_start_utc"].notnull() & hf["job_end_utc"].notnull()].copy()

# drop obviously too-long jobs
dur_days = (hf["job_end_utc"] - hf["job_start_utc"]) / 86400.0
hf = hf[dur_days <= max_job_duration_days].copy()

print(f"Loaded {len(hf)} HF jobs from {f_hf} after cleaning and duration filter")

# pre-extract arrays for speed
hf_lats = hf["lat"].values
hf_lons = hf["lon"].values

# ============================
# 2. loop clusters
# ============================
num_hf_clusts, num_hf_events = 0,0
num_all_clusts, num_all_events = 0,0
for iclust, clust in enumerate(events_clustered):
    if len(clust) == 0: continue

    # cluster events
    ev_lats = clust["lat"].astype(float)
    ev_lons = clust["lon"].astype(float)
    ev_times = clust["ot"]

    # ---- per-well selection: well must have >= min_evt_per_well events within radius ----
    kept_rows = []
    for idx, row in hf.iterrows():
        dists_ev = latlon_to_dist_km_vec(ev_lons, ev_lats, row["lon"], row["lat"])
        n_near = np.sum(dists_ev <= max_dist_km)
        if n_near >= min_evt_per_well:
            kept_rows.append(idx)

    hf_near = hf.loc[kept_rows].copy()

    # time window for plotting
    ots = np.array(ev_times)
    tmin = min(ots) - 30*86400
    tmax = max(ots) + 30*86400

    # -------------------------------------------------------------
    # 1) build TRUE daily HF series (no delay) for plotting
    # -------------------------------------------------------------
    daily_vol = {}
    for _, row in hf_near.iterrows():
        st = row["job_start_utc"]
        et = row["job_end_utc"]
        st_dt = max(st.datetime, tmin.datetime)
        et_dt = min(et.datetime, tmax.datetime)
        job_days = pd.date_range(start=st_dt, end=et_dt, freq="D")
        for d in job_days:
            dd = d.date()
            daily_vol[dd] = daily_vol.get(dd, 0.0) + row["fluid_volume_gal"]

    date_index = pd.date_range(start=tmin.datetime, end=tmax.datetime, freq="D")
    hf_daily = pd.Series([daily_vol.get(d.date(), 0.0) for d in date_index],
                         index=date_index)

    # -------------------------------------------------------------
    # 2) episode detection + trimming (time-domain only)
    #    BUT now we also need to know *which* wells were active on that day
    # -------------------------------------------------------------
    active_days = hf_daily > 0.0
    episodes = []
    if active_days.any():
        in_run = False
        start_i = None
        for i, flag in enumerate(active_days):
            if flag and not in_run:
                in_run = True
                start_i = i
            elif not flag and in_run:
                in_run = False
                episodes.append((start_i, i-1))
        if in_run:
            episodes.append((start_i, len(active_days)-1))

    # we will store (st, et, [(lon,lat), ...]) here
    active_periods_for_test = []

    for (i0, i1) in episodes:
        episode_series = hf_daily.iloc[i0:i1+1]
        ep_max = episode_series.max()
        thr = ep_max * episode_trim_factor
        strong_days = episode_series[episode_series >= thr]

        if strong_days.empty:
            continue

        for day_ts, vol in strong_days.items():
            # ---- find which HF wells were active on THIS day (true start/end, no delay) ----
            day_start = UTCDateTime(day_ts.to_pydatetime())
            day_end   = day_start + 86400.0  # 1 day true period

            active_wells = []
            for _, wrow in hf_near.iterrows():
                wst = wrow["job_start_utc"]
                wet = wrow["job_end_utc"]
                if wst <= day_start <= wet or wst <= day_end <= wet or (day_start <= wst and wet <= day_end):
                    active_wells.append( (wrow["lon"], wrow["lat"]) )

            if not active_wells:
                continue  # no well? skip this day

            # classification window = strong day + delay
            st = day_start
            et = day_start + hf_delay_days * 86400.0
            active_periods_for_test.append( (st, et, active_wells) )

    #if not active_periods_for_test:
    #    continue

    # -------------------------------------------------------------
    # 3) count covered events with spatial filtering per episode
    # -------------------------------------------------------------
    n_ev = len(clust)
    n_in = 0
    for ot, elat, elon in zip(ev_times, ev_lats, ev_lons):
        covered = False
        for (st, et, wells_xy) in active_periods_for_test:
            if not (st <= ot <= et):
                continue
            # event is in time → now check distance to wells active in this episode
            for wlon, wlat in wells_xy:
                d = latlon_to_dist_km_vec(np.array([wlon]), np.array([wlat]), elon, elat)[0]
                if d <= max_dist_km:
                    covered = True
                    break
            if covered:
                break
        if covered:
            n_in += 1

    frac_in = n_in / n_ev
    if frac_in >= hf_cover_thresh:
        num_hf_clusts += 1
        num_hf_events += n_ev
    num_all_clusts += 1
    num_all_events += n_ev

    # ============================
    # 4) plotting
    # ============================
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    # make both axes transparent so we can see through
    ax1.patch.set_alpha(0)
    ax2.patch.set_alpha(0)
    
    # put ax2 under ax1
    ax2.set_zorder(1)
    ax1.set_zorder(2)
    
    # 1) orange patches on ax2 (BOTTOM)
    for st, et, _ in active_periods_for_test:
        if et < tmin or st > tmax:
            continue
        ax2.axvspan(
            st.datetime,
            et.datetime,
            color='orange',
            alpha=0.08,
            zorder=1,
        )
    
    # 2) blue HF curve on ax2 (MIDDLE)
    ax2.plot(
        hf_daily.index,
        hf_daily.values,
        color='tab:blue',
        linewidth=1.6,
        zorder=2,
    )
    ax2.set_ylabel("HF volume (gal)")
    
    # 3) black seismicity dots on ax1 (TOP)
    ax1.scatter(
        [t.datetime for t in ev_times],
        clust["mag"],
        s=15,
        c='k',
        alpha=0.8,
        zorder=3,
    )
    ax1.set_ylabel("Magnitude")
    ax1.set_xlim(tmin.datetime, tmax.datetime)
    ax1.grid(True, linestyle='--', alpha=0.4)
    
    # title + save
    cname = clust_names[iclust] if iclust < len(clust_names) else f"cluster_{iclust}"
    title = (f"Cluster {iclust}: {cname}\n"
             f"HF-correlated (per-episode space+time) {n_in}/{n_ev} = {frac_in*100:.1f}%")
    ax1.set_title(title)
    
    ofig = os.path.join(
        outdir_fig,
        f"clust_{iclust:03d}_{cname.replace(' ','_')}.jpg"
    )
    plt.tight_layout()
    plt.savefig(ofig, dpi=300)
    plt.close(fig)
    
    if frac_in >= hf_cover_thresh:
        print(f"✅ {iclust:03d} {cname}: HF-induced ({n_in}/{n_ev}) → {ofig}")
print(num_all_clusts, num_all_events)
print(num_hf_clusts, num_hf_events)
print('num_hf_clusts, num_hf_events = {} ({:.2f}%), {} ({:.2f}%)'.format(num_hf_clusts, 100*num_hf_clusts/num_all_clusts, num_hf_events, 100*num_hf_events/num_all_events))
