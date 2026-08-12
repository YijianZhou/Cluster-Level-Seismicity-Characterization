# -*- coding: utf-8 -*-
"""
Nearest-neighbor (NN) analysis in the sense of Zaliapin & Ben-Zion.

Given a catalog, this script:
  * reads events from the cluster file (same format as before),
  * sorts them in time,
  * for each event j finds its "parent" i<j that minimizes

        η_ij = t_ij * r_ij^d * 10^(-b * m_i)

    where t_ij is the inter-event time (years), r_ij is the 2D epicentral
    distance (km),
    d is the (possibly fractal) dimension, and b is the GR b-value.

  * computes the rescaled time and distance to the parent:

        T_ij = t_ij * 10^(-q * b * m_i)
        R_ij = r_ij^d * 10^(-p * b * m_i)   with p + q = 1

  * saves all results in a NumPy .npz file.

Time unit: years
Space: 2D epicentral (lon/lat) distance in km
"""

import os
import numpy as np
from obspy import UTCDateTime

# =========================
# user parameters
# =========================
# Input catalog (same format you’ve used for clustering)
fclust   = "input/db-seis_4-manual-cluster_Nmin-100.csv"
mc       = 0.3

# Output file
out_npz  = "output/nn_results_all.npz"
os.makedirs(os.path.dirname(out_npz), exist_ok=True)

# NN parameters
b_value  = 1.0
distance_mode = "epicentral_2d"
d_dim    = 1.6       # fractal dimension paired with 2D epicentral distance
p_param  = 0.5       # p + q = 1
q_param  = 0.5
max_back = 5000      # only test last max_back events as candidate parents

# =========================
# helpers
# =========================
def read_clusters_flat(fclust, mc=0.0):
    """
    Read your cluster file and return flat arrays for the whole catalog:
    times (UTCDateTime), lat, lon, depth_km, mag, evid, cluster_index.
    """
    times = []
    lats  = []
    lons  = []
    deps  = []
    mags  = []
    evids = []
    clids = []

    cur_cluster = -1
    with open(fclust, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                # new cluster
                cur_cluster += 1
                continue

            parts = line.split(",")
            ot = UTCDateTime(parts[0])
            lat, lon, dep, mag = map(float, parts[1:5])
            dep *= 1e0  # it was km already in your FF code; keep as km
            evid = int(parts[-1])

            if mag < mc:
                continue

            times.append(ot)
            lats.append(lat)
            lons.append(lon)
            deps.append(dep)
            mags.append(mag)
            evids.append(evid)
            clids.append(cur_cluster)

    times = np.array(times, dtype=object)
    lats  = np.asarray(lats, dtype=float)
    lons  = np.asarray(lons, dtype=float)
    deps  = np.asarray(deps, dtype=float)
    mags  = np.asarray(mags, dtype=float)
    evids = np.asarray(evids, dtype=int)
    clids = np.asarray(clids, dtype=int)

    return times, lats, lons, deps, mags, evids, clids


def utc_to_year(ot_array):
    """
    Convert an array of UTCDateTime objects to decimal years.
    Only the relative scale matters here.
    """
    ot_array = np.asarray(ot_array, dtype=object)
    # seconds since epoch
    secs = np.array([float(t) for t in ot_array], dtype=float)
    year_sec = 365.25 * 86400.0
    return secs / year_sec


def compute_nn_metrics(t_years, lat, lon, dep_km, mag,
                       b, d, p=0.5, q=0.5, max_back=5000,
                       distance_mode="epicentral_2d"):
    """
    Compute nearest-neighbor parent, η, T, R for each event.

    Parameters
    ----------
    t_years : (N,) array of event times in years (sorted ascending)
    lat, lon, dep_km, mag : (N,) arrays
    b, d, p, q : NN parameters
    distance_mode : {"epicentral_2d", "hypocentral_3d"}
        Spatial distance convention paired with ``d``. The configured
        ``d=1.6`` is intended for 2D epicentral distances. Use 3D only with
        a fractal dimension independently estimated from 3D hypocenters.
    max_back : int
        For event j, only the previous max_back events are considered
        as potential parents (computational speed-up).

    Returns
    -------
    parent : (N,) int, index of parent (or -1 for j=0)
    eta    : (N,) float, nearest-neighbor distance η
    T_res  : (N,) float, rescaled time to parent (T)
    R_res  : (N,) float, rescaled distance to parent (R)
    dt_yr  : (N,) float, raw time difference to parent
    r_km   : (N,) float, raw distance to parent under ``distance_mode``
    """

    valid_distance_modes = {"epicentral_2d", "hypocentral_3d"}
    if distance_mode not in valid_distance_modes:
        raise ValueError(
            f"distance_mode must be one of {sorted(valid_distance_modes)}, "
            f"got {distance_mode!r}"
        )

    N = len(t_years)
    parent = -np.ones(N, dtype=int)
    eta    = np.full(N, np.nan, dtype=float)
    T_res  = np.full(N, np.nan, dtype=float)
    R_res  = np.full(N, np.nan, dtype=float)
    dt_yr  = np.full(N, np.nan, dtype=float)
    r_km   = np.full(N, np.nan, dtype=float)

    for j in range(1, N):
        # candidate parents
        i0 = max(0, j - max_back)
        idxs = np.arange(i0, j, dtype=int)
        if idxs.size == 0:
            continue

        # time differences (years)
        dt = t_years[j] - t_years[idxs]
        valid = dt > 0.0
        if not np.any(valid):
            continue

        idxs = idxs[valid]
        dt   = dt[valid]

        # Horizontal epicentral distance (km). Depth is included only when
        # an explicitly paired 3D distance convention is requested.
        lat_j = lat[j]
        lon_j = lon[j]
        dep_j = dep_km[j]

        lat_i = lat[idxs]
        lon_i = lon[idxs]
        dep_i = dep_km[idxs]

        # local metric: 111 km per degree; adjust x by cos(latitude)
        lat_mean_rad = np.deg2rad(0.5 * (lat_j + lat_i))
        dx = 111.0 * (lon_j - lon_i) * np.cos(lat_mean_rad)
        dy = 111.0 * (lat_j - lat_i)
        r2 = dx*dx + dy*dy
        if distance_mode == "hypocentral_3d":
            dz = dep_j - dep_i
            r2 = r2 + dz*dz
        r = np.sqrt(r2)  # km

        # parent magnitudes
        m_par = mag[idxs]

        # η_ij (eq. 1 in Zaliapin & Ben-Zion 2016)
        # η_ij = t_ij * r_ij^d * 10^(-b * m_i)
        eta_candidates = dt * (r ** d) * (10.0 ** (-b * m_par))

        k = np.argmin(eta_candidates)
        i_par = idxs[k]

        parent[j] = i_par
        eta[j]    = eta_candidates[k]
        dt_yr[j]  = dt[k]
        r_km[j]   = r[k]

        # rescaled variables (eq. 5: T_ij, R_ij)
        m_p = mag[i_par]
        T_res[j] = dt_yr[j] * (10.0 ** (-q * b * m_p))
        R_res[j] = (r_km[j] ** d) * (10.0 ** (-p * b * m_p))

        if (j % 10000) == 0:
            print(f"... processed {j}/{N} events")

    return parent, eta, T_res, R_res, dt_yr, r_km


# =========================
# main
# =========================
if __name__ == "__main__":
    print(f"Reading catalog from {fclust}")
    times, lats, lons, deps, mags, evids, clids = read_clusters_flat(fclust, mc=mc)
    print(f"Loaded {len(times)} events with M >= {mc}")

    # sort by time
    order = np.argsort(times.astype(float))
    times  = times[order]
    lats   = lats[order]
    lons   = lons[order]
    deps   = deps[order]
    mags   = mags[order]
    evids  = evids[order]
    clids  = clids[order]

    t_years = utc_to_year(times)

    print(
        "Computing nearest-neighbor metrics with "
        f"distance_mode={distance_mode}, d={d_dim} ..."
    )
    parent, eta, T_res, R_res, dt_yr, r_km = compute_nn_metrics(
        t_years, lats, lons, deps, mags,
        b=b_value, d=d_dim, p=p_param, q=q_param,
        max_back=max_back, distance_mode=distance_mode
    )

    print(f"Saving results to {out_npz}")
    np.savez_compressed(
        out_npz,
        t_years=t_years,
        lat=lats, lon=lons, dep_km=deps,
        mag=mags, evid=evids, cluster=clids,
        parent=parent,
        eta=eta, T=T_res, R=R_res,
        dt_yr=dt_yr, r_km=r_km,
        b=b_value, d=d_dim, p=p_param, q=q_param,
        max_back=max_back,
        distance_mode=distance_mode
    )
    print("Done.")
