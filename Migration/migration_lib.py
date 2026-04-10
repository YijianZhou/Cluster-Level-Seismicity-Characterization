import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
from obspy import UTCDateTime


def read_clusters(fclust, mc=None):
    clust_names, events_clustered = [], []
    dtype = [('ot', 'O'), ('lat', 'O'), ('lon', 'O'), ('dep', 'O'), ('mag', 'O'), ('evid', 'O')]
    with open(fclust) as f:
        lines = f.readlines()
    for line in lines:
        if line[0] == '#':
            events_clustered.append([])
            clust_names.append(line[2:-1])
            continue
        codes = line.split(',')
        ot = UTCDateTime(codes[0])
        lat, lon, dep, mag = [float(code) for code in codes[1:5]]
        dep *= 1e3
        evid = int(codes[-1])
        if mc is not None and mag < mc:
            continue
        events_clustered[-1].append((ot, lat, lon, dep, mag, evid))
    events_clustered = [np.array(clust, dtype=dtype) for clust in events_clustered]
    return events_clustered, clust_names


def get_init_cent(events, init_events=20, win_len_slope=200 * 86400):
    ots = events['ot']
    n0 = max(1, int(len(ots) * 0.05))
    lat_init = np.median(events[:n0]['lat'])
    lon_init = np.median(events[:n0]['lon'])
    dep_init = np.median(events[:n0]['dep'])
    for ii, ot in enumerate(ots):
        if ii + init_events == len(ots) or ots[ii + init_events] - ot < win_len_slope:
            return ot, lat_init, lon_init, dep_init
    return ots[0], lat_init, lon_init, dep_init


def calc_dist_3d(events, lat_init, lon_init, dep_init):
    dist_3d = []
    cos_lat = np.cos(np.pi / 180 * np.median(events['lat']))
    for [_, lat, lon, dep, _, _] in events:
        dx = 111 * (lon - lon_init) * cos_lat
        dy = 111 * (lat - lat_init)
        dz = (dep - dep_init) / 1e3
        dist_3d.append(1e3 * (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5)
    return np.asarray(dist_3d, dtype=float)


def _bootstrap_single_window_quantiles(args):
    i, data, quantiles, n_bootstrap, min_events_contour, seed = args
    rng = np.random.default_rng(seed)
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    n = data.size
    nq = len(quantiles)
    D_row = np.full(nq, np.nan, dtype=float)
    s_row = np.full(nq, np.nan, dtype=float)
    valid_row = np.zeros(nq, dtype=bool)
    if n == 0:
        return i, D_row, s_row, valid_row, n
    if n == 1:
        D_row[:] = data[0]
        return i, D_row, s_row, valid_row, n
    sample_idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot_samples = data[sample_idx]
    boot_vals = np.quantile(boot_samples, quantiles, axis=1).T
    D_row[:] = np.nanmedian(boot_vals, axis=0)
    s_row[:] = np.nanstd(boot_vals, axis=0, ddof=1)
    if n >= min_events_contour:
        valid_row[:] = True
    return i, D_row, s_row, valid_row, n


def bootstrap_Dij_for_windows(data_list, quantiles=np.arange(0.05, 1.00, 0.05), n_bootstrap=200,
                              min_events_contour=10, random_state=12345, n_jobs=None, s0_frac=0.02):
    quantiles = np.asarray(quantiles, dtype=float)
    n_win = len(data_list)
    n_q = len(quantiles)
    D_ij = np.full((n_win, n_q), np.nan, dtype=float)
    s_ij = np.full((n_win, n_q), np.nan, dtype=float)
    w_ij = np.zeros((n_win, n_q), dtype=float)
    valid_mask = np.zeros((n_win, n_q), dtype=bool)
    num_events = np.zeros(n_win, dtype=int)

    ss = np.random.SeedSequence(random_state)
    seed_ints = [int(cs.generate_state(1, dtype=np.uint32)[0]) for cs in ss.spawn(max(1, n_win))]
    tasks = [(i, data_list[i], quantiles, n_bootstrap, min_events_contour, seed_ints[i]) for i in range(n_win)]
    if n_jobs is None:
        n_jobs = max(1, min(os.cpu_count() or 1, n_win))
    n_jobs = max(1, min(int(n_jobs), max(1, n_win)))
    if n_jobs == 1 or n_win <= 1:
        results = [_bootstrap_single_window_quantiles(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            results = list(ex.map(_bootstrap_single_window_quantiles, tasks))
    for i, D_row, s_row, valid_row, num_i in results:
        D_ij[i, :] = D_row
        s_ij[i, :] = s_row
        valid_mask[i, :] = valid_row
        num_events[i] = num_i

    valid_rows = np.where(num_events >= min_events_contour)[0]
    if valid_rows.size > 0:
        dmin = np.nanmin(D_ij[valid_rows, 0])
        dmax = np.nanmax(D_ij[valid_rows, -1])
        L_ref = float(dmax - dmin)
    else:
        L_ref = np.nan

    if np.isfinite(L_ref) and L_ref > 0:
        s0 = s0_frac * L_ref
        denom = s_ij ** 2 + s0 ** 2
        with np.errstate(divide='ignore', invalid='ignore'):
            w_ij = 1.0 / denom
        w_ij[~np.isfinite(w_ij)] = 0.0
        w_ij[~valid_mask] = 0.0
    return {
        'D_ij': D_ij, 's_ij': s_ij, 'w_ij': w_ij, 'num_events': num_events,
        'L_ref': L_ref, 'quantiles': quantiles, 'valid_mask': valid_mask,
    }


def build_sliding_window_lists(start_time, end_time, events, value_array, win_len_contour=100 * 86400,
                               win_step_contour=10 * 86400):
    num_win = int((end_time - start_time) / win_step_contour) + 2
    t0_list = np.array([start_time - win_len_contour / 2 + ii * win_step_contour for ii in range(num_win)], dtype=object)
    cent_time_out = np.array([start_time + ii * win_step_contour for ii in range(num_win)], dtype=object)
    rel_days_plot = np.array([ii * win_step_contour / 86400.0 for ii in range(num_win)], dtype=float)
    data_list, event_idx_list = [], []
    num_events = np.zeros(num_win, dtype=int)
    for i, t0 in enumerate(t0_list):
        cond_ot = (events['ot'] >= t0) * (events['ot'] <= t0 + win_len_contour)
        idx = np.where(cond_ot)[0]
        vals_i = np.asarray(value_array[cond_ot], dtype=float)
        vals_i = vals_i[np.isfinite(vals_i)]
        vals_i = np.sort(vals_i)
        data_list.append(vals_i)
        event_idx_list.append(idx)
        num_events[i] = idx.size
    return {
        'data_list': data_list, 'event_idx_list': event_idx_list, 't0_list': t0_list,
        'cent_time_out': cent_time_out, 'rel_days_plot': rel_days_plot, 'num_events': num_events,
    }


def build_contour_cache(start_time, end_time, events, dist_3d, quantiles=np.arange(0.05, 1.00, 0.05),
                        win_len_contour=100 * 86400, win_step_contour=10 * 86400, min_events_contour=10,
                        n_bootstrap=200, random_state=12345, n_jobs=None, s0_frac=0.02):
    dep_win = build_sliding_window_lists(start_time, end_time, events, events['dep'], win_len_contour, win_step_contour)
    dist_win = build_sliding_window_lists(start_time, end_time, events, dist_3d, win_len_contour, win_step_contour)
    dep = bootstrap_Dij_for_windows(dep_win['data_list'], quantiles, n_bootstrap, min_events_contour, random_state, n_jobs, s0_frac)
    dist = bootstrap_Dij_for_windows(dist_win['data_list'], quantiles, n_bootstrap, min_events_contour, random_state + 1000003, n_jobs, s0_frac)
    blank_mask = dep_win['num_events'] < min_events_contour
    return {
        'start_time': start_time, 'end_time': end_time, 'events': events, 'dist_3d': np.asarray(dist_3d, dtype=float),
        'quantiles': np.asarray(quantiles, dtype=float), 'win_len_contour': win_len_contour,
        'win_step_contour': win_step_contour, 'min_events_contour': min_events_contour,
        'window': dep_win, 'dep': dep, 'dist': dist, 'blank_mask': blank_mask,
        'blank_cent_times': dep_win['cent_time_out'][blank_mask],
    }


def joint_weighted_mig_fit(D_ij, s_ij, w_ij, rel_days, quantiles, min_fit_windows=4):
    rel_days = np.asarray(rel_days, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    z = 2.0 * quantiles - 1.0
    rows, d_list, w_list = [], [], []
    used_windows = set()
    n_win, n_q = D_ij.shape
    for i in range(n_win):
        t = rel_days[i]
        row_used = False
        for j in range(n_q):
            Dij = D_ij[i, j]
            wij = w_ij[i, j]
            if not np.isfinite(Dij):
                continue
            if not np.isfinite(wij) or wij <= 0:
                continue
            zj = z[j]
            rows.append([1.0, t, zj, t * zj])
            d_list.append(Dij)
            w_list.append(wij)
            row_used = True
        if row_used:
            used_windows.add(i)
    if len(used_windows) < min_fit_windows or len(d_list) < 4:
        raise ValueError('Not enough valid windows/observations to fit 4 parameters.')
    G = np.asarray(rows, dtype=float)
    d = np.asarray(d_list, dtype=float)
    W_diag = np.asarray(w_list, dtype=float)
    sqrtW = np.sqrt(W_diag)
    Gw = G * sqrtW[:, None]
    dw = d * sqrtW
    GTWG = Gw.T @ Gw
    GTWd = Gw.T @ dw
    m_hat = np.linalg.solve(GTWG, GTWd)
    residuals = d - G @ m_hat
    dof = max(len(d) - 4, 1)
    sigma2_hat = np.sum(W_diag * residuals**2) / dof
    cov = sigma2_hat * np.linalg.inv(GTWG)
    std = np.sqrt(np.diag(cov))
    return {'params': {'m0': m_hat[0], 'vm': m_hat[1], 'w0': m_hat[2], 'vw': m_hat[3]},
            'param_std': {'m0_std': std[0], 'vm_std': std[1], 'w0_std': std[2], 'vw_std': std[3]},
            'cov': cov, 'used_window_count': len(used_windows)}


def _fit_one_domain_from_cache(domain_cache, rel_days, idx_sel, min_fit_windows=4):
    idx_sel = np.asarray(idx_sel, dtype=int)
    return joint_weighted_mig_fit(domain_cache['D_ij'][idx_sel, :], domain_cache['s_ij'][idx_sel, :],
                                  domain_cache['w_ij'][idx_sel, :], np.asarray(rel_days, dtype=float)[idx_sel],
                                  domain_cache['quantiles'], min_fit_windows)


def _vel_std_from_cov(cov, z):
    var = cov[1, 1] + (z ** 2) * cov[3, 3] + 2.0 * z * cov[1, 3]
    return float(np.sqrt(max(var, 0.0)))



def _trim_interval_to_events(cache, idx_sel, t_start, t_end):
    """Trim an interval to actual event times contributed by windows whose center times lie in [t_start, t_end]."""
    idx_sel = np.asarray(idx_sel, dtype=int)
    cent = cache['window']['cent_time_out']
    idx_keep = idx_sel[(cent[idx_sel] >= t_start) & (cent[idx_sel] <= t_end)]
    if idx_keep.size == 0:
        return t_start, t_end
    event_idx = []
    for ii in idx_keep:
        event_idx.extend(cache['window']['event_idx_list'][ii].tolist())
    if not event_idx:
        return t_start, t_end
    event_idx = np.unique(np.asarray(event_idx, dtype=int))
    ots = cache['events']['ot'][event_idx]
    ots = ots[(ots >= t_start) & (ots <= t_end)]
    if ots.size == 0:
        return t_start, t_end
    return max(t_start, np.min(ots)), min(t_end, np.max(ots))


def fit_segment_from_cache(cache, idx_sel, min_fit_windows=4, time_start=None, time_end=None):
    idx_sel = np.asarray(idx_sel, dtype=int)
    rel_days = cache['window']['rel_days_plot']
    dep_fit = _fit_one_domain_from_cache(cache['dep'], rel_days, idx_sel, min_fit_windows)
    dist_fit = _fit_one_domain_from_cache(cache['dist'], rel_days, idx_sel, min_fit_windows)
    qmap = {round(float(q), 2): 2.0 * float(q) - 1.0 for q in cache['quantiles']}
    z05, z50, z95 = qmap.get(0.05, -0.9), qmap.get(0.50, 0.0), qmap.get(0.95, 0.9)
    if time_start is None:
        time_start = cache['window']['cent_time_out'][idx_sel[0]]
    if time_end is None:
        time_end = cache['window']['cent_time_out'][idx_sel[-1]]
    time_start, time_end = _trim_interval_to_events(cache, idx_sel, time_start, time_end)
    t_rel0 = (time_start - cache['start_time']) / 86400.0
    t_rel1 = (time_end - cache['start_time']) / 86400.0

    def _pred(fit_out, z, t):
        m0, vm, w0, vw = fit_out['params']['m0'], fit_out['params']['vm'], fit_out['params']['w0'], fit_out['params']['vw']
        return m0 + vm * t + (w0 + vw * t) * z

    def _domain_summary(fit_out):
        vm, vw = fit_out['params']['vm'], fit_out['params']['vw']
        v05 = float(vm + z05 * vw)
        v50 = float(vm + z50 * vw)
        v95 = float(vm + z95 * vw)

        d05_0 = _pred(fit_out, z05, t_rel0)
        d05_1 = _pred(fit_out, z05, t_rel1)
        d50_0 = _pred(fit_out, z50, t_rel0)
        d50_1 = _pred(fit_out, z50, t_rel1)
        d95_0 = _pred(fit_out, z95, t_rel0)
        d95_1 = _pred(fit_out, z95, t_rel1)

        half_width0 = abs(d95_0 - d50_0)
        if half_width0 <= 0:
            rel_len05 = np.nan
            rel_len95 = np.nan
        else:
            rel_len05 = abs(d05_1 - d05_0) / half_width0
            rel_len95 = abs(d95_1 - d95_0) / half_width0

        return {
            'fit': fit_out,
            'v05': v05, 'v50': v50, 'v95': v95,
            'v05_std': _vel_std_from_cov(fit_out['cov'], z05),
            'v50_std': _vel_std_from_cov(fit_out['cov'], z50),
            'v95_std': _vel_std_from_cov(fit_out['cov'], z95),
            'mig_len05': float(d05_1 - d05_0),
            'mig_len95': float(d95_1 - d95_0),
            'rel_mig_len05': float(rel_len05) if np.isfinite(rel_len05) else np.nan,
            'rel_mig_len95': float(rel_len95) if np.isfinite(rel_len95) else np.nan,
        }

    def _xy_from_fit(fit_out):
        d05_0 = _pred(fit_out, z05, t_rel0)
        d05_1 = _pred(fit_out, z05, t_rel1)
        d95_0 = _pred(fit_out, z95, t_rel0)
        d95_1 = _pred(fit_out, z95, t_rel1)
        return {'xy05': np.array([t_rel0, t_rel1, d05_0, d05_1], dtype=float),
                'xy95': np.array([t_rel0, t_rel1, d95_0, d95_1], dtype=float)}

    dep_sum = _domain_summary(dep_fit)
    dist_sum = _domain_summary(dist_fit)
    dep_xy = _xy_from_fit(dep_fit)
    dist_xy = _xy_from_fit(dist_fit)
    return {
        'idx_sel': idx_sel, 'time_start': time_start, 'time_end': time_end,
        'rel_day_range': (t_rel0, t_rel1),
        'dep': dep_sum, 'dist': dist_sum,
        'xy_dep_dist': [dep_xy['xy05'], dep_xy['xy95'], dist_xy['xy05'], dist_xy['xy95']],
        'v_dep_dist_list': [dep_sum['v05'], dep_sum['v50'], dep_sum['v95'], dist_sum['v05'], dist_sum['v50'], dist_sum['v95']],
        'v_dep_dist_std_list': [dep_sum['v05_std'], dep_sum['v50_std'], dep_sum['v95_std'], dist_sum['v05_std'], dist_sum['v50_std'], dist_sum['v95_std']],
        'mig_len05_dep': dep_sum['mig_len05'], 'mig_len95_dep': dep_sum['mig_len95'],
        'mig_len05_dist': dist_sum['mig_len05'], 'mig_len95_dist': dist_sum['mig_len95'],
    }


def _segment_predicted_quantile_line(seg_fit, t_rel_days, z):
    """
    Predict D_q(t) from an existing segment fit at arbitrary relative times (days since cache['start_time']).
    """
    p = seg_fit['fit']['params']
    return p['m0'] + p['vm'] * t_rel_days + (p['w0'] + p['vw'] * t_rel_days) * z


def compute_segment_offset_ratios(seg, cache):
    """
    For one fitted segment, compute off-set point ratios using the Dxx contour points:
      - D95 points above fitted D95 line
      - D05 points below fitted D05 line

    Returns a nested dict:
      out['dep']['q05'], out['dep']['q95'], out['dist']['q05'], out['dist']['q95']
    Each entry contains:
      n_off, n_all, ratio
    """
    q = np.asarray(cache['quantiles'], dtype=float)
    idx05 = int(np.argmin(np.abs(q - 0.05)))
    idx95 = int(np.argmin(np.abs(q - 0.95)))
    z05 = 2.0 * float(q[idx05]) - 1.0
    z95 = 2.0 * float(q[idx95]) - 1.0

    idx_sel = np.asarray(seg['idx_sel'], dtype=int)
    t_rel = np.asarray(cache['window']['rel_days_plot'], dtype=float)[idx_sel]

    out = {}
    for domain_name in ['dep', 'dist']:
        dom = cache[domain_name]
        D = np.asarray(dom['D_ij'], dtype=float)[idx_sel, :]
        W = np.asarray(dom['w_ij'], dtype=float)[idx_sel, :]

        obs05 = D[:, idx05]
        obs95 = D[:, idx95]
        valid05 = np.isfinite(obs05) & np.isfinite(W[:, idx05]) & (W[:, idx05] > 0)
        valid95 = np.isfinite(obs95) & np.isfinite(W[:, idx95]) & (W[:, idx95] > 0)

        pred05 = _segment_predicted_quantile_line(seg[domain_name], t_rel, z05)
        pred95 = _segment_predicted_quantile_line(seg[domain_name], t_rel, z95)

        n_all_05 = int(np.sum(valid05))
        n_all_95 = int(np.sum(valid95))
        n_off_05 = int(np.sum(obs05[valid05] < pred05[valid05])) if n_all_05 > 0 else 0
        n_off_95 = int(np.sum(obs95[valid95] > pred95[valid95])) if n_all_95 > 0 else 0

        out[domain_name] = {
            'q05': {
                'n_off': n_off_05,
                'n_all': n_all_05,
                'ratio': (n_off_05 / n_all_05) if n_all_05 > 0 else np.nan,
            },
            'q95': {
                'n_off': n_off_95,
                'n_all': n_all_95,
                'ratio': (n_off_95 / n_all_95) if n_all_95 > 0 else np.nan,
            },
        }
    return out


def velo_to_mig(v_dep_dist_list, min_velo_exp, min_velo_shr, max_velo_exp=None):
    v_d05_dep, _, v_d95_dep, v_d05_dist, v_d50_dist, v_d95_dist = v_dep_dist_list
    is_dep_exp, is_dist_exp, is_mig = 0, 0, 0

    dep05_ok = (-v_d05_dep >= min_velo_exp) and (max_velo_exp is None or -v_d05_dep <= max_velo_exp)
    dep95_ok = (v_d95_dep >= min_velo_exp) and (max_velo_exp is None or v_d95_dep <= max_velo_exp)
    dist95_ok = (v_d95_dist >= min_velo_exp) and (max_velo_exp is None or v_d95_dist <= max_velo_exp)

    if dep05_ok and not dep95_ok:
        is_dep_exp = 1
    elif (not dep05_ok) and dep95_ok:
        is_dep_exp = 2
    elif dep05_ok and dep95_ok:
        is_dep_exp = 3
    if v_d95_dep - v_d05_dep < -min_velo_shr:
        is_dep_exp = -1
    if dist95_ok:
        is_dist_exp = 1
    if v_d95_dist - v_d05_dist < -min_velo_shr:
        is_dist_exp = -1
    if is_dist_exp == 1 and v_d50_dist < 0:
        is_dist_exp = 0
    if is_dep_exp in [1, 3] and is_dist_exp == 0:
        is_mig = 1
    if is_dep_exp == 2 and is_dist_exp == 0:
        is_mig = 2
    if is_dist_exp == 1 and is_dep_exp == 0:
        is_mig = 3
    if is_dep_exp in [1, 3] and is_dist_exp == 1:
        is_mig = 4
    if is_dep_exp == 2 and is_dist_exp == 1:
        is_mig = 5
    if is_dep_exp < 0 or is_dist_exp < 0:
        is_mig = 0
    return is_dep_exp, is_dist_exp, is_mig


def _type_code(is_dep_exp, is_dist_exp):
    if is_dep_exp > 0 and is_dist_exp > 0:
        return 'both'
    if is_dep_exp > 0 and is_dist_exp == 0:
        return 'dep_only'
    if is_dist_exp > 0 and is_dep_exp == 0:
        return 'dist_only'
    return 'none'


def _is_reversed_dep(dep_a, dep_b):
    return (dep_a == 1 and dep_b == 2) or (dep_a == 2 and dep_b == 1)


def compute_sliding_velocity_series(cache, num_win_mig, min_velo_exp=1.0, min_velo_shr=1.0,
                                    max_velo_exp=None, min_fit_windows=4):
    rel_days = cache['window']['rel_days_plot']
    n_win = len(rel_days)
    if n_win < num_win_mig:
        return {'segments': [], 'v_d05_dep': np.array([]), 'v_d50_dep': np.array([]), 'v_d95_dep': np.array([]),
                'v_d05_dist': np.array([]), 'v_d50_dist': np.array([]), 'v_d95_dist': np.array([]),
                'is_dep_exp': np.array([], dtype=int), 'is_dist_exp': np.array([], dtype=int), 'is_mig': np.array([], dtype=int)}
    segments = []
    for i0 in range(0, n_win - num_win_mig + 1):
        idx_sel = np.arange(i0, i0 + num_win_mig)
        t_start = cache['window']['cent_time_out'][i0] - cache['win_len_contour'] / 2
        t_end = cache['window']['cent_time_out'][i0 + num_win_mig - 1] + cache['win_len_contour'] / 2
        try:
            seg_fit = fit_segment_from_cache(cache, idx_sel, min_fit_windows, t_start, t_end)
            is_dep_exp, is_dist_exp, is_mig = velo_to_mig(
                seg_fit['v_dep_dist_list'], min_velo_exp, min_velo_shr, max_velo_exp
            )
        except Exception:
            seg_fit = None
            is_dep_exp, is_dist_exp, is_mig = 0, 0, 0
        segments.append({'i0': i0, 'i1': i0 + num_win_mig - 1, 'idx_sel': idx_sel, 'time_start': t_start, 'time_end': t_end,
                         'is_dep_exp': is_dep_exp, 'is_dist_exp': is_dist_exp, 'is_mig': is_mig, 'fit_result': seg_fit})

    def _pull(k):
        vals = []
        for seg in segments:
            vals.append(np.nan if seg['fit_result'] is None else seg['fit_result']['v_dep_dist_list'][k])
        return np.asarray(vals, dtype=float)

    return {
        'segments': segments,
        'v_d05_dep': _pull(0), 'v_d50_dep': _pull(1), 'v_d95_dep': _pull(2),
        'v_d05_dist': _pull(3), 'v_d50_dist': _pull(4), 'v_d95_dist': _pull(5),
        'is_dep_exp': np.asarray([s['is_dep_exp'] for s in segments], dtype=int),
        'is_dist_exp': np.asarray([s['is_dist_exp'] for s in segments], dtype=int),
        'is_mig': np.asarray([s['is_mig'] for s in segments], dtype=int),
    }



def _summarize_seg_group_type(seg_group):
    dep_types = [seg['is_dep_exp'] for seg in seg_group if seg['is_dep_exp'] > 0]
    dist_types = [seg['is_dist_exp'] for seg in seg_group if seg['is_dist_exp'] > 0]

    if 3 in dep_types:
        dep_type = 3
    elif 1 in dep_types and 2 in dep_types:
        dep_type = 3
    elif len(dep_types) > 0:
        dep_type = dep_types[-1]
    else:
        dep_type = 0

    dist_type = 1 if len(dist_types) > 0 else 0

    is_mig = 0
    if dep_type in [1, 3] and dist_type == 0:
        is_mig = 1
    elif dep_type == 2 and dist_type == 0:
        is_mig = 2
    elif dist_type == 1 and dep_type == 0:
        is_mig = 3
    elif dep_type in [1, 3] and dist_type == 1:
        is_mig = 4
    elif dep_type == 2 and dist_type == 1:
        is_mig = 5

    return dep_type, dist_type, is_mig


def split_period_by_blank_windows(period, cache, min_mig_len):
    blank_times = sorted([t for t in cache['blank_cent_times'] if period['time_start'] < t < period['time_end']])
    if not blank_times:
        return [period] if (period['time_end'] - period['time_start'] >= min_mig_len) else []

    bounds = [period['time_start']] + blank_times + [period['time_end']]
    out = []
    for ta, tb in zip(bounds[:-1], bounds[1:]):
        if tb - ta < min_mig_len:
            continue
        pp = dict(period)
        pp['time_start'] = ta
        pp['time_end'] = tb
        out.append(pp)
    return out



def _combine_type_pair(dep_a, dist_a, dep_b, dist_b):
    dep_types = [d for d in [dep_a, dep_b] if d > 0]
    if 3 in dep_types:
        dep_out = 3
    elif 1 in dep_types and 2 in dep_types:
        dep_out = 3
    elif len(dep_types) > 0:
        dep_out = dep_types[-1]
    else:
        dep_out = 0
    dist_out = 1 if (dist_a > 0 or dist_b > 0) else 0
    is_mig = 0
    if dep_out in [1, 3] and dist_out == 0:
        is_mig = 1
    elif dep_out == 2 and dist_out == 0:
        is_mig = 2
    elif dist_out == 1 and dep_out == 0:
        is_mig = 3
    elif dep_out in [1, 3] and dist_out == 1:
        is_mig = 4
    elif dep_out == 2 and dist_out == 1:
        is_mig = 5
    return dep_out, dist_out, is_mig


def _candidate_hard_break(pa, pb):
    reversed_dep = _is_reversed_dep(pa['is_dep_exp'], pb['is_dep_exp'])
    ta = _type_code(pa['is_dep_exp'], pa['is_dist_exp'])
    tb = _type_code(pb['is_dep_exp'], pb['is_dist_exp'])
    hard_type_break = ((ta == 'dep_only' and tb == 'dist_only') or (ta == 'dist_only' and tb == 'dep_only'))
    return reversed_dep or hard_type_break



def merge_candidate_migration_periods(sliding_out, cache, min_mig_len):
    mig_segs = [seg for seg in sliding_out['segments'] if seg['is_mig'] > 0]
    if not mig_segs:
        return []

    # Step 1: merge raw sliding segments by overlap and hard-break rules
    merged_groups = []
    cur_group = [mig_segs[0]]
    cur_dep_type = mig_segs[0]['is_dep_exp']
    cur_dist_type = mig_segs[0]['is_dist_exp']
    for nxt in mig_segs[1:]:
        cur_last = cur_group[-1]
        overlaps = nxt['time_start'] <= cur_last['time_end']
        reversed_dep = _is_reversed_dep(cur_dep_type, nxt['is_dep_exp'])
        cur_type = _type_code(cur_dep_type, cur_dist_type)
        nxt_type = _type_code(nxt['is_dep_exp'], nxt['is_dist_exp'])
        hard_type_break = ((cur_type == 'dep_only' and nxt_type == 'dist_only') or
                           (cur_type == 'dist_only' and nxt_type == 'dep_only'))
        if overlaps and (not reversed_dep) and (not hard_type_break):
            cur_group.append(nxt)
            cur_dep_type, cur_dist_type, _ = _combine_type_pair(
                cur_dep_type, cur_dist_type, nxt['is_dep_exp'], nxt['is_dist_exp']
            )
        else:
            merged_groups.append(cur_group)
            cur_group = [nxt]
            cur_dep_type = nxt['is_dep_exp']
            cur_dist_type = nxt['is_dist_exp']
    merged_groups.append(cur_group)

    # Step 2: split each merged group by blank-win center times in abs-time space,
    # then trim each piece to real event-supported abs time before creating candidates.
    candidates = []
    for group in merged_groups:
        group_start = min(seg['time_start'] for seg in group)
        group_end = max(seg['time_end'] for seg in group)

        # hard split by blank-win center times
        bounds = [group_start] + sorted([t for t in cache['blank_cent_times'] if group_start < t < group_end]) + [group_end]

        for ta, tb in zip(bounds[:-1], bounds[1:]):
            if tb - ta < min_mig_len:
                continue

            # collect raw mig segs that overlap this non-blank interval
            sub_group = [seg for seg in group if seg['time_end'] > ta and seg['time_start'] < tb]
            if not sub_group:
                continue

            # trim to real event-supported abs time using only windows whose cent-times
            # lie inside the current blank-split interval
            event_idx = []
            for seg in sub_group:
                for ii in range(seg['i0'], seg['i1'] + 1):
                    ct = cache['window']['cent_time_out'][ii]
                    if ta <= ct <= tb:
                        event_idx.extend(cache['window']['event_idx_list'][ii].tolist())

            if not event_idx:
                continue

            event_idx = np.unique(np.asarray(event_idx, dtype=int))
            ots = cache['events']['ot'][event_idx]
            ots = ots[(ots >= ta) & (ots <= tb)]
            if ots.size == 0:
                continue

            final_start = max(ta, np.min(ots))
            final_end = min(tb, np.max(ots))
            if final_end - final_start < min_mig_len:
                continue

            dep_type, dist_type, is_mig = _summarize_seg_group_type(sub_group)
            candidates.append({
                'time_start': final_start,
                'time_end': final_end,
                'is_dep_exp': dep_type,
                'is_dist_exp': dist_type,
                'is_mig': is_mig,
                'source_i0': min(seg['i0'] for seg in sub_group),
                'source_i1': max(seg['i1'] for seg in sub_group),
            })

    candidates = sorted(
        candidates,
        key=lambda p: (float(p['time_start'].timestamp), float(p['time_end'].timestamp), p['source_i0'], p['source_i1'])
    )

    return candidates


def _window_indices_within_time(cache, t_start, t_end):
    """Return contour-window indices whose center times fall within [t_start, t_end]."""
    cent = cache['window']['cent_time_out']
    return np.where((cent >= t_start) & (cent <= t_end))[0]



def _rebuild_mig_type(is_dep_exp, is_dist_exp):
    is_mig = 0
    if is_dep_exp in [1, 3] and is_dist_exp == 0:
        is_mig = 1
    if is_dep_exp == 2 and is_dist_exp == 0:
        is_mig = 2
    if is_dist_exp == 1 and is_dep_exp == 0:
        is_mig = 3
    if is_dep_exp in [1, 3] and is_dist_exp == 1:
        is_mig = 4
    if is_dep_exp == 2 and is_dist_exp == 1:
        is_mig = 5
    return is_mig


def _dirs_from_class(is_dep_exp, is_dist_exp):
    dirs = set()
    if is_dep_exp == 1:
        dirs.add('dep_up')
    elif is_dep_exp == 2:
        dirs.add('dep_down')
    elif is_dep_exp == 3:
        dirs.update(['dep_up', 'dep_down'])
    if is_dist_exp == 1:
        dirs.add('dist')
    return dirs


def _whole_has_shrink(vlist, min_velo_shr):
    v_d05_dep, _, v_d95_dep, v_d05_dist, _, v_d95_dist = vlist
    dep_shrink = (v_d95_dep - v_d05_dep) < -min_velo_shr
    dist_shrink = (v_d95_dist - v_d05_dist) < -min_velo_shr
    return dep_shrink or dist_shrink


def _filter_dirs_by_off_ratio(dirs, off_ratio, max_off_ratio=None):
    dirs = set(dirs)
    if not dirs or max_off_ratio is None or off_ratio is None:
        return dirs

    keep = set()
    for d in dirs:
        if d == 'dep_up':
            ratio = off_ratio.get('dep', {}).get('q05', {}).get('ratio', np.nan)
        elif d == 'dep_down':
            ratio = off_ratio.get('dep', {}).get('q95', {}).get('ratio', np.nan)
        elif d == 'dist':
            ratio = off_ratio.get('dist', {}).get('q95', {}).get('ratio', np.nan)
        else:
            ratio = np.nan
        if (not np.isfinite(ratio)) or ratio <= max_off_ratio:
            keep.add(d)
    return keep


def _interval_debug_line(tag, item):
    if item is None or item.get('seg') is None:
        return f'[postcheck] {tag}: seg=None'
    v = item['seg']['v_dep_dist_list']
    shrink = item.get('shrink', False)
    return (f'[postcheck] {tag}: pass={item.get("pass", False)} shrink={shrink} '
            f'dep={item.get("is_dep_exp", 0)} dist={item.get("is_dist_exp", 0)} '
            f'mig={item.get("is_mig", 0)} nwin={item.get("nwin", 0)} '
            f'vdep05={v[0]:.2f} vdep50={v[1]:.2f} vdep95={v[2]:.2f} '
            f'vdist05={v[3]:.2f} vdist50={v[4]:.2f} vdist95={v[5]:.2f}')


def _fit_interval_item(cache, ta, tb, min_fit_windows, min_velo_exp, max_velo_exp, min_velo_shr,
                       max_off_ratio=None):
    idx = _window_indices_within_time(cache, ta, tb)
    item = {'time_start': ta, 'time_end': tb, 'idx': idx, 'seg': None,
            'is_dep_exp': 0, 'is_dist_exp': 0, 'is_mig': 0,
            'dirs': set(), 'pass': False, 'shrink': False, 'nwin': int(idx.size), 'off_ratio': None}
    if idx.size < min_fit_windows:
        return item
    try:
        seg = fit_segment_from_cache(cache, idx, min_fit_windows, ta, tb)
    except Exception:
        return item
    vlist = seg['v_dep_dist_list']
    is_dep_exp, is_dist_exp, is_mig = velo_to_mig(vlist, min_velo_exp, min_velo_shr, max_velo_exp)
    shrink = _whole_has_shrink(vlist, min_velo_shr)
    dirs = _dirs_from_class(is_dep_exp, is_dist_exp) if is_mig > 0 else set()
    if shrink:
        dirs = set()
    off_ratio = compute_segment_offset_ratios(seg, cache) if max_off_ratio is not None else None
    dirs = _filter_dirs_by_off_ratio(dirs, off_ratio, max_off_ratio)
    is_dep_exp, is_dist_exp, is_mig = _codes_from_dirs(dirs)
    item.update({
        'seg': seg,
        'is_dep_exp': is_dep_exp,
        'is_dist_exp': is_dist_exp,
        'is_mig': is_mig,
        'dirs': dirs,
        'pass': is_mig > 0,
        'shrink': shrink,
        'off_ratio': off_ratio,
    })
    return item


def _filter_surviving_dirs(seg, dirs, max_velo_exp=None, off_ratio=None, max_off_ratio=None):
    dirs = set(dirs)
    if not dirs:
        return dirs
    v = seg['v_dep_dist_list']
    keep = set()
    for d in dirs:
        ok = True
        if max_velo_exp is not None:
            if d == 'dep_up' and (-v[0] > max_velo_exp):
                ok = False
            elif d == 'dep_down' and (v[2] > max_velo_exp):
                ok = False
            elif d == 'dist' and (v[5] > max_velo_exp):
                ok = False
        if ok and max_off_ratio is not None:
            if d == 'dep_up':
                ratio = off_ratio.get('dep', {}).get('q05', {}).get('ratio', np.nan) if off_ratio is not None else np.nan
            elif d == 'dep_down':
                ratio = off_ratio.get('dep', {}).get('q95', {}).get('ratio', np.nan) if off_ratio is not None else np.nan
            else:
                ratio = off_ratio.get('dist', {}).get('q95', {}).get('ratio', np.nan) if off_ratio is not None else np.nan
            ok = (not np.isfinite(ratio)) or ratio <= max_off_ratio
        if ok:
            keep.add(d)
    return keep


def _codes_from_dirs(dirs):
    dep_dirs = {d for d in dirs if d.startswith('dep_')}
    if dep_dirs == {'dep_up'}:
        dep_code = 1
    elif dep_dirs == {'dep_down'}:
        dep_code = 2
    elif dep_dirs == {'dep_up', 'dep_down'}:
        dep_code = 3
    else:
        dep_code = 0
    dist_code = 1 if 'dist' in dirs else 0
    return dep_code, dist_code, _rebuild_mig_type(dep_code, dist_code)


def _scan_subsegments(cache, start_time, end_time, seg_len, win_step_slope, min_fit_windows,
                      min_velo_exp, max_velo_exp, min_velo_shr, whole_dirs,
                      whole_dep_type=0, require_no_reverse_dep=False,
                      detailed=False, tag='subseg'):
    max_start = end_time - seg_len
    starts = []
    cur_t0 = start_time
    while cur_t0 <= max_start + 1e-6:
        starts.append(cur_t0)
        cur_t0 += win_step_slope
    if (len(starts) == 0) or (starts[-1] < max_start - 1e-6):
        starts.append(max_start)

    subsegments = []
    failed_centers = []
    failed_indices = []
    for kk, cur_t0 in enumerate(starts):
        cur_t1 = min(cur_t0 + seg_len, end_time)
        item = _fit_interval_item(
            cache, cur_t0, cur_t1, min_fit_windows,
            min_velo_exp,
            max_velo_exp,
            min_velo_shr,
        )
        item['whole_consistent_dirs'] = item['dirs'].intersection(whole_dirs)
        item['reversed_dep'] = require_no_reverse_dep and _is_reversed_dep(item['is_dep_exp'], whole_dep_type)
        item['center_time'] = cur_t0 + 0.5 * (cur_t1 - cur_t0)
        subsegments.append(item)
        if detailed:
            print(_interval_debug_line(f'{tag}_try', item))
        if len(item['whole_consistent_dirs']) == 0 or item['reversed_dep']:
            failed_centers.append(item['center_time'])
            failed_indices.append(kk)
    return {
        'subsegments': subsegments,
        'failed_centers': failed_centers,
        'failed_indices': failed_indices,
    }


def postcheck_migration_period(period, cache, min_velo_exp=1.0, min_velo_shr=1.0, max_velo_exp=None,
                               min_fit_windows=4, min_mig_len=None, max_off_ratio=None,
                               win_step_slope=None, detailed=False):
    def _package(active_checks, init_checks_out, final_checks_out, meta_out):
        out = dict(active_checks)
        out['init_checks'] = init_checks_out
        out['final_checks'] = final_checks_out
        out['meta'] = meta_out
        return out

    t0, t1 = period['time_start'], period['time_end']
    dur = t1 - t0
    if dur <= 0:
        return False, {}

    if win_step_slope is None:
        win_step_slope = cache.get('win_step_contour', cache['win_step_contour'])

    # ---------- Initial permissive screening ----------
    seg_len = dur / 2.0
    mid_start = t0 + dur / 4.0
    mid_end = t0 + 3.0 * dur / 4.0
    first_start, first_end = t0, min(t0 + seg_len, t1)
    second_start, second_end = max(t1 - seg_len, t0), t1

    init_checks = {
        'whole': _fit_interval_item(cache, t0, t1, min_fit_windows, min_velo_exp / 2.0, None if max_velo_exp is None else 2.0 * max_velo_exp, 2.0 * min_velo_shr, max_off_ratio=max_off_ratio),
        'first_half': _fit_interval_item(cache, first_start, first_end, min_fit_windows, min_velo_exp / 2.0, None if max_velo_exp is None else 2.0 * max_velo_exp, 2.0 * min_velo_shr),
        'middle': _fit_interval_item(cache, mid_start, mid_end, min_fit_windows, min_velo_exp / 2.0, None if max_velo_exp is None else 2.0 * max_velo_exp, 2.0 * min_velo_shr),
        'second_half': _fit_interval_item(cache, second_start, second_end, min_fit_windows, min_velo_exp / 2.0, None if max_velo_exp is None else 2.0 * max_velo_exp, 2.0 * min_velo_shr),
    }

    seg0 = init_checks['whole']
    seg1 = init_checks['first_half']
    seg2 = init_checks['middle']
    seg3 = init_checks['second_half']

    objective_dirs = set()
    if seg0['pass'] and seg2['pass']:
        objective_dirs = seg0['dirs'].intersection(seg2['dirs'])
    meta = {
        'init_objective_dirs': sorted(objective_dirs),
        'need_head': False,
        'need_tail': False,
        'tuned_t0': t0,
        'tuned_t1': t1,
        'final_dirs': [],
        'fail_stage': '',
        'subsegment_count': 0,
        'failed_subsegment_centers': [],
    }

    if (not seg0['pass']) or (not seg2['pass']) or (len(objective_dirs) == 0):
        meta['fail_stage'] = 'init_screen'
        return False, _package(init_checks, init_checks, None, meta)

    middle_reversed_dep = _is_reversed_dep(seg2['is_dep_exp'], seg0['is_dep_exp'])
    if middle_reversed_dep:
        meta['fail_stage'] = 'middle_reversed_dep'
        return False, _package(init_checks, init_checks, None, meta)

    # ---------- Tuning seg1 / seg3 on objective dir ----------
    tuned_t0, tuned_t1 = t0, t1
    need_head = len(seg1['dirs'].intersection(objective_dirs)) < len(objective_dirs)
    need_tail = len(seg3['dirs'].intersection(objective_dirs)) < len(objective_dirs)
    meta['need_head'] = need_head
    meta['need_tail'] = need_tail

    if need_head:
        found = False
        cur = t0
        while cur <= mid_start + 1e-6:
            ta = cur
            tb = min(cur + seg_len, t1)
            item = _fit_interval_item(cache, ta, tb, min_fit_windows, min_velo_exp / 2.0, None if max_velo_exp is None else 2.0 * max_velo_exp, 2.0 * min_velo_shr)
            if detailed:
                print(_interval_debug_line('tune_head_try', item))
            if len(item['dirs'].intersection(objective_dirs)) == len(objective_dirs):
                tuned_t0 = item['seg']['time_start']
                found = True
                break
            cur += win_step_slope
            seg_len = (t1 - cur) / 2.0 
        if not found:
            meta['fail_stage'] = 'tune_head'
            return False, _package(init_checks, init_checks, None, meta)

    seg_len = (tuned_t1 - tuned_t0) / 2.0 
    if need_tail:
        found = False
        cur = t1
        while cur >= mid_end - 1e-6:
            tb = cur
            ta = max(t0, cur - seg_len)
            item = _fit_interval_item(cache, ta, tb, min_fit_windows, min_velo_exp / 2.0, None if max_velo_exp is None else 2.0 * max_velo_exp, 2.0 * min_velo_shr)
            if detailed:
                print(_interval_debug_line('tune_tail_try', item))
            if len(item['dirs'].intersection(objective_dirs)) == len(objective_dirs):
                tuned_t1 = item['seg']['time_end']
                found = True
                break
            cur -= win_step_slope
            seg_len = (cur - tuned_t0) / 2.0 
        if not found:
            meta['fail_stage'] = 'tune_tail'
            return False, _package(init_checks, init_checks, None, meta)

    # trim tuned interval to actual event-supported time
    idx_whole = _window_indices_within_time(cache, tuned_t0, tuned_t1)
    if idx_whole.size >= min_fit_windows:
        tuned_t0, tuned_t1 = _trim_interval_to_events(cache, idx_whole, tuned_t0, tuned_t1)
    meta['tuned_t0'] = tuned_t0
    meta['tuned_t1'] = tuned_t1

    if (min_mig_len is not None) and ((tuned_t1 - tuned_t0) < min_mig_len):
        meta['fail_stage'] = 'tuned_too_short'
        return False, _package(init_checks, init_checks, None, meta)

    # ---------- Final strict check ----------
    dur2 = tuned_t1 - tuned_t0
    seg_len2 = dur2 / 2.0

    final_checks = {
        'whole': _fit_interval_item(cache, tuned_t0, tuned_t1, min_fit_windows, min_velo_exp, max_velo_exp, min_velo_shr, max_off_ratio=max_off_ratio),
        'first_half': None,
        'middle': None,
        'second_half': None,
        'subsegments': [],
    }

    whole = final_checks['whole']
    if not whole['pass']:
        meta['fail_stage'] = 'final_whole'
        return False, _package(final_checks, init_checks, final_checks, meta)

    whole_dirs = _filter_surviving_dirs(
        whole['seg'], whole['dirs'], max_velo_exp, off_ratio=whole.get('off_ratio'), max_off_ratio=max_off_ratio
    )
    if len(whole_dirs) == 0:
        meta['fail_stage'] = 'final_whole_dirs'
        return False, _package(final_checks, init_checks, final_checks, meta)

    dep_code, dist_code, is_mig = _codes_from_dirs(whole_dirs)
    final_checks['whole']['dirs'] = set(whole_dirs)
    final_checks['whole']['is_dep_exp'] = dep_code
    final_checks['whole']['is_dist_exp'] = dist_code
    final_checks['whole']['is_mig'] = is_mig

    primary_scan = _scan_subsegments(
        cache, tuned_t0, tuned_t1, seg_len2, win_step_slope, min_fit_windows,
        min_velo_exp / 2.0,
        None if max_velo_exp is None else 2.0 * max_velo_exp,
        2.0 * min_velo_shr, whole_dirs,
        whole_dep_type=dep_code,
        require_no_reverse_dep=(dist_code == 1 and dep_code > 0),
        detailed=detailed, tag='final_subseg'
    )
    subsegments = primary_scan['subsegments']
    final_checks['subsegments'] = subsegments
    meta['subsegment_count'] = len(subsegments)
    meta['failed_subsegment_centers'] = primary_scan['failed_centers']

    if len(subsegments) == 0:
        meta['fail_stage'] = 'final_subsegment'
        return False, _package(final_checks, init_checks, final_checks, meta)
    if len(primary_scan['failed_centers']) > 0:
        meta['fail_stage'] = 'final_subsegment'
        meta['final_dirs'] = sorted(whole_dirs)
        return False, _package(final_checks, init_checks, final_checks, meta)

    final_checks['first_half'] = subsegments[0]
    final_checks['middle'] = subsegments[len(subsegments) // 2]
    final_checks['second_half'] = subsegments[-1]
    meta['fail_stage'] = ''
    meta['subsegment_count'] = len(subsegments)
    meta['final_dirs'] = sorted(whole_dirs)
    return True, _package(final_checks, init_checks, final_checks, meta)


def classify_cluster_status(events, ot_init, min_cluster_events=30, min_cluster_duration=400 * 86400):
    if len(events) < min_cluster_events:
        return 'Unknown'
    if (events['ot'][-1] - ot_init) < min_cluster_duration:
        return 'Unknown'
    return 'Classified'


def format_float(x):
    if x is None or not np.isfinite(x):
        return ''
    return f'{x:.2f}'


def _mask_uncertain_contours(D_ij, s_ij, L_ref, rel_unc_max=None):
    arr = np.array(D_ij, copy=True)
    if rel_unc_max is None or not np.isfinite(L_ref) or L_ref <= 0:
        return arr
    arr[(s_ij / L_ref) > rel_unc_max] = np.nan
    return arr


def _setup_fonts(cfg):
    plt.rcParams.update({'font.size': cfg.LABEL_FONTSIZE, 'axes.labelsize': cfg.LABEL_FONTSIZE,
                         'xtick.labelsize': cfg.LABEL_FONTSIZE, 'ytick.labelsize': cfg.LABEL_FONTSIZE,
                         'legend.fontsize': cfg.LABEL_FONTSIZE, 'axes.titlesize': cfg.TITLE_FONTSIZE})


def _format_time_label(t):
    return t.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-4] + 'Z'


def plot_cluster_summary(events, cache, sliding_out, mig_periods_org, mig_periods, fout, cfg, title=''):
    _setup_fonts(cfg)
    rel_days_evt = np.array([(ot - events['ot'][0]) / 86400.0 for ot in events['ot']], dtype=float)
    dep_km = events['dep'] / 1e3
    dist_evt_km = cache['dist_3d'] / 1e3
    q = cache['quantiles']
    idx05 = int(np.argmin(np.abs(q - 0.05)))
    idx95 = int(np.argmin(np.abs(q - 0.95)))
    dep_plot = _mask_uncertain_contours(cache['dep']['D_ij'], cache['dep']['s_ij'], cache['dep']['L_ref'], cfg.DXX_PLOT_REL_UNC_MAX) / 1e3
    dist_plot = _mask_uncertain_contours(cache['dist']['D_ij'], cache['dist']['s_ij'], cache['dist']['L_ref'], cfg.DXX_PLOT_REL_UNC_MAX) / 1e3
    cent_days = np.array([(ct - events['ot'][0]) / 86400.0 for ct in cache['window']['cent_time_out']], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=cfg.FIGSIZE_SUMMARY)
    ax00, ax01 = axes[0]
    ax10, ax11 = axes[1]
    if title:
        ax00.set_title(title)
        ax01.set_title(title)
    for bt in cache['blank_cent_times']:
        x = (bt - events['ot'][0]) / 86400.0
        ax00.axvline(x, color='0.8', lw=cfg.BLANK_LINEWIDTH, zorder=0)
        ax10.axvline(x, color='0.8', lw=cfg.BLANK_LINEWIDTH, zorder=0)
    ax00.scatter(rel_days_evt, dep_km, s=cfg.SCATTER_SIZE, alpha=cfg.SCATTER_ALPHA, edgecolors='none', zorder=2)
    ax10.scatter(rel_days_evt, dist_evt_km, s=cfg.SCATTER_SIZE, alpha=cfg.SCATTER_ALPHA, edgecolors='none', zorder=2)
    ax00.plot(cent_days, dep_plot[:, idx95], color='tab:red', ls='-', lw=cfg.CONTOUR_LINEWIDTH)
    ax00.plot(cent_days, dep_plot[:, idx05], color='tab:red', ls=':', lw=cfg.CONTOUR_LINEWIDTH)
    ax10.plot(cent_days, dist_plot[:, idx95], color='tab:red', ls='-', lw=cfg.CONTOUR_LINEWIDTH)
    ax10.plot(cent_days, dist_plot[:, idx05], color='tab:red', ls=':', lw=cfg.CONTOUR_LINEWIDTH)
    dep_y = np.nanmax(dep_km)
    dist_y = np.nanmin(dist_evt_km)
    for p in mig_periods_org:
        t0 = (p['time_start'] - events['ot'][0]) / 86400.0
        t1 = (p['time_end'] - events['ot'][0]) / 86400.0
        ax00.plot([t0, t1], [dep_y, dep_y], color='k', ls='--', marker='|', lw=1.5)
        ax10.plot([t0, t1], [dist_y, dist_y], color='k', ls='--', marker='|', lw=1.5)
    for p in mig_periods:
        t0 = (p['time_start'] - events['ot'][0]) / 86400.0
        t1 = (p['time_end'] - events['ot'][0]) / 86400.0
        ax00.plot([t0, t1], [dep_y, dep_y], color='k', marker='|', lw=1.5)
        ax10.plot([t0, t1], [dist_y, dist_y], color='k', marker='|', lw=1.5)
    x = np.arange(len(sliding_out['v_d05_dep']))
    ax01.plot(x, sliding_out['v_d95_dep'], color='tab:red', ls='-', label='D95')
    ax01.plot(x, sliding_out['v_d05_dep'], color='tab:red', ls=':', label='D05')
    ax11.plot(x, sliding_out['v_d95_dist'], color='tab:red', ls='-', label='D95')
    ax11.plot(x, sliding_out['v_d05_dist'], color='tab:red', ls=':', label='D05')
    for ax in [ax01, ax11]:
        ax.axhline(-cfg.MIN_VELO_EXP, color='k', ls=':', lw=1)
        ax.axhline(cfg.MIN_VELO_EXP, color='k', ls='-', lw=1)
    is_dep_exp = sliding_out['is_dep_exp']; is_dist_exp = sliding_out['is_dist_exp']; is_mig = sliding_out['is_mig']; blank_idx = np.where(cache['blank_mask'])[0]
    ax01.scatter(np.where(is_dep_exp < 0)[0], np.zeros(np.sum(is_dep_exp < 0)), s=20, color='k', marker='_')
    ax01.scatter(np.where(is_dep_exp > 0)[0], np.zeros(np.sum(is_dep_exp > 0)), s=20, color='k', marker='+')
    ax01.scatter(np.where(is_mig > 0)[0], np.ones(np.sum(is_mig > 0)), s=50, color='k', facecolors='none', marker='*')
    ax01.scatter(blank_idx, np.full(blank_idx.shape, -1.0), s=30, color='k', marker='|')
    ax11.scatter(np.where(is_dist_exp < 0)[0], np.zeros(np.sum(is_dist_exp < 0)), s=20, color='k', marker='_')
    ax11.scatter(np.where(is_dist_exp > 0)[0], np.zeros(np.sum(is_dist_exp > 0)), s=20, color='k', marker='+')
    ax11.scatter(np.where(is_mig > 0)[0], np.ones(np.sum(is_mig > 0)), s=50, color='k', facecolors='none', marker='*')
    ax11.scatter(blank_idx, np.full(blank_idx.shape, -1.0), s=30, color='k', marker='|')
    ax00.set_ylabel('Depth (km)')
    ax00.set_xlabel(f'Time since {_format_time_label(events[0][0])} (days)')
    ax00.invert_yaxis()
    ax10.set_ylabel('Distance 3D (km)')
    ax10.set_xlabel(f'Time since {_format_time_label(events[0][0])} (days)')
    ax01.set_ylabel('Depth Velocity (m/d)')
    ax01.set_xlabel('Window Index')
    ax01.invert_yaxis()
    ax11.set_ylabel('Distance Velocity (m/d)')
    ax11.set_xlabel('Window Index')
    ax01.legend(loc='best'); ax11.legend(loc='best')
    fig.tight_layout(); fig.savefig(fout, dpi=cfg.FIG_DPI, bbox_inches='tight'); plt.close(fig)


def plot_candidate_period_detail(events, cache, period, postcheck_out, fout_idx, cfg, title=''):
    _setup_fonts(cfg)
    init_checks = postcheck_out.get('init_checks', postcheck_out)
    final_checks = postcheck_out.get('final_checks')
    meta = postcheck_out.get('meta', {})
    t_plot0 = period.get('orig_time_start', period['time_start'])
    t_plot1 = period.get('orig_time_end', period['time_end'])
    cond_evt = (events['ot'] >= t_plot0) & (events['ot'] <= t_plot1)
    ev = events[cond_evt] if np.any(cond_evt) else events
    rel_days_evt = np.array([(ot - t_plot0) / 86400.0 for ot in ev['ot']], dtype=float)
    dist_evt = (cache['dist_3d'][cond_evt] / 1e3) if np.any(cond_evt) else (cache['dist_3d'] / 1e3)
    fig, axes = plt.subplots(2, 1, figsize=cfg.FIGSIZE_DETAIL)
    ax0, ax1 = axes
    ax0.scatter(rel_days_evt, ev['dep'] / 1e3, s=cfg.SCATTER_SIZE, alpha=cfg.SCATTER_ALPHA, edgecolors='none', zorder=2)
    ax1.scatter(rel_days_evt, dist_evt, s=cfg.SCATTER_SIZE, alpha=cfg.SCATTER_ALPHA, edgecolors='none', zorder=2)
    for bt in cache['blank_cent_times']:
        if t_plot0 <= bt <= t_plot1:
            x = (bt - t_plot0) / 86400.0
            ax0.axvline(x, color='0.8', lw=cfg.BLANK_LINEWIDTH, zorder=0)
            ax1.axvline(x, color='0.8', lw=cfg.BLANK_LINEWIDTH, zorder=0)
    for ct in meta.get('failed_subsegment_centers', []):
        if t_plot0 <= ct <= t_plot1:
            x = (ct - t_plot0) / 86400.0
            ax0.axvline(x, color='k', lw=0.9, alpha=0.55, zorder=0)
            ax1.axvline(x, color='k', lw=0.9, alpha=0.55, zorder=0)
    q = cache['quantiles']
    idx05 = int(np.argmin(np.abs(q - 0.05)))
    idx95 = int(np.argmin(np.abs(q - 0.95)))
    dep_plot = _mask_uncertain_contours(cache['dep']['D_ij'], cache['dep']['s_ij'], cache['dep']['L_ref'], cfg.DXX_PLOT_REL_UNC_MAX) / 1e3
    dist_plot = _mask_uncertain_contours(cache['dist']['D_ij'], cache['dist']['s_ij'], cache['dist']['L_ref'], cfg.DXX_PLOT_REL_UNC_MAX) / 1e3
    cent = cache['window']['cent_time_out']
    sel = (cent >= t_plot0) & (cent <= t_plot1)
    cent_days = np.array([(ct - t_plot0) / 86400.0 for ct in cent[sel]], dtype=float)
    ax0.plot(cent_days, dep_plot[sel, idx95], color='tab:red', ls='-', lw=cfg.CONTOUR_LINEWIDTH, label='D95')
    ax0.plot(cent_days, dep_plot[sel, idx05], color='tab:red', ls=':', lw=cfg.CONTOUR_LINEWIDTH, label='D05')
    ax1.plot(cent_days, dist_plot[sel, idx95], color='tab:red', ls='-', lw=cfg.CONTOUR_LINEWIDTH, label='D95')
    ax1.plot(cent_days, dist_plot[sel, idx05], color='tab:red', ls=':', lw=cfg.CONTOUR_LINEWIDTH, label='D05')

    def _plot_fit(item, color, lw):
        if item is None or item.get('seg') is None:
            return
        seg = item['seg']
        xy05_dep, xy95_dep, xy05_dist, xy95_dist = seg['xy_dep_dist']
        ta, tb = seg['time_start'], seg['time_end']
        x0 = (ta - t_plot0) / 86400.0; x1 = (tb - t_plot0) / 86400.0
        ax0.plot([x0, x1], [xy95_dep[2] / 1e3, xy95_dep[3] / 1e3], color=color, ls='-', lw=lw)
        ax0.plot([x0, x1], [xy05_dep[2] / 1e3, xy05_dep[3] / 1e3], color=color, ls=':', lw=lw)
        ax1.plot([x0, x1], [xy95_dist[2] / 1e3, xy95_dist[3] / 1e3], color=color, ls='-', lw=lw)
        ax1.plot([x0, x1], [xy05_dist[2] / 1e3, xy05_dist[3] / 1e3], color=color, ls=':', lw=lw)

    _plot_fit(init_checks.get('whole'), 'k', 1.0)
    _plot_fit(init_checks.get('first_half'), 'tab:orange', 1.5)
    _plot_fit(init_checks.get('middle'), 'tab:green', 1.5)
    _plot_fit(init_checks.get('second_half'), 'tab:cyan', 1.5)
    if meta.get('fail_stage', '') == '' and final_checks is not None and final_checks.get('whole') is not None and final_checks['whole'].get('seg') is not None:
        _plot_fit(final_checks.get('whole'), 'k', 2.5)
    ax0.set_xlim(0, max((t_plot1 - t_plot0) / 86400.0, 1e-6))
    ax1.set_xlim(0, max((t_plot1 - t_plot0) / 86400.0, 1e-6))
    ax0.set_title(title); ax0.set_ylabel('Depth (km)'); ax0.invert_yaxis(); ax1.set_ylabel('Distance 3D (km)')
    ax1.set_xlabel(f'Time since {_format_time_label(t_plot0)} (days)')
    ax0.legend(loc='best'); ax1.legend(loc='best')
    fig.tight_layout(); fig.savefig(fout_idx, dpi=cfg.FIG_DPI, bbox_inches='tight'); plt.close(fig)
