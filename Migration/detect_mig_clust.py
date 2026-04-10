import csv
from pathlib import Path
import numpy as np
import config as cfg
from migration_lib import (
    read_clusters, get_init_cent, calc_dist_3d, build_contour_cache,
    compute_sliding_velocity_series, merge_candidate_migration_periods,
    postcheck_migration_period, classify_cluster_status,
    plot_cluster_summary, plot_candidate_period_detail,
    format_float
)


def ensure_dirs():
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def _format_time(t):
    return t.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-4] + 'Z'


def _cluster_header(cluster_id, clust_name, status, ot_init, lat_init, lon_init, dep_init_m):
    return f'# {cluster_id}_{clust_name},{status},{_format_time(ot_init)},{lat_init:.4f},{lon_init:.4f},{dep_init_m/1e3:.2f}\n'


def _write_period_row(writer, cluster_id, clust_name, idx, period, check_whole):
    seg = check_whole['seg']
    row = {
        'cluster_id': cluster_id,
        'cluster_name': clust_name,
        'mig_idx': idx,
        'time_start': _format_time(period['time_start']),
        'time_end': _format_time(period['time_end']),
        'duration_days': format_float((period['time_end'] - period['time_start']) / 86400.0),
        'v_d05_dep': format_float(seg['v_dep_dist_list'][0]),
        'v_d50_dep': format_float(seg['v_dep_dist_list'][1]),
        'v_d95_dep': format_float(seg['v_dep_dist_list'][2]),
        'v_d05_dist': format_float(seg['v_dep_dist_list'][3]),
        'v_d50_dist': format_float(seg['v_dep_dist_list'][4]),
        'v_d95_dist': format_float(seg['v_dep_dist_list'][5]),
        's_d05_dep': format_float(seg['v_dep_dist_std_list'][0]),
        's_d50_dep': format_float(seg['v_dep_dist_std_list'][1]),
        's_d95_dep': format_float(seg['v_dep_dist_std_list'][2]),
        's_d05_dist': format_float(seg['v_dep_dist_std_list'][3]),
        's_d50_dist': format_float(seg['v_dep_dist_std_list'][4]),
        's_d95_dist': format_float(seg['v_dep_dist_std_list'][5]),
        'is_dep_exp': check_whole['is_dep_exp'],
        'is_dist_exp': check_whole['is_dist_exp'],
        'is_mig': check_whole['is_mig'],
    }
    writer.writerow(row)


def _mig_dirs(is_dep_exp, is_dist_exp):
    dirs = []
    if is_dep_exp == 1:
        dirs.append('dep_up')
    elif is_dep_exp == 2:
        dirs.append('dep_down')
    elif is_dep_exp == 3:
        dirs.extend(['dep_up', 'dep_down'])
    if is_dist_exp == 1:
        dirs.append('dist')
    return dirs


def _mig_type_label(is_mig):
    return {
        0: 'none',
        1: 'dep_up',
        2: 'dep_down',
        3: 'dist',
        4: 'dep_up+dist',
        5: 'dep_down+dist',
    }.get(is_mig, str(is_mig))


def _dirs_text(is_dep_exp, is_dist_exp):
    dirs = _mig_dirs(is_dep_exp, is_dist_exp)
    return ','.join(dirs) if dirs else 'none'


def _item_summary(item):
    if item is None or item.get('seg') is None:
        return 'seg=None'
    return f'pass={item.get("pass", False)} dirs={sorted(item.get("dirs", []))} nwin={item.get("nwin", 0)}'


def _print_candidate_summary(cluster_label, cand_idx, period, checks, passed, detailed=False):
    init_checks = checks.get('init_checks', checks)
    final_checks = checks.get('final_checks')
    meta = checks.get('meta', {})
    cand_t0 = period.get('orig_time_start', period['time_start'])
    cand_t1 = period.get('orig_time_end', period['time_end'])

    print(f'[cand_mig] {cluster_label} idx={cand_idx} time={_format_time(cand_t0)} -> {_format_time(cand_t1)} type={_mig_type_label(period.get("is_mig", 0))} dirs={_dirs_text(period.get("is_dep_exp", 0), period.get("is_dist_exp", 0))}')
    print(f'  init_screen: whole={_item_summary(init_checks.get("whole"))}; first={_item_summary(init_checks.get("first_half"))}; middle={_item_summary(init_checks.get("middle"))}; second={_item_summary(init_checks.get("second_half"))}')
    print(f'  init_objective_dirs: {meta.get("init_objective_dirs", [])}')
    print(f'  tuning: need_head={meta.get("need_head", False)} need_tail={meta.get("need_tail", False)} tuned={_format_time(meta.get("tuned_t0", cand_t0))} -> {_format_time(meta.get("tuned_t1", cand_t1))}')
    if final_checks is None:
        print(f'  final_check: not reached, fail_stage={meta.get("fail_stage", "")}')
    else:
        print(f'  final_check: whole={_item_summary(final_checks.get("whole"))} final_dirs={meta.get("final_dirs", [])} subsegments={meta.get("subsegment_count", 0)} fail_stage={meta.get("fail_stage", "") or "passed"}')
    if detailed:
        for name in ['whole', 'first_half', 'middle', 'second_half']:
            item = init_checks.get(name)
            print(f'    init_{name}: {_item_summary(item)}')
        if final_checks is not None:
            print(f'    final_whole: {_item_summary(final_checks.get("whole"))}')
            for kk, item in enumerate(final_checks.get('subsegments', [])):
                print(f'    final_subseg[{kk}]: {_item_summary(item)} whole_consistent={sorted(item.get("whole_consistent_dirs", []))}')
    if passed:
        print(f'  final_mig: detected {_format_time(period["time_start"])} -> {_format_time(period["time_end"])} type={_mig_type_label(checks["whole"]["is_mig"])} dirs={_dirs_text(checks["whole"]["is_dep_exp"], checks["whole"]["is_dist_exp"])}')
    else:
        print(f'  final_mig: rejected at {meta.get("fail_stage", "unknown")}')


def process_cluster(ii, clust_name, events):
    events = np.sort(events, order='ot')
    ot_init, lat_init, lon_init, dep_init = get_init_cent(events)
    dist_3d = calc_dist_3d(events, lat_init, lon_init, dep_init)
    status0 = classify_cluster_status(events, ot_init, cfg.MIN_CLUSTER_EVENTS, cfg.MIN_CLUSTER_DURATION)
    out = {
        'cluster_id': ii,
        'cluster_name': clust_name,
        'status': status0,
        'ot_init': ot_init,
        'lat_init': lat_init,
        'lon_init': lon_init,
        'dep_init': dep_init,
        'cache': None,
        'sliding': None,
        'mig_periods_org': [],
        'mig_periods': [],
        'events': events,
    }
    if status0 == 'Unknown':
        return out

    cache = build_contour_cache(
        start_time=ot_init,
        end_time=events['ot'][-1],
        events=events,
        dist_3d=dist_3d,
        quantiles=cfg.QUANTILES,
        win_len_contour=cfg.WIN_LEN_CONTOUR,
        win_step_contour=cfg.WIN_STEP_CONTOUR,
        min_events_contour=cfg.MIN_EVENTS_CONTOUR,
        n_bootstrap=cfg.N_BOOTSTRAP,
        random_state=cfg.RANDOM_STATE,
        n_jobs=cfg.N_JOBS,
        s0_frac=cfg.S0_FRAC,
    )
    max_velo_exp = getattr(cfg, 'MAX_VELO_EXP', None)
    max_off_ratio = getattr(cfg, 'MAX_OFF_RATIO', None)
    sliding = compute_sliding_velocity_series(
        cache, cfg.NUM_WIN_MIG, cfg.MIN_VELO_EXP, cfg.MIN_VELO_SHR, max_velo_exp, cfg.MIN_FIT_WINDOWS
    )
    mig_periods_org = merge_candidate_migration_periods(sliding, cache, cfg.MIN_MIG_LEN)
    mig_periods = []
    updated_org = []
    detailed_cluster = ii in getattr(cfg, 'DEBUG_CLUSTER_IDS', [])
    for period in mig_periods_org:
        pp = dict(period)
        pp['orig_time_start'] = period['time_start']
        pp['orig_time_end'] = period['time_end']
        cand_idx = len(updated_org)
        passed, checks = postcheck_migration_period(period, cache, cfg.MIN_VELO_EXP, cfg.MIN_VELO_SHR,
                                                    max_velo_exp, cfg.MIN_FIT_WINDOWS,
                                                    cfg.MIN_MIG_LEN, max_off_ratio, cfg.WIN_STEP_SLOPE,
                                                    detailed_cluster)
        pp['postcheck'] = checks
        if 'whole' in checks and checks['whole'].get('seg') is not None:
            pp['time_start'] = checks['whole']['seg']['time_start']
            pp['time_end'] = checks['whole']['seg']['time_end']
        updated_org.append(pp)
        _print_candidate_summary(f'{ii}_{clust_name}', cand_idx, pp, checks, passed, detailed_cluster)
        if passed:
            mig_periods.append(pp)

    out['cache'] = cache
    out['sliding'] = sliding
    out['mig_periods_org'] = updated_org
    out['mig_periods'] = mig_periods
    out['status'] = 'Migration' if mig_periods else 'No migration'
    return out


def write_output_csv(results):
    with open(cfg.FOUT_CSV, 'w', newline='') as f:
        fieldnames = [
            'cluster_id', 'cluster_name', 'mig_idx', 'time_start', 'time_end', 'duration_days',
            'v_d05_dep', 'v_d50_dep', 'v_d95_dep', 'v_d05_dist', 'v_d50_dist', 'v_d95_dist',
            's_d05_dep', 's_d50_dep', 's_d95_dep', 's_d05_dist', 's_d50_dist', 's_d95_dist',
            'is_dep_exp', 'is_dist_exp', 'is_mig'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            f.write(_cluster_header(res['cluster_id'], res['cluster_name'], res['status'],
                                    res['ot_init'], res['lat_init'], res['lon_init'], res['dep_init']))
            if res['status'] == 'Migration':
                for jj, period in enumerate(res['mig_periods']):
                    _write_period_row(writer, res['cluster_id'], res['cluster_name'], jj, period, period['postcheck']['whole'])


def make_plots(results):
    for res in results:
        if res['cache'] is None or res['sliding'] is None:
            continue
        fout = f'{cfg.FOUT_ROOT}-{res["cluster_id"]}.jpg'
        title = f'{res["cluster_id"]}_{res["cluster_name"]}'
        plot_cluster_summary(res['events'], res['cache'], res['sliding'], res['mig_periods_org'], res['mig_periods'], fout, cfg, title=title)
        for jj, period in enumerate(res['mig_periods_org']):
            postcheck_out = period.get('postcheck', {})
            fout_idx = f'{cfg.FOUT_ROOT}-{res["cluster_id"]}_idx-{jj}.pdf'
            plot_candidate_period_detail(res['events'], res['cache'], period, postcheck_out, fout_idx, cfg, title=title)


def main():
    ensure_dirs()
    events_clustered, clust_names = read_clusters(cfg.FCLUST, cfg.MC)
    idx_iter = range(len(events_clustered)) if cfg.PLOT_MAX_CLUSTERS is None else range(min(cfg.PLOT_MAX_CLUSTERS, len(events_clustered)))

    results = []
    for ii in idx_iter:
        print('-' * 20)
        print(f'processing {ii} {clust_names[ii]}')
        results.append(process_cluster(ii, clust_names[ii], events_clustered[ii]))
    write_output_csv(results)
    make_plots(results)
    print(f'Wrote CSV: {cfg.FOUT_CSV}')
    print(f'Wrote figures under: {cfg.OUTPUT_DIR}')


if __name__ == '__main__':
    main()
