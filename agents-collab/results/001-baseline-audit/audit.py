"""Rebuild the development audit from saved CSVs; no simulation imports."""
import argparse
import csv
from itertools import product
from math import isclose, isfinite
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
FAMILIES = ('cycle', 'clique', 'sbm')
BRIDGES = (1, 2, 4, 8)
FORCES = ('mean', 'trim', 'owa', 'vw', 'odd_owa', 'odd_vw', 'huber', 'tanh', 'amp',
          'gauss', 'absvw', 'symowa_ext', 'symowa_med', 'median', 'huber_local',
          'huber_global', 'amp_gated', 'blend_agree', 'huber_anneal', 'edge_memory')
SCHEMES = ('wave', 'damp', 'diff')
SPECTRAL = ('spectral', 'spectral')
SELECTED = tuple(product(('mean', 'huber', 'huber_local'), SCHEMES))
READOUTS = {'raw': 'ari_edge_cut', 'lp': 'ari_edge_cut_lp'}
POLICIES = ('legacy_available', 'legacy_report', 'plan')


def check(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def save(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def ari(text):
    if text in ('', 'nan'):
        return None
    value = float(text)
    check(isfinite(value) and -1 <= value <= 1, f'Invalid ARI: {text}')
    return value


def scores(r, readout):
    value = ari(r[READOUTS[readout]])
    return dict(legacy_available=value, legacy_report=0.0 if value is None else value,
                plan=0.0 if int(r['n_ok']) < int(r['R']) else value)


def load_t310(root):
    path = root / 'GNN_testing/battery/idea3/results/t310_partition.csv'
    meta = {f'two_block:{f}_24x24_bridges{b}_wb1_s{s}':
            dict(gkind=f, bridges=b, seed=s, sizes='24x24', n=48, wb=1.0)
            for f, b, s in product(FAMILIES, BRIDGES, range(4))}
    data = {}
    for r in load(path):
        g, method = r['graph'], (r['force'], r['scheme'])
        key = (g, method)
        check(g in meta and key not in data, f'Unexpected or duplicate key: {key}')
        check(all(str(meta[g][k]) == r[k] for k in ('gkind', 'bridges', 'seed')),
              f'Inconsistent metadata: {key}')
        data[key] = r
        if method == SPECTRAL:
            check(r['R'] == '' and r['n_ok'] == '1' and float(r['fail_frac']) == 0
                  and ari(r['ari_edge_cut']) is not None and ari(r['ari_edge_cut_lp']) is None,
                  f'Invalid spectral row: {key}')
            continue
        R, ok = int(r['R']), int(r['n_ok'])
        check(R == 8 and 0 <= ok <= R and isclose(float(r['fail_frac']), (R-ok)/R),
              f'Invalid failure counts: {key}')
        check(all((ari(r[c]) is not None) == (ok > 0) for c in READOUTS.values()),
              f'Unexpected ARI availability: {key}')
    methods = (*product(FORCES, SCHEMES), SPECTRAL)
    check(data.keys() == set(product(meta, methods)), 'Incomplete t310 Cartesian coverage')
    return meta, data


def strata(meta):
    yield 'overall', 'all', list(meta)
    for column, values in [('gkind', FAMILIES), ('bridges', BRIDGES)]:
        for value in values:
            yield column, value, [g for g, m in meta.items() if m[column] == value]


def avg(values):
    observed = [v for v in values if v is not None]
    return mean(observed) if observed else None


def t310_tables(root, out):
    meta, data = load_t310(root)
    graph_rows, pair_rows, pair_index = [], [], {}
    for g, m in meta.items():
        sp = ari(data[g, SPECTRAL]['ari_edge_cut'])
        row = dict(graph=g, **m, spectral_ari=sp)
        for method in SELECTED:
            r = data[g, method]
            prefix = '_'.join(method)
            for c in ('n_ok', 'R', 'fail_frac'):
                row[f'{prefix}_{c}'] = r[c]
            for readout in READOUTS:
                for policy, value in scores(r, readout).items():
                    row[f'{prefix}_{readout}_{policy}'] = value
        graph_rows.append(row)
        for readout, baseline in product(READOUTS, (('mean', 'wave'), ('huber_local', 'diff'), SPECTRAL)):
            candidate = scores(data[g, ('huber_local', 'wave')], readout)
            base = dict.fromkeys(POLICIES, sp) if baseline == SPECTRAL else scores(data[g, baseline], readout)
            label = '_'.join(baseline) if baseline != SPECTRAL else 'spectral'
            p = dict(graph=g, **m, readout=readout, baseline=label)
            for policy in POLICIES:
                a, b = candidate[policy], base[policy]
                p[policy] = a - b if a is not None and b is not None else None
            pair_rows.append(p)
            pair_index[g, readout, label] = p
    summaries, failures = [], []
    for stratum, level, graphs in strata(meta):
        for readout, baseline, policy in product(READOUTS, ('mean_wave', 'huber_local_diff', 'spectral'), POLICIES):
            vals = [pair_index[g, readout, baseline][policy] for g in graphs]
            summaries.append(dict(stratum=stratum, level=level, readout=readout, baseline=baseline,
                                  policy=policy, graphs=len(graphs), pairs=sum(v is not None for v in vals),
                                  mean_difference=avg(vals)))
        for method in product(FORCES, SCHEMES):
            rows = [data[g, method] for g in graphs]
            runs = [int(r['R']) for r in rows]
            ok = [int(r['n_ok']) for r in rows]
            failed = [a-b for a, b in zip(runs, ok)]
            failure = dict(stratum=stratum, level=level, force=method[0], scheme=method[1],
                           graphs=len(rows), runs=sum(runs), runs_failed=sum(failed),
                           run_failure_fraction=sum(failed)/sum(runs),
                           graphs_any_failed=sum(f > 0 for f in failed),
                           graph_failure_fraction=sum(f > 0 for f in failed)/len(rows),
                           graphs_all_failed=ok.count(0), graphs_partial=sum(0 < n < 8 for n in ok))
            for readout in READOUTS:
                ss = [scores(r, readout) for r in rows]
                for policy in POLICIES:
                    vals = [s[policy] for s in ss]
                    failure[f'{readout}_{policy}_n'] = sum(v is not None for v in vals)
                    failure[f'{readout}_{policy}_mean'] = avg(vals)
                failure[f'{readout}_plan_minus_legacy_report'] = (
                    failure[f'{readout}_plan_mean'] - failure[f'{readout}_legacy_report_mean'])
            failures.append(failure)
    for name, rows in [('t310_graphs', graph_rows), ('t310_pairs', pair_rows),
                       ('t310_pair_summary', summaries), ('t310_failures', failures)]:
        save(out / f'{name}.csv', rows)
    return set((m['gkind'], m['sizes'], m['bridges'], m['wb'], m['seed']) for m in meta.values())


def rank_table(root, out, t310_keys):
    # Read historical evaluation output only; never generate or evaluate those seeds.
    rows = load(root / 'results/partition_quasi.csv')
    def key(r):
        return (r['kind'], r['sizes'], int(r['bridges']), float(r['wb']), int(r['seed']))

    dynamic = set(product(('mean', 'trim', 'owa', 'vw'), ('diff', 'wave_raw', 'wave_lp')))
    baselines = {(a, 'baseline') for a in ('spectral', 'wave_fft_2009_simplified',
                                          'dmd_2022_simplified', 'becchetti_simplified')}
    controls = {'eval_disc': {(f, '24x24', 0, 1.0, s) for f, s in product(FAMILIES, range(100, 108))},
                'eval_neg': {(f, '24x24', 0, 1.0, s) for f, s in product(('er', 'shuffled'), range(100, 108))}}
    positive = set(product(FAMILIES, ('24x24', '16x32'), BRIDGES, (1.0, 0.25), range(100, 108)))
    keys = [(r['role'], key(r), (r['agg'], r['dyn'])) for r in rows]
    expected = {(role, k, m) for role, kk in dict(controls, eval_pos=positive).items()
                for k in kk for m in (dynamic if role == 'eval_neg' else dynamic | baselines)}
    check(len(keys) == len(set(keys)) and set(keys) == expected, 'Invalid rank Cartesian coverage')
    check(not positive & t310_keys, 'Unexpected rank/t310 pairing')
    groups = [('overall', 'all', rows)]
    groups.extend(('kind', f, [r for r in rows if r['kind'] == f]) for f in FAMILIES)
    summary = []
    for stratum, level, group in groups:
        for agg, dyn in sorted(dynamic | baselines):
            subset = [r for r in group if (r['role'], r['agg'], r['dyn']) == ('eval_pos', agg, dyn)]
            historical = [ari(r['ari']) for r in subset]
            failed = [round(float(r['fail_frac'])*8) for r in subset] if dyn != 'baseline' else None
            if failed is not None:
                check(all(0 <= n <= 8 and n/8 == float(r['fail_frac']) for n, r in zip(failed, subset)),
                      f'Invalid rank failure counts: {agg}/{dyn}')
            summary.append(dict(stratum=stratum, level=level, agg=agg, dyn=dyn, graphs=len(subset),
                                historical_n=sum(v is not None for v in historical), historical_mean=avg(historical),
                                runs=len(subset)*8 if failed is not None else None,
                                runs_failed=sum(failed) if failed is not None else None,
                                graphs_any_failed=sum(n > 0 for n in failed) if failed is not None else None,
                                graphs_all_failed=failed.count(8) if failed is not None else None))
    save(out / 'rank_historical.csv', summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=HERE.parents[2], help='repository root')
    parser.add_argument('--out', type=Path, default=HERE, help='output directory')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rank_table(args.root, args.out, t310_tables(args.root, args.out))
    print('Validated 48 t310 graphs (2,928 rows) and 424 historical rank graphs (6,720 rows).')


if __name__ == '__main__':
    main()
