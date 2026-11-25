"""Run the unchanged 006c numerical sweep and join saved dynamic ARIs."""
import argparse
import csv
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / '003-bridge-selectivity'
sys.path[:0] = [str(SOURCE), str(HERE.parent / '006c-static-pcut-check')]
import bridge_selectivity as ref
from bridge_selectivity import C, np, adjusted_rand_score
from check import independent_binary
from pcut import p_rayleigh_sweep


def load(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def run(out):
    spectral_rows = [r for r in load(SOURCE / 'part2_scores.csv') if r['force'] == 'spectral']
    spectral = {r['graph']: float(r['ari_plan']) for r in spectral_rows}
    dynamics = [r for r in load(HERE.parent / '006b-pforce-dynamics/graphs.csv') if r['method'] == 'pforce']
    keys = {(r['graph'], r['dyn'], r['h'], r['window'], r['readout']) for r in dynamics}
    if len(spectral) != len(spectral_rows) or len(keys) != len(dynamics) or len(keys) != 768:
        raise ValueError('Unexpected archived score coverage')
    graph_rows, trials, vectors, static = [], [], [], {}
    for spec in ref.p2_specs():
        g, truth, meta = C.make_graph(spec)
        graph = meta['id']
        row = dict(ref.gmeta(spec, meta), p=1.5, seed_optimizer=0, spectral_ari=spectral[graph],
                   approximate=1, static_ari=np.nan, unscorable=1, reason='')
        try:
            result = p_rayleigh_sweep(g, p=1.5, seed=0)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            row['reason'] = str(exc)
        else:
            trials.extend(dict(graph=graph, **t) for t in result.trials)
            [winner] = [t for t in result.trials if t['selected']]
            row.update(continuous_rayleigh=result.rayleigh, binary_objective=result.cut_objective,
                       threshold=result.threshold, selected_start=winner['start'],
                       selected_optimizer_success=winner['optimizer_success'], selected_status=winner['optimizer_status'],
                       selected_grad_inf=winner['grad_inf'], selected_center_residual=winner['center_residual'])
            vectors.extend(dict(graph=graph, node=i, x=x, label=int(label))
                           for i, (x, label) in enumerate(zip(result.vector, result.labels)))
            if (np.isfinite(result.vector).all() and np.isfinite(result.cut_objective)
                    and np.array_equal(result.labels, result.vector > result.threshold)
                    and np.unique(result.labels).size == 2):
                independent = independent_binary(g.A, result.labels, 1.5)
                error = abs(independent-result.cut_objective)
                row.update(independent_binary_objective=independent, objective_abs_error=error)
                if not error <= 1e-10:
                    raise ValueError(f'Independent objective mismatch: {graph}, {error}')
                row.update(static_ari=float(adjusted_rand_score(truth, result.labels)), unscorable=0)
            else:
                row['reason'] = 'Nonfinite or invalid threshold cut'
        graph_rows.append(row)
        static[graph] = row
    if len(static) != 48 or static.keys() != spectral.keys():
        raise ValueError('Incomplete static/spectral join')
    pairs = []

    def pair(graph, comparator, score, h='', window='', readout='', comparator_fail=''):
        r = static[graph]
        keys = 'graph', 'family', 'sizes', 'bridges', 'seed'
        return dict({k: r[k] for k in keys}, comparator=comparator, h=h, window=window, readout=readout,
                    static_ari=r['static_ari'], static_unscorable=r['unscorable'], comparator_ari=score,
                    comparator_n_fail=comparator_fail, delta_static_minus_comparator=r['static_ari']-score)

    for r in dynamics:
        pairs.append(pair(r['graph'], 'p_'+r['dyn'], float(r['ari_plan']), r['h'], r['window'], r['readout'], r['n_fail']))
    pairs.extend(pair(g, 'spectral', a) for g, a in spectral.items())
    groups = {}
    columns = 'comparator', 'h', 'window', 'readout'
    for p in pairs:
        groups.setdefault(tuple(p[k] for k in columns), []).append(p)
    summary = []
    for key, group in sorted(groups.items()):
        for family in ('all', *ref.FAMILIES):
            subset = [p for p in group if family == 'all' or p['family'] == family]
            if len(subset) != (48 if family == 'all' else 16):
                raise ValueError(f'Incomplete paired group: {key}, {family}')
            ds = np.array([p['delta_static_minus_comparator'] for p in subset])
            ok = np.isfinite(ds)
            summary.append(dict(zip(columns, key), family=family, graphs=len(ds), pairs=int(ok.sum()),
                                mean_delta_static_minus_comparator=float(ds[ok].mean()) if ok.any() else np.nan))
    for name, rows in [('graphs', graph_rows), ('optimizer_trials', trials), ('vectors', vectors),
                       ('paired_graphs', pairs), ('paired_summary', summary)]:
        ref.write_csv(out / f'{name}.csv', rows)
    print(f'Scored {48-sum(r["unscorable"] for r in graph_rows)}/48 graphs; {len(trials)} optimizer records.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    run(args.out)
