"""Compare smoothed p-force and tangent-matched linear dynamics on saved 003 starts."""
import argparse
import csv
from itertools import product
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / '003-bridge-selectivity'
sys.path[:0] = [str(SOURCE), str(HERE.parent / '006a-pforce-check')]
import bridge_selectivity as ref
from bridge_selectivity import C, np
from pforce import make_pforce

P, EPS, FLOOR = 1.5, 0.5, 1e-8
TANGENT = EPS**(P-2)
METHODS = ('pforce', 'matched_linear')


def baseline_rows():
    with (SOURCE / 'part2_scores.csv').open(newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['force'] in ('mean', 'spectral')]
    keys = {(r['graph'], r['force'], r['dyn'], r['h'], r['window'], r['readout']): r for r in rows}
    if len(keys) != len(rows) or len(keys) != 48*(2*2*2*2+1):
        raise ValueError('Bad 003 baseline coverage')
    return keys


def summarize(pairs):
    groups = {}
    keys = 'h', 'window', 'readout', 'dyn', 'baseline'
    for p in pairs:
        groups.setdefault(tuple(p[k] for k in keys), []).append(p)
    result = []
    for key, group in sorted(groups.items()):
        for family in ('all', *ref.FAMILIES):
            rows = [r for r in group if family == 'all' or r['family'] == family]
            if len(rows) != (48 if family == 'all' else 16):
                raise ValueError('Incomplete paired stratum')
            a = np.array([r['delta_ari_plan'] for r in rows])
            ok = np.isfinite(a)
            result.append(dict(zip(keys, key), family=family, graphs=len(rows), pairs=int(ok.sum()),
                               mean_delta_ari_plan=float(a[ok].mean()) if ok.any() else np.nan,
                               pforce_graphs_failed=sum(r['n_fail'] > 0 for r in rows),
                               pforce_runs_failed=sum(r['n_fail'] for r in rows),
                               pforce_runs=len(rows)*ref.R))
    return result


def run(out):
    baselines = baseline_rows()
    with np.load(SOURCE / 'part2_checkpoints.npz') as z:
        ids, inits, truths = z['graph_ids'], z['inits'], z['truth']
    if len(set(ids)) != len(ids) or inits.shape != (48, ref.R, 48):
        raise ValueError('Bad archived start coverage')
    starts = dict(zip(ids, zip(inits, truths)))
    graph_rows, init_rows, pairs = [], [], []
    edge_data, scores_bank = [], []
    for spec in ref.p2_specs():
        g, truth, meta = C.make_graph(spec)
        graph = meta['id']
        u0, archived_truth = starts[graph]
        if not np.array_equal(truth, archived_truth):
            raise ValueError(f'Archive truth mismatch: {graph}')
        metadata, d, bm = ref.gmeta(spec, meta), C.wdeg(g), meta['bridge_mask']
        edge_data.append((graph, g.ei, g.ej, g.ew, bm))
        mean = ref.make_force(g, 'mean')
        forces = {'pforce': make_pforce(g, P, EPS), 'matched_linear': lambda u: TANGENT*mean(u)}
        results = {}
        for h, method, dyn in product(ref.HS, METHODS, ref.DYNS):
            traj = ref.integrate(forces[method], u0, h, dyn, d)
            failures, censors = traj['first_fail'], traj['censor']
            surv = failures < 0
            key = dict(**metadata, method=method, dyn=dyn, h=h)
            for r, (f, c) in enumerate(zip(failures, censors)):
                init_rows.append(dict(**key, init=r, failed=int(f >= 0), first_fail_substep=f,
                                      first_fail_t=f*h if f >= 0 else '', censor_substep=c,
                                      censor_t=c*h if c >= 0 else ''))
            vectors = np.full((2, 2, len(g.ei)), np.nan)
            for wi, (window, t0, t1) in enumerate(ref.WINDOWS):
                a, b = ref.frames(t0, t1)
                U = traj['X'][a:b, surv]
                wkey = dict(**key, window=window, n_init=ref.R, n_fail=int((~surv).sum()),
                            n_censored=int((censors >= 0).sum()), n_scored=int(surv.sum()))
                if method == 'pforce' and surv.any():
                    s = U[..., g.ej] - U[..., g.ei]
                    k = (s*s + EPS*EPS)**((P-2)/2)
                    for label, mask in [('bridge', bm), ('internal', ~bm)]:
                        stiffness = float(k[..., mask].mean())
                        wkey[f'{label}_stiffness'] = stiffness
                        wkey[f'{label}_relative_stiffness'] = stiffness/TANGENT
                for ri, ro in enumerate(ref.READOUTS):
                    ordinary = baselines[graph, 'mean', dyn, str(h), window, ro]
                    spectral = baselines[graph, 'spectral', 'none', 'none', 'none', 'none']
                    row = dict(**wkey, readout=ro, cut_ok=0, ari_survivor=np.nan, auc=np.nan, gap=np.nan,
                               ordinary_mean_ari_plan=float(ordinary['ari_plan']),
                               spectral_ari_plan=float(spectral['ari_plan']))
                    if surv.any():
                        score = C.edge_score(U, g, frac=1, lowpass_fc=ref.FC if ro == 'lp' else None)
                        if np.isfinite(score).all():
                            vectors[wi, ri] = score
                            row.update(ref.score_metrics(g, truth, score, bm))
                            row['both_class_rms_below_1e_minus_8'] = int(max(row['bridge_rms'], row['interior_rms']) < FLOOR)
                    row['unscorable'] = int(not row['cut_ok'])
                    row['ari_plan'] = 0.0 if row['n_fail'] else row['ari_survivor']
                    graph_rows.append(row)
                    results[h, method, dyn, window, ro] = row
            scores_bank.append(vectors)
        for h, dyn, (window, _, _), ro in product(ref.HS, ref.DYNS, ref.WINDOWS, ref.READOUTS):
            a = results[h, 'pforce', dyn, window, ro]
            controls = {f'matched_linear_{dyn}': results[h, 'matched_linear', dyn, window, ro]['ari_plan'],
                        f'ordinary_mean_{dyn}': a['ordinary_mean_ari_plan']}
            if dyn == 'wave':
                controls.update(pforce_diffusion=results[h, 'pforce', 'diffusion', window, ro]['ari_plan'],
                                spectral=a['spectral_ari_plan'])
            for baseline, b in controls.items():
                pairs.append(dict(**metadata, h=h, dyn=dyn, window=window, readout=ro, baseline=baseline,
                                  n_fail=a['n_fail'], pforce_ari_plan=a['ari_plan'], baseline_ari_plan=b,
                                  delta_ari_plan=a['ari_plan']-b))
    if (len(edge_data), len(scores_bank), len(graph_rows), len(init_rows), len(pairs)) != (48, 384, 1536, 3072, 2304):
        raise ValueError('Incomplete design coverage')
    for name, rows in [('graphs', graph_rows), ('initializations', init_rows), ('paired_graphs', pairs),
                       ('paired_summary', summarize(pairs))]:
        ref.write_csv(out / f'{name}.csv', rows)
    edge_ptr = np.cumsum([0]+[len(e[1]) for e in edge_data])
    run_ptr = np.cumsum([0]+[s.shape[-1] for s in scores_bank])
    ref.write_npz(out / 'edges.npz', dict(graph=np.array([e[0] for e in edge_data]), edge_ptr=edge_ptr,
                  i=np.concatenate([e[1] for e in edge_data]), j=np.concatenate([e[2] for e in edge_data]),
                  weight=np.concatenate([e[3] for e in edge_data]), bridge=np.concatenate([e[4] for e in edge_data]),
                  run_graph_index=np.repeat(np.arange(48), 8), run_ptr=run_ptr,
                  run_h=np.tile(np.repeat(ref.HS, 4), 48), run_method=np.tile(np.repeat(METHODS, 2), 96),
                  run_dyn=np.tile(ref.DYNS, 192), windows=np.array([w[0] for w in ref.WINDOWS]),
                  readouts=np.array(ref.READOUTS), scores=np.concatenate(scores_bank, axis=-1)))
    print(f'Scored {len(graph_rows)} graph/readout rows; {sum(r["failed"] for r in init_rows)}/{len(init_rows)} starts failed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    run(args.out)
