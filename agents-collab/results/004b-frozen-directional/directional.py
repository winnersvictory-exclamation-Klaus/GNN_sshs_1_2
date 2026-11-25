"""Freeze endpoint-local stiffness without symmetrizing it."""
import argparse
import csv
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / '004a-frozen-sbm'
sys.path.insert(0, str(PRIOR))
import frozen_sbm as previous
from frozen_sbm import C, np, ref, require

NAME = 'directional_freeze'


def directional_force(g, u):
    tau = ref.local_tau(g, u)
    s = u[:, g.ej] - u[:, g.ei]
    ratio = np.stack([np.divide(np.clip(s, -t, t), s, out=np.ones_like(s), where=s != 0)
                      for t in (tau[:, g.ei], tau[:, g.ej])])
    require(np.isfinite(ratio).all() and ((0 <= ratio) & (ratio <= 1)).all(), 'Invalid multipliers')
    ratio.flags.writeable = False
    A = np.zeros((len(u), g.n, g.n))
    A[:, g.ei, g.ej], A[:, g.ej, g.ei] = g.ew * ratio
    degree = A.sum(2)
    d = C.wdeg(g)
    A.flags.writeable = degree.flags.writeable = False

    def force(v):
        return (np.einsum('rij,rj->ri', A, v) - degree*v) / d

    errors = np.max(np.abs(force(u) - ref.make_force(g, 'huber_local')(u)), axis=1)
    require((errors <= 1e-12).all(), f'Directional/adaptive force mismatch: {errors.max()}')
    return force, ratio, errors


def run(out):
    with (PRIOR / 'graphs.csv').open(newline='') as f:
        baselines = list(csv.DictReader(f))
    keys = {(r['graph'], float(r['h']), r['readout'], r['fork']): r for r in baselines}
    require(len(keys) == len(baselines) == 128, 'Bad 004a score coverage')
    archive = previous.archived_index()
    graphs, inits, pairs, bank = [], [], [], []
    with np.load(previous.SOURCE / 'part2_checkpoints.npz') as z:
        ids, states = z['graph_ids'], z['states']
        run_keys = list(zip(z['run_graph_index'], z['run_force'], z['run_dyn'], z['run_h']))
        checkpoints = {key: i for i, key in enumerate(run_keys)}
        require(len(checkpoints) == len(run_keys), 'Duplicate checkpoint key')
        for spec in ref.p2_specs():
            if spec['block'] != 'sbm':
                continue
            g, truth, meta = C.make_graph(spec)
            graph = meta['id']
            gi, = np.flatnonzero(ids == graph)
            require(np.array_equal(truth, z['truth'][gi]), f'Truth mismatch: {graph}')
            for h in ref.HS:
                start, expected = states[checkpoints[gi, 'huber_local', 'wave', h]][:2]
                prev, cur, state_error = previous.warmup(g, start, h, expected)
                force, ratio, errors = directional_force(g, cur)
                X, ff, cs = previous.continue_wave(force, prev, cur, h)
                ok = ff < 0
                for init, (f, c, err) in enumerate(zip(ff, cs, errors)):
                    inits.append(dict(graph=graph, h=h, init=init, failed=int(f >= 0), first_fail_substep=f,
                                      first_fail_t=f*h if f >= 0 else '', censor_substep=c,
                                      t50_force_max_abs_error=err))
                score_vectors = []
                for ro in ref.READOUTS:
                    row = dict(ref.gmeta(spec, meta), h=h, readout=ro, fork=NAME, window='late',
                               n_init=ref.R, n_fail=int((~ok).sum()), n_censored=int((cs >= 0).sum()),
                               n_scored=int(ok.sum()), t50_state_max_abs_error=state_error,
                               cut_ok=0, ari_survivor=np.nan, auc=np.nan, gap=np.nan)
                    score = (C.edge_score(X[:, ok], g, frac=1.0, lowpass_fc=ref.FC if ro == 'lp' else None)
                             if ok.any() else np.full(len(g.ew), np.nan))
                    if np.isfinite(score).all():
                        row.update(ref.score_metrics(g, truth, score, meta['bridge_mask']))
                    row['unscorable'] = int(not row['cut_ok'])
                    row['ari_plan'] = 0.0 if row['n_fail'] else row['ari_survivor']
                    graphs.append(row)
                    score_vectors.append(score)
                    targets = {fork: keys[graph, h, ro, fork] for fork in previous.FORKS}
                    targets['adaptive'] = archive[graph, 'huber_local', 'wave', str(h), ro]
                    for baseline, b in targets.items():
                        pair = dict(ref.gmeta(spec, meta), h=h, readout=ro, fork=NAME, baseline=baseline,
                                    fork_n_init=ref.R, fork_n_fail=row['n_fail'], baseline_n_fail=int(b['n_fail']))
                        for metric in ('ari_plan', 'ari_survivor', 'auc', 'gap'):
                            x, y = row[metric], float(b[metric]) if b.get(metric, '') != '' else np.nan
                            pair[f'fork_{metric}'], pair[f'baseline_{metric}'] = x, y
                            pair[f'delta_{metric}'] = x-y
                        pairs.append(pair)
                bank.append((graph, h, g.ei, g.ej, g.ew, meta['bridge_mask'], ratio, np.stack(score_vectors)))
    require((len(bank), len(graphs), len(inits), len(pairs)) == (32, 64, 256, 192), 'Incomplete run coverage')
    for name, data in [('graphs', graphs), ('initializations', inits), ('paired_graphs', pairs),
                       ('paired_summary', previous.summarize(pairs))]:
        ref.write_csv(out / f'{name}.csv', data)
    ref.write_npz(out / 'edges.npz', dict(graph=np.array([b[0] for b in bank]), h=np.array([b[1] for b in bank]),
                  edge_ptr=np.cumsum([0] + [len(b[2]) for b in bank]), i=np.concatenate([b[2] for b in bank]),
                  j=np.concatenate([b[3] for b in bank]), w=np.concatenate([b[4] for b in bank]),
                  bridge=np.concatenate([b[5] for b in bank]), multiplier=np.concatenate([b[6] for b in bank], axis=-1),
                  scores=np.concatenate([b[7] for b in bank], axis=-1), readout=np.array(ref.READOUTS)))
    print(f"Max t=50 force error: {max(r['t50_force_max_abs_error'] for r in inits):.3g}; failed starts: {sum(r['failed'] for r in inits)}/256")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    run(args.out)
