"""Fork 003's archived SBM wave states after an exact adaptive warm-up."""
import argparse
import csv
from itertools import product
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / '003-bridge-selectivity'
sys.path.insert(0, str(SOURCE))
import bridge_selectivity as ref
from bridge_selectivity import C, np

FREEZE, UNWEIGHTED = 'symmetric_freeze', 'unweighted_mean'
FORKS = FREEZE, UNWEIGHTED
BASELINES = {'adaptive': ('huber_local', 'wave'), 'pure_mean': ('mean', 'wave'),
             'same_force_diffusion': ('huber_local', 'diffusion')}
TOL = 1e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rows(filename):
    with (SOURCE / filename).open(newline='') as stream:
        return list(csv.DictReader(stream))


def wave_step(force, cur, prev, h):
    F = force(cur)
    a, b, h2 = 1 + ref.GAMMA*h/2, 1 - ref.GAMMA*h/2, h*h
    return cur + 0.5*h2*F if prev is None else (2*cur - b*prev + h2*F)/a


def warmup(g, u0, h, expected):
    force = ref.make_force(g, 'huber_local')
    prev, cur = None, u0.copy()
    for k in range(1, round(50/h)+1):
        prev, cur = cur, wave_step(force, cur, prev, h)
        require((np.abs(cur) < ref.AMP).all(), f'Unexpected warm-up failure at substep {k}')
    error = float(np.max(np.abs(cur - expected)))
    require(np.isfinite(error) and error <= TOL, f't=50 archive mismatch: {error} > {TOL}')
    return prev, cur, error


def frozen_force(g, cur):
    tau = ref.local_tau(g, cur)
    s = cur[:, g.ej] - cur[:, g.ei]
    ai, aj = (np.divide(np.clip(s, -t, t), s, out=np.ones_like(s), where=s != 0)
              for t in (tau[:, g.ei], tau[:, g.ej]))
    multipliers = (ai + aj)/2
    require(np.isfinite(multipliers).all() and ((0 <= multipliers) & (multipliers <= 1)).all(),
            'Invalid frozen stiffness')
    w = g.ew * multipliers
    d = C.wdeg(g)
    A = np.zeros((len(cur), g.n, g.n))
    A[:, g.ei, g.ej] = A[:, g.ej, g.ei] = w
    A.flags.writeable = False
    degree = A.sum(2)
    degree.flags.writeable = False

    def force(u):
        return (np.einsum('rij,rj->ri', A, u) - degree*u) / d

    # Verify summing oriented fluxes gives the same first force and zero weighted total.
    flux = w*s
    direct = np.zeros_like(cur)
    for r in range(len(cur)):
        np.add.at(direct[r], g.ei, flux[r])
        np.add.at(direct[r], g.ej, -flux[r])
    f = force(cur)
    require(np.max(np.abs(f - direct/d)) <= TOL and np.max(np.abs(f @ d)) <= TOL,
            'Frozen force flux/degree check failed')
    return force, multipliers


def continue_wave(force, prev, cur, h):
    every = round(ref.DT_SAVE/h)
    late = []
    fail = np.full(ref.R, -1)
    censor = fail.copy()
    for k in range(round(50/h)+1, round(ref.T_PHYS/h)+1):
        nxt = wave_step(force, cur, prev, h)
        bad = ~(np.abs(nxt) < ref.AMP).all(1)
        stopped = ~np.isfinite(nxt).all(1) | (np.abs(nxt).max(1) > ref.EMERGENCY)
        fail[(fail < 0) & bad] = k
        censor[(censor < 0) & stopped] = k
        nxt[censor >= 0] = np.nan
        prev, cur = cur, nxt
        if k % every == 0 and k*h >= 125:
            late.append(cur)
    X = np.stack(late)
    require(len(X) == 251, 'Wrong number of saved late frames')
    return X, fail, censor


def archived_index():
    index = {}
    for row in rows('part2_scores.csv'):
        if row['family'] != 'sbm' or row['window'] not in ('late', 'none'):
            continue
        key = tuple(row[k] for k in ('graph', 'force', 'dyn', 'h', 'readout'))
        require(key not in index, f'Duplicate archived score {key}')
        index[key] = row
    return index


def summarize(pairs):
    result = []
    keys = ('h', 'readout', 'fork', 'baseline')
    groups = {}
    for r in pairs:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    for key, group in sorted(groups.items()):
        require(len(group) == 16, 'Incomplete pairing')
        row = dict(zip(keys, key), graphs=len(group), runs=sum(r['fork_n_init'] for r in group),
                   runs_failed=sum(r['fork_n_fail'] for r in group),
                   graphs_failed=sum(r['fork_n_fail'] > 0 for r in group))
        for metric in ('ari_plan', 'ari_survivor', 'auc', 'gap'):
            d = np.array([r[f'delta_{metric}'] for r in group])
            ok = np.isfinite(d)
            row[f'n_{metric}'] = int(ok.sum())
            row[f'mean_delta_{metric}'] = float(d[ok].mean()) if ok.any() else np.nan
        result.append(row)
    return result


def main(out):
    index = archived_index()
    scores, inits, checks, pairs = [], [], [], []
    bank = []
    with np.load(SOURCE / 'part2_checkpoints.npz') as z:
        ids, states = z['graph_ids'], z['states']
        run_keys = list(zip(z['run_graph_index'], z['run_force'], z['run_dyn'], z['run_h']))
        run_idx = {key: i for i, key in enumerate(run_keys)}
        require(len(run_idx) == len(run_keys), 'Duplicate checkpoint key')
        for spec in ref.p2_specs():
            if spec['block'] != 'sbm':
                continue
            g, truth, meta = C.make_graph(spec)
            graph = meta['id']
            gi = np.flatnonzero(ids == graph)
            require(len(gi) == 1, f'Bad archived graph ID: {graph}')
            gi = int(gi[0])
            require(np.array_equal(truth, z['truth'][gi]), f'Truth mismatch: {graph}')
            bm = meta['bridge_mask']
            for h in ref.HS:
                try:
                    ckpt = states[run_idx[gi, 'huber_local', 'wave', h]]
                    prev, cur, err = warmup(g, ckpt[0], h, ckpt[1])
                except (KeyError, ValueError) as e:
                    raise RuntimeError(f'Warm-up blocked on {graph}, h={h}: {e}') from e
                checks.append(dict(graph=graph, h=h, max_abs_error_t50=err, tolerance=TOL))
                force, mult = frozen_force(g, cur)
                stored = np.full((2, 2, len(g.ew)), np.nan)
                gscore = {}
                for fi, (fork, Fn) in enumerate(zip(FORKS, (force, ref.make_force(g, 'mean')))):
                    X, fail, censor = continue_wave(Fn, prev, cur, h)
                    survivor = fail < 0
                    for r, (f, c) in enumerate(zip(fail, censor)):
                        inits.append(dict(graph=graph, h=h, fork=fork, init=r, failed=int(f >= 0),
                                          first_fail_substep=f, first_fail_t=f*h if f >= 0 else '',
                                          censor_substep=c, censor_t=c*h if c >= 0 else ''))
                    for ri, readout in enumerate(ref.READOUTS):
                        row = dict(ref.gmeta(spec, meta), h=h, fork=fork, window='late', readout=readout,
                                   n_init=ref.R, n_fail=int((~survivor).sum()), n_censored=int((censor >= 0).sum()),
                                   n_scored=int(survivor.sum()), cut_ok=0, ari_survivor=np.nan, auc=np.nan, gap=np.nan)
                        if survivor.any():
                            sc = C.edge_score(X[:, survivor], g, frac=1.0,
                                              lowpass_fc=ref.FC if readout == 'lp' else None)
                            if np.isfinite(sc).all():
                                stored[fi, ri] = sc
                                row.update(ref.score_metrics(g, truth, sc, bm))
                        row['unscorable'] = int(not row['cut_ok'])
                        row['ari_plan'] = 0.0 if row['n_fail'] else row['ari_survivor']
                        scores.append(row)
                        gscore[fork, readout] = row
                bank.append((graph, h, g.ei, g.ej, g.ew, bm, mult, stored))
                for fork, readout in product(FORKS, ref.READOUTS):
                    a = gscore[fork, readout]
                    targets = {name: index[graph, *method, str(h), readout] for name, method in BASELINES.items()}
                    targets['spectral'] = index[graph, 'spectral', 'none', 'none', 'none']
                    if fork == FREEZE:
                        targets[UNWEIGHTED] = gscore[UNWEIGHTED, readout]
                    for name, b in targets.items():
                        row = dict(ref.gmeta(spec, meta), h=h, readout=readout, fork=fork, baseline=name,
                                   fork_n_init=a['n_init'], fork_n_fail=a['n_fail'],
                                   baseline_n_fail=b['n_fail'] if name != 'spectral' else '')
                        for metric in ('ari_plan', 'ari_survivor', 'auc', 'gap'):
                            value = b.get(metric, '')
                            x, y = float(a[metric]), float(value) if value != '' else np.nan
                            row[f'fork_{metric}'], row[f'baseline_{metric}'] = x, y
                            row[f'delta_{metric}'] = x - y
                        pairs.append(row)
    require(len(checks) == 32 and len(scores) == 128 and len(inits) == 512 and len(pairs) == 576,
            'Incomplete fork coverage')
    for filename, data in [('warmup', checks), ('graphs', scores), ('initializations', inits),
                           ('paired_graphs', pairs), ('paired_summary', summarize(pairs))]:
        ref.write_csv(out / f'{filename}.csv', data)
    lengths = [len(b[2]) for b in bank]
    ref.write_npz(out / 'edges.npz', dict(graph=np.array([b[0] for b in bank]), h=np.array([b[1] for b in bank]),
                  edge_ptr=np.cumsum([0, *lengths]), i=np.concatenate([b[2] for b in bank]),
                  j=np.concatenate([b[3] for b in bank]), w=np.concatenate([b[4] for b in bank]),
                  bridge=np.concatenate([b[5] for b in bank]), multiplier=np.concatenate([b[6] for b in bank], axis=-1),
                  scores=np.concatenate([b[7] for b in bank], axis=-1), fork=np.array(FORKS), readout=np.array(ref.READOUTS)))
    print(f'Validated {len(checks)} warm-ups; max state error {max(r["max_abs_error_t50"] for r in checks):.3g}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    main(out)
