"""SBM amplitude sensitivity using saved amplitude-one starts and scores."""
import argparse
import csv
from itertools import product
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '004b-frozen-directional'))
import directional as direction
from directional import C, np, previous as prev, ref, require

DIRECT = direction.NAME
ARCHIVE = dict(prev.BASELINES)
FORKS = (prev.FREEZE, DIRECT, prev.UNWEIGHTED)
METHODS = ('adaptive', *FORKS, 'pure_mean', 'same_force_diffusion')
AMPS = (0.5, 2.0)


def segment(force, h, A, cur, *, past=None, first=0, last=250, dyn='wave', flags=None):
    """Track first absolute/scaled hits and safety censor on every substep."""
    hits = np.full((3, ref.R), -1) if flags is None else flags.copy()
    frames = []
    every = round(ref.DT_SAVE/h)
    for k in range(round(first/h)+1, round(last/h)+1):
        nxt = prev.wave_step(force, cur, past, h) if dyn == 'wave' else cur + h*force(cur)
        mag = np.max(np.abs(nxt), axis=1)
        finite = np.isfinite(nxt).all(1)
        alive = hits[2] < 0
        for flag, threshold in ((0, ref.AMP), (1, ref.AMP*A)):
            hit = ~finite | (mag >= threshold)
            hits[flag, alive & (hits[flag] < 0) & hit] = k
        hits[2, alive & (~finite | (mag > ref.EMERGENCY))] = k
        nxt[hits[2] >= 0] = 0.0  # keep per-start fixed-force array indexing
        past, cur = cur, nxt
        if k % every == 0 and k*h >= 125:
            frames.append(cur)
    return past, cur, hits, np.stack(frames) if frames else None


def load_csv(folder):
    with (folder / 'graphs.csv').open(newline='') as f:
        return {(r['graph'], float(r['h']), r['fork'], r['readout']): r for r in csv.DictReader(f)}


def baseline_inputs():
    arc = prev.archived_index()
    scores = {}
    zfile = prev.SOURCE / 'part2_edges.npz'
    with np.load(zfile) as z:
        ids, ptr, data = z['graph_ids'], z['edge_ptr'], z['scores']
        for k, (gi, f, d, h) in enumerate(zip(*(z[v] for v in ('run_graph_index', 'run_force', 'run_dyn', 'run_h')))):
            match = [m for m, fd in ARCHIVE.items() if (f, d) == fd]
            if match and ':sbm_' in ids[gi]:
                for j, ro in enumerate(ref.READOUTS):
                    scores[ids[gi], h, match[0], ro] = data[k, 1, j, :ptr[gi+1]-ptr[gi]]
    tables = {**load_csv(prev.HERE), **load_csv(direction.HERE)}
    for folder, forks in ((prev.HERE, prev.FORKS), (direction.HERE, (DIRECT,))):
        with np.load(folder / 'edges.npz') as z:
            data, ptr = z['scores'], z['edge_ptr']
            for k, (graph, h) in enumerate(zip(z['graph'], z['h'])):
                for fi, fork in enumerate(forks):
                    for ri, ro in enumerate(ref.READOUTS):
                        scores[graph, h, fork, ro] = (data[fi, ri] if len(forks) > 1 else data[ri])[ptr[k]:ptr[k+1]]
    for graph, h, method, ro in scores:
        if method in ARCHIVE:
            tables[graph, h, method, ro] = arc[graph, *ARCHIVE[method], str(h), ro]
    require(len(scores) == len(tables) == 16*2*6*2 and scores.keys() == tables.keys(), 'Incomplete amplitude-one joins')
    return tables, scores, arc


def rms(x):
    return float(np.sqrt(np.mean(x*x)))


def main(out):
    base, edge_one, arc = baseline_inputs()
    graphs, inits, edges, pairs, eps_rows = [], [], [], [], []
    with np.load(prev.SOURCE / 'part2_checkpoints.npz') as z:
        starts = dict(zip(z['graph_ids'], z['inits']))
    for spec in ref.p2_specs():
        if spec['block'] != 'sbm':
            continue
        g, truth, meta = C.make_graph(spec)
        graph, metadata = meta['id'], ref.gmeta(spec, meta)
        local, mean = (ref.make_force(g, f) for f in ('huber_local', 'mean'))
        for h, A in product(ref.HS, AMPS):
            u0 = A*starts[graph]
            p, u, warm_hits, _ = segment(local, h, A, u0, last=50)
            require((warm_hits[2] < 0).all(), f'{graph} h={h} A={A}: censored before fork')
            for t, state in ((0, u0), (50, u)):
                delta = local(state)/A - local(state/A)
                bound = ref.EPS*abs(1/A-1)
                require(np.max(np.abs(delta)) <= bound + 1e-14, 'Normalized epsilon bound exceeded')
                for init, v in enumerate(delta):
                    eps_rows.append(dict(**metadata, h=h, amplitude=A, t=t, init=init,
                                         max_abs=float(np.max(np.abs(v))), rms=rms(v), bound=bound))
            sym, _ = prev.frozen_force(g, u)
            direct, _, _ = direction.directional_force(g, u)
            forces = (local, sym, direct, mean, mean, local)
            snapshot = {}
            for k, (method, force) in enumerate(zip(METHODS, forces)):
                dyn = 'diffusion' if method == 'same_force_diffusion' else 'wave'
                kwargs = dict(past=p, first=50, flags=warm_hits) if k < 4 else {}
                _, _, hits, X = segment(force, h, A, u if k < 4 else u0, dyn=dyn, **kwargs)
                require(X.shape == (251, ref.R, g.n), 'Wrong late window size')
                ok = hits[0] < 0
                for init, (absolute, scaled, censor) in enumerate(hits.T):
                    inits.append(dict(**metadata, h=h, amplitude=A, method=method, init=init,
                                      absolute_step=absolute, scaled_step=scaled, censor_step=censor,
                                      absolute_t=absolute*h if absolute >= 0 else '',
                                      scaled_t=scaled*h if scaled >= 0 else ''))
                bank = []
                for ro in ref.READOUTS:
                    key = graph, h, method, ro
                    old, archived = edge_one[key], base[key]
                    row = dict(**metadata, h=h, amplitude=A, method=method, readout=ro, n_init=ref.R,
                               n_absolute_fail=int((~ok).sum()), n_scaled_fail=int((hits[1] >= 0).sum()),
                               n_censored=int((hits[2] >= 0).sum()), n_scored=int(ok.sum()),
                               cut_ok=0, ari_survivor=np.nan, auc=np.nan, gap=np.nan,
                               spectral_ari=float(arc[graph, 'spectral', 'none', 'none', 'none']['ari_plan']))
                    sc = (C.edge_score(X[:, ok], g, frac=1.0, lowpass_fc=ref.FC if ro == 'lp' else None)
                          if ok.any() else np.full_like(old, np.nan))
                    bank.append(sc)
                    if np.isfinite(sc).all():
                        row.update(ref.score_metrics(g, truth, sc, meta['bridge_mask']))
                    delta = sc/A - old
                    sc0 = ref.rx.cut_partition(g, old, 2) if np.isfinite(old).all() else None
                    sc1 = ref.rx.cut_partition(g, sc, 2) if np.isfinite(sc).all() else None
                    if sc0 is not None:
                        gap0 = ref.kruskal_gap(g, old, truth)[0]
                        require(abs(gap0 - float(archived['gap'])) <= 1e-12 and
                                abs(ref.adjusted_rand_score(truth, sc0) - float(archived['ari_survivor'])) <= 1e-12,
                                f'Amplitude-one score/CSV mismatch: {key}')
                    row.update(unscorable=int(not row['cut_ok']), ari_plan=0.0 if (~ok).any() else row['ari_survivor'],
                               scale_max_abs=float(np.max(np.abs(delta))), scale_rms=rms(delta),
                               archive_edge_rms=rms(old), normalized_edge_rms=rms(sc/A),
                               gap_normalized=row['gap']/A, archive_gap=float(archived['gap']),
                               archive_near_tie=int(archived['gap_near_tie']),
                               cut_changed=int(not (np.array_equal(sc0, sc1) or np.array_equal(sc0, 1-sc1)))
                               if sc0 is not None and sc1 is not None else '')
                    graphs.append(row)
                    snapshot[method, ro] = row
                edges.append((graph, h, A, method, g.ei, g.ej, meta['bridge_mask'], np.stack(bank),
                              np.stack([edge_one[graph, h, method, ro] for ro in ref.READOUTS])))
            for method, ro in product(METHODS, ref.READOUTS):
                a = snapshot[method, ro]
                targets = {'amplitude_1': base[graph, h, method, ro]}
                if method in FORKS:
                    targets.update({b: snapshot[b, ro] for b in ('adaptive', prev.UNWEIGHTED) if b != method})
                for label, b in targets.items():
                    row = dict(**metadata, h=h, amplitude=A, method=method, readout=ro, baseline=label)
                    for metric in ('ari_plan', 'ari_survivor'):
                        row[f'delta_{metric}'] = a[metric] - float(b[metric])
                    pairs.append(row)
    require((len(edges), len(graphs), len(inits), len(pairs), len(eps_rows)) == (384, 768, 3072, 1408, 1024),
            'Incomplete run coverage')
    groups = {}
    for p in pairs:
        key = tuple(p[k] for k in ('h', 'amplitude', 'method', 'readout', 'baseline'))
        groups.setdefault(key, []).append(p)
    summary = []
    for key, group in sorted(groups.items()):
        require(len(group) == 16, 'Incomplete paired stratum')
        row = dict(zip(('h', 'amplitude', 'method', 'readout', 'baseline'), key), n_graphs=len(group))
        for metric in ('ari_plan', 'ari_survivor'):
            ds = np.array([p[f'delta_{metric}'] for p in group])
            ok = np.isfinite(ds)
            row[f'n_{metric}'] = int(ok.sum())
            row[f'mean_delta_{metric}'] = float(ds[ok].mean()) if ok.any() else np.nan
        summary.append(row)
    for name, data in [('graphs', graphs), ('initializations', inits), ('paired_graphs', pairs),
                       ('paired_summary', summary), ('epsilon', eps_rows)]:
        ref.write_csv(out / f'{name}.csv', data)
    ref.write_npz(out / 'edges.npz', dict(graph=np.array([e[0] for e in edges]), h=np.array([e[1] for e in edges]),
                  amplitude=np.array([e[2] for e in edges]), method=np.array([e[3] for e in edges]),
                  edge_ptr=np.cumsum([0] + [len(e[4]) for e in edges]), i=np.concatenate([e[4] for e in edges]),
                  j=np.concatenate([e[5] for e in edges]), bridge=np.concatenate([e[6] for e in edges]),
                  scores=np.concatenate([e[7] for e in edges], axis=-1),
                  amplitude_1_scores=np.concatenate([e[8] for e in edges], axis=-1), readout=np.array(ref.READOUTS)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    main(args.out)
