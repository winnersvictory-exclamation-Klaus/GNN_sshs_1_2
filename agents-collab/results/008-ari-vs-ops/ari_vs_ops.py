"""008: ARI vs cumulative edge operations for diffusion variants and label-free-damped wave.

Reuses 003's 48 development graphs, saved random starts (part2_checkpoints.npz), forces and cutter.
Every CHECK_EVERY steps the q=2 partition is cut from the edge-RMS over a trailing window of fixed
physical length (raw and low-pass); ARI is descriptive only.
"""
import argparse
from itertools import product
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / '003-bridge-selectivity'
sys.path.insert(0, str(SOURCE))
import bridge_selectivity as ref  # noqa: E402
from bridge_selectivity import C, np, adjusted_rand_score  # noqa: E402
from scipy.sparse.linalg import LinearOperator, eigsh  # noqa: E402

rx = C.rx
T_PHYS = 250.0
WIN_T = 50.0            # trailing window, physical time
FC_T = 0.04             # low-pass cutoff, cycles per unit time (= 003's 0.02 per 0.5-spaced frame)
CHECK_EVERY = 10
TARGET, STREAK = 0.8, 3
SAFETY = 1.8            # fraction of the stability limit used for "largest stable" steps
LMAX_ITERS = 50         # power-iteration matvecs for lambda_max
LAM_ITERS = 30          # Lanczos matvecs for the smallest nonzero eigenvalue
READOUTS = ('raw', 'lp')
R = ref.R
GRID = (25, 50, 100, 200, 400, 800, 1600)   # ops per start per edge, for the summary curves
FAMILIES = ref.FAMILIES


def operators(g):
    """Symmetric normalized adjacency N = D^-1/2 A D^-1/2 (spectrum of I-N = spectrum of L_rw) and sqrt(d)."""
    d = C.wdeg(g)
    s = np.sqrt(d)
    return g.A / np.outer(s, s), s / np.linalg.norm(s)


def est_lmax(N, seed=0):
    """Power iteration on L = I - N (LMAX_ITERS matvecs, Rayleigh quotient)."""
    x = np.random.default_rng([8, 1, seed]).standard_normal(N.shape[0])
    for _ in range(LMAX_ITERS):
        x /= np.linalg.norm(x)
        y = x - N @ x
        lam = float(x @ y)
        x = y
    return lam


def est_lam2(N, top, seed=0):
    """Lanczos (full reorthogonalization), LAM_ITERS matvecs, on N restricted to the complement of the trivial
    eigenvector. Returns 1 - largest Ritz value, the estimate of the smallest nonzero eigenvalue of L."""
    n = N.shape[0]
    q = np.random.default_rng([8, 2, seed]).standard_normal(n)
    q -= top*(top @ q)
    Q = [q/np.linalg.norm(q)]
    al, be = [], []
    for k in range(LAM_ITERS):
        w = N @ Q[k]
        w -= top*(top @ w)
        al.append(float(Q[k] @ w))
        for v in Q:
            w -= v*(v @ w)
        b = np.linalg.norm(w)
        if k == LAM_ITERS-1 or b < 1e-12:
            break
        be.append(b)
        Q.append(w/b)
    T = np.diag(al) + np.diag(be[:len(al)-1], 1) + np.diag(be[:len(al)-1], -1)
    return float(1 - np.linalg.eigvalsh(T)[-1]), len(al)


def force_for(g, name):
    if name == 'mean':
        d = C.wdeg(g)
        A = g.A
        return lambda u: (u @ A.T)/d - u
    return ref.make_force(g, name)


def trajectory(force, u0, h, dyn, d, gamma, steps):
    """States at every step (steps+1, R, n), plus per-start first failure step (-1 = none; |u|>=20 or nonfinite)."""
    X = np.full((steps+1, *u0.shape), np.nan)
    X[0] = u0
    ff = np.full(u0.shape[0], -1)
    prev, cur = None, u0.astype(float)
    a, b = 1 + gamma*h/2, 1 - gamma*h/2
    with np.errstate(all='ignore'):
        for k in range(steps):
            F = force(cur)
            if dyn == 'diffusion':
                nxt = cur + h*F
            else:
                nxt = cur + 0.5*h*h*F if k == 0 else (2*cur - b*prev + h*h*F)/a
            bad = (ff < 0) & ~(np.abs(nxt) < ref.AMP).all(1)
            ff[bad] = k+1
            nxt = np.where(np.isfinite(nxt) & (np.abs(nxt) < 1e8), nxt, np.nan)   # censor runaway starts
            X[k+1] = nxt
            prev, cur = cur, nxt
    return X, ff


def edge_scores(g, X, h, lowpass):
    """X: (S, R, n) window states -> median over starts of per-start edge RMS."""
    if lowpass:
        X = rx.lowpass(X, FC_T*h)
    s = X[:, :, g.ej] - X[:, :, g.ei]
    return np.median(np.sqrt((s*s).mean(0)), axis=0)


def streak_hit(ok):
    """Index of the first check that starts STREAK consecutive passes; -1 if none."""
    run = 0
    for i, v in enumerate(ok):
        run = run+1 if v else 0
        if run == STREAK:
            return i-STREAK+1
    return -1


def run_config(g, truth, u0, force, h, dyn, gamma, steps, d, setup_ops_edges, m):
    X, ff = trajectory(force, u0, h, dyn, d, gamma, steps)
    win = max(2, int(round(WIN_T/h)) + 1)
    rows = []
    for k in range(CHECK_EVERY, steps+1, CHECK_EVERY):
        failed = (ff >= 0) & (ff <= k)
        alive = np.isfinite(X[k]).all(1)
        W = X[max(0, k-win+1):k+1]
        for ro in READOUTS:
            ari, cut_ok = 0.0, 0
            if alive.any():
                sc = edge_scores(g, W[:, alive], h, ro == 'lp')
                lab = rx.cut_partition(g, sc, 2) if np.isfinite(sc).all() else None
                if lab is not None:
                    cut_ok = 1
                    ari = float(adjusted_rand_score(truth, lab))
            if failed.any():
                ari = 0.0       # plan score: any failed start zeroes the graph
            # per start: one force evaluation and one edge-difference scoring pass per step; setup is shared
            rows.append(dict(readout=ro, check=k//CHECK_EVERY, step=k, t=k*h, ari=ari, cut_ok=cut_ok,
                             n_failed=int(failed.sum()),
                             ops_force=R*m*k, ops_total=R*m*2*k + setup_ops_edges*m))
    return rows


def spectral(g, truth, d, m):
    N, _ = operators(g)
    count = [0]

    def mv(x):
        count[0] += 1
        return N @ x
    op = LinearOperator(N.shape, matvec=mv, dtype=float)
    v0 = np.random.default_rng([8, 3]).standard_normal(g.n)
    w, V = eigsh(op, k=2, which='LA', v0=v0)
    v = V[:, np.argsort(w)[0]]    # second-largest eigenvalue of N
    lab = rx.split_1d(v/np.sqrt(d))
    return float(adjusted_rand_score(truth, lab)), count[0]


def configs(lmax, lam2):
    gstar = 2*np.sqrt(lam2)
    h_fast, h_wave = SAFETY/lmax, SAFETY/np.sqrt(lmax)
    return [
        ('diff_h0.25', 'diffusion', 'mean', 0.25, 0.0, 0),
        ('diff_fast', 'diffusion', 'mean', h_fast, 0.0, LMAX_ITERS),
        ('huber_local_h0.25', 'diffusion', 'huber_local', 0.25, 0.0, 0),
        ('huber_local_h0.5', 'diffusion', 'huber_local', 0.5, 0.0, 0),
        ('wave_gstar', 'wave', 'mean', h_wave, gstar, LMAX_ITERS+LAM_ITERS),
        ('wave_g0.1', 'wave', 'mean', h_wave, 0.1, LMAX_ITERS),
        ('wave_0.5gstar', 'wave', 'mean', h_wave, 0.5*gstar, LMAX_ITERS+LAM_ITERS),
        ('wave_2gstar', 'wave', 'mean', h_wave, 2*gstar, LMAX_ITERS+LAM_ITERS),
    ]


def write(path, rows):
    ref.write_csv(path, rows)


def run(out):
    with np.load(SOURCE / 'part2_checkpoints.npz') as z:
        ids, inits, truths = z['graph_ids'], z['inits'], z['truth']
    if len(set(ids)) != len(ids) or inits.shape != (48, R, 48):
        raise ValueError('Bad archived start coverage')
    starts = dict(zip(ids, zip(inits, truths)))
    curves, graph_rows, targets, specs = [], [], [], ref.p2_specs()
    for spec in specs:
        g, truth, meta = C.make_graph(spec)
        gid, m, d = meta['id'], len(g.ei), C.wdeg(g)
        u0, atruth = starts[gid]
        if not np.array_equal(truth, atruth):
            raise ValueError(f'Archive truth mismatch: {gid}')
        base = ref.gmeta(spec, meta)
        N, top = operators(g)
        lmax, (lam2, nmv) = est_lmax(N), est_lam2(N, top)
        ev = np.linalg.eigvalsh(np.eye(g.n) - N)      # check only; never used to pick steps
        sp_ari, sp_mv = spectral(g, truth, d, m)
        graph_rows.append(dict(base, n=g.n, m=m, lmax_est=lmax, lmax_true=float(ev[-1]), lam2_est=lam2,
                               lam2_true=float(ev[1]), lanczos_matvecs=nmv, gstar=2*np.sqrt(lam2),
                               spectral_ari=sp_ari, spectral_matvecs=sp_mv, spectral_ops=sp_mv*m))
        for name, dyn, fname, h, gamma, setup in configs(lmax, lam2):
            steps = int(np.ceil(T_PHYS/h))
            stable = (h*ev[-1] < 2) if dyn == 'diffusion' else (h*h*ev[-1] < 4)
            force = force_for(g, fname)
            rows = run_config(g, truth, u0, force, h, dyn, gamma, steps, d, setup, m)
            for r in rows:
                curves.append(dict(base, config=name, h=h, gamma=gamma, **r))
            for ro in READOUTS:
                rr = [r for r in rows if r['readout'] == ro]
                i = streak_hit([r['ari'] >= TARGET for r in rr])
                t = dict(base, m=m, config=name, dyn=dyn, h=h, gamma=gamma, readout=ro, steps=steps, stable_true=int(stable),
                         checks=len(rr), n_failed_final=rr[-1]['n_failed'], final_ari=rr[-1]['ari'],
                         max_ari=max(r['ari'] for r in rr), reached=int(i >= 0),
                         target_check=rr[i]['check'] if i >= 0 else '', target_t=rr[i]['t'] if i >= 0 else '',
                         ops_target=rr[i]['ops_total'] if i >= 0 else '',
                         ops_target_confirm=rr[i+STREAK-1]['ops_total'] if i >= 0 else '',
                         ops_force_target=rr[i]['ops_force'] if i >= 0 else '',
                         ari_at_target=rr[i]['ari'] if i >= 0 else '')
                targets.append(t)
    return curves, graph_rows, targets


def paired(targets, num, den, key='ops_target'):
    """Graph-paired ops-to-target ratios, per family and readout; denominators explicit."""
    idx = {(t['graph'], t['config'], t['readout']): t for t in targets}
    out = []
    for fam, ro in product((*FAMILIES, 'all'), READOUTS):
        gs = sorted({t['graph'] for t in targets if fam == 'all' or t['family'] == fam})
        both, only_n, only_d, neither, ratios, an, ad = 0, 0, 0, 0, [], [], []
        for gid in gs:
            a, b = idx[gid, num, ro], idx[gid, den, ro]
            if a['reached'] and b['reached']:
                both += 1
                ratios.append(a[key]/b[key])
                an.append(a['ari_at_target'])
                ad.append(b['ari_at_target'])
            elif a['reached']:
                only_n += 1
            elif b['reached']:
                only_d += 1
            else:
                neither += 1
        r = np.array(ratios)
        out.append(dict(family=fam, readout=ro, numerator=num, denominator=den, metric=key, graphs=len(gs),
                        both_reach=both, only_numerator_reaches=only_n, only_denominator_reaches=only_d,
                        neither_reaches=neither,
                        median_ratio=float(np.median(r)) if both else '', geomean_ratio=float(np.exp(np.log(r).mean())) if both else '',
                        min_ratio=float(r.min()) if both else '', max_ratio=float(r.max()) if both else '',
                        frac_num_cheaper=float((r < 1).mean()) if both else '',
                        mean_ari_at_target_num=float(np.mean(an)) if both else '',
                        mean_ari_at_target_den=float(np.mean(ad)) if both else ''))
    return out


def reach_summary(targets):
    out = []
    keys = sorted({(t['config'], t['readout']) for t in targets})
    for (cfg, ro), fam in product(keys, (*FAMILIES, 'all')):
        rows = [t for t in targets if t['config'] == cfg and t['readout'] == ro and (fam == 'all' or t['family'] == fam)]
        hit = [t for t in rows if t['reached']]
        out.append(dict(config=cfg, readout=ro, family=fam, graphs=len(rows), reached=len(hit),
                        failed_graphs=sum(t['n_failed_final'] > 0 for t in rows),
                        unstable_by_dense_check=sum(not t['stable_true'] for t in rows),
                        median_ops_target=float(np.median([t['ops_target'] for t in hit])) if hit else '',
                        median_ops_per_start_edge=float(np.median([t['ops_target']/(R*t['m']) for t in hit])) if hit else '',
                        mean_final_ari=float(np.mean([t['final_ari'] for t in rows])),
                        mean_max_ari=float(np.mean([t['max_ari'] for t in rows]))))
    return out


def curve_summary(curves, graph_rows):
    """Mean ARI vs ops grid (ops per start per edge): the last check at or below each grid value; graphs with no
    check yet at that budget count as ARI 0 (n_started gives the number that had one)."""
    by = {}
    for c in curves:
        by.setdefault((c['graph'], c['config'], c['readout']), []).append(c)
    meta = {r['graph']: r for r in graph_rows}
    out = []
    keys = sorted({(k[1], k[2]) for k in by})
    for (cfg, ro), fam in product(keys, (*FAMILIES, 'all')):
        gs = [k[0] for k in by if k[1] == cfg and k[2] == ro and (fam == 'all' or meta[k[0]]['family'] == fam)]
        for gr in GRID:
            vals, started = [], 0
            for gid in gs:
                m = meta[gid]['m']
                ok = [c for c in by[gid, cfg, ro] if c['ops_total']/(R*m) <= gr]
                started += bool(ok)
                vals.append(ok[-1]['ari'] if ok else 0.0)
            out.append(dict(config=cfg, readout=ro, family=fam, ops_per_start_edge=gr, graphs=len(gs),
                            n_started=started, mean_ari=float(np.mean(vals)),
                            frac_ge_target=float(np.mean([v >= TARGET for v in vals]))))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    out = p.parse_args().out
    tmp = out.with_name(out.name + '.tmp')
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    curves, graphs, targets = run(tmp.resolve())
    n_cfg = len({t['config'] for t in targets})
    assert len(graphs) == 48 and len(targets) == 48*n_cfg*len(READOUTS) and n_cfg == 8, 'coverage'
    write(tmp / 'graphs.csv', graphs)
    write(tmp / 'curves.csv.gz', curves)
    write(tmp / 'targets.csv', targets)
    write(tmp / 'reach_summary.csv', reach_summary(targets))
    write(tmp / 'curve_summary.csv', curve_summary(curves, graphs))
    pr = []
    for num, den in [('wave_gstar', 'diff_fast'), ('huber_local_h0.25', 'diff_fast'), ('huber_local_h0.5', 'diff_fast'),
                     ('wave_g0.1', 'diff_fast'), ('wave_0.5gstar', 'diff_fast'), ('wave_2gstar', 'diff_fast'),
                     ('diff_h0.25', 'diff_fast')]:
        pr += paired(targets, num, den)
    write(tmp / 'paired_ops_ratio.csv', pr)
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    print(f'{len(targets)} target rows; {len(curves)} curve rows')


if __name__ == '__main__':
    main()
