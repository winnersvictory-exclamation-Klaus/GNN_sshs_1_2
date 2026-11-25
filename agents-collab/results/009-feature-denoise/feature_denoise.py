"""009: feature denoising with mean / fixed Huber / endpoint-local Huber smoothing (diffusion and damped wave).

Node features x0 = m*s + sigma*z, s = +-1 block sign, m = 1, sigma in {1,2,3}. Each method is evolved on a
geometric grid of times; at each time we record sign accuracy (after subtracting the degree-weighted global mean),
MSE to m*s after the optimal scalar rescale, and the bridge/interior contrast of |edge difference| (mean over edges; median over the 8 draws, then median over graphs;
draws with interior mean < 1e-9 are dropped).
"""
import argparse
import csv
import gzip
import io
from itertools import product
from pathlib import Path
import shutil
import sys
import warnings

sys.dont_write_bytecode = True
warnings.filterwarnings('ignore', message='Mean of empty slice')
warnings.filterwarnings('ignore', message='All-NaN slice')
HERE = Path(__file__).resolve().parent
RES = HERE.parent
sys.path[:0] = [str(RES/'003-bridge-selectivity'), str(RES/'008-ari-vs-ops'), str(RES.parent/'graph-maker'/'default-graphs')]
import bridge_selectivity as ref  # noqa: E402
import ari_vs_ops as aop  # noqa: E402  (008: operators, est_lam2)
import generators as gen  # noqa: E402
from bridge_selectivity import C, np  # noqa: E402

H = 0.25                  # one step for all methods (003's h); stable for lmax <= 2
M_SIG = 1.0
SIGMAS = (1.0, 2.0, 3.0)  # sigma/m
DRAWS = 8
STEPS = np.unique(np.round(np.geomspace(1, 2048, 26)).astype(int))   # t = 0.25 .. 512
BLOW = 1e3                # failure: nonfinite or max|x| above this
FORCES = ('mean', 'huber', 'huber_local')
DYNS = ('diffusion', 'wave')
METHODS = [f'{f}_{d}' for f, d in product(FORCES, DYNS)]
BASE = 'mean_diffusion'
TRAIN_SEEDS, TEST_SEEDS = (0, 1), (2, 3)
BOOT = 2000


def graph_list():
    """(family, id, builder args) for every development graph; seeds 0-3."""
    out = []
    for seed in range(4):
        for sizes, nb in product([(24, 24), (16, 32)], (1, 4, 16)):
            out.append(('sbm', f'sbm_{sizes[0]}x{sizes[1]}_b{nb}', seed,
                        lambda s=seed, z=sizes, b=nb: gen.two_block('sbm', z, b, 1.0, s)))
            out.append(('sbm_het', f'sbm_het_{sizes[0]}x{sizes[1]}_b{nb}', seed,
                        lambda s=seed, z=sizes, b=nb: gen.sbm_het(z, b, 1.0, 1.0, 0.3, s)))
        for w in (1, 2, 4):
            out.append(('grid_neck', f'grid_neck_a5_w{w}', seed, lambda w=w: gen.grid_neck(5, w, 1.0)))
        for k in (16, 20):
            out.append(('barbell', f'barbell_k{k}_L4', seed, lambda k=k: gen.barbell_path(k, 4)))
        # q=2 is not allowed by ring_of_cliques: q=4, s=12, sign + for cliques 0,1 and - for cliques 2,3
        out.append(('ring4', 'ring_of_cliques_q4_s12', seed,
                    lambda: (lambda n, e, lab: (n, e, (lab >= 2).astype(int)))(*gen.ring_of_cliques(4, 12))))
    return out


def make(build):
    n, edges, truth = build()
    g = C.rx.G(n, edges)
    return g, np.asarray(truth)


def evolve(force, X0, dyn, gamma, nmax):
    """Return states at STEPS (len(STEPS)+1 recorded incl. t=0), plus first-failure flag per row."""
    rec = {0: X0}
    want = set(STEPS.tolist())
    prev, cur = None, X0.copy()
    a, b, h2 = 1 + gamma*H/2, 1 - gamma*H/2, H*H
    dead = np.zeros(X0.shape[0], bool)
    with np.errstate(all='ignore'):
        for k in range(nmax):
            F = force(cur)
            if dyn == 'diffusion':
                nxt = cur + H*F
            else:
                nxt = cur + 0.5*h2*F if k == 0 else (2*cur - b*prev + h2*F)/a
            dead |= ~np.isfinite(nxt).all(1) | (np.abs(nxt) > BLOW).any(1)
            nxt = np.where(dead[:, None], np.nan, nxt)
            prev, cur = cur, nxt
            if k+1 in want:
                rec[k+1] = nxt
    return rec, dead


def metrics(x, g, d, s, bridge):
    """x (R,n) -> accuracy (threshold 0), best-of-both-signs accuracy, rescaled MSE, bridge/interior contrast."""
    xc = x - (x @ d/d.sum())[:, None]
    pred = np.sign(xc)
    acc = ((pred == s).astype(float) + 0.5*(pred == 0)).mean(1)
    nrm = np.sqrt((xc*xc).sum(1))
    cos = (xc @ s)/np.where(nrm > 0, nrm, np.nan)/np.sqrt(len(s))
    mse = M_SIG**2*(1 - np.nan_to_num(cos, nan=0.0)**2)
    ad = np.abs(x[:, g.ej] - x[:, g.ei])
    inter = ad[:, ~bridge].mean(1)
    contrast = np.where(inter > 1e-9, ad[:, bridge].mean(1)/np.where(inter > 1e-9, inter, 1), np.nan)
    return acc, np.maximum(acc, 1 - acc), mse, contrast


def run():
    rows, glist = [], []
    for fam, gid, seed, build in graph_list():
        g, truth = make(build)
        d = C.wdeg(g)
        s = 1.0 - 2.0*truth                        # +1 for label 0, -1 for label 1
        bridge = truth[g.ei] != truth[g.ej]
        N, top = aop.operators(g)
        lmax = aop.est_lmax(N)
        lam2, _ = aop.est_lam2(N, top)
        gstar = 2*np.sqrt(lam2)
        assert H*lmax < 2 and H*H*lmax < 4
        # features: noise z shared across sigma; row index = sigma_idx*DRAWS + draw
        Z = np.stack([np.random.default_rng([9, seed, dr]).standard_normal(g.n) for dr in range(DRAWS)])
        X0 = np.concatenate([M_SIG*s + sg*Z for sg in SIGMAS])
        glist.append(dict(family=fam, graph=gid, seed=seed, n=g.n, m=len(g.ei), n_bridge_edges=int(bridge.sum()),
                          lmax_est=lmax, lam2_est=lam2, gstar=gstar))
        for f, dyn in product(FORCES, DYNS):
            rec, dead = evolve(ref.make_force(g, f), X0, dyn, gstar if dyn == 'wave' else 0.0, int(STEPS.max()))
            for step in [0, *STEPS.tolist()]:
                acc, accb, mse, con = metrics(rec[step], g, d, s, bridge)
                for si, sg in enumerate(SIGMAS):
                    sl = slice(si*DRAWS, (si+1)*DRAWS)
                    ok = ~dead[sl] if step else np.ones(DRAWS, bool)
                    ok &= np.isfinite(acc[sl])
                    rows.append(dict(family=fam, graph=gid, seed=seed, method=f'{f}_{dyn}', sigma=sg, step=step,
                                     t=step*H, draws=DRAWS, draws_failed=int(DRAWS - ok.sum()),
                                     acc=float(acc[sl][ok].mean()) if ok.any() else np.nan,
                                     acc_best=float(accb[sl][ok].mean()) if ok.any() else np.nan,
                                     mse=float(mse[sl][ok].mean()) if ok.any() else np.nan,
                                     contrast=float(np.nanmedian(con[sl][ok])) if ok.any() and np.isfinite(con[sl][ok]).any() else np.nan))
    return rows, glist


def table(rows):
    """Index per-graph curves: idx[(family, graph, seed, method, sigma)] -> {step: row}."""
    idx = {}
    for r in rows:
        idx.setdefault((r['family'], r['graph'], r['seed'], r['method'], r['sigma']), {})[r['step']] = r
    return idx


def select_and_test(rows):
    """Frozen t* per (family, sigma, method) from seeds 0-1 (label-based, mean accuracy); applied to seeds 2-3."""
    idx = table(rows)
    steps = [0, *STEPS.tolist()]
    fams = sorted({r['family'] for r in rows})
    keys = sorted({k[:3] for k in idx})
    per_graph, sel = [], []
    for fam, sg, meth in product(fams, SIGMAS, [*METHODS]):
        tr = [k for k in idx if k[0] == fam and k[3] == meth and k[4] == sg and k[2] in TRAIN_SEEDS]
        te = [k for k in idx if k[0] == fam and k[3] == meth and k[4] == sg and k[2] in TEST_SEEDS]
        curve = np.array([np.nanmean([idx[k][st]['acc'] for k in tr]) for st in steps])
        best = steps[int(np.argmax(np.where(np.isnan(curve), -1, curve)))]    # ties -> earliest
        sel.append(dict(family=fam, sigma=sg, method=meth, train_graphs=len(tr), test_graphs=len(te),
                        t_star_step=best, t_star=best*H, train_acc=float(np.nanmax(curve))))
        for k in te:
            c = idx[k]
            lin = idx[(k[0], k[1], k[2], BASE, sg)]
            orac = max(steps, key=lambda st: (np.nan_to_num(c[st]['acc'], nan=-1), -st))
            lin_orac = max(steps, key=lambda st: (np.nan_to_num(lin[st]['acc'], nan=-1), -st))
            per_graph.append(dict(family=fam, graph=k[1], seed=k[2], sigma=sg, method=meth,
                                  t_star=best*H, acc_frozen=c[best]['acc'], acc_best_frozen=c[best]['acc_best'],
                                  mse_frozen=c[best]['mse'], contrast_frozen=c[best]['contrast'],
                                  acc_baseline_x0=c[0]['acc'], acc_best_baseline_x0=c[0]['acc_best'],
                                  oracle_t=orac*H, acc_oracle=c[orac]['acc'],
                                  acc_best_oracle=c[orac]['acc_best'],
                                  linear_oracle_t=lin_orac*H, acc_linear_oracle=lin[lin_orac]['acc'],
                                  failed_draws=int(sum(c[st]['draws_failed'] for st in steps))))
    return sel, per_graph


def boot_ci(v, rng):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan, 0
    bs = v[rng.integers(0, len(v), (BOOT, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(v)


def paired_tables(per_graph):
    """Graph-paired differences on test graphs; bootstrap CI over graphs. Also summary of absolute accuracy."""
    rng = np.random.default_rng([9, 99])
    pg = {(p['family'], p['graph'], p['seed'], p['sigma'], p['method']): p for p in per_graph}
    summ, paired = [], []
    fams = sorted({p['family'] for p in per_graph})
    for fam, sg, meth in product([*fams, 'all'], SIGMAS, METHODS):
        rows = [p for p in per_graph if p['method'] == meth and p['sigma'] == sg and (fam == 'all' or p['family'] == fam)]
        row = dict(family=fam, sigma=sg, method=meth, test_graphs=len(rows), draws_per_graph=DRAWS,
                   failed_draw_steps=sum(p['failed_draws'] for p in rows))
        for col in ('acc_baseline_x0', 'acc_best_baseline_x0', 'acc_frozen', 'acc_best_frozen', 'mse_frozen',
                    'contrast_frozen', 'acc_oracle', 'acc_best_oracle', 'acc_linear_oracle'):
            row[col] = float((np.nanmedian if col == 'contrast_frozen' else np.nanmean)([p[col] for p in rows])) if rows else np.nan
        summ.append(row)
        if meth == BASE:
            continue
        diffs = {}
        for name, a, b in [('frozen_vs_frozen_linear', 'acc_frozen', 'acc_frozen'),
                           ('oracle_vs_oracle_linear', 'acc_oracle', 'acc_linear_oracle'),
                           ('frozen_vs_x0', 'acc_frozen', 'acc_baseline_x0'),
                           ('best_sign_frozen_vs_frozen_linear', 'acc_best_frozen', 'acc_best_frozen')]:
            v = []
            for p in rows:
                lin = pg[p['family'], p['graph'], p['seed'], sg, BASE]
                other = lin[b] if name != 'frozen_vs_x0' else p['acc_baseline_x0']
                v.append(p[a] - other)
            mean, lo, hi, n = boot_ci(v, rng)
            paired.append(dict(family=fam, sigma=sg, method=meth, comparison=name, graphs=n, mean_diff=mean,
                               ci_lo=lo, ci_hi=hi))
    return summ, paired


def curve_summary(rows):
    """Mean over graphs of accuracy/MSE/contrast per (family, split, sigma, method, t)."""
    out = []
    groups = {}
    for r in rows:
        split = 'train' if r['seed'] in TRAIN_SEEDS else 'test'
        for fam in (r['family'], 'all'):
            for sp in (split, 'both'):
                groups.setdefault((fam, sp, r['sigma'], r['method'], r['step']), []).append(r)
    for (fam, sp, sg, meth, step), rs in sorted(groups.items()):
        row = dict(family=fam, split=sp, sigma=sg, method=meth, step=step, t=step*H, graphs=len(rs),
                   draws_failed=sum(r['draws_failed'] for r in rs))
        for col in ('acc', 'acc_best', 'mse', 'contrast'):
            row[col] = float((np.nanmedian if col == 'contrast' else np.nanmean)([r[col] for r in rs]))
        out.append(row)
    return out


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    f = io.TextIOWrapper(gzip.GzipFile(path, 'wb', mtime=0), newline='') if str(path).endswith('.gz') else open(path, 'w', newline='')
    with f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow({k: (v.item() if isinstance(v, np.generic) else v) for k, v in r.items()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    out = p.parse_args().out
    tmp = out.with_name(out.name + '.tmp')
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    rows, graphs = run()
    assert len(graphs) == len(graph_list()) and len(rows) == len(graphs)*len(METHODS)*len(SIGMAS)*(len(STEPS)+1)
    sel, per_graph = select_and_test(rows)
    summ, paired = paired_tables(per_graph)
    write_csv(tmp/'graphs.csv', graphs)
    write_csv(tmp/'curves_per_graph.csv.gz', rows)
    write_csv(tmp/'curve_summary.csv', curve_summary(rows))
    write_csv(tmp/'selection.csv', sel)
    write_csv(tmp/'test_per_graph.csv', per_graph)
    write_csv(tmp/'test_summary.csv', summ)
    write_csv(tmp/'paired_vs_linear.csv', paired)
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    print(f'{len(graphs)} graphs; {len(rows)} curve rows; failed draw-steps {sum(r["draws_failed"] for r in rows)}')


if __name__ == '__main__':
    main()
