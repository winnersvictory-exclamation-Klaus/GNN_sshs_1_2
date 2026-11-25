"""Validate the p-cut sweep against independent toy-graph calculations."""
import argparse
import csv
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '003-bridge-selectivity'))
from bridge_selectivity import C, np
from scipy.optimize import minimize_scalar
from pcut import PRayleigh


def independent_binary(A, side, p):
    d = A.sum(axis=1)
    dev = lambda c: float(d @ np.abs(side.astype(float)-c)**p)
    best = minimize_scalar(dev, bounds=(0, 1), method='bounded', options={'xatol': 1e-14})
    if not best.success:
        raise ValueError(best.message)
    return float(A[np.ix_(side, ~side)].sum() / best.fun)


def run(out):
    validation, trials, cuts, vectors = [], [], [], []
    for family in ('clique', 'cycle'):
        g, _, meta = C.make_graph(dict(kind='two_block', block=family, sizes=(4, 5), bridges=1, wb=0.25, seed=0))
        graph = meta['id']
        solver = PRayleigh(g, p=1.5)
        x = np.random.default_rng(12).normal(size=g.n)
        value, grad = solver.value_gradient(x)
        step = 1e-5 * np.eye(g.n)
        fd = np.array([(solver.value_gradient(x+s)[0]-solver.value_gradient(x-s)[0])/2e-5 for s in step])
        grad_error = float(np.max(np.abs(grad-fd)))
        lin = PRayleigh(g, p=2)
        fiedler, fvec = lin.fiedler()
        fiedler_error = abs(lin.value_gradient(fvec)[0]-fiedler)
        p2_error = abs(lin.sweep(seed=0).rayleigh-fiedler)
        result = solver.sweep(seed=0)
        trials.extend(dict(graph=graph, **r) for r in result.trials)
        vectors.extend(dict(graph=graph, node=i, degree=solver.d[i], x=v, selected_side=int(lab))
                       for i, (v, lab) in enumerate(zip(result.vector, result.labels)))
        canon = lambda side: tuple(side ^ side[0])
        threshold_cuts = {canon(result.vector > t) for t in np.unique(result.vector)[:-1]}
        selected = canon(result.labels)
        graph_cuts = []
        for mask in range(1, 1 << (g.n-1)):
            side = np.array([False] + [bool(mask & (1 << bit)) for bit in range(g.n-1)])
            ref = independent_binary(g.A, side, solver.p)
            closed_form = solver.binary_objective(side)
            quotient = solver.value_gradient(side.astype(float))[0]
            graph_cuts.append(dict(graph=graph, membership=''.join(str(int(i)) for i in side),
                                   allowed_threshold=int(tuple(side) in threshold_cuts), selected=int(tuple(side) == selected),
                                   independent_value=ref, helper_binary_value=closed_form, direct_rayleigh_value=quotient,
                                   max_abs_error=max(abs(ref-closed_form), abs(ref-quotient))))
        binary_error = max(r['max_abs_error'] for r in graph_cuts)
        global_best = min(r['independent_value'] for r in graph_cuts)
        best_allowed = min(r['independent_value'] for r in graph_cuts if r['allowed_threshold'])
        [selected_row] = [r for r in graph_cuts if r['selected']]
        sweep_error = max(abs(selected_row['independent_value'] - best_allowed),
                          abs(result.cut_objective - best_allowed))
        for label, error, tol in [('gradient', grad_error, 1e-7), ('p2 Fiedler', fiedler_error, 1e-12),
                                  ('p2 sweep', p2_error, 1e-12), ('binary objective', binary_error, 1e-10),
                                  ('threshold minimum', sweep_error, 1e-10)]:
            if not error <= tol:
                raise ValueError(f'{graph}: {label}: {error} > {tol}')
        cuts.extend(graph_cuts)
        validation.append(dict(graph=graph, nodes=g.n, grad_error=grad_error, gradient_probe_value=value,
                               lambda2=fiedler, p2_fiedler_error=fiedler_error, p2_sweep_error=p2_error,
                               n_binary_cuts=len(graph_cuts), n_threshold_cuts=len(threshold_cuts),
                               binary_max_error=binary_error, threshold_error=sweep_error, selected_threshold=result.threshold,
                               p15_rayleigh=result.rayleigh, selected_binary=result.cut_objective,
                               best_allowed=best_allowed, global_binary_best=global_best,
                               regret_vs_global_binary=selected_row['independent_value']-global_best))
    for name, rows in [('validation', validation), ('optimizer_trials', trials), ('binary_cuts', cuts), ('vectors', vectors)]:
        with (out / f'{name}.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f'Passed all toy checks on {len(validation)} graphs and {len(cuts)} binary cuts.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    run(out)
