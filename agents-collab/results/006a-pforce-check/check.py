"""Validate the p-force and compare common-time traces under step halving."""
import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '003-bridge-selectivity'))
import bridge_selectivity as ref
from bridge_selectivity import C, np
from pforce import make_pforce, potential

P, EPS, SEED = 1.5, 0.5, 0


def run(out):
    checks, screen, traces, differences = [], [], [], []

    def check(graph, name, error, tol):
        checks.append(dict(graph=graph, check=name, max_abs_error=error, tolerance=tol, passed=int(error <= tol)))
        if not error <= tol:
            raise ValueError(f'{graph}: {name}: {error} > {tol}')

    for family in ('clique', 'cycle'):
        spec = dict(kind='two_block', block=family, sizes=(5, 7), bridges=1, wb=0.25, seed=SEED)
        g, _, meta = C.make_graph(spec)
        graph = meta['id']
        d = C.wdeg(g)
        mean = ref.make_force(g, 'mean')
        pforce = make_pforce(g, P, EPS)
        u0 = np.random.default_rng(SEED).uniform(-1, 1, (1, g.n))
        u = u0[0]
        check(graph, 'p2_vs_mean', np.max(np.abs(make_pforce(g, 2, EPS)(u0)-mean(u0))), 1e-12)
        for p in (2.0, P):
            check(graph, f'weighted_net_force_p{p:g}', abs(make_pforce(g, p, EPS)(u) @ d), 1e-12)
        fd = 1e-5 * np.eye(g.n)
        grad = (potential(g, u+fd, P, EPS)-potential(g, u-fd, P, EPS)) / (2e-5)
        check(graph, 'degree_scaled_negative_gradient', np.max(np.abs(pforce(u)+grad/d)), 1e-8)
        small = 1e-5
        stiffness = EPS**(P-2)
        check(graph, 'small_amplitude_normalized_tangent',
              np.max(np.abs(pforce(small*u0)/small-stiffness*mean(u0))), 1e-8)
        lam = float(C.spectrum(g)[0][-1])
        for h in ref.HS:
            for dyn, number, limit in [('wave', h*h*lam*stiffness, 4),
                                       ('diffusion', h*lam*stiffness, 2)]:
                screen.append(dict(graph=graph, h=h, dynamic=dyn, lambda_max=lam, tangent_factor=stiffness,
                                   screen_value=number, upper_limit=limit, inside=int(number < limit)))
                check(graph, f'{dyn}_screen_h{h:g}', int(number >= limit), 0)
        for dyn in ('wave', 'diffusion'):
            series = {}
            for h in ref.HS:
                result = ref.integrate(pforce, u0, h, dyn, d, velocity=dyn == 'wave')
                X = result['X'][:, 0]
                U = potential(g, X, P, EPS)
                K = 0.5*(result['V'][:, 0]**2) @ d if dyn == 'wave' else np.zeros_like(U)
                E = U + K
                m = X @ d / d.sum()
                rms = np.sqrt((X-m[:, None])**2 @ d / d.sum())
                amp = np.max(np.abs(X), axis=1)
                check(graph, f'{dyn}_trace_failures_h{h:g}', int((result['first_fail'] >= 0).sum()), 0)
                if not np.isfinite(E).all():
                    raise ValueError(f'Nonfinite trace: {graph}, {dyn}, {h}')
                series[h] = E, U, K, amp, rms, X
                for k, (e, v, t, a, r, center) in enumerate(zip(E, U, K, amp, rms, m)):
                    traces.append(dict(graph=graph, dyn=dyn, h=h, time=k*ref.DT_SAVE, energy=e, potential=v,
                                       kinetic=t if dyn == 'wave' else '', max_abs=a, centered_weighted_rms=r, center=center))
            coarse, fine = (series[h] for h in (0.5, 0.25))
            for name, a, b in zip(('energy', 'potential', 'kinetic', 'max_abs', 'centered_weighted_rms', 'state'), coarse, fine):
                if name == 'kinetic' and dyn == 'diffusion':
                    continue
                diff = np.abs(a-b)
                differences.append(dict(graph=graph, dyn=dyn, quantity=name, max_abs_same_time_diff=diff.max(),
                                        rms_same_time_diff=np.sqrt(np.mean(diff**2)), final_abs_diff=diff[-1].max(),
                                        coarse_final=float(a[-1]) if a.ndim == 1 else '', fine_final=float(b[-1]) if b.ndim == 1 else '',
                                        coarse_positive_saved_increments=int((np.diff(a) > 1e-12).sum()) if name == 'energy' else '',
                                        fine_positive_saved_increments=int((np.diff(b) > 1e-12).sum()) if name == 'energy' else ''))
    for name, rows in [('checks', checks), ('step_screen', screen), ('traces', traces), ('h_halving', differences)]:
        ref.write_csv(out / f'{name}.csv', rows)
    print(f'Passed {len(checks)} checks; wrote {len(traces)} common-time trace rows.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    run(args.out)
