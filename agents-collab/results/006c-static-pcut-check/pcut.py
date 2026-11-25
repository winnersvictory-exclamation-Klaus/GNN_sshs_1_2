"""Deterministic, approximate p-Rayleigh minimization and binary threshold sweep."""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize


def signed_power(x, exponent):
    return np.sign(x) * np.abs(x)**exponent


@dataclass(frozen=True)
class SweepResult:
    vector: np.ndarray
    labels: np.ndarray
    threshold: float
    rayleigh: float
    cut_objective: float
    trials: tuple


class PRayleigh:
    """Weighted, connected undirected graph with positive original degrees."""

    def __init__(self, g, p=1.5):
        self.i, self.j, self.w, self.p, self.n = g.ei, g.ej, g.ew, float(p), g.n
        self.d = np.bincount(self.i, self.w, minlength=self.n) + np.bincount(self.j, self.w, minlength=self.n)
        if not np.isfinite(p) or p <= 1 or np.any(self.d <= 0):
            raise ValueError('Require finite p>1 and positive original degrees')

    def center(self, x):
        """Return the unique minimizer of the degree-weighted p-deviation."""
        lo, hi = x.min(), x.max()
        if lo == hi:
            return float(lo)
        if self.p == 2:
            return float(x @ self.d / self.d.sum())
        width = hi-lo
        v = (x-lo)/width
        c = brentq(lambda c: self.d @ signed_power(v-c, self.p-1), 0., 1., xtol=1e-15)
        return float(lo + width*c)

    def value_gradient(self, x):
        """Evaluate R and gradient; use the envelope theorem for the minimized denominator."""
        x = np.asarray(x, dtype=float)
        y = x-self.center(x)
        diff = x[self.i]-x[self.j]
        den = self.d @ np.abs(y)**self.p
        if den <= 0 or not np.isfinite(den):
            raise ValueError('Rayleigh quotient requires a finite nonconstant vector')
        value = (self.w @ np.abs(diff)**self.p)/den
        f = self.w * signed_power(diff, self.p-1)
        grad_num = np.bincount(self.i, f, minlength=self.n) - np.bincount(self.j, f, minlength=self.n)
        grad = self.p/den * (grad_num-value*self.d*signed_power(y, self.p-1))
        return float(value), grad

    def binary_objective(self, labels):
        """Binary Rayleigh objective from the two weighted volumes."""
        side = np.asarray(labels, dtype=bool)
        a, b = self.d[side].sum(), self.d[~side].sum()
        if min(a, b) <= 0:
            raise ValueError('Require a nontrivial cut')
        small, large = sorted((a, b))
        # Binary weighted p-center: heavy side's distance is t/(1+t).
        t = (small/large)**(1/(self.p-1))
        den = small/(1+t)**(self.p-1)
        return float(self.w[side[self.i] != side[self.j]].sum()/den)

    def threshold(self, x):
        """Check each distinct observed lower boundary, excluding the maximum."""
        candidates = [(self.binary_objective(x > t), float(t)) for t in np.unique(x)[:-1]]
        value, threshold = min(candidates)
        return x > threshold, threshold, value

    def fiedler(self):
        A = np.zeros((self.n, self.n))
        np.add.at(A, (self.i, self.j), self.w)
        np.add.at(A, (self.j, self.i), self.w)
        droot = np.sqrt(self.d)
        lam, vec = np.linalg.eigh(np.eye(self.n) - A/np.outer(droot, droot))
        x = vec[:, 1]/droot
        if x[np.argmax(np.abs(x))] < 0:
            x = -x
        return float(lam[1]), x

    def sweep(self, seed=0, maxiter=800):
        """Optimize from Fiedler + 3 starts; select by continuous objective, not by labels or cut score."""
        _, fiedler = self.fiedler()
        starts = [fiedler, *np.random.default_rng(seed).standard_normal((3, self.n))]
        trials, vectors = [], []

        def normalize(v):
            x = v-self.center(v)
            return x/(self.d @ np.abs(x)**self.p)**(1/self.p)

        for k, start in enumerate(starts):
            start = normalize(start)
            fit = minimize(self.value_gradient, start, jac=True, method='BFGS',
                           options={'gtol': 1e-8, 'maxiter': maxiter})
            x = normalize(fit.x)
            value, grad = self.value_gradient(x)
            center_resid = abs(self.d @ signed_power(x, self.p-1))
            trials.append(dict(start='fiedler' if k == 0 else f'random_{k}', rayleigh=value,
                               grad_inf=float(np.linalg.norm(grad, np.inf)), center_residual=float(center_resid),
                               optimizer_success=int(fit.success), optimizer_status=int(fit.status),
                               optimizer_message=str(fit.message), iterations=int(fit.nit)))
            vectors.append(x)
        winner = min(range(len(trials)), key=lambda k: trials[k]['rayleigh'])
        for k, r in enumerate(trials):
            r['selected'] = int(k == winner)
        x = vectors[winner]
        labels, t, score = self.threshold(x)
        return SweepResult(x, labels, t, trials[winner]['rayleigh'], score, tuple(trials))


def p_rayleigh_sweep(g, p=1.5, seed=0, maxiter=800):
    """No truth labels are accepted; returns a numerical sweep, not a global certificate."""
    return PRayleigh(g, p).sweep(seed, maxiter)
