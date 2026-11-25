"""Smooth degree-normalized graph p-force and its matching energy."""
import numpy as np


def make_pforce(g, p=1.5, epsilon=0.5):
    """Return F for batched node states (..., n), normalized by original weighted degrees."""
    i, j, w = g.ei, g.ej, g.ew
    degree = np.bincount(i, w, minlength=g.n) + np.bincount(j, w, minlength=g.n)
    if not (np.isfinite([p, epsilon]).all() and p > 0 and epsilon > 0 and (degree > 0).all()):
        raise ValueError('Require positive finite p, epsilon and node degrees')

    def force(u):
        u = np.asarray(u, dtype=float)
        shape = u.shape
        u = u.reshape(-1, g.n)
        s = u[:, j] - u[:, i]
        flux = w * s * (s*s + epsilon*epsilon)**((p-2)/2)
        result = np.zeros_like(u)
        row = np.arange(len(u))[:, None]
        np.add.at(result, (row, i), flux)
        np.add.at(result, (row, j), -flux)
        return (result / degree).reshape(shape)

    return force


def potential(g, u, p=1.5, epsilon=0.5):
    """Weighted edge potential, evaluated with low cancellation near zero."""
    s = u[..., g.ej] - u[..., g.ei]
    return np.sum(g.ew * (epsilon**p/p) * np.expm1((p/2)*np.log1p((s/epsilon)**2)), axis=-1)
