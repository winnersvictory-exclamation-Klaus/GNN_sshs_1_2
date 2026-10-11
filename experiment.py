"""Small deterministic probe of wave response versus linear GCN influence.

Run: python3 experiment.py
Requires: numpy. This is a diagnostic toy, not a trained GNN or DMD replication.
"""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
N = 16
K = 6
C = 1.0


def graph(bridges: int) -> np.ndarray:
    """Two 8-cycles joined by one or four cross edges."""
    A = np.zeros((N, N), dtype=float)
    for offset in (0, 8):
        for i in range(8):
            u, v = offset + i, offset + (i + 1) % 8
            A[u, v] = A[v, u] = 1
    for i in range(bridges):
        u = (2 * i) % 8
        v = 8 + (2 * i) % 8
        A[u, v] = A[v, u] = 1
    return A


def distances(A: np.ndarray) -> np.ndarray:
    D = np.full(A.shape, 10**6, dtype=int)
    np.fill_diagonal(D, 0)
    D[A > 0] = 1
    for k in range(len(A)):
        D = np.minimum(D, D[:, [k]] + D[[k], :])
    return D


def midranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    result = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        result[order[i:j]] = (i + j - 1) / 2
        i = j
    return result


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    a, b = midranks(x), midranks(y)
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def score(A: np.ndarray):
    # P^K[t,s] is the exact sensitivity of K linear mean-aggregation layers
    # at target t to a unit input perturbation at source s.
    P = A + np.eye(N)
    P /= P.sum(axis=1, keepdims=True)
    sensitivity = np.linalg.matrix_power(P, K)

    # Wave response: simultaneously place a unit impulse at each source.
    Lrw = np.eye(N) - A / A.sum(axis=1, keepdims=True)
    prev = np.eye(N)
    cur = prev.copy()
    energy = np.zeros((N, N))
    for _ in range(K):
        nxt = (2 * np.eye(N) - C * C * Lrw) @ cur - prev
        energy += nxt * nxt
        prev, cur = cur, nxt
    energy = np.sqrt(energy / K)

    # Established structural comparator: effective resistance.
    L = np.diag(A.sum(axis=1)) - A
    Lplus = np.linalg.pinv(L, hermitian=True)
    diag = np.diag(Lplus)
    R = diag[:, None] + diag[None, :] - 2 * Lplus
    return sensitivity, energy, R, distances(A)


def main():
    rows = []
    overview = []
    for bridges in (1, 4):
        A = graph(bridges)
        sensitivity, energy, resistance, D = score(A)
        name = f"two_cycles_{bridges}_bridge{'s' if bridges > 1 else ''}"
        for target in range(N):
            for source in range(N):
                if target == source or D[target, source] > K:
                    continue
                rows.append({
                    "graph": name, "source": source, "target": target,
                    "distance": int(D[target, source]),
                    "cross_ring": int((target < 8) != (source < 8)),
                    "linear_gcn_sensitivity": float(sensitivity[target, source]),
                    "wave_rms_K6": float(energy[target, source]),
                    "effective_resistance": float(resistance[target, source]),
                })
        group = [r for r in rows if r["graph"] == name and r["distance"] >= 2]
        # Compare pairs within each graph distance; short pairs cannot drive
        # the apparent relationship for longer pairs.
        for d in sorted({r["distance"] for r in group}):
            part = [r for r in group if r["distance"] == d]
            overview.append({
                "graph": name, "distance": d, "pairs": len(part),
                "rho_wave_vs_sensitivity": spearman(
                    np.array([r["wave_rms_K6"] for r in part]),
                    np.array([r["linear_gcn_sensitivity"] for r in part])),
                "rho_neg_resistance_vs_sensitivity": spearman(
                    np.array([-r["effective_resistance"] for r in part]),
                    np.array([r["linear_gcn_sensitivity"] for r in part])),
            })
    for filename, data in (("pair_results.csv", rows), ("distance_summary.csv", overview)):
        with (ROOT / filename).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    print("Created pair_results.csv and distance_summary.csv")
    for r in overview:
        print(r)


if __name__ == "__main__":
    main()
