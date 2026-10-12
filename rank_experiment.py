"""순위 기반 이웃 집계(Mean/Trim/OWA/Value-weight)의 준평형·분할·자극 전달 실험.

실험방법.md 의 구현. GNN 학습 없음. 실행: python3 rank_experiment.py [--quick]
출력: results/*.csv   (필요: numpy, scipy, scikit-learn)
"""
from __future__ import annotations

import argparse
import csv
import itertools
import time
import tracemalloc
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"

# ---- 고정 설정 -------------------------------------------------------------
T = 500            # 반복 수
W, STRIDE = 50, 25  # 시간 창 길이, 이동 간격
R = 8              # 그래프당 독립 난수 초기값 수
ETA = 0.5          # 확산 갱신 계수
C = 1.0            # 파동 속도
TRIM_K = 1
OWA_GAMMA = 2.0    # 순위 가중 a_r ∝ exp(gamma*(r)/(d-1)), r=0..d-1 (위쪽 순위 가중 증가)
BETA = 1.0         # Value-weight
FAIL_AMP = 20.0    # |u|가 이 값을 넘으면 폭주(실패)로 기록
AGGS = ("mean", "trim", "owa", "vw")
DYNS = ("diff", "wave")
# 사후 평가용 설정은 calibrate()에서 개발 그래프로 정해 PARAMS에 고정한다.
PARAMS = {"eps": 0.5, "delta": 0.2, "rho": 3.0, "fc": 0.04}


# ---- 그래프 ----------------------------------------------------------------
def _connected(n, edges):
    if n == 0:
        return True
    r = np.array([e[0] for e in edges]); c = np.array([e[1] for e in edges])
    m = csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    return connected_components(m, directed=False)[0] == 1


def block_edges(kind, n, rng):
    if kind == "cycle":
        return [(i, (i + 1) % n) for i in range(n)]
    if kind == "clique":
        return list(itertools.combinations(range(n), 2))
    if kind == "sbm":  # degree-corrected: p_ij = p*theta_i*theta_j, theta 로그정규
        for _ in range(200):
            th = rng.lognormal(0.0, 0.6, n); th /= th.mean()
            p = 0.3
            ed = [(i, j) for i in range(n) for j in range(i + 1, n)
                  if rng.random() < min(1.0, p * th[i] * th[j])]
            if ed and _connected(n, ed):
                return ed
        raise RuntimeError("sbm block not connected")
    raise ValueError(kind)


def two_block(kind, sizes, bridges, wb, seed):
    """두 부분(정답 labels)과 bridges개의 연결 간선(가중치 wb). 반환: n, edges[(i,j,w)], labels."""
    rng = np.random.default_rng(seed)
    n0, n1 = sizes
    e = [(i, j, 1.0) for i, j in block_edges(kind, n0, rng)]
    e += [(n0 + i, n0 + j, 1.0) for i, j in block_edges(kind, n1, rng)]
    if bridges:
        a = rng.choice(n0, bridges, replace=False)
        b = n0 + rng.choice(n1, bridges, replace=False)
        e += [(int(i), int(j), wb) for i, j in zip(a, b)]
    return n0 + n1, e, np.array([0] * n0 + [1] * n1)


def er_graph(n, deg, seed):
    rng = np.random.default_rng(seed)
    for _ in range(500):
        ed = [(i, j, 1.0) for i in range(n) for j in range(i + 1, n) if rng.random() < deg / (n - 1)]
        if ed and _connected(n, ed):
            return n, ed
    raise RuntimeError("er not connected")


def shuffled(n, edges, seed):
    """차수 보존 이중 간선 교환으로 부분 구조를 파괴한 그래프."""
    rng = np.random.default_rng(seed)
    E = [(min(i, j), max(i, j)) for i, j, _ in edges]
    S = set(E)
    for _ in range(20 * len(E)):
        a, b = rng.choice(len(E), 2, replace=False)
        (p, q), (r, s) = E[a], E[b]
        if rng.random() < 0.5:
            r, s = s, r
        n1, n2 = (min(p, s), max(p, s)), (min(r, q), max(r, q))
        if p == s or r == q or n1 in S or n2 in S or n1 == n2:
            continue
        S.discard(E[a]); S.discard(E[b]); S.add(n1); S.add(n2)
        E[a], E[b] = n1, n2
    ed = [(i, j, 1.0) for i, j in E]
    return n, ed


class G:
    """이웃을 패딩한 배열 표현."""

    def __init__(self, n, edges):
        self.n = n
        self.edges = edges
        self.ei = np.array([e[0] for e in edges]); self.ej = np.array([e[1] for e in edges])
        self.ew = np.array([e[2] for e in edges], dtype=float)
        nb = [[] for _ in range(n)]
        for i, j, w in edges:
            nb[i].append((j, w)); nb[j].append((i, w))
        self.deg = np.array([len(x) for x in nb])
        self.dmax = int(self.deg.max())
        self.NB = np.zeros((n, self.dmax), dtype=int)
        self.WT = np.zeros((n, self.dmax))
        for i, x in enumerate(nb):
            for r, (j, w) in enumerate(x):
                self.NB[i, r] = j; self.WT[i, r] = w
        self.valid = self.WT > 0
        rank = np.arange(self.dmax)[None, :]
        d = self.deg[:, None]
        k = np.minimum(TRIM_K, (self.deg - 1) // 2)[:, None]
        self.trim_mask = ((rank >= k) & (rank < d - k)).astype(float)
        self.owa_w = np.where(self.valid, np.exp(OWA_GAMMA * rank / np.maximum(d - 1, 1)), 0.0)
        A = np.zeros((n, n))
        for i, j, w in edges:
            A[i, j] += w; A[j, i] += w
        self.A = A


def aggregate(g: G, u: np.ndarray, kind: str) -> np.ndarray:
    """u: (R,n) -> A_i(u) (R,n). 가중 간선은 순위 마스크와 곱해 가중평균."""
    v = u[:, g.NB]                                   # (R,n,dmax)
    if kind == "mean":
        return (g.WT * v).sum(-1) / g.WT.sum(-1)
    if kind == "vw":
        vm = np.where(g.valid, v, -np.inf)
        e = np.where(g.valid, np.exp(BETA * (vm - vm.max(-1, keepdims=True))), 0.0) * g.WT
        return (e * np.where(g.valid, v, 0.0)).sum(-1) / e.sum(-1)
    vs = np.where(g.valid, v, np.inf)
    order = np.argsort(vs, axis=-1, kind="stable")   # 동률은 입력 순서 고정
    sv = np.take_along_axis(vs, order, -1)
    sw = np.take_along_axis(np.broadcast_to(g.WT, v.shape), order, -1)
    sv = np.where(sw > 0, sv, 0.0)
    m = g.trim_mask if kind == "trim" else g.owa_w
    coef = m * sw
    return (coef * sv).sum(-1) / coef.sum(-1)


def simulate(g: G, agg: str, dyn: str, u0: np.ndarray, steps=T, c=C, eta=ETA):
    """반환 traj (steps+1,R,n), failed (R,)."""
    Rn, n = u0.shape
    X = np.empty((steps + 1, Rn, n))
    X[0] = u0
    prev, cur = u0.copy(), u0.copy()
    failed = np.zeros(Rn, bool)
    with np.errstate(all="ignore"):
        for t in range(steps):
            f = aggregate(g, cur, agg) - cur
            nxt = cur + eta * f if dyn == "diff" else 2 * cur - prev + c * c * f
            bad = ~(np.abs(nxt) < FAIL_AMP).all(axis=1)
            failed |= bad
            nxt[failed] = 0.0
            cur_old = cur
            prev, cur = (cur_old, nxt)
            X[t + 1] = cur
    return X, failed


def lowpass(X, fc):
    F = np.fft.rfft(X, axis=0)
    f = np.fft.rfftfreq(X.shape[0])
    F[f > fc] = 0
    return np.fft.irfft(F, n=X.shape[0], axis=0)


# ---- 준평형 측정 -----------------------------------------------------------
def window_stats(X, g: G, labels):
    """창별 (D_in(최대 부분), D_out, 전체 RMS). 각각 (R,nw)."""
    d2 = (X[:, :, g.ei] - X[:, :, g.ej]) ** 2                       # (T+1,R,m)
    cs = np.concatenate([np.zeros_like(d2[:1]), np.cumsum(d2, 0)], 0)
    ns = np.concatenate([np.zeros((1,) + X.shape[1:2]), np.cumsum((X ** 2).sum(-1), 0)], 0)
    starts = np.arange(0, X.shape[0] - W + 1, STRIDE)
    win = cs[starts + W] - cs[starts]                               # (nw,R,m)
    rms = np.sqrt((ns[starts + W] - ns[starts]) / (W * g.n)) + 1e-12  # (nw,R)
    li, lj = labels[g.ei], labels[g.ej]
    inn = li == lj
    din = np.zeros((len(starts), X.shape[1]))
    for p in np.unique(labels):
        sel = inn & (li == p)
        if sel.any():
            din = np.maximum(din, np.sqrt(win[:, :, sel].mean(-1) / W) / rms)
    out = ~inn
    dout = np.sqrt(win[:, :, out].mean(-1) / W) / rms if out.any() else np.zeros_like(din)
    return din.T, dout.T, rms.T


def hold_steps(stats, eps, delta, rho, floor=1e-3):
    din, dout, rms = stats
    ok = (din <= eps) & (dout >= delta) & (dout / (din + 1e-12) >= rho) & (rms >= floor)
    res = np.zeros(ok.shape[0])
    for r in range(ok.shape[0]):
        best = run = 0
        for v in ok[r]:
            run = run + 1 if v else 0
            best = max(best, run)
        res[r] = (best + 1) * STRIDE if best else 0
    return res


# ---- 분할 ------------------------------------------------------------------
def edge_scores(X, g: G):
    d = X[T // 2:, :, g.ei] - X[T // 2:, :, g.ej]
    return np.sqrt((d ** 2).mean(0))                                # (R,m)


def cut_partition(g: G, score, q=2):
    """간선 점수가 큰 것부터 제거해 q개 연결 성분을 얻는 것과 같은 단일연결(Kruskal) 절단."""
    order = np.argsort(score, kind="stable")
    parent = list(range(g.n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    comps = g.n
    for e in order:
        if comps == q:
            break
        a, b = find(g.ei[e]), find(g.ej[e])
        if a != b:
            parent[a] = b; comps -= 1
    if comps != q:
        return None
    roots = [find(i) for i in range(g.n)]
    _, lab = np.unique(roots, return_inverse=True)
    return lab


def conductance(g: G, lab):
    cut = g.ew[lab[g.ei] != lab[g.ej]].sum()
    vol = [g.A[lab == p].sum() for p in np.unique(lab)]
    return float(cut / min(vol)) if min(vol) > 0 else float("nan")


def split_1d(v):
    """1차원 2-means(정렬 후 SSE 최소 분할)."""
    o = np.argsort(v); s = v[o]
    best, bi = np.inf, 1
    for i in range(1, len(s)):
        sse = ((s[:i] - s[:i].mean()) ** 2).sum() + ((s[i:] - s[i:].mean()) ** 2).sum()
        if sse < best:
            best, bi = sse, i
    lab = np.zeros(len(v), int); lab[o[bi:]] = 1
    return lab


def make_metrics(g, lab, truth):
    if lab is None:
        return dict(ari=np.nan, nmi=np.nan, cond=np.nan, minfrac=np.nan, ok=0)
    sizes = np.bincount(lab) / g.n
    out = dict(cond=conductance(g, lab), minfrac=float(sizes.min()), ok=1)
    if truth is not None:
        out.update(ari=adjusted_rand_score(truth, lab), nmi=normalized_mutual_info_score(truth, lab))
    else:
        out.update(ari=np.nan, nmi=np.nan)
    return out


# ---- 기준선(단순화 구현) ---------------------------------------------------
def spectral_labels(g: G):
    d = g.A.sum(1)
    Ls = np.eye(g.n) - g.A / np.sqrt(np.outer(d, d))
    w, V = np.linalg.eigh(Ls)
    return split_1d(V[:, 1] / np.sqrt(d))


def _align(v):
    M = np.stack([v.real, v.imag], 1)
    u, s, vt = np.linalg.svd(M, full_matrices=False)
    return u[:, 0]


def wave_fft_labels(g: G, rng):
    """Sahai 2009 단순화: 평균 파동의 시간 FFT에서 가장 낮은 지배 주파수 성분의 위상 정렬 크기."""
    u0 = rng.uniform(-1, 1, (1, g.n))
    X, _ = simulate(g, "mean", "wave", u0)
    F = np.fft.rfft(X[:, 0, :] - X[:, 0, :].mean(0), axis=0)
    P = (np.abs(F) ** 2).sum(1)
    P[:2] = 0
    peaks = [k for k in range(2, len(P) - 1) if P[k] >= P[k - 1] and P[k] >= P[k + 1] and P[k] > 0.1 * P.max()]
    k = peaks[0] if peaks else int(np.argmax(P))
    return split_1d(_align(F[k]))


def dmd_labels(g: G, rng, rank=40):
    """Zhu 2022 단순화: 시간지연 스택 [u(t);u(t-1)]에 exact DMD, 가장 낮은 0이 아닌 진동수 모드."""
    u0 = rng.uniform(-1, 1, (1, g.n))
    X, _ = simulate(g, "mean", "wave", u0, steps=300)
    Z = X[:, 0, :]
    S = np.concatenate([Z[1:], Z[:-1]], 1).T                          # (2n, 300)
    X1, X2 = S[:, :-1], S[:, 1:]
    U, s, Vt = np.linalg.svd(X1, full_matrices=False)
    r = min(rank, int((s > 1e-8 * s[0]).sum()))
    U, s, Vt = U[:, :r], s[:r], Vt[:r]
    At = U.T @ X2 @ Vt.T / s
    lam, Wv = np.linalg.eig(At)
    modes = X2 @ Vt.T / s @ Wv
    ang = np.abs(np.angle(lam))
    cand = np.where(ang > 1e-3)[0]
    k = cand[np.argmin(ang[cand])]
    return split_1d(_align(modes[:g.n, k]))


def becchetti_labels(g: G, rng, steps=30):
    """Becchetti et al. 단순화: ±1 초기값의 게으른 평균 동역학 후 sign(x_t - x_{t-1})."""
    d = g.A.sum(1)
    P = 0.5 * (np.eye(g.n) + g.A / d[:, None])
    x = rng.choice([-1.0, 1.0], g.n)
    for _ in range(steps - 1):
        x = P @ x
    diff = P @ x - x
    return (diff > np.median(diff)).astype(int)


# ---- 한 그래프에 대한 전체 평가 -------------------------------------------
def prepare(spec):
    kind = spec["kind"]
    if kind in ("cycle", "clique", "sbm"):
        n, e, lab = two_block(kind, spec["sizes"], spec["bridges"], spec["wb"], spec["seed"])
    elif kind == "er":
        n, e = er_graph(sum(spec["sizes"]), 6, spec["seed"]); lab = None
    elif kind == "shuffled":
        n0, e0, lab0 = two_block("sbm", spec["sizes"], 4, 1.0, spec["seed"])
        n, e = shuffled(n0, e0, spec["seed"] + 7); lab = None
    return G(n, e), lab


def graph_row(spec, g):
    return dict(kind=spec["kind"], sizes="x".join(map(str, spec["sizes"])), bridges=spec["bridges"],
                wb=spec["wb"], seed=spec["seed"], n=g.n, m=len(g.edges), role=spec["role"])


def run_graph(args):
    spec, params = args
    g, truth = prepare(spec)
    rng = np.random.default_rng(spec["seed"] * 1000 + 17)
    u0 = rng.uniform(-1, 1, (R, g.n))
    rows = []
    labels_for_hold = truth
    for agg in AGGS:
        for dyn in DYNS:
            X, failed = simulate(g, agg, dyn, u0)
            variants = [("diff", X)] if dyn == "diff" else [("wave_raw", X), ("wave_lp", lowpass(X, params["fc"]))]
            for vname, Xv in variants:
                good = ~failed
                base = dict(graph_row(spec, g), agg=agg, dyn=vname, fail_frac=float(failed.mean()))
                if good.sum() == 0:
                    rows.append(dict(base, ok=0, ari=np.nan, nmi=np.nan, cond=np.nan, minfrac=np.nan,
                                     hold_true=np.nan, hold_rec=np.nan, hold_true_pos=np.nan))
                    continue
                Xg = Xv[:, good]
                sc = np.median(edge_scores(Xg, g), axis=0)
                lab = cut_partition(g, sc, 2)
                met = make_metrics(g, lab, truth)
                row = dict(base, **met)
                for tag, L in (("true", truth), ("rec", lab)):
                    if L is None or len(np.unique(L)) < 2:
                        row[f"hold_{tag}"] = np.nan; row[f"hold_{tag}_pos"] = np.nan
                        continue
                    h = hold_steps(window_stats(Xg, g, L), params["eps"], params["delta"], params["rho"])
                    row[f"hold_{tag}"] = float(h.mean()); row[f"hold_{tag}_pos"] = float((h > 0).mean())
                rows.append(row)
    # 기준선 (참 분할이 있는 그래프만)
    if truth is not None:
        for name, fn in (("spectral", lambda: spectral_labels(g)),
                         ("wave_fft_2009_simplified", lambda: wave_fft_labels(g, rng)),
                         ("dmd_2022_simplified", lambda: dmd_labels(g, rng)),
                         ("becchetti_simplified", lambda: becchetti_labels(g, rng))):
            try:
                lab = fn(); met = make_metrics(g, lab, truth)
            except Exception as ex:  # 실패는 기록
                met = dict(ari=np.nan, nmi=np.nan, cond=np.nan, minfrac=np.nan, ok=0)
            rows.append(dict(graph_row(spec, g), agg=name, dyn="baseline", fail_frac=0.0, **met))
    return rows


def run_stimulus(args):
    spec, a, params = args
    g, truth = prepare(spec)
    n0 = spec["sizes"][0]
    rng = np.random.default_rng(spec["seed"] * 77 + int(a * 10))
    src = rng.choice(n0, R, replace=True)
    sigma = 0.05
    noise = rng.normal(0, sigma, (R, g.n))
    u_s = noise.copy(); u_s[np.arange(R), src] += a
    far = np.arange(g.n) >= n0
    rows = []
    for agg in AGGS:
        for dyn in DYNS:
            Xs, f1 = simulate(g, agg, dyn, u_s, steps=200)
            Xn, f2 = simulate(g, agg, dyn, noise, steps=200)
            ok = ~(f1 | f2)
            if ok.sum() == 0:
                continue
            delta = np.abs(Xs - Xn)[:, ok][:, :, far]                     # (t,R,far)
            peak_t = delta.max(-1)                                        # (t,R)
            peak = peak_t.max(0)
            nz = np.sqrt((Xn[:, ok][:, :, far] ** 2).mean((0, 2)))        # 잡음 단독 반응 RMS
            thr = np.maximum(3 * nz, 1e-3)
            arr = np.array([np.argmax(peak_t[:, r] > thr[r]) if (peak_t[:, r] > thr[r]).any() else np.nan
                            for r in range(peak_t.shape[1])])
            near_peak = np.abs(Xs - Xn)[:, ok][:, :, ~far].max((0, 2))
            rows.append(dict(graph_row(spec, g), agg=agg, dyn=dyn, amp=a, fail_frac=float(1 - ok.mean()),
                             far_peak=float(peak.mean()), near_peak=float(near_peak.mean()),
                             transmit_ratio=float((peak / np.maximum(near_peak, 1e-12)).mean()),
                             arrival=float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan,
                             arrived_frac=float(np.isfinite(arr).mean()),
                             snr=float((peak / np.maximum(nz, 1e-12)).mean()),
                             noise_far_rms_over_sigma=float((nz / sigma).mean())))
    return rows


def run_stab(args):
    spec, c, eta = args
    g, _ = prepare(spec)
    rng = np.random.default_rng(spec["seed"])
    u0 = rng.uniform(-1, 1, (R, g.n))
    rows = []
    for agg in AGGS:
        for dyn, par in (("wave", c), ("diff", eta)):
            _, failed = simulate(g, agg, dyn, u0, c=c, eta=eta)
            rows.append(dict(kind=spec["kind"], seed=spec["seed"], agg=agg, dyn=dyn,
                             param=par, fail_frac=float(failed.mean())))
    return rows


# ---- 보정, 전체 실행 ------------------------------------------------------
def dev_specs():
    sp = []
    for kind in ("cycle", "clique", "sbm"):
        for b in (1, 2):
            for s in range(3):
                sp.append(dict(kind=kind, sizes=(24, 24), bridges=b, wb=1.0, seed=s, role="dev_pos"))
    for kind in ("er", "shuffled"):
        for s in range(3):
            sp.append(dict(kind=kind, sizes=(24, 24), bridges=0, wb=1.0, seed=10 + s, role="dev_neg"))
    return sp


def dev_stats(args):
    spec, fcs = args
    g, truth = prepare(spec)
    rng = np.random.default_rng(spec["seed"] * 1000 + 17)
    u0 = rng.uniform(-1, 1, (R, g.n))
    out = []
    for agg in AGGS:
        for dyn in DYNS:
            X, failed = simulate(g, agg, dyn, u0)
            good = ~failed
            if good.sum() == 0:
                continue
            variants = [("diff", 0.0, X)] if dyn == "diff" else \
                [("wave_raw", 0.0, X)] + [("wave_lp", fc, lowpass(X, fc)) for fc in fcs]
            for vname, fc, Xv in variants:
                Xg = Xv[:, good]
                sc = np.median(edge_scores(Xg, g), 0)
                lab = cut_partition(g, sc, 2)
                L = truth if truth is not None else lab
                if L is None or len(np.unique(L)) < 2:
                    continue
                out.append(dict(role=spec["role"], agg=agg, dyn=vname, fc=fc, stats=window_stats(Xg, g, L)))
    return out


def calibrate(pool, fcs=(0.02, 0.04, 0.08)):
    res = pool.map(dev_stats, [(s, fcs) for s in dev_specs()])
    items = [x for r in res for x in r]
    best, table = None, []
    for fc in fcs:
        sel = [x for x in items if x["dyn"] != "wave_lp" or x["fc"] == fc]
        for eps, delta, rho in itertools.product((0.3, 0.5, 0.7, 1.0), (0.1, 0.2, 0.3), (2.0, 3.0, 5.0)):
            pos, neg = [], []
            for x in sel:
                h = hold_steps(x["stats"], eps, delta, rho)
                (pos if x["role"] == "dev_pos" else neg).append(h)
            posm = float(np.concatenate(pos).mean())
            fp = float((np.concatenate(neg) > 0).mean())
            table.append(dict(fc=fc, eps=eps, delta=delta, rho=rho, pos_hold=posm, neg_false_pos=fp))
            if fp <= 0.05 and (best is None or posm > best[0]):
                best = (posm, dict(eps=eps, delta=delta, rho=rho, fc=fc))
    write_csv(OUT / "calibration.csv", table)
    return best[1] if best else dict(PARAMS), table


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def eval_specs(n_seeds):
    sp = []
    for kind in ("cycle", "clique", "sbm"):
        for sizes in ((24, 24), (16, 32)):
            for b in (1, 2, 4, 8):
                for wb in (1.0, 0.25):
                    for s in range(n_seeds):
                        sp.append(dict(kind=kind, sizes=sizes, bridges=b, wb=wb, seed=100 + s, role="eval_pos"))
    for kind in ("cycle", "clique", "sbm"):                 # 양성 대조군: 끊어진 두 부분
        for s in range(n_seeds):
            sp.append(dict(kind=kind, sizes=(24, 24), bridges=0, wb=1.0, seed=100 + s, role="eval_disc"))
    for kind in ("er", "shuffled"):                         # 음성 대조군
        for s in range(n_seeds):
            sp.append(dict(kind=kind, sizes=(24, 24), bridges=0, wb=1.0, seed=100 + s, role="eval_neg"))
    return sp


def timing():
    rows = []
    for n in (256, 1024, 4096):
        rng = np.random.default_rng(0)
        h = n // 2
        e = [(i, (i + 1) % h, 1.0) for i in range(h)] + [(h + i, h + (i + 1) % h, 1.0) for i in range(h)]
        for _ in range(n // 2):                              # 임의의 지름길로 평균 차수 약 4
            i, j = rng.integers(0, h, 2)
            if i != j:
                e.append((int(i), int(j), 1.0)); e.append((int(h + i), int(h + j), 1.0))
        g = G(n, e)
        u0 = rng.uniform(-1, 1, (1, n))
        for agg in AGGS:
            tracemalloc.start()
            t0 = time.perf_counter()
            simulate(g, agg, "wave", u0, steps=50)
            dt = (time.perf_counter() - t0) / 50
            peak = tracemalloc.get_traced_memory()[1] / 1e6
            tracemalloc.stop()
            rows.append(dict(n=n, m=len(e), agg=agg, sec_per_step=dt, peak_mb=peak))
    write_csv(OUT / "timing.csv", rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    n_seeds = 2 if a.quick else 8
    t0 = time.time()
    with Pool() as pool:
        params, _ = calibrate(pool)
        PARAMS.update(params)
        print("calibrated (dev only):", PARAMS, f"[{time.time()-t0:.0f}s]", flush=True)
        specs = eval_specs(n_seeds)
        res = pool.map(run_graph, [(s, dict(PARAMS)) for s in specs], chunksize=2)
        write_csv(OUT / "partition_quasi.csv", [r for x in res for r in x])
        print("partition done", f"[{time.time()-t0:.0f}s]", flush=True)
        stim = [(s, amp, dict(PARAMS)) for s in specs if s["role"] == "eval_pos" and s["bridges"] in (1, 4)
                and s["wb"] == 1.0 and s["seed"] < 100 + max(2, n_seeds // 2) for amp in (1.0, 5.0)]
        res = pool.map(run_stimulus, stim, chunksize=2)
        write_csv(OUT / "stimulus.csv", [r for x in res for r in x])
        print("stimulus done", f"[{time.time()-t0:.0f}s]", flush=True)
        stab = [(dict(kind=k, sizes=(24, 24), bridges=2, wb=1.0, seed=s, role="stab"), c, eta)
                for k in ("cycle", "clique", "sbm") for s in range(3)
                for c, eta in ((0.5, 0.5), (1.0, 1.0), (1.4, 1.5), (1.8, 1.9))]
        res = pool.map(run_stab, stab)
        write_csv(OUT / "stability.csv", [r for x in res for r in x])
    timing()
    (OUT / "params.txt").write_text(repr(PARAMS))
    print("all done", f"[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
