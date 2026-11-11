"""파동 갱신식의 발산 빈도 테스트: (시간 스킴) × (힘 함수) 조합.

../GNN_sshs_1_2/rank_experiment.py 의 그래프 생성·집계를 읽기 전용으로 가져온다.
한 단계 갱신은 두 부분으로 나뉜다.

    F   = force(ctx, u)                       # 아이디어 3: 평균 대체. A(u) - u 에 해당, (R,n)
    u+  = scheme(ctx, cur, prev, F, Fp)       # 아이디어 1·2: 시간 적분. Fp = 직전 단계의 F

새 함수 추가:
    @force("이름", 파라미터=기본값)
    def _(ctx, u): return F

    @scheme("이름", 파라미터=기본값)
    def _(ctx, cur, prev, F, Fp): return 다음_상태

ctx.pf / ctx.ps: 힘 / 스킴 파라미터, ctx.state: 시뮬레이션 동안 유지되는 저장소,
ctx.u0: 초기값, ctx.t: 단계, ctx.c: 파동 속도, ctx.g: 그래프(rank_experiment.G).

실행 예:
    python3 wave_test.py                                   # 모든 스킴 × 모든 힘
    python3 wave_test.py -s wave damp -f owa huber --c 0.5 1.0
    python3 wave_test.py -s damp -p damp.k=0.3 huber.tau=0.2
    python3 wave_test.py --list
출력: 표(stdout) + results/wave_test.csv (--out 으로 변경)
CSV 열 (한 행 = 그래프 1개 × R개 초기값, 비율 외 값은 폭주하지 않은 실현의 중앙값):
    fail_frac      |u| > FAIL_AMP 또는 NaN 이 된 실현 비율
    mean_fail_step 폭주한 실현의 평균 폭주 단계
    rms_final      마지막 W 단계 RMS / 초기 RMS (DC 포함 — 전체 표류를 봄)
    ac_final       같은 값에서 노드 평균을 뺀 것 — 0 이면 합의 상태로 수렴(파동 소멸)
    rms_peak       전 구간 최대 RMS / 초기 RMS
    dead_frac      폭주는 안 했지만 ac_final < 0.05 인 실현 비율
    ari            후반부 파형으로 q=2 절단한 분할의 ARI (정답이 있는 그래프만)
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("GNN_REPO", HERE.parent / "GNN_sshs_1_2"))
sys.path.insert(0, str(REPO))
import rank_experiment as rx  # noqa: E402

FAIL_AMP = rx.FAIL_AMP
EPS = 1e-12


@dataclass
class Ctx:
    g: rx.G
    c: float
    pf: dict
    ps: dict
    u0: np.ndarray
    t: int = 0
    state: dict = field(default_factory=dict)


FORCES: dict[str, tuple] = {}
SCHEMES: dict[str, tuple] = {}


def _register(table, name, defaults):
    def deco(fn):
        table[name] = (fn, defaults, (fn.__doc__ or "").strip())
        return fn
    return deco


def force(name, **defaults):
    """힘 함수 등록 데코레이터. defaults 는 -p name.key=value 로 덮어쓸 수 있다."""
    return _register(FORCES, name, defaults)


def scheme(name, **defaults):
    """시간 스킴 등록 데코레이터."""
    return _register(SCHEMES, name, defaults)


# ---- 공용 도우미 -----------------------------------------------------------
def rms_rows(a):
    """실현(R)별 RMS, (R,...) -> (R,1)."""
    return np.sqrt((a.reshape(a.shape[0], -1) ** 2).mean(1, keepdims=True))


def diffs(g, u):
    """x_ij = u_j - u_i, 패딩 형태 (R,n,dmax). 패딩 칸은 0."""
    return np.where(g.valid, u[:, g.NB] - u[:, :, None], 0.0)


def edge_mean(g, vals):
    """(1/d_i) Σ_j w_ij vals_ij. d_i = Σ_j w_ij 는 u 와 무관 (형태 A: 에지 퍼텐셜의 기울기)."""
    return (g.WT * vals).sum(-1) / g.WT.sum(-1)


def edge_values(g, a):
    """패딩 배열 (R,n,dmax) 에서 실제 간선 값만 (R, 2m) 로. 각 간선이 양방향으로 두 번 들어간다."""
    return a[:, g.valid]


def sorted_nb(g, x):
    """이웃 차이를 노드별로 오름차순 정렬. 반환 (정렬값, 정렬된 간선 가중치); 패딩은 값 0, 가중치 0."""
    xs = np.where(g.valid, x, np.inf)
    order = np.argsort(xs, axis=-1, kind="stable")
    sx = np.take_along_axis(xs, order, -1)
    sw = np.take_along_axis(np.broadcast_to(g.WT, x.shape), order, -1)
    return np.where(sw > 0, sx, 0.0), sw


def nb_median(g, x):
    """노드별 이웃 값의 중앙값 (가중치 무시). (R,n,dmax) -> (R,n)."""
    sx, _ = sorted_nb(g, x)
    lo = np.broadcast_to(((g.deg - 1) // 2)[None, :, None], sx.shape[:2] + (1,))
    hi = np.broadcast_to((g.deg // 2)[None, :, None], sx.shape[:2] + (1,))
    return 0.5 * (np.take_along_axis(sx, lo, -1) + np.take_along_axis(sx, hi, -1))[..., 0]


def huber_force(g, x, tau):
    return edge_mean(g, np.clip(x, -tau, tau))


def amp_force(g, x, tau, beta):
    return edge_mean(g, x + beta * (x - tau * np.tanh(x / tau)))


# ---- 힘 함수 (아이디어 3) --------------------------------------------------
def _repo_agg(name):
    def fn(ctx, u):
        return rx.aggregate(ctx.g, u, name) - u
    fn.__doc__ = f"repo 집계 {name}: A(u) - u."
    return fn


for _n in rx.AGGS:
    force(_n)(_repo_agg(_n))


@force("odd_owa")
def _(ctx, u):
    """OWA 기함수화: (F(u) - F(-u))/2."""
    return 0.5 * ((rx.aggregate(ctx.g, u, "owa") - u) - (rx.aggregate(ctx.g, -u, "owa") + u))


@force("odd_vw")
def _(ctx, u):
    """Value-weight 기함수화: (F(u) - F(-u))/2."""
    return 0.5 * ((rx.aggregate(ctx.g, u, "vw") - u) - (rx.aggregate(ctx.g, -u, "vw") + u))


@force("huber", tau=0.5)
def _(ctx, u):
    """A1: (1/d_i) Σ w clip(x, -tau, tau)."""
    return huber_force(ctx.g, diffs(ctx.g, u), ctx.pf["tau"])


@force("tanh", tau=0.5)
def _(ctx, u):
    """A2: (1/d_i) Σ w tau tanh(x/tau)."""
    return edge_mean(ctx.g, ctx.pf["tau"] * np.tanh(diffs(ctx.g, u) / ctx.pf["tau"]))


@force("amp", tau=0.5, beta=1.0)
def _(ctx, u):
    """A3: ψ = x + beta(x - tau tanh(x/tau)), 기울기 ∈ [1, 1+beta]. 큰 차이 증폭."""
    return amp_force(ctx.g, diffs(ctx.g, u), ctx.pf["tau"], ctx.pf["beta"])


@force("gauss", sigma=0.5)
def _(ctx, u):
    """B1: Σ w e^{-x²/σ²} x / Σ w e^{-x²/σ²} (남은 가중치로 정규화)."""
    x = diffs(ctx.g, u)
    w = ctx.g.WT * np.exp(-(x / ctx.pf["sigma"]) ** 2)
    return (w * x).sum(-1) / (w.sum(-1) + EPS)


@force("absvw", beta=2.0)
def _(ctx, u):
    """B2: Σ e^{beta|x|} x / Σ e^{beta|x|} (기함수 Value-weight)."""
    x = diffs(ctx.g, u)
    ax = np.where(ctx.g.valid, np.abs(x), -np.inf)
    w = np.where(ctx.g.valid, np.exp(ctx.pf["beta"] * (ax - ax.max(-1, keepdims=True))), 0.0) * ctx.g.WT
    return (w * x).sum(-1) / w.sum(-1)


def _sym_owa(ctx, u, shape):
    """정렬된 차이에 순위 대칭 가중 a_r = a_{d+1-r}. z_r ∈ [-1,1] 은 순위의 중심 대비 위치."""
    g = ctx.g
    key = ("symowa", shape, ctx.pf["gamma"])
    if key not in ctx.state:
        r = np.arange(g.dmax)[None, :]
        d = g.deg[:, None]
        z = np.where(d > 1, (2 * r - (d - 1)) / np.maximum(d - 1, 1), 0.0)
        a = np.cosh(ctx.pf["gamma"] * z) if shape == "ext" else np.exp(-ctx.pf["gamma"] * z ** 2)
        ctx.state[key] = np.where(r < d, a, 0.0)
    sx, sw = sorted_nb(g, diffs(g, u))
    coef = ctx.state[key] * sw
    return (coef * sx).sum(-1) / coef.sum(-1)


@force("symowa_ext", gamma=2.0)
def _(ctx, u):
    """B3: 대칭 OWA, a ∝ cosh(gamma z) — 양 극단 강조."""
    return _sym_owa(ctx, u, "ext")


@force("symowa_med", gamma=2.0)
def _(ctx, u):
    """B3: 대칭 OWA, a ∝ exp(-gamma z²) — soft median."""
    return _sym_owa(ctx, u, "med")


@force("median")
def _(ctx, u):
    """B4: median_j x_ij."""
    return nb_median(ctx.g, diffs(ctx.g, u))


@force("huber_local", kappa=1.0)
def _(ctx, u):
    """C1: Huber, tau_i = kappa · median_j |x_ij| (노드별)."""
    x = diffs(ctx.g, u)
    tau = ctx.pf["kappa"] * nb_median(ctx.g, np.abs(x))[..., None] + EPS
    return edge_mean(ctx.g, np.clip(x, -tau, tau))


@force("huber_global", kappa=1.0)
def _(ctx, u):
    """C1: Huber, tau_t = kappa · RMS_edges(x) (실현별 전역)."""
    x = diffs(ctx.g, u)
    tau = ctx.pf["kappa"] * rms_rows(edge_values(ctx.g, x))[..., None] + EPS
    return edge_mean(ctx.g, np.clip(x, -tau, tau))


@force("amp_gated", kappa=2.0, beta=1.0)
def _(ctx, u):
    """C2: A3 에서 tau = kappa · σ̂, σ̂ = median_edges|x| / 0.6745 (잡음 수준 추정)."""
    x = diffs(ctx.g, u)
    sig = np.median(np.abs(edge_values(ctx.g, x)), axis=1)[:, None, None] / 0.6745
    return amp_force(ctx.g, x, ctx.pf["kappa"] * sig + EPS, ctx.pf["beta"])


@force("blend_agree", tau=0.5)
def _(ctx, u):
    """C3: alpha_i F_mean + (1-alpha_i) F_huber, alpha_i = exp(-(s_i/s̄)²), s_i = 이웃 차이의 표준편차."""
    g = ctx.g
    x = diffs(g, u)
    m = edge_mean(g, x)
    s = np.sqrt(edge_mean(g, (x - m[..., None]) ** 2))
    sbar = np.median(s, axis=1, keepdims=True) + EPS
    alpha = np.exp(-(s / sbar) ** 2)
    return alpha * m + (1 - alpha) * huber_force(g, x, ctx.pf["tau"])


@force("huber_anneal", tau0=1.0, tau_inf=0.2, Ta=100.0)
def _(ctx, u):
    """C4: Huber, tau_t = tau0 e^{-t/Ta} + tau_inf."""
    p = ctx.pf
    tau = p["tau0"] * np.exp(-ctx.t / p["Ta"]) + p["tau_inf"]
    return huber_force(ctx.g, diffs(ctx.g, u), tau)


@force("edge_memory", rho=0.05, sigma=0.5)
def _(ctx, u):
    """C5: (1/d_i) Σ w_ij(t) x_ij, w(t+1) = (1-rho) w + rho e^{-x²/σ²}. w 는 대칭이므로 형태 A 유지."""
    g = ctx.g
    x = diffs(g, u)
    w = ctx.state.setdefault("edge_w", np.ones_like(x))
    F = edge_mean(g, w * x)
    ctx.state["edge_w"] = (1 - ctx.pf["rho"]) * w + ctx.pf["rho"] * np.exp(-(x / ctx.pf["sigma"]) ** 2)
    return F


# ---- 이웃 값 배열 위의 기본 힘 (아이디어 4~6 의 감싸기용) ------------------
BASES = ("mean", "trim", "owa", "vw", "huber")


def gather(g, u):
    """노드 값 (R,n) -> 이웃 값 (R,n,dmax), 패딩 칸 0."""
    return np.where(g.valid, u[:, g.NB], 0.0)


def aggregate_nb(g, v, kind):
    """rx.aggregate 와 같은 집계를 이웃 값 배열 v (R,n,dmax) 에 직접 적용 (노드마다 다른 변환을 거친 값용)."""
    if kind == "mean":
        return (g.WT * v).sum(-1) / g.WT.sum(-1)
    if kind == "vw":
        vm = np.where(g.valid, v, -np.inf)
        e = np.where(g.valid, np.exp(rx.BETA * (vm - vm.max(-1, keepdims=True))), 0.0) * g.WT
        return (e * np.where(g.valid, v, 0.0)).sum(-1) / e.sum(-1)
    vs = np.where(g.valid, v, np.inf)
    order = np.argsort(vs, axis=-1, kind="stable")
    sv = np.take_along_axis(vs, order, -1)
    sw = np.take_along_axis(np.broadcast_to(g.WT, v.shape), order, -1)
    sv = np.where(sw > 0, sv, 0.0)
    coef = (g.trim_mask if kind == "trim" else g.owa_w) * sw
    return (coef * sv).sum(-1) / coef.sum(-1)


def nb_force(g, V, s, kind, tau=0.5):
    """이웃 값 V (R,n,dmax) 와 자기 값 s (R,n) 로 기본 힘 계산. kind ∈ BASES."""
    if kind == "huber":
        return edge_mean(g, np.clip(np.where(g.valid, V - s[..., None], 0.0), -tau, tau))
    return aggregate_nb(g, V, kind) - s


def base_force(g, u, kind):
    return nb_force(g, gather(g, u), u, kind)


# ---- 아이디어 4·5 (공간): 고차 짝수 차수 항 --------------------------------
# G_k(u) = (-1)^{k+1} F^{∘k}(u), F^{∘k} = 힘을 k번 합성. 선형 평균이면 G_k = -L^k u (복원 방향).
# 전체 힘 = Σ_k a_k G_k, a_1 = 1. 평균 집계에서 모드 강성은 g(λ) = Σ a_k λ^k.

def series_force(g, u, kind, weights):
    out, G = 0.0, u
    for k, a in enumerate(weights, 1):
        G = base_force(g, G, kind)
        if a:
            out = out + a * (-1) ** (k + 1) * G
    return out


SERIES_FIXED = {
    "hi_acc": ((1, 1 / 12, 1 / 90, 1 / 560), "아이디어4: 고정밀 라플라시안 급수 L + L²/12 + L³/90 + L⁴/560 (강성 증가)"),
    "hi_alt": ((1, -1 / 12, 1 / 90, -1 / 560), "아이디어4: 부호 교대 L - L²/12 + L³/90 - L⁴/560 (고주파 강성 감소)"),
}
SERIES_EPS = {  # 이름: (기본 eps, 최고 차수 K, 설명)
    "hi_geo": (0.2, 4, "아이디어4: a_k = eps^{k-1}, k=1..K"),
    "hi_geoneg": (-0.2, 4, "아이디어4: a_k = eps^{k-1} (eps<0, 부호 교대)"),
    "d4": (0.1, 2, "아이디어5: L + eps L² (4차 항 하나)"),
    "d4neg": (-0.1, 2, "아이디어5: L + eps L², eps<0"),
    "d4big": (0.3, 2, "아이디어5: L + eps L²"),
    "d4bigneg": (-0.3, 2, "아이디어5: L + eps L², eps<0"),
}


def _make_series_fixed(kind, w, doc):
    def fn(ctx, u):
        return series_force(ctx.g, u, kind, w)
    fn.__doc__ = f"{doc}. 기본 힘 = {kind}."
    return fn


def _make_series_eps(kind, K, doc):
    def fn(ctx, u):
        return series_force(ctx.g, u, kind, [ctx.pf["eps"] ** k for k in range(K)])
    fn.__doc__ = f"{doc}. 기본 힘 = {kind}."
    return fn


# ---- 아이디어 6: 평균에서 멀수록 제곱으로 더 멀리 보내고 힘 계산 -----------
# d = 값 - 평균, s = d 의 RMS (매 단계 다시 계산).
#   quad: φ(d) = d (1 + gamma (d/s)²)     sq: φ(d) = d|d| / s
# global: 전체 노드 평균 기준으로 ũ = m + φ(u - m) 를 만든 뒤 F = A(ũ) - ũ.
# local : 노드 i 마다 이웃 평균 m_i 기준으로 이웃 값과 자기 값을 변환한 뒤 F_i = A_i(Ṽ_i) - s̃_i.

# 반대 방향(평균 쪽으로 당김):
#   sqrt: φ(d) = sign(d) sqrt(|d| s)    (sq 의 역함수)
#   invq: φ(d) = d / (1 + gamma (d/s)²)  (quad 의 배율을 나눔; |d| > s/sqrt(gamma) 에서 감소함수)
#   tanh: φ(d) = s tanh(d/s)            (단조, 기울기 ≤ 1)

def _phi(d, s, mode, gamma):
    if mode == "quad":
        return d * (1 + gamma * (d / s) ** 2)
    if mode == "sq":
        return d * np.abs(d) / s
    if mode == "sqrt":
        return np.sign(d) * np.sqrt(np.abs(d) * s)
    if mode == "invq":
        return d / (1 + gamma * (d / s) ** 2)
    if mode == "tanh":
        return s * np.tanh(d / s)
    raise ValueError(mode)


def spread_force(g, u, kind, mode, scope, gamma):
    if scope == "global":
        m = u.mean(1, keepdims=True)
        d = u - m
        ut = m + _phi(d, rms_rows(d) + EPS, mode, gamma)
        return base_force(g, ut, kind)
    V = gather(g, u)
    m = edge_mean(g, V)
    dV = np.where(g.valid, V - m[..., None], 0.0)
    ds = u - m
    # 자기 값을 포함한 척도: 이웃이 모두 같아도 |d/s| ≤ sqrt(d_i + 1) 로 유한
    s = np.sqrt(((g.WT * dV ** 2).sum(-1) + ds ** 2) / (g.WT.sum(-1) + 1)) + EPS
    Vt = m[..., None] + _phi(dV, s[..., None], mode, gamma)
    return nb_force(g, Vt, m + _phi(ds, s, mode, gamma), kind)


SPREAD = {  # 이름: (mode, scope)
    "quad_g": ("quad", "global"), "sq_g": ("sq", "global"),
    "quad_l": ("quad", "local"), "sq_l": ("sq", "local"),
    "sqrt_g": ("sqrt", "global"), "invq_g": ("invq", "global"), "tanh_g": ("tanh", "global"),
    "sqrt_l": ("sqrt", "local"), "invq_l": ("invq", "local"), "tanh_l": ("tanh", "local"),
}


def _make_spread(kind, mode, scope):
    def fn(ctx, u):
        return spread_force(ctx.g, u, kind, mode, scope, ctx.pf.get("gamma", 1.0))
    fn.__doc__ = f"아이디어6: {mode}, {scope} 평균 기준. 기본 힘 = {kind}."
    return fn


for _b in BASES:
    for _n, (_w, _doc) in SERIES_FIXED.items():
        force(f"{_n}:{_b}")(_make_series_fixed(_b, _w, _doc))
    for _n, (_e, _K, _doc) in SERIES_EPS.items():
        force(f"{_n}:{_b}", eps=_e)(_make_series_eps(_b, _K, _doc))
    for _n, (_m, _s) in SPREAD.items():
        force(f"{_n}:{_b}", **({"gamma": 1.0} if _m in ("quad", "invq") else {}))(_make_spread(_b, _m, _s))


# ---- 시간 스킴 (아이디어 1·2) ----------------------------------------------
# 공통 기호: v = cur - prev, c2 = c², F = F(cur), Fp = F(prev)

@scheme("wave")
def _(ctx, cur, prev, F, Fp):
    """기준: u+ = u + v + c²F."""
    return 2 * cur - prev + ctx.c ** 2 * F


@scheme("damp", k=0.1)
def _(ctx, cur, prev, F, Fp):
    """1.1: u+ = u + (1-k)v + c²F."""
    return cur + (1 - ctx.ps["k"]) * (cur - prev) + ctx.c ** 2 * F


@scheme("viscous", nu=0.2)
def _(ctx, cur, prev, F, Fp):
    """1.2: u+ = u + v + c²F + nu(F - Fp). 감쇠 ∝ λ (고주파만)."""
    return 2 * cur - prev + ctx.c ** 2 * F + ctx.ps["nu"] * (F - Fp)


@scheme("damp_visc", k=0.05, nu=0.1)
def _(ctx, cur, prev, F, Fp):
    """1.3: u+ = u + (1-k)v + c²F + nu(F - Fp)."""
    p = ctx.ps
    return cur + (1 - p["k"]) * (cur - prev) + ctx.c ** 2 * F + p["nu"] * (F - Fp)


@scheme("sched_decay", k0=0.5, Ts=50.0)
def _(ctx, cur, prev, F, Fp):
    """1.4a: damp 에서 k_t = k0 e^{-t/Ts} (초기 과도 제거 후 순수 파동)."""
    k = ctx.ps["k0"] * np.exp(-ctx.t / ctx.ps["Ts"])
    return cur + (1 - k) * (cur - prev) + ctx.c ** 2 * F


@scheme("sched_grow", kinf=0.1, Ts=100.0)
def _(ctx, cur, prev, F, Fp):
    """1.4b: damp 에서 k_t = kinf (1 - e^{-t/Ts})."""
    k = ctx.ps["kinf"] * (1 - np.exp(-ctx.t / ctx.ps["Ts"]))
    return cur + (1 - k) * (cur - prev) + ctx.c ** 2 * F


@scheme("vel_shrink", alpha=1.0, k=0.0)
def _(ctx, cur, prev, F, Fp):
    """2.1: u+ = u + (1-k)v + c²F / (1 + alpha|v_i|/σ), σ = RMS(u0)."""
    v = cur - prev
    s = 1 + ctx.ps["alpha"] * np.abs(v) / rms_rows(ctx.u0)
    return cur + (1 - ctx.ps["k"]) * v + ctx.c ** 2 * F / s


@scheme("drag", alpha=0.5)
def _(ctx, cur, prev, F, Fp):
    """2.2: u+ = u + (1-k_i)v + c²F, k_i = min(1, alpha|v_i|/RMS(u0))."""
    v = cur - prev
    k = np.minimum(1.0, ctx.ps["alpha"] * np.abs(v) / rms_rows(ctx.u0))
    return cur + (1 - k) * v + ctx.c ** 2 * F


@scheme("limiter", vmax=0.3)
def _(ctx, cur, prev, F, Fp):
    """2.3: u+ = u + vm tanh(v/vm) + c²F, vm = vmax · RMS(u0)."""
    vm = ctx.ps["vmax"] * rms_rows(ctx.u0)
    return cur + vm * np.tanh((cur - prev) / vm) + ctx.c ** 2 * F


@scheme("cfl", s=0.9)
def _(ctx, cur, prev, F, Fp):
    """2.4: λ̂ = ‖F - Fp‖/‖v‖ (실현별), c_t² = min(c², s·4/λ̂). 4 = 2(2-k), k=0."""
    v = cur - prev
    nv = rms_rows(v)
    lam = rms_rows(F - Fp) / np.maximum(nv, EPS)
    c2 = np.where(nv > 1e-9, np.minimum(ctx.c ** 2, ctx.ps["s"] * 4 / np.maximum(lam, EPS)), ctx.c ** 2)
    return 2 * cur - prev + c2 * F


@scheme("energy", kmax=0.5)
def _(ctx, cur, prev, F, Fp):
    """2.5: damp 에서 k_t = kmax · clip(E_t/E_0 - 1, 0, 1), E = RMS(u)²."""
    E = rms_rows(cur) ** 2 / rms_rows(ctx.u0) ** 2
    k = ctx.ps["kmax"] * np.clip(E - 1, 0, 1)
    return cur + (1 - k) * (cur - prev) + ctx.c ** 2 * F


@scheme("saturate", kappa=1.0)
def _(ctx, cur, prev, F, Fp):
    """2.6: u+ = u + v + c² φ tanh(F/φ), φ = kappa · RMS(F at t=0) (고정)."""
    if "phi" not in ctx.state:
        ctx.state["phi"] = ctx.ps["kappa"] * rms_rows(F) + EPS
    phi = ctx.state["phi"]
    return 2 * cur - prev + ctx.c ** 2 * phi * np.tanh(F / phi)


# ---- 아이디어 4·5 (시간): 고차 짝수 시간 차분 항 --------------------------
# δ² f (t-1 중심) + Σ_k eps_k δ^{2k} f (t-k 중심) = c² F(f_{t-1}),
# δ^{2k} f = Σ_{j=0}^{2k} (-1)^j C(2k,j) f_{t-j}. 이를 f_t 에 대해 풀어 명시적으로 갱신.

def time_even_step(ctx, cur, prev, F, eps_list):
    from math import comb
    n_hist = 2 * (len(eps_list) + 1)
    h = ctx.state.get("s_hist")
    h = [cur] * n_hist if h is None else [cur] + h[:-1]   # h[j-1] = f_{t-j}, 초기에는 u(-k) = u0
    ctx.state["s_hist"] = h
    lead, rest = 1.0, -2 * h[0] + h[1]
    for k, e in enumerate(eps_list, 2):
        lead += e
        for j in range(1, 2 * k + 1):
            rest = rest + e * (-1) ** j * comb(2 * k, j) * h[j - 1]
    return (ctx.c ** 2 * F - rest) / lead


@scheme("t4", eps=0.1)
def _(ctx, cur, prev, F, Fp):
    """아이디어5(시간): f'' + eps f'''' = c²F."""
    return time_even_step(ctx, cur, prev, F, [ctx.ps["eps"]])


@scheme("t4big", eps=0.5)
def _(ctx, cur, prev, F, Fp):
    """아이디어5(시간): f'' + eps f'''' = c²F."""
    return time_even_step(ctx, cur, prev, F, [ctx.ps["eps"]])


@scheme("t4neg", eps=-0.1)
def _(ctx, cur, prev, F, Fp):
    """아이디어5(시간): f'' + eps f'''' = c²F, eps<0."""
    return time_even_step(ctx, cur, prev, F, [ctx.ps["eps"]])


@scheme("t6", e4=0.1, e6=0.01)
def _(ctx, cur, prev, F, Fp):
    """아이디어4(시간): f'' + e4 f'''' + e6 f'''''' = c²F."""
    return time_even_step(ctx, cur, prev, F, [ctx.ps["e4"], ctx.ps["e6"]])


@scheme("t6neg", e4=-0.1, e6=-0.01)
def _(ctx, cur, prev, F, Fp):
    """아이디어4(시간): f'' + e4 f'''' + e6 f'''''' = c²F, 음수 가중."""
    return time_even_step(ctx, cur, prev, F, [ctx.ps["e4"], ctx.ps["e6"]])


# ---- 시뮬레이션 ------------------------------------------------------------
def simulate(g, sname, fname, ps, pf, u0, c, steps):
    """반환: traj (steps+1,R,n), fail_step (R,) — 폭주하지 않으면 -1."""
    sfn, ffn = SCHEMES[sname][0], FORCES[fname][0]
    Rn, n = u0.shape
    X = np.empty((steps + 1, Rn, n))
    X[0] = u0
    prev, cur = u0.copy(), u0.copy()
    fail_step = np.full(Rn, -1)
    ctx = Ctx(g=g, c=c, pf=pf, ps=ps, u0=u0)
    F_prev = None
    with np.errstate(all="ignore"):
        for t in range(steps):
            ctx.t = t
            F = ffn(ctx, cur)
            nxt = sfn(ctx, cur, prev, F, F if F_prev is None else F_prev)
            bad = ~(np.abs(nxt) < FAIL_AMP).all(axis=1) & (fail_step < 0)
            fail_step[bad] = t + 1
            dead = fail_step >= 0
            nxt[dead] = 0.0
            F[dead] = 0.0
            prev, cur, F_prev = cur, nxt, F
            X[t + 1] = cur
    return X, fail_step


def partition_ari(g, X, truth):
    """후반부 간선 파형 차이로 q=2 절단 후 ARI (rank_experiment 와 같은 절차)."""
    h = X.shape[0] // 2
    d = X[h:, :, g.ei] - X[h:, :, g.ej]
    sc = np.median(np.sqrt((d ** 2).mean(0)), axis=0)
    lab = rx.cut_partition(g, sc, 2)
    return np.nan if lab is None else adjusted_rand_score(truth, lab)


def graph_specs(kinds, seeds, bridges):
    sp = []
    for k in kinds:
        b = 0 if k in ("er", "shuffled") else bridges
        for s in seeds:
            sp.append(dict(kind=k, sizes=(24, 24), bridges=b, wb=1.0, seed=s, role="test"))
    return sp


_GRAPH_CACHE: dict = {}


def run_one(job):
    """한 (그래프, 스킴, 힘, c) 조합. 그래프의 R개 초기값 실현을 함께 돌린다."""
    spec, sname, fname, ps, pf, c, steps, R = job
    key = (spec["kind"], spec["seed"], spec["bridges"])
    if key not in _GRAPH_CACHE:
        _GRAPH_CACHE[key] = rx.prepare(spec)
    g, truth = _GRAPH_CACHE[key]
    rng = np.random.default_rng(spec["seed"] * 1000 + 17)
    u0 = rng.uniform(-1, 1, (R, g.n))
    X, fs = simulate(g, sname, fname, ps, pf, u0, c, steps)
    ok = fs < 0
    r0 = np.sqrt((u0 ** 2).mean(1))
    tail = X[-rx.W:]
    final = np.sqrt((tail ** 2).mean((0, 2))) / r0       # 마지막 W 단계 RMS / 초기 RMS (DC 포함)
    ac = tail - tail.mean(2, keepdims=True)              # 노드 평균(DC) 제거: 합의 상태로 수렴하면 0
    ac_final = np.sqrt((ac ** 2).mean((0, 2))) / r0
    peak = np.sqrt((X ** 2).mean(2)).max(0) / r0
    med = lambda a: float(np.median(a[ok])) if ok.any() else np.nan
    return dict(scheme=sname, force=fname,
                sparams=" ".join(f"{k}={v}" for k, v in ps.items()),
                fparams=" ".join(f"{k}={v}" for k, v in pf.items()),
                c=c, kind=spec["kind"], seed=spec["seed"], runs=len(fs),
                fail_frac=float((~ok).mean()),
                mean_fail_step=float(fs[~ok].mean()) if (~ok).any() else np.nan,
                rms_final=med(final), ac_final=med(ac_final), rms_peak=med(peak),
                dead_frac=float((ok & (ac_final < 0.05)).mean()),
                ari=partition_ari(g, X[:, ok], truth) if ok.any() and truth is not None else np.nan)


# ---- 실행 ------------------------------------------------------------------
def summarize(rows):
    """(스킴, 힘) 별 전체 폭주 %."""
    s_names = list(dict.fromkeys(r["scheme"] for r in rows))
    f_names = list(dict.fromkeys(r["force"] for r in rows))
    print("fail % (all graphs, all c)")
    print(f"{'':<14}" + "".join(f"{s[:9]:>10}" for s in s_names))
    for f in f_names:
        vals = [np.mean([r["fail_frac"] for r in rows if r["scheme"] == s and r["force"] == f]) * 100
                for s in s_names]
        print(f"{f:<14}" + "".join(f"{v:>10.0f}" for v in vals))


def parse_params(items):
    """'name.key=val' 은 그 함수에만, 'key=val' 은 그 키를 가진 모든 함수에 적용."""
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = float(v)
    return out


def resolve(name, defaults, over):
    return {k: over.get(f"{name}.{k}", over.get(k, v)) for k, v in defaults.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--schemes", nargs="+", default=None, help="기본: 전부")
    ap.add_argument("-f", "--forces", nargs="+", default=None, help="기본: 전부")
    ap.add_argument("-k", "--kinds", nargs="+", default=["cycle", "clique", "sbm", "er"])
    ap.add_argument("-p", "--param", nargs="+", help="파라미터 덮어쓰기: name.key=val 또는 key=val")
    ap.add_argument("--c", nargs="+", type=float, default=[0.5, 1.0])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--bridges", type=int, default=2)
    ap.add_argument("--steps", type=int, default=rx.T)
    ap.add_argument("--R", type=int, default=rx.R, help="그래프당 초기값 수")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=str(HERE / "results" / "wave_test.csv"))
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list:
        for title, table in (("schemes", SCHEMES), ("forces", FORCES)):
            print(f"[{title}]")
            for name, (_, d, doc) in table.items():
                print(f"  {name:<14} {d or ''}  {doc}")
        return
    snames = a.schemes or list(SCHEMES)
    fnames = a.forces or list(FORCES)
    bad = [n for n in snames if n not in SCHEMES] + [n for n in fnames if n not in FORCES]
    if bad:
        sys.exit(f"unknown: {bad}. --list 로 확인")
    over = parse_params(a.param)

    specs = graph_specs(a.kinds, range(a.seeds), a.bridges)
    # 같은 그래프의 작업을 이어 붙여 워커의 그래프 캐시가 재사용되도록 정렬
    jobs = [(spec, s, f, resolve(s, SCHEMES[s][1], over), resolve(f, FORCES[f][1], over), c, a.steps, a.R)
            for spec in specs for s in snames for f in fnames for c in a.c]
    t0 = time.time()
    with Pool(a.jobs) as pool:
        rows = []
        for i, r in enumerate(pool.imap(run_one, jobs, chunksize=8), 1):
            rows.append(r)
            if i % 200 == 0:
                print(f"{i}/{len(jobs)} [{time.time() - t0:.0f}s]", file=sys.stderr, flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summarize(rows)
    print(f"saved {out} ({len(rows)} rows, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
