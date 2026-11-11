"""T5.10 equal-cost comparison. d4:mean costs 2 aggregations/step.  At aggregation budgets B in {100,200,400,600,1000,1500}:
d4 runs B/2 steps, plain wave (same c) runs B steps, diffusion B steps.  ARI of edge_cut / edge_cut_lp / fft_lowmode on
the prefix; steps-to-partition in aggregations (first B with ARI>=0.9).  eps in {-0.3,-0.1,0.1,0.3}, c in {0.5,1.0}.
Core zoo, dev seeds 0-3 (REDUCED from 0-7), R=8.  Output results/t10_equal_cost.csv"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib5 import *

EPS = [-0.3, -0.1, 0.1, 0.3]
CS = [0.5, 1.0]
BS = [100, 200, 400, 600, 1000, 1500]
WH = ("edge_cut", "edge_cut_lp", "fft_lowmode")


def arib(run, g, truth, Ts, mult):
    """ARI at prefix lengths Ts (steps); budget = mult*T."""
    res = ari_vs_T(run, g, truth, Ts, which=WH)
    out = {}
    for nm in WH:
        for T in Ts:
            out[f"ari_{nm}_B{T * mult}"] = res[f"ari_{nm}_T{T}"]
        arr = np.array([res[f"ari_{nm}_T{T}"] for T in Ts])
        hit = np.where(arr >= 0.9)[0]
        out[f"stpB_{nm}"] = float(Ts[hit[0]] * mult) if len(hit) else np.nan
    return out


def job(spec):
    g, truth, meta, u0 = graph_job_setup(spec)
    rows = []
    def add(label, runner, mult, ex):
        Ts = [B // mult for B in BS]
        r = runner(g, u0, steps=Ts[-1])
        row = dict(graph=meta["id"], seed=meta["seed"], kind=meta["kind"], bridges=meta["n_bridges"], config=label, **ex)
        row["fail_frac"] = float(r.fail.mean())
        if r.ok.any():
            row.update(arib(r, g, truth, Ts, mult))
        rows.append(row)
    for c in CS:
        add(f"wave c={c:g}", C.make_runner("wave", "mean", c), 1, dict(family="wave", c=c, eps=0.0, cost=1))
        for e in EPS:
            add(f"d4 eps={e:g} c={c:g}", C.make_runner("wave", "d4:mean", c, pf=dict(eps=e)), 2, dict(family="d4", c=c, eps=e, cost=2))
    add("diffusion", C.make_diff_runner("mean", 0.5), 1, dict(family="diff", c=np.nan, eps=np.nan, cost=1))
    return rows


if __name__ == "__main__":
    small = "--small" in sys.argv
    sp = C.zoo("core", seeds=C.DEV_SEEDS[:1] if small else C.DEV_SEEDS[:4])
    out = C.pmap(job, sp, procs=2)
    df = pd.DataFrame([r for rs in out for r in rs])
    C.write_csv(RES / ("t10_small.csv" if small else "t10_equal_cost.csv"), df)
    cols = [f"ari_edge_cut_lp_B{b}" for b in BS]
    print(df.groupby("config", sort=False)[["fail_frac"] + cols].mean().round(3).to_string())
