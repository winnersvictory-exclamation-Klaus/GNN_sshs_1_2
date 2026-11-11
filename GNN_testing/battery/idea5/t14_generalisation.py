"""T5.14 generalisation: dev-selected top-3 configs (results/top3.json) + baselines on zoo 'wide' and 'scale', dev seeds 0-1, R=8, T=500.
(bridges 1-8, unequal sizes etc. are in 'wide'; n=48..384 in 'scale').  Rows per graph x config: fail, dead, ARI of all readouts, edge_margin, hold.
Output results/t14_generalisation.csv"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib5 import *

T = 500


def load_configs():
    top = json.load(open(RES / "top3.json"))
    L = []
    for t in top:
        L.append((t["label"], C.make_runner(t["scheme"], t["force"], t["c"], ps=t.get("ps"), pf=t.get("pf"), steps=T), dict(family="top3")))
    L += [("wave+mean c=0.5", C.make_runner("wave", "mean", 0.5, steps=T), dict(family="base")),
          ("wave+mean c=1", C.make_runner("wave", "mean", 1.0, steps=T), dict(family="base")),
          ("damp(k=.1)+mean c=1", C.make_runner("damp", "mean", 1.0, ps=dict(k=0.1), steps=T), dict(family="base")),
          ("diffusion+mean", C.make_diff_runner("mean", 0.5, steps=T), dict(family="base"))]
    return L


def job(spec):
    g, truth, meta = C.make_graph(spec)
    rng = np.random.default_rng(spec["seed"] * 1000 + 17)
    u0 = C.init("uniform", g, 8, rng, truth=truth)
    q = meta.get("q") or 2
    rows = []
    for label, runner, ex in load_configs():
        r = runner(g, u0)
        row = eval_run(r, g, truth, meta, label, extra=dict(ex, zoo=spec.get("_zoo", ""), q=q, bridges_meta=meta["n_bridges"]))
        rows.append(row)
    rows.append(dict(spectral_row(g, truth, meta), family="spectral"))
    return rows


if __name__ == "__main__":
    small = "--small" in sys.argv
    sp = []
    for z in ("wide", "scale"):
        for s in C.zoo(z, seeds=C.DEV_SEEDS[:1] if small else C.DEV_SEEDS[:2]):
            s = dict(s); s["_zoo"] = z; sp.append(s)
    out = C.pmap(job, sp, procs=2)
    df = pd.DataFrame([r for rs in out for r in rs])
    C.write_csv(RES / ("t14_small.csv" if small else "t14_generalisation.csv"), df)
    print(df.groupby("config", sort=False)[["fail_frac", "ari_edge_cut", "edge_margin", "hold_edge_cut"]].mean().round(3).to_string())
