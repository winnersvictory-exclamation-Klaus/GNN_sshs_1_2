"""T5.16 negative controls: zoo 'neg' (er, shuffled, random_regular d=3,4,6,8, watts_strogatz; no true blocks), dev seeds 0-7.
False-partition rate = P(hold>0 with min_frac fix) on edge_cut / edge_cut_lp / fft_lowmode labels; degenerate-cut rate =
P(minfrac<0.1) of the edge_cut label.  Configs: wave c=0.5, d4 eps in {-0.3,-0.1,0.1,0.3} c=0.5, t4neg eps=-0.1 c=0.5, damp(.1) c=1,
diffusion.  R=8, T=500.  Output results/t16_neg_controls.csv"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib5 import *

T = 500


def configs():
    L = [("wave c=0.5", C.make_runner("wave", "mean", 0.5, steps=T))]
    for e in (-0.3, -0.1, 0.1, 0.3):
        L.append((f"d4 eps={e:g} c=0.5", C.make_runner("wave", "d4:mean", 0.5, pf=dict(eps=e), steps=T)))
    L.append(("t4neg eps=-0.1 c=0.5", C.make_runner("t4neg", "mean", 0.5, ps=dict(eps=-0.1), steps=T)))
    L.append(("damp(k=.1) c=1", C.make_runner("damp", "mean", 1.0, ps=dict(k=0.1), steps=T)))
    L.append(("diffusion", C.make_diff_runner("mean", 0.5, steps=T)))
    return L


def job(spec):
    g, truth, meta = C.make_graph(spec)
    rng = np.random.default_rng(spec["seed"] * 1000 + 17)
    u0 = C.init("uniform", g, 8, rng)
    rows = []
    for label, runner in configs():
        r = runner(g, u0)
        row = dict(graph=meta["id"], seed=spec["seed"], kind=spec["kind"], config=label, **C.traj_summary(r))
        if r.ok.any():
            labs = C.partitions(r.X, g, 2, r.fail_step, which=("edge_cut", "edge_cut_lp", "fft_lowmode"))
            for nm, lab in labs.items():
                h = C.hold(r.X, g, lab, fail_step=r.fail_step)
                h0 = C.hold(r.X, g, lab, fail_step=r.fail_step, min_frac=0.0)
                mf = float(h["minfrac"]) if lab is not None else np.nan
                m = C.partition_metrics(g, lab, None)
                row[f"false_{nm}"] = float(np.nanmean(h["hold_pos"]))
                row[f"false_rx_{nm}"] = float(np.nanmean(h0["hold_pos"]))
                row[f"minfrac_{nm}"] = m["minfrac"]
                row[f"degenerate_{nm}"] = float(m["minfrac"] < 0.1) if np.isfinite(m["minfrac"]) else np.nan
                row[f"cond_{nm}"] = m["cond"]
        rows.append(row)
    return rows


if __name__ == "__main__":
    small = "--small" in sys.argv
    sp = C.zoo("neg", seeds=C.DEV_SEEDS[:1] if small else C.DEV_SEEDS)
    out = C.pmap(job, sp, procs=2)
    df = pd.DataFrame([r for rs in out for r in rs])
    C.write_csv(RES / ("t16_small.csv" if small else "t16_neg_controls.csv"), df)
    print(df.groupby("config", sort=False)[["fail_frac", "false_edge_cut", "false_edge_cut_lp", "false_fft_lowmode", "false_rx_edge_cut", "degenerate_edge_cut"]].mean().round(3).to_string())
