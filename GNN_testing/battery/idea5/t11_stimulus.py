"""T5.11 stimulus transmission.  C.stimulus (n0=24, amps 0.2/1/5, both signs, R=8, T=200, det_trials 32) on 24+24 cycle/clique/sbm,
bridges=2, dev seeds 0-3.  Configs: wave c=0.5/1, d4 eps in {-0.3,-0.1,0.1,0.3} (c=0.5; and c=1 where stable), t4neg
(eps=-0.1,-1/12) c=0.5/1, damp(.1) c=1, diffusion.  Output results/t11_stimulus.csv"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib5 import *

def configs():
    L = []
    for c in (0.5, 1.0):
        L.append((f"wave c={c:g}", C.make_runner("wave", "mean", c, steps=200)))
        for e in (-0.3, -0.1, 0.1, 0.3):
            L.append((f"d4 eps={e:g} c={c:g}", C.make_runner("wave", "d4:mean", c, pf=dict(eps=e), steps=200)))
        for e in (-0.1, -1 / 12):
            L.append((f"t4neg eps={e:.3g} c={c:g}", C.make_runner("t4neg", "mean", c, ps=dict(eps=e), steps=200)))
    L.append(("damp(k=.1) c=1", C.make_runner("damp", "mean", 1.0, ps=dict(k=0.1), steps=200)))
    L.append(("diffusion", C.make_diff_runner("mean", 0.5, steps=200)))
    return L


def job(spec):
    g, truth, meta = C.make_graph(spec)
    rows = []
    for label, runner in configs():
        for r in C.stimulus(g, 24, runner, R=8, steps=200, seed=spec["seed"]):
            rows.append(dict(graph=meta["id"], seed=spec["seed"], kind=spec["kind"], config=label, **r))
    return rows


if __name__ == "__main__":
    small = "--small" in sys.argv
    sp = [dict(kind=k, bridges=2, seed=s) for k in ("cycle", "clique", "sbm") for s in (C.DEV_SEEDS[:1] if small else C.DEV_SEEDS[:4])]
    out = C.pmap(job, sp, procs=2)
    df = pd.DataFrame([r for rs in out for r in rs])
    C.write_csv(RES / ("t11_small.csv" if small else "t11_stimulus.csv"), df)
    print(df[df.amp == 1].groupby("config", sort=False)[["fail_frac", "far_peak", "transmit_ratio", "arrival", "snr", "noise_gain", "det_auc"]].mean().round(3).to_string())
