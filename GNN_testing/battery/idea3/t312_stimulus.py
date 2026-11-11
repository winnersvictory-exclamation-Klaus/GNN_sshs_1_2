"""T3.12 stimulus transmission across the bridge: every force x {wave c=.5, damp(.1) c=1, diffusion} x amp{.2,1,5} x sign +-."""
import i3
from i3 import C, np
import pandas as pd
from t310_partition import FORCES, runner

AMPS = (0.2, 1.0, 5.0)

def job(args):
    spec, = args
    g, truth, meta = C.make_graph(spec)
    n0 = int((truth == truth[0]).sum())
    rows = []
    for f in FORCES:
        for sk in ("wave", "damp", "diff"):
            run = runner(sk, f)
            try:
                st = C.stimulus(g, n0, run, amps=AMPS, signs=(+1, -1), sigma=0.05, R=8, steps=200, seed=spec["seed"], det_trials=32)
            except Exception as e:
                rows.append(dict(graph=meta["id"], force=f, scheme=sk, error=str(e))); continue
            for r in st:
                r = dict(r); r.update(graph=meta["id"], gkind=spec["block"], bridges=meta["n_bridges"], seed=spec["seed"], force=f, scheme=sk)
                rows.append(r)
    return rows

if __name__ == "__main__":
    import sys
    quick = len(sys.argv) > 1
    specs = C.zoo("core", seeds=(0,) if quick else (0, 1))
    if quick: specs = specs[:1]
    out = C.pmap(job, [(s,) for s in specs], procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(i3.RES / ("t312_stimulus_quick.csv" if quick else "t312_stimulus.csv"), rows)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 800)
    print(df[df.amp == 1.0].groupby(["force", "scheme"])[["fail_frac", "transmit_ratio", "noise_gain", "det_auc"]].mean().round(3).to_string())
