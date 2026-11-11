"""T1.15 held-out statistics: top-3 dev-selected configs vs baselines, seeds 100-115, zoo core."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cfgs import *

T = 500; R = 8

def job(args):
    spec, cfgs = args
    g, truth, meta = C.make_graph(spec)
    n0 = spec["sizes"][0]
    u0 = u0_for(g, spec, R)
    rows = []
    for cfg in cfgs:
        run = run_cfg(g, u0, cfg, T)
        row = dict(graph=meta["id"], kind=meta["kind"], seed=spec["seed"], bridges=meta["n_bridges"], config=cfg["label"])
        row.update(metrics(g, truth, meta, run))
        st = C.stimulus(g, n0, runner_cfg(cfg, 200), amps=(1.0,), signs=(+1,), R=8, steps=200, seed=spec["seed"], det_trials=16)[0]
        row.update(transmit_ratio=st["transmit_ratio"], far_peak=st["far_peak"], arrival=st["arrival"], snr=st["snr"],
                   noise_gain=st["noise_gain"], det_auc=st["det_auc"], stim_fail=st["fail_frac"])
        rows.append(row)
    rows.append(dict(graph=meta["id"], kind=meta["kind"], seed=spec["seed"], bridges=meta["n_bridges"], config="spectral",
                     ari_edge_cut=adjusted_rand_score(truth, rx.spectral_labels(g)), fail_frac=0.0))
    return rows

if __name__ == "__main__":
    cfgs = load_cfgs() + BASELINES
    specs = C.zoo("core", seeds=C.HELD_SEEDS)
    print(len(specs), "graphs", len(cfgs), "configs")
    if len(sys.argv) > 1:
        t0 = time.time(); r = job((specs[0], cfgs)); print(len(r), time.time() - t0); sys.exit()
    t0 = time.time()
    rs = C.pmap(job, [(s, cfgs) for s in specs], procs=2)
    rows = [r for x in rs for r in x]
    C.write_csv(OUT / "t15_heldout.csv", rows)
    print(len(rows), time.time() - t0)
