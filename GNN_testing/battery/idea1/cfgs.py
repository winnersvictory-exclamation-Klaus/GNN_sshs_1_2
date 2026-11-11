"""Config helpers for T1.13/14/15: configs read from results/top_configs.json."""
import json
from lib1 import *
from sklearn.metrics import adjusted_rand_score
rx = C.rx

def load_cfgs(path=None):
    p = Path(path) if path else OUT / "top_configs.json"
    return json.load(open(p))

BASELINES = [
    dict(label="wave+mean c=0.5", force="mean", scheme="wave", ps={}, c=0.5),
    dict(label="damp[k=0.1]+mean c=0.5", force="mean", scheme="damp", ps=dict(k=0.1), c=0.5),
    dict(label="diffusion+mean", force="mean", scheme="diffusion", ps={}, c=None),
]

def run_cfg(g, u0, cfg, T):
    if cfg["scheme"] == "diffusion":
        return C.run_diff(g, "mean", u0, 0.5, T)
    return erun(g, cfg["scheme"], cfg["force"], u0, cfg["c"], T, cfg["ps"])

def runner_cfg(cfg, steps):
    if cfg["scheme"] == "diffusion":
        return C.make_diff_runner("mean", 0.5, steps=steps)
    return C.make_runner(cfg["scheme"], cfg["force"], cfg["c"], cfg["ps"], steps=steps)

def metrics(g, truth, meta, run, with_hold=True):
    s = C.traj_summary(run)
    row = dict(fail_frac=s["fail_frac"], dead_frac=s["dead_frac"], ac_final=s["ac_final"], growth_rate=s["growth_rate"])
    if run.ok.any():
        ev = C.evaluate_partitions(run.X, g, truth, run.fail_step)
        for k in ("ari_edge_cut", "ari_edge_cut_lp", "ari_timeavg_2means", "ari_fft_lowmode", "edge_margin", "bridge_auc",
                  "minfrac_edge_cut", "n_ok"):
            row[k] = ev.get(k, np.nan)
        lab = C.partitions(run.X, g, 2 if truth is not None else 2, run.fail_step, which=("edge_cut",))["edge_cut"]
        if lab is not None and with_hold:
            h = C.hold(run.X, g, lab, fail_step=run.fail_step)
            hv = np.asarray(h["hold"], float)
            row["hold_mean"] = float(np.nanmean(hv)); row["hold_frac"] = float(np.nanmean(hv > 0))
            row["minfrac_cut"] = float(C.partition_metrics(g, lab, truth)["minfrac"])
    return row
