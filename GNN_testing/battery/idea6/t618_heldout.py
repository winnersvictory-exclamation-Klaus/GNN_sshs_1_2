"""T6.18 held-out statistics: seeds 100-115 x core. Configs fixed from dev (selected.json) -- no tuning here."""
import json
import i6lib as L
from i6lib import C, np
from rank_experiment import spectral_labels
from sklearn.metrics import adjusted_rand_score

SEL = json.load(open(L.HERE / "selected.json"))
CFGS = L.BASELINES + SEL


def job(spec):
    g, truth, meta = C.make_graph(spec)
    rng = np.random.default_rng(spec["seed"] * 1000 + 17)
    u0 = C.init("uniform", g, 8, rng)
    rows = [L.base_row(meta, block=spec["block"], label="SPECTRAL", ari_edge_cut=adjusted_rand_score(truth, spectral_labels(g)))]
    for cfg in CFGS:
        rn = L.build_runner(cfg)
        run = rn(g, u0)
        row = L.full_row(run, g, truth, meta, spec["block"], readouts=("edge_cut", "edge_cut_lp", "timeavg_2means", "traj_corr_spectral", "fft_lowmode"), label=cfg["label"])
        rn_s = L.build_runner(cfg, steps=200, fail_amp=400.0)
        st = C.stimulus(g, 24, rn_s, amps=(1.0,), signs=(+1, -1), steps=200, seed=spec["seed"])
        row["transmit_ratio"] = float(np.nanmean([r["transmit_ratio"] for r in st]))
        row["det_auc"] = float(np.nanmean([r["det_auc"] for r in st]))
        row["noise_gain"] = float(np.nanmean([r["noise_gain"] for r in st]))
        row["far_peak"] = float(np.nanmean([r["far_peak"] for r in st]))
        rows.append(row)
    return rows


if __name__ == "__main__":
    specs = L.core_specs(C.HELD_SEEDS)
    out = C.pmap(job, specs, procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(L.RES / "t618_heldout.csv", rows)
    print(len(rows), "rows")
