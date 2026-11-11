"""T6.09 spread transforms under the diffusion update (u+ = u + eta F)."""
import i6lib as L
from i6lib import C, np

ETAS = [0.1, 0.25, 0.5]
READS = ("edge_cut", "edge_cut_lp", "timeavg_2means", "traj_corr_spectral", "fft_lowmode")


def job(spec):
    g, truth, meta = C.make_graph(spec)
    rng = np.random.default_rng(spec["seed"] * 1000 + 17)
    u0 = C.init("uniform", g, 8, rng)
    from rank_experiment import spectral_labels
    from sklearn.metrics import adjusted_rand_score
    spec_ari = adjusted_rand_score(truth, spectral_labels(g))
    rows = [L.base_row(meta, block=spec["block"], transform="SPECTRAL", gamma=np.nan, eta=np.nan, ari_edge_cut=spec_ari)]
    cfgs = [("mean", None)] + [(tr, 1.0) for tr in L.TRANSFORMS] + [("quad_g", 0.1), ("quad_l", 0.1), ("invq_g", 0.1), ("invq_l", 0.1)]
    for eta in ETAS:
        for tr, gam in cfgs:
            f = "mean" if tr == "mean" else L.fname(tr)
            pf = None if gam is None else L.pf_for(tr, gam)
            run = C.run_diff(g, f, u0, eta=eta, steps=500, pf=pf)
            row = L.full_row(run, g, truth, meta, spec["block"], readouts=READS, transform=tr, gamma=gam if gam else np.nan, eta=eta)
            rows.append(row)
    return rows


if __name__ == "__main__":
    specs = L.core_specs(C.DEV_SEEDS)
    out = C.pmap(job, specs, procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(L.RES / "t609_diffusion.csv", rows)
    print(len(rows), "rows")
