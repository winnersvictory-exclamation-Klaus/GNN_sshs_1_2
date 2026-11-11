"""T6.09b: stronger diffusion baselines (mean force at eta in {.75,.9,1.0}) to rate-match the transform gains."""
import i6lib as L
from i6lib import C, np


def job(spec):
    g, truth, meta = C.make_graph(spec)
    rng = np.random.default_rng(spec["seed"] * 1000 + 17)
    u0 = C.init("uniform", g, 8, rng)
    rows = []
    for eta in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        run = C.run_diff(g, "mean", u0, eta=eta, steps=500)
        rows.append(L.full_row(run, g, truth, meta, spec["block"], readouts=("edge_cut", "edge_cut_lp", "timeavg_2means", "traj_corr_spectral", "fft_lowmode"), transform="mean", eta=eta))
    return rows


if __name__ == "__main__":
    out = C.pmap(job, L.core_specs(C.DEV_SEEDS), procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(L.RES / "t609b_mean_eta.csv", rows)
    print(len(rows), "rows")
