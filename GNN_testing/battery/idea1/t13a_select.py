"""Dev-seed (0-7) selection scan used to choose the top-3 configs for T1.13/T1.15 (core zoo, T=500)."""
import sys, time, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib1 import *

T = 500; R = 8

def cand_list():
    L = []
    for f in ["mean"]:
        for k in (0.003, 0.01, 0.03, 0.1, 0.3):
            for c in (0.5, 1.0): L.append((f, "damp", dict(k=k), c))
        for nu in (0.05, 0.1, 0.2, 0.4):
            for c in (0.5, 1.0): L.append((f, "viscous", dict(nu=nu), c))
        for k, nu in ((0.01, 0.1), (0.03, 0.2), (0.1, 0.2)):
            L.append((f, "damp_visc", dict(k=k, nu=nu), 1.0))
    for f in ["trim", "huber", "symowa_ext", "symowa_med", "median", "absvw"]:
        for k in (0.03, 0.1, 0.3):
            for c in (0.5, 1.0): L.append((f, "damp", dict(k=k), c))
    for f in ["owa", "vw"]:
        for k in (0.3, 0.5, 0.7):
            for c in (0.5, 1.0): L.append((f, "damp", dict(k=k), c))
    L.append(("mean", "wave", {}, 0.5))
    return L

def label(f, sch, ps, c):
    return f"{sch}[{','.join(f'{k}={v:g}' for k, v in ps.items())}]+{f} c={c:g}" if ps else f"{sch}+{f} c={c:g}"

def one(g, truth, meta, run, row):
    s = C.traj_summary(run)
    row.update(fail_frac=s["fail_frac"], dead_frac=s["dead_frac"], ac_final=s["ac_final"], growth_rate=s["growth_rate"])
    if run.ok.any():
        ev = C.evaluate_partitions(run.X, g, truth, run.fail_step)
        row.update({k: ev[k] for k in ("ari_edge_cut", "ari_edge_cut_lp", "ari_timeavg_2means", "ari_traj_corr_spectral", "ari_fft_lowmode",
                                       "edge_margin", "bridge_auc", "minfrac_edge_cut")})
        lab = C.partitions(run.X, g, 2, run.fail_step, which=("edge_cut",))["edge_cut"]
        h = C.hold(run.X, g, lab, fail_step=run.fail_step) if lab is not None else None
        row["hold_frac"] = float(np.nanmean(np.asarray(h["hold"], float) > 0)) if h is not None else np.nan
    return row

def job(spec):
    g, truth, meta = C.make_graph(spec)
    u0 = u0_for(g, spec, R)
    rows = []
    for f, sch, ps, c in cand_list():
        run = erun(g, sch, f, u0, c, T, ps)
        rows.append(one(g, truth, meta, run, dict(graph=meta["id"], kind=spec["kind"], bridges=spec["bridges"], seed=spec["seed"],
                                                  config=label(f, sch, ps, c), force=f, scheme=sch, ps=str(ps), c=c)))
    run = C.run_diff(g, "mean", u0, 0.5, T)
    rows.append(one(g, truth, meta, run, dict(graph=meta["id"], kind=spec["kind"], bridges=spec["bridges"], seed=spec["seed"],
                                              config="diffusion+mean", force="mean", scheme="diffusion", ps="{}", c=np.nan)))
    return rows

if __name__ == "__main__":
    specs = [dict(kind=k, sizes=(24, 24), bridges=b, wb=1.0, seed=s) for k in ("cycle", "clique", "sbm") for b in (1, 2, 4, 8) for s in C.DEV_SEEDS]
    if len(sys.argv) > 1:
        t0 = time.time(); r = job(specs[0]); print(len(r), time.time() - t0); sys.exit()
    t0 = time.time()
    rs = C.pmap(job, specs, procs=2)
    rows = [r for x in rs for r in x]
    C.write_csv(OUT / "t13a_select.csv", rows)
    print(len(rows), time.time() - t0)
