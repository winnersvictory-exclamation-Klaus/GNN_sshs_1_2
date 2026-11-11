"""T5.08 is it just rewiring?  For the mean base, wave on L+eps L^2 (d4:mean) vs plain wave+mean on an explicit
augmented graph G'.  Derivation: S = D (L + eps L^2) is symmetric with zero row sums, so L+eps L^2 = D^-1 S is a
random-walk Laplacian of the weighted graph with off-diagonal weights w'_ij = -S_ij (= (1+2eps) A_ij - eps (A D^-1 A)_ij)
and extra self-loop mass s_i = D_i - S_ii = -eps D_i (1+q_i), q_i = sum_k P_ik P_ki, so that D'_i = D_i (needed because
rx.G's mean divides by D').  w' >= 0 iff -1/2 <= eps <= 0.  For eps > 0 the 2-hop weights are -eps*(A D^-1 A)_ij < 0
(repulsive): no rx.G exists; we test the nearest non-negative graph (negative weights dropped).
rx.G self-loop edge (i,i,w) is stored twice => effective self weight 2w, so pass w = s_i/2.
Output results/t08_rewiring.csv"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib5 import *

EPS_NEG = [-0.5, -0.4, -0.3, -0.2, -0.1, -0.05]
EPS_POS = [0.05, 0.1, 0.3, 1.0]
T = 500


def rewire(g, eps, clip=False):
    d = C.wdeg(g)
    Lr = C.L_rw(g)
    S = d[:, None] * (Lr + eps * Lr @ Lr)
    asym = float(np.abs(S - S.T).max())
    W = -S.copy()
    np.fill_diagonal(W, 0.0)
    neg_mass = float(-W[W < 0].sum())
    pos_mass = float(W[W > 0].sum())
    Wp = np.where(W > 1e-13, W, 0.0)
    selfw = d - Wp.sum(1)
    min_self = float(selfw.min())
    selfw = np.maximum(selfw, 0.0)
    edges = [(i, j, float(Wp[i, j])) for i in range(g.n) for j in range(i + 1, g.n) if Wp[i, j] > 0]
    edges += [(i, i, float(selfw[i] / 2)) for i in range(g.n) if selfw[i] > 1e-13]
    gp = rx.G(g.n, edges)
    return gp, dict(asym=asym, neg_mass=neg_mass, pos_mass=pos_mass, neg_frac=neg_mass / max(pos_mass, 1e-300),
                    min_self=min_self, m_orig=len(g.edges), m_new=len([e for e in edges if e[0] != e[1]]),
                    self_frac=float(selfw.sum() / d.sum()), W=Wp, selfw=selfw)


def job(spec):
    g, truth, meta, u0 = graph_job_setup(spec)
    d = C.wdeg(g)
    rows = []
    # cut conductance of the true partition: original vs augmented (weights only, self-loops excluded)
    def cutratio(Wm):
        a = truth == 0
        cut = Wm[np.ix_(a, ~a)].sum()
        inn = Wm[np.ix_(a, a)].sum() / 2 + Wm[np.ix_(~a, ~a)].sum() / 2
        return float(cut / inn)
    cr0 = cutratio(g.A - np.diag(np.diag(g.A)))
    for e in EPS_NEG + EPS_POS:
        gp, info = rewire(g, e)
        W = info.pop("W"); info.pop("selfw")
        for c in (0.25, 0.5, 1.0):
            r1 = C.run(g, "wave", "d4:mean", u0, c, T, pf=dict(eps=e))
            r2 = C.run(gp, "wave", "mean", u0, c, T)
            both = r1.ok & r2.ok
            if both.any():
                dX = np.abs(r1.X[:, both] - r2.X[:, both]).max()
                rel = float(dX / max(np.abs(r1.X[:, both]).max(), 1e-300))
            else:
                dX, rel = np.nan, np.nan
            row = dict(graph=meta["id"], seed=spec["seed"], kind=spec["kind"], bridges=spec["bridges"], eps=e, c=c,
                       exact_graph_exists=bool(-0.5 - 1e-12 <= e <= 0), max_traj_diff=float(dX), rel_traj_diff=rel,
                       fail_frac_d4=float(r1.fail.mean()), fail_frac_graph=float(r2.fail.mean()),
                       cut_ratio_orig=cr0, cut_ratio_aug=cutratio(W), **info)
            for tag, r in (("d4", r1), ("rew", r2)):
                if r.ok.any():
                    ev = C.evaluate_partitions(r.X, g, truth, r.fail_step, which=("edge_cut", "edge_cut_lp", "fft_lowmode"))
                    row[f"ari_edge_cut_{tag}"] = ev["ari_edge_cut"]; row[f"ari_fft_{tag}"] = ev["ari_fft_lowmode"]
                    row[f"ari_lp_{tag}"] = ev["ari_edge_cut_lp"]; row[f"em_{tag}"] = ev["edge_margin"]
            if r2.ok.any():
                ev2 = C.evaluate_partitions(r2.X, gp, truth, r2.fail_step, which=("edge_cut",))   # readout on augmented edges
                row["ari_edge_cut_rew_augedges"] = ev2["ari_edge_cut"]; row["em_rew_augedges"] = ev2["edge_margin"]
            rows.append(row)
    return rows


if __name__ == "__main__":
    small = "--small" in sys.argv
    specs = C.zoo("core", seeds=C.DEV_SEEDS[:1] if small else C.DEV_SEEDS[:2])
    out = C.pmap(job, specs, procs=2)
    df = pd.DataFrame([r for rs in out for r in rs])
    C.write_csv(RES / ("t08_small.csv" if small else "t08_rewiring.csv"), df)
    print(df.groupby(["eps", "c"])[["max_traj_diff", "rel_traj_diff", "neg_frac", "m_new", "m_orig", "cut_ratio_orig", "cut_ratio_aug"]].median().round(10).to_string())
