"""T2.06 energy trigger.  `energy` uses E=RMS(u)^2/RMS(u0)^2 (DC included).  Is it triggered by DC drift (node mean) rather than
oscillation?  Observer splits E into DC part and AC part; compares energy vs energy_ac for owa/vw/trim/mean at c=1, kmax grid,
dc in {0, 3}.  Graphs cycle/clique/sbm bridges 2 seeds 0-1; R=8; T=1000."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import i2lib as L
import numpy as np
C = L.C
T = 1000

def job(args):
    spec, R = args
    g, truth, meta = C.make_graph(spec)
    rows = []
    for dc in (0.0, 3.0):
        u0 = L.make_u0(spec, g, R, dc=dc)
        for f in ("mean", "trim", "owa", "vw"):
            for sch in ("energy", "energy_ac", "wave"):
                for km in ((0.1, 0.3, 0.5, 1.0) if sch != "wave" else (None,)):
                    ps = {"kmax": km} if km is not None else {}
                    Ed = np.full((T, R), np.nan); Ea = np.full((T, R), np.nan); Kk = np.full((T, R), np.nan)
                    r2 = (L.rms_rows(u0)[:, 0]) ** 2
                    a2 = (L.ac_rms_rows(u0)[:, 0]) ** 2
                    def obs(t, ctx, cur, prev, F, Fp, nxt, alive):
                        dcp = cur.mean(1) ** 2
                        acp = L.ac_rms_rows(cur)[:, 0] ** 2
                        Ed[t] = np.where(alive, dcp / r2, np.nan)         # DC part of E (rms-normalised)
                        Ea[t] = np.where(alive, acp / a2, np.nan)         # AC energy relative to its initial value
                        if sch == "energy":
                            E = (dcp + acp) / r2
                        elif sch == "energy_ac":
                            E = acp / a2
                        else:
                            E = np.full(R, np.nan)
                        Kk[t] = np.where(alive, np.clip(E - 1, 0, 1), np.nan)
                    run, _ = L.instr_run(g, sch, f, u0, 1.0, T, ps, extra=obs)
                    trig = Kk > 0
                    drift_only = trig & (Ea <= 1.0)          # triggered although AC energy has not exceeded its initial value
                    row = dict(graph=meta["id"], seed=spec["seed"], kind=spec["kind"], dc=dc, force=f, scheme=sch, kmax=km if km else np.nan,
                               fail_frac=float(run.fail.mean()), ac_final=L.nanmed(C.ac_final(run.X, run.fail_step)),
                               dead_frac=float(C.dead(run.X, run.fail_step).mean()),
                               trig_frac=float(np.nanmean(np.where(np.isnan(Kk), np.nan, trig))) if np.isfinite(Kk).any() else np.nan,
                               drift_only_frac_of_trig=float(drift_only.sum() / max(trig.sum(), 1)),
                               dc_part_max=float(np.nanmax(Ed)) if np.isfinite(Ed).any() else np.nan,
                               ac_ratio_max=float(np.nanmax(Ea)) if np.isfinite(Ea).any() else np.nan)
                    # first trigger step relative to first AC>1 step (per realisation, medians)
                    ft = np.where(trig.any(0), trig.argmax(0), -1); fa = np.where((Ea > 1).any(0), (Ea > 1).argmax(0), -1)
                    row["first_trig_med"] = L.nanmed(np.where(ft >= 0, ft, np.nan)); row["first_acgt1_med"] = L.nanmed(np.where(fa >= 0, fa, np.nan))
                    row["dc_drift_dw"] = L.nanmed(C.dc_drift(run.X, g, run.fail_step)["dw"])
                    rows.append(row)
    return rows

if __name__ == "__main__":
    t0 = time.time()
    specs = L.graphs_dev(seeds=(0, 1))
    out = C.pmap(job, [(s, 8) for s in specs], procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(Path(__file__).parent / "results" / "t06_energy_trigger.csv", rows)
    print("rows", len(rows), "time %.0fs" % (time.time() - t0))
