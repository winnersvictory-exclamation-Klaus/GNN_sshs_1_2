import sys
from pathlib import Path
import pandas as pd, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import i2lib as L
C = L.C
d = pd.read_csv(Path(__file__).parent / "results" / "t01_phase_diagram.csv")
d["alive"] = (1 - d.fail_frac - d.dead_frac).clip(lower=0)   # fraction of realisations neither failed nor dead
d["alive_amp"] = d.alive * ((d.ac_final.fillna(0) > 0.05) | (d.fail_frac == 1)).astype(float)
print("n graphs", d.graph.nunique(), "rows", len(d))
F = L.FORCES10
pd.set_option("display.width", 250)
# wave & damp baselines
print("\nBASELINE fail_frac (mean over graphs) force x c")
for s in ("wave", "damp"):
    print(s); print(d[d.scheme == s].pivot_table(index="force", columns="c", values="fail_frac").round(2).T[F].to_string())
for c in (0.5, 1.0, 1.4, 1.8):
    print(f"\n=== c={c}: fail_frac (mean over 12 graphs), rows scheme[param], cols force; 'alive' in () if fail<1 ===")
    for s in L.ORIG:
        sub = d[(d.scheme == s) & (d.c == c)]
        piv = sub.pivot_table(index="param", columns="force", values="fail_frac")[F]
        al = sub.pivot_table(index="param", columns="force", values="alive")[F]
        out = piv.round(2).astype(str) + "/" + al.round(2).astype(str)
        print(s, "(fail/alive)"); print(out.to_string())
# summary: number of (force) cells fully rescued (fail==0 on all graphs & alive>=0.5) per scheme/param/c vs wave
print("\n=== count of nonmean forces with fail_frac==0 on ALL graphs and alive>=0.5 (of 9) ===")
nm = d[d.force != "mean"]
g = nm.groupby(["scheme", "param", "c", "force"]).agg(fail=("fail_frac", "mean"), alive=("alive", "mean")).reset_index()
g["rescued"] = ((g.fail == 0) & (g.alive >= 0.5)).astype(int)
print(g.pivot_table(index=["scheme", "param"], columns="c", values="rescued", aggfunc="sum").to_string())
w = nm[nm.scheme == "wave"].groupby(["c", "force"]).fail_frac.mean().reset_index(); w["resc"] = (w.fail_frac == 0).astype(int)
print("wave rescued count by c:", w.groupby("c").resc.sum().to_dict())
dd = nm[nm.scheme == "damp"].groupby(["c", "force"]).agg(f=("fail_frac", "mean"), a=("alive", "mean")).reset_index(); dd["resc"] = ((dd.f == 0) & (dd.a >= .5)).astype(int)
print("damp0.1 rescued count by c:", dd.groupby("c").resc.sum().to_dict())
# mean force: harm by schemes at c<=1 (fail_frac>0 under stabiliser though wave is stable)
m = d[(d.force == "mean") & (d.c <= 1.0)]
print("\n=== mean force c<=1: schemes that FAIL where plain wave is stable (mean fail_frac) ===")
print(m.pivot_table(index=["scheme", "param"], columns="c", values="fail_frac").round(2).to_string())
