import pandas as pd, numpy as np, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import i2lib as L
C=L.C
pd.set_option("display.width",250); pd.set_option("display.max_rows",600)
d=pd.read_csv('results/t12_combinations.csv')
d["alive"]=((d.fail_frac==0)&(d.dead_frac<0.5)).astype(float)
d["alive_r"]=(1-d.fail_frac-d.dead_frac).clip(lower=0)
print("graphs",d.graph.nunique(),"rows",len(d))
print("\nplain wave fail_frac by force:",d[d.family=='wave'].groupby('force').fail_frac.mean().round(2).to_dict())
for fam,(a,b) in (("vel_shrink",("alpha","k")),("limiter_damp",("vmax","k")),("saturate_visc",("kappa","nu"))):
    print(f"\n=== {fam}: rows {a}, cols {b}; cell = fail_frac / alive_r (mean over 9 graphs); per force ===")
    for f in ("trim","owa","vw","median"):
        s=d[(d.family==fam)&(d.force==f)]
        pf=s.pivot_table(index='p1',columns='p2',values='fail_frac').round(2); pa=s.pivot_table(index='p1',columns='p2',values='alive_r').round(2)
        print(f"-- force {f} ({a} rows x {b} cols)"); print((pf.astype(str)+"/"+pa.astype(str)).to_string())
print("\n=== controls: damp(k), viscous(nu) fail/alive ===")
for fam in ("damp","viscous"):
    s=d[d.family==fam]
    print(fam); print((s.pivot_table(index='force',columns='p2',values='fail_frac').round(2).astype(str)+"/"+s.pivot_table(index='force',columns='p2',values='alive_r').round(2).astype(str)).to_string())
# does the nonlinear part add anything? paired: composite(p1,k) vs pure damp(k) / viscous(nu) with same k: fail_frac and alive
print("\n=== paired composite vs matched linear control (same k / nu): over (graph, force, p1) , mean diff in alive_r (comp - ctrl) and fail_frac ===")
rows=[]
for fam,ctrl in (("vel_shrink","damp"),("limiter_damp","damp"),("saturate_visc","viscous")):
    cs=d[d.family==ctrl][['graph','force','p2','alive_r','fail_frac','ari_edge_cut']].rename(columns={'alive_r':'ca','fail_frac':'cf','ari_edge_cut':'cari'})
    s=d[(d.family==fam)&(d.p2>0)].merge(cs,on=['graph','force','p2'])
    for p1,ss in s.groupby('p1'):
        r=C.paired_wilcoxon(ss.alive_r.values,ss.ca.values)
        rows.append(dict(family=fam,p1=p1,n=r['n'],d_alive=r['mean_diff'],d_fail=float((ss.fail_frac-ss.cf).mean()),p=r['p']))
t=pd.DataFrame(rows); t['p_holm']=C.holm(t.p.fillna(1).values)
print(t.round(4).to_string())
# best ARI among configs that are fail==0 & alive
ok=d[(d.fail_frac==0)&(d.dead_frac<0.5)&(d.ari_edge_cut.notna())]
print("\nconfigs alive (fail 0, dead<.5): mean ARI by family/force:\n",ok.groupby(['family','force']).agg(n=('graph','count'),ari=('ari_edge_cut','mean')).round(3).to_string())
