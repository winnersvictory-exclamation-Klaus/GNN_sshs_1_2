import pandas as pd, numpy as np, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import i2lib as L
C=L.C
pd.set_option("display.width",250); pd.set_option("display.max_rows",500)
d=pd.read_csv('results/t11_stimulus.csv')
print("graphs",d.graph.nunique(),"rows",len(d))
d["fp_per_amp"]=d.far_peak/d.amp
order=list(dict.fromkeys(d.config))
# per graph x config x amp (avg of signs)
a=d.groupby(['graph','config','amp']).agg(fp=('fp_per_amp','mean'),fail=('fail_frac','mean'),auc=('det_auc','mean'),tr=('transmit_ratio','mean'),snr=('snr','mean'),ng=('noise_gain','mean')).reset_index()
p=a.pivot_table(index=['graph','config'],columns='amp',values='fp').reset_index()
p['lin_idx']=p[20.0]/p[0.2]          # far_peak/amp at 20 over at 0.2 (1=linear, <1 compressive)
p['lin_idx5']=p[5.0]/p[0.2]
rows=[]
for cfg in order:
    s=p[p.config==cfg]
    r=dict(config=cfg)
    for k in ('lin_idx5','lin_idx'):
        e,lo,hi=C.boot_ci(s[k].dropna().values) if s[k].notna().any() else (np.nan,)*3
        r[k]=f"{e:.3f} [{lo:.3f},{hi:.3f}]"
    q=a[a.config==cfg]
    for am in (0.2,1.0,5.0,20.0):
        qq=q[q.amp==am]
        r[f"fp/amp@{am:g}"]=round(qq.fp.mean(),3); r[f"auc@{am:g}"]=round(qq.auc.mean(),3); r[f"fail@{am:g}"]=round(qq.fail.mean(),2)
    r["tr@1"]=round(q[q.amp==1.0].tr.mean(),3); r["noise_gain"]=round(q.ng.mean(),3)
    # sign asymmetry at amp 5: |fp(+)-fp(-)|/mean
    s5=d[(d.config==cfg)&(d.amp==5.0)].pivot_table(index='graph',columns='sign',values='far_peak')
    r["sign_asym@5"]=round(float((abs(s5[1]-s5[-1])/((s5[1]+s5[-1])/2)).mean()),3) if len(s5) else np.nan
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
