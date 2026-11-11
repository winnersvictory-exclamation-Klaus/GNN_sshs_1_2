import sys; sys.path.insert(0,'/Users/aidenkoh/Documents/GitHub/GNN_testing/battery')
import common as C, pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_columns',40); pd.set_option('display.max_rows',400)
df=pd.read_csv('/Users/aidenkoh/Documents/GitHub/GNN_testing/battery/idea6/results/t605_blockoffset.csv')
def paired(df, value, baseline, family, delta):
    d=df[(df.delta==delta)&(df.config.isin(family+[baseline]))].copy()
    # keep graphs where the config itself did not fail
    d.loc[d.fail_frac>0, value]=np.nan
    rows=[]
    base=d[d.config==baseline].set_index(['graph','seed'])[value]
    for cfg in family:
        s=d[d.config==cfg].set_index(['graph','seed'])[value]
        a=pd.concat([s,base],axis=1,keys=['a','b']).dropna()
        if len(a)<5: rows.append(dict(config=cfg,n=len(a))); continue
        diff=a.a-a.b; e,lo,hi=C.boot_ci(diff.values); w=C.paired_wilcoxon(a.a.values,a.b.values)
        rows.append(dict(config=cfg,n=len(a),mean=a.a.mean(),base=a.b.mean(),diff=e,lo=lo,hi=hi,p=w['p']))
    r=pd.DataFrame(rows); r['p_holm']=C.holm(r.p.fillna(1).values); return r
fam_d=[c for c in df.config.unique() if c.startswith('diff+') and 'mean eta=.25'!=c.split('+')[1] and c!='diff+mean eta=.5']
fam_w=[c for c in df.config.unique() if c.startswith('wave+') and c not in('wave+mean c=.25','wave+mean c=.5')]
for delta in (0.0,1.0):
  for v in ['env_ret_end','rel_contrast_end','hold_true_ac']:
    print('== diffusion vs diff+mean eta=.25  delta',delta,v); print(paired(df,v,'diff+mean eta=.25',fam_d,delta).round(3).to_string())
    print('== wave vs wave+mean c=.25 delta',delta,v); print(paired(df,v,'wave+mean c=.25',fam_w,delta).round(3).to_string())
