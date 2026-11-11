import pandas as pd, numpy as np, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import i2lib as L
C=L.C
pd.set_option("display.width",250); pd.set_option("display.max_rows",500)
d=pd.read_csv('results/t07_minimal_rescue.csv')
sc=d[d.type=='scan']; rs=d[d.type=='rescue']
print("scan rows",len(sc),"rescue rows",len(rs))
F=L.FORCES10
w=sc[sc.scheme=='wave'].groupby(['kind','force']).fail_frac.mean().unstack('force')[F]
print("plain wave c=1 T=2000 fail_frac\n",w.round(2).to_string())
# minimal rescue param (per kind, force, scheme) = from rescue rows (min_idx)
mr=rs.groupby(['scheme','force','kind']).agg(min_idx=('min_idx','first'),min_param=('min_param','first')).reset_index()
tab=mr.pivot_table(index=['scheme','force'],columns='kind',values='min_idx')
print("\nminimal rescuing grid index (0=weakest of 10; NaN = never rescued) per scheme/force/kind; wave-stable => idx 0")
print(tab.unstack('force').reindex(columns=F,level=1).to_string() if False else tab.to_string())
# count of rescued (force,kind) cells per scheme
allc=pd.MultiIndex.from_product([sorted(sc.scheme.unique()-{'wave'} if False else [s for s in sc.scheme.unique() if s!='wave']),F,['cycle','clique','sbm']],names=['scheme','force','kind'])
res=mr.set_index(['scheme','force','kind']).reindex(allc)
res['rescued']=res.min_idx.notna()
print("\nrescued cells (of 10 forces x 3 kinds) per scheme (any strength in grid):\n",res.groupby('scheme').rescued.sum().to_dict())
nm=res.reset_index(); nm=nm[nm.force!='mean']
print("non-mean forces rescued (of 27):",nm.groupby('scheme').rescued.sum().to_dict())
# alive/dead at 1x,2x,4x
print("\n=== alive/dead at minimal rescue x1, x2, x4 (mean over graphs; non-mean forces only; rescue rows exist only where rescued) ===")
r=rs[rs.force!='mean']
t=r.groupby(['scheme','factor']).agg(n=('graph','count'),fail=('fail_frac','mean'),dead=('dead_frac','mean'),alive=('alive_frac','mean'),life=('lifetime','median'),ac_final=('ac_final','median'),ari=('ari_edge_cut','mean')).round(3)
print(t.to_string())
print("\n=== per force at x1: alive_frac/dead_frac/lifetime/ARI ===")
t=r[r.factor==1].groupby(['scheme','force']).agg(n=('graph','count'),min_param=('min_param','median'),fail=('fail_frac','mean'),dead=('dead_frac','mean'),alive=('alive_frac','mean'),life=('lifetime','median'),ari=('ari_edge_cut','mean')).round(2)
print(t.to_string())
