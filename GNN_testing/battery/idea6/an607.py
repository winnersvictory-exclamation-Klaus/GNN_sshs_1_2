import sys; sys.path.insert(0,'/Users/aidenkoh/Documents/GitHub/GNN_testing/battery')
import common as C, pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_columns',40); pd.set_option('display.max_rows',400)
df=pd.read_csv('/Users/aidenkoh/Documents/GitHub/GNN_testing/battery/idea6/results/t607_frozen.csv').rename(columns={'transform':'tr'})
print(len(df),df.graph.nunique())
p=df.pivot_table(index=['u_scale','tr','variant'],columns='c',values='fail_frac')
print(p.round(2).to_string())
p=df.pivot_table(index=['u_scale','tr','variant'],columns='c',values='dead_frac')
print(p.round(2).to_string())
p=df.pivot_table(index=['u_scale','tr','variant'],columns='c',values='ac_final',aggfunc='median')
print(p.round(3).to_string())
# paired: frozen vs adaptive fail_frac per (scale,tr,c)
rows=[]
for (sc,tr,c),s in df.groupby(['u_scale','tr','c']):
    a=s[s.variant=='adaptive'].set_index(['graph','seed']).fail_frac; f=s[s.variant=='frozen'].set_index(['graph','seed']).fail_frac
    d=(f-a.reindex(f.index)); e,lo,hi=C.boot_ci(d.values); w=C.paired_wilcoxon(f.values,a.reindex(f.index).values)
    rows.append(dict(u_scale=sc,tr=tr,c=c,adapt=a.mean(),frozen=f.mean(),diff=e,lo=lo,hi=hi,p=w['p']))
r=pd.DataFrame(rows); r['p_holm']=C.holm(r.p.values)
print(r[(r.u_scale==1.0)].round(3).to_string())
print('n cells frozen better (holm<.05):',((r.p_holm<.05)&(r['diff']<0)).sum(),' worse:',((r.p_holm<.05)&(r['diff']>0)).sum(),' of',len(r))
