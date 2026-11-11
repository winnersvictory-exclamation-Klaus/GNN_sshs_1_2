import sys; sys.path.insert(0,'/Users/aidenkoh/Documents/GitHub/GNN_testing/battery')
import common as C, pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_columns',40); pd.set_option('display.max_rows',400)
df=pd.read_csv('/Users/aidenkoh/Documents/GitHub/GNN_testing/battery/idea6/results/t609_diffusion.csv').rename(columns={'transform':'tr'})
df=df[df.tr!='SPECTRAL'].copy()
df['cfg']=df.tr+np.where(df.gamma.notna(),'(g='+df.gamma.astype(str)+')','')+'@'+df.eta.astype(str)
df['ari']=np.where(df.fail_frac>0,0.0,df.ari_edge_cut.fillna(0))
mean=df[df.tr=='mean'].set_index(['eta','graph','seed'])
acf=df[df.tr=='mean'].groupby('eta').ac_final.median()
rows=[]
for cfg,s in df[df.tr!='mean'].groupby('cfg'):
    if s.fail_frac.mean()>0.05: continue
    eta=s.eta.iloc[0]
    me=s.ac_final.median()
    # rate-matched mean eta: closest log ac_final
    em=min(acf.index,key=lambda e: abs(np.log(max(acf[e],1e-12))-np.log(max(me,1e-12))))
    for lab,eb in (('same-eta',eta),('rate-matched',em),('mean@.5',0.5)):
        b=mean.loc[eb].ari
        x=s.set_index(['graph','seed']).ari
        a=pd.concat([x,b.reindex(x.index)],axis=1,keys=['a','b']).dropna()
        w=C.paired_wilcoxon(a.a.values,a.b.values); e,lo,hi=C.boot_ci((a.a-a.b).values)
        rows.append(dict(cfg=cfg,base=lab,base_eta=eb,n=len(a),ari=a.a.mean(),ari_base=a.b.mean(),diff=e,lo=lo,hi=hi,p=w['p']))
r=pd.DataFrame(rows)
for lab,s in r.groupby('base'):
    s=s.copy(); s['p_holm']=C.holm(s.p.values)
    print('baseline',lab); print(s.sort_values('diff',ascending=False).head(10).drop(columns='base').round(3).to_string())
    print('n positive Holm<.05:',(((s.p_holm<.05)&(s['diff']>0)).sum()),'of',len(s),'; negative:',(((s.p_holm<.05)&(s['diff']<0)).sum()))
