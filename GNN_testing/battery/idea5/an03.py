from lib5 import *
pd.set_option('display.width',250)
df=pd.read_csv(RES/'t03_mode_selective.csv')
print(len(df), df.graph.nunique())
d=df[df.family.isna() if 'family' in df else df.eps.notna()]
d=df[df.eps.notna()].copy()
# paired vs eps=0 same cmode
res=[]
for cm in ('c0.5','c1','rel0.8'):
    base=d[(d.cmode==cm)&(d.eps==0)].set_index(['graph','seed'])
    for e in sorted(d.eps.unique()):
        if e==0: continue
        sub=d[(d.cmode==cm)&(d.eps==e)].set_index(['graph','seed'])
        j=sub.join(base,rsuffix='_b',how='inner')
        for m in ('ari_edge_cut','edge_margin','high_pot_late','high_disp_late','ari_edge_cut_lp','ari_fft_lowmode'):
            x=j[m]; y=j[m+'_b']; ok=x.notna()&y.notna()
            if ok.sum()>=8:
                w=C.paired_wilcoxon(x[ok].values,y[ok].values)
                res.append(dict(cmode=cm,eps=e,metric=m,n=int(ok.sum()),mean_diff=w['mean_diff'],p=w['p']))
r=pd.DataFrame(res); 
r['p_holm']=np.nan
for (cm,m),sub in r.groupby(['cmode','metric']):
    r.loc[sub.index,'p_holm']=C.holm(sub.p.values)
print(r[r.metric.isin(['ari_edge_cut','high_pot_late'])].round(4).to_string())
# means with CI for key configs c0.5
for m in ('ari_edge_cut','edge_margin','high_pot_late','hold_edge_cut','hold_fft_lowmode'):
    print(m); 
    t=[]
    for (cm,e),sub in d.groupby(['cmode','eps']):
        x=sub[m].dropna().values
        if len(x)>5: t.append((cm,e,len(x))+C.boot_ci(x))
    print(pd.DataFrame(t,columns=['cmode','eps','n','mean','lo','hi']).round(3).query("cmode=='c0.5'").to_string())
print('hold>0 fraction any raw d4:', (d.hold_edge_cut.fillna(0)>0).mean(), (d.hold_edge_cut_lp.fillna(0)>0).mean(), (d.hold_fft_lowmode.fillna(0)>0).mean())
print('max edge_margin among d4 non-dead', d[d.dead_frac<0.5].edge_margin.max())
# diffusion baselines
b=df[df.config.isin(['diffusion+mean','damp(k=.1)+mean c=1','spectral'])]
print(b.groupby('config')[['ari_edge_cut','edge_margin','hold_edge_cut','hold_edge_cut_lp','hold_fft_lowmode']].agg(['mean']).round(3))
