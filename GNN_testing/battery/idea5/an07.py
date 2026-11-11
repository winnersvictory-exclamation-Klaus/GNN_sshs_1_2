from lib5 import *
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
df=pd.read_csv(RES/'t07_plate.csv'); print(len(df), df.graph.nunique())
df['stable_all']=df.fail_frac==0
print('pred_stable vs fail>0 (all rows):'); print(pd.crosstab(df.pred_stable, df.fail_frac>0))
# plate c_crit
p=df[df.form=='plate']; print('plate c_crit_pred median',p.c_crit_pred.median(), p.c_crit_pred.min(), p.c_crit_pred.max())
pl=p.groupby(['T','c']).agg(n=('graph','count'),fail=('fail_frac','mean'),pred_stable=('pred_stable','mean')); print(pl)
# stable-only comparison, T=500
s=df[(df['T']==500)&(df.fail_frac==0)]
cols=['ari_edge_cut','ari_edge_cut_lp','ari_fft_lowmode','edge_margin','edge_margin_lp','hold_edge_cut','high_pot_late','w_block']
print(s.groupby(['form','c']).agg(n=('graph','count'),**{c:(c,'mean') for c in cols}).round(3).to_string())
s2=df[(df['T']==2000)&(df.fail_frac==0)]
print(s2.groupby(['form','c']).agg(n=('graph','count'),**{c:(c,'mean') for c in cols}).round(3).to_string())
# paired plate vs plain at the same c
for T in (500,2000):
  for c in (0.25,0.5,0.75,1.0):
    for m in ('ari_edge_cut','ari_edge_cut_lp','ari_fft_lowmode'):
        a=df[(df['T']==T)&(df.c==c)&(df.form=='plate')&(df.fail_frac==0)].set_index(['graph','seed'])[m]
        b=df[(df['T']==T)&(df.c==c)&(df.form=='plain')&(df.fail_frac==0)].set_index(['graph','seed'])[m]
        j=pd.concat([a,b],axis=1,keys=['a','b']).dropna()
        if len(j)>=8:
            w=C.paired_wilcoxon(j.a.values,j.b.values); print('T',T,'c',c,m,'plate-plain n',len(j),'diff',round(w['mean_diff'],3),'p',round(w['p'],4))
