from lib5 import *
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
df=pd.read_csv(RES/'t09_eps_bridges.csv')
print(len(df), df.graph.nunique())
d=df[df.family=='d4'].copy()
def paired(metric, cfgs_base):
    out=[]
    for c in (0.5,1.0):
        base=d[(d.c==c)&(d.eps==0)].set_index(['graph','seed'])[metric]
        ps=[]; rows=[]
        for e in sorted(d.eps.unique()):
            if e==0: continue
            x=d[(d.c==c)&(d.eps==e)].set_index(['graph','seed'])[metric]
            j=pd.concat([x,base],axis=1,keys=['x','b']).dropna()
            if len(j)>=8:
                w=C.paired_wilcoxon(j.x.values,j.b.values); rows.append(dict(c=c,eps=e,n=len(j),mean_diff=w['mean_diff'],p=w['p']))
        r=pd.DataFrame(rows); r['p_holm']=C.holm(r.p.values); out.append(r)
    return pd.concat(out)
for m in ('ari_edge_cut_lp','ari_edge_cut','ari_fft_lowmode','edge_margin_lp','stp_edge_cut_lp'):
    print('== paired vs eps=0 (same c):',m); print(paired(m,None).round(4).to_string())
# means by bridges
for m in ('ari_edge_cut_lp','ari_fft_lowmode','stp_edge_cut_lp','edge_margin_lp'):
    print('==',m,'c=0.5 by bridges x eps'); print(d[d.c==0.5].pivot_table(index='eps',columns='bridges',values=m,aggfunc='mean').round(3).to_string())
b=df[df.family.isin(['damp','diff','spectral'])]
print(b.groupby(['family','bridges'])[['ari_edge_cut_lp','ari_edge_cut','ari_fft_lowmode','stp_edge_cut_lp','edge_margin_lp']].mean().round(3).to_string())
print(df.groupby('config',sort=False)[['fail_frac_T1500']].mean().T.round(2).to_string())
# CIs overall for key configs
t=[]
for cfg,sub in df.groupby('config',sort=False):
    x=sub.ari_edge_cut_lp.dropna().values
    if len(x): t.append((cfg,len(x))+C.boot_ci(x))
print(pd.DataFrame(t,columns=['config','n','mean','lo','hi']).round(3).to_string())
