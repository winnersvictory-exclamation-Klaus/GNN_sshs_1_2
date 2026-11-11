from lib5 import *
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
df=pd.read_csv(RES/'t04_nonlinear_bases.csv'); dp=pd.read_csv(RES/'t04_props.csv'); print(len(df), df.graph.nunique())
print('fail_frac mean over graphs (rows base x eps; cols c)')
print(df.pivot_table(index=['base','eps'],columns='c',values='fail_frac',aggfunc='mean').round(2).to_string())
print('median first/mean fail step c=0.5'); print(df.pivot_table(index=['base','eps'],columns='c',values='mean_fail_step',aggfunc='median').round(0).to_string())
# paired vs eps=0 (fail_frac) per base,c
rows=[]
for b in ('mean','trim','owa','vw','huber'):
    for c in (0.25,0.5,1.0):
        base=df[(df.base==b)&(df.c==c)&(df.eps==0)].set_index(['graph','seed']).fail_frac
        for e in (-0.3,-0.1,0.1,0.3):
            x=df[(df.base==b)&(df.c==c)&(df.eps==e)].set_index(['graph','seed']).fail_frac
            j=pd.concat([x,base],axis=1,keys=['x','b']).dropna()
            w=C.paired_wilcoxon(j.x.values,j.b.values); rows.append(dict(base=b,c=c,eps=e,n=len(j),mean_diff=w['mean_diff'],p=w['p']))
r=pd.DataFrame(rows); r['p_holm']=C.holm(r.p.values); print(r[r.p<0.1].round(4).to_string()); print('n tests',len(r),'min p_holm',r.p_holm.min())
print('props (median over graphs, states):')
pc=['oddness','momentum_rel','shift_rel','lam_max','eig_sym_max','dsym_rel','c_crit_pred','rho_J']
print(dp.groupby(['base','eps'])[pc].median().round(3).to_string())
