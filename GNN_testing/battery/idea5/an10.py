from lib5 import *
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
df=pd.read_csv(RES/'t10_equal_cost.csv'); print(len(df), df.graph.nunique())
BS=[100,200,400,600,1000,1500]
for nm in ('edge_cut_lp','edge_cut','fft_lowmode'):
    cols=[f'ari_{nm}_B{b}' for b in BS]
    print('==',nm); print(df.groupby('config',sort=False)[cols+['stpB_'+nm]].mean().round(3).to_string())
# paired d4 (B/2 steps) vs wave same c at same budget
for nm in ('edge_cut_lp','edge_cut','fft_lowmode'):
    rows=[]
    for c in (0.5,1.0):
        for e in (-0.3,-0.1,0.1,0.3):
            for B in BS:
                col=f'ari_{nm}_B{B}'
                x=df[df.config==f'd4 eps={e:g} c={c:g}'].set_index(['graph','seed'])[col]
                y=df[df.config==f'wave c={c:g}'].set_index(['graph','seed'])[col]
                j=pd.concat([x,y],axis=1,keys=['x','y']).dropna()
                if len(j)>=8:
                    w=C.paired_wilcoxon(j.x.values,j.y.values); rows.append(dict(c=c,eps=e,B=B,n=len(j),mean_diff=w['mean_diff'],p=w['p']))
    r=pd.DataFrame(rows); r['p_holm']=C.holm(r.p.values)
    print('== paired d4 - wave at equal aggregations, readout',nm,' (Holm over',len(r),'tests)')
    print(r.pivot_table(index=['c','eps'],columns='B',values='mean_diff').round(3).to_string())
    print('min p_holm',r.p_holm.min(),' n sig(<.05):',(r.p_holm<0.05).sum(), ' sig rows:'); print(r[r.p_holm<0.05].round(4).to_string())
