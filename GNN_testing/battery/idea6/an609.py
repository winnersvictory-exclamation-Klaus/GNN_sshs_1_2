import sys; sys.path.insert(0,'/Users/aidenkoh/Documents/GitHub/GNN_testing/battery')
import common as C, pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_columns',40); pd.set_option('display.max_rows',400)
df=pd.read_csv('/Users/aidenkoh/Documents/GitHub/GNN_testing/battery/idea6/results/t609_diffusion.csv').rename(columns={'transform':'tr'})
sp=df[df.tr=='SPECTRAL'][['graph','seed','ari_edge_cut']].rename(columns={'ari_edge_cut':'spec'})
df=df[df.tr!='SPECTRAL'].copy()
df['cfg']=df.tr+np.where(df.gamma.notna(),'(g='+df.gamma.astype(str)+')','')
print('spectral ARI',C.boot_ci(sp.spec))
reads=['ari_edge_cut','ari_edge_cut_lp','ari_timeavg_2means','ari_traj_corr_spectral','ari_fft_lowmode']
for eta,s in df.groupby('eta'):
    s=s.copy()
    for r in reads: s[r+'_eff']=np.where(s.fail_frac>0,0.0,s[r].fillna(0))
    t=s.groupby('cfg').agg(fail=('fail_frac','mean'),dead=('dead_frac','mean'),acfin=('ac_final','median'),
        **{r:(r+'_eff','mean') for r in reads}, margin=('edge_margin','median'), hold=('hold_rec','mean'),hold_ac=('hold_rec_ac','mean'))
    print('eta',eta); print(t.round(3).to_string())
