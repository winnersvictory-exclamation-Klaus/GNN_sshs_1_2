import sys; sys.path.insert(0,'/Users/aidenkoh/Documents/GitHub/GNN_testing/battery/idea4')
import i4, numpy as np, pandas as pd
from scipy.stats import spearmanr
C=i4.C
pd.set_option("display.width",250); pd.set_option("display.max_columns",40)
df=pd.read_csv('results/t404_gap_reshaping.csv')
CENS=3200.0
for rd in ('fft_lowmode','edge_cut_lp','edge_cut'):
    df[f'lt90c_{rd}']=np.log(df[f't90_{rd}'].fillna(CENS))
df['cfg']=df.config+'|'+df.policy
VALS=('ari_end_fft_lowmode','ari_end_edge_cut_lp','ari_fft_lowmode_T400','ari_edge_cut_lp_T400','lt90c_fft_lowmode','lt90c_edge_cut_lp')
for base in ('wave|0.9cp','wave|c0.5'):
    print('######## baseline',base)
    for val in VALS:
        t=C.paired_vs_baseline(df,val,'cfg',base,pair=('graph','seed')).round(4)
        b=C.boot_ci(df[df.cfg==base][val].values)
        print(val,'baseline mean %.3f [%.3f,%.3f] n=%d'%(*b,df[df.cfg==base][val].notna().sum()))
        print(t.to_string(index=False))
for pol in ('0.9cp','c0.5'):
    sub=df[df.policy==pol]
    for rd in ('fft_lowmode','edge_cut_lp'):
        ok=sub[sub[f't90_{rd}'].notna()]
        r,p=spearmanr(1/ok.om2,ok[f't90_{rd}'])
        print(pol,rd,'Spearman(1/om2, t90) among reached: %.3f p=%.2g n=%d'%(r,p,len(ok)))
    print(sub.groupby('config',sort=False)[['gap_ratio','kappa_g','om2','om3','om_sep']].mean().round(3))
print(df.groupby(['block','cfg']).ari_end_edge_cut_lp.mean().unstack(0).round(2))
print(df.groupby(['bridges','cfg']).ari_end_edge_cut_lp.mean().unstack(0).round(2))
