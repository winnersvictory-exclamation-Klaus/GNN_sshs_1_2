import sys; sys.path.insert(0,'/Users/aidenkoh/Documents/GitHub/GNN_testing/battery/idea4')
import i4, numpy as np, pandas as pd
C=i4.C
pd.set_option("display.width",250); pd.set_option("display.max_columns",40)
df=pd.read_csv('results/t405_equal_cost.csv'); df=df[df.config!='spectral'].copy()
CENS=2560.0
for rd in ('fft_lowmode','edge_cut_lp'):
    df[f'lagg90_{rd}']=np.log(df[f'agg90_{rd}'].fillna(CENS))
df['cfg']=df.config+'|'+df.policy
for pol in ('c0.5','0.9cp'):
    base='wave|'+pol
    print('######## policy',pol,'baseline',base)
    for val in ('ari_edge_cut_lp_B160','ari_edge_cut_lp_B320','ari_edge_cut_lp_B640','ari_edge_cut_lp_B1280','ari_fft_lowmode_B640','ari_fft_lowmode_B1280','lagg90_edge_cut_lp','lagg90_fft_lowmode'):
        sub=df[df.policy==pol]
        t=C.paired_vs_baseline(sub.assign(cfg=sub.config),val,'cfg','wave',pair=('graph','seed')).round(4)
        b=C.boot_ci(sub[sub.config=='wave'][val].values)
        print(val,'wave mean %.3f [%.3f,%.3f]'%b)
        print(t[['config','n','mean_diff','ci_lo','ci_hi','p','p_holm']].to_string(index=False))
