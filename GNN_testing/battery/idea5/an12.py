from lib5 import *
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
df=pd.read_csv(RES/'t12_partition_hold.csv'); print(len(df), df.graph.nunique())
cols=['fail_frac','dead_frac','ac_final','ari_edge_cut','ari_edge_cut_lp','ari_timeavg_2means','ari_traj_corr_spectral','ari_fft_lowmode','edge_margin','bridge_auc','hold_edge_cut','hold_edge_cut_lp','hold_fft_lowmode','holdpos_edge_cut','best_window_ari']
g=df.groupby('config',sort=False)[cols].mean()
print(g.round(3).to_string())
print(df.groupby('config',sort=False).edge_margin.median().round(2).to_string())
# bootstrap CI for key
t=[]
for cfg,sub in df.groupby('config',sort=False):
    for m in ('ari_edge_cut','hold_edge_cut'):
        x=sub[m].dropna().values
        if len(x): t.append((cfg,m,len(x))+C.boot_ci(x))
tt=pd.DataFrame(t,columns=['config','metric','n','mean','lo','hi']); print(tt[tt.config.str.contains('t4neg|diff|damp|wave c=1|wave c=0.5')].round(3).to_string())
# paired t4neg c=1 vs baselines
for a in ('t4neg eps=-0.1 c=1','t4neg eps=-0.05 c=1'):
  for b in ('wave c=1','damp(k=.1) c=1','diffusion'):
    for m in ('ari_edge_cut','hold_edge_cut'):
        x=df[df.config==a].set_index(['graph','seed'])[m]; y=df[df.config==b].set_index(['graph','seed'])[m]
        j=pd.concat([x,y],axis=1,keys=['x','y']).dropna(); w=C.paired_wilcoxon(j.x.values,j.y.values)
        print(a,'vs',b,m,'n',len(j),'diff',round(w['mean_diff'],3),'p',w['p'])
# by kind and bridges for t4neg c=1
s=df[df.config=='t4neg eps=-0.1 c=1']
print(s.groupby(['kind','bridges'])[['ari_edge_cut','hold_edge_cut','ac_final','edge_margin']].mean().round(3))
d=df[df.config=='diffusion']; print(d.groupby(['kind','bridges'])[['ari_edge_cut','hold_edge_cut']].mean().round(3))
