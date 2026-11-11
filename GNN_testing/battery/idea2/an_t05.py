import pandas as pd, numpy as np
pd.set_option("display.width",250)
d=pd.read_csv('results/t05_cfl_estimator.csv'); s=pd.read_csv('results/t05_cfl_estimator_samples.csv')
print("graphs",d.graph.nunique(),"samples",len(s))
cols=['fail_frac','wave_fail_frac','frac_ct_lt_c','ct_ratio_median','bias_median','bias_p10','bias_p90','corr','unsafe_frac','unsafe_frac_c','lam_true_med','lam_hat_med','n_samp']
g=d.groupby(['force','c','scheme'])[cols].mean().round(3)
print(g.to_string())
print("\nexact lam_max(L_rw) per graph kind:",d.groupby('kind').lam_exact.mean().round(3).to_dict(),"  c_crit_mean:",d.groupby('kind').c_crit_mean.mean().round(3).to_dict())
# mean force: lam_hat/lam_true by kind
m=d[d.force=='mean']
print("\nmean force bias_median by kind,c:\n",m.groupby(['kind','c','scheme']).bias_median.mean().round(3).unstack().to_string())
