from lib5 import *
df=pd.read_csv(RES/'t01_spatial_theory.csv')
print(len(df), df.graph.nunique(), 'exact fail_step match', df.exact_match.mean(), 'max diff', df.max_fail_diff.max())
df['pred_unst']=~df.pred_stable
print(pd.crosstab(df.pred_unst, df.fail_frac>0))
# disagreements: predicted stable but failed (should be 0), predicted unstable but not failed within T=1000
d=df[(df.pred_unst)&(df.fail_frac==0)]; print('pred unstable, no fail in 1000 steps:',len(d), 'with pred_ln_rho<0.01:', (d.pred_ln_rho<0.01).sum(), 'min pred_ln_rho', d.pred_ln_rho.min(), 'zero-mode(g<=0) cases', (d.pred_nonpos_modes>0).sum())
d2=df[(df.pred_stable)&(df.fail_frac>0)]; print('pred stable but failed', len(d2))
# growth-rate agreement on failing runs
f=df[(df.fail_frac==1)&df.growth_rate.notna()&(df.pred_ln_rho>0.02)]
print('growth meas/pred n',len(f), (f.growth_rate/f.pred_ln_rho).median(), (f.growth_rate/f.pred_ln_rho).quantile([.1,.9]).values)
# stability map: fail frac by eps x c (mean over graphs)
pd.set_option('display.width',250)
m=df.pivot_table(index='eps',columns='c',values='fail_frac',aggfunc='mean'); p=df.pivot_table(index='eps',columns='c',values='pred_stable',aggfunc='mean')
print(pd.concat([m.round(2),(1-p).round(2)],axis=1,keys=['emp_fail','pred_unstable']).to_string())
# c_crit pred: median per eps
print(df.groupby('eps').c_crit_pred.median().round(3).to_string())
