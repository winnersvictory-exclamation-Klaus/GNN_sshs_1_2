from lib5 import *
th=pd.read_csv(RES/'t05_theory_grid.csv')
neg=th[(th.eps<0)&(th.eps>=-0.25)&(th.eps<=-0.0125)]
# fit s_crit = a + b eps
import numpy as np
m=th[(th.eps>-0.25)&(th.eps<0)&np.isfinite(th.s_crit)]
A=np.polyfit(m.eps,m.s_crit,1); print('s_crit fit',A, 'max resid',np.abs(np.polyval(A,m.eps)-m.s_crit).max())
print(th[th.eps.isin([-0.25,-0.2375,-1/12,-0.1,-0.05,0.0125])][['eps','s_crit','par_abs_small_s','phys_abs_small_s']])
print('slope check: max rel dev (rho-1)/s^2 vs eps/2 for |eps|<=0.1:', np.max(np.abs(th[(th.eps.abs()<=0.1)&(th.eps.abs()>0.005)].eval('slope_rho_over_s2/pred_slope-1'))))
df=pd.read_csv(RES/'t05_time_roots.csv')
print('rows',len(df),'exact',df.fail_step_exact_match.mean(),'maxdiff',df.max_fail_step_diff.max(), 'graphs',df.graph.nunique())
# stable/unstable agreement via pred_ln_rho
df['pred_unst']=df.pred_ln_rho_max>1e-9
print(pd.crosstab(df.pred_unst, df.fail_frac>0))
# c_crit pred for eps<0 vs emp: lambda_max per graph
d=df[(df.eps<0)&(df.eps>-0.25)].copy()
d['s_crit']=4*(1+4*d.eps); d['c_crit_pred']=np.sqrt(d.s_crit/d.lam_max)
# check: runs with c<c_crit*0.99 never fail, c>1.01 c_crit fail within 2000 steps?
d['ratio']=d.c/d.c_crit_pred
print(d.groupby(pd.cut(d.ratio,[0,0.9,0.99,1.01,1.1,5])).fail_frac.agg(['mean','count']))
# measured growth vs predicted in failing runs eps>0
p=df[(df.eps>0)&(df.fail_frac==1)]
print('growth meas/pred (eps>0 failing), median',(p.meas_growth_rate/p.pred_ln_rho_max).median(), 'IQR',(p.meas_growth_rate/p.pred_ln_rho_max).quantile([.25,.75]).values)
# t4 default checks
q=df[(df.eps==0.1)&(df.c.isin([0.5,1.0]))].groupby(['kind','c']).med_fail_step.agg(['min','median','max'])
print(q)
