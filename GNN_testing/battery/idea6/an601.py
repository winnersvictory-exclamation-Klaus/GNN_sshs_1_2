import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_columns',40); pd.set_option('display.max_rows',400)
df=pd.read_csv('results/t601_props.csv').rename(columns={'transform':'tr'})
print('rows',len(df),'nonfinite F',(~df.finite).sum(), 'props errors', df.get('err',pd.Series(dtype=object)).notna().sum())
d=df[(df.gamma.isna()|(df.gamma==1.0))&(df.state=='uniform')]
# summary by base: oddness etc medians over transforms
print(d.groupby('base')[['oddness','momentum_rel','homog_rel','dsym_rel','const_defect']].median().round(3))
print(d[d.base=='mean'].groupby('scope')[['oddness','momentum_rel','dsym_rel','eig_sym_max','lam_max']].median().round(3))
print('anti-restoring (eig_sym_max>1e-8) fraction by tr (mean base, uniform):')
print(d[d.base=='mean'].groupby('tr').apply(lambda s:(s.eig_sym_max>1e-8).mean()).round(2).to_dict())
# consensus-state nonlinearity: F(const)
print('max const_defect',df.const_defect.max())
# EPS regime: homogeneity at tiny amplitude
for st in ['uniform','tiny_1e-9','tiny_1e-13','huge_1e6']:
    s=df[(df.state==st)&(df.base=='mean')&(df.gamma.isna()|(df.gamma==1.0))]
    print(st,'homog_rel median', s.homog_rel.median(), ' F_rms/rms(u) median', (s.F_rms).median())
# local scope near-equal neighbours: F relative scale for block_const and near_consensus
s=df[(df.state.isin(['block_const','near_consensus']))&(df.base=='mean')&(df.gamma.isna()|(df.gamma==1.0))]
print(s.groupby(['state','tr']).F_rms.median().unstack(0).round(6))
print('jac_kink>0.1 fraction by state', df.groupby('state').apply(lambda s:(s.jac_kink>0.1).mean()).round(2).to_dict())
