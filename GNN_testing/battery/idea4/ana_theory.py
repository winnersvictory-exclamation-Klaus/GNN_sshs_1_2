import sys; sys.path.insert(0,'/Users/aidenkoh/Documents/GitHub/GNN_testing/battery/idea4')
import i4, numpy as np, pandas as pd
C=i4.C
pd.set_option("display.width",250); pd.set_option("display.max_columns",40); pd.set_option("display.max_rows",300)
print("##### T4.01")
df=pd.read_csv('results/t401_linear_theory.csv'); df['rel']=(df.c_emp-df.c_pred)/df.c_pred
for cfg,s in df.groupby('config',sort=False):
    b=C.boot_ci(s.rel.values)
    print('%-14s n=%d c_pred mean %.4f c_emp mean %.4f rel err mean %.4f [%.4f,%.4f] min %.4f max %.4f | below(0.98cp) fails=%d above(1.02cp) stable=%d lam_max range %.3f-%.3f'%(cfg,len(s),s.c_pred.mean(),s.c_emp.mean(),*b,s.rel.min(),s.rel.max(),(s.ff_below>0).sum(),(s.ff_above<1).sum(),s.lam_max.min(),s.lam_max.max()))
print("##### T4.02")
d=pd.read_csv('results/t402_dispersion.csv')
d['rel_ls']=(d.om_ls-d.om_pred).abs()/d.om_pred
d['cyc']=d.om_pred*4096/(2*np.pi)
print('modes',len(d),'max rel LS err',d.rel_ls.max())
d['rel_fft']=(d.om_fft-d.om_pred).abs()/d.om_pred
print('FFT max rel err (modes with >=5 cycles in window):',d[d.cyc>=5].rel_fft.max(),' all:',d.rel_fft.max(), 'n modes cyc<5:',(d.cyc<5).sum())
d['dev_from_continuum']=(d.om_pred-d.om_plain).abs()/d.om_plain
print(d.groupby(['config','cmode']).dev_from_continuum.agg(['median','max']).round(4))
d['dev_leap']=(d.om_pred-d.om_exact_wave).abs()/d.om_exact_wave
print('plain leapfrog formula check (mean config): max dev',d[d.config=='mean'].dev_leap.max())
print("##### T4.03")
t=pd.read_csv('results/t403_order_weight_sweep.csv')
for part in 'AB':
    s=t[t.part==part]
    for c in ((0.5,1.0,1.4,1.8) if part=='A' else (0.5,1.0)):
        rate=s[f'rate_c{c}']; obs=s[f'ff_c{c}']>0
        pf=rate*1000>8; po=rate<=1e-9
        print(part,c,'n',len(s),'pred_fail&obs',(pf&obs).sum(),'pred_fail&~obs',(pf&~obs).sum(),'pred_ok&obs',(po&obs).sum(),'pred_ok&~obs',(po&~obs).sum(),'ambiguous',(~pf&~po).sum())
b=t[t.part=='B']; print('B vectors with g<0 somewhere: %.1f%% (n=%d)'%(100*(b.n_neg>0).mean(),len(b)), 'rate min among those @c=.5', b[b.n_neg>0]['rate_c0.5'].min())
a=t[t.part=='A']
print('A: cells with g<0:', (a.n_neg>0).sum(),'of',len(a), ' c_pred range',a.c_pred.min(),a.c_pred.max())
print(a.groupby('K').c_pred.agg(['min','max']))
# c_pred relative to plain
plain=a[a.K==1].c_pred.mean(); print('plain mean c_pred',plain)
best=a.groupby(['K','eps']).c_pred.mean(); print('max c_pred over (K,eps):',best.idxmax(), best.max())
print("##### T4.07")
A=pd.read_csv('results/t407_partA_roots.csv'); t6=A[A.e6.notna()]
print('grid points',len(t6),' parasitic-stable at s->0:',(t6.rho_para0<1-1e-9).sum())
fully=t6[(t6.s_star>=2.0)]
print('points with s*>=2 (c up to 1 for lam_max=2):',len(fully)); print(fully[['e4','e6','s_star','rho_para0']].to_string())
print('e4>0 points with s*>0.01:',((t6.e4>0)&(t6.s_star>0.011)).sum(),' e4>0 total',(t6.e4>0).sum())
print('e4>0 s* range', t6[t6.e4>0].s_star.min(), t6[t6.e4>0].s_star.max(),' max s* among e4>0:', t6[t6.e4>0].sort_values('s_star').tail(3)[['e4','e6','s_star']].to_string())
print('e4=0:'); print(t6[t6.e4==0][['e6','s_star','rho_para0']].to_string())
print('named points:')
for e in [(0.1,0.01),(-0.1,-0.01),(-1/12,1/90),(-0.03,0.003),(-0.01,0.0),(-0.1,0.0)]:
    r=i4.time_rho(e,np.linspace(0.01,4.2,840)); bad=np.where(r>1+1e-9)[0]
    s=np.linspace(0.01,4.2,840)[bad[0]-1] if len(bad) else 4.2
    r0=i4.time_roots(e,1e-12); o=np.argsort(abs(r0-1)); pa=np.abs(r0[o[2:]]).max()
    g8=np.log(abs(i4.time_phys_root(e,8e-3)))
    print(e,'s*=%.3f parasitic|z|(s->0)=%.3f gamma(s=8e-3)=%.3e  e4 s^2/2 - e6 s^3 = %.3e'%(s,pa,g8,e[0]*8e-3**2/2-e[1]*8e-3**3))
t4=A[A.e6.isna()]
print('t4 stable window e in:', t4[(t4.s_star>0.01)].e4.min(), t4[(t4.s_star>0.01)].e4.max())
print(t4[t4.e4.isin([-0.25,-0.2,-0.1,-0.05,0.0,0.1,0.5])][['e4','rho_para0','s_star']])
C_=pd.read_csv('results/t407_partC_sweep.csv'); C_['pf']=C_.grow_T>8; C_['po']=C_.rho_pred<=1+1e-9; C_['obs']=C_.fail_frac>0
C_['lead']=1+C_.e4+C_.e6
print('C: n',len(C_),'pred_fail&obs',(C_.pf&C_.obs).sum(),'pred_fail&~obs',(C_.pf&~C_.obs).sum(),'pred_ok&obs',(C_.po&C_.obs).sum(),'pred_ok&~obs',(C_.po&~C_.obs).sum(),'amb',(~C_.pf&~C_.po).sum(),'amb&obs',(~C_.pf&~C_.po&C_.obs).sum())
print('pred_ok&obs rows lead coefficient:', C_[C_.po&C_.obs].lead.unique())
am=C_[~C_.pf&~C_.po]; print('amb: obs fail only if grow_T >=',am[am.obs].grow_T.min(), '; no-fail max grow_T',am[~am.obs].grow_T.max())
B=pd.read_csv('results/t407_partB_rootsim.csv'); B['d']=(B.emp-B.theta)
sig=B[(B.theta.abs()>2e-3)|(B.emp.abs()>2e-3)]
print('B: n',len(B),'|growth|>2e-3: n',len(sig),'median |d| %.2e, 90pct %.2e, max %.2e'%(sig.d.abs().median(),sig.d.abs().quantile(.9),sig.d.abs().max()))
print(B.groupby(['e4','e6','c']).agg(max_theory=('theta','max'),max_emp=('emp','max')).round(5))
print("##### T4.08")
s=pd.read_csv('results/t408_startup.csv')
print(s.groupby('start').fail_frac.mean().round(4))
st=s[s.rho_pred<=1+1e-9]; print('predicted-stable cells fail:',st.groupby('start').fail_frac.mean().to_dict(), 'n',len(st)//4)
print("##### T4.09")
th=pd.read_csv('results/t409_theory.csv'); th['pe']=(th.om_num-th.om_true)/th.om_true
print(th.groupby(['theta','c']).agg(max_abs_phase_err=('pe',lambda x:x.abs().max()),min_logz=('amp_logz','min'),rho_all=('rho_all','max')).round(5))
