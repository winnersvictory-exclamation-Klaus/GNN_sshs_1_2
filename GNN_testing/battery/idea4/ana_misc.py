import sys; sys.path.insert(0,'/Users/aidenkoh/Documents/GitHub/GNN_testing/battery/idea4')
import i4, numpy as np, pandas as pd
C=i4.C
pd.set_option("display.width",250); pd.set_option("display.max_columns",40); pd.set_option("display.max_rows",300)
print("##### T4.06")
d=pd.read_csv('results/t406_nonlinear_bases.csv')
p2=d[d.part==2]
print('n graphs',p2.graph.nunique(), 'seeds',sorted(p2.seed.unique()))
rows=[]
for (b,t,c),s in p2.groupby(['base','term','c']):
    m,lo,hi=C.boot_ci(s.fail_frac.values); rows.append(dict(base=b,term=t,c=c,fail=m,lo=lo,hi=hi,n=len(s)))
r=pd.DataFrame(rows)
print(r.pivot_table(index=['base','term'],columns='c',values='fail').round(3))
# paired: series vs plain base fail_frac
for b in ('trim','huber','owa','vw'):
    for c in (0.25,0.5,1.0):
        sub=p2[(p2.base==b)&(p2.c==c)].copy(); sub['cfg']=sub.term
        t=C.paired_vs_baseline(sub,'fail_frac','cfg','plain',pair=('graph','seed'))
        print(b,c,t[['config','n','mean_diff','ci_lo','ci_hi','p_holm']].round(3).to_string(index=False))
print("##### T4.10")
s=pd.read_csv('results/t410_stimulus.csv')
print('light cone first_step==ceil(hop/K) (1e-12 threshold):', (s.first_step==s.expected_step).groupby(s.config).mean().round(3).to_dict())
print('mismatch by policy', (s.first_step!=s.expected_step).groupby(s.policy).sum().to_dict(), 'mean delay', (s.first_step-s.expected_step)[s.first_step!=s.expected_step].mean())
for pol in ('c0.5','0.9cp'):
    sub=s[s.policy==pol]
    for v in ('arrival_a1','far_peak_a1','transmit_ratio_a1','snr_a1','noise_gain_a1','det_auc_a1','arrival_agg_a1'):
        t=C.paired_vs_baseline(sub,v,'config','wave',pair=('graph','seed')); b=C.boot_ci(sub[sub.config=='wave'][v].values)
        print(pol,v,'wave %.3f [%.3f,%.3f]'%b); print(t[['config','n','mean_diff','ci_lo','ci_hi','p_holm']].round(4).to_string(index=False))
