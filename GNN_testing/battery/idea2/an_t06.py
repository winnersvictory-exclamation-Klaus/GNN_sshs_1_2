import pandas as pd, numpy as np
pd.set_option("display.width",250)
d=pd.read_csv('results/t06_energy_trigger.csv')
print("graphs",d.graph.nunique())
w=d[d.scheme=='wave'].groupby(['dc','force'])[['fail_frac']].mean().round(2)
print("wave fail by dc,force\n",w.T.to_string())
e=d[d.scheme!='wave']
for dc in (0.0,3.0):
    print(f"\n=== dc={dc}: fail_frac / trig_frac / drift_only_frac_of_trig, scheme energy vs energy_ac ===")
    s=e[e.dc==dc]
    t=s.pivot_table(index=['force','kmax'],columns='scheme',values=['fail_frac','trig_frac','drift_only_frac_of_trig','dead_frac','ac_final'],aggfunc='mean').round(2)
    print(t.to_string())
print("\nfirst trigger step vs first step AC energy>initial (medians over graphs), dc=0")
s=e[e.dc==0.0]
print(s.pivot_table(index=['force','scheme'],values=['first_trig_med','first_acgt1_med','dc_part_max','ac_ratio_max','dc_drift_dw'],aggfunc='median').round(2).to_string())
