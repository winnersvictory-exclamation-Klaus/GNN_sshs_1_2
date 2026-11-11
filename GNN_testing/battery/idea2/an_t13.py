import pandas as pd, numpy as np
pd.set_option("display.width",250); pd.set_option("display.max_rows",500)
d=pd.read_csv('results/t13_impulse.csv')
print("graphs",d.graph.nunique(),"rows",len(d))
print("\nfail_frac: impulse vs uniform IC (mean over 6 graphs); rows scheme/level, cols force")
t=d.pivot_table(index=['per_node','scheme','level'],columns=['force','ic'],values='fail_frac',aggfunc='mean').round(2)
print(t.to_string())
im=d[d.ic=='impulse']
print("\nimpulse IC: activation first 20 steps / overall (own criterion), far_peak, arrival, ac_final")
t=im.pivot_table(index=['scheme','level'],columns='force',values=['act_first20','act_mean','ac_final'],aggfunc='mean').round(2)
print(t.to_string())
print("\nfar_peak (impulse amp 1) & arrival")
print(im.pivot_table(index=['scheme','level'],columns='force',values=['far_peak','arrival'],aggfunc='mean').round(3).to_string())
