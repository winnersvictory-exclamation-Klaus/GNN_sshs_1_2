import pandas as pd, numpy as np
pd.set_option("display.width",250)
d=pd.read_csv('results/t09_symmetry.csv')
print("graphs",d.graph.nunique(),"rows",len(d))
f=lambda x:f"{x:.0e}"
g=d.groupby(['scheme','level','force']).agg(neg20=('neg20','median'),perm20=('perm20','median'),ps20=('perm_shuf20','median'),neg100=('neg100','median'),ps100=('perm_shuf100','median')).reset_index()
for col,t in (('neg20','-u0 vs u0, rel dev t<=20 (median over graphs)'),('perm20','order-preserving relabel, t<=20'),('ps20','relabel + shuffled neighbour order, t<=20'),('ps100','relabel+shuffled order, t<=100')):
    print("\n",t); print(g.pivot_table(index=['scheme','level'],columns='force',values=col).map(f).to_string())
m=d[d.force=='mean']
print("\nmean force: max |weighted-mean drift| (/AC-rms u0) and plain mean drift, median over graphs, T=300")
print(m.groupby(['scheme','level'])[['dcdw_max_c0.5','dcplain_max_c0.5','fail_c0.5','dcdw_max_c1.0','dcplain_max_c1.0','fail_c1.0']].median().round(4).to_string())
