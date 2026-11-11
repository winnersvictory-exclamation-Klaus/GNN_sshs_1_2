import sys; sys.path.insert(0,'/Users/aidenkoh/Documents/GitHub/GNN_testing/battery')
import common as C, pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_columns',40); pd.set_option('display.max_rows',400)
df=pd.read_csv('/Users/aidenkoh/Documents/GitHub/GNN_testing/battery/idea6/results/t605_blockoffset.csv')
print(len(df), df.graph.nunique())
for v in ['fail_frac','hold_true_ac','env_ret_end','rel_contrast_end','dout_din_end','ac_rms_end']:
    p=df.groupby(['config','delta'])[v].mean().unstack('delta')
    print(v); print(p.round(2).to_string())
