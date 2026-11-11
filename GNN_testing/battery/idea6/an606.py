import sys; sys.path.insert(0,'/Users/aidenkoh/Documents/GitHub/GNN_testing/battery')
import common as C, pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_columns',40); pd.set_option('display.max_rows',400)
df=pd.read_csv('/Users/aidenkoh/Documents/GitHub/GNN_testing/battery/idea6/results/t606_scope.csv').rename(columns={'transform':'tr'})
print(len(df),df.graph.nunique())
for (delta,dc),s in df.groupby(['delta','dc']):
    print('delta',delta,'dc',dc)
    print(s.pivot_table(index='tr',columns='c',values='fail_frac').round(2).to_string())
s=df[(df.delta==0)&(df.dc==3.0)]
print('dc=+3: dc_dw / dc_plain medians (abs) c=.25'); 
t=s[s.c==0.25].groupby('tr').agg(dc_dw=('dc_dw',lambda x: np.nanmedian(np.abs(x))),dc_plain=('dc_plain',lambda x: np.nanmedian(np.abs(x))),fail=('fail_frac','mean'),n=('dc_dw','count'))
print(t.round(4).to_string())
s=df[(df.delta==0)&(df.dc==0.0)]
t=s[s.c==0.25].groupby('tr').agg(dc_dw=('dc_dw',lambda x: np.nanmedian(np.abs(x))),dc_plain=('dc_plain',lambda x: np.nanmedian(np.abs(x))),fail=('fail_frac','mean'),n=('dc_dw','count'))
print('dc=0, delta=0'); print(t.round(4).to_string())
s=df[(df.delta==1.0)&(df.dc==0.0)&(df.c==0.25)]
print(s.groupby('tr').agg(dc_dw=('dc_dw',lambda x: np.nanmedian(np.abs(x))),dc_plain=('dc_plain',lambda x: np.nanmedian(np.abs(x))),relc=('rel_contrast_end','mean'),envret=('env_ret_end','mean')).round(4).to_string())
