import sys; sys.path.insert(0,'/Users/aidenkoh/Documents/GitHub/GNN_testing/battery')
import common as C, pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_columns',40); pd.set_option('display.max_rows',400)
df=pd.read_csv('/Users/aidenkoh/Documents/GitHub/GNN_testing/battery/idea6/results/t604_invq.csv').rename(columns={'transform':'tr'})
df['cfg']=df.tr+np.where(df.gamma.notna(),'(g='+df.gamma.astype(str)+')','')
print(len(df))
for neg in (False,True):
    d=df[df.neg==neg]
    print('NEG graphs' if neg else 'CORE graphs', d.graph.nunique())
    t=d.groupby(['dyn','par','cfg']).agg(fail=('fail_frac','mean'),alive=('alive_frac','mean'),bc=('bc','mean'),bimodal=('bimodal_frac','mean'),
        kurt=('kurt','mean'),ari_sign=('ari_sign','mean'),cond=('cond_sign','mean'),fied_cond=('fied_cond','mean'),minfrac=('minfrac_sign','mean'),n=('bc','count'))
    print(t.round(3).to_string())
