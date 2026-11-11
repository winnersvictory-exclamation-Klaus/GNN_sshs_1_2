from lib5 import *
from t08_rewiring import rewire
df=pd.read_csv(RES/'t08_rewiring.csv'); pd.set_option('display.width',250)
print(len(df), df.graph.nunique())
neg=df[df.exact_graph_exists]
print('eps<=0: max abs traj diff', neg.max_traj_diff.max(), 'max rel', neg.rel_traj_diff.max(), 'fail identical', (neg.fail_frac_d4==neg.fail_frac_graph).mean())
print('ARI identical (edge_cut, orig readout edges):', np.nanmax(np.abs(neg.ari_edge_cut_d4-neg.ari_edge_cut_rew)))
print(neg.groupby('eps')[['m_orig','m_new','self_frac','cut_ratio_orig','cut_ratio_aug']].median().round(4))
print('ARI orig-edge readout vs aug-edge readout of the same rewired trajectory (eps<0, c=0.5):')
n5=neg[neg.c==0.5].groupby('eps').agg(ari_d4=('ari_edge_cut_d4','mean'),ari_aug=('ari_edge_cut_rew_augedges','mean'),em_d4=('em_d4','median'),em_aug=('em_rew_augedges','median'))
print(n5.round(3))
pos=df[~df.exact_graph_exists]
print(pos.groupby(['eps','c']).agg(neg_frac=('neg_frac','median'),rel_diff=('rel_traj_diff','median'),ff_d4=('fail_frac_d4','mean'),ff_clip=('fail_frac_graph','mean'),ari_d4=('ari_edge_cut_d4','mean'),ari_clip=('ari_edge_cut_rew','mean')).round(3))
# operator-level distance between L+eps L^2 and the clipped graph's operator
g,truth,meta=C.make_graph(dict(kind='sbm',bridges=2,seed=0))
for e in (0.05,0.1,0.3,1.0,-0.2):
    gp,info=rewire(g,e); Lr=C.L_rw(g); M=Lr+e*Lr@Lr; Lp=C.L_rw(gp)
    print(e,'rel operator diff (fro)',np.linalg.norm(M-Lp)/np.linalg.norm(M), 'neg mass frac',info['neg_frac'],'min self',info['min_self'])
# weights check: S symmetric
