import pandas as pd, numpy as np, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import i2lib as L
C = L.C
pd.set_option("display.width",250)
d=pd.read_csv('results/t04_normalisation.csv')
print("graphs",d.graph.nunique())
inv=d[d.cond.isin(["shift+3","scale0.01","scale10","impulse"])]
for c in (0.5,1.0):
    s=inv[inv.c==c]
    print(f"\n=== c={c}: median rel. deviation (AC part, T=300) — mean force. rows scheme/mode/level, cols condition ===")
    t=s.pivot_table(index=['scheme','level','mode'],columns='cond',values='dev',aggfunc='median')
    print(t.map(lambda x:f"{x:.1e}").to_string())
    print("fraction of reference runs that blow up (ac>20x):", s.groupby(['scheme','level','mode']).ref_blowup.mean().round(2).to_dict())
print("\nimpulse: activation (own criterion) mean over first 50 steps / all, mean force")
s=inv[(inv.cond=='impulse')]
print(s.pivot_table(index=['scheme','level','mode'],columns='c',values=['act_first50','act_mean'],aggfunc='mean').round(3).to_string())
o=d[d.cond.str.startswith('dc')]
print("\n=== owa/trim c=1: fail_frac at dc=0 vs dc=3, rms vs ac normalisation (mean over 6 graphs) ===")
t=o.pivot_table(index=['force','scheme','level'],columns=['mode','cond'],values='fail_frac',aggfunc='mean').round(2)
print(t.to_string())
print("ac_final")
print(o.pivot_table(index=['force','scheme','level'],columns=['mode','cond'],values='ac_final',aggfunc='median').round(3).to_string())
