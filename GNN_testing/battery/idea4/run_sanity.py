import sys; sys.path.insert(0,'/Users/aidenkoh/Documents/GitHub/GNN_testing/battery/idea4')
import importlib, time
C=None
for name in sys.argv[1:]:
    m=importlib.import_module(name); C=m.C
    sp=C.zoo('core',seeds=[0])[1]
    t0=time.time()
    r=m.job(sp) if name!='t412_damping_rescue' else m.job(sp)
    print(name,'ok rows',len(r),'time',round(time.time()-t0,1))
