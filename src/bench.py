import time, torch
from manifold import LatentSDE, GeometricFeatures
torch.set_default_dtype(torch.float64)
for k in [2,3,4,5,6,8]:
    sde = LatentSDE(k=k, width=96, depth=3)
    gf = GeometricFeatures(sde, tau=0.3)
    Z = torch.randn(16,k)
    t0=time.time(); _=gf.curvature_only(Z, chunk=16); dt=time.time()-t0
    t1=time.time(); _=gf(Z, chunk=16); dt2=time.time()-t1
    print(f'k={k}: curvature {1000*dt/16:7.1f} ms/pt   full feats {1000*dt2/16:7.1f} ms/pt')
