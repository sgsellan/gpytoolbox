from context import gpytoolbox
import numpy as np
import polyscope as ps

np.random.seed(0)
th = 2*np.pi*np.random.rand(100,1)
P = 0.5*np.concatenate((np.cos(th),np.sin(th)),axis=1)

C,W,CH,PAR,D,A = gpytoolbox.initialize_quadtree(P,graded=True,max_depth=6,vmin=np.array([-1,-1]),vmax=np.array([1,1]))
V,Q = gpytoolbox.bad_quad_mesh_from_quadtree(C,W,CH)


G, stored_at = gpytoolbox.quadtree_gradient(C,W,CH,D,A)
Gx = G[0:stored_at.shape[0],:]
Gy = G[stored_at.shape[0]:(2*stored_at.shape[0]),:]
fun = stored_at[:,0]
# This should be one everywhere
assert(np.all(np.isclose(Gx @ stored_at[:,0],1.0)))
# This will never be exactly zero (unless you subdivide everything at once), but it should be zero "in most places"
assert(np.isclose(np.median(np.abs(Gy @ stored_at[:,0])),0.0))
print("Unit test passed, all asserts passed")

