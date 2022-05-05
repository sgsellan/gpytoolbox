import numpy as np
from context import gpytoolbox
import igl
import polyscope as ps

# Build octree
P = np.random.rand(100,3)
v, q = gpytoolbox.build_octree_as_hex_mesh(P)
v, SVI, SVJ, q = igl.remove_duplicate_vertices(v,q,1e-5)
t = gpytoolbox.libigl_hex_to_polyscope_hex(q)

# Now let's generate random points by barycentric interpolation on the octree
num_samples = 10
indeces = np.random.randint(t.shape[0],size=num_samples)
B = np.random.rand(num_samples,8)
B = B/np.tile(np.linalg.norm(B,axis=1,ord=1),(8,1)).transpose()

queries = np.zeros((num_samples,3))
for dim in range(3):
    queries[:,dim] = (B[:,0]*v[t[indeces,0],dim] + B[:,1]*v[t[indeces,1],dim] + B[:,2]*v[t[indeces,2],dim] + B[:,3]*v[t[indeces,3],dim] + B[:,4]*v[t[indeces,4],dim] + B[:,5]*v[t[indeces,5],dim] + B[:,6]*v[t[indeces,6],dim] + B[:,7]*v[t[indeces,7],dim])/8.0

D = gpytoolbox.interpolate_octree(queries,v,t)

interpolated_points = D @ v
# These should be the same, since we're using the interpolation to compute positions
assert((np.isclose(interpolated_points,queries)).all())
print("Unit test passed, all asserts passed")