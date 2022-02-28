import numpy as np
from context import gpytoolbox
import igl
import polyscope as ps
import tetgen


# Load mesh
v,f = igl.read_triangle_mesh("unit_tests_data/bunny_oded.obj")
# Generate many examples
for m in np.linspace(20,5000,30,dtype=int):
    v,f = gpytoolbox.lazy_cage(v,f,num_faces=m)
    v = gpytoolbox.normalize_points(v)
    tgen = tetgen.TetGen(v,f)
    v, t =  tgen.tetrahedralize()
    d = np.zeros((v.shape[0],2))
    d[:,0] = 1.0 - v[:,2]
    d[:,1] = 1.0 - v[:,1]
    #print(igl.volume(v,t))
    u, g, l = gpytoolbox.upper_envelope(v,t,d)
    # Assert that no tet is flipped:
    assert(np.min(igl.volume(u,g))>-1e-8)

print("Unit test passed, all asserts passed")