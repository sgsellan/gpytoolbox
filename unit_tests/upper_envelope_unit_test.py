import numpy as np
from context import gpytoolbox
import igl
import polyscope as ps
import tetgen


# Load mesh
v,f = igl.read_triangle_mesh("unit_tests_data/bunny_oded.obj")
v = gpytoolbox.normalize_points(v)
tgen = tetgen.TetGen(v,f)
v, t =  tgen.tetrahedralize()
d = np.zeros((v.shape[0],2))
d[:,0] = 1.0 - v[:,2]
d[:,1] = 1.0 - v[:,1]
u, g, l = gpytoolbox.upper_envelope(v,t,d)
faces =igl.boundary_facets(g[l[:,0],:])
oriented_faces,c = igl.bfs_orient(faces)
# tetrahedralize
ps.init()
ps.register_volume_mesh("test tet mesh",vertices=u,tets=g[l[:,0],:])
ps.register_surface_mesh("test surface",vertices=u,faces=oriented_faces)
ps.show()