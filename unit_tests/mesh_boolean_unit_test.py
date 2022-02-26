import numpy as np
from context import gpytoolbox
import igl
import polyscope as ps

# Load mesh
V1, F1 = igl.read_triangle_mesh("unit_tests_data/armadillo.obj")
V1 = gpytoolbox.normalize_points(V1)
F2 = F1.copy()
V2 = V1+0.1
Vinter,Finter = gpytoolbox.mesh_intersection(V1,F1.astype(np.int32),V2,F2.astype(np.int32))
Vdiff,Fdiff = gpytoolbox.mesh_difference(V1,F1.astype(np.int32),V2,F2.astype(np.int32))
Vunion,Funion = gpytoolbox.mesh_union(V1,F1.astype(np.int32),V2,F2.astype(np.int32))

ps.init()
ps_mesh_1 = ps.register_surface_mesh("mesh 1", V1, F1)
ps_mesh_2 = ps.register_surface_mesh("mesh 2", V2, F2)
ps_union = ps.register_surface_mesh("union", Vunion, Funion)
ps_diff = ps.register_surface_mesh("union", Vdiff, Fdiff)
ps_inter = ps.register_surface_mesh("union", Vinter, Finter)
ps.show()