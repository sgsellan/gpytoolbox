import numpy as np
from context import gpytoolbox
import igl
import polyscope as ps

# Load mesh
P = np.random.rand(500,3)
v, q = gpytoolbox.build_octree_as_hex_mesh(P)
v, SVI, SVJ, q = igl.remove_duplicate_vertices(v,q,1e-5)
t = gpytoolbox.libigl_hex_to_polyscope_hex(q)



M = gpytoolbox.massmatrix_octree(v,t)

ps.init()
ps_vol = ps.register_volume_mesh("test volume mesh", v, hexes=t)
ps_vol.add_scalar_quantity("vertex volumes", np.diag(M.toarray()))
ps.show()