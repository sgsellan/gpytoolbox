import numpy as np
import sys
from igl import adjacency_matrix, connected_components
sys.path.append("..")
from regular_square_mesh import regular_square_mesh


# Generate meshes of very diverse sizes
for n in range(10,200,5):
    V,F = regular_square_mesh(n)   
    # Check: vertices are between zero and one
    assert(np.max(V)==1.0)
    assert(np.min(V)==0.0)
    # Check: all triangles are combinatorially connected
    assert(np.max(connected_components(adjacency_matrix(F))[0])==1)
    # Check: all faces properly oriented
    normals = np.cross( V[F[:,1],:] - V[F[:,0],:] , V[F[:,2],:] - V[F[:,0],:] , axis=1)
    assert((normals>0).all())
    
print("Unit test successful")

# To visualize
# import polyscope as ps
# ps.init()
# ps.register_surface_mesh("my mesh", V , F, smooth_shade=False,edge_width=1.0)
# ps.set_navigation_style("planar")
# ps.show()
