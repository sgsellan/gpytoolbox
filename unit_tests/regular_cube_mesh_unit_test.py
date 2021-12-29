import numpy as np
import polyscope as ps
import igl
from igl import adjacency_matrix, connected_components, volume, boundary_facets
from context import gpytoolbox

# Generate meshes of very diverse sizes
for n in range(10,50,5):
    V,T = gpytoolbox.regular_cube_mesh(n)   
    vols = volume(V,T)
    # Assert all volumes are positive
    assert(np.all(vols>0))
    # Check that all tets are combinatorially connected
    assert(np.max(connected_components(adjacency_matrix(T))[0])==1)
    # Check that the number of faces is six times the number of faces in each outer face of the cube, which is 2*(n-1)*(n-1)
    assert(boundary_facets(T).shape[0]==2*(n-1)*(n-1)*6)
    # To-do: check it is reflectionally symmetric

print("Unit test passed, all asserts passed")