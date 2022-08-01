import numpy as np
from gpytoolbox.boundary_edges import boundary_edges

def boundary_vertices(F):
    # Given a triangle mesh with face indices F, returns the indices of all
    # boundary vertices. Works only on manifold meshes.
    #
    # Inputs:
    #       F  #F by 3 face index list of a triangle mesh
    # Outputs:
    #       bV  #bV list of indices into F of boundary vertices

    return np.unique(boundary_edges(F))

