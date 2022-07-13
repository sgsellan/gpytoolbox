import numpy as np
from gpytoolbox.edges import edges

def boundary_edges(F):
    # Given a triangle mesh with face indices F, returns all unique oriented
    # boundary edges as indices into the vertex array.
    # Works only on manifold meshes.
    #
    # Inputs:
    #       F  #F by 3 face index list of a triangle mesh
    # Outputs:
    #       bE  #bE by 2 indices of boundary edges into the vertex array

    E,b = edges(F, return_boundary_indices=True)
    return E[b,:]
